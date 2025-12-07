"""
Motion-specific diffusion utilities.

Extends GaussianDiffusion with motion-specific functionality:
- 6D rotation <-> quaternion conversion
- Forward kinematics integration
- Motion-aware sampling
"""

from typing import Dict, Optional, Tuple, Callable
import torch
import torch.nn as nn

from .gaussian_diffusion import GaussianDiffusion
from .noise_schedule import NoiseSchedule


class MotionDiffusion(GaussianDiffusion):
    """
    Diffusion specialized for skeletal motion.

    Handles:
    - 6D rotation representation conversion
    - Forward kinematics for position computation
    - Motion-aware losses
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: NoiseSchedule,
        kinematic_chain: Optional[nn.Module] = None,
        prediction_type: str = 'x0',
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Args:
            model: Denoising model
            schedule: Noise schedule
            kinematic_chain: KinematicChain for FK computation (optional)
            prediction_type: What the model predicts ('x0', 'epsilon', 'v')
            clip_denoised: Whether to clip denoised samples
            clip_range: Range for clipping
        """
        super().__init__(
            model=model,
            schedule=schedule,
            prediction_type=prediction_type,
            clip_denoised=clip_denoised,
            clip_range=clip_range
        )

        self.kinematic_chain = kinematic_chain

    def rotation_6d_to_quaternion(
        self,
        rot_6d: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert 6D rotation representation to quaternion.

        Args:
            rot_6d: 6D rotation of shape (..., 6)

        Returns:
            Quaternion of shape (..., 4) as [w, x, y, z]
        """
        from ..utils.rotation import rotation_6d_to_quaternion
        return rotation_6d_to_quaternion(rot_6d)

    def quaternion_to_rotation_6d(
        self,
        quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert quaternion to 6D rotation representation.

        Args:
            quat: Quaternion of shape (..., 4) as [w, x, y, z]

        Returns:
            6D rotation of shape (..., 6)
        """
        from ..utils.rotation import quaternion_to_rotation_6d
        return quaternion_to_rotation_6d(quat)

    def compute_joint_positions(
        self,
        rotations_6d: torch.Tensor,
        root_translation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute global joint positions via forward kinematics.

        Optimized to process all frames and batch elements together when possible.

        Args:
            rotations_6d: Joint rotations (batch, num_joints, 6, num_frames) or (batch, num_joints, 6)
            root_translation: Root translation (batch, 3, num_frames) or (batch, 3)

        Returns:
            Joint positions (batch, num_joints, 3, num_frames) or (batch, num_joints, 3)
        """
        if self.kinematic_chain is None:
            raise ValueError("KinematicChain not provided")

        # Handle both single frame and multi-frame input
        has_time_dim = rotations_6d.dim() == 4

        if has_time_dim:
            batch_size, num_joints, _, num_frames = rotations_6d.shape
            device = rotations_6d.device

            # Reshape to (batch * num_frames, num_joints, 6) for batched processing
            rot_flat = rotations_6d.permute(0, 3, 1, 2).reshape(batch_size * num_frames, num_joints, 6)

            # Also flatten root translation if provided
            root_flat = None
            if root_translation is not None:
                root_flat = root_translation.permute(0, 2, 1).reshape(batch_size * num_frames, 3)

            # Compute positions for all (batch, frame) pairs
            positions_flat = self._compute_positions_batched(rot_flat, root_flat)  # (B*T, J, 3)

            # Reshape back to (batch, num_joints, 3, num_frames)
            positions = positions_flat.reshape(batch_size, num_frames, num_joints, 3)
            positions = positions.permute(0, 2, 3, 1)

            return positions
        else:
            return self._compute_positions_batched(rotations_6d, root_translation)

    def _compute_positions_batched(
        self,
        rotations_6d: torch.Tensor,
        root_translation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute positions for batched samples (optimized).

        Uses the kinematic chain but processes batch in parallel where possible.

        Args:
            rotations_6d: (batch, num_joints, 6)
            root_translation: (batch, 3) or None

        Returns:
            positions: (batch, num_joints, 3)
        """
        batch_size = rotations_6d.shape[0]
        num_joints = rotations_6d.shape[1]
        device = rotations_6d.device

        # Convert 6D to quaternion for all samples at once
        rot_flat = rotations_6d.reshape(-1, 6)
        quat_flat = self.rotation_6d_to_quaternion(rot_flat)
        quaternions = quat_flat.reshape(batch_size, num_joints, 4)  # (batch, num_joints, 4)

        # Compute FK for each sample in batch
        # Note: Could be further optimized with batched FK implementation
        all_positions = []
        for b in range(batch_size):
            root_trans = root_translation[b] if root_translation is not None else None

            # Use kinematic chain FK
            transforms = self.kinematic_chain.forward_kinematics(
                local_rotations=quaternions[b],
                root_translation=root_trans
            )

            # Extract positions from transforms
            positions = []
            for name in self.kinematic_chain.joint_order:
                trans, _ = transforms[name]
                positions.append(trans)
            positions = torch.stack(positions)  # (num_joints, 3)
            all_positions.append(positions)

        return torch.stack(all_positions)  # (batch, num_joints, 3)

    def training_loss_with_motion(
        self,
        x_0: torch.Tensor,
        condition: Optional[Dict] = None,
        root_translation: Optional[torch.Tensor] = None,
        lambda_pos: float = 0.0,
        lambda_vel: float = 0.0,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss with optional motion-specific losses.

        Args:
            x_0: Clean motion (batch, num_joints, 6, num_frames)
            condition: Conditioning dictionary
            root_translation: Root translation (batch, 3, num_frames)
            lambda_pos: Weight for FK position loss
            lambda_vel: Weight for velocity loss
            noise: Optional pre-sampled noise

        Returns:
            (loss, metrics): Loss tensor and metrics dictionary
        """
        # Base diffusion loss
        loss, metrics = self.training_loss(x_0, condition, noise)

        # Get prediction for auxiliary losses
        if lambda_pos > 0 or lambda_vel > 0:
            batch_size = x_0.shape[0]
            t = self.schedule.sample_timesteps(batch_size, device=x_0.device)
            x_t, _ = self.q_sample(x_0, t, noise)
            pred_x0, _ = self.model_predictions(x_t, t, condition)

            # Velocity loss
            if lambda_vel > 0:
                pred_vel = pred_x0[:, :, :, 1:] - pred_x0[:, :, :, :-1]
                target_vel = x_0[:, :, :, 1:] - x_0[:, :, :, :-1]
                vel_loss = ((pred_vel - target_vel) ** 2).mean()
                loss = loss + lambda_vel * vel_loss
                metrics['vel_loss'] = vel_loss.item()

            # FK position loss
            if lambda_pos > 0 and self.kinematic_chain is not None:
                try:
                    pred_pos = self.compute_joint_positions(pred_x0, root_translation)
                    target_pos = self.compute_joint_positions(x_0, root_translation)
                    pos_loss = ((pred_pos - target_pos) ** 2).mean()
                    loss = loss + lambda_pos * pos_loss
                    metrics['pos_loss'] = pos_loss.item()
                except Exception:
                    # FK computation may fail; skip position loss
                    pass

        return loss, metrics

    @torch.no_grad()
    def sample_motion(
        self,
        num_samples: int,
        num_joints: int,
        num_frames: int,
        condition: Optional[Dict] = None,
        method: str = 'ddim',
        num_steps: int = 4,
        eta: float = 0.0,
        progress_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Generate motion samples.

        Args:
            num_samples: Number of samples to generate
            num_joints: Number of joints
            num_frames: Number of frames
            condition: Conditioning dictionary
            method: Sampling method ('ddpm', 'ddim')
            num_steps: Number of sampling steps
            eta: DDIM eta parameter
            progress_callback: Optional progress callback

        Returns:
            Generated motion (num_samples, num_joints, 6, num_frames)
        """
        shape = (num_samples, num_joints, 6, num_frames)
        return self.sample(
            shape=shape,
            condition=condition,
            method=method,
            num_steps=num_steps,
            eta=eta,
            progress_callback=progress_callback
        )

    @torch.no_grad()
    def sample_conditioned(
        self,
        past_motion: torch.Tensor,
        trajectory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        style_idx: Optional[torch.Tensor] = None,
        num_frames: int = 20,
        method: str = 'ddim',
        num_steps: int = 4,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Sample motion conditioned on past motion and optional trajectory/style.

        Args:
            past_motion: Past motion (batch, num_joints, 6, past_frames)
            trajectory: Optional (traj_translation, traj_rotation) tuple
            style_idx: Optional style indices (batch,)
            num_frames: Number of frames to generate
            method: Sampling method
            num_steps: Number of sampling steps
            eta: DDIM eta

        Returns:
            Generated motion (batch, num_joints, 6, num_frames)
        """
        batch_size = past_motion.shape[0]
        num_joints = past_motion.shape[1]

        # Build condition dict
        condition = {'past_motion': past_motion}

        if style_idx is not None:
            condition['style_idx'] = style_idx

        if trajectory is not None:
            condition['traj_translation'] = trajectory[0]
            condition['traj_rotation'] = trajectory[1]

        # Sample
        shape = (batch_size, num_joints, 6, num_frames)
        return self.sample(
            shape=shape,
            condition=condition,
            method=method,
            num_steps=num_steps,
            eta=eta
        )

    def normalize_rotation_6d(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """
        Normalize 6D rotation via Gram-Schmidt.

        Args:
            rot_6d: Rotation of shape (..., 6)

        Returns:
            Normalized rotation of shape (..., 6)
        """
        from ..utils.rotation import normalize_rotation_6d
        return normalize_rotation_6d(rot_6d)

    def motion_to_quaternions(
        self,
        motion_6d: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert motion in 6D representation to quaternions.

        Args:
            motion_6d: Motion (batch, num_joints, 6, num_frames)

        Returns:
            Quaternions (batch, num_joints, 4, num_frames)
        """
        batch_size, num_joints, _, num_frames = motion_6d.shape

        # Reshape for conversion: (batch * num_joints * num_frames, 6)
        rot_flat = motion_6d.permute(0, 1, 3, 2).reshape(-1, 6)

        # Convert
        quat_flat = self.rotation_6d_to_quaternion(rot_flat)

        # Reshape back: (batch, num_joints, num_frames, 4) -> (batch, num_joints, 4, num_frames)
        quat = quat_flat.reshape(batch_size, num_joints, num_frames, 4)
        quat = quat.permute(0, 1, 3, 2)

        return quat

    def quaternions_to_motion(
        self,
        quaternions: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert quaternions to motion in 6D representation.

        Args:
            quaternions: Quaternions (batch, num_joints, 4, num_frames)

        Returns:
            Motion (batch, num_joints, 6, num_frames)
        """
        batch_size, num_joints, _, num_frames = quaternions.shape

        # Reshape for conversion: (batch * num_joints * num_frames, 4)
        quat_flat = quaternions.permute(0, 1, 3, 2).reshape(-1, 4)

        # Convert
        rot_flat = self.quaternion_to_rotation_6d(quat_flat)

        # Reshape back: (batch, num_joints, num_frames, 6) -> (batch, num_joints, 6, num_frames)
        rot = rot_flat.reshape(batch_size, num_joints, num_frames, 6)
        rot = rot.permute(0, 1, 3, 2)

        return rot

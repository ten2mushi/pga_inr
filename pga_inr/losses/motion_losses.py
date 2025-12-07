"""
Motion-specific loss functions.

Provides losses for training motion diffusion models:
- Rotation reconstruction loss
- Velocity smoothness loss
- FK position loss
- Foot contact loss

Motion Tensor Shape Convention: (B, J, D, T)
============================================
All motion tensors in this module follow the convention:
    - B: Batch size
    - J: Number of joints
    - D: Data dimension (6 for 6D rotation, 3 for position)
    - T: Number of time frames

This convention groups spatial data (J, D) together, making joint operations
natural and allowing temporal convolutions to operate directly on the last
dimension without permutation.

See pga_inr.core.types for conversion utilities and detailed rationale.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionReconstructionLoss(nn.Module):
    """
    MSE loss on joint rotations (6D representation).
    """

    def __init__(
        self,
        reduction: str = 'mean',
        normalize: bool = False
    ):
        """
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
            normalize: Whether to normalize 6D rotations before comparison
        """
        super().__init__()
        self.reduction = reduction
        self.normalize = normalize

    def forward(
        self,
        pred: torch.Tensor,   # (batch, J, 6, T)
        target: torch.Tensor  # (batch, J, 6, T)
    ) -> torch.Tensor:
        """Compute rotation reconstruction loss."""
        if self.normalize:
            from ..utils.rotation import normalize_rotation_6d
            # Normalize along rotation dimension
            pred_flat = pred.permute(0, 1, 3, 2).reshape(-1, 6)
            target_flat = target.permute(0, 1, 3, 2).reshape(-1, 6)
            pred_flat = normalize_rotation_6d(pred_flat)
            target_flat = normalize_rotation_6d(target_flat)
            pred = pred_flat.reshape(*pred.shape[:2], pred.shape[3], 6).permute(0, 1, 3, 2)
            target = target_flat.reshape(*target.shape[:2], target.shape[3], 6).permute(0, 1, 3, 2)

        loss = (pred - target) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class VelocityLoss(nn.Module):
    """
    Loss on motion velocity (temporal derivatives).

    Encourages smooth motion by matching frame-to-frame velocities.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        order: int = 1  # 1 for velocity, 2 for acceleration
    ):
        """
        Args:
            reduction: Reduction method
            order: Derivative order (1=velocity, 2=acceleration)
        """
        super().__init__()
        self.reduction = reduction
        self.order = order

    def forward(
        self,
        pred: torch.Tensor,   # (batch, J, 6, T)
        target: torch.Tensor  # (batch, J, 6, T)
    ) -> torch.Tensor:
        """Compute velocity loss."""
        # Compute velocities (finite differences)
        pred_deriv = pred
        target_deriv = target

        for _ in range(self.order):
            pred_deriv = pred_deriv[:, :, :, 1:] - pred_deriv[:, :, :, :-1]
            target_deriv = target_deriv[:, :, :, 1:] - target_deriv[:, :, :, :-1]

        loss = (pred_deriv - target_deriv) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FKPositionLoss(nn.Module):
    """
    Loss on 3D joint positions computed via forward kinematics.

    Ensures the predicted rotations produce physically plausible poses.
    """

    def __init__(
        self,
        kinematic_chain: nn.Module,
        reduction: str = 'mean',
        sample_frames: int = 5  # Sample subset of frames for efficiency
    ):
        """
        Args:
            kinematic_chain: KinematicChain for FK computation
            reduction: Reduction method
            sample_frames: Number of frames to sample for FK computation
        """
        super().__init__()
        self.skeleton = kinematic_chain
        self.reduction = reduction
        self.sample_frames = sample_frames

    def forward(
        self,
        pred_rot_6d: torch.Tensor,      # (batch, J, 6, T)
        target_rot_6d: torch.Tensor,    # (batch, J, 6, T)
        root_translation: Optional[torch.Tensor] = None  # (batch, 3, T)
    ) -> torch.Tensor:
        """
        Compute position loss via FK.

        This is computationally expensive, so we sample a subset of frames.
        """
        from ..utils.rotation import rotation_6d_to_quaternion

        batch_size, num_joints, _, num_frames = pred_rot_6d.shape

        # Sample frames
        if self.sample_frames < num_frames:
            frame_indices = torch.linspace(0, num_frames - 1, self.sample_frames).long()
        else:
            frame_indices = torch.arange(num_frames)

        total_loss = 0.0
        num_computed = 0

        for f_idx in frame_indices:
            # Get rotations for this frame
            pred_rot_f = pred_rot_6d[:, :, :, f_idx]  # (batch, J, 6)
            target_rot_f = target_rot_6d[:, :, :, f_idx]  # (batch, J, 6)

            # Convert 6D to quaternion
            pred_quat = rotation_6d_to_quaternion(pred_rot_f.reshape(-1, 6))
            pred_quat = pred_quat.reshape(batch_size, num_joints, 4)

            target_quat = rotation_6d_to_quaternion(target_rot_f.reshape(-1, 6))
            target_quat = target_quat.reshape(batch_size, num_joints, 4)

            # Get root translation
            root_trans_f = root_translation[:, :, f_idx] if root_translation is not None else None

            # Compute FK for each sample
            try:
                pred_pos = self._compute_positions(pred_quat, root_trans_f)
                target_pos = self._compute_positions(target_quat, root_trans_f)

                loss = ((pred_pos - target_pos) ** 2).mean()
                total_loss = total_loss + loss
                num_computed += 1
            except (RuntimeError, ValueError) as e:
                # Log FK failures for debugging but continue processing
                import warnings
                warnings.warn(f"FK computation failed for frame {f_idx}: {e}")
                continue

        if num_computed == 0:
            import warnings
            warnings.warn("All FK computations failed, returning zero loss")
            return torch.tensor(0.0, device=pred_rot_6d.device, requires_grad=True)

        return total_loss / num_computed

    def _compute_positions(
        self,
        quaternions: torch.Tensor,  # (batch, J, 4)
        root_trans: Optional[torch.Tensor]  # (batch, 3)
    ) -> torch.Tensor:
        """Compute joint positions for a batch."""
        batch_size = quaternions.shape[0]

        positions_list = []
        for b in range(batch_size):
            root_t = root_trans[b] if root_trans is not None else None

            transforms = self.skeleton.forward_kinematics(
                local_rotations=quaternions[b],
                root_translation=root_t
            )

            positions = []
            for name in self.skeleton.joint_order:
                trans, _ = transforms[name]
                positions.append(trans)

            positions_list.append(torch.stack(positions))

        return torch.stack(positions_list)  # (batch, J, 3)


class FootContactLoss(nn.Module):
    """
    Loss to prevent foot sliding.

    When a foot is in contact with the ground (low height), its velocity
    should be near zero.
    """

    def __init__(
        self,
        foot_joint_indices: List[int],
        contact_threshold: float = 0.05,
        reduction: str = 'mean'
    ):
        """
        Args:
            foot_joint_indices: Indices of foot joints in skeleton
            contact_threshold: Height threshold for ground contact
            reduction: Reduction method
        """
        super().__init__()
        self.foot_indices = foot_joint_indices
        self.contact_threshold = contact_threshold
        self.reduction = reduction

    def forward(
        self,
        joint_positions: torch.Tensor  # (batch, J, 3, T)
    ) -> torch.Tensor:
        """
        Compute foot contact loss.

        Args:
            joint_positions: 3D positions from FK
        """
        # Extract foot positions
        foot_pos = joint_positions[:, self.foot_indices, :, :]  # (batch, num_feet, 3, T)

        # Foot height (Y coordinate)
        foot_height = foot_pos[:, :, 1, :]  # (batch, num_feet, T)

        # Detect contact using soft threshold (sigmoid) for gradient flow
        # This allows gradients to flow back to foot positions during training
        # Higher sharpness (20.0) makes it approximate a hard threshold while remaining differentiable
        contact_weight = torch.sigmoid((self.contact_threshold - foot_height) * 20.0)

        # Compute foot velocity
        foot_vel = foot_pos[:, :, :, 1:] - foot_pos[:, :, :, :-1]
        foot_vel_xz = foot_vel[:, :, [0, 2], :]  # XZ velocity only

        # Contact weight aligned with velocity (average of current and next frame)
        contact_weight_vel = (contact_weight[:, :, 1:] + contact_weight[:, :, :-1]) * 0.5

        # Penalize XZ velocity when in contact
        loss = (foot_vel_xz ** 2).sum(dim=2)  # (batch, num_feet, T-1)
        loss = loss * contact_weight_vel

        if self.reduction == 'mean':
            # Weight by sum of contact weights (soft count of contact frames)
            num_contacts = contact_weight_vel.sum() + 1e-8
            return loss.sum() / num_contacts
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MotionDiffusionLoss(nn.Module):
    """
    Combined loss for motion diffusion training.

    L = L_recon + lambda_pos * L_FK + lambda_vel * L_vel + lambda_foot * L_foot
    """

    def __init__(
        self,
        kinematic_chain: Optional[nn.Module] = None,
        lambda_pos: float = 0.5,
        lambda_vel: float = 0.2,
        lambda_foot: float = 0.0,
        foot_joint_indices: Optional[List[int]] = None
    ):
        """
        Args:
            kinematic_chain: KinematicChain for FK losses (optional)
            lambda_pos: Weight for FK position loss
            lambda_vel: Weight for velocity loss
            lambda_foot: Weight for foot contact loss
            foot_joint_indices: Indices of foot joints
        """
        super().__init__()

        self.recon_loss = MotionReconstructionLoss()
        self.vel_loss = VelocityLoss()

        self.lambda_pos = lambda_pos
        self.lambda_vel = lambda_vel
        self.lambda_foot = lambda_foot

        if kinematic_chain is not None and lambda_pos > 0:
            self.pos_loss = FKPositionLoss(kinematic_chain)
        else:
            self.pos_loss = None

        if foot_joint_indices is not None and lambda_foot > 0:
            self.foot_loss = FootContactLoss(foot_joint_indices)
        else:
            self.foot_loss = None

    def forward(
        self,
        pred: torch.Tensor,              # (batch, J, 6, T)
        target: torch.Tensor,            # (batch, J, 6, T)
        root_translation: Optional[torch.Tensor] = None,  # (batch, 3, T)
        joint_positions: Optional[torch.Tensor] = None     # (batch, J, 3, T) for foot loss
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined motion loss.

        Args:
            pred: Predicted motion
            target: Target motion
            root_translation: Root trajectory (for FK loss)
            joint_positions: Joint positions (for foot contact loss)

        Returns:
            (total_loss, metrics_dict)
        """
        metrics = {}

        # Reconstruction loss
        recon = self.recon_loss(pred, target)
        metrics['recon'] = recon.item()
        total_loss = recon

        # Velocity loss
        if self.lambda_vel > 0:
            vel = self.vel_loss(pred, target)
            metrics['vel'] = vel.item()
            total_loss = total_loss + self.lambda_vel * vel

        # FK position loss
        if self.pos_loss is not None and self.lambda_pos > 0:
            try:
                pos = self.pos_loss(pred, target, root_translation)
                metrics['pos'] = pos.item()
                total_loss = total_loss + self.lambda_pos * pos
            except (RuntimeError, ValueError) as e:
                # Log FK computation failures for debugging
                import warnings
                warnings.warn(
                    f"FK position loss computation failed: {e}. "
                    "Skipping position loss for this batch."
                )
                metrics['pos'] = 0.0

        # Foot contact loss
        if self.foot_loss is not None and self.lambda_foot > 0 and joint_positions is not None:
            foot = self.foot_loss(joint_positions)
            metrics['foot'] = foot.item()
            total_loss = total_loss + self.lambda_foot * foot

        metrics['total'] = total_loss.item()

        return total_loss, metrics


class GeodesicRotationLoss(nn.Module):
    """
    Geodesic distance loss for rotations.

    Uses the geodesic distance on SO(3) instead of Euclidean distance
    on the 6D representation.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,   # (batch, J, 6, T)
        target: torch.Tensor  # (batch, J, 6, T)
    ) -> torch.Tensor:
        """Compute geodesic rotation loss."""
        from ..utils.rotation import rotation_6d_to_quaternion
        from ..utils.quaternion import quaternion_angle_distance

        batch_size, num_joints, _, num_frames = pred.shape

        # Reshape: (batch * J * T, 6)
        pred_flat = pred.permute(0, 1, 3, 2).reshape(-1, 6)
        target_flat = target.permute(0, 1, 3, 2).reshape(-1, 6)

        # Convert to quaternions
        pred_quat = rotation_6d_to_quaternion(pred_flat)
        target_quat = rotation_6d_to_quaternion(target_flat)

        # Compute geodesic distance
        distances = quaternion_angle_distance(pred_quat, target_quat)

        # Clamp distances to prevent exploding gradients near convergence
        # The arccos function has gradient -1/sqrt(1-x^2) which explodes as x->1
        # Clamping to pi ensures numerical stability while allowing full rotation range
        distances = torch.clamp(distances, max=3.14159)

        # Loss is squared distance
        loss = distances ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.reshape(batch_size, num_joints, num_frames)

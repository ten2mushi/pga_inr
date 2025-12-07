"""
Motion sequence dataset for diffusion training.

Provides sliding-window datasets over animation sequences
for training conditional motion diffusion models.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MotionSequenceDataset(Dataset):
    """
    Sliding window dataset over animation sequences.

    Creates training samples of (past_motion, future_motion, trajectory).
    """

    def __init__(
        self,
        animations: Dict[str, Any],
        past_frames: int = 5,
        future_frames: int = 20,
        stride: int = 3,
        augment: bool = True,
        rotation_augment_range: float = np.pi / 4,  # +/- 45 degrees
        device: torch.device = None
    ):
        """
        Args:
            animations: Dict of AnimationData from load_rigged_character
            past_frames: Number of past frames for conditioning
            future_frames: Number of future frames to predict
            stride: Stride for sliding window
            augment: Whether to apply data augmentation
            rotation_augment_range: Range for random Y-axis rotation
            device: Device for tensors
        """
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.total_window = past_frames + future_frames
        self.stride = stride
        self.augment = augment
        self.rotation_augment_range = rotation_augment_range
        self.device = device or torch.device('cpu')

        # Process all animations
        self.samples = []
        self.style_names = list(animations.keys())

        for style_idx, (anim_name, anim) in enumerate(animations.items()):
            # Get rotation and translation data
            # AnimationData has: keyframe_rotations (T, J, 4), root_translations (T, 3)
            if hasattr(anim, 'keyframe_rotations'):
                rotations = anim.keyframe_rotations  # (T, J, 4) quaternions
                root_trans = anim.root_translations  # (T, 3)
            else:
                # Assume dict-like access
                rotations = anim['keyframe_rotations']
                root_trans = anim['root_translations']

            # Convert to tensors if needed
            if not isinstance(rotations, torch.Tensor):
                rotations = torch.tensor(rotations, dtype=torch.float32)
            if not isinstance(root_trans, torch.Tensor):
                root_trans = torch.tensor(root_trans, dtype=torch.float32)

            # Convert quaternions to 6D rotation
            rotations_6d = self._quaternion_to_6d(rotations)  # (T, J, 6)

            # Create sliding windows
            num_frames = rotations_6d.shape[0]
            for start in range(0, num_frames - self.total_window + 1, stride):
                end = start + self.total_window

                sample = {
                    'rotations_6d': rotations_6d[start:end].clone(),  # (total_window, J, 6)
                    'root_translation': root_trans[start:end].clone(),  # (total_window, 3)
                    'style_idx': style_idx,
                    'anim_name': anim_name,
                    'frame_start': start,
                }
                self.samples.append(sample)

        print(f"Created {len(self.samples)} training samples from {len(animations)} animations")

    def _quaternion_to_6d(self, quats: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternions (T, J, 4) to 6D rotation (T, J, 6).
        """
        from ..utils.rotation import quaternion_to_rotation_6d

        T, J, _ = quats.shape

        # Reshape for batch processing: (T*J, 4)
        quats_flat = quats.reshape(-1, 4)

        # Convert
        rot_6d = quaternion_to_rotation_6d(quats_flat)

        # Reshape back: (T, J, 6)
        return rot_6d.reshape(T, J, 6)

    def _compute_trajectory(
        self,
        root_trans: torch.Tensor,  # (total_window, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute trajectory relative to the start of future frames.

        Returns:
            traj_translation: (2, traj_frames) - XZ plane
            traj_rotation: (6, traj_frames) - 6D rotation (facing direction)
        """
        # Get future portion
        future_trans = root_trans[self.past_frames:]  # (future_frames, 3)

        # Make relative to start of future
        start_pos = future_trans[0:1]  # (1, 3)
        relative_trans = future_trans - start_pos

        # Subsample for trajectory
        traj_frames = self.future_frames // 2
        traj_idx = torch.linspace(0, self.future_frames - 1, traj_frames).long()

        # XZ translation only - select subsampled frames, then X and Z columns
        sampled_trans = relative_trans[traj_idx]  # (traj_frames, 3)
        traj_translation = sampled_trans[:, [0, 2]].T  # (2, traj_frames)

        # Compute facing direction (simplified: use identity or compute from velocity)
        # For now, use identity rotation
        device = root_trans.device
        dtype = root_trans.dtype
        traj_rotation = torch.zeros(6, traj_frames, device=device, dtype=dtype)
        traj_rotation[0] = 1.0  # First column of identity: [1, 0, 0]
        traj_rotation[4] = 1.0  # Second column of identity: [0, 1, 0]

        # Optional: Compute facing from velocity
        if self.future_frames > 1:
            # Compute velocity direction (XZ plane)
            vel = future_trans[1:] - future_trans[:-1]  # (future_frames-1, 3)
            vel_xz = vel[:, [0, 2]]  # (future_frames-1, 2)

            # Average velocity gives facing direction
            avg_vel = vel_xz.mean(dim=0)
            if avg_vel.norm() > 1e-6:
                facing = avg_vel / avg_vel.norm()
                # Convert to 6D rotation (rotation around Y axis)
                # This is a simplification; proper implementation would use atan2
                pass

        return traj_translation, traj_rotation

    def _augment_sample(
        self,
        rotations_6d: torch.Tensor,
        root_trans: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation.

        Args:
            rotations_6d: (total_window, J, 6)
            root_trans: (total_window, 3)

        Returns:
            Augmented (rotations_6d, root_trans)
        """
        device = root_trans.device
        dtype = root_trans.dtype

        # Random Y-axis rotation
        angle = (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * self.rotation_augment_range
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Rotation matrix for Y-axis rotation
        R_y = torch.zeros(3, 3, device=device, dtype=dtype)
        R_y[0, 0] = cos_a
        R_y[0, 2] = sin_a
        R_y[1, 1] = 1.0
        R_y[2, 0] = -sin_a
        R_y[2, 2] = cos_a

        # Rotate root translations
        root_trans_aug = torch.einsum('ij,tj->ti', R_y, root_trans)

        # For joint rotations, we need to rotate only the root joint
        # Other joints are relative to their parent
        # This is a simplification; full implementation would apply to root rotation
        rotations_6d_aug = rotations_6d.clone()

        # Random translation offset (XZ plane)
        offset = torch.randn(3, device=device, dtype=dtype) * 0.1
        offset[1] = 0  # Keep Y unchanged
        root_trans_aug = root_trans_aug + offset

        return rotations_6d_aug, root_trans_aug

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        rotations_6d = sample['rotations_6d'].clone()  # (total_window, J, 6)
        root_trans = sample['root_translation'].clone()  # (total_window, 3)

        # Apply augmentation during training
        if self.augment and self.training_mode:
            rotations_6d, root_trans = self._augment_sample(rotations_6d, root_trans)

        # Split into past and future
        past_rot = rotations_6d[:self.past_frames]  # (past_frames, J, 6)
        future_rot = rotations_6d[self.past_frames:]  # (future_frames, J, 6)

        # Compute trajectory
        traj_trans, traj_rot = self._compute_trajectory(root_trans)

        # Transpose to match model expected format: (J, 6, T)
        past_motion = past_rot.permute(1, 2, 0)  # (J, 6, past_frames)
        future_motion = future_rot.permute(1, 2, 0)  # (J, 6, future_frames)

        return {
            'past_motion': past_motion,
            'future_motion': future_motion,
            'traj_translation': traj_trans,
            'traj_rotation': traj_rot,
            'style_idx': torch.tensor(sample['style_idx'], dtype=torch.long),
            'root_translation': root_trans[self.past_frames:].T,  # (3, future_frames)
        }

    @property
    def training_mode(self) -> bool:
        """Check if in training mode (for augmentation)."""
        return getattr(self, '_training_mode', True)

    def train(self, mode: bool = True):
        """Set training mode."""
        self._training_mode = mode

    def eval(self):
        """Set evaluation mode."""
        self._training_mode = False

    def get_num_joints(self) -> int:
        """Get number of joints from first sample."""
        if len(self.samples) > 0:
            return self.samples[0]['rotations_6d'].shape[1]
        return 0

    def get_style_names(self) -> List[str]:
        """Get list of style names."""
        return self.style_names


def create_motion_dataloader(
    character: Dict,
    past_frames: int = 5,
    future_frames: int = 20,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    stride: int = 3,
    augment: bool = True
) -> DataLoader:
    """
    Create a DataLoader for motion training.

    Args:
        character: Dict from load_rigged_character() containing 'animations'
        past_frames: Past context frames
        future_frames: Future prediction frames
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        stride: Sliding window stride
        augment: Whether to apply augmentation

    Returns:
        DataLoader for motion training
    """
    # Get animations from character dict
    animations = character.get('animations', {})

    if not animations:
        raise ValueError("No animations found in character dict")

    dataset = MotionSequenceDataset(
        animations=animations,
        past_frames=past_frames,
        future_frames=future_frames,
        stride=stride,
        augment=augment
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )


class MotionSequenceDatasetSimple(Dataset):
    """
    Simplified motion dataset for direct tensor input.

    Use this when animations are already preprocessed into tensors.
    """

    def __init__(
        self,
        rotations_6d: torch.Tensor,  # (num_anims, T, J, 6)
        root_translations: torch.Tensor,  # (num_anims, T, 3)
        style_labels: torch.Tensor,  # (num_anims,)
        past_frames: int = 5,
        future_frames: int = 20,
        stride: int = 3
    ):
        """
        Args:
            rotations_6d: All animation rotations
            root_translations: All root translations
            style_labels: Style label for each animation
            past_frames: Number of past frames
            future_frames: Number of future frames
            stride: Sliding window stride
        """
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.total_window = past_frames + future_frames

        self.samples = []

        num_anims = rotations_6d.shape[0]
        for anim_idx in range(num_anims):
            rot = rotations_6d[anim_idx]  # (T, J, 6)
            trans = root_translations[anim_idx]  # (T, 3)
            style = style_labels[anim_idx].item()

            num_frames = rot.shape[0]
            for start in range(0, num_frames - self.total_window + 1, stride):
                end = start + self.total_window
                self.samples.append({
                    'rotations_6d': rot[start:end],
                    'root_translation': trans[start:end],
                    'style_idx': style,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        rotations_6d = sample['rotations_6d']  # (total_window, J, 6)
        root_trans = sample['root_translation']  # (total_window, 3)

        # Split
        past_rot = rotations_6d[:self.past_frames]
        future_rot = rotations_6d[self.past_frames:]

        # Transpose: (J, 6, T)
        past_motion = past_rot.permute(1, 2, 0)
        future_motion = future_rot.permute(1, 2, 0)

        # Simplified trajectory
        traj_frames = self.future_frames // 2
        traj_trans = torch.zeros(2, traj_frames)
        traj_rot = torch.zeros(6, traj_frames)
        traj_rot[0] = 1.0
        traj_rot[4] = 1.0

        return {
            'past_motion': past_motion,
            'future_motion': future_motion,
            'traj_translation': traj_trans,
            'traj_rotation': traj_rot,
            'style_idx': torch.tensor(sample['style_idx'], dtype=torch.long),
        }

"""
Keyframe storage and management for PGA-SLAM.

Handles keyframe selection, storage, retrieval, and covisibility tracking.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from .config import KeyframeConfig, CameraIntrinsics
from .types import Frame, Keyframe


class KeyframeManager:
    """
    Manages keyframe selection, storage, and retrieval.

    Keyframe selection criteria:
    1. Sufficient translation from last keyframe
    2. Sufficient rotation from last keyframe
    3. Tracking error below threshold
    4. Minimum frame interval satisfied
    """

    def __init__(
        self,
        config: KeyframeConfig,
        intrinsics: CameraIntrinsics,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            config: Keyframe configuration
            intrinsics: Camera intrinsics
            device: Device for tensors
        """
        self.config = config
        self.intrinsics = intrinsics
        self.device = device

        self.keyframes: List[Keyframe] = []
        self._last_keyframe_id: int = -1

    def should_add_keyframe(
        self,
        current_translation: torch.Tensor,
        current_quaternion: torch.Tensor,
        tracking_error: float,
        frame_id: int
    ) -> bool:
        """
        Determine if current frame should become a keyframe.

        Args:
            current_translation: Current camera translation (3,)
            current_quaternion: Current camera quaternion (4,) as [w, x, y, z]
            tracking_error: Current tracking loss
            frame_id: Current frame ID

        Returns:
            True if frame should be added as keyframe
        """
        # First frame is always a keyframe
        if len(self.keyframes) == 0:
            return True

        # Check frame interval
        if frame_id - self._last_keyframe_id < self.config.min_frame_interval:
            return False

        # Check tracking error (too high means tracking is unreliable)
        if tracking_error > self.config.max_tracking_error:
            return False

        # Get last keyframe pose
        last_kf = self.keyframes[-1]

        # Compute translation difference
        translation_diff = (current_translation - last_kf.translation).norm()
        if translation_diff > self.config.min_translation:
            return True

        # Compute rotation difference using quaternion geodesic distance
        rotation_diff = self._quaternion_distance(
            current_quaternion, last_kf.quaternion
        )
        if rotation_diff > self.config.min_rotation:
            return True

        return False

    def _quaternion_distance(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor
    ) -> float:
        """
        Compute geodesic distance between two quaternions.

        Returns angle in radians.
        """
        # Ensure same sign (quaternion double cover)
        if (q1 * q2).sum() < 0:
            q2 = -q2

        # Dot product gives cos(angle/2)
        dot = (q1 * q2).sum().clamp(-1, 1)
        angle = 2 * torch.acos(dot.abs())

        return angle.item()

    def add_keyframe(
        self,
        frame: Frame,
        translation: torch.Tensor,
        quaternion: torch.Tensor,
        tracking_error: float
    ) -> Keyframe:
        """
        Add a new keyframe.

        Args:
            frame: The frame to add
            translation: Camera translation (3,)
            quaternion: Camera quaternion (4,)
            tracking_error: Tracking loss at this frame

        Returns:
            Created Keyframe
        """
        # Downsample images for storage
        rgb_down, depth_down = self._downsample_images(frame.rgb, frame.depth)

        # Backproject depth to 3D points
        points_3d, valid_mask = self._backproject_depth(depth_down)

        # Transform points to world frame
        points_world = self._transform_points_to_world(
            points_3d, translation, quaternion
        )

        # Create keyframe
        kf = Keyframe(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            rgb=rgb_down.to(self.device),
            depth=depth_down.to(self.device),
            translation=translation.to(self.device),
            quaternion=quaternion.to(self.device),
            points_3d=points_3d,
            points_world=points_world,
            valid_mask=valid_mask,
            tracking_error=tracking_error
        )

        self.keyframes.append(kf)
        self._last_keyframe_id = frame.frame_id

        # Update covisibility
        self._update_covisibility(kf)

        # Enforce max keyframes (remove oldest)
        if len(self.keyframes) > self.config.max_keyframes:
            self.keyframes.pop(0)

        return kf

    def _downsample_images(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample images for storage efficiency."""
        if self.config.downsample_factor == 1:
            return rgb.clone(), depth.clone()

        scale = 1.0 / self.config.downsample_factor

        # RGB: bilinear interpolation
        rgb_down = F.interpolate(
            rgb.permute(2, 0, 1).unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # Depth: nearest neighbor to avoid interpolating invalid values
        depth_down = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            scale_factor=scale,
            mode='nearest'
        ).squeeze()

        return rgb_down, depth_down

    def _backproject_depth(
        self,
        depth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backproject depth image to 3D points in camera frame.

        Args:
            depth: (H, W) depth in meters

        Returns:
            points_3d: (N, 3) valid 3D points in camera frame
            valid_mask: (H, W) boolean mask
        """
        H, W = depth.shape

        # Adjust intrinsics for downsampling
        scale = 1.0 / self.config.downsample_factor
        fx = self.intrinsics.fx * scale
        fy = self.intrinsics.fy * scale
        cx = self.intrinsics.cx * scale
        cy = self.intrinsics.cy * scale

        # Create pixel coordinates
        v, u = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

        # Valid depth mask
        valid_mask = (depth > 0.1) & (depth < 10.0)

        # Backproject all points
        z = depth.to(self.device)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points_3d_full = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
        points_3d = points_3d_full[valid_mask]  # (N, 3)

        return points_3d, valid_mask.to(self.device)

    def _transform_points_to_world(
        self,
        points_cam: torch.Tensor,
        translation: torch.Tensor,
        quaternion: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform points from camera frame to world frame.

        Args:
            points_cam: (N, 3) points in camera frame
            translation: (3,) camera translation
            quaternion: (4,) camera rotation as [w, x, y, z]

        Returns:
            points_world: (N, 3) points in world frame
        """
        from ..utils.quaternion import quaternion_to_matrix

        # Get rotation matrix
        R = quaternion_to_matrix(quaternion.unsqueeze(0)).squeeze(0)  # (3, 3)

        # Transform: p_world = R @ p_cam + t
        points_world = points_cam @ R.T + translation.unsqueeze(0)

        return points_world

    def _update_covisibility(self, new_kf: Keyframe):
        """
        Update covisibility information for keyframes.

        Two keyframes are covisible if they observe similar regions.
        """
        if len(self.keyframes) < 2:
            return

        # Compute covisibility with other keyframes
        new_pos = new_kf.translation

        for kf in self.keyframes[:-1]:  # Exclude the just-added keyframe
            distance = (new_pos - kf.translation).norm()

            # Simple distance-based covisibility
            if distance < 2.0:  # Within 2 meters
                if new_kf.frame_id not in kf.covisible_keyframes:
                    kf.covisible_keyframes.append(new_kf.frame_id)
                if kf.frame_id not in new_kf.covisible_keyframes:
                    new_kf.covisible_keyframes.append(kf.frame_id)

    def get_recent_keyframes(self, n: int = 10) -> List[Keyframe]:
        """Get the N most recent keyframes."""
        return self.keyframes[-n:]

    def get_keyframe_by_id(self, frame_id: int) -> Optional[Keyframe]:
        """Get keyframe by frame ID."""
        for kf in self.keyframes:
            if kf.frame_id == frame_id:
                return kf
        return None

    def get_keyframe_by_index(self, index: int) -> Optional[Keyframe]:
        """Get keyframe by index in keyframe list."""
        if 0 <= index < len(self.keyframes):
            return self.keyframes[index]
        return None

    def get_covisible_keyframes(self, frame_id: int) -> List[Keyframe]:
        """Get keyframes covisible with given frame."""
        kf = self.get_keyframe_by_id(frame_id)
        if kf is None:
            return []

        covisible = []
        for cov_id in kf.covisible_keyframes:
            cov_kf = self.get_keyframe_by_id(cov_id)
            if cov_kf is not None:
                covisible.append(cov_kf)

        return covisible

    def get_all_points_world(self) -> torch.Tensor:
        """
        Get all 3D points from all keyframes in world coordinates.

        Returns:
            points: (M, 3) concatenated points from all keyframes
        """
        all_points = []
        for kf in self.keyframes:
            if kf.points_world is not None:
                all_points.append(kf.points_world)

        if len(all_points) == 0:
            return torch.empty(0, 3, device=self.device)

        return torch.cat(all_points, dim=0)

    def get_trajectory(self) -> List[torch.Tensor]:
        """Get list of keyframe positions."""
        return [kf.translation for kf in self.keyframes]

    def num_keyframes(self) -> int:
        """Get number of keyframes."""
        return len(self.keyframes)

    def clear(self):
        """Clear all keyframes."""
        self.keyframes.clear()
        self._last_keyframe_id = -1

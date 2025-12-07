"""
Type definitions for PGA-SLAM.

Defines data structures for frames, keyframes, and results.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, NamedTuple, Union
import torch


class Frame(NamedTuple):
    """A single RGB-D frame from a sensor."""
    rgb: torch.Tensor           # (H, W, 3) normalized [0, 1]
    depth: torch.Tensor         # (H, W) in meters
    timestamp: float            # Timestamp in seconds
    frame_id: int               # Unique frame identifier


@dataclass
class Keyframe:
    """
    A stored keyframe for mapping.

    Contains the frame data, estimated pose, and derived 3D points.
    """
    frame_id: int
    timestamp: float

    # Images (potentially downsampled)
    rgb: torch.Tensor           # (H, W, 3)
    depth: torch.Tensor         # (H, W) in meters

    # Pose as translation + quaternion (to avoid circular import with Motor)
    translation: torch.Tensor   # (3,)
    quaternion: torch.Tensor    # (4,) as [w, x, y, z]

    # Derived data (computed once)
    points_3d: Optional[torch.Tensor] = None    # (N, 3) backprojected points in camera frame
    points_world: Optional[torch.Tensor] = None # (N, 3) points in world frame
    valid_mask: Optional[torch.Tensor] = None   # (H, W) valid depth mask

    # Tracking quality
    tracking_error: float = 0.0

    # Covisibility
    covisible_keyframes: List[int] = field(default_factory=list)

    def to_device(self, device: torch.device) -> 'Keyframe':
        """Move all tensors to specified device."""
        return Keyframe(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            rgb=self.rgb.to(device),
            depth=self.depth.to(device),
            translation=self.translation.to(device),
            quaternion=self.quaternion.to(device),
            points_3d=self.points_3d.to(device) if self.points_3d is not None else None,
            points_world=self.points_world.to(device) if self.points_world is not None else None,
            valid_mask=self.valid_mask.to(device) if self.valid_mask is not None else None,
            tracking_error=self.tracking_error,
            covisible_keyframes=self.covisible_keyframes.copy()
        )

    def get_pose_matrix(self) -> torch.Tensor:
        """Get 4x4 camera-to-world transformation matrix."""
        from ..utils.quaternion import quaternion_to_matrix

        R = quaternion_to_matrix(self.quaternion.unsqueeze(0)).squeeze(0)  # (3, 3)
        M = torch.eye(4, device=self.translation.device, dtype=self.translation.dtype)
        M[:3, :3] = R
        M[:3, 3] = self.translation
        return M


@dataclass
class TrackingResult:
    """Result from pose tracking."""
    # Pose as translation + quaternion
    translation: torch.Tensor           # (3,)
    quaternion: torch.Tensor            # (4,) as [w, x, y, z]

    # Optimization status
    converged: bool                     # Did optimization converge
    num_iterations: int                 # Iterations used
    final_loss: float                   # Final loss value
    inlier_ratio: float                 # Ratio of inlier points

    # Per-loss breakdown
    loss_breakdown: Dict[str, float] = field(default_factory=dict)

    def get_pose_matrix(self) -> torch.Tensor:
        """Get 4x4 camera-to-world transformation matrix."""
        from ..utils.quaternion import quaternion_to_matrix

        R = quaternion_to_matrix(self.quaternion.unsqueeze(0)).squeeze(0)
        M = torch.eye(4, device=self.translation.device, dtype=self.translation.dtype)
        M[:3, :3] = R
        M[:3, 3] = self.translation
        return M


@dataclass
class MappingResult:
    """Result from map update."""
    num_iterations: int
    final_loss: float
    loss_history: List[float] = field(default_factory=list)

    # Per-loss breakdown
    loss_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class LoopClosureResult:
    """Result from loop closure detection and optimization."""
    detected: bool
    source_keyframe_id: int
    target_keyframe_id: int

    # Relative transform
    relative_translation: Optional[torch.Tensor] = None
    relative_quaternion: Optional[torch.Tensor] = None

    # Quality metrics
    confidence: float = 0.0
    num_inliers: int = 0


@dataclass
class SLAMState:
    """Current state of the SLAM system."""
    # Current pose estimate
    current_translation: torch.Tensor
    current_quaternion: torch.Tensor

    # Map model (reference, not owned)
    map_model: torch.nn.Module

    # Keyframes
    keyframes: List[Keyframe]

    # Statistics
    total_frames: int
    total_keyframes: int
    tracking_fps: float

    # Status
    is_initialized: bool
    is_lost: bool = False

    def get_trajectory(self) -> List[torch.Tensor]:
        """Get list of keyframe positions."""
        return [kf.translation for kf in self.keyframes]

    def get_current_pose_matrix(self) -> torch.Tensor:
        """Get current 4x4 pose matrix."""
        from ..utils.quaternion import quaternion_to_matrix

        R = quaternion_to_matrix(self.current_quaternion.unsqueeze(0)).squeeze(0)
        M = torch.eye(4, device=self.current_translation.device)
        M[:3, :3] = R
        M[:3, 3] = self.current_translation
        return M


# Type aliases for clarity
PoseType = Union[
    torch.Tensor,  # 4x4 matrix
    tuple  # (translation, quaternion)
]

PointCloud = torch.Tensor  # (N, 3)
DepthMap = torch.Tensor    # (H, W)
RGBImage = torch.Tensor    # (H, W, 3)

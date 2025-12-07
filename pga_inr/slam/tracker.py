"""
Pose optimization for SLAM tracking.

Optimizes camera pose against a fixed neural map using gradient descent
on the SE(3) Lie algebra.
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from .config import TrackerConfig, CameraIntrinsics
from .types import Frame, TrackingResult
from .losses import DirectDepthSDFLoss, TruncatedSDFLoss, FreeSpaceLoss
from ..utils.quaternion import quaternion_to_matrix, matrix_to_quaternion, normalize_quaternion


class PoseOptimizer(nn.Module):
    """
    Optimizes camera pose given a fixed neural map.

    Uses gradient descent on SE(3) parameters (translation + quaternion)
    to find the pose that best explains the observed RGB-D frame.

    Key features:
    - Direct depth supervision (no sphere tracing required)
    - Lie algebra parameterization for stable optimization
    - Support for stratified and importance sampling
    """

    def __init__(
        self,
        config: TrackerConfig,
        intrinsics: CameraIntrinsics,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            config: Tracker configuration
            intrinsics: Camera intrinsics
            device: Device for computation
        """
        super().__init__()
        self.config = config
        self.intrinsics = intrinsics
        self.device = device

        # Learnable pose parameters
        # Translation: 3D vector
        # Rotation: quaternion [w, x, y, z]
        self._translation = nn.Parameter(torch.zeros(3, device=device))
        self._quaternion = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        )

        # Losses
        self.depth_sdf_loss = DirectDepthSDFLoss(
            truncation=config.truncation_distance,
            use_huber=config.use_huber_loss,
            huber_delta=config.huber_delta
        )
        self.truncated_sdf_loss = TruncatedSDFLoss(
            truncation=config.truncation_distance,
            use_huber=config.use_huber_loss
        )
        self.free_space_loss = FreeSpaceLoss(margin=0.01)

    def set_initial_pose(
        self,
        translation: torch.Tensor,
        quaternion: torch.Tensor
    ):
        """
        Set initial pose estimate.

        Args:
            translation: (3,) initial translation
            quaternion: (4,) initial quaternion [w, x, y, z]
        """
        self._translation.data = translation.to(self.device)
        self._quaternion.data = quaternion.to(self.device)

    def get_current_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current pose as (translation, quaternion)."""
        q_normalized = normalize_quaternion(self._quaternion.unsqueeze(0)).squeeze(0)
        return self._translation.detach(), q_normalized.detach()

    def get_pose_matrix(self) -> torch.Tensor:
        """Get current pose as 4x4 matrix."""
        t = self._translation
        q = normalize_quaternion(self._quaternion.unsqueeze(0)).squeeze(0)
        R = quaternion_to_matrix(q.unsqueeze(0)).squeeze(0)

        M = torch.eye(4, device=self.device)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def forward(
        self,
        map_model: nn.Module,
        frame: Frame,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute tracking loss for current pose parameters.

        Args:
            map_model: Neural implicit map (frozen)
            frame: Current RGB-D frame
            num_samples: Number of sample points (default from config)

        Returns:
            (total_loss, loss_dict)
        """
        num_samples = num_samples or self.config.num_depth_samples

        # Get current pose
        t = self._translation
        q = normalize_quaternion(self._quaternion.unsqueeze(0)).squeeze(0)
        R = quaternion_to_matrix(q.unsqueeze(0)).squeeze(0)  # (3, 3)

        # Sample points from observed depth
        surface_points_cam, free_points_cam = self._sample_points(
            frame.depth, num_samples
        )

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # === Surface points loss ===
        # Transform surface points to world frame
        surface_points_world = surface_points_cam @ R.T + t.unsqueeze(0)

        # Query SDF at world points (no observer_pose since points are in world frame)
        with torch.enable_grad():
            surface_outputs = map_model(
                surface_points_world.unsqueeze(0),
                observer_pose=None
            )
        surface_sdf = surface_outputs['sdf'].squeeze(0)

        # Surface points should have SDF = 0
        sdf_loss = self.depth_sdf_loss(surface_sdf, target_sdf=None)
        total_loss = total_loss + self.config.lambda_sdf * sdf_loss
        loss_dict['sdf'] = sdf_loss.item()

        # === Free space points loss ===
        if free_points_cam is not None and len(free_points_cam) > 0:
            free_points_world = free_points_cam @ R.T + t.unsqueeze(0)

            with torch.enable_grad():
                free_outputs = map_model(
                    free_points_world.unsqueeze(0),
                    observer_pose=None
                )
            free_sdf = free_outputs['sdf'].squeeze(0)

            free_loss = self.free_space_loss(free_sdf)
            total_loss = total_loss + self.config.lambda_depth * 0.5 * free_loss
            loss_dict['free_space'] = free_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict

    def _sample_points(
        self,
        depth: torch.Tensor,
        num_samples: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample 3D points from depth image.

        Returns:
            surface_points: Points at observed depth (in camera frame)
            free_points: Points between camera and surface (in camera frame)
        """
        H, W = depth.shape
        K = self.intrinsics

        # Create pixel grid
        v, u = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

        # Valid depth mask
        valid = (depth > 0.1) & (depth < 10.0)
        valid_indices = torch.where(valid.flatten())[0]

        if len(valid_indices) == 0:
            # No valid depth - return empty tensors
            return torch.empty(0, 3, device=self.device), None

        # Split samples: 70% surface, 30% free space
        n_surface = int(num_samples * 0.7)
        n_free = num_samples - n_surface

        # === Sample surface points ===
        if len(valid_indices) < n_surface:
            surface_indices = valid_indices
        else:
            perm = torch.randperm(len(valid_indices), device=self.device)
            surface_indices = valid_indices[perm[:n_surface]]

        # Get sampled pixels
        u_surf = u.flatten()[surface_indices]
        v_surf = v.flatten()[surface_indices]
        z_surf = depth.flatten()[surface_indices].to(self.device)

        # Backproject to 3D
        x_surf = (u_surf - K.cx) * z_surf / K.fx
        y_surf = (v_surf - K.cy) * z_surf / K.fy
        surface_points = torch.stack([x_surf, y_surf, z_surf], dim=-1)

        # === Sample free space points ===
        if n_free > 0:
            if len(valid_indices) < n_free:
                free_indices = valid_indices
            else:
                perm = torch.randperm(len(valid_indices), device=self.device)
                free_indices = valid_indices[perm[:n_free]]

            u_free = u.flatten()[free_indices]
            v_free = v.flatten()[free_indices]
            z_max = depth.flatten()[free_indices].to(self.device)

            # Sample random depth between 0.5 and 0.9 of observed depth
            fractions = torch.rand(len(free_indices), device=self.device) * 0.4 + 0.5
            z_free = z_max * fractions

            x_free = (u_free - K.cx) * z_free / K.fx
            y_free = (v_free - K.cy) * z_free / K.fy
            free_points = torch.stack([x_free, y_free, z_free], dim=-1)
        else:
            free_points = None

        return surface_points, free_points

    def optimize(
        self,
        map_model: nn.Module,
        frame: Frame,
        initial_translation: Optional[torch.Tensor] = None,
        initial_quaternion: Optional[torch.Tensor] = None
    ) -> TrackingResult:
        """
        Full pose optimization loop.

        Args:
            map_model: Neural implicit map (will be frozen)
            frame: RGB-D frame to track
            initial_translation: Initial translation estimate
            initial_quaternion: Initial quaternion estimate

        Returns:
            TrackingResult with optimized pose
        """
        # Freeze map
        map_model.eval()
        for param in map_model.parameters():
            param.requires_grad = False

        # Initialize pose
        if initial_translation is not None:
            self._translation.data = initial_translation.to(self.device)
        else:
            self._translation.data.zero_()

        if initial_quaternion is not None:
            self._quaternion.data = initial_quaternion.to(self.device)
        else:
            self._quaternion.data = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device
            )

        # Ensure parameters require grad
        self._translation.requires_grad_(True)
        self._quaternion.requires_grad_(True)

        # Optimizer for pose parameters only
        optimizer = torch.optim.Adam(
            [self._translation, self._quaternion],
            lr=self.config.learning_rate
        )

        prev_loss = float('inf')
        converged = False
        final_loss_dict = {}

        for iteration in range(self.config.num_iterations):
            optimizer.zero_grad()

            loss, loss_dict = self.forward(map_model, frame)

            loss.backward()
            optimizer.step()

            # Normalize quaternion after update
            with torch.no_grad():
                self._quaternion.data = normalize_quaternion(
                    self._quaternion.data.unsqueeze(0)
                ).squeeze(0)

            # Check convergence
            loss_change = abs(prev_loss - loss.item())
            if loss_change < self.config.convergence_threshold:
                converged = True
                final_loss_dict = loss_dict
                break

            prev_loss = loss.item()
            final_loss_dict = loss_dict

        # Get final pose
        t, q = self.get_current_pose()

        # Unfreeze map for later use
        for param in map_model.parameters():
            param.requires_grad = True

        return TrackingResult(
            translation=t,
            quaternion=q,
            converged=converged,
            num_iterations=iteration + 1,
            final_loss=prev_loss,
            inlier_ratio=1.0,  # TODO: compute actual inlier ratio
            loss_breakdown=final_loss_dict
        )


class MotionModel:
    """
    Simple motion model for pose prediction.

    Uses constant velocity assumption to predict next pose
    from previous poses.
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        self.device = device
        self._prev_translation: Optional[torch.Tensor] = None
        self._prev_quaternion: Optional[torch.Tensor] = None
        self._velocity_translation: Optional[torch.Tensor] = None
        self._velocity_angular: Optional[torch.Tensor] = None

    def update(self, translation: torch.Tensor, quaternion: torch.Tensor):
        """Update motion model with new pose."""
        if self._prev_translation is not None:
            # Compute velocity
            self._velocity_translation = translation - self._prev_translation

        self._prev_translation = translation.clone()
        self._prev_quaternion = quaternion.clone()

    def predict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next pose using constant velocity model.

        Returns:
            (predicted_translation, predicted_quaternion)
        """
        if self._prev_translation is None:
            # No history - return identity
            return (
                torch.zeros(3, device=self.device),
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            )

        if self._velocity_translation is None:
            # Only one pose so far - return it
            return self._prev_translation, self._prev_quaternion

        # Apply velocity
        pred_translation = self._prev_translation + self._velocity_translation

        # For now, keep same rotation (TODO: angular velocity)
        pred_quaternion = self._prev_quaternion

        return pred_translation, pred_quaternion

    def reset(self):
        """Reset motion model."""
        self._prev_translation = None
        self._prev_quaternion = None
        self._velocity_translation = None
        self._velocity_angular = None

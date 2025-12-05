"""
PGA Motor Layer for coordinate-free neural networks.

Transforms query points from world/global space to local/object space
using the PGA sandwich product. This enables observer-independent
learning of implicit neural representations.
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.quaternion import quaternion_to_matrix, normalize_quaternion


class PGAMotorLayer(nn.Module):
    """
    Transforms query points from world space to local object frame.

    Implements: X_local = ~M_observer * P * M_observer (sandwich product)

    For efficiency, uses 4x4 homogeneous matrix representation which is
    isomorphic to the motor algebra for point transformations.

    This layer is the key innovation of PGA-INR: by transforming points
    to a canonical object frame, the network learns geometry independent
    of observer position/orientation.
    """

    def __init__(self):
        super().__init__()

    def build_motor_matrix(
        self,
        translation: torch.Tensor,
        quaternion: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct the 4x4 homogeneous transformation matrix from motor parameters.

        Args:
            translation: Translation vector of shape (B, 3)
            quaternion: Rotation quaternion of shape (B, 4) as [w, x, y, z]

        Returns:
            M: Transformation matrix of shape (B, 4, 4)
        """
        B = translation.shape[0]
        device = translation.device
        dtype = translation.dtype

        # Normalize quaternion to ensure valid rotation
        q = normalize_quaternion(quaternion)

        # Convert quaternion to rotation matrix
        R = quaternion_to_matrix(q)  # (B, 3, 3)

        # Construct 4x4 homogeneous matrix
        M = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        M[:, :3, :3] = R
        M[:, :3, 3] = translation

        return M

    def forward(
        self,
        points: torch.Tensor,
        motor_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Transform points from world space to local (object) frame.

        This implements the inverse transformation: if the object is at
        position T with rotation R, we compute R^{-1}(p - T) to get
        the point coordinates in the object's rest frame.

        Args:
            points: Query points in world space of shape (B, N, 3)
            motor_params: Tuple of (translation, quaternion) defining object pose.
                         - translation: (B, 3)
                         - quaternion: (B, 4) as [w, x, y, z]
                         If None, returns points unchanged (identity transform).

        Returns:
            Local coordinates of shape (B, N, 3)
        """
        if motor_params is None:
            return points

        B, N, _ = points.shape
        translation, quaternion = motor_params

        # Handle single motor for entire batch
        if translation.dim() == 1:
            translation = translation.unsqueeze(0).expand(B, -1)
        if quaternion.dim() == 1:
            quaternion = quaternion.unsqueeze(0).expand(B, -1)

        # Build world-to-local transformation (analytic inverse)
        # M = [R  t]
        #     [0  1]
        # M^-1 = [R^T  -R^T * t]
        #        [0       1    ]
        
        q = normalize_quaternion(quaternion)
        R = quaternion_to_matrix(q) # (B, 3, 3)
        R_inv = R.transpose(1, 2) # (B, 3, 3)
        
        # t_inv = -R^T * t
        # t: (B, 3) -> (B, 3, 1)
        t_inv = -torch.bmm(R_inv, translation.unsqueeze(-1)).squeeze(-1) # (B, 3)
        
        # Construct M_inv
        M_world_to_local = torch.eye(4, device=points.device, dtype=points.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        M_world_to_local[:, :3, :3] = R_inv
        M_world_to_local[:, :3, 3] = t_inv

        # Homogenize points (PGA points have homogeneous coordinate 1)
        ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, ones], dim=-1)  # (B, N, 4)

        # Apply transformation: points_local_h = M_inv @ points
        # M is (B, 4, 4), points_h is (B, N, 4)
        # We want P_local = P_world @ M_inv.T
        points_local_h = points_h @ M_world_to_local.transpose(1, 2)

        # Return 3D coordinates (drop homogeneous component)
        return points_local_h[..., :3]


class LearnableMotorLayer(nn.Module):
    """
    Motor layer with learnable pose parameters.

    Useful for auto-decoding setups where the object pose is optimized
    during training.
    """

    def __init__(
        self,
        init_translation: Optional[torch.Tensor] = None,
        init_quaternion: Optional[torch.Tensor] = None
    ):
        """
        Args:
            init_translation: Initial translation (3,)
            init_quaternion: Initial rotation quaternion (4,) as [w, x, y, z]
        """
        super().__init__()

        if init_translation is None:
            init_translation = torch.zeros(3)
        if init_quaternion is None:
            init_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity

        self.translation = nn.Parameter(init_translation)
        self.quaternion = nn.Parameter(init_quaternion)
        self.motor_layer = PGAMotorLayer()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Transform points using the learned motor.

        Args:
            points: Query points of shape (B, N, 3) or (N, 3)

        Returns:
            Transformed points in local frame
        """
        # Handle unbatched input
        if points.dim() == 2:
            points = points.unsqueeze(0)

        B = points.shape[0]

        # Expand learned parameters to batch size
        t = self.translation.unsqueeze(0).expand(B, -1)
        q = normalize_quaternion(self.quaternion.unsqueeze(0).expand(B, -1))

        return self.motor_layer(points, (t, q))

    def get_motor_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current motor parameters."""
        return self.translation, normalize_quaternion(self.quaternion)


class BatchMotorLayer(nn.Module):
    """
    Motor layer that handles multiple objects with different poses.

    Each object in a scene can have its own pose, and we transform
    query points into each object's local frame in parallel.
    """

    def __init__(self):
        super().__init__()
        self.motor_layer = PGAMotorLayer()

    def forward(
        self,
        points: torch.Tensor,
        translations: torch.Tensor,
        quaternions: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform points for multiple objects.

        Args:
            points: Query points of shape (B, N, 3) or (N, 3)
            translations: Object translations of shape (B, K, 3) for K objects
            quaternions: Object rotations of shape (B, K, 4)

        Returns:
            Transformed points of shape (B, K, N, 3)
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)

        B, N, _ = points.shape
        K = translations.shape[1]

        # Expand points for each object
        points_expanded = points.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, N, 3)
        points_flat = points_expanded.reshape(B * K, N, 3)

        # Flatten object poses
        t_flat = translations.reshape(B * K, 3)
        q_flat = quaternions.reshape(B * K, 4)

        # Transform all at once
        local_points_flat = self.motor_layer(points_flat, (t_flat, q_flat))

        # Reshape back
        return local_points_flat.reshape(B, K, N, 3)


class RelativeMotorLayer(nn.Module):
    """
    Computes relative transformation between two poses.

    Useful for scene composition where we need the transformation
    from one object's frame to another's.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        points: torch.Tensor,
        source_pose: Tuple[torch.Tensor, torch.Tensor],
        target_pose: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Transform points from source frame to target frame.

        Args:
            points: Points in source frame (B, N, 3)
            source_pose: (translation, quaternion) of source frame
            target_pose: (translation, quaternion) of target frame

        Returns:
            Points in target frame (B, N, 3)
        """
        B, N, _ = points.shape
        src_t, src_q = source_pose
        tgt_t, tgt_q = target_pose

        # Handle single pose for batch
        if src_t.dim() == 1:
            src_t = src_t.unsqueeze(0).expand(B, -1)
            src_q = src_q.unsqueeze(0).expand(B, -1)
        if tgt_t.dim() == 1:
            tgt_t = tgt_t.unsqueeze(0).expand(B, -1)
            tgt_q = tgt_q.unsqueeze(0).expand(B, -1)

        # Build transformation matrices
        src_q = normalize_quaternion(src_q)
        tgt_q = normalize_quaternion(tgt_q)

        M_src = self._build_matrix(src_t, src_q)
        M_tgt = self._build_matrix(tgt_t, tgt_q)

        # Relative transformation: M_rel = M_tgt^{-1} @ M_src
        M_rel = torch.bmm(torch.inverse(M_tgt), M_src)

        # Apply to points
        ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, ones], dim=-1)
        result_h = torch.bmm(M_rel, points_h.transpose(1, 2)).transpose(1, 2)

        return result_h[..., :3]

    def _build_matrix(
        self,
        translation: torch.Tensor,
        quaternion: torch.Tensor
    ) -> torch.Tensor:
        """Build 4x4 transformation matrix."""
        B = translation.shape[0]
        R = quaternion_to_matrix(quaternion)
        M = torch.eye(4, device=translation.device, dtype=translation.dtype)
        M = M.unsqueeze(0).expand(B, -1, -1).clone()
        M[:, :3, :3] = R
        M[:, :3, 3] = translation
        return M


class IdentityMotorLayer(nn.Module):
    """
    Identity motor layer (no transformation).

    Useful as a placeholder or for ablation studies.
    """

    def forward(
        self,
        points: torch.Tensor,
        motor_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Return points unchanged."""
        return points

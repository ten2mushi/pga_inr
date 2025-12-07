"""
6D rotation representation utilities for motion diffusion.

The 6D continuous rotation representation uses the first two columns
of a rotation matrix, which provides a continuous and differentiable
parameterization of SO(3) without singularities.

Reference:
    Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    https://arxiv.org/abs/1812.07035

All functions support batched inputs with shape (..., 6) or (..., 3, 3).
"""

from typing import Tuple
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.

    The 6D representation consists of the first two columns of the rotation
    matrix, which are orthonormalized using Gram-Schmidt.

    Args:
        rot_6d: 6D rotation representation of shape (..., 6)

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    batch_shape = rot_6d.shape[:-1]

    # Split into two 3D vectors (first two columns of rotation matrix)
    a1 = rot_6d[..., :3]  # First column
    a2 = rot_6d[..., 3:6]  # Second column

    # Gram-Schmidt orthonormalization
    # b1 = normalize(a1)
    b1 = F.normalize(a1, p=2, dim=-1, eps=1e-12)

    # b2 = normalize(a2 - (b1 · a2) * b1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, p=2, dim=-1, eps=1e-12)

    # b3 = b1 × b2 (cross product)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack columns into rotation matrix
    R = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)

    return R


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 6D representation.

    Extracts the first two columns of the rotation matrix.

    Args:
        R: Rotation matrix of shape (..., 3, 3)

    Returns:
        6D rotation representation of shape (..., 6)
    """
    # Extract first two columns and flatten
    col1 = R[..., :, 0]  # (..., 3)
    col2 = R[..., :, 1]  # (..., 3)

    return torch.cat([col1, col2], dim=-1)  # (..., 6)


def quaternion_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 6D rotation representation.

    Args:
        quat: Quaternion of shape (..., 4) as [w, x, y, z]

    Returns:
        6D rotation representation of shape (..., 6)
    """
    from .quaternion import quaternion_to_matrix

    R = quaternion_to_matrix(quat)  # (..., 3, 3)
    return matrix_to_rotation_6d(R)


def rotation_6d_to_quaternion(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to quaternion.

    Args:
        rot_6d: 6D rotation representation of shape (..., 6)

    Returns:
        Quaternion of shape (..., 4) as [w, x, y, z]
    """
    from .quaternion import matrix_to_quaternion

    R = rotation_6d_to_matrix(rot_6d)  # (..., 3, 3)
    return matrix_to_quaternion(R)


def rotation_6d_to_axis_angle(rot_6d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 6D rotation representation to axis-angle.

    Args:
        rot_6d: 6D rotation representation of shape (..., 6)

    Returns:
        axis: Rotation axis of shape (..., 3)
        angle: Rotation angle in radians of shape (...)
    """
    from .quaternion import quaternion_to_axis_angle

    quat = rotation_6d_to_quaternion(rot_6d)
    return quaternion_to_axis_angle(quat)


def axis_angle_to_rotation_6d(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to 6D rotation.

    Args:
        axis: Rotation axis of shape (..., 3), will be normalized
        angle: Rotation angle in radians of shape (...) or (..., 1)

    Returns:
        6D rotation representation of shape (..., 6)
    """
    from .quaternion import quaternion_from_axis_angle

    quat = quaternion_from_axis_angle(axis, angle)
    return quaternion_to_rotation_6d(quat)


def random_rotation_6d(
    batch_size: int,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Generate random 6D rotation representations.

    Args:
        batch_size: Number of random rotations to generate
        device: Torch device
        dtype: Torch dtype

    Returns:
        Random 6D rotation representations of shape (batch_size, 6)
    """
    from .quaternion import random_quaternion

    quats = random_quaternion(batch_size, device=device, dtype=dtype)
    return quaternion_to_rotation_6d(quats)


def identity_rotation_6d(
    batch_size: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Create identity rotation in 6D representation.

    Args:
        batch_size: Number of identity rotations to create
        device: Torch device
        dtype: Torch dtype

    Returns:
        Identity 6D rotations of shape (batch_size, 6)
    """
    # Identity rotation matrix: I_3x3
    # First two columns: [1,0,0] and [0,1,0]
    rot_6d = torch.zeros(batch_size, 6, device=device, dtype=dtype)
    rot_6d[:, 0] = 1.0  # First column: [1, 0, 0]
    rot_6d[:, 4] = 1.0  # Second column: [0, 1, 0]

    return rot_6d


def rotation_6d_compose(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """
    Compose two rotations in 6D representation.

    Computes R1 @ R2 in matrix form and converts back to 6D.

    Args:
        rot1: First rotation of shape (..., 6)
        rot2: Second rotation of shape (..., 6)

    Returns:
        Composed rotation of shape (..., 6)
    """
    R1 = rotation_6d_to_matrix(rot1)
    R2 = rotation_6d_to_matrix(rot2)
    R = R1 @ R2
    return matrix_to_rotation_6d(R)


def rotation_6d_inverse(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse (transpose) of rotation in 6D representation.

    Args:
        rot_6d: 6D rotation representation of shape (..., 6)

    Returns:
        Inverse rotation of shape (..., 6)
    """
    R = rotation_6d_to_matrix(rot_6d)
    R_inv = R.transpose(-1, -2)  # Rotation matrices are orthogonal
    return matrix_to_rotation_6d(R_inv)


def rotation_6d_apply(rot_6d: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Apply 6D rotation to 3D points.

    Args:
        rot_6d: 6D rotation representation of shape (..., 6)
        points: 3D points of shape (..., 3)

    Returns:
        Rotated points of shape (..., 3)
    """
    R = rotation_6d_to_matrix(rot_6d)  # (..., 3, 3)
    return torch.einsum('...ij,...j->...i', R, points)


def rotation_6d_slerp(
    rot0: torch.Tensor,
    rot1: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Spherical linear interpolation between two 6D rotations.

    Converts to quaternions, performs SLERP, and converts back.

    Args:
        rot0: Start rotation of shape (..., 6)
        rot1: End rotation of shape (..., 6)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated rotation of shape (..., 6)
    """
    from .quaternion import quaternion_slerp

    q0 = rotation_6d_to_quaternion(rot0)
    q1 = rotation_6d_to_quaternion(rot1)
    q_interp = quaternion_slerp(q0, q1, t)
    return quaternion_to_rotation_6d(q_interp)


def rotation_6d_distance(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance between two 6D rotations.

    The geodesic distance is the angle of the rotation needed to
    go from rot1 to rot2.

    Args:
        rot1: First rotation of shape (..., 6)
        rot2: Second rotation of shape (..., 6)

    Returns:
        Angular distance in radians of shape (...)
    """
    from .quaternion import quaternion_angle_distance

    q1 = rotation_6d_to_quaternion(rot1)
    q2 = rotation_6d_to_quaternion(rot2)
    return quaternion_angle_distance(q1, q2)


def normalize_rotation_6d(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Normalize 6D rotation to ensure it represents a valid rotation.

    This applies Gram-Schmidt orthonormalization by converting to
    matrix form and back.

    Args:
        rot_6d: Potentially unnormalized 6D rotation of shape (..., 6)

    Returns:
        Normalized 6D rotation of shape (..., 6)
    """
    R = rotation_6d_to_matrix(rot_6d)  # Gram-Schmidt applied here
    return matrix_to_rotation_6d(R)


def rotation_6d_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Create 6D rotation that aligns v1 to v2.

    Finds the rotation R such that R @ v1 = v2 (after normalization).

    Args:
        v1: Source vector of shape (..., 3)
        v2: Target vector of shape (..., 3)

    Returns:
        6D rotation representation of shape (..., 6)
    """
    v1 = F.normalize(v1, p=2, dim=-1, eps=1e-12)
    v2 = F.normalize(v2, p=2, dim=-1, eps=1e-12)

    # Cross product gives rotation axis
    axis = torch.cross(v1, v2, dim=-1)
    axis_norm = axis.norm(dim=-1, keepdim=True)

    # Dot product gives cos(angle)
    cos_angle = (v1 * v2).sum(dim=-1)

    # Handle parallel vectors (no rotation or 180 degree rotation)
    # For now, use quaternion method for robustness
    from .quaternion import normalize_quaternion

    # Build quaternion: q = [cos(angle/2), sin(angle/2) * axis]
    # Using half-angle formulas
    angle = torch.acos(cos_angle.clamp(-1 + 1e-7, 1 - 1e-7))

    half_angle = angle / 2
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)

    # Normalize axis
    axis = F.normalize(axis, p=2, dim=-1, eps=1e-12)

    # Handle degenerate case (parallel vectors)
    degenerate = axis_norm.squeeze(-1) < 1e-6
    if degenerate.any():
        # Find any perpendicular vector
        perp = torch.zeros_like(v1)
        perp[..., 0] = -v1[..., 1]
        perp[..., 1] = v1[..., 0]
        perp = F.normalize(perp, p=2, dim=-1, eps=1e-12)
        axis = torch.where(degenerate.unsqueeze(-1), perp, axis)

    # Build quaternion
    quat = torch.cat([
        cos_half.unsqueeze(-1),
        sin_half.unsqueeze(-1) * axis
    ], dim=-1)

    quat = normalize_quaternion(quat)
    return quaternion_to_rotation_6d(quat)

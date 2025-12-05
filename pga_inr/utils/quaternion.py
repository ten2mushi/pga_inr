"""
Quaternion operations for rotation handling in PGA-INR.

Quaternions are represented as (w, x, y, z) where w is the scalar part
and (x, y, z) is the vector part. This follows the convention:
    q = w + xi + yj + zk

All operations support batched inputs with shape (..., 4).
"""

from typing import Tuple, Union
import torch
import torch.nn.functional as F


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize quaternion to unit length.

    Args:
        q: Quaternion tensor of shape (..., 4) as [w, x, y, z]
        eps: Small constant for numerical stability

    Returns:
        Normalized quaternion of shape (..., 4)
    """
    return F.normalize(q, p=2, dim=-1, eps=eps)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate: q* = w - xi - yj - zk

    Args:
        q: Quaternion tensor of shape (..., 4) as [w, x, y, z]

    Returns:
        Conjugate quaternion of shape (..., 4)
    """
    # Negate the vector part
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def quaternion_inverse(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute quaternion inverse: q^{-1} = q* / |q|^2

    For unit quaternions, the inverse equals the conjugate.

    Args:
        q: Quaternion tensor of shape (..., 4) as [w, x, y, z]
        eps: Small constant for numerical stability

    Returns:
        Inverse quaternion of shape (..., 4)
    """
    conj = quaternion_conjugate(q)
    norm_sq = (q * q).sum(dim=-1, keepdim=True).clamp(min=eps)
    return conj / norm_sq


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion product: q1 * q2

    Uses the Hamilton product formula:
        (a1 + b1i + c1j + d1k)(a2 + b2i + c2j + d2k)

    Args:
        q1: First quaternion of shape (..., 4) as [w, x, y, z]
        q2: Second quaternion of shape (..., 4) as [w, x, y, z]

    Returns:
        Product quaternion of shape (..., 4)
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion to 3x3 rotation matrix.

    Args:
        q: Unit quaternion of shape (..., 4) as [w, x, y, z]

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Ensure unit quaternion
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(dim=-1)

    # Compute rotation matrix elements
    # Row 1
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)

    # Row 2
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)

    # Row 3
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    # Stack into matrix
    batch_shape = q.shape[:-1]
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)

    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to unit quaternion.

    Uses the Shepperd method for numerical stability.

    Args:
        R: Rotation matrix of shape (..., 3, 3)

    Returns:
        Unit quaternion of shape (..., 4) as [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    B = R.shape[0]

    # Shepperd's method
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4

    # Reshape and normalize
    q = q.reshape(*batch_shape, 4)
    return normalize_quaternion(q)


def quaternion_from_axis_angle(
    axis: torch.Tensor,
    angle: torch.Tensor
) -> torch.Tensor:
    """
    Create quaternion from axis-angle representation.

    q = cos(θ/2) + sin(θ/2) * (ax*i + ay*j + az*k)

    Args:
        axis: Rotation axis of shape (..., 3), will be normalized
        angle: Rotation angle in radians of shape (...) or (..., 1)

    Returns:
        Unit quaternion of shape (..., 4) as [w, x, y, z]
    """
    # Normalize axis
    axis = F.normalize(axis, p=2, dim=-1)

    # Handle angle shape
    if angle.dim() > 0 and angle.shape[-1] == 1:
        angle = angle.squeeze(-1)

    half_angle = angle / 2
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)

    # Expand dimensions for broadcasting
    if axis.dim() > 1:
        sin_half = sin_half.unsqueeze(-1)
        cos_half = cos_half.unsqueeze(-1)

    w = cos_half
    xyz = axis * sin_half

    if w.dim() == 0:
        return torch.tensor([w, xyz[0], xyz[1], xyz[2]], device=axis.device, dtype=axis.dtype)

    return torch.cat([w.unsqueeze(-1) if w.dim() < xyz.dim() else w, xyz], dim=-1)


def quaternion_to_axis_angle(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert quaternion to axis-angle representation.

    Args:
        q: Unit quaternion of shape (..., 4) as [w, x, y, z]

    Returns:
        axis: Rotation axis of shape (..., 3)
        angle: Rotation angle in radians of shape (...)
    """
    q = normalize_quaternion(q)
    w = q[..., 0]
    xyz = q[..., 1:]

    # Compute angle
    # Handle numerical issues near identity rotation
    sin_half_angle = torch.norm(xyz, dim=-1)
    cos_half_angle = w

    # angle = 2 * atan2(sin_half, cos_half)
    angle = 2 * torch.atan2(sin_half_angle, cos_half_angle)

    # Compute axis (handle zero rotation case)
    axis = F.normalize(xyz, p=2, dim=-1, eps=1e-12)

    return axis, angle


def quaternion_slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    t: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Spherical linear interpolation between two quaternions.

    q(t) = sin((1-t)θ)/sin(θ) * q0 + sin(tθ)/sin(θ) * q1

    Args:
        q0: Start quaternion of shape (..., 4)
        q1: End quaternion of shape (..., 4)
        t: Interpolation parameter in [0, 1], scalar or tensor

    Returns:
        Interpolated quaternion of shape (..., 4)
    """
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    # Compute dot product
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # If dot product is negative, negate one quaternion
    # (quaternions q and -q represent the same rotation)
    neg_mask = dot < 0
    q1 = torch.where(neg_mask, -q1, q1)
    dot = torch.abs(dot)

    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)

    # Compute interpolation
    theta = torch.acos(dot)

    # Handle small angles (linear interpolation)
    if isinstance(t, float):
        t = torch.tensor(t, device=q0.device, dtype=q0.dtype)

    # Standard slerp formula
    sin_theta = torch.sin(theta)

    # Avoid division by zero for small angles
    small_angle_mask = sin_theta.abs() < 1e-6

    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta

    # For small angles, use linear interpolation
    s0 = torch.where(small_angle_mask, 1 - t, s0)
    s1 = torch.where(small_angle_mask, t, s1)

    return normalize_quaternion(s0 * q0 + s1 * q1)


def random_quaternion(
    batch_size: int,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Generate random unit quaternions uniformly distributed on S^3.

    Uses the Shoemake method for uniform random rotations.

    Args:
        batch_size: Number of random quaternions to generate
        device: Torch device
        dtype: Torch dtype

    Returns:
        Random unit quaternions of shape (batch_size, 4)
    """
    # Generate three uniform random numbers
    u = torch.rand(batch_size, 3, device=device, dtype=dtype)

    u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]

    # Shoemake's method
    sqrt_1_minus_u0 = torch.sqrt(1 - u0)
    sqrt_u0 = torch.sqrt(u0)
    two_pi_u1 = 2 * torch.pi * u1
    two_pi_u2 = 2 * torch.pi * u2

    w = sqrt_1_minus_u0 * torch.sin(two_pi_u1)
    x = sqrt_1_minus_u0 * torch.cos(two_pi_u1)
    y = sqrt_u0 * torch.sin(two_pi_u2)
    z = sqrt_u0 * torch.cos(two_pi_u2)

    return torch.stack([w, x, y, z], dim=-1)


def identity_quaternion(
    batch_size: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Create identity quaternion (no rotation).

    Args:
        batch_size: Number of identity quaternions to create
        device: Torch device
        dtype: Torch dtype

    Returns:
        Identity quaternions of shape (batch_size, 4)
    """
    q = torch.zeros(batch_size, 4, device=device, dtype=dtype)
    q[:, 0] = 1.0  # w = 1, x = y = z = 0
    return q


def rotate_vector(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate a 3D vector by a quaternion.

    v' = q * v * q^{-1} (quaternion sandwich product)

    Args:
        v: Vector(s) of shape (..., 3)
        q: Unit quaternion(s) of shape (..., 4)

    Returns:
        Rotated vector(s) of shape (..., 3)
    """
    # Convert vector to quaternion (w=0)
    v_quat = torch.zeros(*v.shape[:-1], 4, device=v.device, dtype=v.dtype)
    v_quat[..., 1:] = v

    # Compute q * v * q^{-1}
    q_inv = quaternion_conjugate(q)  # For unit quaternions
    result = quaternion_multiply(quaternion_multiply(q, v_quat), q_inv)

    return result[..., 1:]  # Extract vector part


def quaternion_angle_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute angular distance between two quaternions.

    The distance is the angle of the rotation needed to go from q1 to q2.

    Args:
        q1: First quaternion of shape (..., 4)
        q2: Second quaternion of shape (..., 4)

    Returns:
        Angular distance in radians of shape (...)
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    # Compute dot product (take absolute value since q and -q are same rotation)
    dot = torch.abs((q1 * q2).sum(dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)

    # Angular distance is 2 * acos(|dot|)
    return 2 * torch.acos(dot)

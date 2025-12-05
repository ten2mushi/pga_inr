"""
Motor operations for Projective Geometric Algebra (PGA).

A Motor in PGA represents a rigid body motion (rotation + translation).
Motors are elements of the even subalgebra and have the form:

    M = R * T

where R is a rotor (pure rotation) and T is a translator (pure translation).

Motors have 8 non-zero components:
[s, e01, e02, e03, e12, e31, e23, e0123]

The fundamental operation is the sandwich product:
    X' = M * X * ~M

This transforms any geometric element X (point, line, plane) while
preserving distances and angles.
"""

from __future__ import annotations
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebra import (
    Multivector, sandwich, exp, log,
    IDX_S, IDX_E01, IDX_E02, IDX_E03, IDX_E12, IDX_E31, IDX_E23, IDX_E0123,
    IDX_E012, IDX_E031, IDX_E023, IDX_E123,
)
from ..utils.quaternion import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
    normalize_quaternion,
)


class Motor:
    """
    A Motor representing rigid body motion in PGA.

    Can be constructed from:
    - Translation vector + quaternion rotation
    - 4x4 homogeneous transformation matrix
    - Rotor + translator components
    - Raw multivector components

    The motor is stored both as a Multivector and as a 4x4 matrix
    for efficient point transformations.
    """

    def __init__(
        self,
        translation: Optional[torch.Tensor] = None,
        quaternion: Optional[torch.Tensor] = None,
        multivector: Optional[Multivector] = None
    ):
        """
        Initialize a Motor.

        Args:
            translation: Translation vector of shape (..., 3)
            quaternion: Rotation quaternion of shape (..., 4) as [w, x, y, z]
            multivector: Raw multivector representation
        """
        if multivector is not None:
            self._mv = multivector
            self._matrix = None
        elif translation is not None and quaternion is not None:
            self._mv = self._from_translation_quaternion(translation, quaternion)
            self._matrix = None
        else:
            # Identity motor
            mv = torch.zeros(16)
            mv[IDX_S] = 1.0
            self._mv = Multivector(mv)
            self._matrix = None

    @staticmethod
    def _from_translation_quaternion(
        translation: torch.Tensor,
        quaternion: torch.Tensor
    ) -> Multivector:
        """
        Construct motor from translation and quaternion.

        Motor = Translator * Rotor

        The rotor in PGA uses bivectors:
            R = w + x*e23 + y*e31 + z*e12
        where (w, x, y, z) comes from the quaternion.

        The translator is:
            T = 1 + (d/2)*(dx*e01 + dy*e02 + dz*e03)

        The product M = T * R combines these, with the ideal bivector
        components determined by the geometric product of translator
        and rotor terms.
        """
        # Normalize quaternion
        q = normalize_quaternion(quaternion)
        w, x, y, z = q.unbind(dim=-1)
        tx, ty, tz = translation.unbind(dim=-1)

        # Build multivector components
        batch_shape = translation.shape[:-1]
        mv = torch.zeros(*batch_shape, 16, device=translation.device, dtype=translation.dtype)

        # Rotor components (scalar + Euclidean bivectors)
        # R = w + x*e23 + y*e31 + z*e12
        mv[..., IDX_S] = w
        mv[..., IDX_E23] = x
        mv[..., IDX_E31] = y
        mv[..., IDX_E12] = z

        # Ideal bivector components for motor M = T * R
        # Derived from geometric product of translator and rotor:
        # e01: 0.5 * (tx*w - ty*e12 + tz*e31) = 0.5 * (tx*w - ty*z + tz*y)
        # e02: 0.5 * (tx*e12 + ty*w - tz*e23) = 0.5 * (tx*z + ty*w - tz*x)
        # e03: 0.5 * (-tx*e31 + ty*e23 + tz*w) = 0.5 * (-tx*y + ty*x + tz*w)
        mv[..., IDX_E01] = 0.5 * (tx * w - ty * z + tz * y)
        mv[..., IDX_E02] = 0.5 * (tx * z + ty * w - tz * x)
        mv[..., IDX_E03] = 0.5 * (-tx * y + ty * x + tz * w)

        return Multivector(mv)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> 'Motor':
        """
        Create Motor from 4x4 homogeneous transformation matrix.

        Args:
            matrix: Transformation matrix of shape (..., 4, 4)

        Returns:
            Motor representing the same transformation
        """
        # Extract rotation and translation
        R = matrix[..., :3, :3]
        t = matrix[..., :3, 3]

        # Convert rotation matrix to quaternion
        q = matrix_to_quaternion(R)

        return cls(translation=t, quaternion=q)

    @classmethod
    def from_rotor_translator(
        cls,
        rotor: torch.Tensor,
        translator: torch.Tensor
    ) -> 'Motor':
        """
        Create Motor from separate rotor and translator.

        Args:
            rotor: Rotor components [s, e12, e31, e23] of shape (..., 4)
            translator: Translation vector of shape (..., 3)

        Returns:
            Combined motor M = T * R
        """
        # Convert rotor to quaternion format
        s, e12, e31, e23 = rotor.unbind(dim=-1)
        quaternion = torch.stack([s, e23, e31, e12], dim=-1)

        return cls(translation=translator, quaternion=quaternion)

    @classmethod
    def identity(
        cls,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> 'Motor':
        """Create identity motor (no transformation)."""
        mv = torch.zeros(batch_size, 16, device=device, dtype=dtype or torch.float32)
        mv[:, IDX_S] = 1.0
        return cls(multivector=Multivector(mv))

    @property
    def multivector(self) -> Multivector:
        """Get the multivector representation."""
        return self._mv

    @property
    def device(self) -> torch.device:
        return self._mv.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mv.dtype

    @property
    def shape(self) -> torch.Size:
        """Batch shape."""
        return self._mv.shape

    def to(self, device: torch.device) -> 'Motor':
        """Move to device."""
        return Motor(multivector=self._mv.to(device))

    def to_matrix(self) -> torch.Tensor:
        """
        Convert motor to 4x4 homogeneous transformation matrix.

        Returns:
            Matrix of shape (..., 4, 4)
        """
        if self._matrix is not None:
            return self._matrix

        mv = self._mv.mv

        # Extract rotor (rotation) part
        s = mv[..., IDX_S]
        e12 = mv[..., IDX_E12]
        e31 = mv[..., IDX_E31]
        e23 = mv[..., IDX_E23]

        # Quaternion from rotor: [w, x, y, z] = [s, e23, e31, e12]
        q = torch.stack([s, e23, e31, e12], dim=-1)
        q = normalize_quaternion(q)

        # Build rotation matrix
        R = quaternion_to_matrix(q)

        # Extract translation from ideal bivector part
        e01 = mv[..., IDX_E01]
        e02 = mv[..., IDX_E02]
        e03 = mv[..., IDX_E03]

        # Recover translation from ideal bivector components.
        # The encoding was:
        #   e01 = 0.5 * (tx*w - ty*z + tz*y)
        #   e02 = 0.5 * (tx*z + ty*w - tz*x)
        #   e03 = 0.5 * (-tx*y + ty*x + tz*w)
        #
        # This is a linear system: e_vec = 0.5 * A @ t_vec
        # where A = [[ w, -z,  y],
        #            [ z,  w, -x],
        #            [-y,  x,  w]]
        #
        # For unit quaternion, det(A) = w*(w² + x² + y² + z²) = w
        # The inverse is A^{-1} = adj(A) / w, so t = (2/w) * adj(A) @ e
        #
        # Computing adj(A):
        # adj[0,0] = w² + x²;    adj[0,1] = wz + xy;  adj[0,2] = -wy + xz
        # adj[1,0] = -wz + xy;   adj[1,1] = w² + y²;  adj[1,2] = wx + yz
        # adj[2,0] = wy + xz;    adj[2,1] = -wx + yz; adj[2,2] = w² + z²
        #
        # Then t = (2/w) * adj(A) @ e gives:
        w, x, y, z = q.unbind(dim=-1)

        # For numerical stability, we compute 2 * adj(A) @ e directly,
        # which equals w * t_vec for unit quaternion
        # But since we need t, we'll use the formula with division by w
        # To avoid division by zero when w≈0, we use a safe division.
        w_safe = torch.where(w.abs() < 1e-8, torch.ones_like(w), w)

        # Compute 2 * adj(A) @ e_vec
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        # t_vec * w = 2 * adj(A) @ e_vec
        tx_w = 2.0 * ((ww + xx)*e01 + (wz + xy)*e02 + (-wy + xz)*e03)
        ty_w = 2.0 * ((-wz + xy)*e01 + (ww + yy)*e02 + (wx + yz)*e03)
        tz_w = 2.0 * ((wy + xz)*e01 + (-wx + yz)*e02 + (ww + zz)*e03)

        # Divide by w (with safe division)
        tx = tx_w / w_safe
        ty = ty_w / w_safe
        tz = tz_w / w_safe
        t = torch.stack([tx, ty, tz], dim=-1)

        # Build 4x4 matrix
        batch_shape = mv.shape[:-1]
        M = torch.eye(4, device=mv.device, dtype=mv.dtype)
        M = M.expand(*batch_shape, 4, 4).clone()
        M[..., :3, :3] = R
        M[..., :3, 3] = t

        self._matrix = M
        return M

    def inverse(self) -> 'Motor':
        """
        Compute inverse motor: M^{-1} where M * M^{-1} = 1

        For unit motors (which represent valid rigid transformations),
        the inverse equals the reverse: M^{-1} = ~M
        """
        return Motor(multivector=self._mv.reverse())

    def compose(self, other: 'Motor') -> 'Motor':
        """
        Compose two motors: M_combined = self * other

        This represents applying 'other' first, then 'self'.
        """
        return Motor(multivector=self._mv * other._mv)

    def __mul__(self, other: 'Motor') -> 'Motor':
        """Motor composition."""
        return self.compose(other)

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply motor to 3D points using sandwich product.

        Args:
            points: Points of shape (..., N, 3) in Cartesian coordinates

        Returns:
            Transformed points of shape (..., N, 3)
        """
        # Use matrix representation for efficiency
        M = self.to_matrix()

        # Homogenize points
        ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, ones], dim=-1)  # (..., N, 4)

        # Apply transformation
        # M is (..., 4, 4), points_h is (..., N, 4)
        # We need to handle the batch dimensions correctly
        points_h_t = points_h.unsqueeze(-1)  # (..., N, 4, 1)
        M_expanded = M.unsqueeze(-3)  # (..., 1, 4, 4)
        result_h = (M_expanded @ points_h_t).squeeze(-1)  # (..., N, 4)

        return result_h[..., :3]

    def apply_inverse(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse motor to points.

        Equivalent to transforming points from world frame to local frame.
        """
        return self.inverse().apply(points)

    def interpolate(self, other: 'Motor', t: Union[float, torch.Tensor]) -> 'Motor':
        """
        Geodesic interpolation between two motors.

        Uses log-linear interpolation on SE(3):
            M(t) = exp(t * log(M1^{-1} * M2)) * M1

        Args:
            other: Target motor
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated motor
        """
        # Compute relative motor
        relative = self.inverse() * other

        # Logarithm to get bivector
        bv = log(relative.multivector)

        # Scale bivector
        if isinstance(t, float):
            t = torch.tensor(t, device=self.device, dtype=self.dtype)
        scaled_bv = Multivector(bv.mv * t.unsqueeze(-1) if t.dim() > 0 else bv.mv * t)

        # Exponential and compose
        interp_relative = exp(scaled_bv)
        return Motor(multivector=self._mv * interp_relative)

    def translation(self) -> torch.Tensor:
        """Extract translation vector."""
        M = self.to_matrix()
        return M[..., :3, 3]

    def rotation_quaternion(self) -> torch.Tensor:
        """Extract rotation as quaternion [w, x, y, z]."""
        mv = self._mv.mv
        s = mv[..., IDX_S]
        e12 = mv[..., IDX_E12]
        e31 = mv[..., IDX_E31]
        e23 = mv[..., IDX_E23]
        return normalize_quaternion(torch.stack([s, e23, e31, e12], dim=-1))

    def rotation_matrix(self) -> torch.Tensor:
        """Extract rotation as 3x3 matrix."""
        return self.to_matrix()[..., :3, :3]

    def __repr__(self) -> str:
        return f"Motor(shape={self.shape}, device={self.device})"


def rotor_from_axis_angle(
    axis: torch.Tensor,
    angle: torch.Tensor
) -> Multivector:
    """
    Create a rotation rotor from axis-angle representation.

    R = cos(θ/2) + sin(θ/2) * (ax*e23 + ay*e31 + az*e12)

    Args:
        axis: Rotation axis of shape (..., 3), will be normalized
        angle: Rotation angle in radians of shape (...) or (..., 1)

    Returns:
        Rotor multivector
    """
    # Normalize axis
    axis = F.normalize(axis, p=2, dim=-1)

    # Handle angle shape
    if angle.dim() > 0 and angle.shape[-1] == 1:
        angle = angle.squeeze(-1)

    half_angle = angle / 2
    cos_half = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)

    batch_shape = axis.shape[:-1]
    mv = torch.zeros(*batch_shape, 16, device=axis.device, dtype=axis.dtype)

    mv[..., IDX_S] = cos_half
    mv[..., IDX_E23] = sin_half * axis[..., 0]
    mv[..., IDX_E31] = sin_half * axis[..., 1]
    mv[..., IDX_E12] = sin_half * axis[..., 2]

    return Multivector(mv)


def translator_from_direction(
    direction: torch.Tensor,
    distance: Optional[torch.Tensor] = None
) -> Multivector:
    """
    Create a translation motor from direction and distance.

    T = 1 + (d/2) * (dx*e01 + dy*e02 + dz*e03)

    Args:
        direction: Translation direction of shape (..., 3)
                  If distance is None, this is the full translation vector.
        distance: Translation distance of shape (...) or (..., 1)

    Returns:
        Translator multivector
    """
    if distance is not None:
        # Normalize direction and scale by distance
        direction = F.normalize(direction, p=2, dim=-1)
        if distance.dim() > 0 and distance.shape[-1] == 1:
            distance = distance.squeeze(-1)
        translation = direction * distance.unsqueeze(-1)
    else:
        translation = direction

    batch_shape = translation.shape[:-1]
    mv = torch.zeros(*batch_shape, 16, device=translation.device, dtype=translation.dtype)

    mv[..., IDX_S] = 1.0
    mv[..., IDX_E01] = 0.5 * translation[..., 0]
    mv[..., IDX_E02] = 0.5 * translation[..., 1]
    mv[..., IDX_E03] = 0.5 * translation[..., 2]

    return Multivector(mv)


def motor_from_transform(
    translation: torch.Tensor,
    quaternion: torch.Tensor
) -> Multivector:
    """
    Create a motor from translation and quaternion.

    Convenience function that returns the multivector directly.

    Args:
        translation: Translation vector of shape (..., 3)
        quaternion: Rotation quaternion of shape (..., 4) as [w, x, y, z]

    Returns:
        Motor multivector
    """
    m = Motor(translation=translation, quaternion=quaternion)
    return m.multivector


def motor_log(motor: Motor) -> torch.Tensor:
    """
    Logarithm map: motor → se(3) Lie algebra element.

    Returns a 6D vector [ω, v] where ω is angular velocity
    and v is linear velocity.

    Args:
        motor: Motor to compute log of

    Returns:
        Lie algebra element of shape (..., 6)
    """
    bv = log(motor.multivector)

    # Extract rotation part (ω)
    omega = torch.stack([
        bv.mv[..., IDX_E23],
        bv.mv[..., IDX_E31],
        bv.mv[..., IDX_E12],
    ], dim=-1)

    # Extract translation part (v)
    v = torch.stack([
        bv.mv[..., IDX_E01],
        bv.mv[..., IDX_E02],
        bv.mv[..., IDX_E03],
    ], dim=-1)

    return torch.cat([omega, v], dim=-1)


def motor_exp(lie_algebra: torch.Tensor) -> Motor:
    """
    Exponential map: se(3) Lie algebra → motor.

    Args:
        lie_algebra: 6D vector [ω, v] of shape (..., 6)

    Returns:
        Motor
    """
    omega = lie_algebra[..., :3]
    v = lie_algebra[..., 3:]

    # Build bivector
    batch_shape = lie_algebra.shape[:-1]
    mv = torch.zeros(*batch_shape, 16, device=lie_algebra.device, dtype=lie_algebra.dtype)

    mv[..., IDX_E23] = omega[..., 0]
    mv[..., IDX_E31] = omega[..., 1]
    mv[..., IDX_E12] = omega[..., 2]
    mv[..., IDX_E01] = v[..., 0]
    mv[..., IDX_E02] = v[..., 1]
    mv[..., IDX_E03] = v[..., 2]

    return Motor(multivector=exp(Multivector(mv)))


class MotorLayer(nn.Module):
    """
    Neural network layer that learns a motor transformation.

    Parameters are the translation and rotation (as axis-angle or quaternion).
    """

    def __init__(
        self,
        init_translation: Optional[torch.Tensor] = None,
        init_rotation: Optional[torch.Tensor] = None,
        parameterization: str = 'quaternion'
    ):
        """
        Args:
            init_translation: Initial translation (3,)
            init_rotation: Initial rotation as quaternion (4,) or axis-angle (3,)
            parameterization: 'quaternion' or 'axis_angle'
        """
        super().__init__()

        self.parameterization = parameterization

        # Initialize translation
        if init_translation is None:
            init_translation = torch.zeros(3)
        self.translation = nn.Parameter(init_translation.clone())

        # Initialize rotation
        if parameterization == 'quaternion':
            if init_rotation is None:
                init_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity
            self.rotation = nn.Parameter(init_rotation.clone())
        else:
            if init_rotation is None:
                init_rotation = torch.zeros(3)  # Zero rotation
            self.rotation = nn.Parameter(init_rotation.clone())

    def forward(self) -> Motor:
        """Return the current motor."""
        if self.parameterization == 'quaternion':
            q = normalize_quaternion(self.rotation)
        else:
            # Convert axis-angle to quaternion
            angle = torch.norm(self.rotation)
            axis = self.rotation / (angle + 1e-8)
            half_angle = angle / 2
            q = torch.cat([
                torch.cos(half_angle).unsqueeze(0),
                torch.sin(half_angle) * axis
            ])

        return Motor(translation=self.translation.unsqueeze(0), quaternion=q.unsqueeze(0))

    def get_matrix(self) -> torch.Tensor:
        """Get the 4x4 transformation matrix."""
        return self.forward().to_matrix()

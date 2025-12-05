"""
Motor interpolation methods for smooth motion.

Provides various interpolation schemes on SE(3):
- SLERP (Spherical Linear Interpolation)
- SQUAD (Spherical Quadrangle Interpolation)
- Bezier curves
- Screw motion interpolation
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


def quaternion_slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Spherical linear interpolation for quaternions.

    Args:
        q0: Start quaternion [w, x, y, z] (..., 4)
        q1: End quaternion [w, x, y, z] (..., 4)
        t: Interpolation parameter (...) or scalar in [0, 1]

    Returns:
        Interpolated quaternion (..., 4)
    """
    # Normalize
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)

    # Compute dot product
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # If dot < 0, negate one quaternion (shortest path)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)

    # Clamp for numerical stability
    dot = dot.clamp(-1 + 1e-6, 1 - 1e-6)

    # Compute angle
    theta = torch.acos(dot)

    # Handle t dimensions
    if t.dim() == 0:
        t = t.unsqueeze(0)
    while t.dim() < q0.dim():
        t = t.unsqueeze(-1)

    # SLERP formula
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta

    # Handle near-parallel case (use linear interpolation)
    near_parallel = theta.abs() < 1e-6
    s0 = torch.where(near_parallel, 1 - t, s0)
    s1 = torch.where(near_parallel, t, s1)

    return F.normalize(s0 * q0 + s1 * q1, dim=-1)


def motor_slerp(
    translation0: torch.Tensor,
    quaternion0: torch.Tensor,
    translation1: torch.Tensor,
    quaternion1: torch.Tensor,
    t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Geodesic interpolation of motors (rigid motions) on SE(3).

    Uses logarithm map, linear interpolation, exponential map.

    Args:
        translation0: Start translation (..., 3)
        quaternion0: Start rotation [w, x, y, z] (..., 4)
        translation1: End translation (..., 3)
        quaternion1: End rotation [w, x, y, z] (..., 4)
        t: Interpolation parameter in [0, 1]

    Returns:
        (interpolated_translation, interpolated_quaternion)
    """
    # Interpolate rotation with SLERP
    q_interp = quaternion_slerp(quaternion0, quaternion1, t)

    # Handle t dimensions for translation
    if t.dim() == 0:
        t = t.unsqueeze(0)
    while t.dim() < translation0.dim():
        t = t.unsqueeze(-1)

    # Linear interpolation for translation
    # (This is an approximation; true geodesic would use screw motion)
    t_interp = (1 - t) * translation0 + t * translation1

    return t_interp, q_interp


def screw_interpolation(
    translation0: torch.Tensor,
    quaternion0: torch.Tensor,
    translation1: torch.Tensor,
    quaternion1: torch.Tensor,
    t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constant-velocity screw motion interpolation.

    This is the true geodesic on SE(3).

    Uses: M(t) = M0 * exp(t * log(M0^-1 * M1))

    Args:
        translation0: Start translation (..., 3)
        quaternion0: Start rotation (..., 4)
        translation1: End translation (..., 3)
        quaternion1: End rotation (..., 4)
        t: Interpolation parameter in [0, 1]

    Returns:
        (interpolated_translation, interpolated_quaternion)
    """
    from ..utils.quaternion import (
        quaternion_multiply, quaternion_conjugate, quaternion_to_matrix
    )

    # Compute relative transformation: M0^-1 * M1
    # For rotation: q0^-1 * q1 = conj(q0) * q1
    q0_inv = quaternion_conjugate(quaternion0)
    q_rel = quaternion_multiply(q0_inv, quaternion1)

    # For translation, we need to rotate t1 - t0 by q0^-1
    t_diff = translation1 - translation0
    R0_inv = quaternion_to_matrix(q0_inv)
    if R0_inv.dim() == 2:
        R0_inv = R0_inv.unsqueeze(0)
    if t_diff.dim() == 1:
        t_diff = t_diff.unsqueeze(0)

    t_rel = torch.einsum('...ij,...j->...i', R0_inv, t_diff)

    # Interpolate the relative transformation
    # For quaternion: slerp from identity to q_rel
    identity_q = torch.zeros_like(quaternion0)
    identity_q[..., 0] = 1.0
    q_interp_rel = quaternion_slerp(identity_q, q_rel, t)

    # Handle t dimensions
    if t.dim() == 0:
        t_scalar = t.item()
    else:
        t_scalar = t
        while t_scalar.dim() < t_rel.dim():
            t_scalar = t_scalar.unsqueeze(-1)

    t_interp_rel = t_scalar * t_rel

    # Apply to M0: M(t) = M0 * M_rel(t)
    q_interp = quaternion_multiply(quaternion0, q_interp_rel)

    # Rotate t_interp_rel by q0 and add to t0
    R0 = quaternion_to_matrix(quaternion0)
    if R0.dim() == 2:
        R0 = R0.unsqueeze(0)
    if t_interp_rel.dim() == 1:
        t_interp_rel = t_interp_rel.unsqueeze(0)

    t_interp = translation0 + torch.einsum('...ij,...j->...i', R0, t_interp_rel)

    return t_interp, q_interp


def squad(
    q0: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    q3: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Spherical quadrangle interpolation (SQUAD).

    Provides C1 continuous interpolation through control quaternions.

    Args:
        q0, q1, q2, q3: Control quaternions (..., 4)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated quaternion (..., 4)
    """
    # Inner control points (for C1 continuity)
    s1 = squad_inner_point(q0, q1, q2)
    s2 = squad_inner_point(q1, q2, q3)

    # Double slerp
    q_a = quaternion_slerp(q1, q2, t)
    q_b = quaternion_slerp(s1, s2, t)

    return quaternion_slerp(q_a, q_b, 2 * t * (1 - t))


def squad_inner_point(
    q_prev: torch.Tensor,
    q_curr: torch.Tensor,
    q_next: torch.Tensor
) -> torch.Tensor:
    """
    Compute inner control point for SQUAD.

    Args:
        q_prev: Previous quaternion (..., 4)
        q_curr: Current quaternion (..., 4)
        q_next: Next quaternion (..., 4)

    Returns:
        Inner control point quaternion (..., 4)
    """
    from ..utils.quaternion import quaternion_multiply, quaternion_conjugate

    # s_i = q_i * exp(-0.25 * (log(q_i^-1 * q_{i+1}) + log(q_i^-1 * q_{i-1})))
    q_curr_inv = quaternion_conjugate(q_curr)

    # Relative quaternions
    q_to_next = quaternion_multiply(q_curr_inv, q_next)
    q_to_prev = quaternion_multiply(q_curr_inv, q_prev)

    # Log of quaternion (simplified for unit quaternions)
    log_next = quaternion_log(q_to_next)
    log_prev = quaternion_log(q_to_prev)

    # Average and negate
    log_avg = -0.25 * (log_next + log_prev)

    # Exp of averaged log
    s = quaternion_multiply(q_curr, quaternion_exp(log_avg))

    return F.normalize(s, dim=-1)


def quaternion_log(q: torch.Tensor) -> torch.Tensor:
    """
    Logarithm of unit quaternion.

    log(q) = (0, theta * v) where q = (cos(theta), sin(theta) * v)

    Args:
        q: Unit quaternion (..., 4)

    Returns:
        Pure quaternion (log) (..., 4)
    """
    q = F.normalize(q, dim=-1)
    w = q[..., 0:1]
    v = q[..., 1:4]

    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    theta = torch.atan2(v_norm, w)

    # log = (0, theta * v / |v|)
    log = torch.zeros_like(q)
    log[..., 1:4] = theta * v / v_norm

    return log


def quaternion_exp(log_q: torch.Tensor) -> torch.Tensor:
    """
    Exponential of pure quaternion.

    exp((0, v)) = (cos(|v|), sin(|v|) * v / |v|)

    Args:
        log_q: Pure quaternion (..., 4)

    Returns:
        Unit quaternion (..., 4)
    """
    v = log_q[..., 1:4]
    theta = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    q = torch.zeros_like(log_q)
    q[..., 0:1] = torch.cos(theta)
    q[..., 1:4] = torch.sin(theta) * v / theta

    return F.normalize(q, dim=-1)


def bezier_motor(
    control_translations: List[torch.Tensor],
    control_quaternions: List[torch.Tensor],
    t: torch.Tensor,
    num_subdivisions: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    De Casteljau algorithm for Bezier curves on SE(3).

    Args:
        control_translations: List of control point translations
        control_quaternions: List of control point quaternions
        t: Curve parameter in [0, 1]
        num_subdivisions: Subdivisions for de Casteljau

    Returns:
        (translation, quaternion) at parameter t
    """
    n = len(control_translations)
    if n != len(control_quaternions):
        raise ValueError("Number of translations and quaternions must match")

    # Copy control points
    translations = list(control_translations)
    quaternions = list(control_quaternions)

    # De Casteljau iterations
    for _ in range(n - 1):
        new_translations = []
        new_quaternions = []

        for i in range(len(translations) - 1):
            t_interp, q_interp = motor_slerp(
                translations[i], quaternions[i],
                translations[i + 1], quaternions[i + 1],
                t
            )
            new_translations.append(t_interp)
            new_quaternions.append(q_interp)

        translations = new_translations
        quaternions = new_quaternions

    return translations[0], quaternions[0]


def catmull_rom_motor(
    translations: List[torch.Tensor],
    quaternions: List[torch.Tensor],
    t: torch.Tensor,
    tension: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Catmull-Rom spline interpolation on SE(3).

    Args:
        translations: List of waypoint translations
        quaternions: List of waypoint quaternions
        t: Global parameter in [0, n-1] where n is number of waypoints
        tension: Spline tension parameter

    Returns:
        (translation, quaternion) at parameter t
    """
    n = len(translations)
    if n < 4:
        raise ValueError("Need at least 4 waypoints for Catmull-Rom")

    # Find segment
    if t.dim() == 0:
        segment = int(t.item())
    else:
        segment = t.int()

    segment = max(1, min(segment, n - 3))
    local_t = t - segment

    # Get the 4 control points for this segment
    idx = [segment - 1, segment, segment + 1, segment + 2]

    # Use SQUAD for quaternion interpolation
    q_interp = squad(
        quaternions[idx[0]], quaternions[idx[1]],
        quaternions[idx[2]], quaternions[idx[3]],
        local_t
    )

    # Catmull-Rom for translation
    p0, p1, p2, p3 = [translations[i] for i in idx]

    if local_t.dim() == 0:
        local_t = local_t.unsqueeze(0)
    while local_t.dim() < p0.dim():
        local_t = local_t.unsqueeze(-1)

    t2 = local_t ** 2
    t3 = local_t ** 3

    t_interp = (
        0.5 * (
            (2 * p1) +
            (-p0 + p2) * local_t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
    )

    return t_interp, q_interp


def hermite_motor(
    translation0: torch.Tensor,
    quaternion0: torch.Tensor,
    velocity0: torch.Tensor,
    angular0: torch.Tensor,
    translation1: torch.Tensor,
    quaternion1: torch.Tensor,
    velocity1: torch.Tensor,
    angular1: torch.Tensor,
    t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hermite interpolation with velocity constraints.

    Useful when you have velocity/angular velocity at endpoints.

    Args:
        translation0, translation1: Endpoint translations
        quaternion0, quaternion1: Endpoint quaternions
        velocity0, velocity1: Linear velocities at endpoints
        angular0, angular1: Angular velocities at endpoints
        t: Parameter in [0, 1]

    Returns:
        (translation, quaternion) at parameter t
    """
    # Hermite basis functions
    if t.dim() == 0:
        t = t.unsqueeze(0)
    while t.dim() < translation0.dim():
        t = t.unsqueeze(-1)

    t2 = t ** 2
    t3 = t ** 3

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    # Interpolate translation
    t_interp = (
        h00 * translation0 +
        h10 * velocity0 +
        h01 * translation1 +
        h11 * velocity1
    )

    # For rotation, use SLERP with velocity-based tangent adjustment
    # Simplified: just use SLERP
    q_interp = quaternion_slerp(quaternion0, quaternion1, t)

    return t_interp, q_interp


class MotorTrajectory:
    """
    Represents a continuous motor trajectory.

    Supports evaluation, velocity computation, and arc-length parameterization.
    """

    def __init__(
        self,
        translations: List[torch.Tensor],
        quaternions: List[torch.Tensor],
        method: str = 'catmull_rom'
    ):
        """
        Args:
            translations: List of waypoint translations
            quaternions: List of waypoint quaternions
            method: Interpolation method ('linear', 'slerp', 'catmull_rom', 'bezier')
        """
        self.translations = translations
        self.quaternions = quaternions
        self.method = method
        self.n = len(translations)

    def evaluate(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate trajectory at parameter t.

        Args:
            t: Parameter in [0, 1]

        Returns:
            (translation, quaternion)
        """
        # Scale t to trajectory length
        t_scaled = t * (self.n - 1)

        if self.method == 'linear':
            # Find segment
            segment = t_scaled.int().clamp(0, self.n - 2)
            local_t = t_scaled - segment.float()
            return motor_slerp(
                self.translations[segment], self.quaternions[segment],
                self.translations[segment + 1], self.quaternions[segment + 1],
                local_t
            )
        elif self.method == 'catmull_rom':
            if self.n >= 4:
                return catmull_rom_motor(
                    self.translations, self.quaternions, t_scaled
                )
            else:
                # Fall back to linear
                segment = t_scaled.int().clamp(0, self.n - 2)
                local_t = t_scaled - segment.float()
                return motor_slerp(
                    self.translations[segment], self.quaternions[segment],
                    self.translations[segment + 1], self.quaternions[segment + 1],
                    local_t
                )
        elif self.method == 'bezier':
            return bezier_motor(
                self.translations, self.quaternions, t
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def velocity(
        self,
        t: torch.Tensor,
        dt: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity at parameter t using finite differences.

        Args:
            t: Parameter in [0, 1]
            dt: Finite difference step

        Returns:
            (linear_velocity, angular_velocity)
        """
        t0 = (t - dt).clamp(0, 1)
        t1 = (t + dt).clamp(0, 1)

        trans0, quat0 = self.evaluate(t0)
        trans1, quat1 = self.evaluate(t1)

        # Linear velocity
        linear_vel = (trans1 - trans0) / (2 * dt)

        # Angular velocity (from quaternion difference)
        from ..utils.quaternion import quaternion_multiply, quaternion_conjugate
        q_diff = quaternion_multiply(quat1, quaternion_conjugate(quat0))
        # Extract axis-angle
        log_q = quaternion_log(q_diff)
        angular_vel = 2 * log_q[..., 1:4] / (2 * dt)

        return linear_vel, angular_vel

    def arc_length(
        self,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        Compute approximate arc length of trajectory.

        Args:
            num_samples: Number of samples for approximation

        Returns:
            Arc length
        """
        t_vals = torch.linspace(0, 1, num_samples)
        total_length = 0.0

        prev_trans = None
        for t in t_vals:
            trans, _ = self.evaluate(t)
            if prev_trans is not None:
                total_length += (trans - prev_trans).norm()
            prev_trans = trans

        return total_length

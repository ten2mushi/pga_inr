"""
Geometric transformations using PGA motors.

Provides high-level functions for rotating, translating, and transforming
geometric primitives (points, lines, planes) using the sandwich product.
"""

from __future__ import annotations
from typing import Union, Optional
import torch
import torch.nn.functional as F

from .algebra import Multivector, sandwich
from .motors import (
    Motor,
    rotor_from_axis_angle,
    translator_from_direction,
)
from .primitives import (
    point_from_tensor,
    point_to_cartesian,
    plane_to_normal_distance,
    plane,
    line_to_plucker,
    line_from_plucker,
)


def rotate(
    element: Multivector,
    axis: torch.Tensor,
    angle: torch.Tensor,
    center: Optional[torch.Tensor] = None
) -> Multivector:
    """
    Rotate a geometric element around an axis.

    Args:
        element: Multivector to rotate
        axis: Rotation axis of shape (..., 3)
        angle: Rotation angle in radians
        center: Center of rotation (defaults to origin)

    Returns:
        Rotated multivector
    """
    # Build rotation rotor
    rotor = rotor_from_axis_angle(axis, angle)

    if center is not None:
        # Translate to origin, rotate, translate back
        to_origin = translator_from_direction(-center)
        from_origin = translator_from_direction(center)

        # Combined motor: T_back * R * T_to_origin
        motor_mv = Multivector(from_origin.mv) * Multivector(rotor.mv) * Multivector(to_origin.mv)
    else:
        motor_mv = rotor

    return sandwich(motor_mv, element)


def translate(
    element: Multivector,
    translation: torch.Tensor
) -> Multivector:
    """
    Translate a geometric element.

    Args:
        element: Multivector to translate
        translation: Translation vector of shape (..., 3)

    Returns:
        Translated multivector
    """
    translator = translator_from_direction(translation)
    return sandwich(Multivector(translator.mv), element)


def transform(
    element: Multivector,
    motor: Motor
) -> Multivector:
    """
    Apply a motor transformation to a geometric element.

    Args:
        element: Multivector to transform
        motor: Motor defining the transformation

    Returns:
        Transformed multivector
    """
    return sandwich(motor.multivector, element)


def transform_point(
    point_coords: torch.Tensor,
    motor: Motor
) -> torch.Tensor:
    """
    Transform 3D point coordinates.

    Convenience function that handles conversion to/from multivector.

    Args:
        point_coords: Points of shape (..., 3)
        motor: Motor defining the transformation

    Returns:
        Transformed points of shape (..., 3)
    """
    # Use motor's efficient matrix-based application
    return motor.apply(point_coords)


def transform_points_batch(
    points: torch.Tensor,
    motors: Motor
) -> torch.Tensor:
    """
    Transform multiple points by multiple motors.

    Args:
        points: Points of shape (N, 3) or (B, N, 3)
        motors: Motors with matching batch dimension

    Returns:
        Transformed points
    """
    return motors.apply(points)


def transform_line(
    line: Multivector,
    motor: Motor
) -> Multivector:
    """
    Transform a line by a motor.

    Args:
        line: Line multivector
        motor: Motor defining the transformation

    Returns:
        Transformed line
    """
    return sandwich(motor.multivector, line)


def transform_plane(
    plane_mv: Multivector,
    motor: Motor
) -> Multivector:
    """
    Transform a plane by a motor.

    Args:
        plane_mv: Plane multivector
        motor: Motor defining the transformation

    Returns:
        Transformed plane
    """
    return sandwich(motor.multivector, plane_mv)


def rotate_points(
    points: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor,
    center: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Rotate 3D points around an axis.

    Args:
        points: Points of shape (..., 3)
        axis: Rotation axis of shape (3,)
        angle: Rotation angle in radians
        center: Center of rotation (defaults to origin)

    Returns:
        Rotated points of shape (..., 3)
    """
    # Build rotation matrix from axis-angle
    axis = F.normalize(axis, dim=-1)

    # Rodrigues' rotation formula
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # Cross product matrix
    K = torch.zeros(3, 3, device=axis.device, dtype=axis.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    # Rotation matrix: R = I + sin(a)K + (1-cos(a))K^2
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    R = I + sin_a * K + (1 - cos_a) * (K @ K)

    if center is not None:
        # Translate to origin, rotate, translate back
        points_centered = points - center
        points_rotated = (R @ points_centered.unsqueeze(-1)).squeeze(-1)
        return points_rotated + center
    else:
        return (R @ points.unsqueeze(-1)).squeeze(-1)


def translate_points(
    points: torch.Tensor,
    translation: torch.Tensor
) -> torch.Tensor:
    """
    Translate 3D points.

    Args:
        points: Points of shape (..., 3)
        translation: Translation vector of shape (3,) or (..., 3)

    Returns:
        Translated points
    """
    return points + translation


def scale_points(
    points: torch.Tensor,
    scale: Union[float, torch.Tensor],
    center: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Scale 3D points.

    Note: Scaling is not a rigid transformation and cannot be
    represented by a motor. This function is provided for convenience.

    Args:
        points: Points of shape (..., 3)
        scale: Scale factor (uniform) or per-axis (3,)
        center: Center of scaling (defaults to origin)

    Returns:
        Scaled points
    """
    if center is not None:
        return (points - center) * scale + center
    else:
        return points * scale


def reflect_points_plane(
    points: torch.Tensor,
    normal: torch.Tensor,
    d: float = 0.0
) -> torch.Tensor:
    """
    Reflect points across a plane.

    Args:
        points: Points of shape (..., 3)
        normal: Plane normal (will be normalized)
        d: Plane offset (plane equation: n·x + d = 0)

    Returns:
        Reflected points
    """
    normal = F.normalize(normal, dim=-1)

    # Signed distance from points to plane
    dist = (points * normal).sum(dim=-1, keepdim=True) + d

    # Reflect: p' = p - 2*dist*n
    return points - 2 * dist * normal


def look_at(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor = None
) -> Motor:
    """
    Create a motor that transforms from world space to camera space.

    The camera looks from 'eye' towards 'target' with 'up' as the up direction.

    Args:
        eye: Camera position (3,)
        target: Look-at target position (3,)
        up: Up direction (defaults to [0, 1, 0])

    Returns:
        Motor representing the camera transformation
    """
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=eye.device, dtype=eye.dtype)

    # Camera coordinate frame
    forward = F.normalize(target - eye, dim=-1)
    right = F.normalize(torch.cross(forward, up), dim=-1)
    up = torch.cross(right, forward)

    # Build rotation matrix (world to camera)
    R = torch.stack([right, up, -forward], dim=0)

    # Translation (bring eye to origin)
    t = -R @ eye

    # Build 4x4 matrix
    M = torch.eye(4, device=eye.device, dtype=eye.dtype)
    M[:3, :3] = R
    M[:3, 3] = t

    return Motor.from_matrix(M)


def orbit_camera(
    distance: float,
    azimuth: float,
    elevation: float,
    target: Optional[torch.Tensor] = None,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> Motor:
    """
    Create a camera motor for an orbiting camera.

    Args:
        distance: Distance from target
        azimuth: Horizontal angle in radians
        elevation: Vertical angle in radians
        target: Look-at target (defaults to origin)
        device: Torch device
        dtype: Torch dtype

    Returns:
        Camera motor
    """
    if target is None:
        target = torch.zeros(3, device=device, dtype=dtype or torch.float32)

    # Compute camera position on sphere
    cos_el = torch.cos(torch.tensor(elevation, device=device, dtype=dtype))
    sin_el = torch.sin(torch.tensor(elevation, device=device, dtype=dtype))
    cos_az = torch.cos(torch.tensor(azimuth, device=device, dtype=dtype))
    sin_az = torch.sin(torch.tensor(azimuth, device=device, dtype=dtype))

    eye = target + distance * torch.tensor([
        cos_el * sin_az,
        sin_el,
        cos_el * cos_az,
    ], device=device, dtype=dtype)

    return look_at(eye, target)


def interpolate_transforms(
    motor1: Motor,
    motor2: Motor,
    t: Union[float, torch.Tensor]
) -> Motor:
    """
    Interpolate between two transformations.

    Uses geodesic interpolation on SE(3) (screw linear interpolation).

    Args:
        motor1: Start transformation
        motor2: End transformation
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated motor
    """
    return motor1.interpolate(motor2, t)


def compose_transforms(*motors: Motor) -> Motor:
    """
    Compose multiple transformations.

    The transformations are applied right-to-left:
    compose(M1, M2, M3) means apply M3 first, then M2, then M1.

    Args:
        *motors: Motors to compose

    Returns:
        Composed motor
    """
    if len(motors) == 0:
        return Motor.identity()

    result = motors[0]
    for m in motors[1:]:
        result = result.compose(m)

    return result


def invert_transform(motor: Motor) -> Motor:
    """
    Invert a transformation.

    Args:
        motor: Motor to invert

    Returns:
        Inverse motor
    """
    return motor.inverse()


def world_to_local(
    points: torch.Tensor,
    motor: Motor
) -> torch.Tensor:
    """
    Transform points from world frame to local frame.

    Args:
        points: World-space points (..., 3)
        motor: Motor defining the local frame

    Returns:
        Local-space points
    """
    return motor.apply_inverse(points)


def local_to_world(
    points: torch.Tensor,
    motor: Motor
) -> torch.Tensor:
    """
    Transform points from local frame to world frame.

    Args:
        points: Local-space points (..., 3)
        motor: Motor defining the local frame

    Returns:
        World-space points
    """
    return motor.apply(points)

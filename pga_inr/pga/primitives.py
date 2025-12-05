"""
Geometric primitives in Projective Geometric Algebra (PGA).

PGA represents geometric objects as follows:
- Points: Grade-3 trivectors (normalized: e123 + x*e032 + y*e013 + z*e021)
- Lines: Grade-2 bivectors (Plücker coordinates)
- Planes: Grade-1 vectors (nx*e1 + ny*e2 + nz*e3 + d*e0)

Key operations:
- Join (∨): Creates higher-grade element from lower-grade ones
  - point ∨ point → line
  - point ∨ line → plane
  - line ∨ point → plane
- Meet (∧): Intersection of elements
  - plane ∧ plane → line
  - plane ∧ line → point
  - line ∧ plane → point
"""

from __future__ import annotations
from typing import Union, Optional
import torch
import torch.nn.functional as F

from .algebra import (
    Multivector,
    IDX_S,
    IDX_E0, IDX_E1, IDX_E2, IDX_E3,
    IDX_E01, IDX_E02, IDX_E03, IDX_E12, IDX_E31, IDX_E23,
    IDX_E012, IDX_E031, IDX_E023, IDX_E123,
    IDX_E0123,
)


def point(
    x: Union[float, torch.Tensor],
    y: Union[float, torch.Tensor],
    z: Union[float, torch.Tensor]
) -> Multivector:
    """
    Create a normalized PGA point from Cartesian coordinates.

    In PGA, a point is represented as:
        P = e123 + x*e032 + y*e013 + z*e021

    Where e032 = e0∧e3∧e2, e013 = e0∧e1∧e3, e021 = e0∧e2∧e1
    Note: e032 = -e023, e013 = -e031, e021 = -e012

    Args:
        x, y, z: Cartesian coordinates (scalars or tensors)

    Returns:
        Point multivector (grade-3 trivector)
    """
    # Handle scalar inputs
    if isinstance(x, (int, float)):
        x = torch.tensor(x)
    if isinstance(y, (int, float)):
        y = torch.tensor(y)
    if isinstance(z, (int, float)):
        z = torch.tensor(z)

    # Broadcast to common shape
    x, y, z = torch.broadcast_tensors(x, y, z)
    batch_shape = x.shape

    mv = torch.zeros(*batch_shape, 16, device=x.device, dtype=x.dtype)

    # Homogeneous coordinate (normalized point has weight 1)
    mv[..., IDX_E123] = 1.0

    # Position encoded in trivector components
    # P = e123 + x*e032 + y*e013 + z*e021
    # Note: e032 = -e023, e013 = -e031, e021 = -e012
    mv[..., IDX_E023] = -x  # e032 = -e023
    mv[..., IDX_E031] = -y  # e013 = -e031
    mv[..., IDX_E012] = -z  # e021 = -e012

    return Multivector(mv)


def point_from_tensor(coords: torch.Tensor) -> Multivector:
    """
    Create points from a tensor of coordinates.

    Args:
        coords: Tensor of shape (..., 3) containing [x, y, z]

    Returns:
        Point multivector
    """
    x, y, z = coords.unbind(dim=-1)
    return point(x, y, z)


def point_to_cartesian(p: Multivector) -> torch.Tensor:
    """
    Extract Cartesian coordinates from a PGA point.

    Args:
        p: Point multivector

    Returns:
        Tensor of shape (..., 3) containing [x, y, z]
    """
    mv = p.mv

    # Weight (should be 1 for normalized points)
    w = mv[..., IDX_E123]

    # Handle ideal points (w = 0)
    w = w.clamp(min=1e-12)

    # Extract coordinates (inverse of point encoding)
    x = -mv[..., IDX_E023] / w
    y = -mv[..., IDX_E031] / w
    z = -mv[..., IDX_E012] / w

    return torch.stack([x, y, z], dim=-1)


def ideal_point(
    dx: Union[float, torch.Tensor],
    dy: Union[float, torch.Tensor],
    dz: Union[float, torch.Tensor]
) -> Multivector:
    """
    Create an ideal point (point at infinity) from a direction.

    Ideal points have e123 component = 0 and represent directions.

    Args:
        dx, dy, dz: Direction components

    Returns:
        Ideal point multivector
    """
    if isinstance(dx, (int, float)):
        dx = torch.tensor(dx)
    if isinstance(dy, (int, float)):
        dy = torch.tensor(dy)
    if isinstance(dz, (int, float)):
        dz = torch.tensor(dz)

    dx, dy, dz = torch.broadcast_tensors(dx, dy, dz)
    batch_shape = dx.shape

    mv = torch.zeros(*batch_shape, 16, device=dx.device, dtype=dx.dtype)

    # Ideal point: e123 = 0
    mv[..., IDX_E023] = -dx
    mv[..., IDX_E031] = -dy
    mv[..., IDX_E012] = -dz

    return Multivector(mv)


def plane(
    nx: Union[float, torch.Tensor],
    ny: Union[float, torch.Tensor],
    nz: Union[float, torch.Tensor],
    d: Union[float, torch.Tensor]
) -> Multivector:
    """
    Create a plane from normal and distance.

    In PGA, a plane is represented as:
        π = nx*e1 + ny*e2 + nz*e3 + d*e0

    The plane equation is: nx*x + ny*y + nz*z + d = 0

    Args:
        nx, ny, nz: Normal vector components
        d: Signed distance from origin (offset)

    Returns:
        Plane multivector (grade-1 vector)
    """
    if isinstance(nx, (int, float)):
        nx = torch.tensor(nx)
    if isinstance(ny, (int, float)):
        ny = torch.tensor(ny)
    if isinstance(nz, (int, float)):
        nz = torch.tensor(nz)
    if isinstance(d, (int, float)):
        d = torch.tensor(d)

    nx, ny, nz, d = torch.broadcast_tensors(nx, ny, nz, d)
    batch_shape = nx.shape

    mv = torch.zeros(*batch_shape, 16, device=nx.device, dtype=nx.dtype)

    mv[..., IDX_E1] = nx
    mv[..., IDX_E2] = ny
    mv[..., IDX_E3] = nz
    mv[..., IDX_E0] = d

    return Multivector(mv)


def plane_from_normal_point(
    normal: torch.Tensor,
    point_on_plane: torch.Tensor
) -> Multivector:
    """
    Create a plane from normal vector and a point on the plane.

    Args:
        normal: Normal vector of shape (..., 3)
        point_on_plane: A point on the plane of shape (..., 3)

    Returns:
        Plane multivector
    """
    normal = F.normalize(normal, p=2, dim=-1)
    d = -(normal * point_on_plane).sum(dim=-1)
    nx, ny, nz = normal.unbind(dim=-1)
    return plane(nx, ny, nz, d)


def plane_to_normal_distance(p: Multivector) -> tuple:
    """
    Extract normal and distance from a plane.

    Args:
        p: Plane multivector

    Returns:
        (normal, distance) where normal is (..., 3) and distance is (...)
    """
    mv = p.mv
    normal = torch.stack([mv[..., IDX_E1], mv[..., IDX_E2], mv[..., IDX_E3]], dim=-1)
    distance = mv[..., IDX_E0]
    return normal, distance


def line_from_points(p1: Multivector, p2: Multivector) -> Multivector:
    """
    Create a line through two points using the join operation.

    L = P1 ∨ P2 (regressive product)

    Args:
        p1, p2: Point multivectors

    Returns:
        Line multivector (grade-2 bivector)
    """
    # Join = regressive product = dual of outer product of duals
    return p1.regressive(p2)


def line_from_plucker(
    direction: torch.Tensor,
    moment: torch.Tensor
) -> Multivector:
    """
    Create a line from Plücker coordinates.

    A line in PGA is a bivector with:
    - Euclidean part (e12, e31, e23): direction
    - Ideal part (e01, e02, e03): moment

    The moment satisfies: moment · direction = 0

    Args:
        direction: Line direction of shape (..., 3)
        moment: Plücker moment of shape (..., 3)

    Returns:
        Line multivector
    """
    dx, dy, dz = direction.unbind(dim=-1)
    mx, my, mz = moment.unbind(dim=-1)

    batch_shape = direction.shape[:-1]
    mv = torch.zeros(*batch_shape, 16, device=direction.device, dtype=direction.dtype)

    # Direction in Euclidean bivectors
    mv[..., IDX_E23] = dx
    mv[..., IDX_E31] = dy
    mv[..., IDX_E12] = dz

    # Moment in ideal bivectors
    mv[..., IDX_E01] = mx
    mv[..., IDX_E02] = my
    mv[..., IDX_E03] = mz

    return Multivector(mv)


def line_from_point_direction(
    point_on_line: torch.Tensor,
    direction: torch.Tensor
) -> Multivector:
    """
    Create a line from a point and direction.

    Args:
        point_on_line: A point on the line (..., 3)
        direction: Line direction (..., 3)

    Returns:
        Line multivector
    """
    # Normalize direction
    direction = F.normalize(direction, p=2, dim=-1)

    # Moment = point × direction
    moment = torch.cross(point_on_line, direction, dim=-1)

    return line_from_plucker(direction, moment)


def line_to_plucker(line: Multivector) -> tuple:
    """
    Extract Plücker coordinates from a line.

    Args:
        line: Line multivector

    Returns:
        (direction, moment) each of shape (..., 3)
    """
    mv = line.mv

    direction = torch.stack([
        mv[..., IDX_E23],
        mv[..., IDX_E31],
        mv[..., IDX_E12],
    ], dim=-1)

    moment = torch.stack([
        mv[..., IDX_E01],
        mv[..., IDX_E02],
        mv[..., IDX_E03],
    ], dim=-1)

    return direction, moment


def join(a: Multivector, b: Multivector) -> Multivector:
    """
    Join operation (regressive product).

    Creates the smallest element containing both a and b:
    - point ∨ point → line
    - point ∨ line → plane
    - line ∨ point → plane

    Args:
        a, b: Multivectors to join

    Returns:
        Join multivector
    """
    return a.regressive(b)


def meet(a: Multivector, b: Multivector) -> Multivector:
    """
    Meet operation (outer product / wedge product).

    Finds the intersection of a and b:
    - plane ∧ plane → line (their intersection)
    - plane ∧ line → point (their intersection)
    - line ∧ plane → point (their intersection)

    Args:
        a, b: Multivectors to meet

    Returns:
        Meet multivector
    """
    return a.outer(b)


def distance_point_point(p1: Multivector, p2: Multivector) -> torch.Tensor:
    """
    Euclidean distance between two points.

    Args:
        p1, p2: Point multivectors

    Returns:
        Distance tensor
    """
    c1 = point_to_cartesian(p1)
    c2 = point_to_cartesian(p2)
    return torch.norm(c1 - c2, dim=-1)


def distance_point_plane(p: Multivector, plane_mv: Multivector) -> torch.Tensor:
    """
    Signed distance from point to plane.

    Args:
        p: Point multivector
        plane_mv: Plane multivector

    Returns:
        Signed distance (positive if on normal side)
    """
    coords = point_to_cartesian(p)
    normal, d = plane_to_normal_distance(plane_mv)

    # Normalize plane if necessary
    norm_factor = torch.norm(normal, dim=-1, keepdim=True)
    normal = normal / norm_factor
    d = d / norm_factor.squeeze(-1)

    # Distance = n · p + d
    return (normal * coords).sum(dim=-1) + d


def distance_point_line(p: Multivector, line: Multivector) -> torch.Tensor:
    """
    Distance from point to line.

    Args:
        p: Point multivector
        line: Line multivector

    Returns:
        Distance tensor
    """
    coords = point_to_cartesian(p)
    direction, moment = line_to_plucker(line)

    # Normalize direction
    d_norm = torch.norm(direction, dim=-1, keepdim=True)
    direction = direction / d_norm
    moment = moment / d_norm

    # Point on line closest to origin
    # a = d × m (where d is normalized direction)
    a = torch.cross(direction, moment, dim=-1)

    # Vector from line to point
    v = coords - a

    # Project out direction component
    proj = (v * direction).sum(dim=-1, keepdim=True) * direction
    perp = v - proj

    return torch.norm(perp, dim=-1)


def project_point_plane(p: Multivector, plane_mv: Multivector) -> Multivector:
    """
    Project a point onto a plane.

    Args:
        p: Point to project
        plane_mv: Plane to project onto

    Returns:
        Projected point
    """
    coords = point_to_cartesian(p)
    normal, d = plane_to_normal_distance(plane_mv)

    # Normalize
    norm_factor = torch.norm(normal, dim=-1, keepdim=True)
    normal = normal / norm_factor
    d = d / norm_factor.squeeze(-1)

    # Signed distance
    dist = (normal * coords).sum(dim=-1, keepdim=True) + d.unsqueeze(-1)

    # Project
    projected = coords - dist * normal

    return point_from_tensor(projected)


def project_point_line(p: Multivector, line: Multivector) -> Multivector:
    """
    Project a point onto a line.

    Args:
        p: Point to project
        line: Line to project onto

    Returns:
        Projected point
    """
    coords = point_to_cartesian(p)
    direction, moment = line_to_plucker(line)

    # Normalize direction
    d_norm = torch.norm(direction, dim=-1, keepdim=True)
    direction = direction / d_norm
    moment = moment / d_norm

    # Point on line closest to origin
    a = torch.cross(direction, moment, dim=-1)

    # Vector from that point to input point
    v = coords - a

    # Project onto line direction
    proj_length = (v * direction).sum(dim=-1, keepdim=True)
    projected = a + proj_length * direction

    return point_from_tensor(projected)


def reflect_point_plane(p: Multivector, plane_mv: Multivector) -> Multivector:
    """
    Reflect a point across a plane.

    Uses the sandwich product: P' = π * P * π

    Args:
        p: Point to reflect
        plane_mv: Mirror plane

    Returns:
        Reflected point
    """
    # Normalize plane
    normal, _ = plane_to_normal_distance(plane_mv)
    norm_factor = torch.norm(normal, dim=-1, keepdim=True).unsqueeze(-1)
    normalized_plane = Multivector(plane_mv.mv / norm_factor)

    # Sandwich product
    return normalized_plane * p * normalized_plane


def origin() -> Multivector:
    """Create the origin point (0, 0, 0)."""
    return point(0.0, 0.0, 0.0)


def xy_plane() -> Multivector:
    """Create the XY plane (z = 0)."""
    return plane(0.0, 0.0, 1.0, 0.0)


def xz_plane() -> Multivector:
    """Create the XZ plane (y = 0)."""
    return plane(0.0, 1.0, 0.0, 0.0)


def yz_plane() -> Multivector:
    """Create the YZ plane (x = 0)."""
    return plane(1.0, 0.0, 0.0, 0.0)


def x_axis() -> Multivector:
    """Create the X axis (line through origin in x direction)."""
    return line_from_plucker(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0])
    )


def y_axis() -> Multivector:
    """Create the Y axis."""
    return line_from_plucker(
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0])
    )


def z_axis() -> Multivector:
    """Create the Z axis."""
    return line_from_plucker(
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([0.0, 0.0, 0.0])
    )

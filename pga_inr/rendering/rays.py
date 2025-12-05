"""
Ray generation utilities for rendering PGA-INR models.

Provides functions for:
- Generating rays from camera intrinsics and pose
- PGA-based ray representation
- Ray transformations
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def generate_rays(
    height: int,
    width: int,
    focal: float,
    camera_pose: torch.Tensor,
    center: Optional[Tuple[float, float]] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate camera rays for each pixel.

    Args:
        height: Image height
        width: Image width
        focal: Focal length in pixels
        camera_pose: Camera-to-world transform (4, 4)
        center: Optional (cx, cy) principal point (defaults to center)
        device: Device for tensors

    Returns:
        (origins, directions) each of shape (H, W, 3)
    """
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center

    # Create pixel coordinate grid at pixel CENTERS (add 0.5)
    # Pixel (i, j) covers area from (i, j) to (i+1, j+1), center is at (i+0.5, j+0.5)
    i, j = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device) + 0.5,
        torch.arange(width, dtype=torch.float32, device=device) + 0.5,
        indexing='ij'
    )

    # Convert to normalized camera coordinates
    # Note: y is flipped to match image coordinates (y-up in camera, y-down in image)
    x = (j - cx) / focal
    y = -(i - cy) / focal
    z = -torch.ones_like(x)  # Looking down -z axis

    # Stack to get directions in camera space
    directions_cam = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    # Normalize directions
    directions_cam = F.normalize(directions_cam, dim=-1)

    # Transform to world space
    rotation = camera_pose[:3, :3]  # (3, 3)
    translation = camera_pose[:3, 3]  # (3,)

    # Rotate directions
    directions_world = torch.einsum('ij,...j->...i', rotation, directions_cam)

    # Origin is camera position (same for all rays)
    origins = translation.expand(height, width, 3)

    return origins, directions_world


def generate_rays_from_intrinsics(
    height: int,
    width: int,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from camera intrinsics matrix.

    Args:
        height: Image height
        width: Image width
        intrinsics: Camera intrinsics matrix (3, 3)
        camera_pose: Camera-to-world transform (4, 4)
        device: Device for tensors

    Returns:
        (origins, directions) each of shape (H, W, 3)
    """
    # Extract intrinsics
    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()
    cx = intrinsics[0, 2].item()
    cy = intrinsics[1, 2].item()

    # Create pixel coordinate grid at pixel CENTERS (add 0.5)
    i, j = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device) + 0.5,
        torch.arange(width, dtype=torch.float32, device=device) + 0.5,
        indexing='ij'
    )

    # Convert to camera coordinates
    x = (j - cx) / fx
    y = -(i - cy) / fy  # Flip y (y-up in camera, y-down in image)
    z = -torch.ones_like(x)

    directions_cam = torch.stack([x, y, z], dim=-1)
    directions_cam = F.normalize(directions_cam, dim=-1)

    # Transform to world
    rotation = camera_pose[:3, :3]
    translation = camera_pose[:3, 3]

    directions_world = torch.einsum('ij,...j->...i', rotation, directions_cam)
    origins = translation.expand(height, width, 3)

    return origins, directions_world


def generate_rays_from_pose(
    intrinsics: torch.Tensor,
    translation: torch.Tensor,
    quaternion: torch.Tensor,
    height: int,
    width: int,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from translation and quaternion pose.

    Args:
        intrinsics: Camera intrinsics (3, 3)
        translation: Camera position (3,)
        quaternion: Camera rotation as quaternion [w, x, y, z] (4,)
        height: Image height
        width: Image width
        device: Device for tensors

    Returns:
        (origins, directions) each of shape (H, W, 3)
    """
    from ..utils.quaternion import quaternion_to_matrix

    # Convert quaternion to rotation matrix
    rotation = quaternion_to_matrix(quaternion.unsqueeze(0)).squeeze(0)

    # Build camera-to-world matrix
    camera_pose = torch.eye(4, device=device)
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = translation

    return generate_rays_from_intrinsics(
        height, width, intrinsics, camera_pose, device
    )


def generate_rays_from_matrix(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    height: int,
    width: int,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from intrinsics and extrinsics matrices.

    Args:
        intrinsics: Camera intrinsics (3, 3)
        extrinsics: Camera extrinsics (4, 4) - world-to-camera
        height: Image height
        width: Image width
        device: Device for tensors

    Returns:
        (origins, directions) each of shape (H, W, 3)
    """
    # Invert extrinsics to get camera-to-world
    camera_pose = torch.inverse(extrinsics)

    return generate_rays_from_intrinsics(
        height, width, intrinsics, camera_pose, device
    )


def rays_to_pga(
    origins: torch.Tensor,
    directions: torch.Tensor
) -> torch.Tensor:
    """
    Convert rays to PGA line representation.

    A ray/line in PGA is represented as a bivector (grade 2).

    Args:
        origins: Ray origins (..., 3)
        directions: Ray directions (..., 3)

    Returns:
        PGA bivector representation (..., 6)
    """
    # Line Plücker coordinates
    # Direction part (e12, e31, e23)
    d = directions

    # Moment part (e01, e02, e03)
    m = torch.cross(origins, directions, dim=-1)

    # Pack into bivector [e01, e02, e03, e12, e31, e23]
    return torch.cat([m, d], dim=-1)


def pga_to_rays(
    bivector: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert PGA line to ray representation.

    Args:
        bivector: PGA bivector (..., 6)

    Returns:
        (origins, directions) - Note: origin is the point on ray closest to origin
    """
    # Unpack
    m = bivector[..., :3]  # moment
    d = bivector[..., 3:]  # direction

    # Normalize direction
    d_norm = d / (d.norm(dim=-1, keepdim=True) + 1e-8)

    # Origin is m × d / |d|²
    d_sq = (d ** 2).sum(dim=-1, keepdim=True)
    origin = torch.cross(m, d, dim=-1) / (d_sq + 1e-8)

    return origin, d_norm


def generate_sphere_rays(
    center: torch.Tensor,
    radius: float,
    num_rays: int,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays pointing inward from a sphere.

    Useful for 360° view synthesis.

    Args:
        center: Sphere center (3,)
        radius: Sphere radius
        num_rays: Number of rays
        device: Device for tensors

    Returns:
        (origins, directions)
    """
    # Fibonacci sphere for uniform distribution
    indices = torch.arange(num_rays, dtype=torch.float32, device=device)
    phi = torch.acos(1 - 2 * (indices + 0.5) / num_rays)
    theta = torch.pi * (1 + 5 ** 0.5) * indices

    # Convert to Cartesian
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    # Points on sphere
    points = torch.stack([x, y, z], dim=-1)  # (N, 3)
    origins = center + radius * points

    # Directions point toward center
    directions = -points  # Already normalized

    return origins, directions


def generate_orbit_rays(
    center: torch.Tensor,
    radius: float,
    height: float,
    num_views: int,
    image_height: int,
    image_width: int,
    focal: float,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for orbital camera path around an object.

    Args:
        center: Object center (3,)
        radius: Orbit radius
        height: Camera height above center
        num_views: Number of camera positions
        image_height: Image height
        image_width: Image width
        focal: Focal length
        device: Device for tensors

    Returns:
        (origins, directions) each of shape (num_views, H, W, 3)
    """
    all_origins = []
    all_directions = []

    for i in range(num_views):
        # Camera position on orbit
        angle = 2 * torch.pi * i / num_views
        cam_x = center[0] + radius * torch.cos(torch.tensor(angle))
        cam_y = center[1] + height
        cam_z = center[2] + radius * torch.sin(torch.tensor(angle))
        cam_pos = torch.tensor([cam_x, cam_y, cam_z], device=device)

        # Look at center
        forward = F.normalize(center - cam_pos, dim=0)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = F.normalize(torch.cross(forward, up), dim=0)
        up = torch.cross(right, forward)

        # Build rotation matrix (camera looks down -z)
        rotation = torch.stack([right, up, -forward], dim=1)

        # Camera pose
        pose = torch.eye(4, device=device)
        pose[:3, :3] = rotation
        pose[:3, 3] = cam_pos

        # Generate rays
        origins, directions = generate_rays(
            image_height, image_width, focal, pose, device=device
        )

        all_origins.append(origins)
        all_directions.append(directions)

    return torch.stack(all_origins), torch.stack(all_directions)


def sample_rays_uniform(
    origins: torch.Tensor,
    directions: torch.Tensor,
    num_rays: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly sample rays from a full ray buffer.

    Args:
        origins: All ray origins (H, W, 3)
        directions: All ray directions (H, W, 3)
        num_rays: Number of rays to sample

    Returns:
        (sampled_origins, sampled_directions, indices)
    """
    H, W = origins.shape[:2]
    device = origins.device

    # Random indices
    indices = torch.randint(0, H * W, (num_rays,), device=device)

    # Flatten and sample
    origins_flat = origins.view(-1, 3)
    directions_flat = directions.view(-1, 3)

    return origins_flat[indices], directions_flat[indices], indices


def transform_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    translation: torch.Tensor,
    quaternion: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform rays by a rigid motion.

    Args:
        origins: Ray origins (..., 3)
        directions: Ray directions (..., 3)
        translation: Translation (3,) or (B, 3)
        quaternion: Rotation quaternion [w, x, y, z] (4,) or (B, 4)

    Returns:
        (transformed_origins, transformed_directions)
    """
    from ..utils.quaternion import quaternion_to_matrix

    # Get rotation matrix
    if quaternion.dim() == 1:
        quaternion = quaternion.unsqueeze(0)
    rotation = quaternion_to_matrix(quaternion)

    if rotation.shape[0] == 1:
        rotation = rotation.squeeze(0)
        # Single transform
        new_origins = torch.einsum('ij,...j->...i', rotation, origins) + translation
        new_directions = torch.einsum('ij,...j->...i', rotation, directions)
    else:
        # Batched transform
        new_origins = torch.einsum('bij,...j->...bi', rotation, origins) + translation
        new_directions = torch.einsum('bij,...j->...bi', rotation, directions)

    return new_origins, new_directions


def rays_aabb_intersection(
    origins: torch.Tensor,
    directions: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ray-AABB intersection.

    Args:
        origins: Ray origins (..., 3)
        directions: Ray directions (..., 3)
        box_min: AABB minimum corner (3,)
        box_max: AABB maximum corner (3,)

    Returns:
        (t_near, t_far, mask) where mask indicates valid intersections
    """
    # Avoid division by zero
    inv_dir = 1.0 / (directions + 1e-8)

    # Compute t values for each slab
    t0 = (box_min - origins) * inv_dir
    t1 = (box_max - origins) * inv_dir

    # Handle negative directions
    t_min = torch.minimum(t0, t1)
    t_max = torch.maximum(t0, t1)

    # Find the largest t_min and smallest t_max
    t_near = t_min.max(dim=-1).values
    t_far = t_max.min(dim=-1).values

    # Valid if t_near < t_far and t_far > 0
    mask = (t_near < t_far) & (t_far > 0)

    # Clamp t_near to be positive
    t_near = torch.clamp(t_near, min=0)

    return t_near, t_far, mask


def rays_sphere_intersection(
    origins: torch.Tensor,
    directions: torch.Tensor,
    center: torch.Tensor,
    radius: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ray-sphere intersection.

    Args:
        origins: Ray origins (..., 3)
        directions: Ray directions (..., 3)
        center: Sphere center (3,)
        radius: Sphere radius

    Returns:
        (t_near, t_far, mask) where mask indicates valid intersections
    """
    # Vector from origin to sphere center
    oc = origins - center

    # Quadratic coefficients
    a = (directions ** 2).sum(dim=-1)
    b = 2 * (oc * directions).sum(dim=-1)
    c = (oc ** 2).sum(dim=-1) - radius ** 2

    # Discriminant
    discriminant = b ** 2 - 4 * a * c

    # Valid intersections
    mask = discriminant >= 0

    # Compute t values (handling invalid cases)
    sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0))
    t_near = (-b - sqrt_disc) / (2 * a + 1e-8)
    t_far = (-b + sqrt_disc) / (2 * a + 1e-8)

    # Clamp to positive
    t_near = torch.clamp(t_near, min=0)

    return t_near, t_far, mask


def get_ray_bundle(
    height: int,
    width: int,
    focal: float,
    camera_pose: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Get a complete ray bundle with additional information.

    Args:
        height: Image height
        width: Image width
        focal: Focal length
        camera_pose: Camera pose (4, 4)
        device: Device

    Returns:
        Dictionary with origins, directions, pixel coordinates, etc.
    """
    origins, directions = generate_rays(
        height, width, focal, camera_pose, device=device
    )

    # Pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )

    return {
        'origins': origins,
        'directions': directions,
        'pixel_coords': torch.stack([j, i], dim=-1),  # (x, y) format
        'height': height,
        'width': width,
        'focal': focal,
        'camera_pose': camera_pose,
    }

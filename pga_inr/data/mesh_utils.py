"""
Mesh processing utilities for SDF computation.

Provides functions for:
- Loading meshes from various formats
- Normalizing meshes to unit cube
- Computing SDF values
- Sampling surface points
- Mesh to voxel grid conversion
"""

from typing import Tuple, Optional, Union, List
from pathlib import Path
import numpy as np
import torch

# Type hints for trimesh objects
try:
    import trimesh
    Mesh = trimesh.Trimesh
except ImportError:
    Mesh = object


def load_mesh(path: Union[str, Path]) -> Mesh:
    """
    Load a mesh from file.

    Supports .obj, .ply, .stl, .off, and other formats via trimesh.

    Args:
        path: Path to mesh file

    Returns:
        Loaded mesh
    """
    import trimesh

    mesh = trimesh.load(str(path), force='mesh')

    # If loaded as Scene, extract the geometry
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 1:
            mesh = list(mesh.geometry.values())[0]
        else:
            # Combine all geometries
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    return mesh


def save_mesh(mesh: Mesh, path: Union[str, Path], file_type: Optional[str] = None):
    """
    Save a mesh to file.

    Args:
        mesh: Mesh to save
        path: Output path
        file_type: Optional explicit file type
    """
    mesh.export(str(path), file_type=file_type)


def normalize_mesh(
    mesh: Mesh,
    center: bool = True,
    scale: float = 1.0
) -> Mesh:
    """
    Normalize mesh to fit within a unit cube centered at origin.

    Args:
        mesh: Input mesh
        center: Whether to center at origin
        scale: Scale factor (1.0 = unit cube)

    Returns:
        Normalized mesh copy
    """
    import trimesh

    # Make a copy
    mesh = mesh.copy()

    # Center at origin
    if center:
        centroid = mesh.vertices.mean(axis=0)
        mesh.vertices -= centroid

    # Scale to fit in [-scale, scale]³
    max_extent = np.abs(mesh.vertices).max()
    if max_extent > 0:
        mesh.vertices *= scale / max_extent

    return mesh


def compute_sdf(
    mesh: Mesh,
    points: np.ndarray,
    use_winding_number: bool = True
) -> np.ndarray:
    """
    Compute signed distance field values for query points.

    Args:
        mesh: Reference mesh
        points: Query points (N, 3)
        use_winding_number: Use winding number for sign (more robust)

    Returns:
        SDF values (N,)
    """
    import trimesh

    # Compute unsigned distances
    closest_points, distances, triangle_ids = mesh.nearest.on_surface(points)

    # Compute signs using winding number or ray casting
    if use_winding_number:
        # Winding number method (more robust for non-watertight meshes)
        try:
            from trimesh import proximity
            contains = proximity.check_winding_number(mesh, points)
        except (AttributeError, ImportError):
            # Fallback to ray casting
            contains = mesh.contains(points)
    else:
        # Ray casting (faster but requires watertight mesh)
        contains = mesh.contains(points)

    # Apply signs: negative inside, positive outside
    sdf = distances.copy()
    sdf[contains] *= -1

    return sdf.astype(np.float32)


def compute_sdf_grid(
    mesh: Mesh,
    resolution: int = 64,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    padding: float = 0.1
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute SDF on a regular 3D grid.

    Args:
        mesh: Reference mesh
        resolution: Grid resolution per dimension
        bounds: Optional (min, max) bounds
        padding: Padding fraction around mesh

    Returns:
        (sdf_grid, (min_bound, max_bound)) where sdf_grid is (res, res, res)
    """
    # Compute bounds
    if bounds is None:
        min_bound = mesh.vertices.min(axis=0)
        max_bound = mesh.vertices.max(axis=0)

        # Add padding
        extent = max_bound - min_bound
        min_bound -= padding * extent
        max_bound += padding * extent
    else:
        min_bound, max_bound = bounds

    # Create grid
    x = np.linspace(min_bound[0], max_bound[0], resolution)
    y = np.linspace(min_bound[1], max_bound[1], resolution)
    z = np.linspace(min_bound[2], max_bound[2], resolution)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    # Compute SDF
    sdf = compute_sdf(mesh, points)
    sdf_grid = sdf.reshape(resolution, resolution, resolution)

    return sdf_grid, (min_bound, max_bound)


def sample_surface_points(
    mesh: Mesh,
    num_points: int = 10000,
    return_normals: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample points uniformly on mesh surface.

    Args:
        mesh: Input mesh
        num_points: Number of points to sample
        return_normals: Whether to return normals

    Returns:
        points (N, 3) or (points, normals) if return_normals=True
    """
    # Sample points on surface
    points, face_indices = mesh.sample(num_points, return_index=True)
    points = torch.from_numpy(points.astype(np.float32))

    if return_normals:
        # Get face normals for sampled points
        normals = mesh.face_normals[face_indices]
        normals = torch.from_numpy(normals.astype(np.float32))
        return points, normals
    else:
        return points


def sample_near_surface(
    mesh: Mesh,
    num_points: int = 10000,
    sigma: float = 0.05,
    return_sdf: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample points near the mesh surface with Gaussian offset.

    Args:
        mesh: Input mesh
        num_points: Number of points to sample
        sigma: Standard deviation of offset
        return_sdf: Whether to compute and return SDF values

    Returns:
        points (N, 3) or (points, sdf) if return_sdf=True
    """
    # Sample surface points with normals
    surface_points, normals = sample_surface_points(mesh, num_points, return_normals=True)

    # Add Gaussian offset along normal direction
    offsets = torch.randn(num_points, 1) * sigma
    points = surface_points + offsets * normals

    if return_sdf:
        sdf = compute_sdf(mesh, points.numpy())
        sdf = torch.from_numpy(sdf).float()
        return points, sdf
    else:
        return points


def compute_normals_from_sdf(
    sdf_fn,
    points: torch.Tensor,
    epsilon: float = 1e-3
) -> torch.Tensor:
    """
    Compute normals from SDF using finite differences.

    Args:
        sdf_fn: Function that takes points (N, 3) and returns SDF (N,)
        points: Query points (N, 3)
        epsilon: Finite difference step size

    Returns:
        Normals (N, 3)
    """
    # Create offset points
    dx = torch.tensor([[epsilon, 0, 0]], device=points.device)
    dy = torch.tensor([[0, epsilon, 0]], device=points.device)
    dz = torch.tensor([[0, 0, epsilon]], device=points.device)

    # Central differences
    grad_x = (sdf_fn(points + dx) - sdf_fn(points - dx)) / (2 * epsilon)
    grad_y = (sdf_fn(points + dy) - sdf_fn(points - dy)) / (2 * epsilon)
    grad_z = (sdf_fn(points + dz) - sdf_fn(points - dz)) / (2 * epsilon)

    normals = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

    return normals


def mesh_to_voxels(
    mesh: Mesh,
    resolution: int = 64,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    Convert mesh to binary voxel grid.

    Args:
        mesh: Input mesh
        resolution: Voxel resolution per dimension
        bounds: Optional (min, max) bounds

    Returns:
        Binary voxel grid (res, res, res)
    """
    import trimesh

    # Use trimesh's voxelization
    if bounds is not None:
        # Create transformation to fit bounds
        min_bound, max_bound = bounds
        scale = (max_bound - min_bound).max() / resolution
        origin = min_bound

        voxels = mesh.voxelized(pitch=scale, origin=origin)
    else:
        pitch = mesh.extents.max() / resolution
        voxels = mesh.voxelized(pitch=pitch)

    # Convert to dense grid
    grid = voxels.matrix.astype(np.bool_)

    # Pad or crop to exact resolution
    if grid.shape != (resolution, resolution, resolution):
        result = np.zeros((resolution, resolution, resolution), dtype=np.bool_)
        slices = tuple(slice(0, min(s, resolution)) for s in grid.shape)
        result[slices] = grid[slices]
        grid = result

    return grid


def voxels_to_mesh(
    voxels: np.ndarray,
    level: float = 0.5,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Mesh:
    """
    Convert voxel grid (or SDF grid) to mesh using marching cubes.

    Args:
        voxels: Voxel/SDF grid (X, Y, Z)
        level: Isosurface level
        bounds: Optional (min, max) bounds for vertex positions

    Returns:
        Extracted mesh
    """
    import trimesh
    from skimage import measure

    # Marching cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(voxels, level=level)
    except ValueError:
        # No surface found
        return trimesh.Trimesh()

    # Scale to bounds
    if bounds is not None:
        min_bound, max_bound = bounds
        scale = (max_bound - min_bound) / np.array(voxels.shape)
        vertices = vertices * scale + min_bound

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    return mesh


def sdf_to_mesh(
    sdf_fn,
    resolution: int = 128,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    level: float = 0.0,
    device: torch.device = torch.device('cpu'),
    batch_size: int = 100000
) -> Mesh:
    """
    Extract mesh from a neural SDF using marching cubes.

    Args:
        sdf_fn: Function that takes points (N, 3) and returns SDF (N,) or (N, 1)
        resolution: Grid resolution
        bounds: Coordinate bounds (min, max)
        level: Isosurface level
        device: Device for computation
        batch_size: Batch size for SDF evaluation

    Returns:
        Extracted mesh
    """
    import trimesh
    from skimage import measure

    # Create evaluation grid
    coords = torch.linspace(bounds[0], bounds[1], resolution, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
    points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    # Evaluate SDF in batches
    sdf_values = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        with torch.no_grad():
            sdf = sdf_fn(batch)
            if sdf.dim() == 2:
                sdf = sdf.squeeze(-1)
            sdf_values.append(sdf.cpu())

    sdf_values = torch.cat(sdf_values).numpy()
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)

    # Marching cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(sdf_grid, level=level)
    except ValueError:
        return trimesh.Trimesh()

    # Scale vertices to world coordinates
    vertices = vertices / (resolution - 1) * (bounds[1] - bounds[0]) + bounds[0]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    return mesh


def compute_mesh_metrics(
    pred_mesh: Mesh,
    gt_mesh: Mesh,
    num_samples: int = 100000
) -> dict:
    """
    Compute metrics between predicted and ground truth meshes.

    Args:
        pred_mesh: Predicted mesh
        gt_mesh: Ground truth mesh
        num_samples: Number of samples for metrics

    Returns:
        Dictionary of metrics (chamfer, hausdorff, f1, etc.)
    """
    # Sample points on both meshes
    pred_points = pred_mesh.sample(num_samples)
    gt_points = gt_mesh.sample(num_samples)

    # Compute distances from pred to gt
    _, pred_to_gt, _ = gt_mesh.nearest.on_surface(pred_points)

    # Compute distances from gt to pred
    _, gt_to_pred, _ = pred_mesh.nearest.on_surface(gt_points)

    # Chamfer distance (mean of both directions)
    chamfer = (pred_to_gt.mean() + gt_to_pred.mean()) / 2

    # Hausdorff distance (max of both directions)
    hausdorff = max(pred_to_gt.max(), gt_to_pred.max())

    # F1 score at threshold
    threshold = 0.01
    precision = (pred_to_gt < threshold).mean()
    recall = (gt_to_pred < threshold).mean()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'chamfer': float(chamfer),
        'hausdorff': float(hausdorff),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
    }


def simplify_mesh(
    mesh: Mesh,
    target_faces: int = 10000,
    method: str = 'quadric'
) -> Mesh:
    """
    Simplify mesh to reduce face count.

    Args:
        mesh: Input mesh
        target_faces: Target number of faces
        method: Simplification method ('quadric' or 'vertex_clustering')

    Returns:
        Simplified mesh
    """
    if method == 'quadric':
        simplified = mesh.simplify_quadric_decimation(target_faces)
    else:
        # Vertex clustering (faster but lower quality)
        voxel_size = mesh.extents.max() / (target_faces ** (1/3))
        simplified = mesh.simplify_vertex_clustering(voxel_size)

    return simplified


def repair_mesh(mesh: Mesh) -> Mesh:
    """
    Attempt to repair mesh (fill holes, fix normals, etc.).

    Args:
        mesh: Input mesh

    Returns:
        Repaired mesh
    """
    import trimesh

    # Make copy
    mesh = mesh.copy()

    # Fix normals
    mesh.fix_normals()

    # Fill holes
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    # Remove degenerate faces
    mesh.remove_degenerate_faces()

    # Remove duplicate faces
    mesh.remove_duplicate_faces()

    return mesh


def subdivide_mesh(mesh: Mesh, iterations: int = 1) -> Mesh:
    """
    Subdivide mesh to increase resolution.

    Args:
        mesh: Input mesh
        iterations: Number of subdivision iterations

    Returns:
        Subdivided mesh
    """
    for _ in range(iterations):
        mesh = mesh.subdivide()
    return mesh


def compute_surface_area(mesh: Mesh) -> float:
    """Compute total surface area of mesh."""
    return float(mesh.area)


def compute_volume(mesh: Mesh) -> float:
    """Compute enclosed volume of mesh (requires watertight)."""
    if mesh.is_watertight:
        return float(mesh.volume)
    else:
        # Attempt to make watertight
        try:
            import trimesh
            trimesh.repair.fill_holes(mesh)
            if mesh.is_watertight:
                return float(mesh.volume)
        except Exception:
            pass
        return 0.0


def get_bounding_box(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get axis-aligned bounding box.

    Returns:
        (min_corner, max_corner) each of shape (3,)
    """
    return mesh.bounds[0], mesh.bounds[1]


def transform_mesh(
    mesh: Mesh,
    translation: Optional[np.ndarray] = None,
    rotation: Optional[np.ndarray] = None,
    scale: Optional[float] = None
) -> Mesh:
    """
    Apply transformation to mesh.

    Args:
        mesh: Input mesh
        translation: Translation vector (3,)
        rotation: Rotation matrix (3, 3) or quaternion (4,)
        scale: Uniform scale factor

    Returns:
        Transformed mesh copy
    """
    mesh = mesh.copy()

    if scale is not None:
        mesh.vertices *= scale

    if rotation is not None:
        if rotation.shape == (4,):
            # Quaternion to matrix
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_quat(rotation[[1, 2, 3, 0]]).as_matrix()
        mesh.vertices = mesh.vertices @ rotation.T

    if translation is not None:
        mesh.vertices += translation

    return mesh

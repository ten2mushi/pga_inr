"""
Dataset classes for SDF training.

Provides PyTorch Dataset implementations for:
- Single object SDF training
- Multi-object generative training
- Image-based supervision
- Point cloud data
"""

from typing import Dict, Tuple, Optional, List, Union, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SDFDataset(Dataset):
    """
    Dataset for single object SDF supervision.

    Returns point samples with ground truth SDF values and normals.
    """

    def __init__(
        self,
        points: torch.Tensor,
        sdf_values: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        colors: Optional[torch.Tensor] = None,
        num_samples: int = 10000,
        surface_ratio: float = 0.5
    ):
        """
        Args:
            points: Precomputed sample points (N, 3)
            sdf_values: Ground truth SDF values (N,)
            normals: Ground truth normals (N, 3) (optional)
            colors: Ground truth colors (N, 3) (optional)
            num_samples: Samples per batch
            surface_ratio: Fraction of samples near surface
        """
        self.points = points
        self.sdf_values = sdf_values.view(-1, 1)
        self.normals = normals
        self.colors = colors
        self.num_samples = num_samples
        self.surface_ratio = surface_ratio

        # Identify near-surface points (|sdf| < threshold)
        threshold = 0.05
        self.surface_mask = (self.sdf_values.abs() < threshold).squeeze()
        self.surface_indices = torch.where(self.surface_mask)[0]
        self.volume_indices = torch.where(~self.surface_mask)[0]

    def __len__(self) -> int:
        return len(self.points) // self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample with surface bias
        num_surface = int(self.num_samples * self.surface_ratio)
        num_volume = self.num_samples - num_surface

        # Surface samples
        if len(self.surface_indices) > 0:
            surf_idx = self.surface_indices[
                torch.randint(0, len(self.surface_indices), (num_surface,))
            ]
        else:
            surf_idx = torch.randint(0, len(self.points), (num_surface,))

        # Volume samples
        if len(self.volume_indices) > 0:
            vol_idx = self.volume_indices[
                torch.randint(0, len(self.volume_indices), (num_volume,))
            ]
        else:
            vol_idx = torch.randint(0, len(self.points), (num_volume,))

        indices = torch.cat([surf_idx, vol_idx])

        result = {
            'points': self.points[indices],
            'sdf': self.sdf_values[indices],
        }

        if self.normals is not None:
            result['normals'] = self.normals[indices]

        if self.colors is not None:
            result['colors'] = self.colors[indices]

        return result


class SDFDatasetFromMesh(Dataset):
    """
    SDF Dataset that loads and samples from a mesh file.

    Uses mesh_utils for SDF computation.
    """

    def __init__(
        self,
        mesh_path: Union[str, Path],
        num_samples: int = 10000,
        surface_ratio: float = 0.5,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        normalize: bool = True,
        cache_samples: bool = True,
        cache_size: int = 100000
    ):
        """
        Args:
            mesh_path: Path to mesh file (.obj, .ply, .stl)
            num_samples: Samples per batch
            surface_ratio: Fraction of samples near surface
            bounds: Bounding box for uniform sampling
            normalize: Whether to normalize mesh to unit cube
            cache_samples: Whether to cache precomputed samples
            cache_size: Size of sample cache
        """
        from . import mesh_utils

        self.mesh_path = Path(mesh_path)
        self.num_samples = num_samples
        self.surface_ratio = surface_ratio
        self.bounds = bounds
        self.cache_samples = cache_samples

        # Load and process mesh
        self.mesh = mesh_utils.load_mesh(str(mesh_path))
        if normalize:
            self.mesh = mesh_utils.normalize_mesh(self.mesh)

        # Precompute surface points
        self.surface_points, self.surface_normals = mesh_utils.sample_surface_points(
            self.mesh, num_points=cache_size // 2
        )

        # Precompute cached samples if requested
        if cache_samples:
            self._build_cache(cache_size)

    def _build_cache(self, cache_size: int):
        """Precompute a cache of samples with SDF values."""
        from . import mesh_utils

        # Mixed sampling
        num_surface = cache_size // 2
        num_volume = cache_size - num_surface

        # Surface-near samples
        surface_samples = self.surface_points + torch.randn_like(self.surface_points) * 0.05

        # Uniform volume samples
        volume_samples = torch.rand(num_volume, 3) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        self.cached_points = torch.cat([surface_samples[:num_surface], volume_samples], dim=0)

        # Compute SDF values
        self.cached_sdf = mesh_utils.compute_sdf(
            self.mesh, self.cached_points.numpy()
        )
        self.cached_sdf = torch.from_numpy(self.cached_sdf).float().view(-1, 1)

    def __len__(self) -> int:
        if self.cache_samples:
            return len(self.cached_points) // self.num_samples
        return 1000  # Virtual length for dynamic sampling

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_samples:
            # Sample from cache
            indices = torch.randint(0, len(self.cached_points), (self.num_samples,))
            return {
                'points': self.cached_points[indices],
                'sdf': self.cached_sdf[indices],
            }
        else:
            # Dynamic sampling (slower but more diverse)
            from . import mesh_utils

            num_surface = int(self.num_samples * self.surface_ratio)
            num_volume = self.num_samples - num_surface

            # Surface samples
            surf_idx = torch.randint(0, len(self.surface_points), (num_surface,))
            surface_pts = self.surface_points[surf_idx]
            surface_pts = surface_pts + torch.randn_like(surface_pts) * 0.02

            # Volume samples
            volume_pts = torch.rand(num_volume, 3) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

            points = torch.cat([surface_pts, volume_pts], dim=0)
            sdf = mesh_utils.compute_sdf(self.mesh, points.numpy())

            return {
                'points': points,
                'sdf': torch.from_numpy(sdf).float().view(-1, 1),
            }


class MultiObjectSDFDataset(Dataset):
    """
    Dataset for multiple objects (generative training).

    Returns samples with object indices for auto-decoding.
    """

    def __init__(
        self,
        object_data: List[Dict[str, torch.Tensor]],
        num_samples_per_object: int = 5000,
        surface_ratio: float = 0.5
    ):
        """
        Args:
            object_data: List of dicts with 'points', 'sdf', optional 'normals'
            num_samples_per_object: Samples per object per batch
            surface_ratio: Fraction near surface
        """
        self.objects = object_data
        self.num_objects = len(object_data)
        self.num_samples = num_samples_per_object
        self.surface_ratio = surface_ratio

        # Precompute surface masks for each object
        self.surface_masks = []
        for obj in self.objects:
            threshold = 0.05
            mask = (obj['sdf'].abs() < threshold).squeeze()
            self.surface_masks.append(mask)

    def __len__(self) -> int:
        # One batch per object
        return self.num_objects

    def __getitem__(self, obj_idx: int) -> Dict[str, torch.Tensor]:
        obj = self.objects[obj_idx]
        mask = self.surface_masks[obj_idx]

        surface_indices = torch.where(mask)[0]
        volume_indices = torch.where(~mask)[0]

        num_surface = int(self.num_samples * self.surface_ratio)
        num_volume = self.num_samples - num_surface

        # Sample indices
        if len(surface_indices) > 0:
            surf_idx = surface_indices[
                torch.randint(0, len(surface_indices), (num_surface,))
            ]
        else:
            surf_idx = torch.randint(0, len(obj['points']), (num_surface,))

        if len(volume_indices) > 0:
            vol_idx = volume_indices[
                torch.randint(0, len(volume_indices), (num_volume,))
            ]
        else:
            vol_idx = torch.randint(0, len(obj['points']), (num_volume,))

        indices = torch.cat([surf_idx, vol_idx])

        result = {
            'points': obj['points'][indices],
            'sdf': obj['sdf'][indices].view(-1, 1),
            'object_idx': torch.tensor(obj_idx, dtype=torch.long),
        }

        if 'normals' in obj:
            result['normals'] = obj['normals'][indices]

        return result


class MultiObjectSDFDatasetFromMeshes(Dataset):
    """
    Multi-object dataset that loads from mesh files.
    """

    def __init__(
        self,
        mesh_paths: List[Union[str, Path]],
        num_samples_per_object: int = 5000,
        surface_ratio: float = 0.5,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        cache_size_per_object: int = 50000
    ):
        """
        Args:
            mesh_paths: List of paths to mesh files
            num_samples_per_object: Samples per object per batch
            surface_ratio: Fraction near surface
            bounds: Bounding box
            cache_size_per_object: Cache size for each object
        """
        from . import mesh_utils

        self.mesh_paths = [Path(p) for p in mesh_paths]
        self.num_objects = len(mesh_paths)
        self.num_samples = num_samples_per_object
        self.surface_ratio = surface_ratio
        self.bounds = bounds

        # Load and cache samples for each object
        self.object_caches = []

        for mesh_path in self.mesh_paths:
            mesh = mesh_utils.load_mesh(str(mesh_path))
            mesh = mesh_utils.normalize_mesh(mesh)

            # Sample points
            surface_pts, surface_normals = mesh_utils.sample_surface_points(
                mesh, num_points=cache_size_per_object // 2
            )

            num_surface = cache_size_per_object // 2
            num_volume = cache_size_per_object - num_surface

            # Surface-near samples
            surface_samples = surface_pts + torch.randn_like(surface_pts) * 0.05

            # Volume samples
            volume_samples = torch.rand(num_volume, 3) * (bounds[1] - bounds[0]) + bounds[0]

            points = torch.cat([surface_samples[:num_surface], volume_samples], dim=0)
            sdf = mesh_utils.compute_sdf(mesh, points.numpy())

            self.object_caches.append({
                'points': points,
                'sdf': torch.from_numpy(sdf).float().view(-1, 1),
                'surface_normals': surface_normals,
            })

    def __len__(self) -> int:
        return self.num_objects

    def __getitem__(self, obj_idx: int) -> Dict[str, torch.Tensor]:
        cache = self.object_caches[obj_idx]

        indices = torch.randint(0, len(cache['points']), (self.num_samples,))

        return {
            'points': cache['points'][indices],
            'sdf': cache['sdf'][indices],
            'object_idx': torch.tensor(obj_idx, dtype=torch.long),
        }


class PointCloudDataset(Dataset):
    """
    Dataset from point cloud data (without explicit SDF).

    Suitable for training with occupancy or point-based losses.
    """

    def __init__(
        self,
        point_cloud: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        colors: Optional[torch.Tensor] = None,
        num_samples: int = 10000,
        bounds: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Args:
            point_cloud: Point cloud (N, 3)
            normals: Optional normals (N, 3)
            colors: Optional colors (N, 3)
            num_samples: Samples per batch
            bounds: Bounds for negative samples
        """
        self.points = point_cloud
        self.normals = normals
        self.colors = colors
        self.num_samples = num_samples
        self.bounds = bounds

    def __len__(self) -> int:
        return len(self.points) // self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        num_positive = self.num_samples // 2
        num_negative = self.num_samples - num_positive

        # Positive samples (on surface)
        pos_idx = torch.randint(0, len(self.points), (num_positive,))
        pos_points = self.points[pos_idx]

        # Negative samples (random in volume)
        neg_points = torch.rand(num_negative, 3)
        neg_points = neg_points * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        # Labels: 1 for surface, 0 for off-surface
        labels = torch.cat([
            torch.ones(num_positive),
            torch.zeros(num_negative)
        ])

        result = {
            'points': torch.cat([pos_points, neg_points], dim=0),
            'labels': labels,
        }

        if self.normals is not None:
            # Only for positive samples
            pos_normals = self.normals[pos_idx]
            neg_normals = torch.zeros(num_negative, 3)
            result['normals'] = torch.cat([pos_normals, neg_normals], dim=0)

        if self.colors is not None:
            pos_colors = self.colors[pos_idx]
            neg_colors = torch.zeros(num_negative, 3)
            result['colors'] = torch.cat([pos_colors, neg_colors], dim=0)

        return result


class ImagePoseDataset(Dataset):
    """
    Dataset with images and camera poses for NeRF-style training.
    """

    def __init__(
        self,
        images: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        num_rays: int = 1024,
        near: float = 0.0,
        far: float = 5.0
    ):
        """
        Args:
            images: Image tensor (N, H, W, 3)
            poses: Camera poses (N, 4, 4) or (N, 7) [trans, quat]
            intrinsics: Camera intrinsics (3, 3) or (N, 3, 3)
            num_rays: Number of rays per batch
            near: Near plane
            far: Far plane
        """
        self.images = images
        self.poses = poses
        self.intrinsics = intrinsics
        self.num_rays = num_rays
        self.near = near
        self.far = far

        self.num_images, self.H, self.W, _ = images.shape

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx]
        pose = self.poses[idx]

        # Get intrinsics
        if self.intrinsics.dim() == 2:
            K = self.intrinsics
        else:
            K = self.intrinsics[idx]

        # Sample random pixels
        ray_indices = torch.randint(0, self.H * self.W, (self.num_rays,))
        y = ray_indices // self.W
        x = ray_indices % self.W

        # Get ground truth colors
        gt_colors = image[y, x]

        # Generate rays
        from ..rendering import rays as ray_utils

        if pose.shape == (4, 4):
            # Matrix pose
            origins, directions = ray_utils.generate_rays_from_matrix(
                K, pose, self.H, self.W
            )
        else:
            # Translation + quaternion
            origins, directions = ray_utils.generate_rays_from_pose(
                K, pose[:3], pose[3:], self.H, self.W
            )

        # Select sampled rays
        origins = origins.view(-1, 3)[ray_indices]
        directions = directions.view(-1, 3)[ray_indices]

        return {
            'origins': origins,
            'directions': directions,
            'gt_colors': gt_colors,
            'near': torch.full((self.num_rays,), self.near),
            'far': torch.full((self.num_rays,), self.far),
            'image_idx': torch.tensor(idx),
        }


class PosedObjectDataset(Dataset):
    """
    Dataset with objects at various poses.

    Useful for training pose-equivariant models.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        num_poses: int = 10,
        rotation_range: Tuple[float, float] = (-3.14159, 3.14159),
        translation_range: Tuple[float, float] = (-0.5, 0.5)
    ):
        """
        Args:
            base_dataset: Underlying SDF dataset
            num_poses: Number of random poses per object
            rotation_range: Range for random rotations (radians)
            translation_range: Range for random translations
        """
        from ..utils import quaternion

        self.base_dataset = base_dataset
        self.num_poses = num_poses
        self.rotation_range = rotation_range
        self.translation_range = translation_range

        # Pre-generate poses
        self.poses = []
        for _ in range(num_poses):
            # Random translation
            t = torch.rand(3) * (translation_range[1] - translation_range[0]) + translation_range[0]

            # Random rotation (quaternion)
            q = quaternion.random_quaternion(1).squeeze(0)

            self.poses.append((t, q))

    def __len__(self) -> int:
        return len(self.base_dataset) * self.num_poses

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_idx = idx // self.num_poses
        pose_idx = idx % self.num_poses

        # Get base data
        data = self.base_dataset[base_idx]

        # Get pose
        translation, quaternion = self.poses[pose_idx]

        # Add pose to data
        data['translation'] = translation
        data['quaternion'] = quaternion

        return data


def collate_variable_points(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-sized point batches.

    Pads all point tensors to the same size.
    """
    # Find max number of points
    max_points = max(item['points'].shape[0] for item in batch)

    collated = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if key == 'points' or key == 'normals' or key == 'colors':
            # Pad to max_points
            padded = []
            for v in values:
                if len(v) < max_points:
                    padding = torch.zeros(max_points - len(v), v.shape[-1])
                    v = torch.cat([v, padding], dim=0)
                padded.append(v)
            collated[key] = torch.stack(padded)
        elif key == 'sdf' or key == 'labels':
            # Pad scalar values
            padded = []
            for v in values:
                if len(v) < max_points:
                    padding = torch.zeros(max_points - len(v), *v.shape[1:])
                    v = torch.cat([v, padding], dim=0)
                padded.append(v)
            collated[key] = torch.stack(padded)
        else:
            # Default stacking
            collated[key] = torch.stack(values)

    return collated


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with appropriate settings.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_points if kwargs.pop('variable_points', False) else None,
        **kwargs
    )

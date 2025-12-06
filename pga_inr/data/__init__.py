"""
Data loading and processing for PGA-INR.

Includes SDF dataset classes, point sampling strategies, mesh utilities,
and FBX loading for rigged characters.
"""

from .datasets import (
    SDFDataset,
    SDFDatasetFromMesh,
    MultiObjectSDFDataset,
    MultiObjectSDFDatasetFromMeshes,
    PointCloudDataset,
    ImagePoseDataset,
    PosedObjectDataset,
    create_dataloader,
)
from .sampling import (
    UniformSampler,
    SurfaceSampler,
    MixedSampler,
    ImportanceSampler,
    StratifiedSampler,
    NearSurfaceSampler,
    AdaptiveSampler,
    HierarchicalSampler,
    sample_on_sphere,
    sample_in_sphere,
    sample_rays,
)
from .mesh_utils import (
    load_mesh,
    save_mesh,
    normalize_mesh,
    compute_sdf,
    compute_sdf_grid,
    sample_surface_points,
    sample_near_surface,
    sdf_to_mesh,
    compute_mesh_metrics,
)
from .fbx_loader import (
    load_rigged_mesh,
    load_animation,
    load_rigged_character,
    MeshData,
    SkeletonData,
    SkinningData,
    AnimationData,
    FBXLoadError,
    FBXMeshError,
    FBXSkeletonError,
    FBXAnimationError,
)
from .rigged_dataset import (
    CanonicalMeshDataset,
    RiggedMeshDataset,
    AnimatedMeshDataset,
)

__all__ = [
    # Datasets
    "SDFDataset",
    "SDFDatasetFromMesh",
    "MultiObjectSDFDataset",
    "MultiObjectSDFDatasetFromMeshes",
    "PointCloudDataset",
    "ImagePoseDataset",
    "PosedObjectDataset",
    "create_dataloader",
    # Samplers
    "UniformSampler",
    "SurfaceSampler",
    "MixedSampler",
    "ImportanceSampler",
    "StratifiedSampler",
    "NearSurfaceSampler",
    "AdaptiveSampler",
    "HierarchicalSampler",
    "sample_on_sphere",
    "sample_in_sphere",
    "sample_rays",
    # Mesh utilities
    "load_mesh",
    "save_mesh",
    "normalize_mesh",
    "compute_sdf",
    "compute_sdf_grid",
    "sample_surface_points",
    "sample_near_surface",
    "sdf_to_mesh",
    "compute_mesh_metrics",
    # FBX loading
    "load_rigged_mesh",
    "load_animation",
    "load_rigged_character",
    "MeshData",
    "SkeletonData",
    "SkinningData",
    "AnimationData",
    "FBXLoadError",
    "FBXMeshError",
    "FBXSkeletonError",
    "FBXAnimationError",
    # Rigged datasets
    "CanonicalMeshDataset",
    "RiggedMeshDataset",
    "AnimatedMeshDataset",
]

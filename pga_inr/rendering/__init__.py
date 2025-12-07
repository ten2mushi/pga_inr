"""
Rendering module for PGA-INR.

Includes sphere tracing, ray generation, and shading models.
"""

from .rays import (
    generate_rays,
    generate_rays_from_pose,
    generate_rays_from_intrinsics,
    generate_rays_from_matrix,
    get_ray_bundle,
    transform_rays,
    rays_aabb_intersection,
    rays_sphere_intersection,
    rays_to_pga,
    pga_to_rays,
    sample_rays_uniform,
)
from .sphere_tracing import PGASphereTracer, DifferentiableSphereTracer, BatchSphereTracer
from .shading import (
    PGAShader,
    phong_shading,
    pbr_shading,
    normal_to_color,
    depth_to_color,
)
from .diff_mesh import (
    DifferentiableMarchingCubes,
    MeshSupervisedLoss,
    SurfaceSampler,
    MeshQualityLoss,
)

__all__ = [
    # Ray generation
    "generate_rays",
    "generate_rays_from_pose",
    "generate_rays_from_intrinsics",
    "generate_rays_from_matrix",
    "get_ray_bundle",
    "transform_rays",
    "rays_aabb_intersection",
    "rays_sphere_intersection",
    "rays_to_pga",
    "pga_to_rays",
    "sample_rays_uniform",
    # Sphere tracing
    "PGASphereTracer",
    "DifferentiableSphereTracer",
    "BatchSphereTracer",
    # Shading
    "PGAShader",
    "phong_shading",
    "pbr_shading",
    "normal_to_color",
    "depth_to_color",
    # Differentiable mesh
    "DifferentiableMarchingCubes",
    "MeshSupervisedLoss",
    "SurfaceSampler",
    "MeshQualityLoss",
]

"""
Shading models for PGA-INR rendering.

Provides various shading methods:
- Phong shading
- PBR (Physically Based Rendering)
- PGA-based geometric shading
- Normal-based visualization
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def phong_shading(
    normals: torch.Tensor,
    albedo: torch.Tensor,
    light_dir: torch.Tensor,
    view_dir: torch.Tensor,
    ambient: float = 0.1,
    diffuse: float = 0.7,
    specular: float = 0.2,
    shininess: float = 32.0
) -> torch.Tensor:
    """
    Classic Phong shading model.

    Args:
        normals: Surface normals (..., 3)
        albedo: Surface color (..., 3)
        light_dir: Light direction (3,) or (..., 3), pointing toward light
        view_dir: View direction (3,) or (..., 3), pointing toward camera
        ambient: Ambient light intensity
        diffuse: Diffuse reflection coefficient
        specular: Specular reflection coefficient
        shininess: Specular shininess exponent

    Returns:
        Shaded colors (..., 3)
    """
    # Normalize inputs
    normals = F.normalize(normals, dim=-1)
    light_dir = F.normalize(light_dir, dim=-1)
    view_dir = F.normalize(view_dir, dim=-1)

    # Broadcast light/view if needed
    if light_dir.dim() == 1:
        light_dir = light_dir.expand_as(normals)
    if view_dir.dim() == 1:
        view_dir = view_dir.expand_as(normals)

    # Ambient component
    ambient_color = ambient * albedo

    # Diffuse component (Lambert)
    n_dot_l = (normals * light_dir).sum(dim=-1, keepdim=True).clamp(min=0)
    diffuse_color = diffuse * n_dot_l * albedo

    # Specular component (Blinn-Phong)
    halfway = F.normalize(light_dir + view_dir, dim=-1)
    n_dot_h = (normals * halfway).sum(dim=-1, keepdim=True).clamp(min=0)
    specular_color = specular * (n_dot_h ** shininess)

    # Combine
    color = ambient_color + diffuse_color + specular_color

    return color.clamp(0, 1)


def pbr_shading(
    normals: torch.Tensor,
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    light_dir: torch.Tensor,
    view_dir: torch.Tensor,
    light_color: torch.Tensor = None,
    ambient_intensity: float = 0.03
) -> torch.Tensor:
    """
    Physically Based Rendering (simplified Cook-Torrance).

    Args:
        normals: Surface normals (..., 3)
        albedo: Base color (..., 3)
        roughness: Surface roughness (..., 1)
        metallic: Metallic factor (..., 1)
        light_dir: Light direction
        view_dir: View direction
        light_color: Light color (3,), defaults to white
        ambient_intensity: Ambient light intensity

    Returns:
        Shaded colors (..., 3)
    """
    if light_color is None:
        light_color = torch.ones(3, device=normals.device)

    # Normalize
    normals = F.normalize(normals, dim=-1)
    light_dir = F.normalize(light_dir, dim=-1)
    view_dir = F.normalize(view_dir, dim=-1)

    if light_dir.dim() == 1:
        light_dir = light_dir.expand_as(normals)
    if view_dir.dim() == 1:
        view_dir = view_dir.expand_as(normals)

    halfway = F.normalize(light_dir + view_dir, dim=-1)

    # Dot products
    n_dot_l = (normals * light_dir).sum(dim=-1, keepdim=True).clamp(min=0.001)
    n_dot_v = (normals * view_dir).sum(dim=-1, keepdim=True).clamp(min=0.001)
    n_dot_h = (normals * halfway).sum(dim=-1, keepdim=True).clamp(min=0.001)
    v_dot_h = (view_dir * halfway).sum(dim=-1, keepdim=True).clamp(min=0.001)

    # Fresnel (Schlick approximation)
    f0 = 0.04 * (1 - metallic) + albedo * metallic
    fresnel = f0 + (1 - f0) * (1 - v_dot_h) ** 5

    # Distribution (GGX)
    alpha = roughness ** 2
    alpha_sq = alpha ** 2
    denom = n_dot_h ** 2 * (alpha_sq - 1) + 1
    distribution = alpha_sq / (torch.pi * denom ** 2 + 1e-8)

    # Geometry (Smith GGX)
    k = (roughness + 1) ** 2 / 8
    g1_l = n_dot_l / (n_dot_l * (1 - k) + k)
    g1_v = n_dot_v / (n_dot_v * (1 - k) + k)
    geometry = g1_l * g1_v

    # Specular BRDF
    specular = (fresnel * distribution * geometry) / (4 * n_dot_l * n_dot_v + 1e-8)

    # Diffuse BRDF (Lambert for non-metals)
    kd = (1 - fresnel) * (1 - metallic)
    diffuse = kd * albedo / torch.pi

    # Combine with light
    lo = (diffuse + specular) * light_color * n_dot_l

    # Ambient
    ambient = ambient_intensity * albedo

    color = lo + ambient

    # Tone mapping (Reinhard)
    color = color / (color + 1)

    return color.clamp(0, 1)


def normal_to_color(normals: torch.Tensor) -> torch.Tensor:
    """
    Convert normals to RGB visualization.

    Maps normal components from [-1, 1] to [0, 1].

    Args:
        normals: Surface normals (..., 3)

    Returns:
        RGB colors (..., 3)
    """
    return (F.normalize(normals, dim=-1) + 1) / 2


def depth_to_color(
    depth: torch.Tensor,
    near: float = 0.0,
    far: float = 5.0,
    colormap: str = 'viridis'
) -> torch.Tensor:
    """
    Convert depth values to RGB visualization.

    Args:
        depth: Depth values (..., 1) or (...)
        near: Near plane for normalization
        far: Far plane for normalization
        colormap: Color map name ('viridis', 'plasma', 'magma', 'turbo')

    Returns:
        RGB colors (..., 3)
    """
    if depth.dim() > 0 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    # Normalize to [0, 1]
    depth_norm = (depth - near) / (far - near + 1e-8)
    depth_norm = depth_norm.clamp(0, 1)

    # Apply colormap
    if colormap == 'viridis':
        # Simplified viridis
        r = 0.267004 + 0.993248 * depth_norm - 0.757398 * depth_norm ** 2
        g = 0.004874 + 1.177485 * depth_norm - 0.617661 * depth_norm ** 2
        b = 0.329415 + 0.340155 * depth_norm + 0.302596 * depth_norm ** 2
    elif colormap == 'plasma':
        r = 0.050383 + 0.843252 * depth_norm + 0.106365 * depth_norm ** 2
        g = 0.029803 + 0.790150 * depth_norm + 0.179947 * depth_norm ** 2
        b = 0.527975 - 0.375342 * depth_norm + 0.847367 * depth_norm ** 2
    elif colormap == 'turbo':
        r = 0.13572 + 4.61539 * depth_norm - 42.6603 * depth_norm ** 2 + 132.6059 * depth_norm ** 3 - 152.9489 * depth_norm ** 4 + 59.2872 * depth_norm ** 5
        g = 0.09140 + 2.26410 * depth_norm + 7.64320 * depth_norm ** 2 - 33.4024 * depth_norm ** 3 + 39.3779 * depth_norm ** 4 - 15.7197 * depth_norm ** 5
        b = 0.10667 + 12.7577 * depth_norm - 60.5827 * depth_norm ** 2 + 109.8371 * depth_norm ** 3 - 86.6428 * depth_norm ** 4 + 24.5415 * depth_norm ** 5
    else:  # grayscale
        r = g = b = depth_norm

    return torch.stack([r, g, b], dim=-1).clamp(0, 1)


def matcap_shading(
    normals: torch.Tensor,
    view_dir: torch.Tensor,
    matcap: torch.Tensor
) -> torch.Tensor:
    """
    MatCap (Material Capture) shading.

    Uses a 2D texture to define the material appearance.

    Args:
        normals: Surface normals (..., 3)
        view_dir: View direction (3,)
        matcap: MatCap texture (H, W, 3)

    Returns:
        Shaded colors (..., 3)
    """
    # Transform normals to view space
    # Simplified: just use normal x, y components
    normals = F.normalize(normals, dim=-1)

    # Map to texture coordinates
    u = (normals[..., 0] + 1) / 2
    v = (normals[..., 1] + 1) / 2

    # Sample matcap texture
    H, W = matcap.shape[:2]
    u_idx = (u * (W - 1)).long().clamp(0, W - 1)
    v_idx = (v * (H - 1)).long().clamp(0, H - 1)

    # Gather colors
    shape = normals.shape[:-1]
    u_idx = u_idx.view(-1)
    v_idx = v_idx.view(-1)
    colors = matcap[v_idx, u_idx]
    colors = colors.view(*shape, 3)

    return colors


def ambient_occlusion(
    points: torch.Tensor,
    normals: torch.Tensor,
    sdf_fn,
    num_samples: int = 8,
    radius: float = 0.1
) -> torch.Tensor:
    """
    Compute ambient occlusion from SDF.

    Args:
        points: Surface points (..., 3)
        normals: Surface normals (..., 3)
        sdf_fn: SDF function
        num_samples: Number of AO samples
        radius: Maximum AO radius

    Returns:
        AO values (..., 1) in [0, 1]
    """
    shape = points.shape[:-1]
    device = points.device

    # Sample points along normal direction
    t_vals = torch.linspace(0, 1, num_samples, device=device) ** 2 * radius

    ao = torch.zeros(*shape, device=device)

    for t in t_vals:
        sample_points = points + t * normals
        sample_points_flat = sample_points.view(-1, 3)

        # Query SDF
        with torch.no_grad():
            sdf = sdf_fn(sample_points_flat)
            if sdf.dim() == 2:
                sdf = sdf.squeeze(-1)
            sdf = sdf.view(*shape)

        # Accumulate occlusion (closer points contribute more)
        weight = 1 - t / radius
        ao = ao + weight * (sdf < 0).float()

    ao = ao / num_samples
    ao = 1 - ao

    return ao.unsqueeze(-1)


def subsurface_scattering(
    points: torch.Tensor,
    normals: torch.Tensor,
    light_dir: torch.Tensor,
    sdf_fn,
    albedo: torch.Tensor,
    scatter_distance: float = 0.1,
    num_samples: int = 4
) -> torch.Tensor:
    """
    Simple subsurface scattering approximation.

    Args:
        points: Surface points (..., 3)
        normals: Surface normals (..., 3)
        light_dir: Light direction (3,)
        sdf_fn: SDF function
        albedo: Base color (..., 3)
        scatter_distance: Scattering distance
        num_samples: Number of samples

    Returns:
        SSS contribution (..., 3)
    """
    device = points.device
    light_dir = F.normalize(light_dir, dim=-1)

    if light_dir.dim() == 1:
        light_dir = light_dir.expand_as(normals)

    # Sample points behind surface in light direction
    sss = torch.zeros_like(albedo)

    for i in range(num_samples):
        t = (i + 1) / num_samples * scatter_distance
        sample_points = points - t * light_dir

        # Query SDF
        with torch.no_grad():
            sdf = sdf_fn(sample_points.view(-1, 3))
            if sdf.dim() == 2:
                sdf = sdf.squeeze(-1)
            sdf = sdf.view(*points.shape[:-1])

        # Points inside contribute to SSS
        inside = (sdf < 0).float()
        weight = torch.exp(-t / scatter_distance)
        sss = sss + weight * inside.unsqueeze(-1) * albedo

    sss = sss / num_samples

    return sss


def rim_lighting(
    normals: torch.Tensor,
    view_dir: torch.Tensor,
    rim_color: torch.Tensor = None,
    rim_power: float = 3.0,
    rim_intensity: float = 0.5
) -> torch.Tensor:
    """
    Rim/fresnel lighting effect.

    Args:
        normals: Surface normals (..., 3)
        view_dir: View direction (3,)
        rim_color: Rim light color (3,), defaults to white
        rim_power: Rim falloff power
        rim_intensity: Rim intensity

    Returns:
        Rim lighting contribution (..., 3)
    """
    if rim_color is None:
        rim_color = torch.ones(3, device=normals.device)

    normals = F.normalize(normals, dim=-1)
    view_dir = F.normalize(view_dir, dim=-1)

    if view_dir.dim() == 1:
        view_dir = view_dir.expand_as(normals)

    # Fresnel term
    n_dot_v = (normals * view_dir).sum(dim=-1, keepdim=True).clamp(min=0)
    rim = (1 - n_dot_v) ** rim_power * rim_intensity

    return rim * rim_color


class PGAShader:
    """
    PGA-based shading using geometric products.

    Uses PGA operations for lighting calculations.
    """

    def __init__(
        self,
        ambient: float = 0.1,
        diffuse: float = 0.7,
        specular: float = 0.2,
        shininess: float = 32.0
    ):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess

    def shade(
        self,
        normals: torch.Tensor,
        albedo: torch.Tensor,
        light_dir: torch.Tensor,
        view_dir: torch.Tensor
    ) -> torch.Tensor:
        """
        Shade using PGA operations.

        The dot product in PGA is computed via the scalar part of
        the geometric product.

        Args:
            normals: Surface normals (..., 3)
            albedo: Surface colors (..., 3)
            light_dir: Light direction (3,)
            view_dir: View direction (3,)

        Returns:
            Shaded colors (..., 3)
        """
        # For now, fall back to standard Phong
        # Full PGA implementation would use multivector operations
        return phong_shading(
            normals, albedo, light_dir, view_dir,
            self.ambient, self.diffuse, self.specular, self.shininess
        )


def combine_shading(
    base_color: torch.Tensor,
    ao: Optional[torch.Tensor] = None,
    rim: Optional[torch.Tensor] = None,
    sss: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Combine multiple shading contributions.

    Args:
        base_color: Base shaded color (..., 3)
        ao: Ambient occlusion (..., 1)
        rim: Rim lighting (..., 3)
        sss: Subsurface scattering (..., 3)

    Returns:
        Combined color (..., 3)
    """
    color = base_color

    if ao is not None:
        color = color * ao

    if rim is not None:
        color = color + rim

    if sss is not None:
        color = color + sss

    return color.clamp(0, 1)

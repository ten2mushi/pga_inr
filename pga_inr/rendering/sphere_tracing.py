"""
Sphere tracing renderer for PGA-INR models.

Provides efficient SDF-based rendering via ray marching with
adaptive step sizes based on SDF values.
"""

from typing import Dict, Tuple, Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGASphereTracer:
    """
    Sphere tracing renderer for PGA-INR.

    Uses adaptive step sizes based on SDF values for efficient
    surface intersection finding.
    """

    def __init__(
        self,
        model: nn.Module,
        width: int = 256,
        height: int = 256,
        fov: float = 60.0,
        max_steps: int = 128,
        epsilon: float = 1e-4,
        near: float = 0.1,
        far: float = 5.0,
        step_scale: float = 1.0
    ):
        """
        Args:
            model: PGA-INR model with SDF output
            width: Image width
            height: Image height
            fov: Field of view in degrees
            max_steps: Maximum tracing steps
            epsilon: Surface intersection threshold
            near: Near plane distance
            far: Far plane distance
            step_scale: Scale factor for step sizes (< 1 for safety)
        """
        self.model = model
        self.width = width
        self.height = height
        self.fov = fov
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.near = near
        self.far = far
        self.step_scale = step_scale

        # Compute focal length from FOV
        self.focal = self.height / (2 * torch.tan(torch.tensor(fov * torch.pi / 360)))

    def _pose_to_matrix(
        self,
        camera_pose,
        device: torch.device
    ) -> torch.Tensor:
        """
        Convert camera pose to 4x4 matrix.

        Args:
            camera_pose: Either a 4x4 matrix or (translation, quaternion) tuple
            device: Device for tensors

        Returns:
            4x4 camera-to-world transform matrix
        """
        if isinstance(camera_pose, tuple):
            # (translation, quaternion) format
            from ..utils.quaternion import quaternion_to_matrix
            translation, quaternion = camera_pose
            rotation = quaternion_to_matrix(quaternion.unsqueeze(0)).squeeze(0)
            matrix = torch.eye(4, device=device)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation
            return matrix
        else:
            # Already a 4x4 matrix
            return camera_pose.to(device)

    def generate_rays(
        self,
        camera_pose,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for all pixels.

        Args:
            camera_pose: Camera-to-world transform, either (4, 4) matrix or
                        (translation, quaternion) tuple
            device: Device for tensors

        Returns:
            (origins, directions) each of shape (H*W, 3)
        """
        from .rays import generate_rays

        # Convert pose to matrix if needed
        pose_matrix = self._pose_to_matrix(camera_pose, device)

        origins, directions = generate_rays(
            self.height, self.width,
            self.focal.item() if isinstance(self.focal, torch.Tensor) else self.focal,
            pose_matrix,
            device=device
        )

        return origins.view(-1, 3), directions.view(-1, 3)

    @torch.no_grad()
    def trace(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        object_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        latent_code: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform sphere tracing with ray masking for efficiency.

        Only evaluates the SDF for active (non-converged) rays, providing
        significant speedup when many rays converge early or miss the object.

        Args:
            origins: Ray origins (N, 3)
            directions: Ray directions (N, 3), normalized
            object_pose: Optional object pose (translation, quaternion)
            latent_code: Optional latent code for generative models

        Returns:
            (points, mask, depths) where:
                points: Intersection points (N, 3)
                mask: Valid intersection mask (N,)
                depths: Ray depths at intersection (N,)
        """
        device = origins.device
        N = origins.shape[0]

        # Initialize
        depths = torch.full((N,), self.near, device=device)
        active = torch.ones(N, dtype=torch.bool, device=device)  # Rays still being traced
        converged = torch.zeros(N, dtype=torch.bool, device=device)  # Rays that hit surface

        for step in range(self.max_steps):
            # Early exit when all rays have terminated
            if not active.any():
                break

            # Get indices of active rays
            active_indices = torch.where(active)[0]

            # Current points for active rays only
            active_origins = origins[active_indices]
            active_directions = directions[active_indices]
            active_depths = depths[active_indices]
            points_active = active_origins + active_depths.unsqueeze(-1) * active_directions

            # Query SDF only for active rays
            if latent_code is not None:
                outputs = self.model(points_active.unsqueeze(0), object_pose, latent_code)
            else:
                outputs = self.model(points_active.unsqueeze(0), object_pose)

            sdf = outputs.get('sdf', outputs.get('density')).squeeze(0).squeeze(-1)

            # Update depths for active rays
            # Use absolute value to ensure we always step forward along the ray
            new_depths = active_depths + self.step_scale * torch.abs(sdf)
            depths[active_indices] = new_depths

            # Check for intersection (SDF < epsilon) among active rays
            hit = sdf.abs() < self.epsilon
            # Check for out of bounds
            out_of_bounds = (new_depths >= self.far) | (new_depths <= self.near)

            # Update converged status - rays that hit the surface
            converged[active_indices[hit]] = True

            # Deactivate rays that hit or went out of bounds
            active[active_indices] = ~(hit | out_of_bounds)

        # Final points
        points = origins + depths.unsqueeze(-1) * directions

        # Valid intersections: converged and within bounds
        valid = converged & (depths < self.far) & (depths > self.near)

        return points, valid, depths

    @torch.no_grad()
    def render(
        self,
        camera_pose: torch.Tensor,
        object_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        latent_code: Optional[torch.Tensor] = None,
        return_depth: bool = False,
        return_normals: bool = False,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, torch.Tensor]:
        """
        Render image via sphere tracing.

        Args:
            camera_pose: Camera-to-world transform (4, 4)
            object_pose: Optional object pose (translation, quaternion)
            latent_code: Optional latent code
            return_depth: Whether to return depth map
            return_normals: Whether to return normal map
            background_color: Background RGB color

        Returns:
            Dictionary with 'rgb' and optionally 'depth', 'normals'
        """
        device = next(self.model.parameters()).device

        # Generate rays
        origins, directions = self.generate_rays(camera_pose, device)

        # Sphere trace
        points, mask, depths = self.trace(
            origins, directions, object_pose, latent_code
        )

        # Query model at intersection points for color
        valid_points = points[mask]

        if len(valid_points) > 0:
            if latent_code is not None:
                outputs = self.model(valid_points.unsqueeze(0), object_pose, latent_code)
            else:
                outputs = self.model(valid_points.unsqueeze(0), object_pose)

            # Handle both full PGA-INR (with rgb) and SDF-only models
            if 'rgb' in outputs:
                valid_colors = outputs['rgb'].squeeze(0)
            else:
                # Default gray color for SDF models without color
                valid_colors = torch.full((valid_points.shape[0], 3), 0.7, device=device)

            valid_normals = outputs.get('normal', None)
            if valid_normals is not None:
                valid_normals = valid_normals.squeeze(0)
        else:
            valid_colors = torch.empty(0, 3, device=device)
            valid_normals = torch.empty(0, 3, device=device) if return_normals else None

        # Initialize output images
        bg = torch.tensor(background_color, device=device)
        rgb = bg.unsqueeze(0).expand(origins.shape[0], 3).clone()
        rgb[mask] = valid_colors

        result = {
            'rgb': rgb.view(self.height, self.width, 3),
            'hit_mask': mask.view(self.height, self.width)
        }

        if return_depth:
            depth = torch.zeros(origins.shape[0], device=device)
            depth[mask] = depths[mask]
            result['depth'] = depth.view(self.height, self.width)

        if return_normals and valid_normals is not None:
            normals = torch.zeros(origins.shape[0], 3, device=device)
            normals[mask] = valid_normals
            result['normals'] = normals.view(self.height, self.width, 3)

        return result

    def render_with_shading(
        self,
        camera_pose: torch.Tensor,
        object_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        latent_code: Optional[torch.Tensor] = None,
        light_dir: Optional[torch.Tensor] = None,
        ambient: float = 0.1,
        diffuse: float = 0.7,
        specular: float = 0.2,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, torch.Tensor]:
        """
        Render with Phong shading.

        Args:
            camera_pose: Camera pose
            object_pose: Object pose
            latent_code: Latent code
            light_dir: Light direction (3,), defaults to camera direction
            ambient: Ambient light intensity
            diffuse: Diffuse intensity
            specular: Specular intensity
            background_color: Background color

        Returns:
            Dictionary with shaded 'rgb'
        """
        from .shading import phong_shading

        # Render with normals
        result = self.render(
            camera_pose, object_pose, latent_code,
            return_depth=True, return_normals=True,
            background_color=background_color
        )

        # Handle camera_pose format for light/view direction extraction
        if isinstance(camera_pose, tuple):
            from ..utils.quaternion import quaternion_to_matrix
            _, quaternion = camera_pose
            rotation = quaternion_to_matrix(quaternion.unsqueeze(0)).squeeze(0)
            cam_forward = rotation[:, 2]  # Third column is forward direction
        else:
            cam_forward = camera_pose[:3, 2]

        if light_dir is None:
            # Light from camera
            light_dir = -cam_forward

        # View direction
        view_dir = -cam_forward

        # Apply shading
        mask = result['hit_mask']
        shaded = phong_shading(
            result['normals'],
            result['rgb'],
            light_dir,
            view_dir,
            ambient, diffuse, specular
        )

        # Apply background
        bg = torch.tensor(background_color, device=result['rgb'].device)
        result['rgb'] = torch.where(
            mask.unsqueeze(-1),
            shaded,
            bg.view(1, 1, 3)
        )

        return result


class DifferentiableSphereTracer(PGASphereTracer):
    """
    Differentiable sphere tracer for end-to-end training.

    Uses implicit differentiation at surface intersection.
    """

    def trace_differentiable(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        object_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        latent_code: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Differentiable sphere tracing.

        Uses gradient of SDF at intersection for implicit differentiation.
        """
        # First, trace without gradients to find intersection
        with torch.no_grad():
            points, mask, depths = self.trace(
                origins, directions, object_pose, latent_code
            )

        # Now compute with gradients at intersection points
        depths = depths.detach().requires_grad_(True)
        points = origins + depths.unsqueeze(-1) * directions

        return points, mask, depths

    def render(
        self,
        camera_pose: torch.Tensor,
        object_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        latent_code: Optional[torch.Tensor] = None,
        return_depth: bool = False,
        return_normals: bool = False,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, torch.Tensor]:
        """
        Differentiable render.
        """
        device = next(self.model.parameters()).device

        # Generate rays
        origins, directions = self.generate_rays(camera_pose, device)

        # Differentiable trace
        points, mask, depths = self.trace_differentiable(
            origins, directions, object_pose, latent_code
        )

        # Query model (with gradients)
        valid_points = points[mask]

        if len(valid_points) > 0:
            if latent_code is not None:
                outputs = self.model(valid_points.unsqueeze(0), object_pose, latent_code)
            else:
                outputs = self.model(valid_points.unsqueeze(0), object_pose)

            # Handle both full PGA-INR (with rgb) and SDF-only models
            if 'rgb' in outputs:
                valid_colors = outputs['rgb'].squeeze(0)
            else:
                # Default gray color for SDF models without color
                valid_colors = torch.full((valid_points.shape[0], 3), 0.7, device=device)

            valid_normals = outputs.get('normal', None)
            if valid_normals is not None:
                valid_normals = valid_normals.squeeze(0)
        else:
            valid_colors = torch.empty(0, 3, device=device, requires_grad=True)
            valid_normals = None

        # Build output
        bg = torch.tensor(background_color, device=device)
        rgb = bg.unsqueeze(0).expand(origins.shape[0], 3).clone()
        rgb[mask] = valid_colors

        result = {
            'rgb': rgb.view(self.height, self.width, 3),
            'hit_mask': mask.view(self.height, self.width)
        }

        if return_depth:
            depth = torch.zeros(origins.shape[0], device=device)
            depth[mask] = depths[mask]
            result['depth'] = depth.view(self.height, self.width)

        if return_normals and valid_normals is not None:
            normals = torch.zeros(origins.shape[0], 3, device=device)
            normals[mask] = valid_normals
            result['normals'] = normals.view(self.height, self.width, 3)

        return result


def volume_render(
    density: torch.Tensor,
    rgb: torch.Tensor,
    t_vals: torch.Tensor,
    white_background: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Volume rendering for NeRF-style models.

    Args:
        density: Density/sigma values (N, num_samples)
        rgb: RGB values (N, num_samples, 3)
        t_vals: Sample t values along ray (N, num_samples)
        white_background: Whether to use white background

    Returns:
        (colors, weights) where colors is (N, 3) and weights is (N, num_samples)
    """
    # Compute deltas
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

    # Convert density to alpha
    alpha = 1 - torch.exp(-density * dists)

    # Compute transmittance
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-10], dim=-1),
        dim=-1
    )[..., :-1]

    # Compute weights
    weights = alpha * transmittance

    # Compute final color
    colors = (weights.unsqueeze(-1) * rgb).sum(dim=-2)

    # Add background
    if white_background:
        acc = weights.sum(dim=-1, keepdim=True)
        colors = colors + (1 - acc)

    return colors, weights


def marching_cubes_render(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    resolution: int = 256,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    level: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract mesh from SDF using marching cubes.

    Args:
        sdf_fn: SDF function
        resolution: Grid resolution
        bounds: Coordinate bounds
        level: Isosurface level

    Returns:
        (vertices, faces)
    """
    from ..data.mesh_utils import sdf_to_mesh

    mesh = sdf_to_mesh(sdf_fn, resolution, bounds, level)
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    return vertices, faces


class BatchSphereTracer(PGASphereTracer):
    """
    Batched sphere tracer for multiple objects/views.
    """

    @torch.no_grad()
    def render_batch(
        self,
        camera_poses,
        object_poses: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        latent_codes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Render multiple views.

        Args:
            camera_poses: Camera poses - either (B, 4, 4) tensor or tuple of
                         (translations (B, 3), quaternions (B, 4))
            object_poses: Optional list of object poses
            latent_codes: Optional latent codes (B, latent_dim)

        Returns:
            Dictionary with batched outputs
        """
        device = next(self.model.parameters()).device

        # Handle tuple format (translations, quaternions)
        if isinstance(camera_poses, tuple):
            translations, quaternions = camera_poses
            batch_size = translations.shape[0]
        else:
            batch_size = camera_poses.shape[0]

        all_rgb = []
        all_depth = []
        all_hit_mask = []

        for i in range(batch_size):
            object_pose = object_poses[i] if object_poses is not None else None
            latent_code = latent_codes[i:i+1] if latent_codes is not None else None

            # Extract single camera pose
            if isinstance(camera_poses, tuple):
                camera_pose = (translations[i], quaternions[i])
            else:
                camera_pose = camera_poses[i]

            result = self.render(
                camera_pose,
                object_pose,
                latent_code,
                return_depth=True
            )

            all_rgb.append(result['rgb'])
            all_depth.append(result['depth'])
            all_hit_mask.append(result['hit_mask'])

        return {
            'rgb': torch.stack(all_rgb),
            'depth': torch.stack(all_depth),
            'hit_mask': torch.stack(all_hit_mask)
        }


def render_turntable(
    model: nn.Module,
    num_frames: int = 36,
    radius: float = 2.0,
    height: float = 0.5,
    image_size: int = 256,
    fov: float = 60.0,
    object_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    latent_code: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Render a turntable animation.

    Args:
        model: PGA-INR model
        num_frames: Number of frames
        radius: Camera orbit radius
        height: Camera height
        image_size: Image size
        fov: Field of view
        object_pose: Object pose
        latent_code: Latent code

    Returns:
        Frames tensor (num_frames, H, W, 3)
    """
    device = next(model.parameters()).device

    tracer = PGASphereTracer(
        model,
        width=image_size,
        height=image_size,
        fov=fov
    )

    frames = []

    for i in range(num_frames):
        # Camera position on orbit
        angle = 2 * torch.pi * i / num_frames
        cam_x = radius * torch.cos(torch.tensor(angle))
        cam_z = radius * torch.sin(torch.tensor(angle))
        cam_y = height

        # Look at origin
        cam_pos = torch.tensor([cam_x, cam_y, cam_z], device=device)
        forward = F.normalize(-cam_pos, dim=0)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = F.normalize(torch.cross(forward, up), dim=0)
        up = torch.cross(right, forward)

        # Camera pose
        pose = torch.eye(4, device=device)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos

        # Render
        result = tracer.render(
            pose, object_pose, latent_code,
            return_normals=True
        )

        frames.append(result['rgb'])

    return torch.stack(frames)

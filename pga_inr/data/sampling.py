"""
Point sampling strategies for SDF training.

Provides various sampling methods for generating training points:
- Uniform sampling in a bounding box
- Surface-biased sampling (near the mesh surface)
- Importance sampling based on SDF gradient
- Stratified sampling for better coverage
- Curriculum importance sampling (progressive surface focus)
- Newton projection sampling (exact surface points)
- Error-driven sampling (adaptive to training loss)
"""

from typing import Tuple, Optional, Callable, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformSampler:
    """
    Uniform random sampling within a bounding box.

    Simple baseline sampling strategy.
    """

    def __init__(
        self,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            bounds: (min, max) coordinate bounds
            device: Device for generated tensors
        """
        self.bounds = bounds
        self.device = device

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Generate uniform random samples.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches

        Returns:
            Points of shape (batch_size, num_samples, 3)
        """
        points = torch.rand(
            batch_size, num_samples, 3,
            device=self.device
        )
        # Scale to bounds
        points = points * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return points


class SurfaceSampler:
    """
    Sample points on or near a mesh surface.

    Uses precomputed surface points and optionally perturbs them.
    """

    def __init__(
        self,
        surface_points: torch.Tensor,
        surface_normals: Optional[torch.Tensor] = None,
        noise_std: float = 0.01,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            surface_points: Precomputed surface points (N, 3)
            surface_normals: Optional surface normals (N, 3)
            noise_std: Standard deviation of offset noise
            device: Device for generated tensors
        """
        self.surface_points = surface_points.to(device)
        self.surface_normals = surface_normals.to(device) if surface_normals is not None else None
        self.noise_std = noise_std
        self.device = device

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample points near the surface.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches
            add_noise: Whether to add offset noise

        Returns:
            (points, normals) where:
                points: (batch_size, num_samples, 3)
                normals: (batch_size, num_samples, 3) or None
        """
        # Random indices
        indices = torch.randint(
            0, len(self.surface_points),
            (batch_size, num_samples),
            device=self.device
        )

        # Gather surface points
        points = self.surface_points[indices]

        # Optional normals
        normals = None
        if self.surface_normals is not None:
            normals = self.surface_normals[indices]

        # Add noise
        if add_noise and self.noise_std > 0:
            if normals is not None:
                # Offset along normal direction
                offset = torch.randn(batch_size, num_samples, 1, device=self.device)
                points = points + offset * normals * self.noise_std
            else:
                # Random offset
                noise = torch.randn_like(points) * self.noise_std
                points = points + noise

        return points, normals


class MixedSampler:
    """
    Mix of uniform and surface sampling.

    Combines uniform samples (for global coverage) with surface samples
    (for detail near the surface).
    """

    def __init__(
        self,
        surface_points: torch.Tensor,
        surface_normals: Optional[torch.Tensor] = None,
        surface_ratio: float = 0.5,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        surface_noise_std: float = 0.01,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            surface_points: Precomputed surface points (N, 3)
            surface_normals: Optional surface normals (N, 3)
            surface_ratio: Fraction of samples near surface
            bounds: Bounds for uniform sampling
            surface_noise_std: Noise for surface samples
            device: Device for generated tensors
        """
        self.uniform_sampler = UniformSampler(bounds, device)
        self.surface_sampler = SurfaceSampler(
            surface_points, surface_normals,
            surface_noise_std, device
        )
        self.surface_ratio = surface_ratio
        self.device = device

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample mixed points.

        Args:
            num_samples: Total number of points per batch
            batch_size: Number of batches

        Returns:
            (all_points, is_surface, normals) where:
                all_points: (batch_size, num_samples, 3)
                is_surface: (batch_size, num_samples) boolean mask
                normals: (batch_size, num_surface, 3) or None
        """
        num_surface = int(num_samples * self.surface_ratio)
        num_uniform = num_samples - num_surface

        # Surface samples
        surface_pts, normals = self.surface_sampler.sample(num_surface, batch_size)

        # Uniform samples
        uniform_pts = self.uniform_sampler.sample(num_uniform, batch_size)

        # Combine
        all_points = torch.cat([surface_pts, uniform_pts], dim=1)

        # Create mask
        is_surface = torch.zeros(batch_size, num_samples, dtype=torch.bool, device=self.device)
        is_surface[:, :num_surface] = True

        return all_points, is_surface, normals


class ImportanceSampler:
    """
    Importance sampling based on a criterion function.

    Generates more samples in regions where the criterion is high.
    """

    def __init__(
        self,
        criterion_fn: Callable[[torch.Tensor], torch.Tensor],
        bounds: Tuple[float, float] = (-1.0, 1.0),
        num_candidates: int = 10000,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            criterion_fn: Function that takes points (N, 3) and returns
                         importance scores (N,)
            bounds: Coordinate bounds
            num_candidates: Number of candidates to evaluate
            device: Device for generated tensors
        """
        self.criterion_fn = criterion_fn
        self.bounds = bounds
        self.num_candidates = num_candidates
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Generate importance-weighted samples.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches

        Returns:
            Points of shape (batch_size, num_samples, 3)
        """
        all_samples = []

        for _ in range(batch_size):
            # Generate candidates
            candidates = torch.rand(
                self.num_candidates, 3,
                device=self.device
            )
            candidates = candidates * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

            # Evaluate importance
            importance = self.criterion_fn(candidates)
            importance = importance.clamp(min=1e-8)

            # Sample based on importance
            probs = importance / importance.sum()
            indices = torch.multinomial(probs, num_samples, replacement=True)
            samples = candidates[indices]

            all_samples.append(samples)

        return torch.stack(all_samples, dim=0)


class StratifiedSampler:
    """
    Stratified sampling for better coverage.

    Divides space into a grid and samples uniformly within each cell.
    """

    def __init__(
        self,
        resolution: int = 8,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            resolution: Grid resolution per dimension
            bounds: Coordinate bounds
            device: Device for generated tensors
        """
        self.resolution = resolution
        self.bounds = bounds
        self.device = device

        # Cell size
        self.cell_size = (bounds[1] - bounds[0]) / resolution

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1,
        jitter: bool = True
    ) -> torch.Tensor:
        """
        Generate stratified samples.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches
            jitter: Whether to add jitter within cells

        Returns:
            Points of shape (batch_size, num_samples, 3)
        """
        # Number of samples per cell
        num_cells = self.resolution ** 3
        samples_per_cell = max(1, num_samples // num_cells)

        # Generate cell indices
        x_idx = torch.arange(self.resolution, device=self.device)
        y_idx = torch.arange(self.resolution, device=self.device)
        z_idx = torch.arange(self.resolution, device=self.device)

        # Grid of cell corners
        gx, gy, gz = torch.meshgrid(x_idx, y_idx, z_idx, indexing='ij')
        cell_corners = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).float()

        all_samples = []

        for _ in range(batch_size):
            batch_samples = []

            for corner in cell_corners:
                # Sample within cell
                if jitter:
                    offsets = torch.rand(samples_per_cell, 3, device=self.device)
                else:
                    offsets = torch.full((samples_per_cell, 3), 0.5, device=self.device)

                # Convert to world coordinates
                cell_samples = (corner + offsets) * self.cell_size + self.bounds[0]
                batch_samples.append(cell_samples)

            batch_samples = torch.cat(batch_samples, dim=0)

            # Randomly select if we have too many
            if len(batch_samples) > num_samples:
                indices = torch.randperm(len(batch_samples), device=self.device)[:num_samples]
                batch_samples = batch_samples[indices]
            # Pad if we have too few
            elif len(batch_samples) < num_samples:
                extra = num_samples - len(batch_samples)
                extra_samples = torch.rand(extra, 3, device=self.device)
                extra_samples = extra_samples * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
                batch_samples = torch.cat([batch_samples, extra_samples], dim=0)

            all_samples.append(batch_samples)

        return torch.stack(all_samples, dim=0)


class NearSurfaceSampler:
    """
    Sample points at varying distances from surface.

    Uses stratified distance bands for better SDF learning.
    """

    def __init__(
        self,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        distance_bands: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2),
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            surface_points: Surface points (N, 3)
            surface_normals: Surface normals (N, 3)
            distance_bands: Distance band boundaries
            device: Device for generated tensors
        """
        self.surface_points = surface_points.to(device)
        self.surface_normals = surface_normals.to(device)
        self.distance_bands = distance_bands
        self.device = device

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points at varying distances from surface.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches

        Returns:
            (points, signed_distances) where:
                points: (batch_size, num_samples, 3)
                signed_distances: (batch_size, num_samples, 1)
        """
        num_bands = len(self.distance_bands) - 1
        samples_per_band = num_samples // num_bands

        all_points = []
        all_distances = []

        for b in range(batch_size):
            batch_points = []
            batch_distances = []

            for i in range(num_bands):
                d_min, d_max = self.distance_bands[i], self.distance_bands[i + 1]

                # Random surface points
                indices = torch.randint(
                    0, len(self.surface_points),
                    (samples_per_band,),
                    device=self.device
                )
                pts = self.surface_points[indices]
                normals = self.surface_normals[indices]

                # Random distances in band (with sign)
                distances = torch.rand(samples_per_band, 1, device=self.device)
                distances = distances * (d_max - d_min) + d_min
                signs = torch.sign(torch.rand(samples_per_band, 1, device=self.device) - 0.5)
                signed_distances = distances * signs

                # Offset along normal
                points = pts + signed_distances * normals

                batch_points.append(points)
                batch_distances.append(signed_distances)

            batch_points = torch.cat(batch_points, dim=0)
            batch_distances = torch.cat(batch_distances, dim=0)

            # Handle remaining samples
            remaining = num_samples - len(batch_points)
            if remaining > 0:
                indices = torch.randint(
                    0, len(self.surface_points),
                    (remaining,),
                    device=self.device
                )
                extra_pts = self.surface_points[indices]
                extra_dists = torch.zeros(remaining, 1, device=self.device)
                batch_points = torch.cat([batch_points, extra_pts], dim=0)
                batch_distances = torch.cat([batch_distances, extra_dists], dim=0)

            all_points.append(batch_points)
            all_distances.append(batch_distances)

        return torch.stack(all_points), torch.stack(all_distances)


class AdaptiveSampler:
    """
    Adaptive sampling that focuses on high-error regions.

    Maintains a history of errors and samples more densely
    where the model is performing poorly.
    """

    def __init__(
        self,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        resolution: int = 16,
        temperature: float = 1.0,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            bounds: Coordinate bounds
            resolution: Grid resolution for error tracking
            temperature: Sampling temperature (higher = more uniform)
            device: Device for generated tensors
        """
        self.bounds = bounds
        self.resolution = resolution
        self.temperature = temperature
        self.device = device

        # Error grid (uniform initialization)
        self.error_grid = torch.ones(
            resolution, resolution, resolution,
            device=device
        )
        self.cell_size = (bounds[1] - bounds[0]) / resolution

    def update_errors(
        self,
        points: torch.Tensor,
        errors: torch.Tensor
    ):
        """
        Update error grid based on observed errors.

        Args:
            points: Sample points (N, 3)
            errors: Error values (N,)
        """
        # Convert to grid indices
        normalized = (points - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        indices = (normalized * self.resolution).long().clamp(0, self.resolution - 1)

        # Update grid with exponential moving average
        alpha = 0.1
        for i, (idx, err) in enumerate(zip(indices, errors)):
            self.error_grid[idx[0], idx[1], idx[2]] = (
                (1 - alpha) * self.error_grid[idx[0], idx[1], idx[2]] +
                alpha * err.item()
            )

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Sample based on error distribution.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches

        Returns:
            Points of shape (batch_size, num_samples, 3)
        """
        # Compute sampling probabilities
        probs = (self.error_grid / self.temperature).flatten().softmax(dim=0)

        all_samples = []

        for _ in range(batch_size):
            # Sample cell indices
            cell_indices = torch.multinomial(probs, num_samples, replacement=True)

            # Convert to 3D indices
            z_idx = cell_indices % self.resolution
            y_idx = (cell_indices // self.resolution) % self.resolution
            x_idx = cell_indices // (self.resolution ** 2)

            # Sample within cells
            offsets = torch.rand(num_samples, 3, device=self.device)

            points = torch.stack([x_idx, y_idx, z_idx], dim=-1).float()
            points = (points + offsets) * self.cell_size + self.bounds[0]

            all_samples.append(points)

        return torch.stack(all_samples, dim=0)


def sample_on_sphere(
    num_samples: int,
    radius: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Sample points uniformly on a sphere surface.

    Args:
        num_samples: Number of points
        radius: Sphere radius
        center: Sphere center
        device: Device for generated tensors

    Returns:
        Points of shape (num_samples, 3)
    """
    # Uniform on sphere using Gaussian normalization
    points = torch.randn(num_samples, 3, device=device)
    points = points / points.norm(dim=-1, keepdim=True)
    points = points * radius

    # Add center
    center = torch.tensor(center, device=device)
    points = points + center

    return points


def sample_in_sphere(
    num_samples: int,
    radius: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Sample points uniformly inside a sphere.

    Args:
        num_samples: Number of points
        radius: Sphere radius
        center: Sphere center
        device: Device for generated tensors

    Returns:
        Points of shape (num_samples, 3)
    """
    # Direction
    directions = torch.randn(num_samples, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Radius (cube root for uniform volume sampling)
    r = torch.rand(num_samples, 1, device=device) ** (1/3) * radius

    points = directions * r

    # Add center
    center = torch.tensor(center, device=device)
    points = points + center

    return points


def sample_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    stratified: bool = True,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays.

    Args:
        origins: Ray origins (N, 3)
        directions: Ray directions (N, 3), should be normalized
        near: Near plane distance
        far: Far plane distance
        num_samples: Number of samples per ray
        stratified: Whether to use stratified sampling
        device: Device for generated tensors

    Returns:
        (points, t_values) where:
            points: (N, num_samples, 3)
            t_values: (N, num_samples)
    """
    N = origins.shape[0]

    # Generate t values
    t_vals = torch.linspace(near, far, num_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(N, -1)

    if stratified:
        # Add uniform noise within each bin
        bin_size = (far - near) / num_samples
        noise = torch.rand(N, num_samples, device=device) * bin_size
        t_vals = t_vals + noise

    # Compute points: o + t * d
    points = origins.unsqueeze(1) + t_vals.unsqueeze(-1) * directions.unsqueeze(1)

    return points, t_vals


class HierarchicalSampler:
    """
    Hierarchical sampling for NeRF-style rendering.

    First samples coarsely, then refines based on density.
    """

    def __init__(
        self,
        near: float = 0.0,
        far: float = 5.0,
        num_coarse: int = 64,
        num_fine: int = 128,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            near: Near plane distance
            far: Far plane distance
            num_coarse: Number of coarse samples
            num_fine: Number of fine samples
            device: Device for generated tensors
        """
        self.near = near
        self.far = far
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.device = device

    def sample_coarse(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate coarse samples.

        Args:
            origins: Ray origins (N, 3)
            directions: Ray directions (N, 3)

        Returns:
            (points, t_values)
        """
        return sample_rays(
            origins, directions,
            self.near, self.far,
            self.num_coarse,
            stratified=True,
            device=self.device
        )

    def sample_fine(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        t_coarse: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate fine samples based on coarse weights.

        Args:
            origins: Ray origins (N, 3)
            directions: Ray directions (N, 3)
            t_coarse: Coarse t values (N, num_coarse)
            weights: Coarse sample weights (N, num_coarse)

        Returns:
            (points, t_values) for combined coarse + fine samples
        """
        N = origins.shape[0]

        # Normalize weights for PDF
        weights = weights + 1e-5
        pdf = weights / weights.sum(dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

        # Sample from CDF using inverse transform
        u = torch.rand(N, self.num_fine, device=self.device)

        # Find indices in CDF
        indices = torch.searchsorted(cdf, u, right=True)
        below = (indices - 1).clamp(min=0)
        above = indices.clamp(max=cdf.shape[-1] - 1)

        # Gather CDF values
        cdf_below = torch.gather(cdf, 1, below)
        cdf_above = torch.gather(cdf, 1, above)

        # Gather t values
        t_vals_expanded = torch.cat([t_coarse, t_coarse[:, -1:]], dim=-1)
        t_below = torch.gather(t_vals_expanded, 1, below)
        t_above = torch.gather(t_vals_expanded, 1, above)

        # Linear interpolation
        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t_fine = t_below + (u - cdf_below) / denom * (t_above - t_below)

        # Combine and sort
        t_combined, _ = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1)

        # Compute points
        points = origins.unsqueeze(1) + t_combined.unsqueeze(-1) * directions.unsqueeze(1)

        return points, t_combined


# =============================================================================
# Curriculum and Adaptive Sampling (Enhanced)
# =============================================================================


class CurriculumImportanceSampler:
    """
    Progressive sampling from uniform distribution to surface-focused.

    Implements curriculum learning for SDF training by gradually increasing
    the concentration of samples near the surface. Early training uses
    uniform sampling for global coverage, while later stages focus on
    surface details.

    The curriculum follows: uniform -> near-surface -> exact surface
    """

    def __init__(
        self,
        sdf_fn: Callable[[torch.Tensor], torch.Tensor],
        bounds: Tuple[float, float] = (-1.0, 1.0),
        curriculum_steps: int = 1000,
        surface_band: float = 0.1,
        min_surface_ratio: float = 0.0,
        max_surface_ratio: float = 0.9,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            sdf_fn: Function mapping points (N, 3) -> SDF values (N,)
            bounds: Coordinate bounds for uniform sampling
            curriculum_steps: Total steps in curriculum
            surface_band: Width of band around surface for importance
            min_surface_ratio: Starting surface focus ratio
            max_surface_ratio: Final surface focus ratio
            device: Device for generated tensors
        """
        self.sdf_fn = sdf_fn
        self.bounds = bounds
        self.curriculum_steps = curriculum_steps
        self.surface_band = surface_band
        self.min_surface_ratio = min_surface_ratio
        self.max_surface_ratio = max_surface_ratio
        self.device = device

        # Current step
        self._current_step = 0
        self._surface_ratio = min_surface_ratio

    def set_step(self, step: int):
        """
        Update curriculum step and compute new surface ratio.

        Args:
            step: Current training step
        """
        self._current_step = min(step, self.curriculum_steps)
        progress = self._current_step / max(self.curriculum_steps, 1)
        # Smooth progression using cosine schedule
        self._surface_ratio = (
            self.min_surface_ratio +
            (self.max_surface_ratio - self.min_surface_ratio) *
            (1 - np.cos(progress * np.pi)) / 2
        )

    @property
    def surface_ratio(self) -> float:
        """Current ratio of surface-focused samples."""
        return self._surface_ratio

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate curriculum-aware samples.

        Returns points biased toward surface based on curriculum progress.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches

        Returns:
            (points, weights) where:
                points: (batch_size, num_samples, 3)
                weights: (batch_size, num_samples) importance weights
        """
        num_surface = int(num_samples * self._surface_ratio)
        num_uniform = num_samples - num_surface

        all_points = []
        all_weights = []

        for _ in range(batch_size):
            batch_points = []
            batch_weights = []

            # Uniform samples
            if num_uniform > 0:
                uniform_pts = torch.rand(num_uniform, 3, device=self.device)
                uniform_pts = uniform_pts * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
                batch_points.append(uniform_pts)
                batch_weights.append(torch.ones(num_uniform, device=self.device))

            # Surface-biased samples via rejection sampling
            if num_surface > 0:
                # Generate candidates (oversample for rejection)
                num_candidates = num_surface * 10
                candidates = torch.rand(num_candidates, 3, device=self.device)
                candidates = candidates * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

                # Evaluate SDF
                sdf_vals = self.sdf_fn(candidates)
                if sdf_vals.dim() > 1:
                    sdf_vals = sdf_vals.squeeze(-1)

                # Importance based on distance to surface
                importance = torch.exp(-torch.abs(sdf_vals) / self.surface_band)
                importance = importance / importance.sum()

                # Sample based on importance
                indices = torch.multinomial(importance, num_surface, replacement=True)
                surface_pts = candidates[indices]
                surface_weights = 1.0 / (importance[indices] * num_candidates + 1e-8)

                batch_points.append(surface_pts)
                batch_weights.append(surface_weights)

            # Combine
            points = torch.cat(batch_points, dim=0)
            weights = torch.cat(batch_weights, dim=0)

            # Shuffle
            perm = torch.randperm(num_samples, device=self.device)
            points = points[perm]
            weights = weights[perm]

            # Normalize weights
            weights = weights / weights.mean()

            all_points.append(points)
            all_weights.append(weights)

        return torch.stack(all_points), torch.stack(all_weights)


class NewtonProjectionSampler:
    """
    Project random points onto the SDF surface via Newton's method.

    Generates exact surface points by iteratively projecting random points
    toward the zero level set using gradient descent. Useful for:
    - Generating ground truth surface samples
    - Finding surface correspondences
    - Computing surface metrics
    """

    def __init__(
        self,
        sdf_fn: Callable[[torch.Tensor], torch.Tensor],
        bounds: Tuple[float, float] = (-1.0, 1.0),
        num_iterations: int = 5,
        step_size: float = 1.0,
        convergence_threshold: float = 1e-4,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            sdf_fn: Differentiable SDF function mapping (N, 3) -> (N,)
            bounds: Coordinate bounds for initial random points
            num_iterations: Maximum Newton iterations
            step_size: Step size multiplier (1.0 = full Newton step)
            convergence_threshold: Early stopping threshold
            device: Device for generated tensors
        """
        self.sdf_fn = sdf_fn
        self.bounds = bounds
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.convergence_threshold = convergence_threshold
        self.device = device

    def _compute_gradient(self, points: torch.Tensor) -> torch.Tensor:
        """Compute SDF gradient using autograd."""
        points = points.requires_grad_(True)
        sdf = self.sdf_fn(points)
        if sdf.dim() > 1:
            sdf = sdf.squeeze(-1)

        grad = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=points,
            create_graph=False,
            retain_graph=False
        )[0]

        return grad

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1,
        return_convergence: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate surface samples via Newton projection.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches
            return_convergence: If True, return convergence info

        Returns:
            If return_convergence=False:
                surface_points: (batch_size, num_samples, 3)
            If return_convergence=True:
                (surface_points, converged) where converged is boolean mask
        """
        all_points = []
        all_converged = []

        for _ in range(batch_size):
            # Initialize with random points
            points = torch.rand(num_samples, 3, device=self.device)
            points = points * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

            converged = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

            for _ in range(self.num_iterations):
                # Skip already converged points
                active = ~converged
                if not active.any():
                    break

                active_points = points[active].clone()

                # Compute SDF and gradient
                with torch.enable_grad():
                    grad = self._compute_gradient(active_points)

                with torch.no_grad():
                    sdf = self.sdf_fn(active_points)
                    if sdf.dim() > 1:
                        sdf = sdf.squeeze(-1)

                    # Newton step: p_new = p - sdf * grad / |grad|^2
                    grad_norm_sq = (grad ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    step = sdf.unsqueeze(-1) * grad / grad_norm_sq

                    # Update points
                    new_points = active_points - self.step_size * step

                    # Clamp to bounds
                    new_points = new_points.clamp(self.bounds[0], self.bounds[1])

                    # Check convergence
                    point_converged = torch.abs(sdf) < self.convergence_threshold
                    converged[active] = point_converged

                    # Update
                    points[active] = new_points

            all_points.append(points.detach())
            all_converged.append(converged)

        points_out = torch.stack(all_points)

        if return_convergence:
            converged_out = torch.stack(all_converged)
            return points_out, converged_out

        return points_out

    def project(
        self,
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Project given points onto the surface.

        Args:
            points: Points to project (N, 3) or (B, N, 3)

        Returns:
            Projected surface points with same shape
        """
        input_shape = points.shape
        if points.dim() == 2:
            points = points.unsqueeze(0)

        batch_size, num_points, _ = points.shape

        # Flatten for processing
        points_flat = points.reshape(-1, 3)

        for _ in range(self.num_iterations):
            with torch.enable_grad():
                grad = self._compute_gradient(points_flat)

            with torch.no_grad():
                sdf = self.sdf_fn(points_flat)
                if sdf.dim() > 1:
                    sdf = sdf.squeeze(-1)

                grad_norm_sq = (grad ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
                step = sdf.unsqueeze(-1) * grad / grad_norm_sq
                points_flat = points_flat - self.step_size * step
                points_flat = points_flat.clamp(self.bounds[0], self.bounds[1])

        result = points_flat.reshape(input_shape)
        return result.detach()


class ErrorDrivenSampler:
    """
    Adaptive sampler that focuses on high-error regions during training.

    Maintains a 3D error grid that tracks reconstruction loss across space.
    Samples are biased toward regions with higher historical errors, enabling
    the model to focus on difficult areas.

    Key features:
    - Exponential moving average of errors
    - Temperature-controlled exploration/exploitation
    - Grid-based spatial tracking
    """

    def __init__(
        self,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        grid_resolution: int = 32,
        error_decay: float = 0.9,
        temperature: float = 1.0,
        min_probability: float = 0.01,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            bounds: Coordinate bounds
            grid_resolution: Resolution of error tracking grid
            error_decay: Decay factor for error EMA (0-1)
            temperature: Sampling temperature (higher = more uniform)
            min_probability: Minimum sampling probability per cell
            device: Device for generated tensors
        """
        self.bounds = bounds
        self.grid_resolution = grid_resolution
        self.error_decay = error_decay
        self.temperature = temperature
        self.min_probability = min_probability
        self.device = device

        # Initialize error grid with uniform prior
        self.error_grid = torch.ones(
            grid_resolution, grid_resolution, grid_resolution,
            device=device
        )

        # Cell size for coordinate conversion
        self.cell_size = (bounds[1] - bounds[0]) / grid_resolution

        # Statistics
        self._update_count = 0
        self._total_samples = 0

    def _point_to_cell(self, points: torch.Tensor) -> torch.Tensor:
        """Convert points to grid cell indices."""
        normalized = (points - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        indices = (normalized * self.grid_resolution).long()
        return indices.clamp(0, self.grid_resolution - 1)

    def update_errors(
        self,
        points: torch.Tensor,
        errors: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ):
        """
        Update error grid based on observed training errors.

        Uses scatter_add for efficient batch updates with exponential
        moving average.

        Args:
            points: Sample points (N, 3) or (B, N, 3)
            errors: Error values (N,) or (B, N)
            weights: Optional per-point weights (N,) or (B, N)
        """
        # Flatten inputs
        if points.dim() == 3:
            points = points.reshape(-1, 3)
            errors = errors.reshape(-1)
            if weights is not None:
                weights = weights.reshape(-1)

        if weights is None:
            weights = torch.ones_like(errors)

        # Convert to cell indices
        cell_indices = self._point_to_cell(points)

        # Compute flat indices
        flat_indices = (
            cell_indices[:, 0] * self.grid_resolution ** 2 +
            cell_indices[:, 1] * self.grid_resolution +
            cell_indices[:, 2]
        )

        # Accumulate weighted errors per cell
        weighted_errors = errors * weights

        # Create accumulators
        error_sum = torch.zeros(
            self.grid_resolution ** 3, device=self.device
        ).scatter_add_(0, flat_indices, weighted_errors)

        weight_sum = torch.zeros(
            self.grid_resolution ** 3, device=self.device
        ).scatter_add_(0, flat_indices, weights)

        # Compute mean errors per cell (avoid division by zero)
        mask = weight_sum > 0
        new_errors = torch.zeros_like(error_sum)
        new_errors[mask] = error_sum[mask] / weight_sum[mask]

        # Reshape to grid
        new_errors = new_errors.reshape(
            self.grid_resolution, self.grid_resolution, self.grid_resolution
        )
        update_mask = mask.reshape(
            self.grid_resolution, self.grid_resolution, self.grid_resolution
        )

        # EMA update only for observed cells
        self.error_grid = torch.where(
            update_mask,
            self.error_decay * self.error_grid + (1 - self.error_decay) * new_errors,
            self.error_grid
        )

        self._update_count += 1
        self._total_samples += len(points)

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Sample points based on error distribution.

        Higher error regions are sampled more frequently.

        Args:
            num_samples: Number of points per batch
            batch_size: Number of batches

        Returns:
            Points of shape (batch_size, num_samples, 3)
        """
        # Compute sampling probabilities with temperature
        log_probs = torch.log(self.error_grid.flatten() + 1e-8) / self.temperature
        probs = F.softmax(log_probs, dim=0)

        # Apply minimum probability
        probs = probs.clamp(min=self.min_probability)
        probs = probs / probs.sum()

        all_samples = []

        for _ in range(batch_size):
            # Sample cells
            cell_indices = torch.multinomial(probs, num_samples, replacement=True)

            # Convert to 3D indices
            x_idx = cell_indices // (self.grid_resolution ** 2)
            y_idx = (cell_indices // self.grid_resolution) % self.grid_resolution
            z_idx = cell_indices % self.grid_resolution

            # Random offset within each cell
            offsets = torch.rand(num_samples, 3, device=self.device)

            # Convert to world coordinates
            cell_coords = torch.stack([x_idx, y_idx, z_idx], dim=-1).float()
            points = (cell_coords + offsets) * self.cell_size + self.bounds[0]

            all_samples.append(points)

        return torch.stack(all_samples, dim=0)

    def get_error_heatmap(self, axis: int = 2) -> torch.Tensor:
        """
        Get 2D heatmap of errors by averaging along an axis.

        Args:
            axis: Axis to average along (0=x, 1=y, 2=z)

        Returns:
            2D error heatmap
        """
        return self.error_grid.mean(dim=axis)

    def reset(self):
        """Reset error grid to uniform."""
        self.error_grid.fill_(1.0)
        self._update_count = 0
        self._total_samples = 0


# Import numpy for curriculum sampler's cosine schedule
import numpy as np

"""
Tests for point sampling strategies.

Following the Yoneda philosophy: these tests completely DEFINE how each
sampler generates points. The tests serve as executable documentation for:

1. UniformSampler - Uniform random sampling in bounding box
2. SurfaceSampler - Sampling near mesh surfaces
3. MixedSampler - Combined uniform and surface sampling
4. ImportanceSampler - Importance-weighted sampling
5. StratifiedSampler - Stratified grid-based sampling
6. NearSurfaceSampler - Distance-banded surface sampling
7. AdaptiveSampler - Error-driven adaptive sampling
8. HierarchicalSampler - NeRF-style hierarchical ray sampling
9. Utility functions: sample_on_sphere, sample_in_sphere, sample_rays
"""

import pytest
import torch
import math

from pga_inr.data.sampling import (
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


# =============================================================================
# UniformSampler Tests
# =============================================================================

class TestUniformSamplerCreation:
    """Tests for UniformSampler construction."""

    def test_default_bounds(self):
        """Default bounds are [-1, 1]."""
        sampler = UniformSampler()
        assert sampler.bounds == (-1.0, 1.0)

    def test_custom_bounds(self):
        """Custom bounds are stored correctly."""
        sampler = UniformSampler(bounds=(-5.0, 5.0))
        assert sampler.bounds == (-5.0, 5.0)

    def test_device_specification(self):
        """Device is stored correctly."""
        sampler = UniformSampler(device=torch.device('cpu'))
        assert sampler.device == torch.device('cpu')


class TestUniformSamplerSample:
    """Tests for UniformSampler sample method."""

    def test_output_shape_single_batch(self):
        """Output shape for single batch."""
        sampler = UniformSampler()
        points = sampler.sample(num_samples=100, batch_size=1)
        assert points.shape == (1, 100, 3)

    def test_output_shape_multiple_batches(self):
        """Output shape for multiple batches."""
        sampler = UniformSampler()
        points = sampler.sample(num_samples=50, batch_size=4)
        assert points.shape == (4, 50, 3)

    def test_points_within_bounds(self):
        """All points are within specified bounds."""
        bounds = (-2.0, 3.0)
        sampler = UniformSampler(bounds=bounds)
        points = sampler.sample(num_samples=1000)
        assert points.min() >= bounds[0]
        assert points.max() <= bounds[1]

    def test_uniform_distribution(self):
        """Points are approximately uniformly distributed."""
        sampler = UniformSampler(bounds=(-1.0, 1.0))
        points = sampler.sample(num_samples=10000)

        # Mean should be approximately 0
        assert torch.abs(points.mean()) < 0.1

        # Check each dimension
        for dim in range(3):
            dim_mean = points[..., dim].mean()
            assert torch.abs(dim_mean) < 0.1

    def test_randomness(self):
        """Different calls produce different samples."""
        sampler = UniformSampler()
        points1 = sampler.sample(num_samples=100)
        points2 = sampler.sample(num_samples=100)
        assert not torch.allclose(points1, points2)


# =============================================================================
# SurfaceSampler Tests
# =============================================================================

class TestSurfaceSamplerCreation:
    """Tests for SurfaceSampler construction."""

    def test_surface_points_stored(self):
        """Surface points are stored correctly."""
        surface_pts = torch.randn(100, 3)
        sampler = SurfaceSampler(surface_points=surface_pts)
        assert sampler.surface_points.shape == (100, 3)

    def test_optional_normals(self):
        """Normals are optional."""
        surface_pts = torch.randn(100, 3)
        sampler = SurfaceSampler(surface_points=surface_pts)
        assert sampler.surface_normals is None

    def test_normals_stored(self):
        """When provided, normals are stored."""
        surface_pts = torch.randn(100, 3)
        normals = torch.randn(100, 3)
        sampler = SurfaceSampler(surface_points=surface_pts, surface_normals=normals)
        assert sampler.surface_normals.shape == (100, 3)


class TestSurfaceSamplerSample:
    """Tests for SurfaceSampler sample method."""

    def test_output_shape(self):
        """Output has correct shape."""
        surface_pts = torch.randn(100, 3)
        sampler = SurfaceSampler(surface_points=surface_pts)
        points, normals = sampler.sample(num_samples=50, batch_size=2)
        assert points.shape == (2, 50, 3)
        assert normals is None

    def test_returns_normals_when_available(self):
        """Returns normals when they were provided."""
        surface_pts = torch.randn(100, 3)
        surface_normals = torch.randn(100, 3)
        sampler = SurfaceSampler(surface_points=surface_pts, surface_normals=surface_normals)
        points, normals = sampler.sample(num_samples=50)
        assert normals is not None
        assert normals.shape == (1, 50, 3)

    def test_no_noise_option(self):
        """Can disable noise addition."""
        surface_pts = torch.randn(100, 3)
        sampler = SurfaceSampler(surface_points=surface_pts, noise_std=0.0)
        points, _ = sampler.sample(num_samples=100, add_noise=False)
        # Points should be exactly on surface
        # Check that all sampled points are in the surface_pts set
        for i in range(points.shape[1]):
            point = points[0, i]
            distances = torch.norm(surface_pts - point, dim=-1)
            assert distances.min() < 1e-5

    def test_noise_offset_applied(self):
        """Noise is applied when add_noise=True."""
        surface_pts = torch.tensor([[0.0, 0.0, 0.0]])  # Single point
        sampler = SurfaceSampler(surface_points=surface_pts, noise_std=0.5)
        points, _ = sampler.sample(num_samples=100, add_noise=True)
        # Points should be offset from origin
        distances = torch.norm(points, dim=-1)
        assert distances.mean() > 0.1  # Should be offset by noise

    def test_normal_directed_noise(self):
        """When normals provided, noise is along normal direction."""
        surface_pts = torch.tensor([[0.0, 0.0, 0.0]])
        surface_normals = torch.tensor([[0.0, 0.0, 1.0]])  # Z direction
        sampler = SurfaceSampler(
            surface_points=surface_pts,
            surface_normals=surface_normals,
            noise_std=1.0
        )
        points, _ = sampler.sample(num_samples=1000, add_noise=True)
        # X and Y should remain near 0
        assert torch.abs(points[..., 0].mean()) < 0.1
        assert torch.abs(points[..., 1].mean()) < 0.1


# =============================================================================
# MixedSampler Tests
# =============================================================================

class TestMixedSamplerCreation:
    """Tests for MixedSampler construction."""

    def test_combines_samplers(self):
        """Creates both uniform and surface samplers."""
        surface_pts = torch.randn(100, 3)
        sampler = MixedSampler(surface_points=surface_pts, surface_ratio=0.5)
        assert sampler.uniform_sampler is not None
        assert sampler.surface_sampler is not None


class TestMixedSamplerSample:
    """Tests for MixedSampler sample method."""

    def test_output_shape(self):
        """Output has correct shape."""
        surface_pts = torch.randn(100, 3)
        sampler = MixedSampler(surface_points=surface_pts)
        points, is_surface, normals = sampler.sample(num_samples=100)
        assert points.shape == (1, 100, 3)
        assert is_surface.shape == (1, 100)

    def test_surface_ratio(self):
        """Surface ratio determines fraction of surface samples."""
        surface_pts = torch.randn(100, 3)
        sampler = MixedSampler(surface_points=surface_pts, surface_ratio=0.7)
        _, is_surface, _ = sampler.sample(num_samples=100)
        # 70 samples should be surface samples
        assert is_surface.sum() == 70

    def test_mask_correct(self):
        """is_surface mask marks first samples as surface."""
        surface_pts = torch.randn(100, 3)
        sampler = MixedSampler(surface_points=surface_pts, surface_ratio=0.5)
        _, is_surface, _ = sampler.sample(num_samples=100)
        # First 50 should be surface
        assert is_surface[0, :50].all()
        # Last 50 should be uniform
        assert not is_surface[0, 50:].any()


# =============================================================================
# ImportanceSampler Tests
# =============================================================================

class TestImportanceSamplerCreation:
    """Tests for ImportanceSampler construction."""

    def test_criterion_stored(self):
        """Criterion function is stored."""
        criterion = lambda x: torch.ones(x.shape[0])
        sampler = ImportanceSampler(criterion_fn=criterion)
        assert sampler.criterion_fn is not None


class TestImportanceSamplerSample:
    """Tests for ImportanceSampler sample method."""

    def test_output_shape(self):
        """Output has correct shape."""
        criterion = lambda x: torch.ones(x.shape[0])
        sampler = ImportanceSampler(criterion_fn=criterion, num_candidates=1000)
        points = sampler.sample(num_samples=100)
        assert points.shape == (1, 100, 3)

    def test_importance_weighting(self):
        """Samples concentrate in high-importance regions."""
        # Criterion: importance is high for x > 0
        def criterion(pts):
            return (pts[:, 0] > 0).float() + 0.01

        sampler = ImportanceSampler(
            criterion_fn=criterion,
            bounds=(-1.0, 1.0),
            num_candidates=10000
        )
        points = sampler.sample(num_samples=1000)

        # Most points should have x > 0
        positive_x_ratio = (points[..., 0] > 0).float().mean()
        assert positive_x_ratio > 0.7  # Much more than 50%

    def test_points_within_bounds(self):
        """Samples are within bounds."""
        criterion = lambda x: torch.ones(x.shape[0])
        bounds = (-2.0, 2.0)
        sampler = ImportanceSampler(criterion_fn=criterion, bounds=bounds)
        points = sampler.sample(num_samples=100)
        assert points.min() >= bounds[0]
        assert points.max() <= bounds[1]


# =============================================================================
# StratifiedSampler Tests
# =============================================================================

class TestStratifiedSamplerCreation:
    """Tests for StratifiedSampler construction."""

    def test_default_resolution(self):
        """Default resolution is 8."""
        sampler = StratifiedSampler()
        assert sampler.resolution == 8

    def test_cell_size_calculation(self):
        """Cell size is computed correctly."""
        sampler = StratifiedSampler(resolution=4, bounds=(-2.0, 2.0))
        assert sampler.cell_size == 1.0  # 4 / 4


class TestStratifiedSamplerSample:
    """Tests for StratifiedSampler sample method."""

    def test_output_shape(self):
        """Output has correct shape."""
        sampler = StratifiedSampler(resolution=4)
        points = sampler.sample(num_samples=100)
        assert points.shape == (1, 100, 3)

    def test_points_within_bounds(self):
        """All points within bounds."""
        bounds = (-1.0, 1.0)
        sampler = StratifiedSampler(resolution=4, bounds=bounds)
        points = sampler.sample(num_samples=100)
        assert points.min() >= bounds[0]
        assert points.max() <= bounds[1]

    def test_jitter_option(self):
        """Jitter adds randomness within cells."""
        sampler = StratifiedSampler(resolution=2)
        # Without jitter, points would be at cell centers
        points_no_jitter = sampler.sample(num_samples=8, jitter=False)
        points_jitter = sampler.sample(num_samples=8, jitter=True)
        # With jitter, points should differ
        assert not torch.allclose(points_no_jitter, points_jitter)


# =============================================================================
# NearSurfaceSampler Tests
# =============================================================================

class TestNearSurfaceSamplerCreation:
    """Tests for NearSurfaceSampler construction."""

    def test_distance_bands_stored(self):
        """Distance bands are stored correctly."""
        surface_pts = torch.randn(100, 3)
        surface_normals = torch.nn.functional.normalize(torch.randn(100, 3), dim=-1)
        bands = (0.0, 0.01, 0.05, 0.1)
        sampler = NearSurfaceSampler(
            surface_points=surface_pts,
            surface_normals=surface_normals,
            distance_bands=bands
        )
        assert sampler.distance_bands == bands


class TestNearSurfaceSamplerSample:
    """Tests for NearSurfaceSampler sample method."""

    def test_output_shape(self):
        """Output has correct shape."""
        surface_pts = torch.randn(100, 3)
        surface_normals = torch.nn.functional.normalize(torch.randn(100, 3), dim=-1)
        sampler = NearSurfaceSampler(surface_pts, surface_normals)
        points, distances = sampler.sample(num_samples=100)
        assert points.shape == (1, 100, 3)
        assert distances.shape == (1, 100, 1)

    def test_signed_distances_returned(self):
        """Returns signed distances."""
        surface_pts = torch.zeros(10, 3)  # All at origin
        surface_normals = torch.tensor([[1.0, 0.0, 0.0]]).expand(10, 3)
        sampler = NearSurfaceSampler(
            surface_pts, surface_normals,
            distance_bands=(0.0, 0.1, 0.2)
        )
        _, distances = sampler.sample(num_samples=100)
        # Should have both positive and negative distances
        assert (distances > 0).any()
        assert (distances < 0).any()

    def test_distances_within_bands(self):
        """Distances are within expected bands."""
        surface_pts = torch.randn(100, 3)
        surface_normals = torch.nn.functional.normalize(torch.randn(100, 3), dim=-1)
        bands = (0.0, 0.1, 0.2)
        sampler = NearSurfaceSampler(surface_pts, surface_normals, distance_bands=bands)
        _, distances = sampler.sample(num_samples=100)
        # Absolute distances should be within max band
        assert distances.abs().max() <= 0.2 + 1e-5


# =============================================================================
# AdaptiveSampler Tests
# =============================================================================

class TestAdaptiveSamplerCreation:
    """Tests for AdaptiveSampler construction."""

    def test_error_grid_initialized(self):
        """Error grid is initialized to uniform."""
        sampler = AdaptiveSampler(resolution=8)
        assert sampler.error_grid.shape == (8, 8, 8)
        assert torch.allclose(sampler.error_grid, torch.ones(8, 8, 8))


class TestAdaptiveSamplerSample:
    """Tests for AdaptiveSampler sample method."""

    def test_output_shape(self):
        """Output has correct shape."""
        sampler = AdaptiveSampler(resolution=4)
        points = sampler.sample(num_samples=100)
        assert points.shape == (1, 100, 3)

    def test_points_within_bounds(self):
        """Points are within bounds."""
        bounds = (-1.0, 1.0)
        sampler = AdaptiveSampler(bounds=bounds, resolution=4)
        points = sampler.sample(num_samples=100)
        assert points.min() >= bounds[0]
        assert points.max() <= bounds[1]

    def test_update_errors(self):
        """Error update affects sampling distribution."""
        sampler = AdaptiveSampler(resolution=4, temperature=0.1)

        # Update errors: high error in positive octant
        points = torch.tensor([[0.5, 0.5, 0.5]])
        errors = torch.tensor([100.0])
        sampler.update_errors(points, errors)

        # Sample many points
        samples = sampler.sample(num_samples=1000)

        # Should sample more from high-error region
        in_positive_octant = (samples > 0).all(dim=-1).float().mean()
        assert in_positive_octant > 0.2  # More than 1/8


# =============================================================================
# HierarchicalSampler Tests
# =============================================================================

class TestHierarchicalSamplerCreation:
    """Tests for HierarchicalSampler construction."""

    def test_parameters_stored(self):
        """Parameters are stored correctly."""
        sampler = HierarchicalSampler(near=0.0, far=5.0, num_coarse=64, num_fine=128)
        assert sampler.near == 0.0
        assert sampler.far == 5.0
        assert sampler.num_coarse == 64
        assert sampler.num_fine == 128


class TestHierarchicalSamplerCoarse:
    """Tests for coarse sampling."""

    def test_coarse_output_shape(self):
        """Coarse samples have correct shape."""
        sampler = HierarchicalSampler(num_coarse=32)
        origins = torch.zeros(10, 3)
        directions = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
        points, t_vals = sampler.sample_coarse(origins, directions)
        assert points.shape == (10, 32, 3)
        assert t_vals.shape == (10, 32)

    def test_coarse_points_on_rays(self):
        """Coarse points lie on rays."""
        sampler = HierarchicalSampler(near=0.0, far=5.0, num_coarse=32)
        origins = torch.zeros(5, 3)
        directions = torch.tensor([[1.0, 0.0, 0.0]]).expand(5, 3)
        points, t_vals = sampler.sample_coarse(origins, directions)

        # Points should be o + t * d
        for i in range(5):
            for j in range(32):
                expected = origins[i] + t_vals[i, j] * directions[i]
                assert torch.allclose(points[i, j], expected, atol=1e-5)

    def test_t_values_within_range(self):
        """t values are within [near, far]."""
        near, far = 1.0, 10.0
        sampler = HierarchicalSampler(near=near, far=far, num_coarse=64)
        origins = torch.zeros(10, 3)
        directions = torch.randn(10, 3)
        _, t_vals = sampler.sample_coarse(origins, directions)
        assert t_vals.min() >= near
        assert t_vals.max() <= far + (far - near) / 64  # Account for bin noise


class TestHierarchicalSamplerFine:
    """Tests for fine sampling."""

    def test_fine_output_shape(self):
        """Fine samples have correct shape."""
        sampler = HierarchicalSampler(num_coarse=32, num_fine=64)
        origins = torch.zeros(10, 3)
        directions = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)

        # First get coarse samples
        _, t_coarse = sampler.sample_coarse(origins, directions)
        weights = torch.rand(10, 32)

        # Then refine
        points, t_vals = sampler.sample_fine(origins, directions, t_coarse, weights)
        # Combined coarse + fine
        assert t_vals.shape == (10, 32 + 64)
        assert points.shape == (10, 32 + 64, 3)

    def test_fine_t_values_sorted(self):
        """Fine t values are sorted."""
        sampler = HierarchicalSampler(num_coarse=16, num_fine=32)
        origins = torch.zeros(5, 3)
        directions = torch.randn(5, 3)
        _, t_coarse = sampler.sample_coarse(origins, directions)
        weights = torch.rand(5, 16)
        _, t_fine = sampler.sample_fine(origins, directions, t_coarse, weights)

        # Check sorted
        t_sorted, _ = torch.sort(t_fine, dim=-1)
        assert torch.allclose(t_fine, t_sorted)


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestSampleOnSphere:
    """Tests for sample_on_sphere utility."""

    def test_output_shape(self):
        """Output has correct shape."""
        points = sample_on_sphere(num_samples=100)
        assert points.shape == (100, 3)

    def test_points_on_sphere(self):
        """All points lie on sphere surface."""
        radius = 2.5
        points = sample_on_sphere(num_samples=1000, radius=radius)
        distances = torch.norm(points, dim=-1)
        assert torch.allclose(distances, torch.full((1000,), radius), atol=1e-5)

    def test_center_offset(self):
        """Center offset works correctly."""
        center = (1.0, 2.0, 3.0)
        points = sample_on_sphere(num_samples=100, center=center)
        centroid = points.mean(dim=0)
        # Centroid should be approximately at center
        assert torch.allclose(centroid, torch.tensor(center), atol=0.2)

    def test_uniform_distribution(self):
        """Points are approximately uniformly distributed on sphere."""
        points = sample_on_sphere(num_samples=10000, radius=1.0)
        # Mean should be near origin
        mean = points.mean(dim=0)
        assert torch.norm(mean) < 0.1


class TestSampleInSphere:
    """Tests for sample_in_sphere utility."""

    def test_output_shape(self):
        """Output has correct shape."""
        points = sample_in_sphere(num_samples=100)
        assert points.shape == (100, 3)

    def test_points_inside_sphere(self):
        """All points are inside sphere."""
        radius = 2.0
        points = sample_in_sphere(num_samples=1000, radius=radius)
        distances = torch.norm(points, dim=-1)
        assert (distances <= radius).all()

    def test_volume_uniform(self):
        """Points are volume-uniformly distributed (more near surface)."""
        points = sample_in_sphere(num_samples=10000, radius=1.0)
        distances = torch.norm(points, dim=-1)
        # Mean distance should be 3/4 * R for uniform volume
        expected_mean = 0.75
        assert torch.abs(distances.mean() - expected_mean) < 0.05

    def test_center_offset(self):
        """Center offset applied correctly."""
        center = (5.0, 5.0, 5.0)
        points = sample_in_sphere(num_samples=1000, radius=1.0, center=center)
        centroid = points.mean(dim=0)
        assert torch.allclose(centroid, torch.tensor(center), atol=0.1)


class TestSampleRays:
    """Tests for sample_rays utility."""

    def test_output_shapes(self):
        """Output shapes are correct."""
        origins = torch.zeros(10, 3)
        directions = torch.randn(10, 3)
        points, t_vals = sample_rays(
            origins, directions,
            near=0.0, far=5.0, num_samples=32
        )
        assert points.shape == (10, 32, 3)
        assert t_vals.shape == (10, 32)

    def test_points_on_rays(self):
        """Points lie on the specified rays."""
        origins = torch.randn(5, 3)
        directions = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
        points, t_vals = sample_rays(
            origins, directions,
            near=1.0, far=10.0, num_samples=16
        )

        # Verify: point = origin + t * direction
        for i in range(5):
            for j in range(16):
                expected = origins[i] + t_vals[i, j] * directions[i]
                assert torch.allclose(points[i, j], expected, atol=1e-5)

    def test_stratified_vs_uniform(self):
        """Stratified sampling adds jitter within bins."""
        origins = torch.zeros(10, 3)
        directions = torch.tensor([[1.0, 0.0, 0.0]]).expand(10, 3)

        _, t_strat = sample_rays(origins, directions, 0.0, 1.0, 10, stratified=True)
        _, t_uniform = sample_rays(origins, directions, 0.0, 1.0, 10, stratified=False)

        # Stratified should have some variation, uniform should be identical
        assert not torch.allclose(t_strat, t_uniform)

    def test_t_values_in_range(self):
        """t values fall within [near, far]."""
        near, far = 2.0, 8.0
        origins = torch.zeros(10, 3)
        directions = torch.randn(10, 3)
        _, t_vals = sample_rays(origins, directions, near, far, 64)
        assert t_vals.min() >= near
        # With stratified sampling, max can be slightly beyond far
        assert t_vals.max() <= far + (far - near) / 64


# =============================================================================
# Device Compatibility Tests
# =============================================================================

class TestSamplerDeviceCompatibility:
    """Tests for CPU device compatibility."""

    def test_uniform_sampler_cpu(self):
        """UniformSampler works on CPU."""
        sampler = UniformSampler(device=torch.device('cpu'))
        points = sampler.sample(100)
        assert points.device == torch.device('cpu')

    def test_surface_sampler_cpu(self):
        """SurfaceSampler works on CPU."""
        surface_pts = torch.randn(100, 3)
        sampler = SurfaceSampler(surface_points=surface_pts, device=torch.device('cpu'))
        points, _ = sampler.sample(50)
        assert points.device == torch.device('cpu')

    def test_stratified_sampler_cpu(self):
        """StratifiedSampler works on CPU."""
        sampler = StratifiedSampler(device=torch.device('cpu'))
        points = sampler.sample(100)
        assert points.device == torch.device('cpu')


# =============================================================================
# Dtype Preservation Tests
# =============================================================================

class TestSamplerDtypePreservation:
    """Tests for dtype handling."""

    def test_uniform_sampler_dtype(self):
        """UniformSampler outputs default float32."""
        sampler = UniformSampler()
        points = sampler.sample(100)
        assert points.dtype == torch.float32

    def test_surface_sampler_preserves_dtype(self):
        """SurfaceSampler preserves input dtype."""
        surface_pts = torch.randn(100, 3, dtype=torch.float64)
        sampler = SurfaceSampler(surface_points=surface_pts)
        points, _ = sampler.sample(50)
        assert points.dtype == torch.float64


# =============================================================================
# Batch Dimension Tests
# =============================================================================

class TestSamplerBatchDimensions:
    """Tests for batch dimension handling."""

    def test_uniform_batch_size_1(self):
        """Batch size 1 works."""
        sampler = UniformSampler()
        points = sampler.sample(100, batch_size=1)
        assert points.shape == (1, 100, 3)

    def test_uniform_batch_size_many(self):
        """Large batch size works."""
        sampler = UniformSampler()
        points = sampler.sample(50, batch_size=16)
        assert points.shape == (16, 50, 3)

    def test_surface_batch_size(self):
        """SurfaceSampler handles batch size."""
        surface_pts = torch.randn(100, 3)
        sampler = SurfaceSampler(surface_points=surface_pts)
        points, _ = sampler.sample(30, batch_size=8)
        assert points.shape == (8, 30, 3)


# =============================================================================
# Edge Cases
# =============================================================================

class TestSamplerEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self):
        """Handle num_samples=1."""
        sampler = UniformSampler()
        points = sampler.sample(num_samples=1)
        assert points.shape == (1, 1, 3)

    def test_zero_noise(self):
        """SurfaceSampler with zero noise."""
        surface_pts = torch.randn(10, 3)
        sampler = SurfaceSampler(surface_points=surface_pts, noise_std=0.0)
        points, _ = sampler.sample(10, add_noise=False)
        assert points.shape == (1, 10, 3)

    def test_single_surface_point(self):
        """SurfaceSampler with single surface point."""
        surface_pts = torch.tensor([[1.0, 2.0, 3.0]])
        sampler = SurfaceSampler(surface_points=surface_pts, noise_std=0.0)
        points, _ = sampler.sample(5, add_noise=False)
        # All samples should be the same point
        assert torch.allclose(points[0], torch.tensor([[1.0, 2.0, 3.0]]).expand(5, 3))

    def test_sphere_zero_radius(self):
        """sample_on_sphere with zero radius."""
        points = sample_on_sphere(num_samples=10, radius=0.0)
        assert torch.allclose(points, torch.zeros(10, 3))

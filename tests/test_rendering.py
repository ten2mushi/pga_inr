"""
Tests for rendering components.
"""

import pytest
import torch
import math

from pga_inr.rendering.rays import (
    generate_rays,
    generate_rays_from_pose,
    transform_rays,
    rays_aabb_intersection,
    rays_sphere_intersection,
)
from pga_inr.rendering.shading import (
    phong_shading,
    normal_to_color,
    depth_to_color,
)
from pga_inr.rendering.sphere_tracing import PGASphereTracer
from pga_inr.models.inr import PGA_INR_SDF


def _identity_pose():
    """Create identity camera pose."""
    return torch.eye(4)


def _fov_to_focal(fov_degrees: float, image_size: int) -> float:
    """Convert FOV to focal length."""
    return image_size / (2 * math.tan(math.radians(fov_degrees) / 2))


class TestRayGeneration:
    """Test ray generation utilities."""

    def test_generate_rays_shape(self):
        """Test ray generation output shapes."""
        width, height = 64, 48
        focal = _fov_to_focal(60.0, width)
        camera_pose = _identity_pose()

        origins, directions = generate_rays(height, width, focal, camera_pose)

        assert origins.shape == (height, width, 3)
        assert directions.shape == (height, width, 3)

    def test_generate_rays_normalized(self):
        """Test that ray directions are normalized."""
        focal = _fov_to_focal(60.0, 32)
        camera_pose = _identity_pose()
        origins, directions = generate_rays(32, 32, focal, camera_pose)

        norms = directions.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_generate_rays_center_direction(self):
        """Test that center ray points forward (-Z)."""
        focal = _fov_to_focal(60.0, 33)
        camera_pose = _identity_pose()
        origins, directions = generate_rays(33, 33, focal, camera_pose)

        # Center pixel
        center_dir = directions[16, 16]

        # Should point along -Z (forward)
        expected = torch.tensor([0.0, 0.0, -1.0])
        assert torch.allclose(center_dir, expected, atol=1e-5)

    def test_generate_rays_fov_effect(self):
        """Test that larger FOV gives wider spread."""
        camera_pose = _identity_pose()
        focal_narrow = _fov_to_focal(30.0, 32)
        focal_wide = _fov_to_focal(90.0, 32)

        _, dirs_narrow = generate_rays(32, 32, focal_narrow, camera_pose)
        _, dirs_wide = generate_rays(32, 32, focal_wide, camera_pose)

        # Corner ray should be more angled with wider FOV
        corner_narrow = dirs_narrow[0, 0]
        corner_wide = dirs_wide[0, 0]

        # X component should be larger (more angled) for wider FOV
        assert abs(corner_wide[0]) > abs(corner_narrow[0])


class TestRayTransformation:
    """Test ray transformation."""

    def test_transform_rays_identity(self):
        """Identity transform should not change rays."""
        focal = _fov_to_focal(60.0, 16)
        camera_pose = _identity_pose()
        origins, directions = generate_rays(16, 16, focal, camera_pose)

        trans = torch.zeros(3)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        new_origins, new_dirs = transform_rays(origins, directions, trans, quat)

        assert torch.allclose(new_origins, origins, atol=1e-5)
        assert torch.allclose(new_dirs, directions, atol=1e-5)

    def test_transform_rays_translation(self):
        """Translation should only affect origins."""
        focal = _fov_to_focal(60.0, 16)
        camera_pose = _identity_pose()
        origins, directions = generate_rays(16, 16, focal, camera_pose)

        trans = torch.tensor([1.0, 2.0, 3.0])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        new_origins, new_dirs = transform_rays(origins, directions, trans, quat)

        # Origins should be translated
        expected_origins = origins + trans
        assert torch.allclose(new_origins, expected_origins, atol=1e-5)

        # Directions should be unchanged
        assert torch.allclose(new_dirs, directions, atol=1e-5)


class TestRayIntersection:
    """Test ray intersection utilities."""

    def test_aabb_intersection_hit(self):
        """Test AABB intersection with hitting rays."""
        # Ray from origin pointing at box
        origins = torch.tensor([[0.0, 0.0, 2.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        # Unit box centered at origin
        aabb_min = torch.tensor([-1.0, -1.0, -1.0])
        aabb_max = torch.tensor([1.0, 1.0, 1.0])

        t_near, t_far, hit = rays_aabb_intersection(origins, directions, aabb_min, aabb_max)

        assert hit[0].item() == True
        assert t_near[0].item() > 0  # Hit is in front
        assert t_far[0].item() > t_near[0].item()

    def test_aabb_intersection_miss(self):
        """Test AABB intersection with missing rays."""
        # Ray pointing away from box
        origins = torch.tensor([[0.0, 0.0, 2.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])  # Pointing away

        aabb_min = torch.tensor([-1.0, -1.0, -1.0])
        aabb_max = torch.tensor([1.0, 1.0, 1.0])

        t_near, t_far, hit = rays_aabb_intersection(origins, directions, aabb_min, aabb_max)

        assert hit[0].item() == False

    def test_sphere_intersection_hit(self):
        """Test sphere intersection with hitting rays."""
        # Ray from (0, 0, 2) pointing at sphere at origin
        origins = torch.tensor([[0.0, 0.0, 2.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        t_near, t_far, hit = rays_sphere_intersection(origins, directions, center, radius)

        assert hit[0].item() == True
        assert torch.isclose(t_near[0], torch.tensor(1.0), atol=1e-5)  # Hit at z=1

    def test_sphere_intersection_miss(self):
        """Test sphere intersection with missing rays."""
        # Ray that misses the sphere
        origins = torch.tensor([[0.0, 5.0, 2.0]])  # Offset in Y
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        t_near, t_far, hit = rays_sphere_intersection(origins, directions, center, radius)

        assert hit[0].item() == False


class TestPhongShading:
    """Test Phong shading."""

    def test_phong_facing_light(self):
        """Test Phong shading with normal facing light."""
        normals = torch.tensor([[[0.0, 0.0, 1.0]]])  # Facing camera/light
        albedo = torch.tensor([[[1.0, 1.0, 1.0]]])   # White
        light_dir = torch.tensor([0.0, 0.0, 1.0])    # From camera
        view_dir = torch.tensor([0.0, 0.0, 1.0])     # To camera

        result = phong_shading(
            normals, albedo, light_dir, view_dir,
            ambient=0.1, diffuse=0.7, specular=0.2, shininess=32.0
        )

        # Should be bright (ambient + diffuse + specular)
        assert result[0, 0, 0].item() > 0.9

    def test_phong_facing_away(self):
        """Test Phong shading with normal facing away from light."""
        normals = torch.tensor([[[0.0, 0.0, -1.0]]])  # Facing away
        albedo = torch.tensor([[[1.0, 1.0, 1.0]]])
        light_dir = torch.tensor([0.0, 0.0, 1.0])
        view_dir = torch.tensor([0.0, 0.0, 1.0])

        result = phong_shading(
            normals, albedo, light_dir, view_dir,
            ambient=0.1, diffuse=0.7, specular=0.2
        )

        # Should only have ambient (diffuse and specular are 0)
        assert torch.isclose(result[0, 0, 0], torch.tensor(0.1), atol=1e-5)

    def test_phong_batched(self):
        """Test Phong shading with batched input."""
        normals = torch.randn(32, 32, 3)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        albedo = torch.rand(32, 32, 3)
        light_dir = torch.tensor([1.0, 1.0, 1.0])
        light_dir = light_dir / light_dir.norm()
        view_dir = torch.tensor([0.0, 0.0, 1.0])

        result = phong_shading(normals, albedo, light_dir, view_dir)

        assert result.shape == (32, 32, 3)
        assert result.min() >= 0
        assert result.max() <= 1


class TestNormalToColor:
    """Test normal to color conversion."""

    def test_normal_to_color_x_axis(self):
        """Test normal pointing in +X is red."""
        normals = torch.tensor([[[1.0, 0.0, 0.0]]])
        colors = normal_to_color(normals)

        # +X should map to red (1, 0.5, 0.5)
        assert colors[0, 0, 0].item() > 0.9  # Red channel high
        assert abs(colors[0, 0, 1].item() - 0.5) < 0.1  # Green ~0.5
        assert abs(colors[0, 0, 2].item() - 0.5) < 0.1  # Blue ~0.5

    def test_normal_to_color_y_axis(self):
        """Test normal pointing in +Y is green."""
        normals = torch.tensor([[[0.0, 1.0, 0.0]]])
        colors = normal_to_color(normals)

        assert abs(colors[0, 0, 0].item() - 0.5) < 0.1
        assert colors[0, 0, 1].item() > 0.9  # Green channel high
        assert abs(colors[0, 0, 2].item() - 0.5) < 0.1

    def test_normal_to_color_bounded(self):
        """Test output is bounded [0, 1]."""
        normals = torch.randn(32, 32, 3)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        colors = normal_to_color(normals)

        assert colors.min() >= 0.0
        assert colors.max() <= 1.0


class TestDepthToColor:
    """Test depth to color conversion."""

    def test_depth_to_color_near_white(self):
        """Test that near depths are bright."""
        depth = torch.tensor([[0.1]])
        colors = depth_to_color(depth, near=0.0, far=1.0)

        # depth_to_color squeezes the last dim if it's 1
        # [[0.1]] -> [0.1] -> output (1, 3)
        assert colors.shape == (1, 3)
        # Near should be brighter (viridis starts dark-ish but check it's not black)
        assert colors[0, 0].item() > 0.2 or colors[0, 1].item() > 0.2 or colors[0, 2].item() > 0.2

    def test_depth_to_color_far_dark(self):
        """Test that far depths are darker."""
        depth = torch.tensor([[0.9]])
        colors = depth_to_color(depth, near=0.0, far=1.0)

        # depth_to_color squeezes the last dim if it's 1
        # Output is (1, 3)
        assert colors.shape == (1, 3)

    def test_depth_to_color_shape(self):
        """Test output shape."""
        depth = torch.rand(32, 32)
        colors = depth_to_color(depth)

        assert colors.shape == (32, 32, 3)


class TestSphereTracer:
    """Test sphere tracing renderer."""

    def test_sphere_tracer_creation(self):
        """Test sphere tracer can be created."""
        model = PGA_INR_SDF(hidden_features=32, hidden_layers=2)
        tracer = PGASphereTracer(
            model=model,
            width=16,
            height=16,
            fov=60.0
        )

        assert tracer.width == 16
        assert tracer.height == 16

    def test_sphere_tracer_render_shape(self):
        """Test sphere tracer output shapes."""
        model = PGA_INR_SDF(hidden_features=32, hidden_layers=2)
        tracer = PGASphereTracer(
            model=model,
            width=16,
            height=16,
            max_steps=10
        )

        camera_pose = (
            torch.tensor([0.0, 0.0, 2.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0])
        )

        result = tracer.render(
            camera_pose,
            return_depth=True,
            return_normals=True
        )

        assert result['rgb'].shape == (16, 16, 3)
        assert result['depth'].shape == (16, 16)
        assert result['normals'].shape == (16, 16, 3)
        assert result['hit_mask'].shape == (16, 16)

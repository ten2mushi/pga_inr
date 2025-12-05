"""
Tests for PGA-INR neural network models.
"""

import pytest
import torch
import torch.nn as nn
import math

from pga_inr.models import PGA_INR, PGA_INR_SDF
from pga_inr.models.generative import Generative_PGA_INR, Generative_PGA_INR_SDF, LatentCodeBank
from pga_inr.models.layers.sine import SineLayer, SirenMLP
from pga_inr.models.layers.motor import PGAMotorLayer
from pga_inr.models.layers.hyper import HyperNetwork, HyperLayer
from pga_inr.utils.quaternion import quaternion_from_axis_angle


class TestSineLayer:
    """Test SIREN layers."""

    def test_sine_layer_output_shape(self):
        """Test output shape is correct."""
        layer = SineLayer(in_features=3, out_features=64)
        x = torch.randn(2, 100, 3)
        y = layer(x)

        assert y.shape == (2, 100, 64)

    def test_sine_layer_bounded(self):
        """Test output is bounded by [-1, 1]."""
        layer = SineLayer(in_features=3, out_features=64)
        x = torch.randn(2, 100, 3)
        y = layer(x)

        assert y.max() <= 1.0
        assert y.min() >= -1.0

    def test_sine_layer_omega_0(self):
        """Test that omega_0 scales input."""
        layer_low = SineLayer(in_features=3, out_features=64, omega_0=1.0)
        layer_high = SineLayer(in_features=3, out_features=64, omega_0=30.0)

        # Copy weights to make them identical
        with torch.no_grad():
            layer_high.linear.weight.copy_(layer_low.linear.weight)
            layer_high.linear.bias.copy_(layer_low.linear.bias)

        x = torch.randn(2, 100, 3)
        y_low = layer_low(x)
        y_high = layer_high(x)

        # Higher omega should have more variation
        assert y_high.std() != y_low.std()


class TestSirenMLP:
    """Test SIREN MLP."""

    def test_siren_mlp_output_shape(self):
        """Test output shape."""
        mlp = SirenMLP(
            in_features=3,
            hidden_features=64,
            hidden_layers=3,
            out_features=1
        )
        x = torch.randn(2, 100, 3)
        y = mlp(x)

        assert y.shape == (2, 100, 1)

    def test_siren_mlp_gradient_flow(self):
        """Test gradients flow through network."""
        mlp = SirenMLP(
            in_features=3,
            hidden_features=32,
            hidden_layers=2,
            out_features=1
        )
        x = torch.randn(2, 100, 3, requires_grad=True)
        y = mlp(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestPGAMotorLayer:
    """Test PGA motor layer."""

    def test_motor_layer_identity(self):
        """Identity motor should not transform points."""
        layer = PGAMotorLayer()

        points = torch.randn(2, 100, 3)
        trans = torch.zeros(2, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        transformed = layer(points, (trans, quat))

        assert torch.allclose(transformed, points, atol=1e-5)

    def test_motor_layer_translation(self):
        """Test pure translation (world->local transform).

        If the object is at position [1,2,3] in world space, then the world
        origin (0,0,0) appears at position [-1,-2,-3] in the object's local frame.
        This is the inverse transformation: p_local = p_world - translation.
        """
        layer = PGAMotorLayer()

        points = torch.zeros(1, 10, 3)  # Points at world origin
        trans = torch.tensor([[1.0, 2.0, 3.0]])  # Object is at (1,2,3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # No rotation

        transformed = layer(points, (trans, quat))

        # In object's local frame, world origin is at -trans
        expected = -trans.unsqueeze(1).expand_as(points)
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_motor_layer_rotation(self):
        """Test rotation around Z axis (world->local transform).

        If the object is rotated +90° around Z axis, then a point at (1,0,0)
        in world space appears at (0,-1,0) in the object's local frame.
        This is the inverse rotation: R^{-1} @ p_world.

        Forward: R rotates (1,0,0) -> (0,1,0)
        Inverse: R^{-1} rotates (1,0,0) -> (0,-1,0)
        """
        layer = PGAMotorLayer()

        points = torch.tensor([[[1.0, 0.0, 0.0]]])  # Point on +X axis
        trans = torch.zeros(1, 3)  # No translation
        axis = torch.tensor([[0.0, 0.0, 1.0]])  # Rotate around Z
        angle = torch.tensor([[math.pi / 2]])  # 90 degrees
        quat = quaternion_from_axis_angle(axis, angle)

        transformed = layer(points, (trans, quat))

        # Inverse rotation: (1,0,0) -> (0,-1,0)
        expected = torch.tensor([[[0.0, -1.0, 0.0]]])
        assert torch.allclose(transformed, expected, atol=1e-5)


class TestPGA_INR:
    """Test basic PGA-INR model."""

    def test_pga_inr_output_shape(self):
        """Test output shapes are correct."""
        model = PGA_INR(
            hidden_features=64,
            hidden_layers=2,
            output_normals=True
        )

        batch_size = 2
        num_points = 100
        points = torch.randn(batch_size, num_points, 3)
        trans = torch.randn(batch_size, 3)
        quat = torch.randn(batch_size, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)

        outputs = model(points, (trans, quat))

        assert outputs['density'].shape == (batch_size, num_points, 1)
        assert outputs['rgb'].shape == (batch_size, num_points, 3)
        assert outputs['normal'].shape == (batch_size, num_points, 3)
        assert outputs['local_coords'].shape == (batch_size, num_points, 3)

    def test_pga_inr_no_observer(self):
        """Test model works without observer pose."""
        model = PGA_INR(hidden_features=64, hidden_layers=2)

        points = torch.randn(2, 100, 3)
        outputs = model(points, observer_pose=None)

        assert 'density' in outputs
        assert outputs['density'].shape == (2, 100, 1)

    def test_pga_inr_rgb_bounded(self):
        """Test RGB output is in [0, 1]."""
        model = PGA_INR(hidden_features=64, hidden_layers=2)

        points = torch.randn(2, 100, 3)
        outputs = model(points)

        rgb = outputs['rgb']
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0


class TestPGA_INR_SDF:
    """Test SDF-specialized PGA-INR."""

    def test_sdf_model_output_shape(self):
        """Test SDF model outputs."""
        model = PGA_INR_SDF(
            hidden_features=64,
            hidden_layers=2
        )

        points = torch.randn(2, 100, 3)
        outputs = model(points)

        assert outputs['sdf'].shape == (2, 100, 1)
        assert outputs['normal'].shape == (2, 100, 3)

    def test_sdf_geometric_init(self):
        """Test geometric initialization approximates unit sphere."""
        model = PGA_INR_SDF(
            hidden_features=256,
            hidden_layers=4,
            geometric_init=True
        )

        # Test at origin (should be ~-1 for unit sphere)
        origin = torch.tensor([[[0.0, 0.0, 0.0]]])
        sdf_origin = model(origin)['sdf'].item()
        assert abs(sdf_origin - (-1.0)) < 0.3  # Within 0.3 of expected

        # Test at surface (should be ~0)
        surface = torch.tensor([[[1.0, 0.0, 0.0]]])
        sdf_surface = model(surface)['sdf'].item()
        assert abs(sdf_surface) < 0.3


class TestHyperNetwork:
    """Test HyperNetwork layers."""

    def test_hyper_network_output_shapes(self):
        """Test HyperNetwork generates correct weight shapes."""
        target_shapes = [(32, 3), (32, 32), (1, 32)]

        hyper = HyperNetwork(
            latent_dim=16,
            target_shapes=target_shapes,
            hidden_dim=64
        )

        z = torch.randn(2, 16)
        weights, biases = hyper(z)

        assert len(weights) == 3
        assert len(biases) == 3

        # HyperNetwork returns weights of shape (B, out_dim, in_dim)
        # target_shapes = [(fan_in, fan_out), ...]
        # First layer: fan_in=32, fan_out=3 -> weight shape (2, 3, 32)
        assert weights[0].shape == (2, 3, 32)
        # Second layer: fan_in=32, fan_out=32 -> weight shape (2, 32, 32)
        assert weights[1].shape == (2, 32, 32)
        # Third layer: fan_in=1, fan_out=32 -> weight shape (2, 32, 1)
        assert weights[2].shape == (2, 32, 1)

        assert biases[0].shape == (2, 3)
        assert biases[1].shape == (2, 32)
        assert biases[2].shape == (2, 32)

    def test_hyper_layer(self):
        """Test HyperLayer applies weights correctly."""
        layer = HyperLayer(activation=torch.sin)

        x = torch.randn(2, 100, 8)
        weights = torch.randn(2, 16, 8)
        bias = torch.randn(2, 16)

        y = layer(x, weights, bias)

        assert y.shape == (2, 100, 16)


class TestGenerativePGA_INR:
    """Test generative PGA-INR models."""

    def test_generative_output_shape(self):
        """Test generative model output."""
        model = Generative_PGA_INR(
            latent_dim=16,
            hidden_features=32,
            hidden_layers=2
        )

        batch_size = 2
        num_points = 100
        points = torch.randn(batch_size, num_points, 3)
        z = torch.randn(batch_size, 16)

        outputs = model(points, observer_pose=None, latent_code=z)

        assert outputs.shape == (batch_size, num_points, 4)  # density + RGB

    def test_generative_sdf_output_shape(self):
        """Test generative SDF model."""
        model = Generative_PGA_INR_SDF(
            latent_dim=16,
            hidden_features=32,
            hidden_layers=2
        )

        points = torch.randn(2, 100, 3)
        z = torch.randn(2, 16)

        outputs = model.forward_dict(points, observer_pose=None, latent_code=z)

        assert outputs['sdf'].shape == (2, 100, 1)

    def test_latent_code_bank(self):
        """Test latent code bank."""
        bank = LatentCodeBank(num_objects=10, latent_dim=16)

        indices = torch.tensor([0, 5, 9])
        codes = bank(indices)

        assert codes.shape == (3, 16)

        # Same index should give same code
        code1 = bank(torch.tensor([0]))
        code2 = bank(torch.tensor([0]))
        assert torch.allclose(code1, code2)


class TestObserverInvariance:
    """Test observer-independence property."""

    def test_same_local_from_different_observers(self):
        """Different observers should get different local coords for same world point."""
        model = PGA_INR(hidden_features=64, hidden_layers=2)

        # Same world point
        world_point = torch.tensor([[[0.5, 0.5, 0.5]]])
        world_point = world_point.expand(2, -1, -1)

        # Different observer poses
        trans = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        quat = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ])

        outputs = model(world_point, (trans, quat))
        local_coords = outputs['local_coords']

        # Local coords should be different for different observers
        assert not torch.allclose(local_coords[0], local_coords[1])

    def test_identity_observer_preserves_coords(self):
        """Identity observer should preserve world coordinates."""
        model = PGA_INR(hidden_features=64, hidden_layers=2)

        points = torch.randn(1, 100, 3)
        trans = torch.zeros(1, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        outputs = model(points, (trans, quat))

        # With identity observer, local coords should equal world coords
        assert torch.allclose(outputs['local_coords'], points, atol=1e-5)


class TestGradientComputation:
    """Test gradient computation for losses."""

    def test_sdf_gradient_computation(self):
        """Test computing SDF gradients for Eikonal loss."""
        model = PGA_INR_SDF(hidden_features=64, hidden_layers=2)

        points = torch.randn(2, 100, 3, requires_grad=True)
        outputs = model(points)
        sdf = outputs['sdf']

        # Compute gradient
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True
        )[0]

        assert grad.shape == (2, 100, 3)
        assert not torch.isnan(grad).any()

    def test_eikonal_property(self):
        """Test Eikonal property |∇f| ≈ 1 for trained SDF."""
        # Use geometric init which approximates unit sphere SDF
        model = PGA_INR_SDF(
            hidden_features=256,
            hidden_layers=4,
            geometric_init=True
        )

        points = torch.randn(1, 1000, 3, requires_grad=True)
        outputs = model(points)
        sdf = outputs['sdf']

        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=False
        )[0]

        grad_norm = grad.norm(dim=-1)

        # With geometric init, gradient norm should be close to 1
        mean_norm = grad_norm.mean().item()
        assert 0.5 < mean_norm < 1.5  # Reasonable range

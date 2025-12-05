"""
Tests for geometric loss functions.
"""

import pytest
import torch
import torch.nn as nn
import math

from pga_inr.losses import (
    EikonalLoss,
    NormalAlignmentLoss,
    SDFLoss,
    GeometricConsistencyLoss,
)
from pga_inr.losses.regularization import (
    LatentRegularization,
    WeightDecay,
    LipschitzRegularization,
)


class TestEikonalLoss:
    """Test Eikonal loss."""

    def test_eikonal_perfect_gradient(self):
        """Test that unit gradient gives zero loss."""
        loss_fn = EikonalLoss()

        # Create mock SDF and coordinates
        coords = torch.randn(2, 100, 3, requires_grad=True)

        # Create SDF that has unit gradient (f(x) = x[..., 0])
        sdf = coords[..., 0:1]  # Just x coordinate

        loss = loss_fn(sdf, coords)

        # Gradient is (1, 0, 0), norm = 1, so loss should be 0
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_eikonal_non_unit_gradient(self):
        """Test that non-unit gradient gives positive loss."""
        loss_fn = EikonalLoss()

        coords = torch.randn(2, 100, 3, requires_grad=True)

        # Create SDF with gradient magnitude 2 (f(x) = 2*x[0])
        sdf = 2 * coords[..., 0:1]

        loss = loss_fn(sdf, coords)

        # Gradient norm is 2, loss = (2-1)^2 = 1
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-4)

    def test_eikonal_reduction(self):
        """Test different reduction modes."""
        coords = torch.randn(2, 100, 3, requires_grad=True)
        sdf = coords[..., 0:1]

        loss_mean = EikonalLoss(reduction='mean')
        loss_sum = EikonalLoss(reduction='sum')
        loss_none = EikonalLoss(reduction='none')

        l_mean = loss_mean(sdf, coords)
        l_sum = loss_sum(sdf, coords)
        l_none = loss_none(sdf, coords)

        assert l_mean.shape == ()  # Scalar
        assert l_sum.shape == ()
        assert l_none.shape == (2, 100)


class TestNormalAlignmentLoss:
    """Test normal alignment loss."""

    def test_aligned_normals(self):
        """Test that aligned normals give zero loss."""
        loss_fn = NormalAlignmentLoss()

        # Predicted normals
        pred_normals = torch.tensor([
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]]
        ])

        # Coordinates for gradient computation
        coords = torch.randn(2, 1, 3, requires_grad=True)

        # SDF whose gradient matches pred_normals
        # For simplicity, use a SDF that gives correct gradient
        sdf = (coords * pred_normals).sum(dim=-1, keepdim=True)

        loss = loss_fn(pred_normals, sdf, coords)

        # Should be close to 0 for aligned normals
        assert loss.item() < 0.1

    def test_misaligned_normals(self):
        """Test that misaligned normals give positive loss."""
        loss_fn = NormalAlignmentLoss()

        # Predicted normals pointing in X direction
        pred_normals = torch.tensor([[[1.0, 0.0, 0.0]]])

        coords = torch.randn(1, 1, 3, requires_grad=True)

        # SDF with gradient in Y direction
        sdf = coords[..., 1:2]

        loss = loss_fn(pred_normals, sdf, coords)

        # Should be positive (1 - cos(90°) = 1)
        assert loss.item() > 0.5


class TestSDFLoss:
    """Test SDF reconstruction loss."""

    def test_sdf_l1_loss(self):
        """Test L1 SDF loss."""
        loss_fn = SDFLoss(loss_type='l1')

        pred = torch.tensor([[[0.5], [0.3]]])
        target = torch.tensor([[[0.5], [0.1]]])

        loss = loss_fn(pred, target)

        expected = torch.tensor(0.1)  # (0 + 0.2) / 2
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_sdf_l2_loss(self):
        """Test L2 SDF loss."""
        loss_fn = SDFLoss(loss_type='l2')

        pred = torch.tensor([[[1.0], [0.0]]])
        target = torch.tensor([[[0.0], [0.0]]])

        loss = loss_fn(pred, target)

        expected = torch.tensor(0.5)  # (1 + 0) / 2
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_sdf_clamped_loss(self):
        """Test clamped SDF loss."""
        loss_fn = SDFLoss(loss_type='l1', clamp=0.5)

        # Values outside clamp range should be clamped
        pred = torch.tensor([[[2.0], [-2.0]]])
        target = torch.tensor([[[0.5], [-0.5]]])

        loss = loss_fn(pred, target)

        # Both should be clamped to ±0.5, so loss is 0
        assert loss.item() < 0.01


class TestGeometricConsistencyLoss:
    """Test combined geometric loss."""

    def test_combined_loss_components(self):
        """Test that combined loss returns all components."""
        loss_fn = GeometricConsistencyLoss(
            lambda_sdf=1.0,
            lambda_eikonal=0.1,
            lambda_align=0.05,
            lambda_normal=1.0
        )

        batch_size = 2
        num_points = 100

        # Create mock outputs - local_coords must be in outputs dict
        local_coords = torch.randn(batch_size, num_points, 3, requires_grad=True)
        outputs = {
            'sdf': local_coords[..., 0:1],  # Simple SDF
            'normal': torch.randn(batch_size, num_points, 3),
            'local_coords': local_coords
        }
        outputs['normal'] = outputs['normal'] / outputs['normal'].norm(dim=-1, keepdim=True)

        gt_sdf = torch.randn(batch_size, num_points, 1)
        gt_normals = torch.randn(batch_size, num_points, 3)
        gt_normals = gt_normals / gt_normals.norm(dim=-1, keepdim=True)

        total_loss, metrics = loss_fn(outputs, gt_sdf, gt_normals)

        # Actual API returns keys like 'sdf', 'eikonal', 'align', 'normal'
        assert 'sdf' in metrics
        assert 'eikonal' in metrics
        assert 'align' in metrics
        assert 'normal' in metrics

        assert total_loss.requires_grad

    def test_combined_loss_without_normals(self):
        """Test combined loss without ground truth normals."""
        loss_fn = GeometricConsistencyLoss(lambda_normal=0.0)

        local_coords = torch.randn(2, 100, 3, requires_grad=True)
        outputs = {
            'sdf': local_coords[..., 0:1],
            'normal': torch.randn(2, 100, 3),
            'local_coords': local_coords
        }

        gt_sdf = torch.randn(2, 100, 1)

        total_loss, metrics = loss_fn(outputs, gt_sdf, gt_normals=None)

        # No normal key when gt_normals is None
        assert 'normal' not in metrics or metrics.get('normal', 0.0) == 0.0
        assert total_loss.item() > 0


class TestLatentRegularization:
    """Test latent code regularization."""

    def test_l2_regularization(self):
        """Test L2 regularization."""
        reg = LatentRegularization(regularization_type='l2')

        z = torch.tensor([[1.0, 0.0, 0.0]])
        loss = reg(z)

        # Implementation uses sum(dim=-1) then mean over batch
        # For [[1, 0, 0]], sum of squares is 1+0+0 = 1
        expected = torch.tensor(1.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_l1_regularization(self):
        """Test L1 regularization."""
        reg = LatentRegularization(regularization_type='l1')

        z = torch.tensor([[1.0, -1.0, 0.5]])
        loss = reg(z)

        # Implementation uses sum(dim=-1) then mean over batch
        # For [[1, -1, 0.5]], sum of abs is 1+1+0.5 = 2.5
        expected = torch.tensor(2.5)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_kl_regularization(self):
        """Test KL divergence regularization (VAE-style)."""
        reg = LatentRegularization(regularization_type='kl')

        # For standard normal, mu=0, logvar=0 gives 0 KL
        z = torch.zeros(1, 10)
        loss = reg(z)

        # KL of N(0,1) with N(0,1) is 0
        assert loss.item() < 0.5  # Should be small


class TestWeightDecay:
    """Test weight decay regularization."""

    def test_weight_decay(self):
        """Test weight decay computation."""
        reg = WeightDecay(decay_rate=0.01)

        # Create simple model
        model = nn.Linear(10, 5)

        loss = reg(model)

        # Should be positive
        assert loss.item() > 0

    def test_weight_decay_zero_weights(self):
        """Test weight decay with zero weights."""
        reg = WeightDecay(decay_rate=0.01)

        model = nn.Linear(10, 5)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()

        loss = reg(model)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


class TestLipschitzRegularization:
    """Test Lipschitz regularization."""

    def test_lipschitz_small_gradient(self):
        """Test Lipschitz regularization with small gradient."""
        lip = LipschitzRegularization(target_lipschitz=1.0)

        x = torch.randn(10, 3, requires_grad=True)
        y = 0.5 * x[..., 0:1]  # Gradient norm = 0.5 < 1

        loss = lip(y, x)

        # Should be 0 since gradient norm < max
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_lipschitz_large_gradient(self):
        """Test Lipschitz regularization with large gradient."""
        lip = LipschitzRegularization(target_lipschitz=1.0)

        x = torch.randn(10, 3, requires_grad=True)
        y = 2 * x[..., 0:1]  # Gradient norm = 2 > 1

        loss = lip(y, x)

        # Should be positive
        assert loss.item() > 0

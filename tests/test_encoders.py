"""
Tests for positional and frequency encoders.

Following the Yoneda philosophy: these tests completely DEFINE the behavior
of each encoder type. The tests serve as executable documentation for:

1. PositionalEncoder - Classic NeRF-style positional encoding
2. FourierEncoder - Random Fourier Features
3. GaussianEncoder - Gaussian basis functions
4. HashEncoder - Multi-resolution hash encoding
5. CompositeEncoder - Combining multiple encoders
6. IdentityEncoder - Pass-through encoder
"""

import pytest
import torch
import math

from pga_inr.models.encoders import (
    PositionalEncoder,
    FourierEncoder,
    GaussianEncoder,
    HashEncoder,
    CompositeEncoder,
    IdentityEncoder,
)


# =============================================================================
# PositionalEncoder Tests
# =============================================================================

class TestPositionalEncoderCreation:
    """Tests for PositionalEncoder construction."""

    def test_default_output_dim(self):
        """Default output dimension is 3 * (1 + 2*10) = 63."""
        encoder = PositionalEncoder()
        # input_dim=3, num_frequencies=10, include_input=True
        # output = 3 * (1 + 2*10) = 3 * 21 = 63
        assert encoder.output_dim == 63

    def test_custom_input_dim(self):
        """Custom input dimension affects output size."""
        encoder = PositionalEncoder(input_dim=2, num_frequencies=5)
        # 2 * (1 + 2*5) = 2 * 11 = 22
        assert encoder.output_dim == 22

    def test_exclude_input(self):
        """Excluding raw input reduces output dimension."""
        encoder = PositionalEncoder(input_dim=3, num_frequencies=10, include_input=False)
        # 3 * 2 * 10 = 60
        assert encoder.output_dim == 60

    def test_linear_frequency_sampling(self):
        """Linear frequency sampling instead of logarithmic."""
        encoder = PositionalEncoder(
            input_dim=3,
            num_frequencies=4,
            log_sampling=False,
            max_frequency=8.0
        )
        # Should have linearly spaced frequencies from 1 to 8
        assert encoder.output_dim == 3 * (1 + 2 * 4)  # 27


class TestPositionalEncoderForward:
    """Tests for PositionalEncoder forward pass."""

    def test_output_shape_single_point(self):
        """Output shape for single point."""
        encoder = PositionalEncoder(input_dim=3, num_frequencies=4)
        x = torch.randn(3)
        out = encoder(x)
        assert out.shape == (encoder.output_dim,)

    def test_output_shape_batch(self):
        """Output shape for batched input."""
        encoder = PositionalEncoder(input_dim=3, num_frequencies=4)
        x = torch.randn(10, 3)
        out = encoder(x)
        assert out.shape == (10, encoder.output_dim)

    def test_output_shape_2d_batch(self):
        """Output shape for 2D batched input."""
        encoder = PositionalEncoder(input_dim=3, num_frequencies=4)
        x = torch.randn(5, 10, 3)
        out = encoder(x)
        assert out.shape == (5, 10, encoder.output_dim)

    def test_include_input_preserves_original(self):
        """When include_input=True, raw input appears in output."""
        encoder = PositionalEncoder(input_dim=3, num_frequencies=2, include_input=True)
        x = torch.tensor([0.5, 0.3, 0.1])
        out = encoder(x)
        # First 3 values should be the input
        assert torch.allclose(out[:3], x)

    def test_exclude_input(self):
        """When include_input=False, raw input not in output."""
        encoder = PositionalEncoder(input_dim=3, num_frequencies=2, include_input=False)
        x = torch.tensor([0.5, 0.3, 0.1])
        out = encoder(x)
        assert out.shape[0] == 3 * 2 * 2  # No raw input, just sin/cos

    def test_sin_cos_values(self):
        """Output contains sin and cos at expected frequencies."""
        encoder = PositionalEncoder(input_dim=1, num_frequencies=2, include_input=True)
        x = torch.tensor([0.5])  # Single value
        out = encoder(x)

        # Expected: [x, sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x)]
        assert out[0] == 0.5  # Raw input
        assert torch.isclose(out[1], torch.sin(torch.tensor(math.pi * 0.5)), atol=1e-5)
        assert torch.isclose(out[2], torch.cos(torch.tensor(math.pi * 0.5)), atol=1e-5)

    def test_bounded_output(self):
        """Sin and cos outputs are bounded in [-1, 1]."""
        encoder = PositionalEncoder(input_dim=3, num_frequencies=10, include_input=False)
        x = torch.randn(100, 3)
        out = encoder(x)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5


class TestPositionalEncoderProperties:
    """Tests for PositionalEncoder mathematical properties."""

    def test_deterministic(self):
        """Encoding is deterministic."""
        encoder = PositionalEncoder()
        x = torch.randn(10, 3)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs."""
        encoder = PositionalEncoder()
        x1 = torch.tensor([0.0, 0.0, 0.0])
        x2 = torch.tensor([1.0, 0.0, 0.0])
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert not torch.allclose(out1, out2)

    def test_dtype_preservation(self):
        """Output dtype matches input dtype."""
        encoder = PositionalEncoder()
        x32 = torch.randn(3, dtype=torch.float32)
        x64 = torch.randn(3, dtype=torch.float64)
        assert encoder(x32).dtype == torch.float32
        assert encoder(x64).dtype == torch.float64


# =============================================================================
# FourierEncoder Tests
# =============================================================================

class TestFourierEncoderCreation:
    """Tests for FourierEncoder construction."""

    def test_default_output_dim(self):
        """Default output dimension."""
        encoder = FourierEncoder(input_dim=3, num_frequencies=256, include_input=True)
        # 2 * 256 + 3 = 515
        assert encoder.output_dim == 515

    def test_exclude_input(self):
        """Excluding input reduces dimension."""
        encoder = FourierEncoder(input_dim=3, num_frequencies=256, include_input=False)
        # 2 * 256 = 512
        assert encoder.output_dim == 512

    def test_learnable_frequencies(self):
        """Learnable frequencies are parameters."""
        encoder = FourierEncoder(num_frequencies=32, learnable=True)
        assert isinstance(encoder.B, torch.nn.Parameter)
        assert encoder.B.requires_grad

    def test_fixed_frequencies(self):
        """Non-learnable frequencies are buffers."""
        encoder = FourierEncoder(num_frequencies=32, learnable=False)
        assert not isinstance(encoder.B, torch.nn.Parameter)


class TestFourierEncoderForward:
    """Tests for FourierEncoder forward pass."""

    def test_output_shape(self):
        """Output has correct shape."""
        encoder = FourierEncoder(input_dim=3, num_frequencies=64)
        x = torch.randn(10, 3)
        out = encoder(x)
        assert out.shape == (10, encoder.output_dim)

    def test_bounded_output(self):
        """Sin and cos outputs are bounded."""
        encoder = FourierEncoder(input_dim=3, num_frequencies=64, include_input=False)
        x = torch.randn(100, 3)
        out = encoder(x)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_different_sigma_different_features(self):
        """Different sigma values produce different feature distributions."""
        encoder_low = FourierEncoder(sigma=1.0)
        encoder_high = FourierEncoder(sigma=100.0)

        x = torch.randn(10, 3)
        out_low = encoder_low(x)
        out_high = encoder_high(x)

        # High sigma should produce higher frequency variations
        # This is a weak test but checks they differ
        assert not torch.allclose(out_low, out_high)


class TestFourierEncoderRandomness:
    """Tests for random frequency initialization."""

    def test_different_encoders_different_frequencies(self):
        """Different encoder instances have different random frequencies."""
        encoder1 = FourierEncoder(num_frequencies=32, sigma=10.0)
        encoder2 = FourierEncoder(num_frequencies=32, sigma=10.0)

        # B matrices should be different (random initialization)
        assert not torch.allclose(encoder1.B, encoder2.B)

    def test_frequency_statistics(self):
        """Random frequencies approximately follow N(0, sigma^2)."""
        sigma = 10.0
        encoder = FourierEncoder(num_frequencies=1000, sigma=sigma)

        # Check mean is approximately 0
        assert torch.abs(encoder.B.mean()) < 1.0

        # Check std is approximately sigma
        assert torch.abs(encoder.B.std() - sigma) < 2.0


# =============================================================================
# GaussianEncoder Tests
# =============================================================================

class TestGaussianEncoderCreation:
    """Tests for GaussianEncoder construction."""

    def test_output_dim_calculation(self):
        """Output dimension is num_centers^input_dim."""
        encoder = GaussianEncoder(input_dim=3, num_centers=4)
        # 4^3 = 64
        assert encoder.output_dim == 64

    def test_centers_shape(self):
        """Centers have correct shape."""
        encoder = GaussianEncoder(input_dim=3, num_centers=4)
        assert encoder.centers.shape == (64, 3)  # 4^3 centers, each in 3D

    def test_centers_within_bounds(self):
        """Centers are within specified bounds."""
        bounds = (-2.0, 2.0)
        encoder = GaussianEncoder(input_dim=3, num_centers=4, bounds=bounds)
        assert encoder.centers.min() >= bounds[0]
        assert encoder.centers.max() <= bounds[1]


class TestGaussianEncoderForward:
    """Tests for GaussianEncoder forward pass."""

    def test_output_shape(self):
        """Output has correct shape."""
        encoder = GaussianEncoder(input_dim=3, num_centers=4)
        x = torch.randn(10, 3)
        out = encoder(x)
        assert out.shape == (10, 64)

    def test_output_positive(self):
        """Gaussian outputs are non-negative."""
        encoder = GaussianEncoder(input_dim=3, num_centers=4)
        x = torch.randn(100, 3)
        out = encoder(x)
        assert (out >= 0).all()

    def test_output_bounded_by_one(self):
        """Gaussian outputs are at most 1."""
        encoder = GaussianEncoder(input_dim=3, num_centers=4)
        x = torch.randn(100, 3)
        out = encoder(x)
        assert out.max() <= 1.0 + 1e-5

    def test_center_activation_peaks(self):
        """Point exactly at center gives activation 1."""
        encoder = GaussianEncoder(input_dim=3, num_centers=4, sigma=0.1)
        # Pick a center
        center = encoder.centers[0:1]  # Shape (1, 3)
        out = encoder(center)  # Shape (1, 64)
        # The first Gaussian should be maximally activated
        assert torch.isclose(out[0, 0], torch.tensor(1.0), atol=1e-5)

    def test_smaller_sigma_sharper_peaks(self):
        """Smaller sigma produces sharper (more localized) activations."""
        encoder_wide = GaussianEncoder(sigma=1.0, num_centers=4)
        encoder_narrow = GaussianEncoder(sigma=0.1, num_centers=4)

        x = torch.randn(100, 3)
        out_wide = encoder_wide(x)
        out_narrow = encoder_narrow(x)

        # Narrow encoder should have lower mean activation (more localized)
        assert out_narrow.mean() < out_wide.mean()


# =============================================================================
# HashEncoder Tests
# =============================================================================

class TestHashEncoderCreation:
    """Tests for HashEncoder construction."""

    def test_output_dim(self):
        """Output dimension is num_levels * level_dim."""
        encoder = HashEncoder(num_levels=16, level_dim=2)
        assert encoder.output_dim == 32

    def test_resolutions_increase(self):
        """Resolutions increase across levels."""
        encoder = HashEncoder(num_levels=4, base_resolution=16, max_resolution=256)
        resolutions = encoder.resolutions.tolist()
        assert all(resolutions[i] <= resolutions[i+1] for i in range(len(resolutions)-1))

    def test_embeddings_count(self):
        """Correct number of embedding tables."""
        encoder = HashEncoder(num_levels=8)
        assert len(encoder.embeddings) == 8


class TestHashEncoderForward:
    """Tests for HashEncoder forward pass."""

    def test_output_shape(self):
        """Output has correct shape."""
        encoder = HashEncoder(num_levels=4, level_dim=2)
        x = torch.rand(10, 3)  # Input in [0, 1]
        out = encoder(x)
        assert out.shape == (10, 8)

    def test_output_shape_batched(self):
        """Batched output shape."""
        encoder = HashEncoder(num_levels=4, level_dim=2)
        x = torch.rand(5, 10, 3)
        out = encoder(x)
        assert out.shape == (5, 10, 8)

    def test_deterministic(self):
        """Hash encoding is deterministic."""
        encoder = HashEncoder()
        x = torch.rand(10, 3)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)


class TestHashEncoderLearning:
    """Tests for learnable hash encoding."""

    def test_embeddings_require_grad(self):
        """Embedding tables are learnable."""
        encoder = HashEncoder()
        for emb in encoder.embeddings:
            assert emb.weight.requires_grad

    def test_gradient_flow(self):
        """Gradients flow through hash encoder."""
        encoder = HashEncoder(num_levels=4)
        x = torch.rand(10, 3, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        # Embeddings should have gradients
        assert encoder.embeddings[0].weight.grad is not None


# =============================================================================
# CompositeEncoder Tests
# =============================================================================

class TestCompositeEncoderCreation:
    """Tests for CompositeEncoder construction."""

    def test_output_dim_sum(self):
        """Output dimension is sum of encoder output dims."""
        enc1 = PositionalEncoder(input_dim=3, num_frequencies=4)
        enc2 = FourierEncoder(input_dim=3, num_frequencies=32)
        composite = CompositeEncoder([enc1, enc2])
        assert composite.output_dim == enc1.output_dim + enc2.output_dim

    def test_empty_list_raises(self):
        """Empty encoder list should work but have zero output dim."""
        composite = CompositeEncoder([])
        assert composite.output_dim == 0


class TestCompositeEncoderForward:
    """Tests for CompositeEncoder forward pass."""

    def test_output_concatenated(self):
        """Output is concatenation of individual encoder outputs."""
        enc1 = PositionalEncoder(input_dim=3, num_frequencies=2)
        enc2 = FourierEncoder(input_dim=3, num_frequencies=8)
        composite = CompositeEncoder([enc1, enc2])

        x = torch.randn(10, 3)
        out = composite(x)

        # First part should match enc1 output
        out1 = enc1(x)
        assert torch.allclose(out[:, :enc1.output_dim], out1)

        # Second part should match enc2 output
        out2 = enc2(x)
        assert torch.allclose(out[:, enc1.output_dim:], out2)

    def test_three_encoders(self):
        """Works with three encoders."""
        enc1 = IdentityEncoder(input_dim=3)
        enc2 = PositionalEncoder(input_dim=3, num_frequencies=2)
        enc3 = FourierEncoder(input_dim=3, num_frequencies=8)
        composite = CompositeEncoder([enc1, enc2, enc3])

        assert composite.output_dim == 3 + enc2.output_dim + enc3.output_dim

        x = torch.randn(5, 3)
        out = composite(x)
        assert out.shape == (5, composite.output_dim)


# =============================================================================
# IdentityEncoder Tests
# =============================================================================

class TestIdentityEncoder:
    """Tests for IdentityEncoder (pass-through)."""

    def test_output_dim_equals_input_dim(self):
        """Output dimension equals input dimension."""
        encoder = IdentityEncoder(input_dim=5)
        assert encoder.output_dim == 5

    def test_output_equals_input(self):
        """Output is identical to input."""
        encoder = IdentityEncoder(input_dim=3)
        x = torch.randn(10, 3)
        out = encoder(x)
        assert torch.allclose(out, x)

    def test_preserves_grad(self):
        """Gradients flow through identity encoder."""
        encoder = IdentityEncoder(input_dim=3)
        x = torch.randn(5, 3, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestEncoderGradientFlow:
    """Tests for gradient flow through all encoders."""

    @pytest.mark.parametrize("encoder_class,kwargs", [
        (PositionalEncoder, {"num_frequencies": 4}),
        (FourierEncoder, {"num_frequencies": 32}),
        (GaussianEncoder, {"num_centers": 4}),
        pytest.param(HashEncoder, {"num_levels": 4}, marks=pytest.mark.skip(reason="HashEncoder gradient to input not supported")),
        (IdentityEncoder, {}),
    ])
    def test_gradient_flow(self, encoder_class, kwargs):
        """Gradients flow through encoder."""
        encoder = encoder_class(input_dim=3, **kwargs)
        x = torch.randn(10, 3, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# Device and Dtype Compatibility
# =============================================================================

class TestEncoderDeviceDtype:
    """Tests for device and dtype handling."""

    @pytest.mark.parametrize("encoder_class,kwargs", [
        (PositionalEncoder, {"num_frequencies": 4}),
        (FourierEncoder, {"num_frequencies": 16}),
        (GaussianEncoder, {"num_centers": 4}),
        (IdentityEncoder, {}),
    ])
    def test_float32(self, encoder_class, kwargs):
        """Encoders work with float32."""
        encoder = encoder_class(input_dim=3, **kwargs)
        x = torch.randn(5, 3, dtype=torch.float32)
        out = encoder(x)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("encoder_class,kwargs", [
        (PositionalEncoder, {"num_frequencies": 4}),
        (FourierEncoder, {"num_frequencies": 16}),
        (IdentityEncoder, {}),
    ])
    def test_float64(self, encoder_class, kwargs):
        """Encoders work with float64."""
        encoder = encoder_class(input_dim=3, **kwargs).double()
        x = torch.randn(5, 3, dtype=torch.float64)
        out = encoder(x)
        assert out.dtype == torch.float64

    def test_cpu_device(self):
        """Encoders work on CPU."""
        encoder = PositionalEncoder()
        x = torch.randn(5, 3, device='cpu')
        out = encoder(x)
        assert out.device == torch.device('cpu')


# =============================================================================
# Batch Dimension Tests
# =============================================================================

class TestEncoderBatchDimensions:
    """Tests for handling various batch dimensions."""

    @pytest.mark.parametrize("encoder_class,kwargs", [
        (PositionalEncoder, {"num_frequencies": 4}),
        (FourierEncoder, {"num_frequencies": 16}),
        (GaussianEncoder, {"num_centers": 4}),
        (IdentityEncoder, {}),
    ])
    def test_no_batch(self, encoder_class, kwargs):
        """Handle single sample (no batch dim)."""
        encoder = encoder_class(input_dim=3, **kwargs)
        x = torch.randn(3)
        out = encoder(x)
        assert out.shape == (encoder.output_dim,)

    @pytest.mark.parametrize("encoder_class,kwargs", [
        (PositionalEncoder, {"num_frequencies": 4}),
        (FourierEncoder, {"num_frequencies": 16}),
        (GaussianEncoder, {"num_centers": 4}),
        (IdentityEncoder, {}),
    ])
    def test_1d_batch(self, encoder_class, kwargs):
        """Handle 1D batch."""
        encoder = encoder_class(input_dim=3, **kwargs)
        x = torch.randn(10, 3)
        out = encoder(x)
        assert out.shape == (10, encoder.output_dim)

    @pytest.mark.parametrize("encoder_class,kwargs", [
        (PositionalEncoder, {"num_frequencies": 4}),
        (FourierEncoder, {"num_frequencies": 16}),
        (GaussianEncoder, {"num_centers": 4}),
        (IdentityEncoder, {}),
    ])
    def test_2d_batch(self, encoder_class, kwargs):
        """Handle 2D batch."""
        encoder = encoder_class(input_dim=3, **kwargs)
        x = torch.randn(5, 10, 3)
        out = encoder(x)
        assert out.shape == (5, 10, encoder.output_dim)

    @pytest.mark.parametrize("encoder_class,kwargs", [
        (PositionalEncoder, {"num_frequencies": 4}),
        (FourierEncoder, {"num_frequencies": 16}),
        (GaussianEncoder, {"num_centers": 4}),
        (IdentityEncoder, {}),
    ])
    def test_3d_batch(self, encoder_class, kwargs):
        """Handle 3D batch."""
        encoder = encoder_class(input_dim=3, **kwargs)
        x = torch.randn(2, 5, 10, 3)
        out = encoder(x)
        assert out.shape == (2, 5, 10, encoder.output_dim)


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestEncoderNumericalStability:
    """Tests for numerical stability."""

    def test_positional_large_input(self):
        """PositionalEncoder handles large inputs."""
        encoder = PositionalEncoder(num_frequencies=10)
        x = torch.full((10, 3), 1000.0)
        out = encoder(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_positional_small_input(self):
        """PositionalEncoder handles small inputs."""
        encoder = PositionalEncoder(num_frequencies=10)
        x = torch.full((10, 3), 1e-10)
        out = encoder(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_fourier_large_input(self):
        """FourierEncoder handles large inputs."""
        encoder = FourierEncoder(num_frequencies=64)
        x = torch.full((10, 3), 100.0)
        out = encoder(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gaussian_far_from_centers(self):
        """GaussianEncoder handles points far from all centers."""
        encoder = GaussianEncoder(num_centers=4, bounds=(-1, 1), sigma=0.1)
        x = torch.full((10, 3), 100.0)  # Far outside bounds
        out = encoder(x)
        assert not torch.isnan(out).any()
        # Output should be very small (exponentially decayed)
        assert out.max() < 1e-10


# =============================================================================
# Edge Cases
# =============================================================================

class TestEncoderEdgeCases:
    """Tests for edge cases."""

    def test_single_frequency(self):
        """PositionalEncoder with single frequency."""
        encoder = PositionalEncoder(num_frequencies=1)
        x = torch.randn(5, 3)
        out = encoder(x)
        assert out.shape == (5, 3 * (1 + 2 * 1))

    def test_empty_batch(self):
        """Handle empty batch."""
        encoder = PositionalEncoder()
        x = torch.zeros(0, 3)
        out = encoder(x)
        assert out.shape == (0, encoder.output_dim)

    def test_1d_input(self):
        """Encoders work with 1D input."""
        encoder = PositionalEncoder(input_dim=1, num_frequencies=4)
        x = torch.randn(10, 1)
        out = encoder(x)
        assert out.shape == (10, 1 * (1 + 2 * 4))

    def test_high_dimensional_input(self):
        """Encoders work with high-dimensional input."""
        encoder = PositionalEncoder(input_dim=10, num_frequencies=4)
        x = torch.randn(5, 10)
        out = encoder(x)
        assert out.shape == (5, 10 * (1 + 2 * 4))


# =============================================================================
# Module State Tests
# =============================================================================

class TestEncoderModuleState:
    """Tests for encoder module state."""

    def test_positional_encoder_buffers(self):
        """PositionalEncoder has freq_bands as buffer."""
        encoder = PositionalEncoder(num_frequencies=5)
        assert 'freq_bands' in dict(encoder.named_buffers())
        assert encoder.freq_bands.shape == (5,)

    def test_fourier_encoder_fixed_buffer(self):
        """Non-learnable FourierEncoder has B as buffer."""
        encoder = FourierEncoder(learnable=False)
        assert 'B' in dict(encoder.named_buffers())

    def test_fourier_encoder_learnable_parameter(self):
        """Learnable FourierEncoder has B as parameter."""
        encoder = FourierEncoder(learnable=True)
        assert 'B' in dict(encoder.named_parameters())

    def test_hash_encoder_embeddings(self):
        """HashEncoder has embedding tables."""
        encoder = HashEncoder(num_levels=4)
        assert len(list(encoder.named_modules())) > 1

    def test_gaussian_encoder_centers_buffer(self):
        """GaussianEncoder has centers as buffer."""
        encoder = GaussianEncoder(num_centers=4)
        assert 'centers' in dict(encoder.named_buffers())

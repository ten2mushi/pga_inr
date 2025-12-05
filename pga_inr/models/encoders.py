"""
Positional and Fourier encoders for implicit neural representations.

These encoders map low-dimensional coordinates to higher-dimensional
feature spaces, enabling networks to learn high-frequency details.
"""

from typing import Optional
import math
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Classic positional encoding as used in NeRF.

    Encodes each input coordinate x as:
    [x, sin(2^0 π x), cos(2^0 π x), sin(2^1 π x), cos(2^1 π x), ...,
     sin(2^{L-1} π x), cos(2^{L-1} π x)]

    Output dimension: input_dim * (1 + 2 * num_frequencies) if include_input
                      input_dim * (2 * num_frequencies) otherwise
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_frequencies: int = 10,
        log_sampling: bool = True,
        include_input: bool = True,
        max_frequency: Optional[float] = None
    ):
        """
        Args:
            input_dim: Dimension of input coordinates
            num_frequencies: Number of frequency bands (L)
            log_sampling: If True, frequencies are 2^k for k=0..L-1
                         If False, frequencies are linearly spaced
            include_input: Whether to include raw input in output
            max_frequency: Maximum frequency (only used if log_sampling=False)
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Compute frequency bands
        if log_sampling:
            # Exponential: 2^0, 2^1, ..., 2^{L-1}
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            # Linear spacing
            if max_frequency is None:
                max_frequency = 2.0 ** (num_frequencies - 1)
            freq_bands = torch.linspace(1.0, max_frequency, num_frequencies)

        # Scale by π
        self.register_buffer('freq_bands', freq_bands * math.pi)

        # Compute output dimension
        self.output_dim = input_dim * 2 * num_frequencies
        if include_input:
            self.output_dim += input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input coordinates.

        Args:
            x: Input of shape (..., input_dim)

        Returns:
            Encoded output of shape (..., output_dim)
        """
        # Compute all frequency encodings
        # x: (..., D), freq_bands: (L,)
        # Result shape: (..., D, L)
        x_freq = x.unsqueeze(-1) * self.freq_bands

        # Compute sin and cos
        # Shape: (..., D, L)
        sin_x = torch.sin(x_freq)
        cos_x = torch.cos(x_freq)

        # Interleave sin and cos and flatten
        # Shape: (..., D * 2 * L)
        encoded = torch.stack([sin_x, cos_x], dim=-1)
        # Use explicit dimension to handle empty batch case
        encoded_dim = self.input_dim * 2 * self.num_frequencies
        encoded = encoded.reshape(*x.shape[:-1], encoded_dim)

        # Optionally prepend raw input
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


class FourierEncoder(nn.Module):
    """
    Random Fourier Features encoding.

    Uses random frequency matrices for a more flexible encoding:
    γ(x) = [sin(Bx), cos(Bx)]

    where B is a matrix of random frequencies sampled from N(0, σ²).

    Reference:
        Tancik et al., "Fourier Features Let Networks Learn High Frequency
        Functions in Low Dimensional Domains", NeurIPS 2020.
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_frequencies: int = 256,
        sigma: float = 10.0,
        include_input: bool = True,
        learnable: bool = False
    ):
        """
        Args:
            input_dim: Dimension of input coordinates
            num_frequencies: Number of random frequencies
            sigma: Standard deviation of frequency distribution
            include_input: Whether to include raw input in output
            learnable: Whether frequencies should be learnable parameters
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Random frequency matrix B ~ N(0, σ²)
        B = torch.randn(num_frequencies, input_dim) * sigma

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

        # Output dimension
        self.output_dim = 2 * num_frequencies
        if include_input:
            self.output_dim += input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input coordinates.

        Args:
            x: Input of shape (..., input_dim)

        Returns:
            Encoded output of shape (..., output_dim)
        """
        # x: (..., D), B: (F, D)
        # x @ B.T: (..., F)
        x_proj = x @ self.B.T * 2 * math.pi

        # Compute sin and cos
        sin_x = torch.sin(x_proj)
        cos_x = torch.cos(x_proj)

        # Concatenate
        encoded = torch.cat([sin_x, cos_x], dim=-1)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


class GaussianEncoder(nn.Module):
    """
    Gaussian encoding for smooth interpolation.

    Uses Gaussian basis functions centered at grid points:
    γ(x) = [exp(-||x - c_1||² / 2σ²), ..., exp(-||x - c_K||² / 2σ²)]

    Useful when smooth interpolation is more important than high-frequency detail.
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_centers: int = 64,
        sigma: float = 0.1,
        bounds: tuple = (-1, 1)
    ):
        """
        Args:
            input_dim: Dimension of input coordinates
            num_centers: Number of Gaussian centers per dimension
            sigma: Standard deviation of Gaussians
            bounds: Coordinate bounds for center placement
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_centers = num_centers
        self.sigma = sigma

        # Create grid of centers
        # For 3D with 4 centers: 4^3 = 64 total centers
        centers_1d = torch.linspace(bounds[0], bounds[1], num_centers)
        grids = torch.meshgrid(*[centers_1d] * input_dim, indexing='ij')
        centers = torch.stack([g.flatten() for g in grids], dim=-1)

        self.register_buffer('centers', centers)
        self.output_dim = centers.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input coordinates.

        Args:
            x: Input of shape (..., input_dim)

        Returns:
            Encoded output of shape (..., num_centers^input_dim)
        """
        # x: (..., D), centers: (K, D)
        # Compute squared distances
        # Expand x: (..., 1, D), centers: (K, D)
        x_exp = x.unsqueeze(-2)
        diff = x_exp - self.centers
        dist_sq = (diff ** 2).sum(dim=-1)

        # Gaussian activations
        encoded = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        return encoded


class HashEncoder(nn.Module):
    """
    Multi-resolution hash encoding (simplified version).

    Inspired by InstantNGP, uses multiple resolution levels with
    hash tables for efficient lookup.

    Note: This is a simplified version without spatial hashing.
    For full performance, use the original InstantNGP implementation.
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_levels: int = 16,
        level_dim: int = 2,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        log2_hashmap_size: int = 19
    ):
        """
        Args:
            input_dim: Dimension of input coordinates
            num_levels: Number of resolution levels
            level_dim: Features per level
            base_resolution: Resolution at first level
            max_resolution: Resolution at last level
            log2_hashmap_size: Log2 of hash table size
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim

        # Compute resolution at each level
        b = (max_resolution / base_resolution) ** (1 / (num_levels - 1))
        resolutions = [int(base_resolution * (b ** i)) for i in range(num_levels)]
        self.register_buffer('resolutions', torch.tensor(resolutions))

        # Create embedding tables for each level
        hash_size = 2 ** log2_hashmap_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(hash_size, level_dim)
            for _ in range(num_levels)
        ])

        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

        self.output_dim = num_levels * level_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input coordinates.

        Args:
            x: Input of shape (..., input_dim), normalized to [0, 1]

        Returns:
            Encoded output of shape (..., output_dim)
        """
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)

        features = []

        for level, (res, emb) in enumerate(zip(self.resolutions, self.embeddings)):
            # Scale coordinates to grid
            x_scaled = x_flat * res

            # Get grid indices (floor)
            x_floor = torch.floor(x_scaled).long()

            # Simple hash (for demonstration)
            # Real implementation uses proper spatial hashing
            primes = torch.tensor([1, 2654435761, 805459861], device=x.device)[:self.input_dim]
            hash_idx = ((x_floor * primes).sum(dim=-1) % emb.num_embeddings).long()

            # Lookup
            feat = emb(hash_idx)
            features.append(feat)

        encoded = torch.cat(features, dim=-1)
        return encoded.reshape(*batch_shape, -1)


class CompositeEncoder(nn.Module):
    """
    Combines multiple encoders.

    Useful for mixing different encoding strategies.
    """

    def __init__(self, encoders: list):
        """
        Args:
            encoders: List of encoder modules
        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.output_dim = sum(e.output_dim for e in encoders)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate outputs from all encoders."""
        outputs = [enc(x) for enc in self.encoders]
        return torch.cat(outputs, dim=-1)


class IdentityEncoder(nn.Module):
    """
    Identity encoder (no encoding).

    Passes input through unchanged. Useful as a baseline or placeholder.
    """

    def __init__(self, input_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

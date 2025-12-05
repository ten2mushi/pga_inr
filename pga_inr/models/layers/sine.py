"""
SIREN (Sinusoidal Representation Networks) layers.

SIREN uses periodic sinusoidal activation functions to better represent
high-frequency details in implicit neural representations.

Reference:
    Sitzmann et al., "Implicit Neural Representations with Periodic
    Activation Functions", NeurIPS 2020.
"""

from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """
    A single SIREN layer with sinusoidal activation.

    f(x) = sin(ω₀ · (Wx + b))

    The initialization scheme is critical for SIREN to work properly:
    - First layer: W ~ U(-1/n, 1/n)
    - Hidden layers: W ~ U(-√(6/n)/ω₀, √(6/n)/ω₀)

    Where n is the input dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0
    ):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
            is_first: Whether this is the first layer (affects initialization)
            omega_0: Frequency multiplier (ω₀)
        """
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/n, 1/n]
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers: uniform in [-√(6/n)/ω₀, √(6/n)/ω₀]
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_intermediate(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning intermediate values for analysis.

        Returns:
            (output, preactivation) where preactivation = ω₀ * (Wx + b)
        """
        preact = self.omega_0 * self.linear(x)
        return torch.sin(preact), preact


class SirenMLP(nn.Module):
    """
    A complete SIREN network (stack of SineLayer).

    Standard architecture for implicit neural representations.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        final_activation: Optional[nn.Module] = None
    ):
        """
        Args:
            in_features: Input dimension (typically 3 for coordinates)
            hidden_features: Hidden layer width
            hidden_layers: Number of hidden layers
            out_features: Output dimension
            omega_0: Frequency for first layer
            omega_hidden: Frequency for hidden layers
            final_activation: Optional activation for output layer
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Build network
        layers = []

        # First layer
        layers.append(SineLayer(
            in_features,
            hidden_features,
            is_first=True,
            omega_0=omega_0
        ))

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(SineLayer(
                hidden_features,
                hidden_features,
                is_first=False,
                omega_0=omega_hidden
            ))

        self.layers = nn.ModuleList(layers)

        # Final linear layer (no activation by default)
        self.final_linear = nn.Linear(hidden_features, out_features)
        self._init_final_layer()

        self.final_activation = final_activation

    def _init_final_layer(self):
        """Initialize final layer weights."""
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.final_linear.in_features) / 30.0
            self.final_linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input coordinates of shape (..., in_features)

        Returns:
            Output of shape (..., out_features)
        """
        for layer in self.layers:
            x = layer(x)

        x = self.final_linear(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def forward_with_features(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning intermediate features.

        Useful for skip connections or analysis.

        Returns:
            (output, features) where features is the last hidden layer output
        """
        for layer in self.layers:
            x = layer(x)

        features = x
        output = self.final_linear(x)

        if self.final_activation is not None:
            output = self.final_activation(output)

        return output, features


class ModulatedSineLayer(nn.Module):
    """
    SIREN layer with modulation for conditioning.

    Supports FiLM-style modulation: γ * sin(ω₀ * (Wx + b)) + β

    Where γ (scale) and β (shift) come from a conditioning network.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False
    ):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional FiLM modulation.

        Args:
            x: Input tensor
            gamma: Scale modulation (same shape as output or broadcastable)
            beta: Shift modulation (same shape as output or broadcastable)

        Returns:
            Modulated output
        """
        out = torch.sin(self.omega_0 * self.linear(x))

        if gamma is not None:
            out = gamma * out

        if beta is not None:
            out = out + beta

        return out


class ModulatedSirenMLP(nn.Module):
    """
    Complete modulated SIREN network.

    Accepts per-layer modulation parameters from a conditioning network.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        omega_0: float = 30.0
    ):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features

        # Build layers
        layers = []
        layers.append(ModulatedSineLayer(
            in_features, hidden_features,
            omega_0=omega_0, is_first=True
        ))

        for _ in range(hidden_layers):
            layers.append(ModulatedSineLayer(
                hidden_features, hidden_features,
                omega_0=omega_0, is_first=False
            ))

        self.layers = nn.ModuleList(layers)
        self.final_linear = nn.Linear(hidden_features, out_features)

        # Initialize final layer
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_features) / omega_0
            self.final_linear.weight.uniform_(-bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        modulations: Optional[List[tuple]] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional modulation.

        Args:
            x: Input coordinates
            modulations: List of (gamma, beta) tuples for each layer.
                        If None, no modulation is applied.

        Returns:
            Output tensor
        """
        if modulations is None:
            modulations = [(None, None)] * (len(self.layers))

        for i, layer in enumerate(self.layers):
            gamma, beta = modulations[i] if i < len(modulations) else (None, None)
            x = layer(x, gamma=gamma, beta=beta)

        return self.final_linear(x)


class GradientSiren(nn.Module):
    """
    SIREN variant that also computes analytical gradients.

    Useful for SDF learning where we need both f(x) and ∇f(x).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int = 1,
        omega_0: float = 30.0
    ):
        super().__init__()
        self.siren = SirenMLP(
            in_features, hidden_features, hidden_layers, out_features, omega_0
        )

    def forward(
        self,
        x: torch.Tensor,
        compute_grad: bool = True
    ) -> tuple:
        """
        Forward pass optionally computing gradients.

        Args:
            x: Input coordinates of shape (..., 3)
            compute_grad: Whether to compute gradients

        Returns:
            (output, gradient) if compute_grad else output
            gradient has shape (..., 3)
        """
        if compute_grad:
            x = x.requires_grad_(True)

        output = self.siren(x)

        if compute_grad:
            gradient = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            return output, gradient
        else:
            return output

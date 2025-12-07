"""
Phase-Coherent SIREN for Weight-Space Interpolation.

The problem: Standard SIREN networks independently trained on different shapes
have different phase configurations in their first layer. This makes naive
weight interpolation produce artifacts because the phases don't align.

The solution: Share the first SIREN layer across all networks in an ensemble.
This forces all networks to start from the same phase configuration, enabling
meaningful weight-space interpolation in the remaining layers.

Key insight from the pga_inr discoveries:
- Output-space interpolation ALWAYS works: lerp(model_a(x), model_b(x), alpha)
- Weight-space interpolation requires phase alignment
- Sharing the first layer provides this alignment naturally

Classes:
    - SharedFirstLayerSIREN: A shared first layer for phase alignment
    - PhaseCoherentSIREN_SDF: SDF network enabling weight-space interpolation
    - PhaseAlignmentTrainer: Training utilities for phase consistency
"""

from typing import Dict, List, Optional, Tuple, Union
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base import BaseINR
from ..core.constants import (
    DEFAULT_HIDDEN_FEATURES,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_OMEGA_0,
    OUTPUT_SDF,
    OUTPUT_NORMAL,
    OUTPUT_FEATURES,
)
from ..core.types import TensorDict, ObserverPose
from .layers import SineLayer


class SharedFirstLayerSIREN(nn.Module):
    """
    A shared first SIREN layer for phase alignment across networks.

    When multiple SIREN networks share this first layer, their activations
    start from the same phase configuration. This enables meaningful
    weight-space interpolation in the subsequent layers.

    The layer is frozen during individual network training but can be
    jointly optimized during alignment training.
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        omega_0: float = DEFAULT_OMEGA_0,
        freeze: bool = True,
    ):
        """
        Args:
            in_features: Input dimension (3 for 3D coordinates)
            hidden_features: Output dimension (typically same as network width)
            omega_0: SIREN frequency parameter for first layer
            freeze: Whether to freeze weights after initialization
        """
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.omega_0 = omega_0

        # The shared sine layer
        self.layer = SineLayer(
            in_features=in_features,
            out_features=hidden_features,
            is_first=True,
            omega_0=omega_0,
        )

        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze layer weights."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze layer weights for joint optimization."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared first layer.

        Args:
            x: Input coordinates of shape (..., in_features)

        Returns:
            Activated features of shape (..., hidden_features)
        """
        return self.layer(x)

    def get_phase_signature(self, sample_points: torch.Tensor) -> torch.Tensor:
        """
        Compute phase signature for debugging/visualization.

        The phase signature shows how the first layer maps spatial
        positions to activation phases.

        Args:
            sample_points: Points to evaluate (N, 3)

        Returns:
            Phase values (N, hidden_features)
        """
        with torch.no_grad():
            preact = self.omega_0 * self.layer.linear(sample_points)
            # Return phases in [-pi, pi]
            return torch.remainder(preact + np.pi, 2 * np.pi) - np.pi


class PhaseCoherentSIREN_SDF(BaseINR):
    """
    SDF network with shared first layer for weight-space interpolation.

    Architecture:
        1. SharedFirstLayerSIREN (shared, frozen) - provides phase alignment
        2. SIREN hidden layers (per-network) - learnable
        3. Output head (per-network) - produces SDF values

    This design enables:
        - Independent training of shapes (each with own hidden layers)
        - Meaningful weight interpolation between trained networks
        - Direct latent space arithmetic via weight manipulation

    Usage:
        # Create shared layer once
        shared = SharedFirstLayerSIREN()

        # Create coherent networks
        net_a = PhaseCoherentSIREN_SDF(shared)
        net_b = PhaseCoherentSIREN_SDF(shared)

        # Train independently
        train(net_a, shape_a_data)
        train(net_b, shape_b_data)

        # Interpolate in weight space!
        interpolate_weights(net_a, net_b, 0.5, net_target)
    """

    def __init__(
        self,
        shared_layer: SharedFirstLayerSIREN,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        omega_hidden: float = DEFAULT_OMEGA_0,
    ):
        """
        Args:
            shared_layer: Pre-created shared first layer
            hidden_features: Width of hidden layers (should match shared layer)
            hidden_layers: Number of hidden layers (excluding shared first)
            omega_hidden: SIREN frequency for hidden layers
        """
        super().__init__()

        self.shared_layer = shared_layer
        self.hidden_features = hidden_features
        self.hidden_layers_count = hidden_layers
        self.omega_hidden = omega_hidden

        # Verify dimension compatibility
        if shared_layer.hidden_features != hidden_features:
            raise ValueError(
                f"Shared layer output ({shared_layer.hidden_features}) must match "
                f"hidden_features ({hidden_features})"
            )

        # Build hidden layers (these are interpolatable)
        self.layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(
                SineLayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    is_first=False,
                    omega_0=omega_hidden,
                )
            )

        # SDF output head
        self.sdf_head = nn.Linear(hidden_features, 1)
        self._init_sdf_head()

    def _init_sdf_head(self):
        """Initialize SDF head for reasonable initial values."""
        with torch.no_grad():
            # Small initialization for stability
            bound = np.sqrt(6.0 / self.hidden_features) / self.omega_hidden
            self.sdf_head.weight.uniform_(-bound, bound)
            self.sdf_head.bias.zero_()

    def get_output_type(self) -> str:
        """Return 'sdf' as primary output type."""
        return OUTPUT_SDF

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Forward pass.

        Args:
            query_points: Coordinates of shape (..., 3) or (B, N, 3)
            observer_pose: Not used (for API compatibility)

        Returns:
            Dictionary with:
                - 'sdf': Signed distance values (..., 1) or (B, N, 1)
                - 'features': Last layer features for analysis
        """
        # Handle different input shapes
        original_shape = query_points.shape
        if query_points.dim() == 2:
            # (N, 3) -> (1, N, 3)
            x = query_points.unsqueeze(0)
        else:
            x = query_points

        # Shared first layer (frozen, provides phase alignment)
        x = self.shared_layer(x)

        # Hidden layers (interpolatable)
        for layer in self.layers:
            x = layer(x)

        # Store features before final projection
        features = x

        # SDF output
        sdf = self.sdf_head(x)

        # Restore original shape if needed
        if query_points.dim() == 2:
            sdf = sdf.squeeze(0)
            features = features.squeeze(0)

        return {
            OUTPUT_SDF: sdf,
            OUTPUT_FEATURES: features,
        }

    def get_interpolatable_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get the weights that can be interpolated.

        Returns a dictionary of state_dict entries for hidden layers and
        output head (excludes shared first layer).

        Returns:
            Dictionary mapping parameter names to tensors
        """
        weights = {}

        # Hidden layers
        for i, layer in enumerate(self.layers):
            for name, param in layer.named_parameters():
                weights[f"layers.{i}.{name}"] = param.data.clone()

        # SDF head
        for name, param in self.sdf_head.named_parameters():
            weights[f"sdf_head.{name}"] = param.data.clone()

        return weights

    def set_interpolatable_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Set the interpolatable weights from a dictionary.

        Args:
            weights: Dictionary from get_interpolatable_weights()
        """
        current_state = self.state_dict()
        for name, tensor in weights.items():
            if name in current_state:
                current_state[name] = tensor
        self.load_state_dict(current_state)

    @staticmethod
    def interpolate_weights(
        model_a: "PhaseCoherentSIREN_SDF",
        model_b: "PhaseCoherentSIREN_SDF",
        alpha: float,
        target: Optional["PhaseCoherentSIREN_SDF"] = None,
    ) -> "PhaseCoherentSIREN_SDF":
        """
        Interpolate weights between two phase-coherent networks.

        Because both networks share the same first layer (phase alignment),
        linear interpolation of the remaining weights produces smooth
        shape morphing.

        Args:
            model_a: First model (alpha=0)
            model_b: Second model (alpha=1)
            alpha: Interpolation factor in [0, 1]
            target: Optional target model to store result (avoids allocation)

        Returns:
            Model with interpolated weights (target if provided, else new model)
        """
        # Verify same shared layer
        if model_a.shared_layer is not model_b.shared_layer:
            raise ValueError(
                "Models must share the same SharedFirstLayerSIREN for "
                "weight interpolation to work correctly."
            )

        # Create or use target
        if target is None:
            target = PhaseCoherentSIREN_SDF(
                shared_layer=model_a.shared_layer,
                hidden_features=model_a.hidden_features,
                hidden_layers=model_a.hidden_layers_count,
                omega_hidden=model_a.omega_hidden,
            )

        # Get weights
        weights_a = model_a.get_interpolatable_weights()
        weights_b = model_b.get_interpolatable_weights()

        # Interpolate
        interpolated = {}
        for name in weights_a:
            interpolated[name] = (
                (1 - alpha) * weights_a[name] + alpha * weights_b[name]
            )

        # Set interpolated weights
        target.set_interpolatable_weights(interpolated)

        return target


class PhaseCoherentEnsemble(nn.Module):
    """
    Ensemble of phase-coherent SIREN networks sharing a first layer.

    Useful for:
        - Training multiple shapes with guaranteed interpolatability
        - Shape collections where any pair can be smoothly morphed
        - Latent space exploration via weight combinations
    """

    def __init__(
        self,
        num_networks: int,
        in_features: int = 3,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        omega_0: float = DEFAULT_OMEGA_0,
        omega_hidden: float = DEFAULT_OMEGA_0,
    ):
        """
        Args:
            num_networks: Number of networks in ensemble
            in_features: Input dimension
            hidden_features: Hidden layer width
            hidden_layers: Number of hidden layers per network
            omega_0: First layer frequency
            omega_hidden: Hidden layer frequency
        """
        super().__init__()

        # Shared first layer
        self.shared_layer = SharedFirstLayerSIREN(
            in_features=in_features,
            hidden_features=hidden_features,
            omega_0=omega_0,
            freeze=True,  # Frozen by default
        )

        # Individual networks
        self.networks = nn.ModuleList([
            PhaseCoherentSIREN_SDF(
                shared_layer=self.shared_layer,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                omega_hidden=omega_hidden,
            )
            for _ in range(num_networks)
        ])

    def __getitem__(self, idx: int) -> PhaseCoherentSIREN_SDF:
        """Get network by index."""
        return self.networks[idx]

    def __len__(self) -> int:
        """Number of networks in ensemble."""
        return len(self.networks)

    def forward(
        self,
        query_points: torch.Tensor,
        network_idx: int = 0,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Forward pass through a specific network.

        Args:
            query_points: Input coordinates
            network_idx: Which network to use
            observer_pose: Not used (API compatibility)

        Returns:
            Output dictionary from selected network
        """
        return self.networks[network_idx](query_points, observer_pose)

    def interpolate(
        self,
        idx_a: int,
        idx_b: int,
        alpha: float,
    ) -> PhaseCoherentSIREN_SDF:
        """
        Create interpolated network between two ensemble members.

        Args:
            idx_a: Index of first network
            idx_b: Index of second network
            alpha: Interpolation factor [0, 1]

        Returns:
            New network with interpolated weights
        """
        return PhaseCoherentSIREN_SDF.interpolate_weights(
            self.networks[idx_a],
            self.networks[idx_b],
            alpha,
        )

    def blend(
        self,
        weights: List[float],
    ) -> PhaseCoherentSIREN_SDF:
        """
        Blend multiple networks with given weights.

        Args:
            weights: Weight for each network (should sum to 1)

        Returns:
            New network with blended weights
        """
        if len(weights) != len(self.networks):
            raise ValueError(
                f"Need {len(self.networks)} weights, got {len(weights)}"
            )

        # Normalize weights
        weights = torch.tensor(weights)
        weights = weights / weights.sum()

        # Get all interpolatable weights
        all_weights = [net.get_interpolatable_weights() for net in self.networks]

        # Blend
        blended = {}
        for name in all_weights[0]:
            blended[name] = sum(
                w.item() * ws[name] for w, ws in zip(weights, all_weights)
            )

        # Create result network
        result = PhaseCoherentSIREN_SDF(
            shared_layer=self.shared_layer,
            hidden_features=self.networks[0].hidden_features,
            hidden_layers=self.networks[0].hidden_layers_count,
            omega_hidden=self.networks[0].omega_hidden,
        )
        result.set_interpolatable_weights(blended)

        return result


class PhaseAlignmentTrainer:
    """
    Training utilities for phase consistency across networks.

    Provides methods for:
        - Computing phase alignment loss between networks
        - Joint optimization of shared layer
        - Analyzing phase distributions
    """

    def __init__(
        self,
        networks: Union[PhaseCoherentEnsemble, List[PhaseCoherentSIREN_SDF]],
        alignment_weight: float = 0.1,
        sample_points: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            networks: Ensemble or list of phase-coherent networks
            alignment_weight: Weight for alignment loss term
            sample_points: Fixed points for alignment computation
        """
        if isinstance(networks, PhaseCoherentEnsemble):
            self.networks = list(networks.networks)
            self.shared_layer = networks.shared_layer
        else:
            self.networks = networks
            # Verify all share same layer
            shared = networks[0].shared_layer
            for net in networks[1:]:
                if net.shared_layer is not shared:
                    raise ValueError("All networks must share the same first layer")
            self.shared_layer = shared

        self.alignment_weight = alignment_weight

        # Default sample points
        if sample_points is None:
            sample_points = torch.randn(1000, 3) * 0.5
        self.sample_points = sample_points

    def compute_alignment_loss(
        self,
        points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute phase alignment loss across all networks.

        This measures how well the activations align across networks
        at the same spatial locations.

        Args:
            points: Sample points (uses default if None)

        Returns:
            Scalar alignment loss
        """
        if points is None:
            points = self.sample_points.to(
                next(self.networks[0].parameters()).device
            )

        # Get activations from each network's first hidden layer
        # (after the shared layer)
        activations = []
        for net in self.networks:
            shared_out = self.shared_layer(points)
            if len(net.layers) > 0:
                first_hidden = net.layers[0](shared_out)
                activations.append(first_hidden)

        if len(activations) < 2:
            return torch.tensor(0.0)

        # Compute pairwise correlation loss
        # We want activations to be similar in structure
        loss = 0.0
        count = 0
        for i in range(len(activations)):
            for j in range(i + 1, len(activations)):
                # Cosine similarity between activation patterns
                a = activations[i].flatten(1)
                b = activations[j].flatten(1)
                sim = F.cosine_similarity(a, b, dim=1).mean()
                loss = loss + (1 - sim)  # Convert to loss (0 = perfect alignment)
                count += 1

        return loss / max(count, 1)

    def get_phase_statistics(
        self,
        points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute phase statistics for analysis.

        Args:
            points: Sample points

        Returns:
            Dictionary with phase statistics
        """
        if points is None:
            points = self.sample_points

        device = next(self.networks[0].parameters()).device
        points = points.to(device)

        with torch.no_grad():
            # Get phase signature from shared layer
            phase_sig = self.shared_layer.get_phase_signature(points)

            # Compute statistics
            return {
                "phase_mean": phase_sig.mean(dim=0),
                "phase_std": phase_sig.std(dim=0),
                "phase_range": phase_sig.max(dim=0).values - phase_sig.min(dim=0).values,
                "phase_coverage": (phase_sig.abs() > 0.1).float().mean(dim=0),
            }

    def optimize_shared_layer(
        self,
        num_steps: int = 100,
        lr: float = 1e-4,
        sample_points: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """
        Jointly optimize the shared layer for better phase distribution.

        This can help ensure the first layer provides good coverage
        of the activation space.

        Args:
            num_steps: Number of optimization steps
            lr: Learning rate
            sample_points: Points to optimize over

        Returns:
            List of loss values during optimization
        """
        if sample_points is None:
            sample_points = self.sample_points

        device = next(self.networks[0].parameters()).device
        sample_points = sample_points.to(device)

        # Unfreeze shared layer
        self.shared_layer.unfreeze()

        optimizer = torch.optim.Adam(self.shared_layer.parameters(), lr=lr)

        losses = []
        for _ in range(num_steps):
            optimizer.zero_grad()

            # Compute phase distribution
            phase_sig = self.shared_layer.get_phase_signature(sample_points)

            # Loss: encourage spread across phase space
            # Maximize variance (good coverage)
            phase_var = phase_sig.var(dim=0).mean()

            # Minimize correlation between neurons (diversity)
            phase_centered = phase_sig - phase_sig.mean(dim=0, keepdim=True)
            corr_matrix = torch.mm(phase_centered.T, phase_centered) / len(sample_points)
            # Off-diagonal elements should be small
            identity_mask = torch.eye(corr_matrix.shape[0], device=device)
            off_diag = (corr_matrix * (1 - identity_mask)).abs().mean()

            loss = -phase_var + 0.1 * off_diag

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Re-freeze shared layer
        self.shared_layer.freeze()

        return losses

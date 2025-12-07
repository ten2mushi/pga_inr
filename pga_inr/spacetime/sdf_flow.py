"""
Neural SDF Flow for 4D Shape Evolution.

Implements neural networks that learn time-varying SDFs satisfying the
level-set equation. This enables physically-meaningful shape evolution
like melting, growth, or fluid interfaces.

The level-set equation:
    d(SDF)/dt + v · grad(SDF) = 0

Where v is a velocity field that transports the zero level set.

Key classes:
    - NeuralSDFFlow: 4D SDF with velocity field
    - LevelSetLoss: Enforces level-set equation
    - FlowRegularization: Smoothness constraints on flow
    - SDFEvolver: Time-stepping utilities

This module extends the spacetime INR framework with explicit
level-set physics constraints.
"""

from typing import Dict, List, Optional, Tuple, Callable
import math

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
)
from ..core.types import TensorDict, ObserverPose
from ..models.layers import SineLayer


class NeuralSDFFlow(BaseINR):
    """
    4D SDF with velocity field satisfying level-set equation.

    Learns a time-varying SDF where the surface evolution follows
    the level-set equation. The velocity field can be:
    - Learned (implicit flow)
    - Specified (explicit flow)
    - Physics-constrained (e.g., mean curvature flow)

    Architecture:
        Input: (x, y, z, t) - 4D spacetime coordinates
        Output: SDF value, velocity vector, surface normal

    The network learns to satisfy:
        d(SDF)/dt + v · grad(SDF) = 0

    This ensures the zero level set moves with velocity v.
    """

    def __init__(
        self,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        omega_0: float = DEFAULT_OMEGA_0,
        velocity_mode: str = "learned",
        time_encoding_dim: int = 32,
    ):
        """
        Args:
            hidden_features: Width of hidden layers
            hidden_layers: Number of hidden layers
            omega_0: SIREN frequency parameter
            velocity_mode: 'learned', 'curvature', or 'constant'
            time_encoding_dim: Dimension of time encoding
        """
        nn.Module.__init__(self)

        self.hidden_features = hidden_features
        self.omega_0 = omega_0
        self.velocity_mode = velocity_mode
        self.time_encoding_dim = time_encoding_dim

        # Time encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_encoding_dim),
            nn.SiLU(),
            nn.Linear(time_encoding_dim, time_encoding_dim),
        )

        # SDF network: (x, y, z, time_encoding) -> SDF
        input_dim = 3 + time_encoding_dim

        self.sdf_layers = nn.ModuleList()
        self.sdf_layers.append(
            SineLayer(input_dim, hidden_features, is_first=True, omega_0=omega_0)
        )
        for _ in range(hidden_layers):
            self.sdf_layers.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0)
            )

        self.sdf_head = nn.Linear(hidden_features, 1)

        # Velocity network (if learned)
        if velocity_mode == "learned":
            self.velocity_layers = nn.ModuleList()
            self.velocity_layers.append(
                SineLayer(input_dim, hidden_features, is_first=True, omega_0=omega_0)
            )
            for _ in range(hidden_layers // 2):
                self.velocity_layers.append(
                    SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0)
                )
            self.velocity_head = nn.Linear(hidden_features, 3)

        self._init_heads()

    def _init_heads(self):
        """Initialize output heads."""
        with torch.no_grad():
            # SDF head: small initialization
            nn.init.xavier_uniform_(self.sdf_head.weight, gain=0.1)
            nn.init.zeros_(self.sdf_head.bias)

            # Velocity head: small initialization for stability
            if hasattr(self, 'velocity_head'):
                nn.init.xavier_uniform_(self.velocity_head.weight, gain=0.01)
                nn.init.zeros_(self.velocity_head.bias)

    def get_output_type(self) -> str:
        return OUTPUT_SDF

    def _forward_sdf_only(
        self,
        query_points: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal forward pass that computes only SDF and velocity.

        This method avoids the recursion in _compute_spatial_gradient by
        not computing normals.

        Returns:
            Tuple of (sdf, velocity)
        """
        # Handle shapes
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)
        B, N, _ = query_points.shape

        # Handle time
        if isinstance(time, (int, float)):
            time = torch.tensor([time], device=query_points.device)
        if time.dim() == 0:
            time = time.unsqueeze(0)
        if time.shape[0] == 1 and B > 1:
            time = time.expand(B)

        # Encode time
        time_encoded = self.time_encoder(time.unsqueeze(-1))  # (B, time_dim)
        time_expanded = time_encoded.unsqueeze(1).expand(-1, N, -1)  # (B, N, time_dim)

        # Combine spatial and temporal features
        x = torch.cat([query_points, time_expanded], dim=-1)  # (B, N, 3 + time_dim)

        # SDF network
        sdf_features = x
        for layer in self.sdf_layers:
            sdf_features = layer(sdf_features)
        sdf = self.sdf_head(sdf_features)

        # Velocity
        if self.velocity_mode == "learned":
            vel_features = x
            for layer in self.velocity_layers:
                vel_features = layer(vel_features)
            velocity = self.velocity_head(vel_features)
        else:
            # Constant/zero velocity (curvature mode handled in full forward)
            velocity = torch.zeros_like(query_points)

        return sdf, velocity

    def forward(
        self,
        query_points: torch.Tensor,
        time: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
        compute_normal: bool = True,
    ) -> TensorDict:
        """
        Forward pass through 4D SDF flow.

        Args:
            query_points: Spatial coordinates (B, N, 3) or (N, 3)
            time: Time value (B,) or (1,) or scalar
            observer_pose: Not used (API compatibility)
            compute_normal: Whether to compute spatial gradient (default True)

        Returns:
            Dictionary with:
                - 'sdf': SDF values (B, N, 1)
                - 'velocity': Velocity vectors (B, N, 3)
                - 'normal': Spatial gradients (B, N, 3) if compute_normal=True
        """
        sdf, velocity = self._forward_sdf_only(query_points, time)

        # Handle curvature velocity mode (needs gradient)
        if self.velocity_mode == "curvature":
            velocity = self._compute_curvature_velocity(query_points, time)

        result = {
            OUTPUT_SDF: sdf,
            'velocity': velocity,
        }

        # Normal (spatial gradient of SDF) - compute only if requested
        if compute_normal:
            normal = self._compute_spatial_gradient(query_points, time)
            result[OUTPUT_NORMAL] = normal

        return result

    def _compute_spatial_gradient(
        self,
        points: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial gradient (normal) using autograd."""
        points_grad = points.clone().requires_grad_(True)

        # Use _forward_sdf_only to avoid recursion
        sdf, _ = self._forward_sdf_only(points_grad, time)

        grad = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=points_grad,
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad

    def _compute_curvature_velocity(
        self,
        points: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity for mean curvature flow."""
        # Mean curvature: H = div(n) where n = grad(SDF) / |grad(SDF)|
        # Velocity: v = -H * n

        points_grad = points.clone().requires_grad_(True)
        output = self.forward(points_grad, time)
        sdf = output[OUTPUT_SDF]

        # First derivatives
        grad = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=points_grad,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normal = grad / grad_norm

        # Approximate mean curvature via Laplacian (trace of Hessian)
        # For SDF: H ~ -0.5 * laplacian(SDF) when |grad| = 1
        laplacian = 0.0
        for i in range(3):
            grad_i = grad[..., i:i+1]
            grad_ii = torch.autograd.grad(
                outputs=grad_i.sum(),
                inputs=points_grad,
                create_graph=False,
                retain_graph=True,
            )[0][..., i:i+1]
            laplacian = laplacian + grad_ii

        curvature = -0.5 * laplacian
        velocity = -curvature * normal

        return velocity.detach()

    def advect(
        self,
        points: torch.Tensor,
        time: torch.Tensor,
        dt: float,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Advect points along the velocity field.

        Args:
            points: Initial points (B, N, 3)
            time: Starting time
            dt: Time step
            num_steps: Number of Euler steps

        Returns:
            Advected points (B, N, 3)
        """
        current_points = points.clone()
        current_time = time

        for _ in range(num_steps):
            output = self.forward(current_points, current_time)
            velocity = output['velocity']

            # Euler step
            current_points = current_points + dt * velocity
            current_time = current_time + dt

        return current_points

    def trace_surface_particle(
        self,
        initial_point: torch.Tensor,
        time_start: float,
        time_end: float,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """
        Trace a surface particle through time.

        Follows a point on the zero level set as it evolves.

        Args:
            initial_point: Starting point (3,)
            time_start: Initial time
            time_end: Final time
            num_steps: Number of time steps

        Returns:
            Trajectory (num_steps + 1, 3)
        """
        dt = (time_end - time_start) / num_steps

        trajectory = [initial_point.unsqueeze(0)]
        current = initial_point.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        current_time = torch.tensor([time_start], device=initial_point.device)

        for step in range(num_steps):
            # Get velocity
            output = self.forward(current, current_time)
            velocity = output['velocity']

            # Euler step
            current = current + dt * velocity
            current_time = current_time + dt

            # Project back to surface (optional, helps stability)
            # This uses Newton's method to find nearest surface point
            for _ in range(2):
                output = self.forward(current, current_time)
                sdf = output[OUTPUT_SDF]
                normal = output[OUTPUT_NORMAL]
                normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-8)
                current = current - sdf * normal

            trajectory.append(current.squeeze(0))

        return torch.cat(trajectory, dim=0)


class LevelSetLoss(nn.Module):
    """
    Level-set equation constraint loss.

    Enforces: d(SDF)/dt + v · grad(SDF) = 0

    This ensures the zero level set moves with velocity v.
    """

    def __init__(
        self,
        weight: float = 1.0,
        surface_weight: float = 2.0,
        surface_band: float = 0.05,
    ):
        """
        Args:
            weight: Overall loss weight
            surface_weight: Extra weight for points near surface
            surface_band: Band width for surface weighting
        """
        super().__init__()
        self.weight = weight
        self.surface_weight = surface_weight
        self.surface_band = surface_band

    def forward(
        self,
        model: NeuralSDFFlow,
        points: torch.Tensor,
        time: torch.Tensor,
        dt: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute level-set equation residual.

        Uses finite difference for time derivative:
            d(SDF)/dt ≈ (SDF(t+dt) - SDF(t)) / dt

        Args:
            model: NeuralSDFFlow model
            points: Sample points (B, N, 3)
            time: Current time (B,) or scalar
            dt: Time step for finite difference

        Returns:
            Scalar loss
        """
        points = points.requires_grad_(True)

        # SDF and velocity at current time
        output_t = model(points, time)
        sdf_t = output_t[OUTPUT_SDF]
        velocity = output_t['velocity']

        # Spatial gradient
        grad_sdf = torch.autograd.grad(
            outputs=sdf_t.sum(),
            inputs=points,
            create_graph=True,
        )[0]

        # SDF at t + dt
        if isinstance(time, (int, float)):
            time_next = time + dt
        else:
            time_next = time + dt

        with torch.no_grad():
            output_t_plus = model(points.detach(), time_next)
            sdf_t_plus = output_t_plus[OUTPUT_SDF]

        # Time derivative (finite difference)
        dsdf_dt = (sdf_t_plus - sdf_t) / dt

        # Level-set equation residual: dsdf/dt + v · grad(sdf) = 0
        advection = (velocity * grad_sdf).sum(dim=-1, keepdim=True)
        residual = dsdf_dt + advection

        # Weight by distance to surface
        surface_weight = self.surface_weight * torch.exp(
            -torch.abs(sdf_t.detach()) / self.surface_band
        )
        weighted_residual = (residual ** 2) * (1 + surface_weight)

        return self.weight * weighted_residual.mean()


class FlowRegularization(nn.Module):
    """
    Regularization for velocity field smoothness.

    Encourages smooth, physically-plausible velocity fields.
    """

    def __init__(
        self,
        spatial_smoothness: float = 0.1,
        temporal_smoothness: float = 0.1,
        divergence_weight: float = 0.0,
    ):
        """
        Args:
            spatial_smoothness: Weight for spatial gradient penalty
            temporal_smoothness: Weight for temporal gradient penalty
            divergence_weight: Weight for divergence penalty (0 = incompressible)
        """
        super().__init__()
        self.spatial_smoothness = spatial_smoothness
        self.temporal_smoothness = temporal_smoothness
        self.divergence_weight = divergence_weight

    def forward(
        self,
        model: NeuralSDFFlow,
        points: torch.Tensor,
        time: torch.Tensor,
        dt: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute flow regularization loss.

        Args:
            model: NeuralSDFFlow model
            points: Sample points (B, N, 3)
            time: Current time
            dt: Time step for temporal gradient

        Returns:
            Scalar regularization loss
        """
        points = points.requires_grad_(True)
        loss = 0.0

        # Get velocity at current time
        output = model(points, time)
        velocity = output['velocity']

        # Spatial smoothness: penalize velocity gradient magnitude
        if self.spatial_smoothness > 0:
            vel_grad = torch.autograd.grad(
                outputs=velocity.sum(),
                inputs=points,
                create_graph=True,
            )[0]
            loss = loss + self.spatial_smoothness * (vel_grad ** 2).mean()

        # Temporal smoothness
        if self.temporal_smoothness > 0:
            if isinstance(time, (int, float)):
                time_next = time + dt
            else:
                time_next = time + dt

            output_next = model(points, time_next)
            velocity_next = output_next['velocity']
            temporal_diff = velocity_next - velocity
            loss = loss + self.temporal_smoothness * (temporal_diff ** 2).mean()

        # Divergence (for incompressible flow)
        if self.divergence_weight > 0:
            div = 0.0
            for i in range(3):
                vi = velocity[..., i:i+1]
                div_i = torch.autograd.grad(
                    outputs=vi.sum(),
                    inputs=points,
                    create_graph=True,
                )[0][..., i:i+1]
                div = div + div_i
            loss = loss + self.divergence_weight * (div ** 2).mean()

        return loss


class SDFEvolver:
    """
    Utilities for time-stepping SDF evolution.

    Provides methods for:
    - Forward simulation
    - Surface extraction at each time step
    - Animation generation
    """

    def __init__(
        self,
        model: NeuralSDFFlow,
        grid_resolution: int = 64,
        bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Args:
            model: Trained NeuralSDFFlow model
            grid_resolution: Resolution for SDF grid extraction
            bounds: Spatial bounds
        """
        self.model = model
        self.grid_resolution = grid_resolution
        self.bounds = bounds

        # Precompute grid
        device = next(model.parameters()).device
        coords_1d = torch.linspace(bounds[0], bounds[1], grid_resolution, device=device)
        x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        self.grid_points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)

    @torch.no_grad()
    def extract_sdf_grid(self, time: float) -> torch.Tensor:
        """
        Extract SDF grid at a given time.

        Args:
            time: Time value

        Returns:
            SDF grid (grid_resolution, grid_resolution, grid_resolution)
        """
        device = self.grid_points.device
        time_tensor = torch.tensor([time], device=device)

        output = self.model(self.grid_points, time_tensor)
        sdf = output[OUTPUT_SDF]

        return sdf.reshape(
            self.grid_resolution,
            self.grid_resolution,
            self.grid_resolution
        )

    @torch.no_grad()
    def evolve_sequence(
        self,
        time_start: float,
        time_end: float,
        num_frames: int,
    ) -> List[torch.Tensor]:
        """
        Extract SDF grids for an animation sequence.

        Args:
            time_start: Starting time
            time_end: Ending time
            num_frames: Number of frames

        Returns:
            List of SDF grids
        """
        times = torch.linspace(time_start, time_end, num_frames)
        grids = []

        for t in times:
            grid = self.extract_sdf_grid(t.item())
            grids.append(grid.cpu())

        return grids

    @torch.no_grad()
    def extract_velocity_field(
        self,
        time: float,
        resolution: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract velocity field for visualization.

        Args:
            time: Time value
            resolution: Sampling resolution

        Returns:
            Tuple of (points, velocities)
        """
        device = next(self.model.parameters()).device
        bounds = self.bounds

        coords_1d = torch.linspace(bounds[0], bounds[1], resolution, device=device)
        x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)

        time_tensor = torch.tensor([time], device=device)
        output = self.model(points, time_tensor)

        return points.squeeze(0).cpu(), output['velocity'].squeeze(0).cpu()

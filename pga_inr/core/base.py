"""
Abstract base classes for PGA-INR models.

This module defines the standard interface that all INR models should implement.
Using these base classes ensures consistent API design across the library.

Class Hierarchy:
    BaseINR (abstract)
    ├── PGA_INR
    │   ├── PGA_INR_SDF
    │   └── PGA_INR_NeRF
    ├── PGA_INR_SDF_V2
    └── BaseGenerativeINR (abstract)
        ├── Generative_PGA_INR
        └── Generative_PGA_INR_SDF
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from .constants import (
    OUTPUT_SDF,
    OUTPUT_DENSITY,
    OUTPUT_RGB,
    OUTPUT_NORMAL,
    OUTPUT_GRADIENT,
    OUTPUT_LOCAL_COORDS,
    OUTPUT_FEATURES,
)
from .types import TensorDict, ObserverPose


class BaseINR(nn.Module, ABC):
    """
    Abstract base class for all Implicit Neural Representation models.

    This class defines the standard interface that all INR models must implement.
    It ensures consistent return types and method signatures across the library.

    Standard Output Keys:
        - 'sdf' or 'density': Primary scalar field (mutually exclusive)
        - 'rgb': RGB color values in [0, 1], shape (B, N, 3)
        - 'normal': Unit surface normals, shape (B, N, 3)
        - 'local_coords': Points in local frame, shape (B, N, 3)
        - 'gradient': SDF gradient (if computed), shape (B, N, 3)
        - 'features': Hidden features (optional), shape (B, N, hidden_dim)

    Subclasses must implement:
        - forward(): Return Dict[str, Tensor]
        - forward_with_gradient(): Return Dict[str, Tensor] including gradient

    Example:
        class MyINR(BaseINR):
            def forward(self, query_points, observer_pose=None):
                # ... compute outputs ...
                return {
                    'sdf': sdf_values,
                    'normal': normals,
                    'local_coords': local_points,
                }
    """

    # Standard output key names (class attributes for easy access)
    KEY_SDF = OUTPUT_SDF
    KEY_DENSITY = OUTPUT_DENSITY
    KEY_RGB = OUTPUT_RGB
    KEY_NORMAL = OUTPUT_NORMAL
    KEY_GRADIENT = OUTPUT_GRADIENT
    KEY_LOCAL_COORDS = OUTPUT_LOCAL_COORDS
    KEY_FEATURES = OUTPUT_FEATURES

    @abstractmethod
    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Forward pass computing the implicit field values.

        Args:
            query_points: World-space coordinates of shape (B, N, 3)
            observer_pose: Optional tuple of (translation, quaternion) defining
                          object pose. Translation is (B, 3) or (3,), quaternion
                          is (B, 4) or (4,) in [w, x, y, z] convention.
                          If None, points are assumed to be in local frame.

        Returns:
            Dictionary with at least one of 'sdf' or 'density', plus:
                - 'local_coords': Coordinates in local frame (B, N, 3)
                - 'normal': Surface normals (B, N, 3) if applicable
                - 'rgb': Colors (B, N, 3) if applicable
        """
        pass

    def forward_with_gradient(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Forward pass that also computes the gradient of the scalar field.

        The gradient of an SDF at a point equals the surface normal direction.
        This method enables computing analytical normals via autograd.

        Args:
            query_points: World-space coordinates (B, N, 3)
            observer_pose: Optional object pose

        Returns:
            Dictionary with all outputs from forward() plus:
                - 'gradient': Gradient of scalar field w.r.t. input (B, N, 3)

        Note:
            Default implementation uses autograd. Subclasses may override
            for more efficient gradient computation.
        """
        # Enable gradient computation
        query_points = query_points.requires_grad_(True)

        # Forward pass
        outputs = self.forward(query_points, observer_pose)

        # Determine which scalar field to differentiate
        if self.KEY_SDF in outputs:
            scalar_field = outputs[self.KEY_SDF]
        elif self.KEY_DENSITY in outputs:
            scalar_field = outputs[self.KEY_DENSITY]
        else:
            raise ValueError(
                "Model output must contain 'sdf' or 'density' for gradient computation"
            )

        # Compute gradient
        gradient = torch.autograd.grad(
            outputs=scalar_field,
            inputs=query_points,
            grad_outputs=torch.ones_like(scalar_field),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        outputs[self.KEY_GRADIENT] = gradient
        return outputs

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_type(self) -> str:
        """
        Return the primary output type of this model.

        Returns:
            'sdf' for signed distance function models
            'density' for density/occupancy models
        """
        # Default implementation - subclasses should override if needed
        return "sdf"


class BaseGenerativeINR(BaseINR):
    """
    Abstract base class for generative INR models with latent conditioning.

    Generative INR models disentangle:
        - Intrinsic shape (encoded in latent code z)
        - Extrinsic pose (handled by PGA motor transformation)

    This enables:
        - Shape interpolation in latent space
        - Multi-object scene composition
        - Zero-shot pose generalization

    Subclasses must implement:
        - forward(): Accept additional latent_code parameter
        - latent_dim property: Return dimension of latent space

    Example:
        class MyGenerativeINR(BaseGenerativeINR):
            @property
            def latent_dim(self) -> int:
                return 64

            def forward(self, query_points, observer_pose, latent_code):
                # ... compute outputs conditioned on latent_code ...
                return {'sdf': sdf, 'normal': normal, 'local_coords': local}
    """

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        pass

    @abstractmethod
    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> TensorDict:
        """
        Forward pass with latent conditioning.

        Args:
            query_points: World coordinates (B, N, 3)
            observer_pose: Tuple (translation, quaternion) for object pose
            latent_code: Shape latent codes (B, latent_dim)

        Returns:
            Dictionary with 'sdf' or 'density', 'local_coords', and optionally
            'normal', 'rgb', 'features'.
        """
        pass

    def forward_with_gradient(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> TensorDict:
        """
        Forward pass with gradient computation.

        Args:
            query_points: World coordinates (B, N, 3)
            observer_pose: Object pose tuple
            latent_code: Shape latent codes (B, latent_dim)

        Returns:
            Dictionary with outputs plus 'gradient'
        """
        query_points = query_points.requires_grad_(True)

        outputs = self.forward(query_points, observer_pose, latent_code)

        if self.KEY_SDF in outputs:
            scalar_field = outputs[self.KEY_SDF]
        elif self.KEY_DENSITY in outputs:
            scalar_field = outputs[self.KEY_DENSITY]
        else:
            raise ValueError(
                "Model output must contain 'sdf' or 'density' for gradient computation"
            )

        gradient = torch.autograd.grad(
            outputs=scalar_field,
            inputs=query_points,
            grad_outputs=torch.ones_like(scalar_field),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        outputs[self.KEY_GRADIENT] = gradient
        return outputs

    def sample_latent(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample random latent codes from prior distribution.

        Default prior is standard normal N(0, I).

        Args:
            batch_size: Number of samples
            device: Device for output tensor

        Returns:
            Latent codes of shape (batch_size, latent_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(batch_size, self.latent_dim, device=device)

    def interpolate_latent(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        t: float,
        mode: str = "linear",
    ) -> torch.Tensor:
        """
        Interpolate between two latent codes.

        Args:
            z1: Start latent code (latent_dim,) or (1, latent_dim)
            z2: End latent code (latent_dim,) or (1, latent_dim)
            t: Interpolation parameter in [0, 1]
            mode: 'linear' or 'spherical' (slerp)

        Returns:
            Interpolated latent code
        """
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        if z2.dim() == 1:
            z2 = z2.unsqueeze(0)

        if mode == "linear":
            return (1 - t) * z1 + t * z2
        elif mode == "spherical":
            # Spherical linear interpolation
            z1_norm = z1 / (z1.norm(dim=-1, keepdim=True) + 1e-8)
            z2_norm = z2 / (z2.norm(dim=-1, keepdim=True) + 1e-8)

            dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
            omega = torch.acos(dot)

            if omega.abs().max() < 1e-6:
                return z1

            sin_omega = torch.sin(omega)
            return (
                torch.sin((1 - t) * omega) * z1 + torch.sin(t * omega) * z2
            ) / sin_omega
        else:
            raise ValueError(f"Unknown interpolation mode: {mode}")

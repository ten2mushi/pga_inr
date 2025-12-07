"""
Neural CSG (Constructive Solid Geometry) Operations.

This module provides operations for combining and morphing neural SDFs:
- SDFMorpher: Output-space interpolation between SDF models
- SmoothCSG: Smooth boolean operations (union, intersection, subtraction)
- MultiShapeBlend: Weighted blending of N shapes
- GradientAwareComposition: CSG with proper gradient/normal handling

Key insight: Weight interpolation fails for SIREN networks due to phase
misalignment. Output-space interpolation is the mathematically correct
approach for morphing between trained SDFs.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base import BaseINR
from ..core.types import TensorDict, ObserverPose
from ..core.constants import OUTPUT_SDF, OUTPUT_NORMAL, OUTPUT_LOCAL_COORDS, OUTPUT_FEATURES


# =============================================================================
# Smooth Min/Max Functions
# =============================================================================

def smooth_min(a: torch.Tensor, b: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    """
    Polynomial smooth minimum (smooth union for SDFs).

    Creates a smooth blend between two values, eliminating the sharp
    transition at min(a, b).

    Args:
        a: First tensor
        b: Second tensor
        k: Smoothness parameter (larger = smoother blend)

    Returns:
        Smoothly blended minimum

    Note:
        When k -> 0, this approaches torch.min(a, b)
    """
    h = torch.clamp(0.5 + 0.5 * (b - a) / (k + 1e-8), 0.0, 1.0)
    return torch.lerp(b, a, h) - k * h * (1 - h)


def smooth_max(a: torch.Tensor, b: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    """
    Polynomial smooth maximum (smooth intersection for SDFs).

    Args:
        a: First tensor
        b: Second tensor
        k: Smoothness parameter (larger = smoother blend)

    Returns:
        Smoothly blended maximum
    """
    return -smooth_min(-a, -b, k)


def smooth_min_exp(a: torch.Tensor, b: torch.Tensor, k: float = 32.0) -> torch.Tensor:
    """
    Exponential smooth minimum using logsumexp.

    More numerically stable for large k values.

    Args:
        a: First tensor
        b: Second tensor
        k: Sharpness parameter (larger = sharper transition)

    Returns:
        Smoothly blended minimum
    """
    stacked = torch.stack([a, b], dim=0)
    return -torch.logsumexp(-k * stacked, dim=0) / k


def smooth_max_exp(a: torch.Tensor, b: torch.Tensor, k: float = 32.0) -> torch.Tensor:
    """
    Exponential smooth maximum using logsumexp.

    Args:
        a: First tensor
        b: Second tensor
        k: Sharpness parameter (larger = sharper transition)

    Returns:
        Smoothly blended maximum
    """
    stacked = torch.stack([a, b], dim=0)
    return torch.logsumexp(k * stacked, dim=0) / k


# =============================================================================
# Base Class
# =============================================================================

class BaseCSGOperation(ABC):
    """
    Abstract base class for CSG operations.

    All CSG operations take query points and return a TensorDict
    with at least 'sdf' key.
    """

    @abstractmethod
    def __call__(
        self,
        points: torch.Tensor,
        **kwargs
    ) -> TensorDict:
        """
        Evaluate the CSG operation at query points.

        Args:
            points: Query points (B, N, 3) or (N, 3)
            **kwargs: Operation-specific parameters

        Returns:
            TensorDict with 'sdf' and optionally 'normal', 'local_coords'
        """
        pass

    def with_gradient(
        self,
        points: torch.Tensor,
        **kwargs
    ) -> TensorDict:
        """
        Evaluate CSG operation with analytical gradient computation.

        Args:
            points: Query points with requires_grad=True
            **kwargs: Operation-specific parameters

        Returns:
            TensorDict with 'sdf', 'gradient', and optionally 'normal'
        """
        points = points.requires_grad_(True)
        outputs = self(points, **kwargs)

        sdf = outputs[OUTPUT_SDF]
        gradient = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        outputs['gradient'] = gradient
        return outputs


# =============================================================================
# SDFMorpher - Output-Space Interpolation
# =============================================================================

class SDFMorpher(BaseCSGOperation):
    """
    Output-space interpolation between SDF models.

    Key insight: Weight interpolation FAILS for SIREN networks because
    sinusoidal activations encode spatial information in phase configurations.
    Two independently trained SIRENs have different phase alignments, so
    interpolating weights causes destructive interference.

    The CORRECT approach is output-space interpolation:
        sdf_morph(x) = (1 - alpha) * sdf_A(x) + alpha * sdf_B(x)

    This preserves the valid SDF property and creates smooth morphs.

    Example:
        >>> morpher = SDFMorpher(model_a, model_b, blend_mode='linear')
        >>> # Morph from shape A to shape B
        >>> for alpha in torch.linspace(0, 1, 10):
        ...     output = morpher(points, alpha)
        ...     mesh = marching_cubes(output['sdf'])
    """

    def __init__(
        self,
        model_a: BaseINR,
        model_b: BaseINR,
        blend_mode: str = 'linear',
        k: float = 0.3,
    ):
        """
        Initialize the SDFMorpher.

        Args:
            model_a: First SDF model (alpha=0)
            model_b: Second SDF model (alpha=1)
            blend_mode: Interpolation mode:
                - 'linear': Simple linear interpolation (default)
                - 'smoothmin': CSG-like smooth blend
                - 'smoothmax': Inverse CSG-like blend
            k: Smoothness parameter for non-linear blend modes
        """
        self.model_a = model_a
        self.model_b = model_b
        self.blend_mode = blend_mode
        self.k = k

        # Validate blend mode
        valid_modes = {'linear', 'smoothmin', 'smoothmax'}
        if blend_mode not in valid_modes:
            raise ValueError(f"blend_mode must be one of {valid_modes}")

    def __call__(
        self,
        points: torch.Tensor,
        alpha: float,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Compute morphed SDF at query points.

        Args:
            points: Query points (B, N, 3) or (N, 3)
            alpha: Interpolation factor in [0, 1]
                   - alpha=0: returns model_a output
                   - alpha=1: returns model_b output
            observer_pose: Optional observer pose for both models

        Returns:
            TensorDict with:
                - 'sdf': Interpolated SDF values
                - 'normal': Interpolated normals (if both models provide them)
                - 'local_coords': Local coordinates from model_a
        """
        with torch.no_grad():
            # Evaluate both models
            out_a = self.model_a(points, observer_pose) if observer_pose else self.model_a(points)
            out_b = self.model_b(points, observer_pose) if observer_pose else self.model_b(points)

            # Extract SDFs
            sdf_a = out_a.get(OUTPUT_SDF, out_a.get('density'))
            sdf_b = out_b.get(OUTPUT_SDF, out_b.get('density'))

            # Interpolate SDF based on blend mode
            if self.blend_mode == 'linear':
                sdf_interp = (1 - alpha) * sdf_a + alpha * sdf_b
            elif self.blend_mode == 'smoothmin':
                # CSG-like smooth blend
                h = torch.clamp(0.5 + 0.5 * (sdf_b - sdf_a) / self.k, 0.0, 1.0)
                h = h * (1 - alpha) + (1 - h) * alpha
                sdf_interp = torch.lerp(sdf_a, sdf_b, h) - self.k * h * (1 - h)
            elif self.blend_mode == 'smoothmax':
                h = torch.clamp(0.5 + 0.5 * (sdf_a - sdf_b) / self.k, 0.0, 1.0)
                h = h * alpha + (1 - h) * (1 - alpha)
                sdf_interp = torch.lerp(sdf_b, sdf_a, h) + self.k * h * (1 - h)

            result = {OUTPUT_SDF: sdf_interp}

            # Interpolate normals if available
            if OUTPUT_NORMAL in out_a and OUTPUT_NORMAL in out_b:
                normal_a = out_a[OUTPUT_NORMAL]
                normal_b = out_b[OUTPUT_NORMAL]
                normal_interp = F.normalize(
                    (1 - alpha) * normal_a + alpha * normal_b,
                    dim=-1
                )
                result[OUTPUT_NORMAL] = normal_interp

            # Use local_coords from model_a (arbitrary choice)
            if OUTPUT_LOCAL_COORDS in out_a:
                result[OUTPUT_LOCAL_COORDS] = out_a[OUTPUT_LOCAL_COORDS]

            return result

    def morph_sequence(
        self,
        points: torch.Tensor,
        num_steps: int = 10,
        observer_pose: Optional[ObserverPose] = None,
    ) -> List[TensorDict]:
        """
        Generate a sequence of morphed SDFs from model_a to model_b.

        Args:
            points: Query points
            num_steps: Number of interpolation steps
            observer_pose: Optional observer pose

        Returns:
            List of TensorDicts for each interpolation step
        """
        alphas = torch.linspace(0, 1, num_steps)
        return [self(points, float(alpha), observer_pose) for alpha in alphas]


# =============================================================================
# SmoothCSG - Smooth Boolean Operations
# =============================================================================

class SmoothCSG(BaseCSGOperation):
    """
    Smooth CSG (Constructive Solid Geometry) operations.

    Implements smooth versions of boolean operations:
        - Union: Combine shapes (smooth minimum of SDFs)
        - Intersection: Overlap region (smooth maximum of SDFs)
        - Subtraction: Remove one shape from another

    The smooth versions eliminate sharp edges at boolean boundaries,
    creating organic-looking results.

    Example:
        >>> union = SmoothCSG([model_a, model_b], operation='union', k=0.1)
        >>> result = union(points)
        >>> # result['sdf'] contains the smooth union
    """

    OPERATIONS = {'union', 'intersection', 'subtraction'}

    def __init__(
        self,
        models: List[BaseINR],
        operation: str,
        k: float = 0.1,
        use_exponential: bool = False,
        gradient_mode: str = 'blend',
    ):
        """
        Initialize SmoothCSG operation.

        Args:
            models: List of SDF models to combine
            operation: CSG operation type:
                - 'union': Combine all shapes
                - 'intersection': Keep overlapping regions
                - 'subtraction': Remove model[1:] from model[0]
            k: Smoothness parameter (larger = smoother blend)
            use_exponential: Use exponential (logsumexp) instead of polynomial
            gradient_mode: How to handle normals/gradients:
                - 'blend': Interpolate based on SDF weights
                - 'dominant': Use normal from dominant shape
        """
        if operation not in self.OPERATIONS:
            raise ValueError(f"operation must be one of {self.OPERATIONS}")

        if len(models) < 2:
            raise ValueError("Need at least 2 models for CSG operation")

        if operation == 'subtraction' and len(models) != 2:
            raise ValueError("Subtraction requires exactly 2 models")

        self.models = models
        self.operation = operation
        self.k = k
        self.use_exponential = use_exponential
        self.gradient_mode = gradient_mode

    def __call__(
        self,
        points: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Evaluate CSG operation at query points.

        Args:
            points: Query points (B, N, 3) or (N, 3)
            observer_pose: Optional observer pose for all models

        Returns:
            TensorDict with 'sdf', optionally 'normal' and 'object_idx'
        """
        # Evaluate all models
        outputs = []
        for model in self.models:
            if observer_pose is not None:
                out = model(points, observer_pose)
            else:
                out = model(points)
            outputs.append(out)

        # Stack SDFs: (num_models, B, N, 1) or (num_models, N, 1)
        sdfs = torch.stack([
            out.get(OUTPUT_SDF, out.get('density'))
            for out in outputs
        ], dim=0)

        # Apply CSG operation
        if self.operation == 'union':
            result_sdf, weights = self._smooth_union(sdfs)
        elif self.operation == 'intersection':
            result_sdf, weights = self._smooth_intersection(sdfs)
        elif self.operation == 'subtraction':
            result_sdf, weights = self._smooth_subtraction(sdfs)

        result = {OUTPUT_SDF: result_sdf}

        # Handle normals
        if OUTPUT_NORMAL in outputs[0]:
            normals = torch.stack([out[OUTPUT_NORMAL] for out in outputs], dim=0)
            if self.gradient_mode == 'blend':
                # Weighted blend of normals
                weights_expanded = weights.unsqueeze(-1)  # Add channel dim
                blended_normal = (weights_expanded * normals).sum(dim=0)
                result[OUTPUT_NORMAL] = F.normalize(blended_normal, dim=-1)
            elif self.gradient_mode == 'dominant':
                # Use normal from dominant shape
                idx = weights.argmax(dim=0, keepdim=True)
                idx_expanded = idx.unsqueeze(-1).expand_as(normals[0:1])
                result[OUTPUT_NORMAL] = torch.gather(normals, 0, idx_expanded).squeeze(0)

        # Track which object is dominant
        result['object_idx'] = weights.argmax(dim=0)

        return result

    def _smooth_union(
        self,
        sdfs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute smooth union (min of SDFs)."""
        if self.use_exponential:
            result = -torch.logsumexp(-self.k * sdfs, dim=0) / self.k
            # Weights based on contribution to logsumexp
            weights = F.softmax(-self.k * sdfs, dim=0)
        else:
            # Iteratively apply smooth_min
            result = sdfs[0]
            for i in range(1, len(sdfs)):
                result = smooth_min(result, sdfs[i], self.k)

            # Compute weights based on distance to result
            distances = (sdfs - result.unsqueeze(0)).abs()
            weights = F.softmax(-distances / (self.k + 1e-8), dim=0).squeeze(-1)

        return result, weights

    def _smooth_intersection(
        self,
        sdfs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute smooth intersection (max of SDFs)."""
        if self.use_exponential:
            result = torch.logsumexp(self.k * sdfs, dim=0) / self.k
            weights = F.softmax(self.k * sdfs, dim=0)
        else:
            result = sdfs[0]
            for i in range(1, len(sdfs)):
                result = smooth_max(result, sdfs[i], self.k)

            distances = (sdfs - result.unsqueeze(0)).abs()
            weights = F.softmax(-distances / (self.k + 1e-8), dim=0).squeeze(-1)

        return result, weights

    def _smooth_subtraction(
        self,
        sdfs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute smooth subtraction: model[0] - model[1]."""
        sdf_a = sdfs[0]
        sdf_b = sdfs[1]

        # Subtraction: A - B = A AND NOT(B) = max(sdf_A, -sdf_B)
        neg_sdf_b = -sdf_b

        if self.use_exponential:
            stacked = torch.stack([sdf_a, neg_sdf_b], dim=0)
            result = torch.logsumexp(self.k * stacked, dim=0) / self.k
            weights = F.softmax(self.k * stacked, dim=0)
        else:
            result = smooth_max(sdf_a, neg_sdf_b, self.k)

            distances = torch.stack([
                (sdf_a - result).abs(),
                (neg_sdf_b - result).abs()
            ], dim=0)
            weights = F.softmax(-distances / (self.k + 1e-8), dim=0).squeeze(-1)

        return result, weights


# =============================================================================
# MultiShapeBlend - Weighted N-Shape Blending
# =============================================================================

class MultiShapeBlend(BaseCSGOperation, nn.Module):
    """
    Blend multiple SDF shapes with learnable or fixed weights.

    Computes weighted combination:
        sdf_blend(x) = sum_i(w_i * sdf_i(x))

    where weights can be:
        - Fixed (provided at init)
        - Learnable (optimized during training)
        - Per-point (spatially varying)

    Example:
        >>> blend = MultiShapeBlend(models, learnable_weights=True)
        >>> # Train to find optimal blend
        >>> output = blend(points)
        >>> loss = some_loss(output['sdf'], target)
        >>> loss.backward()  # Gradients flow to blend weights
    """

    def __init__(
        self,
        models: List[BaseINR],
        weights: Optional[torch.Tensor] = None,
        learnable_weights: bool = False,
        normalize_weights: bool = True,
        blend_normals: bool = True,
    ):
        """
        Initialize MultiShapeBlend.

        Args:
            models: List of SDF models to blend
            weights: Initial blend weights (num_models,). If None, uniform.
            learnable_weights: Whether weights should be optimized
            normalize_weights: Whether to softmax-normalize weights
            blend_normals: Whether to blend normals (vs using dominant)
        """
        nn.Module.__init__(self)

        self.models = nn.ModuleList(models)
        self.normalize_weights = normalize_weights
        self.blend_normals = blend_normals
        self.num_models = len(models)

        # Initialize weights
        if weights is None:
            weights = torch.ones(self.num_models) / self.num_models

        if learnable_weights:
            self.weights = nn.Parameter(weights.clone())
        else:
            self.register_buffer('weights', weights)

    def get_normalized_weights(self) -> torch.Tensor:
        """Get normalized blend weights."""
        if self.normalize_weights:
            return F.softmax(self.weights, dim=0)
        return self.weights

    def __call__(
        self,
        points: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Evaluate blended SDF at query points.

        Args:
            points: Query points (B, N, 3) or (N, 3)
            observer_pose: Optional observer pose

        Returns:
            TensorDict with 'sdf', 'normal', 'weights'
        """
        weights = self.get_normalized_weights()

        # Evaluate all models
        outputs = []
        for model in self.models:
            if observer_pose is not None:
                out = model(points, observer_pose)
            else:
                out = model(points)
            outputs.append(out)

        # Stack SDFs: (num_models, ...)
        sdfs = torch.stack([
            out.get(OUTPUT_SDF, out.get('density'))
            for out in outputs
        ], dim=0)

        # Weighted blend
        # weights shape: (num_models,) -> (num_models, 1, 1, 1) for broadcasting
        weight_shape = [self.num_models] + [1] * (sdfs.dim() - 1)
        weights_expanded = weights.view(*weight_shape)

        blended_sdf = (weights_expanded * sdfs).sum(dim=0)

        result = {
            OUTPUT_SDF: blended_sdf,
            'weights': weights,
        }

        # Blend normals if available
        if OUTPUT_NORMAL in outputs[0] and self.blend_normals:
            normals = torch.stack([out[OUTPUT_NORMAL] for out in outputs], dim=0)
            blended_normal = (weights_expanded * normals).sum(dim=0)
            result[OUTPUT_NORMAL] = F.normalize(blended_normal, dim=-1)

        return result


# =============================================================================
# GradientAwareComposition - CSG with Proper Gradient Handling
# =============================================================================

class GradientAwareComposition(BaseCSGOperation):
    """
    CSG composition with gradient-aware blending at boundaries.

    Unlike simple CSG which can have gradient discontinuities at
    boundaries, this class uses signed distance information to
    smoothly blend gradients/normals near surface intersections.

    Key features:
        - Smooth gradient transitions at CSG boundaries
        - Proper normal interpolation based on surface distance
        - Optional feature blending for neural features

    Example:
        >>> comp = GradientAwareComposition(models, boundary_width=0.05)
        >>> output = comp.with_gradient(points)
        >>> # output['gradient'] is smooth across boundaries
    """

    def __init__(
        self,
        models: List[BaseINR],
        operation: str = 'union',
        boundary_width: float = 0.05,
        k: float = 0.1,
    ):
        """
        Initialize GradientAwareComposition.

        Args:
            models: List of SDF models
            operation: CSG operation ('union', 'intersection', 'subtraction')
            boundary_width: Width of gradient blending region
            k: Smoothness parameter for CSG operation
        """
        self.models = models
        self.operation = operation
        self.boundary_width = boundary_width
        self.k = k

        # Internal CSG for SDF computation
        self._csg = SmoothCSG(
            models=models,
            operation=operation,
            k=k,
            gradient_mode='blend'
        )

    def __call__(
        self,
        points: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Evaluate gradient-aware CSG at query points.

        Args:
            points: Query points (B, N, 3) or (N, 3)
            observer_pose: Optional observer pose

        Returns:
            TensorDict with 'sdf', 'normal', 'boundary_weight'
        """
        # Get base CSG result
        result = self._csg(points, observer_pose)

        # Evaluate individual SDFs for gradient blending weights
        individual_sdfs = []
        individual_normals = []

        for model in self.models:
            if observer_pose is not None:
                out = model(points, observer_pose)
            else:
                out = model(points)

            individual_sdfs.append(out.get(OUTPUT_SDF, out.get('density')))
            if OUTPUT_NORMAL in out:
                individual_normals.append(out[OUTPUT_NORMAL])

        # Stack SDFs
        sdfs = torch.stack(individual_sdfs, dim=0)

        # Compute boundary weights
        # Points near boundaries (where SDFs are similar) get blended normals
        sdf_diffs = (sdfs - result[OUTPUT_SDF].unsqueeze(0)).abs()
        boundary_weight = torch.sigmoid(
            (self.boundary_width - sdf_diffs.min(dim=0)[0]) / (self.boundary_width * 0.5)
        )

        result['boundary_weight'] = boundary_weight

        # If we have normals, do gradient-aware blending
        if individual_normals:
            normals = torch.stack(individual_normals, dim=0)

            # Compute blending weights based on SDF proximity
            # Closer to surface = higher weight
            weights = F.softmax(-sdf_diffs.abs() / (self.boundary_width + 1e-8), dim=0)
            weights_expanded = weights.unsqueeze(-1)

            # Blend normals
            blended_normal = (weights_expanded * normals).sum(dim=0)
            result[OUTPUT_NORMAL] = F.normalize(blended_normal, dim=-1)

        return result

    def with_gradient(
        self,
        points: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
    ) -> TensorDict:
        """
        Evaluate with analytical gradient computation.

        Computes smooth gradients even at CSG boundaries.

        Args:
            points: Query points with requires_grad capability
            observer_pose: Optional observer pose

        Returns:
            TensorDict with 'sdf', 'gradient', 'normal'
        """
        points = points.requires_grad_(True)

        # Get CSG SDF
        result = self(points, observer_pose)
        sdf = result[OUTPUT_SDF]

        # Compute gradient
        gradient = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        result['gradient'] = gradient
        result[OUTPUT_NORMAL] = F.normalize(gradient, dim=-1)

        return result


# =============================================================================
# Utility Functions
# =============================================================================

def morph_sdf(
    sdf_fn_a: Callable,
    sdf_fn_b: Callable,
    points: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Simple functional morphing between two SDF functions.

    Useful for analytic SDFs or quick prototyping.

    Args:
        sdf_fn_a: First SDF function (x -> sdf)
        sdf_fn_b: Second SDF function (x -> sdf)
        points: Query points
        alpha: Interpolation factor [0, 1]

    Returns:
        Interpolated SDF values
    """
    sdf_a = sdf_fn_a(points)
    sdf_b = sdf_fn_b(points)
    return (1 - alpha) * sdf_a + alpha * sdf_b


def csg_union(
    sdf_a: torch.Tensor,
    sdf_b: torch.Tensor,
    smooth: bool = False,
    k: float = 0.1,
) -> torch.Tensor:
    """
    CSG union of two SDF tensors.

    Args:
        sdf_a: First SDF tensor
        sdf_b: Second SDF tensor
        smooth: Whether to use smooth union
        k: Smoothness parameter (if smooth=True)

    Returns:
        Union SDF
    """
    if smooth:
        return smooth_min(sdf_a, sdf_b, k)
    return torch.min(sdf_a, sdf_b)


def csg_intersection(
    sdf_a: torch.Tensor,
    sdf_b: torch.Tensor,
    smooth: bool = False,
    k: float = 0.1,
) -> torch.Tensor:
    """
    CSG intersection of two SDF tensors.

    Args:
        sdf_a: First SDF tensor
        sdf_b: Second SDF tensor
        smooth: Whether to use smooth intersection
        k: Smoothness parameter (if smooth=True)

    Returns:
        Intersection SDF
    """
    if smooth:
        return smooth_max(sdf_a, sdf_b, k)
    return torch.max(sdf_a, sdf_b)


def csg_subtraction(
    sdf_a: torch.Tensor,
    sdf_b: torch.Tensor,
    smooth: bool = False,
    k: float = 0.1,
) -> torch.Tensor:
    """
    CSG subtraction: A - B.

    Args:
        sdf_a: SDF of shape to keep
        sdf_b: SDF of shape to subtract
        smooth: Whether to use smooth subtraction
        k: Smoothness parameter (if smooth=True)

    Returns:
        Subtraction SDF
    """
    if smooth:
        return smooth_max(sdf_a, -sdf_b, k)
    return torch.max(sdf_a, -sdf_b)

"""
Operators for SDF manipulation and composition.

This module provides CSG (Constructive Solid Geometry) operations,
morphing utilities, and gradient-aware blending for neural SDFs.

Key classes:
    - SDFMorpher: Output-space interpolation between SDF models
    - SmoothCSG: Smooth boolean operations (union, intersection, subtraction)
    - MultiShapeBlend: Weighted blending of N shapes
    - GradientAwareComposition: CSG with proper gradient/normal handling
"""

from .csg import (
    BaseCSGOperation,
    SDFMorpher,
    SmoothCSG,
    MultiShapeBlend,
    GradientAwareComposition,
    smooth_min,
    smooth_max,
    smooth_min_exp,
    smooth_max_exp,
)

__all__ = [
    "BaseCSGOperation",
    "SDFMorpher",
    "SmoothCSG",
    "MultiShapeBlend",
    "GradientAwareComposition",
    "smooth_min",
    "smooth_max",
    "smooth_min_exp",
    "smooth_max_exp",
]

"""
Core module for PGA-INR.

Contains:
- Constants: Centralized default values and numeric constants
- Types: Type aliases for common tensor shapes
- Base: Abstract base classes defining standard interfaces
"""

from .constants import (
    # Numeric constants
    DEFAULT_EPS,
    DEFAULT_EPS_NORM,
    DEFAULT_GRADIENT_CLIP,
    # Model architecture defaults
    DEFAULT_HIDDEN_FEATURES,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_OMEGA_0,
    DEFAULT_OMEGA_HIDDEN,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_FREQUENCIES,
    # SDF/Surface defaults
    DEFAULT_SURFACE_THRESHOLD,
    DEFAULT_CONTACT_THRESHOLD,
    DEFAULT_SPHERE_TRACE_EPS,
    DEFAULT_SPHERE_TRACE_STEPS,
    # Training defaults
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WEIGHT_DECAY,
    # Loss weights
    DEFAULT_LAMBDA_SDF,
    DEFAULT_LAMBDA_EIKONAL,
    DEFAULT_LAMBDA_NORMAL,
    DEFAULT_LAMBDA_LATENT,
    # Output keys
    OUTPUT_SDF,
    OUTPUT_DENSITY,
    OUTPUT_RGB,
    OUTPUT_NORMAL,
    OUTPUT_GRADIENT,
    OUTPUT_LOCAL_COORDS,
    OUTPUT_FEATURES,
)

from .types import (
    # Type aliases
    TensorDict,
    ObserverPose,
    MotionTensor,
    RotationTensor,
    JointPositionTensor,
    TrajectoryTensor,
    # Shape documentation
    MOTION_SHAPE_CONVENTION,
)

from .base import (
    BaseINR,
    BaseGenerativeINR,
)

__all__ = [
    # Constants
    "DEFAULT_EPS",
    "DEFAULT_EPS_NORM",
    "DEFAULT_GRADIENT_CLIP",
    "DEFAULT_HIDDEN_FEATURES",
    "DEFAULT_HIDDEN_LAYERS",
    "DEFAULT_OMEGA_0",
    "DEFAULT_OMEGA_HIDDEN",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_NUM_FREQUENCIES",
    "DEFAULT_SURFACE_THRESHOLD",
    "DEFAULT_CONTACT_THRESHOLD",
    "DEFAULT_SPHERE_TRACE_EPS",
    "DEFAULT_SPHERE_TRACE_STEPS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_LAMBDA_SDF",
    "DEFAULT_LAMBDA_EIKONAL",
    "DEFAULT_LAMBDA_NORMAL",
    "DEFAULT_LAMBDA_LATENT",
    "OUTPUT_SDF",
    "OUTPUT_DENSITY",
    "OUTPUT_RGB",
    "OUTPUT_NORMAL",
    "OUTPUT_GRADIENT",
    "OUTPUT_LOCAL_COORDS",
    "OUTPUT_FEATURES",
    # Types
    "TensorDict",
    "ObserverPose",
    "MotionTensor",
    "RotationTensor",
    "JointPositionTensor",
    "TrajectoryTensor",
    "MOTION_SHAPE_CONVENTION",
    # Base classes
    "BaseINR",
    "BaseGenerativeINR",
]

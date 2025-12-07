"""
Loss functions for PGA-INR training.

Includes geometric losses (Eikonal constraint, normal alignment),
regularization terms, and motion-specific losses.
"""

from .geometric import (
    EikonalLoss,
    NormalAlignmentLoss,
    SDFLoss,
    GeometricConsistencyLoss,
    compute_gradient,
)
from .regularization import (
    LatentRegularization,
    LipschitzRegularization,
)
from .motion_losses import (
    MotionReconstructionLoss,
    VelocityLoss,
    FKPositionLoss,
    FootContactLoss,
    MotionDiffusionLoss,
    GeodesicRotationLoss,
)

__all__ = [
    # Geometric losses
    "EikonalLoss",
    "NormalAlignmentLoss",
    "SDFLoss",
    "GeometricConsistencyLoss",
    "compute_gradient",
    # Regularization
    "LatentRegularization",
    "LipschitzRegularization",
    # Motion losses
    "MotionReconstructionLoss",
    "VelocityLoss",
    "FKPositionLoss",
    "FootContactLoss",
    "MotionDiffusionLoss",
    "GeodesicRotationLoss",
]

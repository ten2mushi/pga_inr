"""
Loss functions for PGA-INR training.

Includes geometric losses (Eikonal constraint, normal alignment)
and regularization terms.
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

__all__ = [
    "EikonalLoss",
    "NormalAlignmentLoss",
    "SDFLoss",
    "GeometricConsistencyLoss",
    "compute_gradient",
    "LatentRegularization",
    "LipschitzRegularization",
]

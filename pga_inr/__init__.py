"""
PGA-INR: Projective Geometric Algebra Implicit Neural Representations

A PyTorch library for coordinate-free learning of 3D shapes with guaranteed
rotational invariance, compositionality, and geometric consistency.

Key Features:
- Full PGA algebra implementation (16-component multivectors)
- Observer-independent neural fields
- Geometric regularization (Eikonal + normal alignment)
- HyperNetwork-based generative models
- Sphere tracing renderer
- 4D spacetime extensions for articulated models

Example:
    >>> import pga_inr
    >>> model = pga_inr.models.PGA_INR(hidden_features=256)
    >>> outputs = model(query_points, observer_pose)
"""

__version__ = "0.1.0"
__author__ = "PGA-INR Contributors"

from . import pga
from . import models
from . import losses
from . import rendering
from . import data
from . import spacetime
from . import training
from . import utils

__all__ = [
    "pga",
    "models",
    "losses",
    "rendering",
    "data",
    "spacetime",
    "training",
    "utils",
]

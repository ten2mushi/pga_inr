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
- Diffusion-based motion generation with coordinate-free INR

API Design:
- All INR models inherit from BaseINR or BaseGenerativeINR
- All forward() methods return Dict[str, Tensor]
- Standard output keys: 'sdf'/'density', 'rgb', 'normal', 'local_coords'
- Motion tensors follow (B, J, D, T) shape convention

Example:
    >>> import pga_inr
    >>> from pga_inr.core import DEFAULT_HIDDEN_FEATURES
    >>> model = pga_inr.models.PGA_INR(hidden_features=DEFAULT_HIDDEN_FEATURES)
    >>> outputs = model(query_points, observer_pose)
    >>> sdf = outputs['density']  # or outputs['sdf'] for SDF models
"""

__version__ = "0.1.0"
__author__ = "PGA-INR Contributors"

from . import core
from . import pga
from . import models
from . import losses
from . import rendering
from . import data
from . import spacetime
from . import training
from . import utils
from . import diffusion
from . import ops
from . import slam

__all__ = [
    "core",
    "pga",
    "models",
    "losses",
    "rendering",
    "data",
    "spacetime",
    "training",
    "utils",
    "diffusion",
    "ops",
    "slam",
]

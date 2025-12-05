"""
Utility functions for PGA-INR.

Includes quaternion operations, visualization helpers, and configuration management.
"""

from .quaternion import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_from_axis_angle,
    quaternion_slerp,
    random_quaternion,
    normalize_quaternion,
)
from .visualization import (
    plot_sdf_slice,
    plot_point_cloud,
    render_turntable,
    save_mesh,
)
from .config import Config, load_config, save_config

__all__ = [
    # Quaternion operations
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_from_axis_angle",
    "quaternion_slerp",
    "random_quaternion",
    "normalize_quaternion",
    # Visualization
    "plot_sdf_slice",
    "plot_point_cloud",
    "render_turntable",
    "save_mesh",
    # Config
    "Config",
    "load_config",
    "save_config",
]

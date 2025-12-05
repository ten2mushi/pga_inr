"""
PGA (Projective Geometric Algebra) module.

Implements the full algebra of G(3,0,1) with 16-component multivectors,
geometric products, and rigid body transformations (motors).
"""

from .algebra import (
    Multivector,
    geometric_product,
    outer_product,
    inner_product,
    sandwich,
    exp,
    log,
    scalar,
    e0, e1, e2, e3,
    e01, e02, e03, e12, e31, e23,
    e012, e031, e023, e123,
    e0123,
)

from .motors import (
    Motor,
    rotor_from_axis_angle,
    translator_from_direction,
    motor_from_transform,
    motor_log,
    motor_exp,
)

from .primitives import (
    point,
    plane,
    line_from_points,
    line_from_plucker,
    join,
    meet,
)

from .transforms import (
    rotate,
    translate,
    transform_point,
    transform_line,
    transform_plane,
)

__all__ = [
    # Algebra
    "Multivector",
    "geometric_product",
    "outer_product",
    "inner_product",
    "sandwich",
    "exp",
    "log",
    "scalar",
    # Basis elements
    "e0", "e1", "e2", "e3",
    "e01", "e02", "e03", "e12", "e31", "e23",
    "e012", "e031", "e023", "e123",
    "e0123",
    # Motors
    "Motor",
    "rotor_from_axis_angle",
    "translator_from_direction",
    "motor_from_transform",
    "motor_log",
    "motor_exp",
    # Primitives
    "point",
    "plane",
    "line_from_points",
    "line_from_plucker",
    "join",
    "meet",
    # Transforms
    "rotate",
    "translate",
    "transform_point",
    "transform_line",
    "transform_plane",
]

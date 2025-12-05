"""
4D Spacetime extensions for PGA-INR.

Includes time-varying motors, articulated skeletons, and 4D neural fields
for dynamic scene reconstruction and physics simulation.
"""

from .interpolation import (
    quaternion_slerp,
    motor_slerp,
    screw_interpolation,
    squad,
    bezier_motor,
    catmull_rom_motor,
    hermite_motor,
    MotorTrajectory,
)
from .temporal_motor import (
    TemporalMotor,
    LearnableKeyframes,
    NeuralODEMotor,
    ControlledMotor,
    CompositMotor,
    PeriodicMotor,
)
from .kinematic_chain import (
    Joint,
    KinematicChain,
    ArticulatedMotor,
    LinearBlendSkinning,
    DualQuaternionSkinning,
)
from .spacetime_inr import (
    Spacetime_PGA_INR,
    DeformableNeuralField,
    ArticulatedNeuralField,
    TemporalConsistencyLoss,
    FlowFieldLoss,
    render_dynamic_scene,
)

__all__ = [
    # Interpolation
    "quaternion_slerp",
    "motor_slerp",
    "screw_interpolation",
    "squad",
    "bezier_motor",
    "catmull_rom_motor",
    "hermite_motor",
    "MotorTrajectory",
    # Temporal motors
    "TemporalMotor",
    "LearnableKeyframes",
    "NeuralODEMotor",
    "ControlledMotor",
    "CompositMotor",
    "PeriodicMotor",
    # Kinematic chains
    "Joint",
    "KinematicChain",
    "ArticulatedMotor",
    "LinearBlendSkinning",
    "DualQuaternionSkinning",
    # 4D INR
    "Spacetime_PGA_INR",
    "DeformableNeuralField",
    "ArticulatedNeuralField",
    "TemporalConsistencyLoss",
    "FlowFieldLoss",
    "render_dynamic_scene",
]

"""
Neural network models for PGA-INR.

Includes SIREN layers, PGA motor layers, HyperNetworks, and the main
PGA-INR and Generative PGA-INR architectures.
"""

from .layers import SineLayer, SirenMLP, PGAMotorLayer, HyperLayer, HyperNetwork
from .encoders import FourierEncoder, PositionalEncoder
from .inr import PGA_INR, PGA_INR_SDF, PGA_INR_SDF_V2, PGA_INR_NeRF, compose_scenes
from .generative import (
    Generative_PGA_INR,
    Generative_PGA_INR_SDF,
    LatentCodeBank,
    MotorFieldINR,
    DeformableSDF,
    LatentSpaceArithmetic,
    StyleTransfer,
    create_shape_manifold,
    interpolate_shapes,
)
from .phase_coherent import (
    SharedFirstLayerSIREN,
    PhaseCoherentSIREN_SDF,
    PhaseCoherentEnsemble,
    PhaseAlignmentTrainer,
)
from .motion import (
    ConditionalMotionPGAINR,
    MotionINRWrapper,
    MotionEncoder,
    TrajectoryEncoder,
    TimestepEmbedding,
    QueryEncoder,
    StyleEmbedding,
    LearnableStyleBank,
    ConditionalStyleEmbedding,
)

__all__ = [
    # Layers
    "SineLayer",
    "SirenMLP",
    "PGAMotorLayer",
    "HyperLayer",
    "HyperNetwork",
    # Encoders
    "FourierEncoder",
    "PositionalEncoder",
    # Models
    "PGA_INR",
    "PGA_INR_SDF",
    "PGA_INR_SDF_V2",
    "PGA_INR_NeRF",
    "Generative_PGA_INR",
    "Generative_PGA_INR_SDF",
    "LatentCodeBank",
    "MotorFieldINR",
    "DeformableSDF",
    "LatentSpaceArithmetic",
    "StyleTransfer",
    "create_shape_manifold",
    "interpolate_shapes",
    # Phase-coherent models
    "SharedFirstLayerSIREN",
    "PhaseCoherentSIREN_SDF",
    "PhaseCoherentEnsemble",
    "PhaseAlignmentTrainer",
    # Motion models
    "ConditionalMotionPGAINR",
    "MotionINRWrapper",
    "MotionEncoder",
    "TrajectoryEncoder",
    "TimestepEmbedding",
    "QueryEncoder",
    "StyleEmbedding",
    "LearnableStyleBank",
    "ConditionalStyleEmbedding",
    # Functions
    "compose_scenes",
]

"""
Neural network models for PGA-INR.

Includes SIREN layers, PGA motor layers, HyperNetworks, and the main
PGA-INR and Generative PGA-INR architectures.
"""

from .layers import SineLayer, SirenMLP, PGAMotorLayer, HyperLayer, HyperNetwork
from .encoders import FourierEncoder, PositionalEncoder
from .inr import PGA_INR, PGA_INR_SDF, PGA_INR_NeRF, compose_scenes
from .generative import Generative_PGA_INR, Generative_PGA_INR_SDF, LatentCodeBank

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
    "PGA_INR_NeRF",
    "Generative_PGA_INR",
    "Generative_PGA_INR_SDF",
    "LatentCodeBank",
    # Functions
    "compose_scenes",
]

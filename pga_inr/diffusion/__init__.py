"""
Diffusion module for PGA-INR.

Provides Gaussian diffusion processes with various noise schedules
for generative modeling of motion and other signals.
"""

from .noise_schedule import (
    NoiseSchedule,
    LinearSchedule,
    CosineSchedule,
    QuadraticSchedule,
    SigmoidSchedule,
    get_schedule,
)
from .gaussian_diffusion import GaussianDiffusion
from .motion_diffusion import MotionDiffusion

__all__ = [
    # Noise schedules
    "NoiseSchedule",
    "LinearSchedule",
    "CosineSchedule",
    "QuadraticSchedule",
    "SigmoidSchedule",
    "get_schedule",
    # Diffusion
    "GaussianDiffusion",
    "MotionDiffusion",
]

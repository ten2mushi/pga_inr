"""
Training utilities for PGA-INR.

Includes training loops, schedulers, curriculum learning, and logging.
"""

from .trainer import PGAINRTrainer, GenerativePGAINRTrainer, create_optimizer
from .schedulers import (
    WarmupScheduler,
    WarmupCosineScheduler,
    CyclicCosineScheduler,
    ExponentialDecayScheduler,
    ExponentialWarmupScheduler,
    SIRENScheduler,
    OneCycleLR,
    LatentCodeScheduler,
    create_scheduler,
)
from .curriculum import (
    AdaptiveTrainingSampler,
    CurriculumScheduler,
    LossWeightScheduler,
    DifficultyEstimator,
    ProgressiveTrainingConfig,
)

__all__ = [
    "PGAINRTrainer",
    "GenerativePGAINRTrainer",
    "create_optimizer",
    "WarmupScheduler",
    "WarmupCosineScheduler",
    "CyclicCosineScheduler",
    "ExponentialDecayScheduler",
    "ExponentialWarmupScheduler",
    "SIRENScheduler",
    "OneCycleLR",
    "LatentCodeScheduler",
    "create_scheduler",
    # Curriculum learning
    "AdaptiveTrainingSampler",
    "CurriculumScheduler",
    "LossWeightScheduler",
    "DifficultyEstimator",
    "ProgressiveTrainingConfig",
]

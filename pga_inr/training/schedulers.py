"""
Learning rate schedulers for PGA-INR training.

Provides custom schedulers tailored for implicit neural representations:
- Warmup schedulers
- Cyclic schedulers with geometric decay
- Multi-phase schedulers
"""

from typing import List, Optional
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.

    Useful for stabilizing early training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    Common choice for transformer-style training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]


class ExponentialDecayScheduler(_LRScheduler):
    """
    Exponential decay learning rate scheduler.

    LR decays exponentially: lr = base_lr * decay_rate^step
    Optionally includes a warmup phase.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        decay_rate: float = 0.99,
        decay_steps: int = 1,
        min_lr: float = 1e-7,
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            decay_rate: Multiplicative factor for decay (applied every decay_steps)
            decay_steps: Number of steps between each decay application
            min_lr: Minimum learning rate floor
            warmup_steps: Number of linear warmup steps (0 for no warmup)
            last_epoch: Last epoch index
        """
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Exponential decay phase
            decay_epoch = self.last_epoch - self.warmup_steps
            num_decays = decay_epoch // self.decay_steps
            factor = self.decay_rate ** num_decays
            return [
                max(self.min_lr, base_lr * factor)
                for base_lr in self.base_lrs
            ]


class CyclicCosineScheduler(_LRScheduler):
    """
    Cyclic cosine annealing with restarts.

    Learning rate follows cosine decay, then resets.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        cycle_length: int,
        num_cycles: int = 4,
        cycle_mult: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            cycle_length: Initial cycle length in steps
            num_cycles: Number of cycles
            cycle_mult: Factor to multiply cycle length after each cycle
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.cycle_length = cycle_length
        self.num_cycles = num_cycles
        self.cycle_mult = cycle_mult
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Determine current cycle
        current_length = self.cycle_length
        step = self.last_epoch
        cycle = 0

        while step >= current_length and cycle < self.num_cycles - 1:
            step -= current_length
            current_length = int(current_length * self.cycle_mult)
            cycle += 1

        # Position within cycle
        progress = step / current_length
        progress = min(progress, 1.0)

        # Cosine factor
        factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


class MultiPhaseScheduler(_LRScheduler):
    """
    Multi-phase scheduler with different rates per phase.

    Useful for curriculum learning or multi-stage training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        phase_steps: List[int],
        phase_lrs: List[float],
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            phase_steps: List of cumulative step counts for each phase
            phase_lrs: List of learning rate multipliers for each phase
            last_epoch: Last epoch index
        """
        self.phase_steps = phase_steps
        self.phase_lrs = phase_lrs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Find current phase
        phase = 0
        for i, steps in enumerate(self.phase_steps):
            if self.last_epoch < steps:
                phase = i
                break
            phase = i + 1

        if phase >= len(self.phase_lrs):
            phase = len(self.phase_lrs) - 1

        factor = self.phase_lrs[phase]
        return [base_lr * factor for base_lr in self.base_lrs]


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial decay learning rate scheduler.

    LR decays polynomially from initial to minimum value.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            total_steps: Total training steps
            power: Decay power (1.0 = linear)
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = min(self.last_epoch / self.total_steps, 1.0)
        factor = (1 - progress) ** self.power

        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


class ExponentialWarmupScheduler(_LRScheduler):
    """
    Exponential warmup followed by exponential decay.

    Smoother than linear warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        decay_rate: float = 0.99,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            decay_rate: Decay rate after warmup
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Exponential warmup
            factor = 1 - math.exp(-5 * (self.last_epoch + 1) / self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Exponential decay
            decay_steps = self.last_epoch - self.warmup_steps
            factor = self.decay_rate ** decay_steps
            return [
                max(self.min_lr, base_lr * factor)
                for base_lr in self.base_lrs
            ]


class SIRENScheduler(_LRScheduler):
    """
    Scheduler designed for SIREN networks.

    Starts with lower learning rate for first layer stability,
    then increases.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_scale: float = 0.1,
        ramp_steps: int = 1000,
        decay_start: int = 10000,
        decay_rate: float = 0.5,
        decay_steps: int = 5000,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            initial_scale: Initial LR scale factor
            ramp_steps: Steps to ramp up to full LR
            decay_start: Step to start decay
            decay_rate: Decay factor
            decay_steps: Steps between decay applications
            last_epoch: Last epoch index
        """
        self.initial_scale = initial_scale
        self.ramp_steps = ramp_steps
        self.decay_start = decay_start
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.ramp_steps:
            # Ramp up
            progress = self.last_epoch / self.ramp_steps
            factor = self.initial_scale + (1 - self.initial_scale) * progress
        elif self.last_epoch < self.decay_start:
            # Full learning rate
            factor = 1.0
        else:
            # Step decay
            decay_count = (self.last_epoch - self.decay_start) // self.decay_steps
            factor = self.decay_rate ** decay_count

        return [base_lr * factor for base_lr in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """
    1cycle learning rate policy.

    LR increases to max, then decreases to min over training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            max_lr: Maximum learning rate
            total_steps: Total training steps
            pct_start: Percentage of cycle spent increasing LR
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = max_lr / final_div_factor
            last_epoch: Last epoch index
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps

        if progress <= self.pct_start:
            # Increasing phase
            phase_progress = progress / self.pct_start
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * phase_progress
        else:
            # Decreasing phase
            phase_progress = (progress - self.pct_start) / (1 - self.pct_start)
            lr = self.max_lr - (self.max_lr - self.final_lr) * phase_progress

        return [lr for _ in self.base_lrs]


class LatentCodeScheduler(_LRScheduler):
    """
    Scheduler specifically for latent code optimization.

    Higher learning rate initially for faster convergence,
    then aggressive decay for refinement.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 500,
        decay_start: int = 2000,
        decay_rate: float = 0.9,
        decay_interval: int = 500,
        min_lr: float = 1e-5,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Warmup period
            decay_start: Step to start decay
            decay_rate: Decay factor
            decay_interval: Steps between decay
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_steps
        elif self.last_epoch < self.decay_start:
            factor = 1.0
        else:
            # Step decay
            decay_count = (self.last_epoch - self.decay_start) // self.decay_interval
            factor = self.decay_rate ** decay_count

        return [
            max(self.min_lr, base_lr * factor)
            for base_lr in self.base_lrs
        ]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    **kwargs
) -> _LRScheduler:
    """
    Factory function for creating schedulers.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific arguments

    Returns:
        Configured scheduler
    """
    schedulers = {
        'warmup': WarmupScheduler,
        'warmup_cosine': WarmupCosineScheduler,
        'cyclic_cosine': CyclicCosineScheduler,
        'multi_phase': MultiPhaseScheduler,
        'polynomial': PolynomialDecayScheduler,
        'exponential_decay': ExponentialDecayScheduler,
        'exponential_warmup': ExponentialWarmupScheduler,
        'siren': SIRENScheduler,
        'one_cycle': OneCycleLR,
        'latent_code': LatentCodeScheduler,
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return schedulers[scheduler_type](optimizer, **kwargs)

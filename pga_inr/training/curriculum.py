"""
Curriculum Learning Utilities for SDF Training.

Implements progressive training strategies that start with simple tasks
and gradually increase complexity. This improves convergence and final
quality for neural SDF learning.

Key concepts:
- AdaptiveTrainingSampler: Samples more from high-error regions
- CurriculumScheduler: Manages stage progression and hyperparameters
- LossWeightScheduler: Progressive adjustment of loss term weights
- DifficultyEstimator: Estimates sample difficulty for curriculum

Based on findings that:
1. Starting with uniform sampling helps establish global structure
2. Progressive surface focus improves detail quality
3. Adaptive sampling accelerates convergence in difficult regions
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTrainingSampler:
    """
    Training-integrated adaptive sampler.

    Tracks reconstruction errors during training and biases sampling
    toward regions with higher historical errors. This focuses training
    on the most difficult parts of the shape.

    Key features:
    - Seamless integration with training loop
    - Configurable error weighting
    - Automatic update interval management
    - Support for mixed sampling strategies
    """

    def __init__(
        self,
        base_sampler,
        error_weight: float = 0.5,
        update_interval: int = 100,
        grid_resolution: int = 32,
        error_decay: float = 0.95,
        min_uniform_ratio: float = 0.2,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            base_sampler: Base sampler for uniform/structured sampling
            error_weight: Weight for error-based sampling (0-1)
            update_interval: Steps between error grid updates
            grid_resolution: Resolution of error tracking grid
            error_decay: Decay factor for historical errors
            min_uniform_ratio: Minimum fraction of uniform samples
            device: Device for computations
        """
        self.base_sampler = base_sampler
        self.error_weight = error_weight
        self.update_interval = update_interval
        self.min_uniform_ratio = min_uniform_ratio
        self.device = device

        # Error tracking grid
        self.grid_resolution = grid_resolution
        self.error_decay = error_decay
        self.error_grid = torch.ones(
            grid_resolution, grid_resolution, grid_resolution,
            device=device
        )

        # Statistics
        self._step = 0
        self._pending_errors = []
        self._pending_points = []

    def record_loss(
        self,
        points: torch.Tensor,
        losses: torch.Tensor,
    ):
        """
        Record per-point losses for future sampling bias.

        Args:
            points: Sample points (N, 3) or (B, N, 3)
            losses: Per-point losses (N,) or (B, N)
        """
        # Flatten if batched
        if points.dim() == 3:
            points = points.reshape(-1, 3)
            losses = losses.reshape(-1)

        self._pending_points.append(points.detach())
        self._pending_errors.append(losses.detach())

        # Update grid periodically
        self._step += 1
        if self._step % self.update_interval == 0:
            self._update_error_grid()

    def _update_error_grid(self):
        """Update error grid from accumulated samples."""
        if not self._pending_points:
            return

        # Concatenate all pending data
        all_points = torch.cat(self._pending_points, dim=0)
        all_errors = torch.cat(self._pending_errors, dim=0)

        # Get bounds from base sampler if available
        bounds = getattr(self.base_sampler, 'bounds', (-1.0, 1.0))

        # Convert points to grid indices
        normalized = (all_points - bounds[0]) / (bounds[1] - bounds[0])
        indices = (normalized * self.grid_resolution).long()
        indices = indices.clamp(0, self.grid_resolution - 1)

        # Compute flat indices
        flat_idx = (
            indices[:, 0] * self.grid_resolution ** 2 +
            indices[:, 1] * self.grid_resolution +
            indices[:, 2]
        )

        # Accumulate errors
        error_sum = torch.zeros(
            self.grid_resolution ** 3, device=self.device
        ).scatter_add_(0, flat_idx, all_errors)

        count = torch.zeros(
            self.grid_resolution ** 3, device=self.device
        ).scatter_add_(0, flat_idx, torch.ones_like(all_errors))

        # Compute mean errors
        mask = count > 0
        new_errors = torch.zeros_like(error_sum)
        new_errors[mask] = error_sum[mask] / count[mask]

        # Reshape
        new_errors = new_errors.reshape(
            self.grid_resolution, self.grid_resolution, self.grid_resolution
        )
        update_mask = mask.reshape(
            self.grid_resolution, self.grid_resolution, self.grid_resolution
        )

        # EMA update
        self.error_grid = torch.where(
            update_mask,
            self.error_decay * self.error_grid + (1 - self.error_decay) * new_errors,
            self.error_grid
        )

        # Clear pending
        self._pending_points.clear()
        self._pending_errors.clear()

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Sample points with adaptive error-based weighting.

        Args:
            num_samples: Total samples per batch
            batch_size: Number of batches

        Returns:
            Points (batch_size, num_samples, 3)
        """
        # Compute number of each sample type
        num_uniform = max(
            int(num_samples * self.min_uniform_ratio),
            int(num_samples * (1 - self.error_weight))
        )
        num_error_weighted = num_samples - num_uniform

        all_samples = []

        for _ in range(batch_size):
            batch_points = []

            # Base/uniform samples
            if num_uniform > 0:
                uniform_pts = self.base_sampler.sample(num_uniform, 1)
                if uniform_pts.dim() == 3:
                    uniform_pts = uniform_pts.squeeze(0)
                batch_points.append(uniform_pts)

            # Error-weighted samples
            if num_error_weighted > 0:
                error_pts = self._sample_from_error_grid(num_error_weighted)
                batch_points.append(error_pts)

            # Combine and shuffle
            combined = torch.cat(batch_points, dim=0)
            perm = torch.randperm(len(combined), device=self.device)
            combined = combined[perm]

            all_samples.append(combined)

        return torch.stack(all_samples, dim=0)

    def _sample_from_error_grid(self, num_samples: int) -> torch.Tensor:
        """Sample points biased toward high-error cells."""
        # Compute sampling probabilities
        probs = self.error_grid.flatten()
        probs = probs / probs.sum()

        # Sample cells
        cell_indices = torch.multinomial(probs, num_samples, replacement=True)

        # Convert to 3D indices
        x_idx = cell_indices // (self.grid_resolution ** 2)
        y_idx = (cell_indices // self.grid_resolution) % self.grid_resolution
        z_idx = cell_indices % self.grid_resolution

        # Random offset within cells
        offsets = torch.rand(num_samples, 3, device=self.device)

        # Convert to world coordinates
        bounds = getattr(self.base_sampler, 'bounds', (-1.0, 1.0))
        cell_size = (bounds[1] - bounds[0]) / self.grid_resolution

        cell_coords = torch.stack([x_idx, y_idx, z_idx], dim=-1).float()
        points = (cell_coords + offsets) * cell_size + bounds[0]

        return points


class CurriculumScheduler:
    """
    Curriculum learning scheduler for SDF training.

    Manages progression through training stages, each with different
    hyperparameters and sampling strategies.

    Typical curriculum:
    Stage 1 (Coarse): Uniform sampling, high eikonal weight
    Stage 2 (Detail): Surface-biased sampling, balanced weights
    Stage 3 (Fine): Adaptive sampling, low regularization

    Each stage can specify:
    - Surface sampling ratio
    - Loss weights (sdf, eikonal, normal)
    - Learning rate multiplier
    - Importance sampling parameters
    """

    def __init__(
        self,
        total_steps: int,
        num_stages: int = 3,
        stage_ratios: Optional[List[float]] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            total_steps: Total training steps
            num_stages: Number of curriculum stages
            stage_ratios: Relative duration of each stage (will be normalized)
            warmup_steps: Initial warmup steps before curriculum starts
        """
        self.total_steps = total_steps
        self.num_stages = num_stages
        self.warmup_steps = warmup_steps

        # Compute stage boundaries
        if stage_ratios is None:
            stage_ratios = [1.0] * num_stages
        stage_ratios = [r / sum(stage_ratios) for r in stage_ratios]

        curriculum_steps = total_steps - warmup_steps
        self.stage_boundaries = [warmup_steps]
        cumulative = warmup_steps
        for ratio in stage_ratios:
            cumulative += int(ratio * curriculum_steps)
            self.stage_boundaries.append(cumulative)

        # Default stage configs
        self._stage_configs = self._default_stage_configs()

    def _default_stage_configs(self) -> List[Dict[str, Any]]:
        """Generate default stage configurations."""
        configs = []

        for stage in range(self.num_stages):
            progress = stage / max(self.num_stages - 1, 1)

            # Surface ratio: 0 -> 0.8
            surface_ratio = progress * 0.8

            # Eikonal weight: 1.0 -> 0.1
            eikonal_weight = 1.0 - 0.9 * progress

            # Normal weight: 0.0 -> 0.5
            normal_weight = 0.5 * progress

            # Learning rate multiplier: 1.0 -> 0.1
            lr_multiplier = 1.0 - 0.9 * progress

            # Surface band: 0.2 -> 0.05
            surface_band = 0.2 - 0.15 * progress

            configs.append({
                'surface_ratio': surface_ratio,
                'eikonal_weight': eikonal_weight,
                'normal_weight': normal_weight,
                'lr_multiplier': lr_multiplier,
                'surface_band': surface_band,
            })

        return configs

    def set_stage_config(
        self,
        stage: int,
        config: Dict[str, Any],
    ):
        """
        Override configuration for a specific stage.

        Args:
            stage: Stage index (0-indexed)
            config: Configuration dictionary
        """
        if stage < len(self._stage_configs):
            self._stage_configs[stage].update(config)

    def get_current_stage(self, step: int) -> int:
        """Get current stage index based on step."""
        for i, boundary in enumerate(self.stage_boundaries[1:]):
            if step < boundary:
                return i
        return self.num_stages - 1

    def get_stage_progress(self, step: int) -> float:
        """Get progress within current stage (0 to 1)."""
        stage = self.get_current_stage(step)
        start = self.stage_boundaries[stage]
        end = self.stage_boundaries[stage + 1]
        return (step - start) / max(end - start, 1)

    def get_stage_params(self, step: int) -> Dict[str, Any]:
        """
        Get hyperparameters for current training step.

        Interpolates between stage configurations for smooth transitions.

        Args:
            step: Current training step

        Returns:
            Dictionary of hyperparameters
        """
        # Warmup phase
        if step < self.warmup_steps:
            warmup_progress = step / max(self.warmup_steps, 1)
            return {
                'surface_ratio': 0.0,
                'eikonal_weight': 1.0,
                'normal_weight': 0.0,
                'lr_multiplier': warmup_progress,
                'surface_band': 0.2,
                'stage': -1,
            }

        stage = self.get_current_stage(step)
        progress = self.get_stage_progress(step)

        # Get current and next stage configs
        current = self._stage_configs[stage]
        if stage < self.num_stages - 1:
            next_stage = self._stage_configs[stage + 1]
        else:
            next_stage = current

        # Interpolate with cosine schedule for smooth transition
        t = 0.5 * (1 - math.cos(progress * math.pi))

        params = {}
        for key in current:
            if isinstance(current[key], (int, float)):
                params[key] = (1 - t) * current[key] + t * next_stage[key]
            else:
                params[key] = current[key]

        params['stage'] = stage

        return params

    def should_transition(self, step: int) -> bool:
        """Check if we're at a stage transition point."""
        return step in self.stage_boundaries


class LossWeightScheduler:
    """
    Schedule loss term weights during training.

    Manages the relative importance of different loss terms
    (SDF reconstruction, eikonal, normal, etc.) across training.
    """

    def __init__(
        self,
        sdf_weight: float = 1.0,
        eikonal_weight: float = 0.1,
        normal_weight: float = 0.1,
        grad_weight: float = 0.0,
        total_steps: int = 10000,
        schedule_type: str = 'constant',
    ):
        """
        Args:
            sdf_weight: Weight for SDF reconstruction loss
            eikonal_weight: Weight for eikonal constraint
            normal_weight: Weight for normal alignment
            grad_weight: Weight for gradient smoothness
            total_steps: Total training steps
            schedule_type: 'constant', 'linear', or 'cosine'
        """
        self.initial_weights = {
            'sdf': sdf_weight,
            'eikonal': eikonal_weight,
            'normal': normal_weight,
            'grad': grad_weight,
        }
        self.total_steps = total_steps
        self.schedule_type = schedule_type

        # Final weights (for scheduled types)
        self.final_weights = {
            'sdf': sdf_weight,
            'eikonal': eikonal_weight * 0.1,  # Decrease regularization
            'normal': normal_weight * 2.0,    # Increase normal importance
            'grad': grad_weight,
        }

    def get_weights(self, step: int) -> Dict[str, float]:
        """
        Get loss weights for current step.

        Args:
            step: Current training step

        Returns:
            Dictionary of loss weights
        """
        if self.schedule_type == 'constant':
            return self.initial_weights.copy()

        progress = min(step / self.total_steps, 1.0)

        if self.schedule_type == 'linear':
            t = progress
        elif self.schedule_type == 'cosine':
            t = 0.5 * (1 - math.cos(progress * math.pi))
        else:
            t = 0

        weights = {}
        for key in self.initial_weights:
            weights[key] = (
                (1 - t) * self.initial_weights[key] +
                t * self.final_weights[key]
            )

        return weights

    def set_final_weight(self, key: str, value: float):
        """Set final weight for a loss term."""
        self.final_weights[key] = value


class DifficultyEstimator:
    """
    Estimate sample difficulty for curriculum learning.

    Tracks which samples/regions are harder to fit and provides
    difficulty scores that can be used for adaptive sampling.
    """

    def __init__(
        self,
        ema_decay: float = 0.99,
        difficulty_threshold: float = 0.5,
    ):
        """
        Args:
            ema_decay: Decay for exponential moving average
            difficulty_threshold: Threshold for "hard" samples
        """
        self.ema_decay = ema_decay
        self.difficulty_threshold = difficulty_threshold

        self._running_mean = None
        self._running_var = None
        self._num_updates = 0

    def update(self, losses: torch.Tensor):
        """
        Update difficulty statistics from batch losses.

        Args:
            losses: Per-sample losses (N,)
        """
        batch_mean = losses.mean().item()
        batch_var = losses.var().item()

        if self._running_mean is None:
            self._running_mean = batch_mean
            self._running_var = batch_var
        else:
            self._running_mean = (
                self.ema_decay * self._running_mean +
                (1 - self.ema_decay) * batch_mean
            )
            self._running_var = (
                self.ema_decay * self._running_var +
                (1 - self.ema_decay) * batch_var
            )

        self._num_updates += 1

    def get_difficulty_scores(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized difficulty scores for samples.

        Args:
            losses: Per-sample losses (N,)

        Returns:
            Difficulty scores (N,) in [0, 1]
        """
        if self._running_mean is None:
            return torch.zeros_like(losses)

        # Standardize losses
        std = max(self._running_var ** 0.5, 1e-8)
        z_scores = (losses - self._running_mean) / std

        # Convert to [0, 1] via sigmoid
        scores = torch.sigmoid(z_scores)

        return scores

    def is_hard_sample(self, loss: float) -> bool:
        """Check if a loss value indicates a hard sample."""
        if self._running_mean is None:
            return False

        std = max(self._running_var ** 0.5, 1e-8)
        z_score = (loss - self._running_mean) / std

        return z_score > self.difficulty_threshold

    @property
    def current_mean(self) -> float:
        """Get current running mean loss."""
        return self._running_mean or 0.0

    @property
    def current_std(self) -> float:
        """Get current running standard deviation."""
        return (self._running_var or 0.0) ** 0.5


class ProgressiveTrainingConfig:
    """
    Configuration for progressive SDF training.

    Bundles together all curriculum-related settings.
    """

    def __init__(
        self,
        total_steps: int = 10000,
        num_stages: int = 3,
        use_adaptive_sampling: bool = True,
        use_curriculum_loss: bool = True,
        initial_surface_ratio: float = 0.0,
        final_surface_ratio: float = 0.8,
        initial_lr: float = 1e-4,
        final_lr: float = 1e-5,
        warmup_steps: int = 500,
    ):
        """
        Args:
            total_steps: Total training steps
            num_stages: Number of curriculum stages
            use_adaptive_sampling: Enable error-based adaptive sampling
            use_curriculum_loss: Enable scheduled loss weights
            initial_surface_ratio: Starting surface sample ratio
            final_surface_ratio: Final surface sample ratio
            initial_lr: Starting learning rate
            final_lr: Final learning rate
            warmup_steps: Learning rate warmup steps
        """
        self.total_steps = total_steps
        self.num_stages = num_stages
        self.use_adaptive_sampling = use_adaptive_sampling
        self.use_curriculum_loss = use_curriculum_loss
        self.initial_surface_ratio = initial_surface_ratio
        self.final_surface_ratio = final_surface_ratio
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps

    def create_scheduler(self) -> CurriculumScheduler:
        """Create a CurriculumScheduler with this config."""
        scheduler = CurriculumScheduler(
            total_steps=self.total_steps,
            num_stages=self.num_stages,
            warmup_steps=self.warmup_steps,
        )

        # Configure stages for progressive surface focus
        for stage in range(self.num_stages):
            progress = stage / max(self.num_stages - 1, 1)
            surface_ratio = (
                self.initial_surface_ratio +
                progress * (self.final_surface_ratio - self.initial_surface_ratio)
            )
            lr_multiplier = (
                1.0 - progress * (1.0 - self.final_lr / self.initial_lr)
            )
            scheduler.set_stage_config(stage, {
                'surface_ratio': surface_ratio,
                'lr_multiplier': lr_multiplier,
            })

        return scheduler

    def create_loss_scheduler(self) -> LossWeightScheduler:
        """Create a LossWeightScheduler with default curriculum settings."""
        return LossWeightScheduler(
            sdf_weight=1.0,
            eikonal_weight=0.1,
            normal_weight=0.1,
            total_steps=self.total_steps,
            schedule_type='cosine' if self.use_curriculum_loss else 'constant',
        )

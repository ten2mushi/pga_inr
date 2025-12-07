"""
Noise schedules for diffusion models.

Provides various beta schedules for controlling the diffusion process:
- Linear schedule: Standard DDPM schedule
- Cosine schedule: Improved schedule from "Improved DDPM" paper

Reference:
    Ho et al., "Denoising Diffusion Probabilistic Models"
    Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
"""

from abc import ABC, abstractmethod
from typing import Optional
import math
import torch
import numpy as np


class NoiseSchedule(ABC):
    """
    Base class for diffusion noise schedules.

    Precomputes and stores all necessary coefficients for the diffusion process.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        device: torch.device = None
    ):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            device: Device for tensors
        """
        self.num_timesteps = num_timesteps
        self.device = device or torch.device('cpu')

        # Compute betas
        betas = self._compute_betas()
        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)

        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=self.device),
            self.alphas_cumprod[:-1]
        ])
        self.alphas_cumprod_next = torch.cat([
            self.alphas_cumprod[1:],
            torch.tensor([0.0], device=self.device)
        ])

        # Precompute coefficients for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Precompute coefficients for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        # Clamp to avoid log(0)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) /
            (1.0 - self.alphas_cumprod)
        )

    @abstractmethod
    def _compute_betas(self) -> np.ndarray:
        """Compute the beta schedule. Must be implemented by subclasses."""
        pass

    def to(self, device: torch.device) -> 'NoiseSchedule':
        """Move schedule tensors to device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.alphas_cumprod_next = self.alphas_cumprod_next.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get cumulative product of alphas at timestep t.

        Args:
            t: Timestep indices of shape (batch_size,)

        Returns:
            alpha_bar values of shape (batch_size,)
        """
        return self.alphas_cumprod[t]

    def sample_timesteps(
        self,
        batch_size: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Sample random timesteps for training.

        Args:
            batch_size: Number of timesteps to sample
            device: Device for output tensor

        Returns:
            Random timesteps of shape (batch_size,)
        """
        device = device or self.device
        return torch.randint(
            0, self.num_timesteps, (batch_size,),
            device=device, dtype=torch.long
        )

    def get_inference_timesteps(
        self,
        num_steps: int,
        method: str = 'uniform'
    ) -> torch.Tensor:
        """
        Get timesteps for inference.

        Args:
            num_steps: Number of inference steps
            method: Selection method ('uniform', 'quadratic')

        Returns:
            Timesteps for inference of shape (num_steps,)
        """
        if method == 'uniform':
            # Uniform spacing
            step_size = self.num_timesteps // num_steps
            timesteps = torch.arange(
                0, self.num_timesteps, step_size,
                device=self.device
            )[:num_steps]
            timesteps = torch.flip(timesteps, dims=[0])
        elif method == 'quadratic':
            # Quadratic spacing (more steps near t=0)
            timesteps = (
                np.linspace(0, np.sqrt(self.num_timesteps * 0.8), num_steps) ** 2
            ).astype(int)
            timesteps = torch.tensor(timesteps, device=self.device)
            timesteps = torch.flip(timesteps, dims=[0])
        else:
            raise ValueError(f"Unknown method: {method}")

        return timesteps

    def _extract(
        self,
        arr: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: tuple
    ) -> torch.Tensor:
        """
        Extract values from array at given timesteps and reshape for broadcasting.

        Args:
            arr: Array to extract from
            timesteps: Timestep indices
            broadcast_shape: Shape to broadcast to

        Returns:
            Extracted and reshaped values
        """
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)


class LinearSchedule(NoiseSchedule):
    """
    Linear beta schedule from the original DDPM paper.

    beta_t = beta_start + t * (beta_end - beta_start) / (T - 1)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = None
    ):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            device: Device for tensors
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> np.ndarray:
        return np.linspace(
            self.beta_start, self.beta_end, self.num_timesteps
        )


class CosineSchedule(NoiseSchedule):
    """
    Cosine beta schedule from "Improved DDPM".

    This schedule preserves more signal at high noise levels, which is
    beneficial for many generation tasks including motion.

    alpha_bar(t) = cos((t/T + s) / (1+s) * pi/2)^2
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        s: float = 0.008,
        max_beta: float = 0.999,
        device: torch.device = None
    ):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            s: Small offset to prevent beta_0 = 0
            max_beta: Maximum beta value to prevent instability
            device: Device for tensors
        """
        self.s = s
        self.max_beta = max_beta
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> np.ndarray:
        def alpha_bar(t):
            return math.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2

        betas = []
        for i in range(self.num_timesteps):
            t1 = i / self.num_timesteps
            t2 = (i + 1) / self.num_timesteps
            beta = min(1 - alpha_bar(t2) / alpha_bar(t1), self.max_beta)
            betas.append(beta)

        return np.array(betas)


class QuadraticSchedule(NoiseSchedule):
    """
    Quadratic beta schedule.

    beta_t = beta_start + (t/T)^2 * (beta_end - beta_start)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = None
    ):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            device: Device for tensors
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> np.ndarray:
        return (
            np.linspace(
                self.beta_start ** 0.5,
                self.beta_end ** 0.5,
                self.num_timesteps
            ) ** 2
        )


class SigmoidSchedule(NoiseSchedule):
    """
    Sigmoid beta schedule.

    Provides smooth transition with adjustable steepness.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        steepness: float = 6.0,
        device: torch.device = None
    ):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            steepness: Sigmoid steepness parameter
            device: Device for tensors
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steepness = steepness
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> np.ndarray:
        x = np.linspace(-self.steepness, self.steepness, self.num_timesteps)
        sigmoid = 1 / (1 + np.exp(-x))
        return self.beta_start + (self.beta_end - self.beta_start) * sigmoid


def get_schedule(
    name: str,
    num_timesteps: int = 1000,
    device: torch.device = None,
    **kwargs
) -> NoiseSchedule:
    """
    Factory function to create noise schedules.

    Args:
        name: Schedule name ('linear', 'cosine', 'quadratic', 'sigmoid')
        num_timesteps: Number of diffusion timesteps
        device: Device for tensors
        **kwargs: Additional arguments for the schedule

    Returns:
        NoiseSchedule instance
    """
    schedules = {
        'linear': LinearSchedule,
        'cosine': CosineSchedule,
        'quadratic': QuadraticSchedule,
        'sigmoid': SigmoidSchedule,
    }

    if name not in schedules:
        raise ValueError(f"Unknown schedule: {name}. Available: {list(schedules.keys())}")

    return schedules[name](num_timesteps=num_timesteps, device=device, **kwargs)

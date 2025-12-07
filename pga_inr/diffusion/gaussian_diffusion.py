"""
Gaussian diffusion process for generative modeling.

Implements DDPM and DDIM sampling for diffusion models.

Reference:
    Ho et al., "Denoising Diffusion Probabilistic Models"
    Song et al., "Denoising Diffusion Implicit Models"
"""

from typing import Dict, Optional, Tuple, Callable
import torch
import torch.nn as nn

from .noise_schedule import NoiseSchedule


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process with DDPM/DDIM sampling.

    Supports multiple prediction types:
    - 'x0': Model directly predicts clean data x_0
    - 'epsilon': Model predicts noise epsilon
    - 'v': Model predicts velocity v = sqrt(alpha) * epsilon - sqrt(1-alpha) * x_0
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: NoiseSchedule,
        prediction_type: str = 'x0',
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Args:
            model: Denoising model that takes (noisy_x, timestep, condition) -> prediction
            schedule: Noise schedule instance
            prediction_type: What the model predicts ('x0', 'epsilon', 'v')
            clip_denoised: Whether to clip denoised samples
            clip_range: Range for clipping
        """
        super().__init__()

        self.model = model
        self.schedule = schedule
        self.prediction_type = prediction_type
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range

    def to(self, device: torch.device) -> 'GaussianDiffusion':
        """Move to device."""
        super().to(device)
        self.schedule = self.schedule.to(device)
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t given x_0.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_0: Clean data of shape (batch, ...)
            t: Timesteps of shape (batch,)
            noise: Optional pre-sampled noise

        Returns:
            (x_t, noise): Noisy data and the noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def predict_x0_from_epsilon(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted epsilon.

        x_0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
        """
        sqrt_recip_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * epsilon

    def predict_epsilon_from_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict epsilon from x_t and predicted x_0.

        epsilon = (x_t - sqrt(alpha_bar) * x_0) / sqrt(1 - alpha_bar)
        """
        sqrt_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_alphas_cumprod, t, x_t.shape
        )
        sqrt_one_minus_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        return (x_t - sqrt_alpha_bar * x_0) / sqrt_one_minus_alpha_bar.clamp(min=1e-8)

    def predict_v_from_x0_epsilon(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity parameterization v.

        v = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x_0
        """
        sqrt_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alpha_bar * epsilon - sqrt_one_minus_alpha_bar * x_0

    def predict_x0_epsilon_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict x_0 and epsilon from velocity v.
        """
        sqrt_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_alphas_cumprod, t, x_t.shape
        )
        sqrt_one_minus_alpha_bar = self.schedule._extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        x_0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * v
        epsilon = sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * v

        return x_0, epsilon

    def model_predictions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model predictions and convert to (x_0, epsilon).

        Args:
            x_t: Noisy data
            t: Timesteps
            condition: Conditioning dictionary

        Returns:
            (pred_x0, pred_epsilon): Predicted clean data and noise
        """
        # Get model output
        model_output = self.model(x_t, t, condition)

        # Convert to x_0 and epsilon based on prediction type
        if self.prediction_type == 'x0':
            pred_x0 = model_output
            pred_epsilon = self.predict_epsilon_from_x0(x_t, t, pred_x0)
        elif self.prediction_type == 'epsilon':
            pred_epsilon = model_output
            pred_x0 = self.predict_x0_from_epsilon(x_t, t, pred_epsilon)
        elif self.prediction_type == 'v':
            pred_x0, pred_epsilon = self.predict_x0_epsilon_from_v(x_t, t, model_output)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Optionally clip x_0
        if self.clip_denoised:
            pred_x0 = pred_x0.clamp(self.clip_range[0], self.clip_range[1])

        return pred_x0, pred_epsilon

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).

        Returns:
            (posterior_mean, posterior_variance, posterior_log_variance)
        """
        posterior_mean_coef1 = self.schedule._extract(
            self.schedule.posterior_mean_coef1, t, x_t.shape
        )
        posterior_mean_coef2 = self.schedule._extract(
            self.schedule.posterior_mean_coef2, t, x_t.shape
        )

        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t

        posterior_variance = self.schedule._extract(
            self.schedule.posterior_variance, t, x_t.shape
        )
        posterior_log_variance = self.schedule._extract(
            self.schedule.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance

    def training_loss(
        self,
        x_0: torch.Tensor,
        condition: Optional[Dict] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.

        Args:
            x_0: Clean data of shape (batch, ...)
            condition: Conditioning dictionary
            noise: Optional pre-sampled noise

        Returns:
            (loss, metrics): Loss tensor and metrics dictionary
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = self.schedule.sample_timesteps(batch_size, device=x_0.device)

        # Add noise
        x_t, noise = self.q_sample(x_0, t, noise)

        # Get model prediction
        model_output = self.model(x_t, t, condition)

        # Compute target based on prediction type
        if self.prediction_type == 'x0':
            target = x_0
        elif self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'v':
            target = self.predict_v_from_x0_epsilon(x_0, t, noise)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # MSE loss
        loss = ((model_output - target) ** 2).mean()

        metrics = {
            'diffusion_loss': loss.item(),
            'mse': loss.item(),
        }

        return loss, metrics

    @torch.no_grad()
    def p_sample_ddpm(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Single DDPM sampling step.

        Samples x_{t-1} from p(x_{t-1} | x_t).

        Args:
            x_t: Current noisy sample
            t: Current timestep
            condition: Conditioning dictionary

        Returns:
            x_{t-1}: Sample at previous timestep
        """
        # Get predictions
        pred_x0, pred_epsilon = self.model_predictions(x_t, t, condition)

        # Compute posterior
        posterior_mean, posterior_variance, _ = self.q_posterior(pred_x0, x_t, t)

        # Sample
        noise = torch.randn_like(x_t)

        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

        return x_prev

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        condition: Optional[Dict] = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Single DDIM sampling step.

        DDIM allows deterministic (eta=0) or stochastic sampling.

        Args:
            x_t: Current noisy sample
            t: Current timestep
            t_prev: Previous timestep
            condition: Conditioning dictionary
            eta: DDIM eta parameter (0 = deterministic)

        Returns:
            x_{t_prev}: Sample at previous timestep
        """
        # Get predictions
        pred_x0, pred_epsilon = self.model_predictions(x_t, t, condition)

        # Get alpha values
        alpha_bar_t = self.schedule._extract(
            self.schedule.alphas_cumprod, t, x_t.shape
        )

        # Handle t_prev = -1 case (final step)
        # Need to match shapes for torch.where
        t_prev_cond = (t_prev >= 0).reshape(-1, *([1] * (x_t.dim() - 1)))  # (batch, 1, 1, ...)
        alpha_bar_t_prev = torch.where(
            t_prev_cond,
            self.schedule._extract(self.schedule.alphas_cumprod, t_prev.clamp(min=0), x_t.shape),
            torch.ones_like(alpha_bar_t)
        )

        # DDIM sampling formula
        sigma = (
            eta *
            torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) *
            torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
        )

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * pred_epsilon

        # Random noise
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        # Sample
        x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[Dict] = None,
        method: str = 'ddim',
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        progress_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Generate samples from noise.

        Args:
            shape: Shape of samples to generate (batch, ...)
            condition: Conditioning dictionary
            method: Sampling method ('ddpm', 'ddim')
            num_steps: Number of sampling steps (None = use all timesteps)
            eta: DDIM eta parameter
            progress_callback: Optional callback for progress

        Returns:
            Generated samples
        """
        device = self.schedule.device

        # Start from noise
        x = torch.randn(shape, device=device)

        # Get timesteps
        if num_steps is None:
            num_steps = self.schedule.num_timesteps

        timesteps = self.schedule.get_inference_timesteps(num_steps)

        # Sampling loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            if method == 'ddpm':
                x = self.p_sample_ddpm(x, t_batch, condition)
            elif method == 'ddim':
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1)
                t_prev_batch = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
                x = self.p_sample_ddim(x, t_batch, t_prev_batch, condition, eta)
            else:
                raise ValueError(f"Unknown method: {method}")

            if progress_callback is not None:
                progress_callback(i, len(timesteps), x)

        return x

    @torch.no_grad()
    def sample_loop(
        self,
        x_T: torch.Tensor,
        condition: Optional[Dict] = None,
        method: str = 'ddim',
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Sample from a given initial noise.

        Args:
            x_T: Initial noise
            condition: Conditioning dictionary
            method: Sampling method
            num_steps: Number of steps
            eta: DDIM eta
            return_intermediates: Whether to return all intermediate samples

        Returns:
            Final sample or list of intermediate samples
        """
        device = x_T.device
        batch_size = x_T.shape[0]

        if num_steps is None:
            num_steps = self.schedule.num_timesteps

        timesteps = self.schedule.get_inference_timesteps(num_steps)

        x = x_T
        intermediates = [x] if return_intermediates else None

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if method == 'ddpm':
                x = self.p_sample_ddpm(x, t_batch, condition)
            elif method == 'ddim':
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1)
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                x = self.p_sample_ddim(x, t_batch, t_prev_batch, condition, eta)
            else:
                raise ValueError(f"Unknown method: {method}")

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return intermediates
        return x

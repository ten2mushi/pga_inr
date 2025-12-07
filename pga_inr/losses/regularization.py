"""
Regularization losses for PGA-INR training.

These losses encourage desirable properties in the learned representations
beyond basic reconstruction accuracy.
"""

from typing import Optional
import torch
import torch.nn as nn


class LatentRegularization(nn.Module):
    """
    Regularization for latent codes in generative models.

    Encourages latent codes to stay close to a prior distribution,
    typically N(0, I).
    """

    def __init__(
        self,
        regularization_type: str = 'l2',
        sigma: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            regularization_type: 'l2' (MSE), 'l1', or 'kl' (KL divergence)
            sigma: Standard deviation of prior for KL regularization
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.regularization_type = regularization_type
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        Compute latent regularization loss.

        Args:
            latent_codes: Latent codes of shape (B, latent_dim)

        Returns:
            Regularization loss
        """
        if self.regularization_type == 'l2':
            # L2 norm penalty
            loss = (latent_codes ** 2).sum(dim=-1)
        elif self.regularization_type == 'l1':
            # L1 norm penalty (encourages sparsity)
            loss = latent_codes.abs().sum(dim=-1)
        elif self.regularization_type == 'kl':
            # KL divergence from N(0, σ²I)
            # Assuming latent_codes are deterministic, treat as mean of N(z, ε)
            loss = 0.5 * (latent_codes ** 2).sum(dim=-1) / (self.sigma ** 2)
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LipschitzRegularization(nn.Module):
    """
    Encourages Lipschitz continuity of the network.

    A network with Lipschitz constant L satisfies:
    ||f(x) - f(y)|| ≤ L ||x - y||

    This regularization penalizes large gradients.
    """

    def __init__(
        self,
        target_lipschitz: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            target_lipschitz: Target Lipschitz constant
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.target_lipschitz = target_lipschitz
        self.reduction = reduction

    def forward(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lipschitz regularization.

        Args:
            outputs: Network outputs (B, N, out_dim)
            inputs: Network inputs (B, N, in_dim), must have requires_grad=True

        Returns:
            Regularization loss
        """
        # Compute Jacobian norm (gradient of output w.r.t. input)
        grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Gradient norm as proxy for Lipschitz constant
        # Clamp for numerical stability (min prevents division by zero in downstream ops,
        # max prevents extremely large gradients from dominating the loss)
        grad_norm = grad.norm(p=2, dim=-1).clamp(min=1e-12, max=100.0)

        # Penalize exceeding target Lipschitz constant
        loss = torch.relu(grad_norm - self.target_lipschitz) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SmoothnessRegularization(nn.Module):
    """
    Encourages smooth outputs by penalizing high-frequency variations.

    Uses the Laplacian (sum of second derivatives) as a smoothness measure.
    This penalizes high curvature in the output field.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        outputs: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smoothness loss using Laplacian regularization.

        Args:
            outputs: Network outputs at coords (B, N, out_dim)
            coords: Sample coordinates (B, N, 3), must have requires_grad=True

        Returns:
            Smoothness loss (penalizes high curvature / second derivatives)
        """
        B, N, D = coords.shape

        # Compute first-order gradient
        grad = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute Laplacian (sum of squared second derivatives)
        # This penalizes high curvature in the field
        laplacian_sq = torch.zeros(B, N, device=coords.device, dtype=coords.dtype)
        for i in range(D):
            grad_i = grad[..., i]
            grad2_i = torch.autograd.grad(
                outputs=grad_i.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True
            )[0][..., i]
            laplacian_sq = laplacian_sq + grad2_i ** 2

        loss = laplacian_sq

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightDecay(nn.Module):
    """
    Standard L2 weight decay on model parameters.
    """

    def __init__(self, decay_rate: float = 1e-4):
        super().__init__()
        self.decay_rate = decay_rate

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute weight decay loss.

        Args:
            model: Neural network module

        Returns:
            Weight decay loss
        """
        loss = 0
        for param in model.parameters():
            loss = loss + (param ** 2).sum()

        return self.decay_rate * loss


class DivergenceLoss(nn.Module):
    """
    Penalizes divergence of the normal field.

    For valid SDFs, the normals form a conservative field.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        normals: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute divergence loss.

        Args:
            normals: Normal field (B, N, 3)
            coords: Coordinates (B, N, 3), must have requires_grad=True

        Returns:
            Divergence loss
        """
        # Compute divergence: ∂nx/∂x + ∂ny/∂y + ∂nz/∂z
        divergence = 0
        for i in range(3):
            grad_ni = torch.autograd.grad(
                outputs=normals[..., i].sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True
            )[0][..., i]
            divergence = divergence + grad_ni

        loss = divergence ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MinimalSurfaceLoss(nn.Module):
    """
    Encourages minimal surface area (mean curvature = 0).

    Based on the mean curvature flow equation.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        sdf: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute minimal surface loss based on mean curvature.

        Args:
            sdf: SDF values (B, N, 1)
            coords: Coordinates (B, N, 3)

        Returns:
            Mean curvature loss
        """
        # Compute gradient
        grad = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Compute Hessian diagonal (for mean curvature approximation)
        mean_curvature = 0
        for i in range(3):
            grad_i = grad[..., i]
            hess_ii = torch.autograd.grad(
                outputs=grad_i.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True
            )[0][..., i]
            mean_curvature = mean_curvature + hess_ii

        # Normalize by gradient magnitude
        mean_curvature = mean_curvature / grad_norm.squeeze(-1)

        loss = mean_curvature ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

"""
Geometric loss functions for PGA-INR training.

These losses enforce geometric constraints on the learned implicit field:
- Eikonal: Ensures valid SDF (|∇f| = 1)
- Normal alignment: Predicted normals match gradient direction
- SDF supervision: Match ground truth distance values

The key insight is that these constraints are enforced in the LOCAL frame,
making them invariant to observer position/rotation.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.autograd as autograd


def compute_gradient(
    y: torch.Tensor,
    x: torch.Tensor,
    grad_outputs: Optional[torch.Tensor] = None,
    create_graph: bool = True
) -> torch.Tensor:
    """
    Compute gradient of y with respect to x.

    This is the fundamental operation for computing ∇f(x).

    Args:
        y: Output tensor (typically SDF values) of shape (..., 1)
        x: Input tensor (coordinates) of shape (..., 3), must have requires_grad=True
        grad_outputs: Gradient seed (defaults to ones)
        create_graph: Whether to create computation graph for higher-order derivatives

    Returns:
        Gradient tensor of shape (..., 3)
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)

    grad = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True
    )[0]

    return grad


class EikonalLoss(nn.Module):
    """
    Eikonal equation constraint: |∇f(x)| = 1

    This ensures the network learns a valid Signed Distance Function.
    The loss penalizes deviations from unit gradient magnitude.

    L_eikonal = mean((||∇f(x)|| - 1)²)
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        sdf: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Eikonal loss.

        Args:
            sdf: SDF values of shape (B, N, 1)
            coords: Coordinates of shape (B, N, 3), must have requires_grad=True

        Returns:
            Eikonal loss value
        """
        # Compute gradient
        gradient = compute_gradient(sdf, coords)

        # Compute gradient magnitude
        grad_norm = gradient.norm(p=2, dim=-1)

        # Eikonal loss: (||∇f|| - 1)²
        loss = (grad_norm - 1.0) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class NormalAlignmentLoss(nn.Module):
    """
    Aligns predicted normals with the gradient of the SDF.

    The gradient ∇f points in the direction of steepest ascent, which is
    the outward surface normal for SDFs. The predicted normal head should
    match this direction.

    L_align = mean(1 - cos(n_pred, ∇f))
           = mean(1 - (n_pred · ∇f / |∇f|))
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_normals: torch.Tensor,
        sdf: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normal alignment loss.

        Args:
            pred_normals: Predicted normals (B, N, 3), should be unit vectors
            sdf: SDF values (B, N, 1)
            coords: Coordinates (B, N, 3), must have requires_grad=True

        Returns:
            Alignment loss value
        """
        # Compute gradient
        gradient = compute_gradient(sdf, coords)

        # Normalize gradient
        grad_norm = gradient.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        grad_normalized = gradient / grad_norm

        # Cosine similarity (dot product of unit vectors)
        alignment = (pred_normals * grad_normalized).sum(dim=-1)

        # Loss: 1 - cosine (minimized when aligned)
        loss = 1.0 - alignment

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SDFLoss(nn.Module):
    """
    Supervised SDF loss.

    Compares predicted SDF values to ground truth.
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        clamp: Optional[float] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            loss_type: 'l1', 'l2', or 'huber'
            clamp: Optional clamping value for truncated SDF
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.loss_type = loss_type
        self.clamp = clamp
        self.reduction = reduction

        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SDF loss.

        Args:
            pred_sdf: Predicted SDF values (B, N, 1)
            gt_sdf: Ground truth SDF values (B, N, 1)

        Returns:
            Loss value
        """
        # Optional clamping (for truncated SDF)
        if self.clamp is not None:
            pred_sdf = torch.clamp(pred_sdf, -self.clamp, self.clamp)
            gt_sdf = torch.clamp(gt_sdf, -self.clamp, self.clamp)

        loss = self.loss_fn(pred_sdf, gt_sdf)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class NormalSupervisionLoss(nn.Module):
    """
    Supervised normal loss.

    Compares predicted normals to ground truth surface normals.
    """

    def __init__(
        self,
        loss_type: str = 'cosine',
        reduction: str = 'mean'
    ):
        """
        Args:
            loss_type: 'cosine' (1 - cos) or 'l2' (MSE)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        pred_normals: torch.Tensor,
        gt_normals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normal supervision loss.

        Args:
            pred_normals: Predicted normals (B, N, 3)
            gt_normals: Ground truth normals (B, N, 3)

        Returns:
            Loss value
        """
        if self.loss_type == 'cosine':
            # Cosine loss: 1 - cos(pred, gt)
            cos_sim = (pred_normals * gt_normals).sum(dim=-1)
            loss = 1.0 - cos_sim
        else:
            # L2 loss
            loss = ((pred_normals - gt_normals) ** 2).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GeometricConsistencyLoss(nn.Module):
    """
    Combined geometric loss for SDF learning.

    L = L_sdf + λ₁·L_eikonal + λ₂·L_align + λ₃·L_normal

    This loss enforces:
    1. SDF values match ground truth
    2. Gradient magnitude equals 1 (valid SDF)
    3. Predicted normals align with computed gradients
    4. Normals match ground truth (if available)
    """

    def __init__(
        self,
        lambda_sdf: float = 1.0,
        lambda_eikonal: float = 0.1,
        lambda_align: float = 0.05,
        lambda_normal: float = 1.0,
        sdf_clamp: Optional[float] = None
    ):
        """
        Args:
            lambda_sdf: Weight for SDF reconstruction loss
            lambda_eikonal: Weight for Eikonal constraint
            lambda_align: Weight for normal alignment loss
            lambda_normal: Weight for normal supervision loss
            sdf_clamp: Optional clamping for truncated SDF
        """
        super().__init__()

        self.lambda_sdf = lambda_sdf
        self.lambda_eikonal = lambda_eikonal
        self.lambda_align = lambda_align
        self.lambda_normal = lambda_normal

        self.sdf_loss = SDFLoss(loss_type='l1', clamp=sdf_clamp)
        self.eikonal_loss = EikonalLoss()
        self.align_loss = NormalAlignmentLoss()
        self.normal_loss = NormalSupervisionLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_sdf: torch.Tensor,
        gt_normals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined geometric loss.

        Args:
            outputs: Model outputs containing:
                - 'density' or 'sdf': Predicted SDF values (B, N, 1)
                - 'normal': Predicted normals (B, N, 3) (optional)
                - 'local_coords': Local coordinates (B, N, 3)
            gt_sdf: Ground truth SDF values (B, N, 1)
            gt_normals: Ground truth normals (B, N, 3) (optional)

        Returns:
            (total_loss, metrics_dict) where metrics_dict contains individual losses
        """
        # Extract predictions
        pred_sdf = outputs.get('sdf', outputs.get('density'))
        pred_normal = outputs.get('normal')
        local_coords = outputs['local_coords']

        metrics = {}

        # 1. SDF reconstruction loss
        sdf_loss = self.sdf_loss(pred_sdf, gt_sdf)
        metrics['sdf'] = sdf_loss.item()
        total_loss = self.lambda_sdf * sdf_loss

        # 2. Eikonal constraint (gradient magnitude = 1)
        eikonal_loss = self.eikonal_loss(pred_sdf, local_coords)
        metrics['eikonal'] = eikonal_loss.item()
        total_loss = total_loss + self.lambda_eikonal * eikonal_loss

        # 3. Normal alignment (predicted normals match gradients)
        if pred_normal is not None:
            align_loss = self.align_loss(pred_normal, pred_sdf, local_coords)
            metrics['align'] = align_loss.item()
            total_loss = total_loss + self.lambda_align * align_loss

        # 4. Normal supervision (if ground truth available)
        if gt_normals is not None and pred_normal is not None:
            normal_loss = self.normal_loss(pred_normal, gt_normals)
            metrics['normal'] = normal_loss.item()
            total_loss = total_loss + self.lambda_normal * normal_loss

        return total_loss, metrics


class SurfaceLoss(nn.Module):
    """
    Loss for points known to be on the surface.

    For surface points, SDF should be exactly zero.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_sdf: torch.Tensor) -> torch.Tensor:
        """
        Compute surface loss.

        Args:
            pred_sdf: Predicted SDF for surface points (B, N, 1)

        Returns:
            Loss value (SDF values should be zero)
        """
        loss = pred_sdf.abs()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class InteriorExteriorLoss(nn.Module):
    """
    Loss for points with known inside/outside labels.

    Encourages correct sign of SDF (negative inside, positive outside).
    """

    def __init__(self, margin: float = 0.01, reduction: str = 'mean'):
        """
        Args:
            margin: Minimum absolute SDF value
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        pred_sdf: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interior/exterior loss.

        Args:
            pred_sdf: Predicted SDF (B, N, 1)
            labels: Labels (B, N, 1): 1 for outside, -1 for inside

        Returns:
            Loss value
        """
        # SDF should have same sign as labels with margin
        target = labels * self.margin
        loss = torch.relu(target - pred_sdf * labels)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ColorLoss(nn.Module):
    """
    Loss for color supervision.
    """

    def __init__(self, loss_type: str = 'l2', reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute color loss.

        Args:
            pred_rgb: Predicted colors (B, N, 3)
            gt_rgb: Ground truth colors (B, N, 3)

        Returns:
            Loss value
        """
        loss = self.loss_fn(pred_rgb, gt_rgb)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

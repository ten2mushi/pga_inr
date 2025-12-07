"""
SLAM-specific loss functions.

Provides losses for direct depth supervision, free space constraints,
photometric alignment, and pose graph optimization.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectDepthSDFLoss(nn.Module):
    """
    Direct depth supervision for SDF learning.

    Key insight: Points backprojected from observed depth should have SDF = 0.

    L_depth = |SDF(M^{-1} * backproject(u, d_obs))|

    where M is the camera pose and backproject gives 3D point from pixel + depth.
    """

    def __init__(
        self,
        truncation: float = 0.1,
        use_huber: bool = True,
        huber_delta: float = 0.02,
        reduction: str = 'mean'
    ):
        """
        Args:
            truncation: TSDF truncation distance in meters
            use_huber: Use Huber loss instead of L1 for robustness
            huber_delta: Huber loss delta parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.truncation = truncation
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.reduction = reduction

    def forward(
        self,
        pred_sdf: torch.Tensor,
        target_sdf: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute direct depth loss.

        Args:
            pred_sdf: Predicted SDF at sampled points (N, 1) or (N,)
            target_sdf: Target SDF values (N, 1) or (N,). If None, assumes 0.
            weights: Optional per-point weights (N,)

        Returns:
            Loss value
        """
        pred_sdf = pred_sdf.squeeze(-1) if pred_sdf.dim() > 1 else pred_sdf

        if target_sdf is None:
            # Surface points should have SDF = 0
            target_sdf = torch.zeros_like(pred_sdf)
        else:
            target_sdf = target_sdf.squeeze(-1) if target_sdf.dim() > 1 else target_sdf

        if self.use_huber:
            loss = F.smooth_l1_loss(
                pred_sdf,
                target_sdf,
                beta=self.huber_delta,
                reduction='none'
            )
        else:
            loss = (pred_sdf - target_sdf).abs()

        # Apply weights
        if weights is not None:
            loss = loss * weights

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TruncatedSDFLoss(nn.Module):
    """
    Truncated SDF supervision for TSDF-style training.

    SDF values are clamped to [-truncation, truncation] before computing loss.
    This prevents the network from fitting to noise far from surfaces.
    """

    def __init__(
        self,
        truncation: float = 0.1,
        use_huber: bool = True,
        huber_delta: float = 0.02,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.truncation = truncation
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.reduction = reduction

    def forward(
        self,
        pred_sdf: torch.Tensor,
        target_sdf: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute truncated SDF loss.

        Args:
            pred_sdf: Predicted SDF (N, 1) or (N,)
            target_sdf: Target SDF (N, 1) or (N,)
            weights: Optional per-point weights

        Returns:
            Loss value
        """
        pred_sdf = pred_sdf.squeeze(-1) if pred_sdf.dim() > 1 else pred_sdf
        target_sdf = target_sdf.squeeze(-1) if target_sdf.dim() > 1 else target_sdf

        # Clamp both predictions and targets
        pred_clamped = pred_sdf.clamp(-self.truncation, self.truncation)
        target_clamped = target_sdf.clamp(-self.truncation, self.truncation)

        if self.use_huber:
            loss = F.smooth_l1_loss(
                pred_clamped,
                target_clamped,
                beta=self.huber_delta,
                reduction='none'
            )
        else:
            loss = (pred_clamped - target_clamped).abs()

        if weights is not None:
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FreeSpaceLoss(nn.Module):
    """
    Loss for free space points (between camera and surface).

    These points should have positive SDF (outside any surface).
    Uses hinge loss: max(0, margin - sdf)
    """

    def __init__(
        self,
        margin: float = 0.01,
        reduction: str = 'mean'
    ):
        """
        Args:
            margin: Minimum required positive SDF value
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        pred_sdf: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Penalize negative SDF values in free space.

        Args:
            pred_sdf: Predicted SDF (N, 1) or (N,)
            weights: Optional per-point weights

        Returns:
            Loss value
        """
        pred_sdf = pred_sdf.squeeze(-1) if pred_sdf.dim() > 1 else pred_sdf

        # Hinge loss: penalize when sdf < margin
        loss = F.relu(self.margin - pred_sdf)

        if weights is not None:
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class OccupiedSpaceLoss(nn.Module):
    """
    Loss for occupied space points (behind the observed surface).

    These points should have negative SDF (inside some surface).
    Uses hinge loss: max(0, sdf + margin)
    """

    def __init__(
        self,
        margin: float = 0.01,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        pred_sdf: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Penalize positive SDF values in occupied space.

        Args:
            pred_sdf: Predicted SDF (N, 1) or (N,)
            weights: Optional per-point weights

        Returns:
            Loss value
        """
        pred_sdf = pred_sdf.squeeze(-1) if pred_sdf.dim() > 1 else pred_sdf

        # Hinge loss: penalize when sdf > -margin
        loss = F.relu(pred_sdf + self.margin)

        if weights is not None:
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PhotometricLoss(nn.Module):
    """
    Photometric (RGB) loss for dense alignment.

    Compares rendered RGB with observed RGB at corresponding pixels.
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        reduction: str = 'mean'
    ):
        """
        Args:
            loss_type: 'l1' or 'l2'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute photometric loss.

        Args:
            pred_rgb: Predicted RGB (H, W, 3) or (N, 3)
            target_rgb: Target RGB (H, W, 3) or (N, 3)
            mask: Optional validity mask

        Returns:
            Loss value
        """
        if mask is not None:
            if mask.dim() == 2:  # (H, W)
                mask = mask.unsqueeze(-1).expand_as(pred_rgb)
            pred_rgb = pred_rgb[mask]
            target_rgb = target_rgb[mask]

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_rgb, target_rgb, reduction=self.reduction)
        else:
            loss = F.mse_loss(pred_rgb, target_rgb, reduction=self.reduction)

        return loss


class DepthLoss(nn.Module):
    """
    Depth map supervision loss.

    Compares rendered depth with observed depth.
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        max_depth: float = 10.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.loss_type = loss_type
        self.max_depth = max_depth
        self.reduction = reduction

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute depth loss.

        Args:
            pred_depth: Predicted depth (H, W) or (N,)
            target_depth: Target depth (H, W) or (N,)
            mask: Optional validity mask

        Returns:
            Loss value
        """
        # Create validity mask
        if mask is None:
            mask = (target_depth > 0) & (target_depth < self.max_depth)

        pred_valid = pred_depth[mask]
        target_valid = target_depth[mask]

        if len(pred_valid) == 0:
            return torch.tensor(0.0, device=pred_depth.device)

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_valid, target_valid, reduction=self.reduction)
        else:
            loss = F.mse_loss(pred_valid, target_valid, reduction=self.reduction)

        return loss


class PoseGraphLoss(nn.Module):
    """
    Pose graph optimization loss using relative motor constraints.

    L = sum_ij ||log(M_i^{-1} * M_j * M_ij^{-1})||^2

    where M_ij is the measured relative transform from i to j.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        poses_lie: torch.Tensor,       # (N, 6) lie algebra params
        edges: torch.Tensor,           # (E, 2) edge indices
        measurements_lie: torch.Tensor,  # (E, 6) relative transforms in lie algebra
        information: Optional[torch.Tensor] = None  # (E, 6, 6) information matrices
    ) -> torch.Tensor:
        """
        Compute pose graph residuals.

        Args:
            poses_lie: Poses in se(3) Lie algebra (N, 6)
            edges: Edge indices (E, 2)
            measurements_lie: Measured relative transforms in se(3) (E, 6)
            information: Optional information matrices (E, 6, 6)

        Returns:
            Loss value
        """
        from ..pga.motors import motor_exp, motor_log

        residuals = []

        for e in range(edges.shape[0]):
            i, j = edges[e]

            # Current poses
            M_i = motor_exp(poses_lie[i])
            M_j = motor_exp(poses_lie[j])

            # Measured relative transform
            M_ij_measured = motor_exp(measurements_lie[e])

            # Compute residual: should be identity if consistent
            # residual = M_i^{-1} * M_j * M_ij^{-1}
            M_ij_current = M_i.inverse().compose(M_j)
            M_residual = M_ij_measured.inverse().compose(M_ij_current)

            # Log map to get se(3) residual
            residual = motor_log(M_residual)
            residuals.append(residual)

        residuals = torch.stack(residuals, dim=0)  # (E, 6)

        # Compute weighted squared norms
        if information is not None:
            # Mahalanobis distance: r^T @ I @ r
            loss = torch.einsum('ei,eij,ej->e', residuals, information, residuals)
        else:
            # Euclidean norm
            loss = (residuals ** 2).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SLAMLoss(nn.Module):
    """
    Combined SLAM loss for tracking and mapping.

    Combines multiple loss terms with configurable weights.
    """

    def __init__(
        self,
        lambda_sdf: float = 1.0,
        lambda_eikonal: float = 0.1,
        lambda_free_space: float = 0.5,
        lambda_depth: float = 1.0,
        lambda_rgb: float = 0.1,
        truncation: float = 0.1,
        use_huber: bool = True
    ):
        super().__init__()

        self.lambda_sdf = lambda_sdf
        self.lambda_eikonal = lambda_eikonal
        self.lambda_free_space = lambda_free_space
        self.lambda_depth = lambda_depth
        self.lambda_rgb = lambda_rgb

        # Component losses
        self.sdf_loss = DirectDepthSDFLoss(
            truncation=truncation,
            use_huber=use_huber
        )
        self.truncated_sdf_loss = TruncatedSDFLoss(
            truncation=truncation,
            use_huber=use_huber
        )
        self.free_space_loss = FreeSpaceLoss()
        self.depth_loss = DepthLoss()
        self.photometric_loss = PhotometricLoss()

    def forward(
        self,
        pred_sdf: torch.Tensor,
        target_sdf: Optional[torch.Tensor] = None,
        pred_gradient: Optional[torch.Tensor] = None,
        surface_mask: Optional[torch.Tensor] = None,
        free_space_mask: Optional[torch.Tensor] = None,
        pred_depth: Optional[torch.Tensor] = None,
        target_depth: Optional[torch.Tensor] = None,
        pred_rgb: Optional[torch.Tensor] = None,
        target_rgb: Optional[torch.Tensor] = None,
        rgb_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Compute combined SLAM loss.

        Returns:
            (total_loss, loss_dict)
        """
        device = pred_sdf.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # SDF loss (surface points)
        if target_sdf is not None or surface_mask is not None:
            sdf_input = pred_sdf[surface_mask] if surface_mask is not None else pred_sdf
            sdf_target = target_sdf[surface_mask] if surface_mask is not None and target_sdf is not None else target_sdf
            sdf_loss = self.sdf_loss(sdf_input, sdf_target)
            total_loss = total_loss + self.lambda_sdf * sdf_loss
            loss_dict['sdf'] = sdf_loss.item()

        # Eikonal loss
        if pred_gradient is not None:
            grad_norm = pred_gradient.norm(dim=-1)
            eikonal_loss = ((grad_norm - 1.0) ** 2).mean()
            total_loss = total_loss + self.lambda_eikonal * eikonal_loss
            loss_dict['eikonal'] = eikonal_loss.item()

        # Free space loss
        if free_space_mask is not None and free_space_mask.any():
            free_sdf = pred_sdf[free_space_mask]
            free_loss = self.free_space_loss(free_sdf)
            total_loss = total_loss + self.lambda_free_space * free_loss
            loss_dict['free_space'] = free_loss.item()

        # Depth loss
        if pred_depth is not None and target_depth is not None:
            depth_loss = self.depth_loss(pred_depth, target_depth)
            total_loss = total_loss + self.lambda_depth * depth_loss
            loss_dict['depth'] = depth_loss.item()

        # RGB loss
        if pred_rgb is not None and target_rgb is not None:
            rgb_loss = self.photometric_loss(pred_rgb, target_rgb, rgb_mask)
            total_loss = total_loss + self.lambda_rgb * rgb_loss
            loss_dict['rgb'] = rgb_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict

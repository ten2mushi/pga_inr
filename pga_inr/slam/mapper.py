"""
Neural map optimization for PGA-SLAM.

Updates the neural implicit representation (INR) weights based on
keyframe observations using direct depth supervision.
"""

from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn

from .config import MapperConfig
from .types import Keyframe, MappingResult
from .losses import DirectDepthSDFLoss, TruncatedSDFLoss, FreeSpaceLoss
from ..models.inr import PGA_INR_SDF_V2
from ..losses.geometric import EikonalLoss


class NeuralMapper:
    """
    Manages neural implicit map (INR weights).

    Updates the map based on keyframe observations using direct
    depth supervision without requiring sphere tracing.

    Key features:
    - Direct depth supervision (surface points → SDF = 0)
    - Free space regularization (SDF > 0 between camera and surface)
    - Eikonal loss for valid SDF
    - Support for incremental and joint optimization
    """

    def __init__(
        self,
        config: MapperConfig,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            config: Mapper configuration
            device: Device for computation
        """
        self.config = config
        self.device = device

        # Initialize neural map
        self.model = PGA_INR_SDF_V2(
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            omega_0=config.omega_0,
            omega_hidden=config.omega_hidden,
            use_positional_encoding=config.use_positional_encoding,
            num_frequencies=config.num_frequencies,
            geometric_init=config.geometric_init
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.999
        )

        # Losses
        self.sdf_loss = DirectDepthSDFLoss(use_huber=True)
        self.truncated_sdf_loss = TruncatedSDFLoss(
            truncation=0.1, use_huber=True
        )
        self.free_space_loss = FreeSpaceLoss(margin=0.01)
        self.eikonal_loss = EikonalLoss()

        # Statistics
        self._total_iterations = 0

    def update_from_keyframe(
        self,
        keyframe: Keyframe,
        num_iterations: Optional[int] = None
    ) -> MappingResult:
        """
        Update map from a single keyframe.

        Uses direct depth supervision:
        - Points at observed depth: SDF = 0
        - Points between camera and surface: SDF > 0 (free space)
        - Eikonal regularization: |∇SDF| = 1

        Args:
            keyframe: Keyframe to learn from
            num_iterations: Override iteration count

        Returns:
            MappingResult with loss history
        """
        num_iterations = num_iterations or self.config.num_iterations

        self.model.train()
        loss_history = []
        final_breakdown = {}

        for iteration in range(num_iterations):
            self.optimizer.zero_grad()

            # Sample training points
            points, sdf_targets, point_types = self._sample_points_from_keyframe(
                keyframe
            )

            # Enable gradients for eikonal loss
            points = points.requires_grad_(True)

            # Forward pass
            outputs = self.model(points.unsqueeze(0), observer_pose=None)
            pred_sdf = outputs['sdf'].squeeze(0)
            local_coords = outputs['local_coords'].squeeze(0)

            # Compute losses
            total_loss = torch.tensor(0.0, device=self.device)
            breakdown = {}

            # === Surface points: SDF = 0 ===
            surface_mask = point_types == 0
            if surface_mask.any():
                surface_sdf = pred_sdf[surface_mask]
                surface_target = sdf_targets[surface_mask]
                sdf_loss = self.sdf_loss(surface_sdf, surface_target)
                total_loss = total_loss + self.config.lambda_sdf * sdf_loss
                breakdown['sdf'] = sdf_loss.item()

            # === Free space points: SDF > 0 ===
            free_mask = point_types == 1
            if free_mask.any():
                free_sdf = pred_sdf[free_mask]
                free_loss = self.free_space_loss(free_sdf)
                total_loss = total_loss + self.config.lambda_free_space * free_loss
                breakdown['free_space'] = free_loss.item()

            # === Truncated SDF for behind-surface points ===
            behind_mask = point_types == 2
            if behind_mask.any():
                behind_sdf = pred_sdf[behind_mask]
                behind_target = sdf_targets[behind_mask]
                truncated_loss = self.truncated_sdf_loss(behind_sdf, behind_target)
                total_loss = total_loss + self.config.lambda_sdf * 0.5 * truncated_loss
                breakdown['truncated'] = truncated_loss.item()

            # === Eikonal loss ===
            eikonal_loss = self._compute_eikonal_loss(pred_sdf, local_coords)
            total_loss = total_loss + self.config.lambda_eikonal * eikonal_loss
            breakdown['eikonal'] = eikonal_loss.item()

            # Backward and step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss_history.append(total_loss.item())
            final_breakdown = breakdown

        self._total_iterations += num_iterations

        # Step scheduler
        self.scheduler.step()

        return MappingResult(
            num_iterations=num_iterations,
            final_loss=loss_history[-1] if loss_history else 0.0,
            loss_history=loss_history,
            loss_breakdown=final_breakdown
        )

    def _compute_eikonal_loss(
        self,
        sdf: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """Compute Eikonal loss: (|∇SDF| - 1)²."""
        gradient = torch.autograd.grad(
            outputs=sdf,
            inputs=coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]

        if gradient is None:
            return torch.tensor(0.0, device=sdf.device)

        grad_norm = gradient.norm(dim=-1)
        eikonal = ((grad_norm - 1.0) ** 2).mean()
        return eikonal

    def update_from_keyframes(
        self,
        keyframes: List[Keyframe],
        num_iterations: int = 100
    ) -> MappingResult:
        """
        Joint optimization over multiple keyframes (bundle adjustment).

        Args:
            keyframes: List of keyframes
            num_iterations: Total iterations

        Returns:
            MappingResult with loss history
        """
        if len(keyframes) == 0:
            return MappingResult(
                num_iterations=0,
                final_loss=0.0,
                loss_history=[],
                loss_breakdown={}
            )

        self.model.train()
        loss_history = []
        final_breakdown = {}

        samples_per_kf = self.config.batch_size // len(keyframes)

        for iteration in range(num_iterations):
            self.optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=self.device)
            total_sdf_loss = 0.0
            total_eikonal_loss = 0.0

            # Accumulate gradients from all keyframes
            for kf in keyframes:
                points, sdf_targets, point_types = self._sample_points_from_keyframe(
                    kf, num_samples=samples_per_kf
                )

                points = points.requires_grad_(True)
                outputs = self.model(points.unsqueeze(0), observer_pose=None)
                pred_sdf = outputs['sdf'].squeeze(0)
                local_coords = outputs['local_coords'].squeeze(0)

                # Surface loss
                surface_mask = point_types == 0
                if surface_mask.any():
                    sdf_loss = self.sdf_loss(
                        pred_sdf[surface_mask],
                        sdf_targets[surface_mask]
                    )
                    total_loss = total_loss + self.config.lambda_sdf * sdf_loss
                    total_sdf_loss += sdf_loss.item()

                # Eikonal
                eikonal_loss = self._compute_eikonal_loss(pred_sdf, local_coords)
                total_loss = total_loss + self.config.lambda_eikonal * eikonal_loss
                total_eikonal_loss += eikonal_loss.item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss_history.append(total_loss.item())
            final_breakdown = {
                'sdf': total_sdf_loss / len(keyframes),
                'eikonal': total_eikonal_loss / len(keyframes),
                'total': total_loss.item()
            }

        self._total_iterations += num_iterations

        return MappingResult(
            num_iterations=num_iterations,
            final_loss=loss_history[-1] if loss_history else 0.0,
            loss_history=loss_history,
            loss_breakdown=final_breakdown
        )

    def _sample_points_from_keyframe(
        self,
        keyframe: Keyframe,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample training points from a keyframe.

        Point types:
        - 0: Surface points (SDF = 0)
        - 1: Free space points (SDF > 0)
        - 2: Behind surface points (SDF < 0, truncated)

        Returns:
            points: (N, 3) world coordinates
            sdf_targets: (N, 1) target SDF values
            point_types: (N,) int tensor of point types
        """
        num_samples = num_samples or self.config.batch_size

        # Get 3D points from keyframe
        if keyframe.points_world is None:
            # Compute on the fly
            points_cam = keyframe.points_3d
            if points_cam is None:
                # Return empty if no points
                return (
                    torch.empty(0, 3, device=self.device),
                    torch.empty(0, 1, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device)
                )
            from ..utils.quaternion import quaternion_to_matrix
            R = quaternion_to_matrix(keyframe.quaternion.unsqueeze(0)).squeeze(0)
            points_world = points_cam @ R.T + keyframe.translation.unsqueeze(0)
        else:
            points_world = keyframe.points_world
            points_cam = keyframe.points_3d

        M = points_world.shape[0]
        if M == 0:
            return (
                torch.empty(0, 3, device=self.device),
                torch.empty(0, 1, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device)
            )

        # Sample counts
        n_surface = int(num_samples * self.config.surface_sample_ratio)
        n_free = int(num_samples * self.config.free_space_sample_ratio)
        n_volume = num_samples - n_surface - n_free

        all_points = []
        all_targets = []
        all_types = []

        # === 1. Surface samples (type 0) ===
        if n_surface > 0:
            idx = torch.randint(0, M, (n_surface,), device=self.device)
            surface_pts = points_world[idx]

            # Add small noise
            if self.config.surface_noise_std > 0:
                noise = torch.randn_like(surface_pts) * self.config.surface_noise_std
                surface_pts = surface_pts + noise

            all_points.append(surface_pts)
            all_targets.append(torch.zeros(n_surface, 1, device=self.device))
            all_types.append(torch.zeros(n_surface, dtype=torch.long, device=self.device))

        # === 2. Free space samples (type 1) ===
        if n_free > 0 and points_cam is not None:
            idx = torch.randint(0, M, (n_free,), device=self.device)
            depth_pts_cam = points_cam[idx]

            # Sample random fraction [0.3, 0.8] of observed depth
            fractions = torch.rand(n_free, 1, device=self.device) * 0.5 + 0.3
            free_pts_cam = depth_pts_cam * fractions

            # Transform to world
            from ..utils.quaternion import quaternion_to_matrix
            R = quaternion_to_matrix(keyframe.quaternion.unsqueeze(0)).squeeze(0)
            free_pts_world = free_pts_cam @ R.T + keyframe.translation.unsqueeze(0)

            # SDF target: positive (distance to surface)
            depths = depth_pts_cam[:, 2:3]
            free_depths = free_pts_cam[:, 2:3]
            sdf_targets = (depths - free_depths).clamp(max=0.1)

            all_points.append(free_pts_world)
            all_targets.append(sdf_targets)
            all_types.append(torch.ones(n_free, dtype=torch.long, device=self.device))

        # === 3. Volume samples (uniform in bounds) ===
        if n_volume > 0:
            bounds = self.config.bounds
            volume_pts = torch.rand(n_volume, 3, device=self.device)
            volume_pts[:, 0] = volume_pts[:, 0] * (bounds[1] - bounds[0]) + bounds[0]
            volume_pts[:, 1] = volume_pts[:, 1] * (bounds[3] - bounds[2]) + bounds[2]
            volume_pts[:, 2] = volume_pts[:, 2] * (bounds[5] - bounds[4]) + bounds[4]

            all_points.append(volume_pts)
            # Large positive target for volume points (assumed free space)
            all_targets.append(torch.ones(n_volume, 1, device=self.device) * 0.5)
            all_types.append(torch.ones(n_volume, dtype=torch.long, device=self.device))

        if len(all_points) == 0:
            return (
                torch.empty(0, 3, device=self.device),
                torch.empty(0, 1, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device)
            )

        points = torch.cat(all_points, dim=0)
        targets = torch.cat(all_targets, dim=0)
        types = torch.cat(all_types, dim=0)

        return points, targets, types

    def get_model(self) -> nn.Module:
        """Get the neural map model."""
        return self.model

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_iterations': self._total_iterations,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'total_iterations' in checkpoint:
            self._total_iterations = checkpoint['total_iterations']

    def reset(self):
        """Reset model to initial state."""
        self.model = PGA_INR_SDF_V2(
            hidden_features=self.config.hidden_features,
            hidden_layers=self.config.hidden_layers,
            omega_0=self.config.omega_0,
            omega_hidden=self.config.omega_hidden,
            use_positional_encoding=self.config.use_positional_encoding,
            num_frequencies=self.config.num_frequencies,
            geometric_init=self.config.geometric_init
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.999
        )
        self._total_iterations = 0

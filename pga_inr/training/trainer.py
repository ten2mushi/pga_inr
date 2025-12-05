"""
Training utilities for PGA-INR models.

Provides training loops, logging, and checkpoint management for:
- Basic PGA-INR (single object)
- Generative PGA-INR (multi-object with auto-decoding)
"""

from typing import Dict, Optional, Tuple, List, Callable, Any
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm


class PGAINRTrainer:
    """
    Training manager for PGA-INR models.

    Handles:
    - Training and validation loops
    - Geometric loss computation with gradient tracking
    - Logging and checkpointing
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device = torch.device('cuda'),
        scheduler: Optional[_LRScheduler] = None,
        gradient_clip: Optional[float] = None,
        log_interval: int = 100,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            model: PGA-INR model
            optimizer: Optimizer for model parameters
            loss_fn: Loss function (e.g., GeometricConsistencyLoss)
            device: Device for training
            scheduler: Optional learning rate scheduler
            gradient_clip: Optional gradient clipping value
            log_interval: Steps between logging
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval

        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.

        Args:
            batch: Dictionary with 'points', 'sdf', optional 'normals'
            observer_pose: Optional (translation, quaternion) tuple

        Returns:
            (loss, metrics_dict)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move batch to device
        points = batch['points'].to(self.device)
        gt_sdf = batch['sdf'].to(self.device)
        gt_normals = batch.get('normals')
        if gt_normals is not None:
            gt_normals = gt_normals.to(self.device)

        # Handle pose
        if observer_pose is not None:
            translation = observer_pose[0].to(self.device)
            quaternion = observer_pose[1].to(self.device)
            pose = (translation, quaternion)
        elif 'translation' in batch and 'quaternion' in batch:
            pose = (
                batch['translation'].to(self.device),
                batch['quaternion'].to(self.device)
            )
        else:
            pose = None

        # Enable gradient tracking for Eikonal loss
        points.requires_grad_(True)

        # Forward pass
        outputs = self.model(points, pose)

        # Ensure local_coords has gradients
        if 'local_coords' in outputs and not outputs['local_coords'].requires_grad:
            outputs['local_coords'] = outputs['local_coords'].requires_grad_(True)

        # Compute loss
        loss, metrics = self.loss_fn(outputs, gt_sdf, gt_normals)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

        # Optimizer step
        self.optimizer.step()

        self.global_step += 1

        return loss, metrics

    def validate_step(
        self,
        batch: Dict[str, torch.Tensor],
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single validation step.

        Args:
            batch: Dictionary with 'points', 'sdf', optional 'normals'
            observer_pose: Optional (translation, quaternion) tuple

        Returns:
            (loss, metrics_dict)
        """
        self.model.eval()

        with torch.no_grad():
            points = batch['points'].to(self.device)
            gt_sdf = batch['sdf'].to(self.device)
            gt_normals = batch.get('normals')
            if gt_normals is not None:
                gt_normals = gt_normals.to(self.device)

            if observer_pose is not None:
                pose = (
                    observer_pose[0].to(self.device),
                    observer_pose[1].to(self.device)
                )
            elif 'translation' in batch and 'quaternion' in batch:
                pose = (
                    batch['translation'].to(self.device),
                    batch['quaternion'].to(self.device)
                )
            else:
                pose = None

            outputs = self.model(points, pose)
            loss, metrics = self.loss_fn(outputs, gt_sdf, gt_normals)

        return loss, metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        self.epoch = epoch

        total_loss = 0.0
        all_metrics = {}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in pbar:
            loss, metrics = self.train_step(batch)

            total_loss += loss.item()
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0) + v
            num_batches += 1

            # Update progress bar
            if self.global_step % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    **{k: f'{v:.4f}' for k, v in metrics.items()}
                })

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()

        self.history['train_loss'].append(avg_loss)

        return {'loss': avg_loss, **avg_metrics}

    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Run validation.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of average metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_metrics = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                loss, metrics = self.validate_step(batch)

                total_loss += loss.item()
                for k, v in metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0) + v
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

        self.history['val_loss'].append(avg_loss)

        return {'loss': avg_loss, **avg_metrics}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs
            early_stopping_patience: Epochs without improvement before stopping
            callbacks: Optional list of callback functions

        Returns:
            Training history dictionary
        """
        callbacks = callbacks or []
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if self.checkpoint_dir:
                        self.save_checkpoint('best.pt')
                else:
                    patience_counter += 1
            else:
                val_metrics = {}

            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')

            # Log
            elapsed = time.time() - start_time
            print(f'Epoch {epoch + 1}/{epochs} ({elapsed:.1f}s)')
            print(f'  Train: ' + ', '.join(f'{k}={v:.4f}' for k, v in train_metrics.items()))
            if val_metrics:
                print(f'  Val:   ' + ', '.join(f'{k}={v:.4f}' for k, v in val_metrics.items()))

            # Run callbacks
            for callback in callbacks:
                callback(self, epoch, train_metrics, val_metrics)

            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break

        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint('final.pt')

        return self.history

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f'Saved checkpoint: {path}')

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f'Loaded checkpoint from epoch {self.epoch}')


class GenerativePGAINRTrainer(PGAINRTrainer):
    """
    Trainer for generative models with auto-decoding.

    Optimizes both network weights and latent codes.
    """

    def __init__(
        self,
        model: nn.Module,
        latent_bank: nn.Module,
        optimizer_net: Optimizer,
        optimizer_codes: Optimizer,
        loss_fn: nn.Module,
        latent_reg: Optional[nn.Module] = None,
        latent_reg_weight: float = 0.001,
        device: torch.device = torch.device('cuda'),
        scheduler_net: Optional[_LRScheduler] = None,
        scheduler_codes: Optional[_LRScheduler] = None,
        gradient_clip: Optional[float] = None,
        log_interval: int = 100,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            model: Generative PGA-INR model
            latent_bank: LatentCodeBank module
            optimizer_net: Optimizer for network parameters
            optimizer_codes: Optimizer for latent codes
            loss_fn: Loss function
            latent_reg: Optional latent regularization module
            latent_reg_weight: Weight for latent regularization
            device: Device for training
            scheduler_net: Optional scheduler for network optimizer
            scheduler_codes: Optional scheduler for codes optimizer
            gradient_clip: Optional gradient clipping value
            log_interval: Steps between logging
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(
            model=model,
            optimizer=optimizer_net,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler_net,
            gradient_clip=gradient_clip,
            log_interval=log_interval,
            checkpoint_dir=checkpoint_dir
        )

        self.latent_bank = latent_bank.to(device)
        self.optimizer_codes = optimizer_codes
        self.scheduler_codes = scheduler_codes
        self.latent_reg = latent_reg
        self.latent_reg_weight = latent_reg_weight

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training step for generative model.

        Args:
            batch: Dictionary with 'points', 'sdf', 'object_idx', optional 'normals'
            observer_pose: Optional (translation, quaternion) tuple

        Returns:
            (loss, metrics_dict)
        """
        self.model.train()
        self.latent_bank.train()

        self.optimizer.zero_grad()
        self.optimizer_codes.zero_grad()

        # Move batch to device
        points = batch['points'].to(self.device)
        gt_sdf = batch['sdf'].to(self.device)
        object_idx = batch['object_idx'].to(self.device)
        gt_normals = batch.get('normals')
        if gt_normals is not None:
            gt_normals = gt_normals.to(self.device)

        # Handle pose
        if observer_pose is not None:
            pose = (
                observer_pose[0].to(self.device),
                observer_pose[1].to(self.device)
            )
        elif 'translation' in batch and 'quaternion' in batch:
            pose = (
                batch['translation'].to(self.device),
                batch['quaternion'].to(self.device)
            )
        else:
            pose = None

        # Get latent codes
        latent_codes = self.latent_bank(object_idx)

        # Enable gradient tracking
        points.requires_grad_(True)

        # Forward pass
        outputs = self.model(points, pose, latent_codes)

        # Compute reconstruction loss
        loss, metrics = self.loss_fn(outputs, gt_sdf, gt_normals)

        # Add latent regularization
        if self.latent_reg is not None:
            reg_loss = self.latent_reg(latent_codes)
            loss = loss + self.latent_reg_weight * reg_loss
            metrics['latent_reg'] = reg_loss.item()

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
            torch.nn.utils.clip_grad_norm_(
                self.latent_bank.parameters(),
                self.gradient_clip
            )

        # Optimizer steps
        self.optimizer.step()
        self.optimizer_codes.step()

        self.global_step += 1

        return loss, metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.latent_bank.train()
        self.epoch = epoch

        total_loss = 0.0
        all_metrics = {}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in pbar:
            loss, metrics = self.train_step(batch)

            total_loss += loss.item()
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0) + v
            num_batches += 1

            if self.global_step % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    **{k: f'{v:.4f}' for k, v in metrics.items()}
                })

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        if self.scheduler_codes is not None:
            self.scheduler_codes.step()

        self.history['train_loss'].append(avg_loss)

        return {'loss': avg_loss, **avg_metrics}

    def save_checkpoint(self, filename: str):
        """Save training checkpoint including latent codes."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'latent_bank_state_dict': self.latent_bank.state_dict(),
            'optimizer_net_state_dict': self.optimizer.state_dict(),
            'optimizer_codes_state_dict': self.optimizer_codes.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_net_state_dict'] = self.scheduler.state_dict()
        if self.scheduler_codes is not None:
            checkpoint['scheduler_codes_state_dict'] = self.scheduler_codes.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f'Saved checkpoint: {path}')

    def load_checkpoint(self, path: str):
        """Load training checkpoint including latent codes."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.latent_bank.load_state_dict(checkpoint['latent_bank_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_net_state_dict'])
        self.optimizer_codes.load_state_dict(checkpoint['optimizer_codes_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_net_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_net_state_dict'])
        if self.scheduler_codes is not None and 'scheduler_codes_state_dict' in checkpoint:
            self.scheduler_codes.load_state_dict(checkpoint['scheduler_codes_state_dict'])

        print(f'Loaded checkpoint from epoch {self.epoch}')

    def infer_latent(
        self,
        target_sdf_fn: Callable[[torch.Tensor], torch.Tensor],
        num_iterations: int = 1000,
        lr: float = 0.01,
        num_samples: int = 10000,
        bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> torch.Tensor:
        """
        Infer latent code for a new shape (not in training set).

        Uses gradient descent to find z that minimizes reconstruction error.

        Args:
            target_sdf_fn: Function returning target SDF values
            num_iterations: Optimization iterations
            lr: Learning rate
            num_samples: Number of sample points
            bounds: Coordinate bounds

        Returns:
            Optimized latent code
        """
        self.model.eval()

        # Initialize random latent code
        z = torch.randn(1, self.latent_bank.latent_dim, device=self.device)
        z.requires_grad_(True)

        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Sample random points
            points = torch.rand(1, num_samples, 3, device=self.device)
            points = points * (bounds[1] - bounds[0]) + bounds[0]
            points.requires_grad_(True)

            # Forward pass
            outputs = self.model(points, None, z)
            pred_sdf = outputs.get('sdf', outputs.get('density'))

            # Get target SDF
            with torch.no_grad():
                target_sdf = target_sdf_fn(points.squeeze(0))
                target_sdf = target_sdf.view(1, -1, 1)

            # Loss
            loss = ((pred_sdf - target_sdf) ** 2).mean()

            # Add regularization
            if self.latent_reg is not None:
                loss = loss + self.latent_reg_weight * self.latent_reg(z)

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Iteration {i + 1}/{num_iterations}, Loss: {loss.item():.6f}')

        return z.detach()


class NeRFTrainer(PGAINRTrainer):
    """
    Trainer for NeRF-style models with ray-based supervision.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device = torch.device('cuda'),
        scheduler: Optional[_LRScheduler] = None,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        gradient_clip: Optional[float] = None,
        log_interval: int = 100,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            model: NeRF model
            optimizer: Optimizer
            device: Device
            scheduler: LR scheduler
            num_coarse_samples: Coarse samples per ray
            num_fine_samples: Fine samples per ray
            gradient_clip: Gradient clipping
            log_interval: Log interval
            checkpoint_dir: Checkpoint directory
        """
        # Use MSE loss for color supervision
        loss_fn = nn.MSELoss()

        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler,
            gradient_clip=gradient_clip,
            log_interval=log_interval,
            checkpoint_dir=checkpoint_dir
        )

        self.num_coarse = num_coarse_samples
        self.num_fine = num_fine_samples

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training step for NeRF.

        Args:
            batch: Dictionary with 'origins', 'directions', 'gt_colors', 'near', 'far'
            observer_pose: Optional object pose

        Returns:
            (loss, metrics_dict)
        """
        from ..data.sampling import HierarchicalSampler
        from ..rendering.sphere_tracing import volume_render

        self.model.train()
        self.optimizer.zero_grad()

        # Move batch to device
        origins = batch['origins'].to(self.device)
        directions = batch['directions'].to(self.device)
        gt_colors = batch['gt_colors'].to(self.device)
        near = batch['near'][0].item()
        far = batch['far'][0].item()

        # Sample points along rays
        sampler = HierarchicalSampler(
            near=near, far=far,
            num_coarse=self.num_coarse,
            num_fine=self.num_fine,
            device=self.device
        )

        # Coarse samples
        points, t_vals = sampler.sample_coarse(origins, directions)

        # Query model
        outputs = self.model(points.view(-1, 3).unsqueeze(0), observer_pose)
        density = outputs['density'].view(*points.shape[:-1])
        rgb = outputs['rgb'].view(*points.shape[:-1], 3)

        # Volume rendering
        pred_colors, weights = volume_render(density, rgb, t_vals)

        # Color loss
        loss = self.loss_fn(pred_colors, gt_colors)

        # Backward pass
        loss.backward()

        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

        self.optimizer.step()
        self.global_step += 1

        # Compute PSNR
        mse = ((pred_colors - gt_colors) ** 2).mean()
        psnr = -10 * torch.log10(mse + 1e-8)

        return loss, {'mse': mse.item(), 'psnr': psnr.item()}


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    **kwargs
) -> Optimizer:
    """
    Create optimizer for model.

    Args:
        model: Model to optimize
        optimizer_type: 'adam', 'adamw', 'sgd'
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer
    """
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'cosine',
    epochs: int = 100,
    **kwargs
) -> _LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: 'cosine', 'step', 'exponential'
        epochs: Total training epochs
        **kwargs: Additional scheduler arguments

    Returns:
        Configured scheduler
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            **kwargs
        )
    elif scheduler_type == 'step':
        step_size = kwargs.pop('step_size', epochs // 3)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            **kwargs
        )
    elif scheduler_type == 'exponential':
        gamma = kwargs.pop('gamma', 0.99)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            **kwargs
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

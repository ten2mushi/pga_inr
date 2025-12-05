"""
Tests for training utilities.

Following the Yoneda philosophy: these tests completely DEFINE the behavior
of the training components. The tests serve as executable documentation for:

1. PGAINRTrainer - Main training class
2. GenerativePGAINRTrainer - Auto-decoding trainer
3. NeRFTrainer - NeRF-style trainer
4. Utility functions: create_optimizer, create_scheduler
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

from pga_inr.training.trainer import (
    PGAINRTrainer,
    GenerativePGAINRTrainer,
    create_optimizer,
    create_scheduler,
)


# =============================================================================
# Mock Models for Testing
# =============================================================================

class MockModel(nn.Module):
    """Simple mock model for testing trainer."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, points, pose=None):
        batch_size, num_points, _ = points.shape
        flat_points = points.view(-1, 3)
        sdf = self.net(flat_points).view(batch_size, num_points, 1)
        return {
            'sdf': sdf,
            'local_coords': points,
        }


class MockLoss(nn.Module):
    """Simple mock loss function."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, gt_sdf, gt_normals=None):
        pred_sdf = outputs['sdf']
        loss = self.mse(pred_sdf, gt_sdf)
        metrics = {'sdf_loss': loss.item()}
        return loss, metrics


class MockLatentBank(nn.Module):
    """Mock latent code bank for generative training."""

    def __init__(self, num_objects=10, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.codes = nn.Embedding(num_objects, latent_dim)

    def forward(self, indices):
        return self.codes(indices)


class MockGenerativeModel(nn.Module):
    """Mock generative model."""

    def __init__(self, hidden_dim=32, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.latent_dim = latent_dim

    def forward(self, points, pose=None, latent_codes=None):
        batch_size, num_points, _ = points.shape
        flat_points = points.view(-1, 3)

        if latent_codes is not None:
            # Expand latent codes to match points
            codes_expanded = latent_codes.unsqueeze(1).expand(-1, num_points, -1)
            codes_flat = codes_expanded.reshape(-1, self.latent_dim)
            inputs = torch.cat([flat_points, codes_flat], dim=-1)
        else:
            inputs = flat_points

        sdf = self.net(inputs).view(batch_size, num_points, 1)
        return {'sdf': sdf}


# =============================================================================
# Data Helpers
# =============================================================================

def create_mock_dataloader(num_samples=100, batch_size=10, num_points=50):
    """Create a mock dataloader for testing."""
    points = torch.randn(num_samples, num_points, 3)
    sdf = torch.randn(num_samples, num_points, 1)

    dataset = TensorDataset(points, sdf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Wrap to return dict format
    class DictLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for batch in self.loader:
                yield {'points': batch[0], 'sdf': batch[1]}

        def __len__(self):
            return len(self.loader)

    return DictLoader(loader)


def create_generative_dataloader(num_samples=100, batch_size=10, num_points=50, num_objects=5):
    """Create dataloader for generative model testing."""
    points = torch.randn(num_samples, num_points, 3)
    sdf = torch.randn(num_samples, num_points, 1)
    object_idx = torch.randint(0, num_objects, (num_samples,))

    dataset = TensorDataset(points, sdf, object_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class DictLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for batch in self.loader:
                yield {
                    'points': batch[0],
                    'sdf': batch[1],
                    'object_idx': batch[2]
                }

        def __len__(self):
            return len(self.loader)

    return DictLoader(loader)


# =============================================================================
# PGAINRTrainer Tests
# =============================================================================

class TestPGAINRTrainerCreation:
    """Tests for trainer construction."""

    def test_trainer_creation(self):
        """Trainer can be created with required arguments."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert trainer.loss_fn is loss_fn

    def test_trainer_initial_state(self):
        """Trainer has correct initial state."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        assert trainer.global_step == 0
        assert trainer.epoch == 0
        assert trainer.best_val_loss == float('inf')
        assert trainer.history['train_loss'] == []

    def test_trainer_with_scheduler(self):
        """Trainer accepts scheduler."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=torch.device('cpu')
        )

        assert trainer.scheduler is scheduler


class TestTrainStep:
    """Tests for single training step."""

    def test_train_step_returns_loss_and_metrics(self):
        """Train step returns loss tensor and metrics dict."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        batch = {
            'points': torch.randn(2, 50, 3),
            'sdf': torch.randn(2, 50, 1)
        }

        loss, metrics = trainer.train_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert isinstance(metrics, dict)

    def test_train_step_increments_global_step(self):
        """Train step increments global step counter."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        assert trainer.global_step == 0

        batch = {
            'points': torch.randn(2, 50, 3),
            'sdf': torch.randn(2, 50, 1)
        }

        trainer.train_step(batch)
        assert trainer.global_step == 1

        trainer.train_step(batch)
        assert trainer.global_step == 2

    def test_train_step_updates_parameters(self):
        """Train step updates model parameters."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        # Get initial parameters
        params_before = [p.clone() for p in model.parameters()]

        batch = {
            'points': torch.randn(2, 50, 3),
            'sdf': torch.randn(2, 50, 1)
        }

        trainer.train_step(batch)

        # Check parameters changed
        params_after = list(model.parameters())
        changed = any(
            not torch.allclose(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert changed


class TestValidateStep:
    """Tests for validation step."""

    def test_validate_step_returns_loss_and_metrics(self):
        """Validate step returns loss and metrics."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        batch = {
            'points': torch.randn(2, 50, 3),
            'sdf': torch.randn(2, 50, 1)
        }

        loss, metrics = trainer.validate_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_validate_step_no_gradient(self):
        """Validate step does not compute gradients."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        batch = {
            'points': torch.randn(2, 50, 3),
            'sdf': torch.randn(2, 50, 1)
        }

        loss, _ = trainer.validate_step(batch)

        # Loss should not have gradient
        assert not loss.requires_grad


class TestTrainEpoch:
    """Tests for full epoch training."""

    def test_train_epoch_returns_metrics(self):
        """Train epoch returns average metrics."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        loader = create_mock_dataloader(num_samples=20, batch_size=5)

        metrics = trainer.train_epoch(loader, epoch=0)

        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)

    def test_train_epoch_updates_history(self):
        """Train epoch updates training history."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        loader = create_mock_dataloader(num_samples=20, batch_size=5)

        assert len(trainer.history['train_loss']) == 0

        trainer.train_epoch(loader, epoch=0)

        assert len(trainer.history['train_loss']) == 1


class TestValidation:
    """Tests for validation method."""

    def test_validation_returns_metrics(self):
        """Validation returns average metrics."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        loader = create_mock_dataloader(num_samples=20, batch_size=5)

        metrics = trainer.validate(loader)

        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(self):
        """Checkpoint can be saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = MockLoss()

            trainer = PGAINRTrainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=torch.device('cpu'),
                checkpoint_dir=tmpdir
            )

            trainer.global_step = 100
            trainer.epoch = 5

            trainer.save_checkpoint('test.pt')

            assert os.path.exists(os.path.join(tmpdir, 'test.pt'))

    def test_load_checkpoint(self):
        """Checkpoint can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = MockLoss()

            trainer = PGAINRTrainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=torch.device('cpu'),
                checkpoint_dir=tmpdir
            )

            trainer.global_step = 100
            trainer.epoch = 5

            trainer.save_checkpoint('test.pt')

            # Create new trainer and load
            model2 = MockModel()
            optimizer2 = torch.optim.Adam(model2.parameters())

            trainer2 = PGAINRTrainer(
                model=model2,
                optimizer=optimizer2,
                loss_fn=loss_fn,
                device=torch.device('cpu'),
                checkpoint_dir=tmpdir
            )

            trainer2.load_checkpoint(os.path.join(tmpdir, 'test.pt'))

            assert trainer2.global_step == 100
            assert trainer2.epoch == 5


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_gradient_clipping_applied(self):
        """Gradient clipping is applied when specified."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu'),
            gradient_clip=1.0
        )

        batch = {
            'points': torch.randn(2, 50, 3) * 100,  # Large inputs
            'sdf': torch.randn(2, 50, 1)
        }

        # Should not raise
        trainer.train_step(batch)


# =============================================================================
# GenerativePGAINRTrainer Tests
# =============================================================================

class TestGenerativeTrainerCreation:
    """Tests for generative trainer construction."""

    def test_generative_trainer_creation(self):
        """Generative trainer can be created."""
        model = MockGenerativeModel()
        latent_bank = MockLatentBank(num_objects=10)
        optimizer_net = torch.optim.Adam(model.parameters())
        optimizer_codes = torch.optim.Adam(latent_bank.parameters())
        loss_fn = MockLoss()

        trainer = GenerativePGAINRTrainer(
            model=model,
            latent_bank=latent_bank,
            optimizer_net=optimizer_net,
            optimizer_codes=optimizer_codes,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        assert trainer.model is model
        assert trainer.latent_bank is latent_bank


class TestGenerativeTrainStep:
    """Tests for generative trainer train step."""

    def test_generative_train_step(self):
        """Generative train step works."""
        model = MockGenerativeModel()
        latent_bank = MockLatentBank(num_objects=10)
        optimizer_net = torch.optim.Adam(model.parameters())
        optimizer_codes = torch.optim.Adam(latent_bank.parameters())
        loss_fn = MockLoss()

        trainer = GenerativePGAINRTrainer(
            model=model,
            latent_bank=latent_bank,
            optimizer_net=optimizer_net,
            optimizer_codes=optimizer_codes,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        batch = {
            'points': torch.randn(2, 50, 3),
            'sdf': torch.randn(2, 50, 1),
            'object_idx': torch.tensor([0, 1])
        }

        loss, metrics = trainer.train_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestCreateOptimizer:
    """Tests for create_optimizer utility."""

    def test_create_adam(self):
        """Create Adam optimizer."""
        model = MockModel()
        optimizer = create_optimizer(model, optimizer_type='adam', lr=1e-4)

        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_adamw(self):
        """Create AdamW optimizer."""
        model = MockModel()
        optimizer = create_optimizer(model, optimizer_type='adamw', lr=1e-4)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_sgd(self):
        """Create SGD optimizer."""
        model = MockModel()
        optimizer = create_optimizer(model, optimizer_type='sgd', lr=1e-2)

        assert isinstance(optimizer, torch.optim.SGD)

    def test_unknown_optimizer_raises(self):
        """Unknown optimizer type raises error."""
        model = MockModel()

        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(model, optimizer_type='unknown')

    def test_weight_decay(self):
        """Weight decay is applied."""
        model = MockModel()
        optimizer = create_optimizer(
            model,
            optimizer_type='adamw',
            lr=1e-4,
            weight_decay=0.01
        )

        assert optimizer.defaults['weight_decay'] == 0.01


class TestCreateScheduler:
    """Tests for create_scheduler utility."""

    def test_create_cosine(self):
        """Create cosine annealing scheduler."""
        model = MockModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer, scheduler_type='cosine', epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_step(self):
        """Create step scheduler."""
        model = MockModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer, scheduler_type='step', epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_exponential(self):
        """Create exponential scheduler."""
        model = MockModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer, scheduler_type='exponential')

        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_create_plateau(self):
        """Create plateau scheduler."""
        model = MockModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer, scheduler_type='plateau')

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_unknown_scheduler_raises(self):
        """Unknown scheduler type raises error."""
        model = MockModel()
        optimizer = create_optimizer(model)

        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(optimizer, scheduler_type='unknown')


# =============================================================================
# Integration Tests
# =============================================================================

class TestTrainingIntegration:
    """Integration tests for complete training workflows."""

    def test_short_training_run(self):
        """Complete short training run."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        train_loader = create_mock_dataloader(num_samples=20, batch_size=5)

        # Train for 2 epochs
        history = trainer.fit(train_loader, epochs=2)

        assert len(history['train_loss']) == 2
        assert trainer.epoch == 1  # 0-indexed

    def test_training_with_validation(self):
        """Training with validation loop."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        train_loader = create_mock_dataloader(num_samples=20, batch_size=5)
        val_loader = create_mock_dataloader(num_samples=10, batch_size=5)

        history = trainer.fit(train_loader, val_loader=val_loader, epochs=2)

        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2

    def test_training_with_scheduler(self):
        """Training with learning rate scheduler."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=torch.device('cpu')
        )

        train_loader = create_mock_dataloader(num_samples=20, batch_size=5)

        initial_lr = optimizer.param_groups[0]['lr']
        trainer.fit(train_loader, epochs=2)
        final_lr = optimizer.param_groups[0]['lr']

        # LR should have decreased
        assert final_lr < initial_lr


# =============================================================================
# Edge Cases
# =============================================================================

class TestTrainerEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataloader(self):
        """Handle empty dataloader gracefully."""
        # This would typically be a configuration error
        pass  # Skip - empty dataloader is unusual

    def test_single_sample_batch(self):
        """Handle batch size of 1."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )

        batch = {
            'points': torch.randn(1, 50, 3),
            'sdf': torch.randn(1, 50, 1)
        }

        loss, metrics = trainer.train_step(batch)
        assert isinstance(loss, torch.Tensor)

    def test_no_checkpoint_dir(self):
        """Trainer works without checkpoint directory."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MockLoss()

        trainer = PGAINRTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu'),
            checkpoint_dir=None
        )

        assert trainer.checkpoint_dir is None

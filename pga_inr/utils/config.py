"""
Configuration management for PGA-INR.

Provides configuration classes and utilities for managing model hyperparameters.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class Config:
    """
    Configuration for PGA-INR models and training.

    Attributes:
        # Model architecture
        hidden_features: Number of hidden features in network
        hidden_layers: Number of hidden layers
        omega_0: SIREN frequency parameter for first layer
        omega_hidden: SIREN frequency for hidden layers
        output_normals: Whether to output surface normals
        geometric_init: Use geometric initialization for SDF

        # Generative model
        latent_dim: Dimension of latent code
        use_hyper_network: Whether to use HyperNetwork

        # Training
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        num_epochs: Number of training epochs
        num_points_per_sample: Points sampled per training step

        # Loss weights
        lambda_sdf: Weight for SDF reconstruction loss
        lambda_eikonal: Weight for Eikonal loss
        lambda_normal: Weight for normal alignment loss
        lambda_latent: Weight for latent regularization

        # Data
        surface_sample_ratio: Ratio of surface vs uniform samples

        # Device
        device: Device to use ('cuda', 'mps', 'cpu')
    """

    # Model architecture
    hidden_features: int = 256
    hidden_layers: int = 4
    omega_0: float = 30.0
    omega_hidden: float = 30.0
    output_normals: bool = True
    geometric_init: bool = True

    # Generative model
    latent_dim: int = 64
    use_hyper_network: bool = False

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 100
    num_points_per_sample: int = 5000

    # Loss weights
    lambda_sdf: float = 1.0
    lambda_eikonal: float = 0.1
    lambda_normal: float = 1.0
    lambda_latent: float = 0.001

    # Data
    surface_sample_ratio: float = 0.5

    # Device
    device: str = 'cuda'

    # Additional fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        # Extract known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        known_kwargs = {k: v for k, v in config_dict.items() if k in known_fields}
        extra_kwargs = {k: v for k, v in config_dict.items() if k not in known_fields}

        config = cls(**known_kwargs)
        config.extra = extra_kwargs
        return config

    def update(self, **kwargs) -> 'Config':
        """Return a new config with updated values."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return Config.from_dict(config_dict)


def load_config(filepath: str) -> Config:
    """
    Load configuration from JSON file.

    Args:
        filepath: Path to JSON config file

    Returns:
        Config object
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return Config.from_dict(config_dict)


def save_config(config: Config, filepath: str) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Config object to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

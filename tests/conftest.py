"""
Pytest configuration and fixtures for PGA-INR tests.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@pytest.fixture
def cpu_device():
    """Force CPU device for consistent testing."""
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def num_points():
    """Default number of points for tests."""
    return 100


@pytest.fixture
def identity_quaternion():
    """Identity quaternion [w, x, y, z]."""
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]])


@pytest.fixture
def identity_pose(identity_quaternion):
    """Identity pose (zero translation, identity rotation)."""
    trans = torch.zeros(1, 3)
    return (trans, identity_quaternion)


@pytest.fixture
def random_points(batch_size, num_points):
    """Random points in [-1, 1]^3."""
    return torch.rand(batch_size, num_points, 3) * 2 - 1


@pytest.fixture
def simple_sdf_model():
    """Simple SDF model for testing."""
    from pga_inr.models import PGA_INR_SDF

    return PGA_INR_SDF(
        hidden_features=32,
        hidden_layers=2,
        omega_0=30.0
    )


@pytest.fixture
def simple_joint_tree():
    """Simple 2-joint kinematic chain definition."""
    return {
        'root': {
            'parent': None,
            'rest_translation': [0.0, 0.0, 0.0],
            'rest_rotation': [1.0, 0.0, 0.0, 0.0],
            'axis': [0.0, 0.0, 1.0],
            'children': ['end']
        },
        'end': {
            'parent': 'root',
            'rest_translation': [1.0, 0.0, 0.0],
            'rest_rotation': [1.0, 0.0, 0.0, 0.0],
            'axis': [0.0, 0.0, 1.0],
            'children': []
        }
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

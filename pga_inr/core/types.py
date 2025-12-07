"""
Type aliases and shape conventions for PGA-INR.

This module defines type aliases for common tensor shapes used throughout
the library. It also documents the shape conventions, particularly for
motion data.

Shape Conventions:
==================

Motion Tensor Shape Convention: (B, J, D, T)
--------------------------------------------
All motion tensors in this library follow the convention:
    - B: Batch size
    - J: Number of joints
    - D: Data dimension (6 for 6D rotation, 3 for position, etc.)
    - T: Number of time frames

This differs from some other motion libraries that use (B, T, J, D).
The (B, J, D, T) convention was chosen because:
    1. It groups spatial data (J, D) together, making joint operations natural
    2. Temporal convolutions can operate on the last dimension directly
    3. It mirrors how animation data is often stored (per-joint channels over time)

Rationale:
    - Joint-centric operations (FK, IK, joint selection) index on dims 1-2
    - Temporal operations (velocity, smoothing) operate on dim 3
    - This layout enables efficient Conv1d over time without permutation

Example:
    rotations: Tensor[B, J, 6, T]  # 6D rotations for each joint over time
    positions: Tensor[B, J, 3, T]  # 3D positions for each joint over time
    velocities: Tensor[B, J, 3, T] # 3D velocities for each joint over time

To convert from (B, T, J, D) to our convention:
    tensor_bjdt = tensor_btjd.permute(0, 2, 3, 1)

To convert to (B, T, J, D) from our convention:
    tensor_btjd = tensor_bjdt.permute(0, 3, 1, 2)
"""

from typing import Dict, Tuple, Optional, Union, TypeVar
import torch


# =============================================================================
# Generic Type Variables
# =============================================================================

T = TypeVar('T', bound=torch.Tensor)


# =============================================================================
# Basic Type Aliases
# =============================================================================

# Output dictionary from model forward pass
TensorDict = Dict[str, torch.Tensor]

# Observer pose as (translation, quaternion) tuple
# - translation: (B, 3) or (3,) world-space translation
# - quaternion: (B, 4) or (4,) as [w, x, y, z] convention
ObserverPose = Tuple[torch.Tensor, torch.Tensor]


# =============================================================================
# Motion Type Aliases
# =============================================================================

# Motion tensor: (B, J, D, T) - rotations or positions over time
# See module docstring for shape convention rationale
MotionTensor = torch.Tensor

# Rotation tensor: (B, J, 6, T) - 6D rotation representation
# The 6D representation is continuous and differentiable (Zhou et al., 2019)
RotationTensor = torch.Tensor

# Joint position tensor: (B, J, 3, T) - 3D world positions
JointPositionTensor = torch.Tensor

# Trajectory tensor: (B, D, T) - root trajectory (usually D=2 for XZ or D=3)
TrajectoryTensor = torch.Tensor


# =============================================================================
# Shape Convention Documentation
# =============================================================================

MOTION_SHAPE_CONVENTION: str = """
Motion Tensor Shape Convention: (B, J, D, T)
============================================

All motion tensors follow this axis ordering:
    Axis 0 (B): Batch dimension
    Axis 1 (J): Joint dimension (number of skeleton joints)
    Axis 2 (D): Data dimension (6 for rotation, 3 for position)
    Axis 3 (T): Temporal dimension (number of frames)

Standard shapes:
    - Rotations (6D):   (batch_size, num_joints, 6, num_frames)
    - Positions (3D):   (batch_size, num_joints, 3, num_frames)
    - Velocities (3D):  (batch_size, num_joints, 3, num_frames)
    - Root trajectory:  (batch_size, 2, num_frames)  # XZ plane
    - Root rotation:    (batch_size, 6, num_frames)  # 6D rotation

Conversion utilities:
    # From (B, T, J, D) to (B, J, D, T):
    our_format = other_format.permute(0, 2, 3, 1)

    # From (B, J, D, T) to (B, T, J, D):
    other_format = our_format.permute(0, 3, 1, 2)

Why this convention?
    1. Joint operations are contiguous in memory (dims 1-2)
    2. Conv1d temporal smoothing works on last dim without permute
    3. Matches animation channel layout in many DCC tools
"""


def validate_motion_shape(
    tensor: torch.Tensor,
    expected_joints: Optional[int] = None,
    expected_dim: Optional[int] = None,
    name: str = "tensor"
) -> None:
    """
    Validate that a tensor follows the motion shape convention (B, J, D, T).

    Args:
        tensor: Tensor to validate
        expected_joints: Expected number of joints (axis 1)
        expected_dim: Expected data dimension (axis 2)
        name: Name for error messages

    Raises:
        ValueError: If tensor doesn't match expected shape convention
    """
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} should have 4 dimensions (B, J, D, T), got {tensor.ndim}"
        )

    if expected_joints is not None and tensor.shape[1] != expected_joints:
        raise ValueError(
            f"{name} should have {expected_joints} joints (axis 1), "
            f"got {tensor.shape[1]}"
        )

    if expected_dim is not None and tensor.shape[2] != expected_dim:
        raise ValueError(
            f"{name} should have dimension {expected_dim} (axis 2), "
            f"got {tensor.shape[2]}"
        )


def convert_to_motion_convention(
    tensor: torch.Tensor,
    source_format: str = "BTJD"
) -> torch.Tensor:
    """
    Convert tensor to PGA-INR motion convention (B, J, D, T).

    Args:
        tensor: Input tensor
        source_format: Source format string, e.g., "BTJD", "TJDB"

    Returns:
        Tensor in (B, J, D, T) format
    """
    source_format = source_format.upper()

    if source_format == "BJDT":
        return tensor
    elif source_format == "BTJD":
        return tensor.permute(0, 2, 3, 1)
    elif source_format == "TJDB":
        return tensor.permute(3, 1, 2, 0)
    elif source_format == "TJD":
        # Single sequence without batch - add batch dim
        return tensor.permute(1, 2, 0).unsqueeze(0)
    else:
        raise ValueError(
            f"Unknown source format: {source_format}. "
            f"Supported: BJDT, BTJD, TJDB, TJD"
        )


def convert_from_motion_convention(
    tensor: torch.Tensor,
    target_format: str = "BTJD"
) -> torch.Tensor:
    """
    Convert tensor from PGA-INR motion convention (B, J, D, T) to other format.

    Args:
        tensor: Input tensor in (B, J, D, T) format
        target_format: Target format string, e.g., "BTJD", "TJDB"

    Returns:
        Tensor in target format
    """
    target_format = target_format.upper()

    if target_format == "BJDT":
        return tensor
    elif target_format == "BTJD":
        return tensor.permute(0, 3, 1, 2)
    elif target_format == "TJDB":
        return tensor.permute(3, 1, 2, 0)
    elif target_format == "TJD":
        # Remove batch dim and permute
        if tensor.shape[0] != 1:
            raise ValueError("Cannot convert to TJD format with batch size > 1")
        return tensor.squeeze(0).permute(2, 0, 1)
    else:
        raise ValueError(
            f"Unknown target format: {target_format}. "
            f"Supported: BJDT, BTJD, TJDB, TJD"
        )

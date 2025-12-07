"""
PGA-SLAM: Neural Implicit SLAM with Projective Geometric Algebra.

A complete SLAM system combining:
- Pose tracking with Lie algebra optimization on SE(3)
- Neural implicit mapping with PGA-INR
- Keyframe-based bundle adjustment
- Loop closure detection and pose graph optimization
- Real-time frontend/backend architecture

Example Usage
-------------
>>> from pga_inr.slam import SLAMConfig, SLAMSystem
>>> from pga_inr.data import FMDatasetLoader
>>>
>>> # Load dataset
>>> loader = FMDatasetLoader("input/slam/fm_dataset/extracted/livingroom_medium")
>>> frames = loader.get_frame_generator()
>>>
>>> # Create SLAM system
>>> config = SLAMConfig.for_fm_dataset_high_end()
>>> slam = SLAMSystem(config)
>>>
>>> # Run SLAM
>>> slam.start_backend()
>>> state = slam.run(frames, max_frames=500)
>>> slam.finalize()
>>>
>>> # Visualize results
>>> from pga_inr.slam import SLAMVisualizer
>>> viz = SLAMVisualizer()
>>> viz.plot_slam_dashboard(state)
"""

# Configuration
from .config import (
    CameraIntrinsics,
    TrackerConfig,
    MapperConfig,
    KeyframeConfig,
    BackendConfig,
    SLAMConfig,
)

# Type definitions
from .types import (
    Frame,
    Keyframe,
    TrackingResult,
    MappingResult,
    LoopClosureResult,
    SLAMState,
)

# Core components
from .tracker import PoseOptimizer, MotionModel
from .mapper import NeuralMapper
from .keyframe import KeyframeManager
from .pose_graph import PoseGraph, PoseGraphEdge

# High-level systems
from .frontend import SLAMFrontend
from .backend import SLAMBackend, SLAMSystem

# Losses
from .losses import (
    DirectDepthSDFLoss,
    TruncatedSDFLoss,
    FreeSpaceLoss,
    PhotometricLoss,
    DepthLoss,
    PoseGraphLoss,
    SLAMLoss,
)

# Visualization
from .visualization import (
    SLAMVisualizer,
    create_trajectory_animation,
    visualize_tracking_result,
)

# Ground truth generation (pseudo-GT from IMU/ICP)
from .ground_truth import (
    IMUIntegrator,
    DepthICPTracker,
    load_imu_data,
    generate_imu_trajectory,
    generate_icp_trajectory,
)

__all__ = [
    # Configuration
    "CameraIntrinsics",
    "TrackerConfig",
    "MapperConfig",
    "KeyframeConfig",
    "BackendConfig",
    "SLAMConfig",
    # Types
    "Frame",
    "Keyframe",
    "TrackingResult",
    "MappingResult",
    "LoopClosureResult",
    "SLAMState",
    # Core
    "PoseOptimizer",
    "MotionModel",
    "NeuralMapper",
    "KeyframeManager",
    "PoseGraph",
    "PoseGraphEdge",
    # Systems
    "SLAMFrontend",
    "SLAMBackend",
    "SLAMSystem",
    # Losses
    "DirectDepthSDFLoss",
    "TruncatedSDFLoss",
    "FreeSpaceLoss",
    "PhotometricLoss",
    "DepthLoss",
    "PoseGraphLoss",
    "SLAMLoss",
    # Visualization
    "SLAMVisualizer",
    "create_trajectory_animation",
    "visualize_tracking_result",
    # Ground truth generation
    "IMUIntegrator",
    "DepthICPTracker",
    "load_imu_data",
    "generate_imu_trajectory",
    "generate_icp_trajectory",
]

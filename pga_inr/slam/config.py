"""
Configuration dataclasses for PGA-SLAM.

Provides structured configuration for all SLAM components including
tracking, mapping, keyframe management, and backend optimization.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 1000.0  # Divide raw depth by this to get meters

    def to_matrix(self) -> torch.Tensor:
        """Convert to 3x3 intrinsics matrix."""
        return torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)

    def scaled(self, scale: float) -> 'CameraIntrinsics':
        """Return scaled intrinsics (for downsampling)."""
        return CameraIntrinsics(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=int(self.width * scale),
            height=int(self.height * scale),
            depth_scale=self.depth_scale
        )


@dataclass
class TrackerConfig:
    """Configuration for pose tracking."""
    # Optimization
    num_iterations: int = 30          # Iterations per frame (fewer with more samples)
    learning_rate: float = 0.01       # Adam LR for pose
    convergence_threshold: float = 1e-5

    # Sampling - High-end GPU settings
    num_rays: int = 4096              # Rays per iteration
    num_depth_samples: int = 2048     # Points sampled from depth
    use_stratified_sampling: bool = True
    sample_near_surface: bool = True
    surface_band: float = 0.05        # Meters

    # Loss weights
    lambda_depth: float = 1.0
    lambda_rgb: float = 0.1
    lambda_sdf: float = 1.0
    lambda_eikonal: float = 0.01

    # Truncation
    truncation_distance: float = 0.1  # TSDF truncation in meters

    # Robustness
    use_huber_loss: bool = True
    huber_delta: float = 0.02


@dataclass
class MapperConfig:
    """Configuration for neural map updates."""
    # Network architecture - High-end GPU settings
    hidden_features: int = 512
    hidden_layers: int = 8
    omega_0: float = 30.0
    omega_hidden: float = 30.0
    use_positional_encoding: bool = True
    num_frequencies: int = 8
    geometric_init: bool = True

    # Optimization
    num_iterations: int = 100         # Iterations per keyframe
    learning_rate: float = 1e-4
    batch_size: int = 16384           # Points per batch

    # Loss weights
    lambda_sdf: float = 1.0
    lambda_eikonal: float = 0.1
    lambda_free_space: float = 0.5
    lambda_smoothness: float = 0.01

    # Sampling
    surface_sample_ratio: float = 0.5
    volume_sample_ratio: float = 0.3
    free_space_sample_ratio: float = 0.2
    bounds: Tuple[float, float, float, float, float, float] = (-3, 3, -3, 3, -3, 3)

    # Surface noise for augmentation
    surface_noise_std: float = 0.01   # 1cm noise on surface points


@dataclass
class KeyframeConfig:
    """Configuration for keyframe management."""
    # Selection criteria
    min_translation: float = 0.1      # Meters
    min_rotation: float = 0.1         # Radians
    max_tracking_error: float = 0.5   # SDF error threshold (relaxed for robustness)
    min_frame_interval: int = 5       # Minimum frames between keyframes

    # Storage
    max_keyframes: int = 100          # Maximum stored keyframes
    downsample_factor: int = 4        # Depth image downsampling for storage

    # Covisibility
    min_covisibility: float = 0.3     # Minimum overlap for covisibility


@dataclass
class BackendConfig:
    """Configuration for backend optimization."""
    # Bundle adjustment
    ba_window_size: int = 10          # Keyframes in local BA
    ba_iterations: int = 200
    ba_learning_rate_pose: float = 0.001
    ba_learning_rate_map: float = 1e-4

    # Loop closure
    enable_loop_closure: bool = True
    loop_detection_threshold: float = 2.0   # Distance threshold in meters
    min_loop_interval: int = 20       # Minimum keyframes between loops

    # Scheduling
    ba_frequency: int = 5             # Run BA every N keyframes

    # Pose graph
    information_scale: float = 1.0    # Scale for information matrix


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    # Rendering
    render_width: int = 640
    render_height: int = 480
    render_fov: float = 60.0

    # Trajectory visualization
    trajectory_color: Tuple[float, float, float] = (0.0, 0.5, 1.0)
    keyframe_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    # Map visualization
    map_slice_resolution: int = 256
    map_bounds: Tuple[float, float] = (-3.0, 3.0)

    # Output
    save_renders: bool = True
    render_every_n_frames: int = 10


@dataclass
class SLAMConfig:
    """Complete SLAM configuration."""
    # Sub-configs
    camera: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        fx=608.0, fy=608.0, cx=331.0, cy=246.0, width=640, height=480
    ))
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # General
    device: str = 'cuda'
    dtype: str = 'float32'  # 'float32' or 'float16'
    verbose: bool = True

    # Paths
    output_dir: str = 'output/slam'
    checkpoint_dir: str = 'checkpoints/slam'

    @classmethod
    def for_fm_dataset(cls) -> 'SLAMConfig':
        """Factory for FMDataset settings."""
        return cls(
            camera=CameraIntrinsics(
                fx=608.0, fy=608.0, cx=331.0, cy=246.0,
                width=640, height=480, depth_scale=1000.0
            )
        )

    @classmethod
    def for_fm_dataset_high_end(cls) -> 'SLAMConfig':
        """Factory for FMDataset with high-end GPU settings."""
        config = cls.for_fm_dataset()
        # Increase capacity for high-end GPU
        config.tracker.num_rays = 4096
        config.tracker.num_depth_samples = 2048
        config.mapper.hidden_features = 512
        config.mapper.hidden_layers = 8
        config.mapper.batch_size = 16384
        config.mapper.num_frequencies = 8
        return config

    @classmethod
    def for_testing(cls) -> 'SLAMConfig':
        """Factory for quick testing with reduced settings."""
        config = cls.for_fm_dataset()
        config.tracker.num_iterations = 10
        config.tracker.num_depth_samples = 512
        config.mapper.num_iterations = 20
        config.mapper.hidden_features = 128
        config.mapper.hidden_layers = 4
        config.mapper.batch_size = 4096
        config.keyframe.max_keyframes = 20
        return config

    def get_dtype(self) -> torch.dtype:
        """Get torch dtype from string."""
        if self.dtype == 'float16':
            return torch.float16
        return torch.float32

    def get_device(self) -> torch.device:
        """Get torch device."""
        return torch.device(self.device)

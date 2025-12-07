"""
Tests for the SLAM module.

Tests cover:
- Configuration dataclasses
- Type definitions
- Loss functions
- Pose graph operations
- Tracker optimization
- Mapper updates
- Frontend integration
"""

import pytest
import torch
import numpy as np

from pga_inr.slam import (
    # Config
    CameraIntrinsics,
    TrackerConfig,
    MapperConfig,
    KeyframeConfig,
    BackendConfig,
    SLAMConfig,
    # Types
    Frame,
    Keyframe,
    TrackingResult,
    MappingResult,
    LoopClosureResult,
    SLAMState,
    # Core components
    PoseOptimizer,
    MotionModel,
    NeuralMapper,
    KeyframeManager,
    PoseGraph,
    PoseGraphEdge,
    # Losses
    DirectDepthSDFLoss,
    TruncatedSDFLoss,
    FreeSpaceLoss,
    PoseGraphLoss,
)
from pga_inr.models.inr import PGA_INR_SDF_V2


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def camera_intrinsics():
    """Create test camera intrinsics."""
    return CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        depth_scale=1000.0
    )


@pytest.fixture
def slam_config(camera_intrinsics):
    """Create test SLAM config."""
    return SLAMConfig(
        camera=camera_intrinsics,
        tracker=TrackerConfig(),
        mapper=MapperConfig(
            hidden_features=64,
            hidden_layers=2,
            batch_size=256,
            num_iterations=5
        ),
        keyframe=KeyframeConfig(),
        backend=BackendConfig()
    )


class TestConfiguration:
    """Test configuration dataclasses."""

    def test_camera_intrinsics_creation(self):
        """Test CameraIntrinsics creation."""
        intrinsics = CameraIntrinsics(
            fx=608.0, fy=608.0, cx=331.0, cy=246.0,
            width=640, height=480, depth_scale=1000.0
        )
        assert intrinsics.fx == 608.0
        assert intrinsics.width == 640
        assert intrinsics.depth_scale == 1000.0

    def test_tracker_config_defaults(self):
        """Test TrackerConfig has sensible defaults."""
        config = TrackerConfig()
        assert config.num_iterations > 0
        assert config.learning_rate > 0
        assert config.num_depth_samples > 0

    def test_mapper_config_defaults(self):
        """Test MapperConfig has sensible defaults."""
        config = MapperConfig()
        assert config.hidden_features > 0
        assert config.hidden_layers > 0
        assert config.batch_size > 0

    def test_slam_config_for_fm_dataset(self):
        """Test preset configuration creation."""
        config = SLAMConfig.for_fm_dataset()
        assert config.camera.fx == 608.0  # FMDataset intrinsics
        assert config.camera.width == 640

    def test_slam_config_high_end(self):
        """Test high-end GPU configuration."""
        config = SLAMConfig.for_fm_dataset_high_end()
        # High-end should have larger batch sizes
        assert config.mapper.batch_size > 4096
        assert config.mapper.hidden_features >= 256


class TestTypes:
    """Test type definitions."""

    def test_frame_creation(self, device):
        """Test Frame creation."""
        rgb = torch.rand(480, 640, 3, device=device)
        depth = torch.rand(480, 640, device=device) * 5.0
        frame = Frame(rgb=rgb, depth=depth, timestamp=0.0, frame_id=0)

        assert frame.rgb.shape == (480, 640, 3)
        assert frame.depth.shape == (480, 640)
        assert frame.frame_id == 0

    def test_keyframe_creation(self, device):
        """Test Keyframe creation."""
        rgb = torch.rand(480, 640, 3, device=device)
        depth = torch.rand(480, 640, device=device) * 5.0
        translation = torch.zeros(3, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        kf = Keyframe(
            frame_id=0,
            timestamp=0.0,
            rgb=rgb,
            depth=depth,
            translation=translation,
            quaternion=quaternion,
            tracking_error=0.01
        )

        assert kf.frame_id == 0
        assert kf.translation.shape == (3,)
        assert kf.quaternion.shape == (4,)

    def test_tracking_result(self, device):
        """Test TrackingResult creation."""
        result = TrackingResult(
            translation=torch.zeros(3, device=device),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            converged=True,
            num_iterations=10,
            final_loss=0.001,
            inlier_ratio=0.95,
            loss_breakdown={'sdf': 0.001}
        )

        assert result.converged
        assert result.final_loss == 0.001


class TestLosses:
    """Test SLAM loss functions."""

    def test_direct_depth_sdf_loss(self, device):
        """Test DirectDepthSDFLoss."""
        loss_fn = DirectDepthSDFLoss(truncation=0.1, use_huber=False)

        # SDF predictions (should be near 0 at surface)
        pred_sdf = torch.tensor([[0.01], [-0.02], [0.0]], device=device)

        loss = loss_fn(pred_sdf, target_sdf=None)
        assert loss >= 0
        assert not torch.isnan(loss)

    def test_truncated_sdf_loss(self, device):
        """Test TruncatedSDFLoss."""
        loss_fn = TruncatedSDFLoss(truncation=0.1, use_huber=True)

        pred_sdf = torch.tensor([[0.15], [-0.05], [0.02]], device=device)
        target_sdf = torch.tensor([[0.1], [-0.1], [0.0]], device=device)

        loss = loss_fn(pred_sdf, target_sdf)
        assert loss >= 0
        assert not torch.isnan(loss)

    def test_free_space_loss(self, device):
        """Test FreeSpaceLoss."""
        loss_fn = FreeSpaceLoss(margin=0.01)

        # Free space should have positive SDF
        positive_sdf = torch.tensor([[0.1], [0.2], [0.05]], device=device)
        negative_sdf = torch.tensor([[-0.1], [-0.05]], device=device)

        loss_pos = loss_fn(positive_sdf)
        loss_neg = loss_fn(negative_sdf)

        # Positive SDF should have lower loss
        assert loss_pos < loss_neg

    def test_pose_graph_loss(self, device):
        """Test PoseGraphLoss."""
        loss_fn = PoseGraphLoss()

        # Create poses in lie algebra (se3: [tx, ty, tz, rx, ry, rz])
        poses_lie = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Identity
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Translated by 1
        ], device=device)

        # Edge from 0 to 1
        edges = torch.tensor([[0, 1]], device=device)

        # Measured relative transform (same as actual)
        measurements_lie = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ], device=device)

        loss = loss_fn(poses_lie, edges, measurements_lie)

        # Identical transforms should have near-zero loss
        assert loss >= 0
        assert not torch.isnan(loss)


class TestPoseGraph:
    """Test pose graph operations."""

    def test_add_nodes(self, device):
        """Test adding nodes to pose graph."""
        graph = PoseGraph(device)

        graph.add_node(0, torch.zeros(3, device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        graph.add_node(1, torch.tensor([1.0, 0.0, 0.0], device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

        assert graph.num_nodes() == 2

    def test_add_edges(self, device):
        """Test adding edges to pose graph."""
        graph = PoseGraph(device)

        # Add nodes
        graph.add_node(0, torch.zeros(3, device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        graph.add_node(1, torch.tensor([1.0, 0.0, 0.0], device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

        # Add odometry edge
        graph.add_odometry_edge(0, 1)

        assert graph.num_edges() == 1

    def test_optimization(self, device):
        """Test pose graph optimization."""
        graph = PoseGraph(device)

        # Create a simple chain of poses
        for i in range(5):
            graph.add_node(
                i,
                torch.tensor([float(i), 0.0, 0.0], device=device),
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            )
            if i > 0:
                graph.add_odometry_edge(i-1, i)

        # Run optimization
        loss_history = graph.optimize(num_iterations=10)

        assert len(loss_history) == 10
        # Loss should decrease (or stay flat for consistent poses)
        assert loss_history[-1] <= loss_history[0] + 0.1  # Allow small tolerance

    def test_loop_closure_detection(self, device):
        """Test loop closure candidate detection."""
        graph = PoseGraph(device)

        # Create a loop (poses 0 and 4 are spatially close)
        graph.add_node(0, torch.tensor([0.0, 0.0, 0.0], device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        graph.add_node(1, torch.tensor([2.0, 0.0, 0.0], device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        graph.add_node(2, torch.tensor([2.0, 2.0, 0.0], device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        graph.add_node(3, torch.tensor([0.0, 2.0, 0.0], device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        graph.add_node(4, torch.tensor([0.1, 0.1, 0.0], device=device),  # Close to node 0
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

        # Node 4 should find node 0 as loop closure candidate
        candidates = graph.detect_loop_candidates(
            4, distance_threshold=0.5, min_interval=3
        )

        assert 0 in candidates

    def test_get_trajectory(self, device):
        """Test trajectory extraction."""
        graph = PoseGraph(device)

        graph.add_node(0, torch.tensor([0.0, 0.0, 0.0], device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        graph.add_node(1, torch.tensor([1.0, 0.0, 0.0], device=device),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

        trajectory = graph.get_trajectory()

        assert len(trajectory) == 2
        assert trajectory[0][0] == 0
        assert trajectory[1][0] == 1


class TestMotionModel:
    """Test motion model for pose prediction."""

    def test_first_prediction(self, device):
        """Test prediction with no history."""
        model = MotionModel(device)
        t, q = model.predict()

        # Should return identity with no history
        assert torch.allclose(t, torch.zeros(3, device=device))
        assert torch.allclose(q, torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

    def test_constant_velocity(self, device):
        """Test constant velocity prediction."""
        model = MotionModel(device)

        # Update with two poses
        model.update(
            torch.tensor([0.0, 0.0, 0.0], device=device),
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        )
        model.update(
            torch.tensor([1.0, 0.0, 0.0], device=device),
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        )

        # Predict should extrapolate
        t, q = model.predict()

        # Should predict motion of +1 in x
        assert t[0] > 1.0


class TestKeyframeManager:
    """Test keyframe management."""

    def test_add_keyframe(self, device, camera_intrinsics):
        """Test adding keyframes."""
        config = KeyframeConfig()
        manager = KeyframeManager(config, camera_intrinsics, device)

        rgb = torch.rand(480, 640, 3, device=device)
        depth = torch.rand(480, 640, device=device) * 5.0
        frame = Frame(rgb=rgb, depth=depth, timestamp=0.0, frame_id=0)

        kf = manager.add_keyframe(
            frame,
            translation=torch.zeros(3, device=device),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            tracking_error=0.01
        )

        assert len(manager.keyframes) == 1
        assert kf.frame_id == 0

    def test_keyframe_selection(self, device, camera_intrinsics):
        """Test keyframe selection criteria."""
        config = KeyframeConfig(
            min_translation=0.5,
            min_rotation=0.1,
            min_frame_interval=5
        )
        manager = KeyframeManager(config, camera_intrinsics, device)

        # Add first keyframe
        rgb = torch.rand(480, 640, 3, device=device)
        depth = torch.rand(480, 640, device=device) * 5.0
        frame = Frame(rgb=rgb, depth=depth, timestamp=0.0, frame_id=0)

        manager.add_keyframe(
            frame,
            translation=torch.zeros(3, device=device),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            tracking_error=0.01
        )

        # Check if we should add a new keyframe (small movement)
        should_add = manager.should_add_keyframe(
            current_translation=torch.tensor([0.1, 0.0, 0.0], device=device),
            current_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            tracking_error=0.01,
            frame_id=1
        )
        assert not should_add  # Too close and too few frames

        # Large movement
        should_add = manager.should_add_keyframe(
            current_translation=torch.tensor([1.0, 0.0, 0.0], device=device),
            current_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            tracking_error=0.01,
            frame_id=10
        )
        assert should_add  # Large translation


class TestNeuralMapper:
    """Test neural mapper."""

    def test_mapper_initialization(self, device):
        """Test mapper initialization."""
        config = MapperConfig(
            hidden_features=64,
            hidden_layers=2,
            batch_size=256
        )
        mapper = NeuralMapper(config, device)

        model = mapper.get_model()
        assert model is not None
        assert isinstance(model, PGA_INR_SDF_V2)

    def test_update_from_keyframe(self, device, camera_intrinsics):
        """Test map update from keyframe."""
        config = MapperConfig(
            hidden_features=64,
            hidden_layers=2,
            batch_size=256,
            num_iterations=3
        )
        mapper = NeuralMapper(config, device)

        # Create a simple keyframe with some 3D points
        rgb = torch.rand(480, 640, 3, device=device)
        depth = torch.rand(480, 640, device=device) * 2.0 + 1.0  # 1-3m depth

        # Compute 3D points from depth
        u = torch.arange(640, device=device, dtype=torch.float32)
        v = torch.arange(480, device=device, dtype=torch.float32)
        v, u = torch.meshgrid(v, u, indexing='ij')

        z = depth
        x = (u - camera_intrinsics.cx) * z / camera_intrinsics.fx
        y = (v - camera_intrinsics.cy) * z / camera_intrinsics.fy
        points_3d = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        kf = Keyframe(
            frame_id=0,
            timestamp=0.0,
            rgb=rgb,
            depth=depth,
            translation=torch.zeros(3, device=device),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            tracking_error=0.01,
            points_3d=points_3d[:1000],  # Subset
            points_world=points_3d[:1000]
        )

        # Update map
        result = mapper.update_from_keyframe(kf, num_iterations=3)

        assert isinstance(result, MappingResult)
        assert len(result.loss_history) == 3
        assert result.final_loss >= 0


class TestIntegration:
    """Integration tests for SLAM components."""

    def test_slam_module_imports(self):
        """Test that all module imports work."""
        from pga_inr import slam
        assert hasattr(slam, 'SLAMConfig')
        assert hasattr(slam, 'SLAMSystem')
        assert hasattr(slam, 'SLAMFrontend')
        assert hasattr(slam, 'SLAMBackend')
        assert hasattr(slam, 'SLAMVisualizer')

    def test_slam_config_creation_chain(self):
        """Test creating a full SLAM config."""
        config = SLAMConfig.for_fm_dataset()

        # All sub-configs should be present
        assert config.camera is not None
        assert config.tracker is not None
        assert config.mapper is not None
        assert config.keyframe is not None
        assert config.backend is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

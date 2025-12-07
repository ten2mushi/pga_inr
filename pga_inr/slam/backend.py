"""
Background optimization for SLAM.

Runs in separate thread to perform:
- Local bundle adjustment
- Loop closure detection and optimization
- Global pose graph optimization
"""

from typing import List, Optional, Callable
import torch
import threading
import queue
import time

from .config import BackendConfig
from .types import Keyframe, LoopClosureResult
from .mapper import NeuralMapper
from .pose_graph import PoseGraph


class SLAMBackend:
    """
    Background optimization for SLAM.

    Runs in separate thread to perform:
    - Local bundle adjustment on recent keyframes
    - Loop closure detection based on spatial proximity
    - Global pose graph optimization

    The backend communicates with the frontend via callbacks and queues.
    """

    def __init__(
        self,
        config: BackendConfig,
        mapper: NeuralMapper,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            config: Backend configuration
            mapper: Reference to neural mapper
            device: Device for computation
        """
        self.config = config
        self.mapper = mapper
        self.device = device

        # Pose graph
        self.pose_graph = PoseGraph(device)

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._keyframe_queue: queue.Queue = queue.Queue()

        # Callbacks
        self._loop_closure_callback: Optional[Callable[[LoopClosureResult], None]] = None
        self._optimization_callback: Optional[Callable[[List[float]], None]] = None

        # Statistics
        self._keyframes_processed = 0
        self._loop_closures_detected = 0

    def start(self):
        """Start background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def add_keyframe(self, keyframe: Keyframe):
        """Add keyframe to processing queue."""
        self._keyframe_queue.put(keyframe)

    def _run_loop(self):
        """Main backend processing loop."""
        keyframes_since_ba = 0

        while self._running:
            try:
                # Get new keyframe (with timeout for checking _running)
                kf = self._keyframe_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Add to pose graph
            self.pose_graph.add_node(
                kf.frame_id,
                kf.translation,
                kf.quaternion
            )

            # Add edge to previous keyframe
            if self.pose_graph.num_nodes() > 1:
                node_ids = sorted(self.pose_graph.nodes.keys())
                if len(node_ids) >= 2:
                    prev_id = node_ids[-2]
                    self.pose_graph.add_odometry_edge(prev_id, kf.frame_id)

            keyframes_since_ba += 1
            self._keyframes_processed += 1

            # Loop closure detection
            if self.config.enable_loop_closure:
                loop_result = self._detect_loop_closure(kf)
                if loop_result.detected:
                    self._add_loop_closure_constraint(loop_result)
                    self._loop_closures_detected += 1

                    # Notify callback
                    if self._loop_closure_callback is not None:
                        self._loop_closure_callback(loop_result)

            # Run bundle adjustment periodically
            if keyframes_since_ba >= self.config.ba_frequency:
                self._run_local_ba()
                keyframes_since_ba = 0

    def _detect_loop_closure(self, keyframe: Keyframe) -> LoopClosureResult:
        """
        Detect loop closure candidates.

        Uses spatial proximity as primary criterion.
        """
        candidates = self.pose_graph.detect_loop_candidates(
            keyframe.frame_id,
            distance_threshold=self.config.loop_detection_threshold,
            min_interval=self.config.min_loop_interval
        )

        if not candidates:
            return LoopClosureResult(
                detected=False,
                source_keyframe_id=keyframe.frame_id,
                target_keyframe_id=-1
            )

        # Find closest candidate
        min_dist = float('inf')
        best_candidate = -1

        for cand_id in candidates:
            cand_t, _ = self.pose_graph.nodes[cand_id]
            dist = (keyframe.translation - cand_t).norm().item()
            if dist < min_dist:
                min_dist = dist
                best_candidate = cand_id

        if best_candidate < 0:
            return LoopClosureResult(
                detected=False,
                source_keyframe_id=keyframe.frame_id,
                target_keyframe_id=-1
            )

        # Compute relative transform
        loop_t, loop_q = self.pose_graph.nodes[best_candidate]

        # Relative transform from loop candidate to current
        from ..utils.quaternion import quaternion_to_matrix
        R_loop = quaternion_to_matrix(loop_q.unsqueeze(0)).squeeze(0)
        R_curr = quaternion_to_matrix(keyframe.quaternion.unsqueeze(0)).squeeze(0)

        R_rel = R_loop.T @ R_curr
        t_rel = R_loop.T @ (keyframe.translation - loop_t)

        from ..utils.quaternion import matrix_to_quaternion
        q_rel = matrix_to_quaternion(R_rel.unsqueeze(0)).squeeze(0)

        return LoopClosureResult(
            detected=True,
            source_keyframe_id=best_candidate,
            target_keyframe_id=keyframe.frame_id,
            relative_translation=t_rel,
            relative_quaternion=q_rel,
            confidence=1.0 / (min_dist + 0.01)
        )

    def _add_loop_closure_constraint(self, loop_result: LoopClosureResult):
        """Add loop closure constraint to pose graph."""
        self.pose_graph.add_edge(
            loop_result.source_keyframe_id,
            loop_result.target_keyframe_id,
            loop_result.relative_translation,
            loop_result.relative_quaternion,
            is_loop_closure=True
        )

        # Run pose graph optimization after loop closure
        loss_history = self.pose_graph.optimize(
            num_iterations=50,
            learning_rate=0.005
        )

        if self._optimization_callback is not None:
            self._optimization_callback(loss_history)

    def _run_local_ba(self):
        """Run local bundle adjustment on recent keyframes."""
        # Get recent node IDs
        node_ids = sorted(self.pose_graph.nodes.keys())

        if len(node_ids) < 2:
            return

        # Select window of recent keyframes
        window_ids = node_ids[-self.config.ba_window_size:]

        # Run pose graph optimization on window
        # (For full BA, would also optimize map - but that's expensive)
        loss_history = self.pose_graph.optimize(
            num_iterations=self.config.ba_iterations,
            learning_rate=self.config.ba_learning_rate_pose
        )

        if self._optimization_callback is not None:
            self._optimization_callback(loss_history)

    def run_global_optimization(self) -> List[float]:
        """
        Run global pose graph optimization.

        Should be called after all frames processed.
        """
        loss_history = self.pose_graph.optimize(
            num_iterations=200,
            learning_rate=0.001
        )
        return loss_history

    def set_loop_closure_callback(
        self,
        callback: Callable[[LoopClosureResult], None]
    ):
        """Set callback for loop closure detections."""
        self._loop_closure_callback = callback

    def set_optimization_callback(
        self,
        callback: Callable[[List[float]], None]
    ):
        """Set callback for optimization completion."""
        self._optimization_callback = callback

    def get_optimized_poses(self):
        """Get optimized poses from pose graph."""
        return self.pose_graph.get_poses()

    def get_trajectory(self):
        """Get optimized trajectory."""
        return self.pose_graph.get_trajectory()

    def get_statistics(self):
        """Get backend statistics."""
        return {
            'keyframes_processed': self._keyframes_processed,
            'loop_closures_detected': self._loop_closures_detected,
            'pose_graph_nodes': self.pose_graph.num_nodes(),
            'pose_graph_edges': self.pose_graph.num_edges(),
            'loop_closure_edges': self.pose_graph.num_loop_closures()
        }


class SLAMSystem:
    """
    Complete SLAM system combining frontend and backend.

    Provides a unified interface for running SLAM with background optimization.
    """

    def __init__(
        self,
        config,  # SLAMConfig
        device: Optional[torch.device] = None
    ):
        """
        Args:
            config: Complete SLAM configuration
            device: Device for computation
        """
        from .frontend import SLAMFrontend

        self.config = config
        self.device = device or torch.device(config.device)

        # Create frontend
        self.frontend = SLAMFrontend(config, self.device)

        # Create backend
        self.backend = SLAMBackend(
            config.backend,
            self.frontend.mapper,
            self.device
        )

        # Connect frontend to backend
        self.frontend.set_keyframe_callback(self.backend.add_keyframe)

    def start_backend(self):
        """Start background optimization thread."""
        self.backend.start()

    def stop_backend(self):
        """Stop background optimization thread."""
        self.backend.stop()

    def process_frame(self, frame):
        """Process a single frame."""
        return self.frontend.process_frame(frame)

    def run(self, frame_generator, max_frames=None, start_backend=True):
        """
        Run SLAM on a sequence.

        Args:
            frame_generator: Generator yielding frames
            max_frames: Maximum frames to process
            start_backend: Whether to start backend thread

        Returns:
            Final SLAM state
        """
        if start_backend:
            self.start_backend()

        try:
            state = self.frontend.run(frame_generator, max_frames)
        finally:
            if start_backend:
                self.stop_backend()

        return state

    def get_state(self):
        """Get current SLAM state."""
        return self.frontend.get_state()

    def finalize(self):
        """
        Finalize SLAM after processing.

        Runs global optimization and updates poses.
        """
        self.stop_backend()

        # Run final global optimization
        loss_history = self.backend.run_global_optimization()

        return loss_history

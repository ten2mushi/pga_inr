"""
Real-time SLAM frontend.

Orchestrates tracking and keyframe selection for live operation.
"""

from typing import Optional, Generator, Callable, List
import torch
import time

from .config import SLAMConfig
from .types import Frame, Keyframe, TrackingResult, SLAMState
from .tracker import PoseOptimizer, MotionModel
from .mapper import NeuralMapper
from .keyframe import KeyframeManager


class SLAMFrontend:
    """
    Real-time SLAM frontend.

    Orchestrates tracking and keyframe selection for live operation.
    This is the main entry point for running SLAM on a sequence.

    Pipeline:
    1. Receive new frame
    2. Predict pose using motion model
    3. Optimize pose against neural map
    4. Check keyframe criteria
    5. If keyframe: add to manager, update map
    6. Return tracking result
    """

    def __init__(
        self,
        config: SLAMConfig,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            config: Complete SLAM configuration
            device: Device for computation (default from config)
        """
        self.config = config
        self.device = device or torch.device(config.device)

        # Initialize components
        self.tracker = PoseOptimizer(
            config.tracker, config.camera, self.device
        )
        self.mapper = NeuralMapper(config.mapper, self.device)
        self.keyframe_manager = KeyframeManager(
            config.keyframe, config.camera, self.device
        )
        self.motion_model = MotionModel(self.device)

        # State
        self.current_translation = torch.zeros(3, device=self.device)
        self.current_quaternion = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )
        self.is_initialized = False
        self.is_lost = False
        self.frame_count = 0
        self.tracking_times: List[float] = []

        # Callbacks
        self._keyframe_callback: Optional[Callable[[Keyframe], None]] = None
        self._tracking_callback: Optional[Callable[[TrackingResult], None]] = None

    def process_frame(self, frame: Frame) -> TrackingResult:
        """
        Process a single RGB-D frame.

        Args:
            frame: Input RGB-D frame

        Returns:
            TrackingResult with estimated pose
        """
        start_time = time.time()

        # Move frame data to device
        frame = Frame(
            rgb=frame.rgb.to(self.device),
            depth=frame.depth.to(self.device),
            timestamp=frame.timestamp,
            frame_id=frame.frame_id
        )

        if not self.is_initialized:
            # First frame: initialize map and pose
            result = self._initialize(frame)
        else:
            # Track against current map
            result = self._track_frame(frame)

        # Record timing
        elapsed = time.time() - start_time
        self.tracking_times.append(elapsed)
        self.frame_count += 1

        # Call tracking callback
        if self._tracking_callback is not None:
            self._tracking_callback(result)

        return result

    def _initialize(self, frame: Frame) -> TrackingResult:
        """
        Initialize SLAM from first frame.

        Creates first keyframe at identity pose and initializes neural map.
        """
        # Initialize at identity pose
        self.current_translation = torch.zeros(3, device=self.device)
        self.current_quaternion = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )

        # Add first keyframe
        kf = self.keyframe_manager.add_keyframe(
            frame,
            self.current_translation.clone(),
            self.current_quaternion.clone(),
            tracking_error=0.0
        )

        # Initialize map from first keyframe (use more iterations)
        if self.config.verbose:
            print(f"Initializing map from first frame...")

        init_iterations = self.config.mapper.num_iterations * 2
        self.mapper.update_from_keyframe(kf, num_iterations=init_iterations)

        if self.config.verbose:
            print(f"Map initialized.")

        # Initialize motion model
        self.motion_model.update(
            self.current_translation,
            self.current_quaternion
        )

        self.is_initialized = True

        # Call keyframe callback
        if self._keyframe_callback is not None:
            self._keyframe_callback(kf)

        return TrackingResult(
            translation=self.current_translation.clone(),
            quaternion=self.current_quaternion.clone(),
            converged=True,
            num_iterations=0,
            final_loss=0.0,
            inlier_ratio=1.0,
            loss_breakdown={}
        )

    def _track_frame(self, frame: Frame) -> TrackingResult:
        """
        Track camera pose for a new frame.

        Args:
            frame: New RGB-D frame

        Returns:
            TrackingResult
        """
        # Predict initial pose from motion model
        pred_t, pred_q = self.motion_model.predict()

        # Optimize pose against map
        result = self.tracker.optimize(
            map_model=self.mapper.get_model(),
            frame=frame,
            initial_translation=pred_t,
            initial_quaternion=pred_q
        )

        # Check if tracking is lost
        if result.final_loss > self.config.keyframe.max_tracking_error * 2:
            self.is_lost = True
            if self.config.verbose:
                print(f"Warning: Tracking may be lost (loss={result.final_loss:.4f})")

        # Update current pose
        self.current_translation = result.translation.clone()
        self.current_quaternion = result.quaternion.clone()

        # Update motion model
        self.motion_model.update(result.translation, result.quaternion)

        # Check if we should add a keyframe
        should_add = self.keyframe_manager.should_add_keyframe(
            result.translation,
            result.quaternion,
            result.final_loss,
            frame.frame_id
        )

        if should_add:
            # Add keyframe
            kf = self.keyframe_manager.add_keyframe(
                frame,
                result.translation.clone(),
                result.quaternion.clone(),
                result.final_loss
            )

            # Update map with new keyframe
            if self.config.verbose:
                print(f"Frame {frame.frame_id}: Adding keyframe #{len(self.keyframe_manager.keyframes)}")

            self.mapper.update_from_keyframe(kf)

            # Call keyframe callback
            if self._keyframe_callback is not None:
                self._keyframe_callback(kf)

        return result

    def run(
        self,
        frame_generator: Generator[Frame, None, None],
        max_frames: Optional[int] = None
    ) -> SLAMState:
        """
        Run SLAM on a sequence of frames.

        Args:
            frame_generator: Generator yielding Frame objects
            max_frames: Optional maximum frames to process

        Returns:
            Final SLAM state
        """
        for i, frame in enumerate(frame_generator):
            if max_frames is not None and i >= max_frames:
                break

            result = self.process_frame(frame)

            if self.config.verbose and i % 10 == 0:
                fps = self._get_current_fps()
                print(
                    f"Frame {i}: loss={result.final_loss:.4f}, "
                    f"keyframes={len(self.keyframe_manager.keyframes)}, "
                    f"fps={fps:.1f}"
                )

        return self.get_state()

    def _get_current_fps(self) -> float:
        """Compute current tracking FPS."""
        if len(self.tracking_times) == 0:
            return 0.0
        recent = self.tracking_times[-10:]
        return len(recent) / (sum(recent) + 1e-8)

    def get_state(self) -> SLAMState:
        """Get current SLAM state."""
        avg_fps = self._get_current_fps()

        return SLAMState(
            current_translation=self.current_translation.clone(),
            current_quaternion=self.current_quaternion.clone(),
            map_model=self.mapper.get_model(),
            keyframes=self.keyframe_manager.keyframes.copy(),
            total_frames=self.frame_count,
            total_keyframes=len(self.keyframe_manager.keyframes),
            tracking_fps=avg_fps,
            is_initialized=self.is_initialized,
            is_lost=self.is_lost
        )

    def set_keyframe_callback(self, callback: Callable[[Keyframe], None]):
        """Set callback for new keyframes (for backend thread)."""
        self._keyframe_callback = callback

    def set_tracking_callback(self, callback: Callable[[TrackingResult], None]):
        """Set callback for tracking results."""
        self._tracking_callback = callback

    def save_checkpoint(self, path: str):
        """
        Save SLAM checkpoint.

        Saves:
        - Neural map model
        - Keyframe poses
        - Current state
        """
        keyframe_data = []
        for kf in self.keyframe_manager.keyframes:
            keyframe_data.append({
                'frame_id': kf.frame_id,
                'timestamp': kf.timestamp,
                'translation': kf.translation.cpu(),
                'quaternion': kf.quaternion.cpu(),
                'tracking_error': kf.tracking_error
            })

        torch.save({
            'model_state_dict': self.mapper.get_model().state_dict(),
            'keyframes': keyframe_data,
            'current_translation': self.current_translation.cpu(),
            'current_quaternion': self.current_quaternion.cpu(),
            'frame_count': self.frame_count,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load SLAM checkpoint.

        Note: Only loads map and poses, not images.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.mapper.get_model().load_state_dict(checkpoint['model_state_dict'])
        self.current_translation = checkpoint['current_translation'].to(self.device)
        self.current_quaternion = checkpoint['current_quaternion'].to(self.device)
        self.frame_count = checkpoint.get('frame_count', 0)
        self.is_initialized = True

    def reset(self):
        """Reset SLAM to initial state."""
        self.mapper.reset()
        self.keyframe_manager.clear()
        self.motion_model.reset()

        self.current_translation = torch.zeros(3, device=self.device)
        self.current_quaternion = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )
        self.is_initialized = False
        self.is_lost = False
        self.frame_count = 0
        self.tracking_times.clear()

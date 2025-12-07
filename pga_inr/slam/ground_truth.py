"""
Ground truth trajectory generation for FMDataset.

Since the FMDataset does not include ground truth poses, we can generate
reference trajectories using:
1. IMU integration (gyroscope for orientation)
2. Depth-based ICP registration (for position)
3. Hybrid approaches combining both

These serve as pseudo ground truth for qualitative evaluation.
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import torch

from .types import Frame
from .config import CameraIntrinsics


class IMUIntegrator:
    """
    Integrate IMU measurements to estimate camera trajectory.

    Uses gyroscope for orientation (reliable short-term) and
    accelerometer for position (drifts rapidly - use with caution).
    """

    def __init__(
        self,
        initial_position: np.ndarray = None,
        initial_orientation: np.ndarray = None,
        gravity: float = 9.81,
        bias_estimation_frames: int = 50
    ):
        """
        Args:
            initial_position: Initial position [x, y, z]
            initial_orientation: Initial orientation as quaternion [w, x, y, z]
            gravity: Gravity magnitude (m/s²)
            bias_estimation_frames: Frames to use for bias estimation
        """
        self.position = initial_position if initial_position is not None else np.zeros(3)
        self.velocity = np.zeros(3)

        if initial_orientation is not None:
            self.orientation = initial_orientation / np.linalg.norm(initial_orientation)
        else:
            self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        self.gravity = gravity
        self.gravity_vector = np.array([0, -gravity, 0])  # Assuming Y-up

        # Bias estimation
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.bias_samples = []
        self.bias_estimation_frames = bias_estimation_frames
        self.bias_estimated = False

        # History
        self.trajectory: List[Tuple[float, np.ndarray, np.ndarray]] = []
        self.prev_timestamp: Optional[float] = None

    def update(
        self,
        timestamp: float,
        gyro: np.ndarray,
        accel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state with new IMU measurement.

        Args:
            timestamp: Time in seconds
            gyro: Angular velocity [rx, ry, rz] in rad/s
            accel: Linear acceleration [ax, ay, az] in m/s²

        Returns:
            (position, quaternion)
        """
        # Bias estimation phase
        if not self.bias_estimated and len(self.bias_samples) < self.bias_estimation_frames:
            self.bias_samples.append((gyro.copy(), accel.copy()))
            if len(self.bias_samples) >= self.bias_estimation_frames:
                self._estimate_bias()
            return self.position.copy(), self.orientation.copy()

        # Remove bias
        gyro_corrected = gyro - self.gyro_bias
        accel_corrected = accel - self.accel_bias

        # Compute dt
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return self.position.copy(), self.orientation.copy()

        dt = timestamp - self.prev_timestamp
        if dt <= 0 or dt > 0.5:  # Skip invalid dt
            self.prev_timestamp = timestamp
            return self.position.copy(), self.orientation.copy()

        self.prev_timestamp = timestamp

        # Integrate orientation (quaternion integration)
        omega_norm = np.linalg.norm(gyro_corrected)
        if omega_norm > 1e-8:
            # Rodrigues formula for quaternion update
            axis = gyro_corrected / omega_norm
            angle = omega_norm * dt
            dq = self._axis_angle_to_quaternion(axis, angle)
            self.orientation = self._quaternion_multiply(self.orientation, dq)
            self.orientation = self.orientation / np.linalg.norm(self.orientation)

        # Transform acceleration to world frame and remove gravity
        R = self._quaternion_to_matrix(self.orientation)
        accel_world = R @ accel_corrected
        accel_world = accel_world - self.gravity_vector

        # Integrate velocity and position
        self.velocity = self.velocity + accel_world * dt
        self.position = self.position + self.velocity * dt

        # Store trajectory point
        self.trajectory.append((
            timestamp,
            self.position.copy(),
            self.orientation.copy()
        ))

        return self.position.copy(), self.orientation.copy()

    def _estimate_bias(self):
        """Estimate sensor biases from static period."""
        gyro_samples = np.array([s[0] for s in self.bias_samples])
        accel_samples = np.array([s[1] for s in self.bias_samples])

        # Gyroscope bias: mean during static period
        self.gyro_bias = np.mean(gyro_samples, axis=0)

        # Accelerometer bias: assume static and vertical
        mean_accel = np.mean(accel_samples, axis=0)
        accel_norm = np.linalg.norm(mean_accel)

        # Assuming the sensor is roughly level, gravity points down (-Y or -Z)
        # Find the axis most aligned with gravity
        abs_accel = np.abs(mean_accel)
        gravity_axis = np.argmax(abs_accel)
        gravity_sign = np.sign(mean_accel[gravity_axis])

        expected_gravity = np.zeros(3)
        expected_gravity[gravity_axis] = gravity_sign * self.gravity

        self.accel_bias = mean_accel - expected_gravity
        self.gravity_vector = expected_gravity

        self.bias_estimated = True

    def _axis_angle_to_quaternion(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to quaternion [w, x, y, z]."""
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return np.array([w, xyz[0], xyz[1], xyz[2]])

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def get_trajectory(self) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """Get full trajectory as list of (timestamp, position, quaternion)."""
        return self.trajectory

    def get_positions(self) -> np.ndarray:
        """Get positions as (N, 3) array."""
        if not self.trajectory:
            return np.zeros((0, 3))
        return np.array([t[1] for t in self.trajectory])

    def get_orientations(self) -> np.ndarray:
        """Get orientations as (N, 4) quaternion array."""
        if not self.trajectory:
            return np.zeros((0, 4))
        return np.array([t[2] for t in self.trajectory])


class DepthICPTracker:
    """
    Estimate camera trajectory using depth-based ICP registration.

    Uses point cloud registration between consecutive depth frames
    to estimate relative poses, then chains them for global trajectory.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        max_correspondence_distance: float = 0.1,
        max_iterations: int = 50,
        downsample_voxel_size: float = 0.02,
        min_points: int = 100
    ):
        """
        Args:
            intrinsics: Camera intrinsics
            max_correspondence_distance: ICP correspondence threshold (meters)
            max_iterations: Maximum ICP iterations
            downsample_voxel_size: Voxel size for point cloud downsampling
            min_points: Minimum points required for ICP
        """
        self.intrinsics = intrinsics
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations
        self.downsample_voxel_size = downsample_voxel_size
        self.min_points = min_points

        # State
        self.prev_points: Optional[np.ndarray] = None
        self.global_transform = np.eye(4)
        self.trajectory: List[Tuple[int, np.ndarray]] = []

    def process_frame(
        self,
        frame_id: int,
        depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process a depth frame and update trajectory.

        Args:
            frame_id: Frame identifier
            depth: Depth image in meters (H, W)
            valid_mask: Optional mask for valid depth pixels

        Returns:
            Current global position (3,)
        """
        # Convert depth to point cloud
        points = self._depth_to_pointcloud(depth, valid_mask)

        if len(points) < self.min_points:
            # Not enough points, keep previous pose
            position = self.global_transform[:3, 3]
            self.trajectory.append((frame_id, position.copy()))
            return position

        # Downsample
        points = self._voxel_downsample(points, self.downsample_voxel_size)

        if self.prev_points is None:
            # First frame
            self.prev_points = points
            position = np.zeros(3)
            self.trajectory.append((frame_id, position))
            return position

        # Estimate relative transform using ICP
        relative_transform = self._icp(self.prev_points, points)

        # Update global transform
        self.global_transform = self.global_transform @ relative_transform

        # Update previous points
        self.prev_points = points

        position = self.global_transform[:3, 3]
        self.trajectory.append((frame_id, position.copy()))

        return position

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert depth image to 3D point cloud."""
        H, W = depth.shape
        K = self.intrinsics

        # Create pixel grid
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        # Valid depth mask
        if valid_mask is None:
            valid_mask = (depth > 0.1) & (depth < 10.0)

        # Get valid pixels
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]

        # Backproject to 3D
        x = (u_valid - K.cx) * z_valid / K.fx
        y = (v_valid - K.cy) * z_valid / K.fy

        points = np.stack([x, y, z_valid], axis=-1)
        return points

    def _voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Downsample point cloud using voxel grid."""
        if voxel_size <= 0:
            return points

        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # Get unique voxels and average points within each
        unique_indices, inverse = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        # Average points in each voxel
        downsampled = np.zeros((len(unique_indices), 3))
        counts = np.zeros(len(unique_indices))

        np.add.at(downsampled, inverse, points)
        np.add.at(counts, inverse, 1)

        downsampled = downsampled / counts[:, np.newaxis]
        return downsampled

    def _icp(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Simple point-to-point ICP implementation.

        For production use, consider using Open3D or scipy's implementations.
        """
        # Initialize transform
        transform = np.eye(4)

        source_h = np.hstack([source, np.ones((len(source), 1))])

        for iteration in range(self.max_iterations):
            # Transform source points
            source_transformed = (transform @ source_h.T).T[:, :3]

            # Find nearest neighbors (simple brute force for small clouds)
            correspondences = self._find_correspondences(
                source_transformed, target
            )

            if len(correspondences) < 3:
                break

            src_pts = source_transformed[correspondences[:, 0]]
            tgt_pts = target[correspondences[:, 1]]

            # Compute transform update using SVD
            delta_transform = self._compute_rigid_transform(src_pts, tgt_pts)

            # Update total transform
            transform = delta_transform @ transform

            # Check convergence
            delta_translation = np.linalg.norm(delta_transform[:3, 3])
            if delta_translation < 1e-6:
                break

        return transform

    def _find_correspondences(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Find nearest neighbor correspondences."""
        correspondences = []

        # Subsample for efficiency
        step = max(1, len(source) // 500)
        source_subset = source[::step]

        for i, src_pt in enumerate(source_subset):
            # Find nearest target point
            distances = np.linalg.norm(target - src_pt, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist < self.max_correspondence_distance:
                correspondences.append([i * step, min_idx])

        return np.array(correspondences) if correspondences else np.zeros((0, 2), dtype=int)

    def _compute_rigid_transform(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Compute rigid transform between point sets using SVD."""
        # Compute centroids
        centroid_src = np.mean(source, axis=0)
        centroid_tgt = np.mean(target, axis=0)

        # Center points
        src_centered = source - centroid_src
        tgt_centered = target - centroid_tgt

        # Compute cross-covariance matrix
        H = src_centered.T @ tgt_centered

        # SVD
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_tgt - R @ centroid_src

        # Build 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        return transform

    def get_trajectory(self) -> List[Tuple[int, np.ndarray]]:
        """Get trajectory as list of (frame_id, position)."""
        return self.trajectory

    def get_positions(self) -> np.ndarray:
        """Get positions as (N, 3) array."""
        if not self.trajectory:
            return np.zeros((0, 3))
        return np.array([t[1] for t in self.trajectory])


def load_imu_data(imu_path: str) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Load IMU data from FMDataset IMU.txt file.

    Returns:
        List of (timestamp_seconds, gyro, accel) tuples
    """
    data = []

    with open(imu_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) != 7:
                continue

            timestamp_us = float(parts[0])
            timestamp_s = timestamp_us / 1e6

            # Gyroscope: rx, ry, rz (rad/s)
            gyro = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

            # Accelerometer: ax, ay, az (m/s²)
            accel = np.array([float(parts[4]), float(parts[5]), float(parts[6])])

            data.append((timestamp_s, gyro, accel))

    return data


def generate_imu_trajectory(
    imu_path: str,
    frame_timestamps: List[float]
) -> List[Tuple[int, torch.Tensor]]:
    """
    Generate trajectory from IMU data synchronized to frame timestamps.

    Args:
        imu_path: Path to IMU.txt
        frame_timestamps: List of frame timestamps in seconds

    Returns:
        List of (frame_id, position) tuples with torch tensors
    """
    # Load IMU data
    imu_data = load_imu_data(imu_path)

    if not imu_data:
        return []

    # Create integrator
    integrator = IMUIntegrator()

    # Process all IMU samples
    for timestamp, gyro, accel in imu_data:
        integrator.update(timestamp, gyro, accel)

    # Get positions at frame timestamps
    trajectory = []
    imu_trajectory = integrator.get_trajectory()

    if not imu_trajectory:
        return []

    imu_times = np.array([t[0] for t in imu_trajectory])
    imu_positions = np.array([t[1] for t in imu_trajectory])

    for frame_id, frame_ts in enumerate(frame_timestamps):
        # Find nearest IMU sample
        idx = np.argmin(np.abs(imu_times - frame_ts))
        position = imu_positions[idx]
        trajectory.append((frame_id, torch.tensor(position, dtype=torch.float32)))

    return trajectory


def generate_icp_trajectory(
    frames: List[Frame],
    intrinsics: CameraIntrinsics,
    skip_frames: int = 1
) -> List[Tuple[int, torch.Tensor]]:
    """
    Generate trajectory using depth-based ICP.

    Args:
        frames: List of Frame objects
        intrinsics: Camera intrinsics
        skip_frames: Process every N frames (for speed)

    Returns:
        List of (frame_id, position) tuples
    """
    tracker = DepthICPTracker(intrinsics)

    trajectory = []
    for i, frame in enumerate(frames):
        if i % skip_frames != 0:
            continue

        depth = frame.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()

        position = tracker.process_frame(frame.frame_id, depth)
        trajectory.append((frame.frame_id, torch.tensor(position, dtype=torch.float32)))

    return trajectory

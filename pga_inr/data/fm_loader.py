"""
FMDataset (Fast-Motion RGB-D) loader.

Loads RGB-D sequences from the FMDataset for SLAM evaluation.
Supports both extracted directories and nested zip files.
"""

import os
import zipfile
from pathlib import Path
from typing import Optional, Iterator, List, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import io

from ..slam.types import Frame
from ..slam.config import CameraIntrinsics


class FMDatasetLoader(Dataset):
    """
    Loader for FMDataset sequences.

    FMDataset structure (after extraction):
    sequence_name/
        color/*.png          # RGB images
        filtered/*.png       # Depth images (16-bit, scale=1000)
        TIMESTAMP.txt        # Frame timestamps
        IMU.txt             # Optional IMU data

    Camera intrinsics (fixed for all sequences):
        Color: fx=fy=608, cx=331, cy=246
        Depth: fx=fy=583, cx=325, cy=240
        Depth scale: 1000 (divide by 1000 for meters)
    """

    # Default intrinsics for FMDataset
    COLOR_INTRINSICS = CameraIntrinsics(
        fx=608.0, fy=608.0, cx=331.0, cy=246.0,
        width=640, height=480, depth_scale=1000.0
    )

    DEPTH_INTRINSICS = CameraIntrinsics(
        fx=583.0, fy=583.0, cx=325.0, cy=240.0,
        width=640, height=480, depth_scale=1000.0
    )

    def __init__(
        self,
        sequence_path: Union[str, Path],
        use_color_intrinsics: bool = True,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        preload: bool = False,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize FMDataset loader.

        Args:
            sequence_path: Path to extracted sequence directory or .zip file
            use_color_intrinsics: Use color camera intrinsics (True) or depth (False)
            max_depth: Maximum valid depth in meters
            min_depth: Minimum valid depth in meters
            preload: Whether to preload all frames into memory
            device: Device to load tensors to
        """
        self.sequence_path = Path(sequence_path)
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.device = device
        self.intrinsics = self.COLOR_INTRINSICS if use_color_intrinsics else self.DEPTH_INTRINSICS

        # Determine if we're loading from zip or directory
        self._is_zip = self.sequence_path.suffix == '.zip'
        self._zip_file: Optional[zipfile.ZipFile] = None

        # Parse structure
        self._parse_structure()

        # Load timestamps
        self._load_timestamps()

        # Optional preloading
        self._preloaded_frames: Optional[List[Frame]] = None
        if preload:
            self._preload_all()

    def _parse_structure(self):
        """Parse directory/zip structure to find color and depth files."""
        if self._is_zip:
            self._zip_file = zipfile.ZipFile(self.sequence_path, 'r')
            all_files = self._zip_file.namelist()

            # Find the base directory in the zip
            color_files = [f for f in all_files if '/color/' in f and f.endswith('.png')]
            if not color_files:
                color_files = [f for f in all_files if 'color/' in f and f.endswith('.png')]

            depth_files = [f for f in all_files if '/filtered/' in f and f.endswith('.png')]
            if not depth_files:
                depth_files = [f for f in all_files if 'filtered/' in f and f.endswith('.png')]

            self.color_files = sorted(color_files)
            self.depth_files = sorted(depth_files)
        else:
            # Direct directory loading
            color_dir = self.sequence_path / "color"
            depth_dir = self.sequence_path / "filtered"

            if not color_dir.exists():
                raise ValueError(f"Color directory not found: {color_dir}")
            if not depth_dir.exists():
                raise ValueError(f"Depth directory not found: {depth_dir}")

            self.color_files = sorted(list(color_dir.glob("*.png")))
            self.depth_files = sorted(list(depth_dir.glob("*.png")))

        # Validate
        if len(self.color_files) == 0:
            raise ValueError(f"No color images found in {self.sequence_path}")
        if len(self.depth_files) == 0:
            raise ValueError(f"No depth images found in {self.sequence_path}")

        # Match color and depth files by filename (timestamp)
        self._match_color_depth_files()

        self.num_frames = len(self.matched_pairs)

    def _match_color_depth_files(self):
        """Match color and depth files by timestamp/filename."""
        # Extract timestamps from filenames
        def get_timestamp(path):
            if isinstance(path, Path):
                return path.stem
            else:
                # String path from zip
                return Path(path).stem

        color_timestamps = {get_timestamp(f): f for f in self.color_files}
        depth_timestamps = {get_timestamp(f): f for f in self.depth_files}

        # Find matching pairs
        common_timestamps = sorted(set(color_timestamps.keys()) & set(depth_timestamps.keys()))

        if len(common_timestamps) == 0:
            raise ValueError("No matching color/depth pairs found")

        self.matched_pairs = [
            (color_timestamps[ts], depth_timestamps[ts])
            for ts in common_timestamps
        ]

    def _load_timestamps(self):
        """Load timestamps from TIMESTAMP.txt."""
        self.timestamps = []

        timestamp_file = None
        if self._is_zip:
            all_files = self._zip_file.namelist()
            timestamp_files = [f for f in all_files if 'TIMESTAMP' in f.upper()]
            if timestamp_files:
                timestamp_file = timestamp_files[0]
        else:
            ts_path = self.sequence_path / "TIMESTAMP.txt"
            if ts_path.exists():
                timestamp_file = ts_path

        if timestamp_file is not None:
            try:
                if self._is_zip:
                    with self._zip_file.open(timestamp_file) as f:
                        content = f.read().decode('utf-8')
                else:
                    with open(timestamp_file, 'r') as f:
                        content = f.read()

                for line in content.strip().split('\n'):
                    try:
                        ts = float(line.strip())
                        self.timestamps.append(ts)
                    except ValueError:
                        continue
            except Exception:
                pass

        # Fallback: generate sequential timestamps if file missing or parsing failed
        if len(self.timestamps) != self.num_frames:
            self.timestamps = [float(i) / 30.0 for i in range(self.num_frames)]  # Assume 30 FPS

    def _preload_all(self):
        """Preload all frames into memory."""
        print(f"Preloading {self.num_frames} frames...")
        self._preloaded_frames = []
        for i in range(self.num_frames):
            frame = self._load_frame(i)
            self._preloaded_frames.append(frame)
        print("Preloading complete.")

    def _load_image(self, path) -> np.ndarray:
        """Load image from path or zip."""
        if self._is_zip:
            with self._zip_file.open(path) as f:
                img = Image.open(io.BytesIO(f.read()))
                return np.array(img)
        else:
            return np.array(Image.open(path))

    def _load_frame(self, index: int) -> Frame:
        """Load a single frame by index."""
        color_path, depth_path = self.matched_pairs[index]

        # Load color image
        color_np = self._load_image(color_path).astype(np.float32) / 255.0

        # Load depth image
        depth_np = self._load_image(depth_path).astype(np.float32)

        # Convert depth to meters
        depth_np = depth_np / self.intrinsics.depth_scale

        # Invalidate depths outside valid range
        depth_np[(depth_np < self.min_depth) | (depth_np > self.max_depth)] = 0

        # Convert to tensors
        rgb = torch.from_numpy(color_np).to(self.device)
        depth = torch.from_numpy(depth_np).to(self.device)

        return Frame(
            rgb=rgb,
            depth=depth,
            timestamp=self.timestamps[index],
            frame_id=index
        )

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> Frame:
        if index < 0:
            index = self.num_frames + index
        if index >= self.num_frames or index < 0:
            raise IndexError(f"Frame index {index} out of range [0, {self.num_frames})")

        if self._preloaded_frames is not None:
            return self._preloaded_frames[index]
        return self._load_frame(index)

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.num_frames):
            yield self[i]

    def get_intrinsics(self) -> CameraIntrinsics:
        """Get camera intrinsics."""
        return self.intrinsics

    def get_sequence_name(self) -> str:
        """Get sequence name."""
        return self.sequence_path.stem

    def close(self):
        """Close zip file if open."""
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class FMDatasetSequenceIterator:
    """
    Iterator over all sequences in an FMDataset directory.

    Useful for batch processing multiple sequences.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        use_color_intrinsics: bool = True,
        max_depth: float = 10.0,
        preload: bool = False
    ):
        """
        Args:
            dataset_dir: Directory containing extracted sequences
            use_color_intrinsics: Use color camera intrinsics
            max_depth: Maximum valid depth
            preload: Preload frames for each sequence
        """
        self.dataset_dir = Path(dataset_dir)
        self.use_color_intrinsics = use_color_intrinsics
        self.max_depth = max_depth
        self.preload = preload

        # Find all sequences
        self.sequences = self._find_sequences()

    def _find_sequences(self) -> List[Path]:
        """Find all valid sequence directories."""
        sequences = []

        for item in sorted(self.dataset_dir.iterdir()):
            if item.is_dir():
                # Check for required structure
                color_dir = item / "color"
                depth_dir = item / "filtered"
                if color_dir.exists() and depth_dir.exists():
                    sequences.append(item)
            elif item.suffix == '.zip' and 'FMDataset' not in item.name:
                # Individual sequence zip
                sequences.append(item)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterator[Tuple[str, FMDatasetLoader]]:
        for seq_path in self.sequences:
            loader = FMDatasetLoader(
                seq_path,
                use_color_intrinsics=self.use_color_intrinsics,
                max_depth=self.max_depth,
                preload=self.preload
            )
            yield seq_path.stem, loader
            loader.close()

    def get_sequence(self, name: str) -> Optional[FMDatasetLoader]:
        """Get loader for specific sequence by name."""
        for seq_path in self.sequences:
            if seq_path.stem == name:
                return FMDatasetLoader(
                    seq_path,
                    use_color_intrinsics=self.use_color_intrinsics,
                    max_depth=self.max_depth,
                    preload=self.preload
                )
        return None

    def list_sequences(self) -> List[str]:
        """List available sequence names."""
        return [seq.stem for seq in self.sequences]

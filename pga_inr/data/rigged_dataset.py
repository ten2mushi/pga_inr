"""
Dataset classes for rigged character training.

Provides PyTorch Dataset implementations for:
- Canonical mesh SDF training (T-pose)
- Articulated character SDF training at various poses
- Dynamic pose sequence training
"""

from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

from .fbx_loader import load_rigged_character, AnimationData
from .mesh_utils import compute_sdf


class CanonicalMeshDataset(Dataset):
    """
    Dataset for training SDF on canonical (T-pose) mesh.

    Samples points and computes SDF values for the rest pose mesh.
    This is used to train the base PGA_INR_SDF model before articulation.
    """

    def __init__(
        self,
        mesh,
        num_samples: int = 10000,
        surface_ratio: float = 0.5,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        surface_noise_std: float = 0.02,
        cache_size: int = 100000,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            mesh: trimesh.Trimesh object (from load_rigged_character)
            num_samples: Points per batch
            surface_ratio: Fraction of points near surface (vs volume)
            bounds: Sampling bounds for volume points
            surface_noise_std: Std dev for surface point perturbation
            cache_size: Number of samples to precompute
            device: Torch device
        """
        self.mesh = mesh
        self.num_samples = num_samples
        self.surface_ratio = surface_ratio
        self.bounds = bounds
        self.surface_noise_std = surface_noise_std
        self.device = device

        # Precompute surface points
        self.surface_points, self.surface_normals = self._sample_surface(cache_size)

        # Build cache
        self._build_cache(cache_size)

    def _sample_surface(self, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points on the mesh surface."""
        points, face_indices = self.mesh.sample(num_points, return_index=True)
        normals = self.mesh.face_normals[face_indices]
        return points.astype(np.float32), normals.astype(np.float32)

    def _build_cache(self, cache_size: int):
        """Precompute cached samples with SDF values."""
        print(f"    Building SDF cache ({cache_size} points)...")
        num_surface = int(cache_size * self.surface_ratio)
        num_volume = cache_size - num_surface

        # Surface-near samples
        indices = np.random.randint(0, len(self.surface_points), num_surface)
        surface_pts = self.surface_points[indices].copy()
        surface_normals = self.surface_normals[indices]
        # Add perturbation along normals
        surface_pts += surface_normals * np.random.randn(num_surface, 1).astype(np.float32) * self.surface_noise_std

        # Volume samples
        volume_pts = np.random.uniform(
            self.bounds[0], self.bounds[1],
            size=(num_volume, 3)
        ).astype(np.float32)

        # Combine
        self.cached_points = np.concatenate([surface_pts, volume_pts], axis=0)

        # Compute SDF (uses batched processing to limit memory)
        print("    Computing SDF values (this may take a moment)...")
        self.cached_sdf = compute_sdf(self.mesh, self.cached_points).astype(np.float32)
        print("    Cache built successfully.")

    def __len__(self) -> int:
        return len(self.cached_points) // self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random sample from cache
        indices = np.random.randint(0, len(self.cached_points), self.num_samples)
        points = self.cached_points[indices]
        sdf = self.cached_sdf[indices]

        return {
            'points': torch.from_numpy(points).float(),
            'sdf': torch.from_numpy(sdf).float().view(-1, 1)
        }


class RiggedMeshDataset(Dataset):
    """
    Dataset for training SDF on rigged characters at various poses.

    Samples points and computes SDF values for the deformed mesh.
    Uses animations from FBX to generate diverse poses.
    """

    def __init__(
        self,
        mesh_path: Union[str, Path],
        animation_paths: Optional[List[Union[str, Path]]] = None,
        num_samples: int = 10000,
        surface_ratio: float = 0.5,
        bounds: Tuple[float, float] = (-1.5, 1.5),
        num_poses: int = 20,
        pose_source: str = 'animation',
        surface_noise_std: float = 0.02,
        device: torch.device = torch.device('cpu'),
        cache_poses: bool = True
    ):
        """
        Args:
            mesh_path: Path to FBX file with mesh
            animation_paths: Paths to animation FBX files
            num_samples: Points per sample
            surface_ratio: Fraction near surface
            bounds: Sampling bounds
            num_poses: Number of poses to sample
            pose_source: 'animation' or 'random'
            surface_noise_std: Surface perturbation std
            device: Torch device
            cache_poses: Whether to precompute deformed meshes
        """
        self.num_samples = num_samples
        self.surface_ratio = surface_ratio
        self.bounds = bounds
        self.num_poses = num_poses
        self.surface_noise_std = surface_noise_std
        self.device = device

        # Load character
        self.character = load_rigged_character(
            mesh_path,
            animation_paths=animation_paths,
            device=device,
            normalize=True
        )

        # Generate or extract poses
        self.poses = self._generate_poses(pose_source)

        # Cache deformed meshes if requested
        self.cached_meshes = None
        if cache_poses:
            self._cache_deformed_meshes()

    def _generate_poses(
        self,
        pose_source: str
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate poses for dataset."""
        chain = self.character['kinematic_chain']
        if chain is None:
            # No skeleton, return identity poses
            return [(torch.eye(4), torch.zeros(3)) for _ in range(self.num_poses)]

        num_joints = len(chain.joints)

        if pose_source == 'animation' and self.character['animations']:
            # Sample from animations
            poses = []
            all_anims = list(self.character['animations'].values())

            for i in range(self.num_poses):
                # Cycle through animations
                anim = all_anims[i % len(all_anims)]
                # Sample a random frame
                frame_idx = np.random.randint(0, anim.keyframe_rotations.shape[0])
                rotations = anim.keyframe_rotations[frame_idx]
                root_trans = anim.root_translations[frame_idx]
                poses.append((rotations.to(self.device), root_trans.to(self.device)))

            return poses
        else:
            # Generate random poses (small perturbations from rest)
            from ..utils.quaternion import quaternion_from_axis_angle

            poses = []
            for _ in range(self.num_poses):
                # Random joint angles (small perturbations)
                angles = torch.randn(num_joints, device=self.device) * 0.3  # ~17 degrees std

                rotations = []
                for j, name in enumerate(chain.joint_order):
                    joint = chain.joints[name]
                    axis = joint.axis if joint.axis is not None else torch.tensor([0.0, 0.0, 1.0])
                    axis = axis.to(self.device)

                    quat = quaternion_from_axis_angle(
                        axis.unsqueeze(0),
                        angles[j:j+1].unsqueeze(0)
                    ).squeeze()
                    rotations.append(quat)

                rotations = torch.stack(rotations)
                root_trans = torch.zeros(3, device=self.device)
                poses.append((rotations, root_trans))

            return poses

    def _cache_deformed_meshes(self):
        """Precompute deformed meshes for all poses."""
        import trimesh

        self.cached_meshes = []
        lbs = self.character['lbs']
        chain = self.character['kinematic_chain']

        if chain is None or lbs is None:
            # No skeleton, just use rest mesh
            for _ in self.poses:
                self.cached_meshes.append(self.character['mesh'])
            return

        for rotations, root_trans in self.poses:
            transforms = chain.forward_kinematics(rotations, root_trans)
            deformed_verts = lbs(transforms).detach().cpu().numpy()

            mesh = trimesh.Trimesh(
                vertices=deformed_verts,
                faces=self.character['mesh'].faces.copy()
            )
            self.cached_meshes.append(mesh)

    def __len__(self) -> int:
        return self.num_poses

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rotations, root_trans = self.poses[idx]

        # Get deformed mesh
        if self.cached_meshes:
            mesh = self.cached_meshes[idx]
        else:
            chain = self.character['kinematic_chain']
            lbs = self.character['lbs']

            if chain is None or lbs is None:
                mesh = self.character['mesh']
            else:
                transforms = chain.forward_kinematics(rotations, root_trans)
                deformed_verts = lbs(transforms).detach().cpu().numpy()

                import trimesh
                mesh = trimesh.Trimesh(
                    vertices=deformed_verts,
                    faces=self.character['mesh'].faces.copy()
                )

        # Sample points
        num_surface = int(self.num_samples * self.surface_ratio)
        num_volume = self.num_samples - num_surface

        # Surface samples
        surface_pts, face_indices = mesh.sample(num_surface, return_index=True)
        surface_normals = mesh.face_normals[face_indices]
        surface_pts = surface_pts + surface_normals * np.random.randn(num_surface, 1) * self.surface_noise_std

        # Volume samples
        volume_pts = np.random.uniform(
            self.bounds[0], self.bounds[1],
            size=(num_volume, 3)
        )

        points = np.concatenate([surface_pts, volume_pts], axis=0).astype(np.float32)

        # Compute SDF
        sdf = compute_sdf(mesh, points).astype(np.float32)

        return {
            'points': torch.from_numpy(points).float(),
            'sdf': torch.from_numpy(sdf).float().view(-1, 1),
            'pose_rotations': rotations,
            'pose_translation': root_trans,
            'pose_idx': torch.tensor(idx, dtype=torch.long)
        }


class AnimatedMeshDataset(Dataset):
    """
    Dataset for training on animated character sequences.

    Provides time-conditioned samples for spacetime neural fields.
    Samples uniformly across the animation timeline.
    """

    def __init__(
        self,
        mesh_path: Union[str, Path],
        animation_paths: List[Union[str, Path]],
        animation_name: Optional[str] = None,
        num_samples: int = 10000,
        num_time_samples: int = 50,
        surface_ratio: float = 0.5,
        bounds: Tuple[float, float] = (-1.5, 1.5),
        surface_noise_std: float = 0.02,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            mesh_path: Path to FBX file with mesh
            animation_paths: Paths to animation FBX files
            animation_name: Specific animation to use (None = first)
            num_samples: Points per time sample
            num_time_samples: Number of time steps
            surface_ratio: Fraction near surface
            bounds: Spatial sampling bounds
            surface_noise_std: Surface perturbation std
            device: Torch device
        """
        self.character = load_rigged_character(
            mesh_path,
            animation_paths=animation_paths,
            device=device,
            normalize=True
        )

        # Get animation
        if not self.character['animations']:
            raise ValueError("No animations found in provided files")

        if animation_name:
            if animation_name not in self.character['animations']:
                raise ValueError(f"Animation '{animation_name}' not found")
            self.animation = self.character['animations'][animation_name]
        else:
            self.animation = list(self.character['animations'].values())[0]

        self.num_samples = num_samples
        self.num_time_samples = num_time_samples
        self.surface_ratio = surface_ratio
        self.bounds = bounds
        self.surface_noise_std = surface_noise_std
        self.device = device

        # Precompute time samples (normalized to [0, 1])
        self.time_samples = torch.linspace(0, 1, num_time_samples, device=device)

    def _interpolate_pose(self, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interpolate animation at normalized time t in [0, 1]."""
        anim = self.animation
        num_frames = anim.keyframe_rotations.shape[0]

        frame_idx = t * (num_frames - 1)
        idx0 = int(frame_idx)
        idx1 = min(idx0 + 1, num_frames - 1)
        alpha = frame_idx - idx0

        # Linear interpolation for positions
        root_trans = (1 - alpha) * anim.root_translations[idx0] + alpha * anim.root_translations[idx1]

        # SLERP for rotations (simplified - just linear blend and normalize)
        rotations = (1 - alpha) * anim.keyframe_rotations[idx0] + alpha * anim.keyframe_rotations[idx1]
        rotations = rotations / rotations.norm(dim=-1, keepdim=True)

        return rotations, root_trans

    def __len__(self) -> int:
        return self.num_time_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.time_samples[idx]
        rotations, root_trans = self._interpolate_pose(t.item())

        # Compute deformed mesh
        chain = self.character['kinematic_chain']
        lbs = self.character['lbs']

        if chain is None or lbs is None:
            mesh = self.character['mesh']
        else:
            transforms = chain.forward_kinematics(rotations, root_trans)
            deformed_verts = lbs(transforms).detach().cpu().numpy()

            import trimesh
            mesh = trimesh.Trimesh(
                vertices=deformed_verts,
                faces=self.character['mesh'].faces.copy()
            )

        # Sample points
        num_surface = int(self.num_samples * self.surface_ratio)
        num_volume = self.num_samples - num_surface

        surface_pts, face_indices = mesh.sample(num_surface, return_index=True)
        surface_normals = mesh.face_normals[face_indices]
        surface_pts = surface_pts + surface_normals * np.random.randn(num_surface, 1) * self.surface_noise_std

        volume_pts = np.random.uniform(
            self.bounds[0], self.bounds[1],
            size=(num_volume, 3)
        )

        points = np.concatenate([surface_pts, volume_pts], axis=0).astype(np.float32)
        sdf = compute_sdf(mesh, points).astype(np.float32)

        return {
            'points': torch.from_numpy(points).float(),
            'time': t.unsqueeze(0),
            'sdf': torch.from_numpy(sdf).float().view(-1, 1),
            'pose_rotations': rotations,
            'pose_translation': root_trans
        }

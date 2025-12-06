"""
Tests for FBX loading functionality.

Run with: pytest tests/test_fbx_loader.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_FBX_PATH = PROJECT_ROOT / "input/3d_meshes/x_bot_t_pose.fbx"
TEST_ANIM_PATHS = [
    PROJECT_ROOT / "input/animations/x_bot_walking.fbx",
    PROJECT_ROOT / "input/animations/x_bot_breakdance_1990.fbx",
]


class TestFBXLoaderImports:
    """Test that FBX loader modules can be imported."""

    def test_import_fbx_loader(self):
        """Test basic imports work."""
        from pga_inr.data.fbx_loader import (
            load_rigged_mesh,
            load_animation,
            load_rigged_character,
        )

    def test_import_data_classes(self):
        """Test data class imports."""
        from pga_inr.data.fbx_loader import (
            MeshData,
            SkeletonData,
            SkinningData,
            AnimationData,
        )

    def test_import_exceptions(self):
        """Test exception imports."""
        from pga_inr.data.fbx_loader import (
            FBXLoadError,
            FBXMeshError,
            FBXSkeletonError,
            FBXAnimationError,
        )

    def test_import_from_data_module(self):
        """Test imports from pga_inr.data work."""
        from pga_inr.data import (
            load_rigged_mesh,
            load_animation,
            load_rigged_character,
            MeshData,
            SkeletonData,
            SkinningData,
            AnimationData,
        )

    def test_import_datasets(self):
        """Test dataset imports."""
        from pga_inr.data import (
            CanonicalMeshDataset,
            RiggedMeshDataset,
            AnimatedMeshDataset,
        )


class TestCoordinateSystem:
    """Test coordinate system conversion."""

    def test_conversion_matrix_identity(self):
        """OpenGL to OpenGL should be identity."""
        from pga_inr.data.fbx_loader import CoordinateSystem

        matrix = CoordinateSystem.get_conversion_matrix('opengl', 'opengl')
        assert np.allclose(matrix, np.eye(4))

    def test_blender_to_opengl_conversion(self):
        """Test Blender (Z-up) to OpenGL (Y-up) conversion."""
        from pga_inr.data.fbx_loader import CoordinateSystem

        matrix = CoordinateSystem.get_conversion_matrix('blender', 'opengl')
        assert not np.allclose(matrix, np.eye(4))
        assert matrix.shape == (4, 4)

    def test_apply_to_vertices(self):
        """Test vertex transformation."""
        from pga_inr.data.fbx_loader import CoordinateSystem

        vertices = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        identity = np.eye(4, dtype=np.float32)

        result = CoordinateSystem.apply_to_vertices(vertices, identity)
        assert np.allclose(result, vertices)

    def test_apply_to_normals(self):
        """Test normal transformation."""
        from pga_inr.data.fbx_loader import CoordinateSystem

        normals = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        identity = np.eye(4, dtype=np.float32)

        result = CoordinateSystem.apply_to_normals(normals, identity)
        assert np.allclose(result, normals)


class TestMatrixDecomposition:
    """Test matrix decomposition utilities."""

    def test_decompose_identity_matrix(self):
        """Identity matrix should decompose to zero translation and identity quaternion."""
        from pga_inr.data.fbx_loader import _decompose_matrix

        identity = np.eye(4, dtype=np.float32)
        trans, quat = _decompose_matrix(identity)

        assert np.allclose(trans, [0, 0, 0])
        assert np.allclose(quat, [1, 0, 0, 0], atol=1e-5)

    def test_decompose_translation_matrix(self):
        """Test pure translation matrix decomposition."""
        from pga_inr.data.fbx_loader import _decompose_matrix

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = [1, 2, 3]

        trans, quat = _decompose_matrix(matrix)

        assert np.allclose(trans, [1, 2, 3])
        assert np.allclose(quat, [1, 0, 0, 0], atol=1e-5)

    def test_decompose_rotation_matrix(self):
        """Test rotation matrix decomposition produces unit quaternion."""
        from pga_inr.data.fbx_loader import _decompose_matrix

        # 90 degree rotation around Z-axis
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]

        trans, quat = _decompose_matrix(matrix)

        # Check unit quaternion
        assert np.allclose(np.linalg.norm(quat), 1.0)
        # Translation should be zero
        assert np.allclose(trans, [0, 0, 0])


class TestQuaternionSlerp:
    """Test quaternion SLERP implementation."""

    def test_slerp_t0_returns_q0(self):
        """SLERP at t=0 should return q0."""
        from pga_inr.data.fbx_loader import _quaternion_slerp_np

        q0 = np.array([1, 0, 0, 0], dtype=np.float32)
        q1 = np.array([0.707, 0.707, 0, 0], dtype=np.float32)

        result = _quaternion_slerp_np(q0, q1, 0.0)
        assert np.allclose(result, q0, atol=1e-5)

    def test_slerp_t1_returns_q1(self):
        """SLERP at t=1 should return q1 (normalized)."""
        from pga_inr.data.fbx_loader import _quaternion_slerp_np

        q0 = np.array([1, 0, 0, 0], dtype=np.float32)
        q1 = np.array([0.707, 0.707, 0, 0], dtype=np.float32)
        q1_norm = q1 / np.linalg.norm(q1)

        result = _quaternion_slerp_np(q0, q1, 1.0)
        assert np.allclose(result, q1_norm, atol=1e-3)

    def test_slerp_returns_unit_quaternion(self):
        """SLERP should always return unit quaternion."""
        from pga_inr.data.fbx_loader import _quaternion_slerp_np

        q0 = np.array([1, 0, 0, 0], dtype=np.float32)
        q1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _quaternion_slerp_np(q0, q1, t)
            assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-5)

    def test_slerp_handles_antipodal(self):
        """SLERP should handle antipodal quaternions."""
        from pga_inr.data.fbx_loader import _quaternion_slerp_np

        q0 = np.array([1, 0, 0, 0], dtype=np.float32)
        q1 = np.array([-1, 0, 0, 0], dtype=np.float32)  # Antipodal

        result = _quaternion_slerp_np(q0, q1, 0.5)
        assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-5)


class TestFBXLoaderWithoutFile:
    """Tests that don't require actual FBX file."""

    def test_file_not_found_error(self):
        """Test error handling for missing file."""
        from pga_inr.data.fbx_loader import load_rigged_mesh, FBXLoadError

        with pytest.raises((FileNotFoundError, FBXLoadError, ImportError)):
            load_rigged_mesh("nonexistent.fbx")

    def test_mesh_data_namedtuple(self):
        """Test MeshData structure."""
        from pga_inr.data.fbx_loader import MeshData

        mesh = MeshData(
            vertices=np.zeros((10, 3), dtype=np.float32),
            faces=np.zeros((5, 3), dtype=np.int32),
            normals=np.zeros((10, 3), dtype=np.float32)
        )

        assert mesh.vertices.shape == (10, 3)
        assert mesh.faces.shape == (5, 3)
        assert mesh.normals.shape == (10, 3)

    def test_skeleton_data_namedtuple(self):
        """Test SkeletonData structure."""
        from pga_inr.data.fbx_loader import SkeletonData

        skeleton = SkeletonData(
            joint_tree={'root': {'parent': None, 'translation': [0, 0, 0], 'quaternion': [1, 0, 0, 0]}},
            joint_names=['root'],
            bind_matrices=np.eye(4, dtype=np.float32)[np.newaxis]
        )

        assert 'root' in skeleton.joint_tree
        assert skeleton.joint_names == ['root']
        assert skeleton.bind_matrices.shape == (1, 4, 4)

    def test_animation_data_namedtuple(self):
        """Test AnimationData structure."""
        from pga_inr.data.fbx_loader import AnimationData

        anim = AnimationData(
            name='walk',
            duration=2.0,
            fps=30.0,
            keyframe_rotations=np.zeros((60, 10, 4), dtype=np.float32),
            root_translations=np.zeros((60, 3), dtype=np.float32),
            keyframe_times=np.linspace(0, 2.0, 60, dtype=np.float32)
        )

        assert anim.name == 'walk'
        assert anim.duration == 2.0
        assert anim.fps == 30.0
        assert anim.keyframe_rotations.shape == (60, 10, 4)


def _gltf_exists(fbx_path):
    """Check if glTF conversion exists for FBX file."""
    return Path(fbx_path).with_suffix('.gltf').exists()


@pytest.mark.skipif(
    not (TEST_FBX_PATH.exists() or _gltf_exists(TEST_FBX_PATH)),
    reason="Test FBX/glTF file not found"
)
class TestFBXLoadingWithFile:
    """Tests that require actual FBX or glTF file."""

    def test_load_rigged_mesh_returns_tuple(self):
        """Test basic FBX/glTF loading returns correct types."""
        from pga_inr.data.fbx_loader import load_rigged_mesh, MeshData, SkeletonData, SkinningData

        mesh, skeleton, skinning = load_rigged_mesh(str(TEST_FBX_PATH))

        assert isinstance(mesh, MeshData)
        assert isinstance(skeleton, SkeletonData)
        assert isinstance(skinning, SkinningData)

    def test_load_mesh_has_vertices(self):
        """Test loaded mesh has vertices."""
        from pga_inr.data.fbx_loader import load_rigged_mesh

        mesh, _, _ = load_rigged_mesh(str(TEST_FBX_PATH))

        assert mesh.vertices.shape[0] > 0
        assert mesh.vertices.shape[1] == 3

    def test_load_mesh_normalized(self):
        """Test loaded mesh is normalized to unit cube."""
        from pga_inr.data.fbx_loader import load_rigged_mesh

        mesh, _, _ = load_rigged_mesh(str(TEST_FBX_PATH), normalize=True)

        assert mesh.vertices.min() >= -1.1
        assert mesh.vertices.max() <= 1.1

    def test_load_skeleton_creates_kinematic_chain(self):
        """Test skeleton can be used to create KinematicChain."""
        from pga_inr.data.fbx_loader import load_rigged_mesh
        from pga_inr.spacetime import KinematicChain

        _, skeleton, _ = load_rigged_mesh(str(TEST_FBX_PATH))

        if skeleton.joint_names:  # Only if skeleton exists
            chain = KinematicChain(skeleton.joint_tree, device=torch.device('cpu'))
            assert len(chain.joints) == len(skeleton.joint_names)

    def test_skinning_weights_normalized(self):
        """Test that skinning weights sum to 1."""
        from pga_inr.data.fbx_loader import load_rigged_mesh

        _, _, skinning = load_rigged_mesh(str(TEST_FBX_PATH))

        weight_sums = skinning.bone_weights.sum(axis=1)
        assert np.allclose(weight_sums, np.ones_like(weight_sums), atol=1e-5)

    def test_load_rigged_character_complete(self):
        """Test complete character loading."""
        from pga_inr.data.fbx_loader import load_rigged_character

        character = load_rigged_character(str(TEST_FBX_PATH))

        assert 'mesh' in character
        assert 'kinematic_chain' in character
        assert 'lbs' in character
        assert 'dqs' in character
        assert 'rest_vertices' in character
        assert 'bone_weights' in character
        assert 'bone_indices' in character
        assert 'metadata' in character


@pytest.mark.skipif(
    not ((TEST_FBX_PATH.exists() or _gltf_exists(TEST_FBX_PATH)) and
         any(p.exists() or _gltf_exists(p) for p in TEST_ANIM_PATHS)),
    reason="Test FBX/glTF or animation files not found"
)
class TestAnimationLoading:
    """Tests for animation loading."""

    def test_load_animation(self):
        """Test loading animation from separate file."""
        from pga_inr.data.fbx_loader import load_rigged_mesh, load_animation

        _, skeleton, _ = load_rigged_mesh(str(TEST_FBX_PATH))

        anim_path = next((p for p in TEST_ANIM_PATHS if p.exists() or _gltf_exists(p)), None)
        if anim_path:
            animations = load_animation(str(anim_path), skeleton, sample_fps=30.0)
            assert len(animations) > 0

    def test_load_character_with_animations(self):
        """Test loading character with animations."""
        from pga_inr.data.fbx_loader import load_rigged_character

        anim_paths = [str(p) for p in TEST_ANIM_PATHS if p.exists() or _gltf_exists(p)]
        if anim_paths:
            character = load_rigged_character(
                str(TEST_FBX_PATH),
                animation_paths=anim_paths
            )

            assert character['metadata']['num_animations'] > 0


class TestRiggedDatasets:
    """Tests for rigged character datasets."""

    def test_canonical_mesh_dataset_creation(self):
        """Test creating CanonicalMeshDataset with mock mesh."""
        import trimesh
        from pga_inr.data.rigged_dataset import CanonicalMeshDataset

        # Create simple cube mesh
        mesh = trimesh.creation.box([1, 1, 1])

        dataset = CanonicalMeshDataset(
            mesh=mesh,
            num_samples=100,
            cache_size=1000
        )

        assert len(dataset) > 0

    def test_canonical_mesh_dataset_getitem(self):
        """Test CanonicalMeshDataset returns correct format."""
        import trimesh
        from pga_inr.data.rigged_dataset import CanonicalMeshDataset

        mesh = trimesh.creation.box([1, 1, 1])

        dataset = CanonicalMeshDataset(
            mesh=mesh,
            num_samples=100,
            cache_size=1000
        )

        sample = dataset[0]

        assert 'points' in sample
        assert 'sdf' in sample
        assert sample['points'].shape == (100, 3)
        assert sample['sdf'].shape == (100, 1)

    @pytest.mark.skipif(
        not (TEST_FBX_PATH.exists() or _gltf_exists(TEST_FBX_PATH)),
        reason="Test FBX/glTF file not found"
    )
    def test_rigged_mesh_dataset(self):
        """Test RiggedMeshDataset creation and sampling."""
        from pga_inr.data.rigged_dataset import RiggedMeshDataset

        dataset = RiggedMeshDataset(
            mesh_path=str(TEST_FBX_PATH),
            num_samples=100,
            num_poses=2
        )

        assert len(dataset) == 2

        sample = dataset[0]
        assert 'points' in sample
        assert 'sdf' in sample
        assert sample['points'].shape == (100, 3)
        assert sample['sdf'].shape == (100, 1)

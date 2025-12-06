"""
FBX file loading utilities for rigged character import.

Provides functions for extracting:
- Mesh geometry (vertices, faces, normals)
- Skeleton hierarchy (joint tree compatible with KinematicChain)
- Skinning weights (bone weights and indices for LBS/DQS)
- Animations (keyframe rotations and root translations)

Requires PyAssimp: pip install pyassimp
Also requires Assimp native library: brew install assimp (macOS)
"""

from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
from pathlib import Path
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class MeshData(NamedTuple):
    """Container for mesh geometry data."""
    vertices: np.ndarray          # (V, 3) float32
    faces: np.ndarray             # (F, 3) int32
    normals: Optional[np.ndarray]  # (V, 3) float32 or None


class SkeletonData(NamedTuple):
    """Container for skeleton hierarchy data."""
    joint_tree: Dict[str, Dict[str, Any]]  # KinematicChain format
    joint_names: List[str]                   # Ordered joint names
    bind_matrices: np.ndarray               # (J, 4, 4) inverse bind matrices


class SkinningData(NamedTuple):
    """Container for skinning weight data."""
    bone_weights: np.ndarray   # (V, max_bones) float32
    bone_indices: np.ndarray   # (V, max_bones) int32
    max_bones_per_vertex: int


class AnimationData(NamedTuple):
    """Container for animation data."""
    name: str
    duration: float                         # Total duration in seconds
    fps: float                              # Frames per second
    keyframe_rotations: np.ndarray         # (T, J, 4) quaternions [w,x,y,z]
    root_translations: np.ndarray          # (T, 3) root translations
    keyframe_times: np.ndarray             # (T,) time stamps


# =============================================================================
# Exceptions
# =============================================================================

class FBXLoadError(Exception):
    """Base exception for FBX loading errors."""
    pass


class FBXMeshError(FBXLoadError):
    """Error loading mesh data from FBX."""
    pass


class FBXSkeletonError(FBXLoadError):
    """Error loading skeleton data from FBX."""
    pass


class FBXAnimationError(FBXLoadError):
    """Error loading animation data from FBX."""
    pass


# =============================================================================
# Assimp Availability Check
# =============================================================================

def _check_assimp_available() -> bool:
    """Check if pyassimp is available."""
    try:
        import pyassimp
        return True
    except ImportError:
        return False


def _get_assimp_install_message() -> str:
    """Get installation instructions for pyassimp."""
    return (
        "PyAssimp is not installed. To load FBX files, install it:\n"
        "  pip install pyassimp\n"
        "  brew install assimp  # macOS\n"
        "  apt-get install libassimp-dev  # Ubuntu/Debian"
    )


# =============================================================================
# Coordinate System Handling
# =============================================================================

class CoordinateSystem:
    """Coordinate system conversion utilities."""

    OPENGL = 'opengl'      # Y-up, right-handed (target)
    BLENDER = 'blender'    # Z-up, right-handed
    DIRECTX = 'directx'    # Y-up, left-handed

    # Conversion matrices to OpenGL (Y-up, right-handed)
    _CONVERSIONS = {
        ('blender', 'opengl'): np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32),
        ('directx', 'opengl'): np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32),
    }

    @staticmethod
    def detect_from_scene(scene) -> str:
        """Detect coordinate system from Assimp scene metadata."""
        # Check metadata for up-axis
        if hasattr(scene, 'metadata') and scene.metadata:
            for key, value in scene.metadata.items():
                key_lower = key.lower() if isinstance(key, str) else str(key).lower()
                if 'upaxis' in key_lower:
                    if value in (1, 'Y', 'y'):
                        return CoordinateSystem.OPENGL
                    elif value in (2, 'Z', 'z'):
                        return CoordinateSystem.BLENDER
        # Default to OpenGL (most FBX files are Y-up)
        return CoordinateSystem.OPENGL

    @staticmethod
    def get_conversion_matrix(
        from_system: str,
        to_system: str = 'opengl'
    ) -> np.ndarray:
        """Get 4x4 transformation matrix for coordinate conversion."""
        if from_system == to_system:
            return np.eye(4, dtype=np.float32)

        key = (from_system, to_system)
        if key in CoordinateSystem._CONVERSIONS:
            return CoordinateSystem._CONVERSIONS[key]

        # Try inverse
        inverse_key = (to_system, from_system)
        if inverse_key in CoordinateSystem._CONVERSIONS:
            return np.linalg.inv(CoordinateSystem._CONVERSIONS[inverse_key]).astype(np.float32)

        logger.warning(f"Unknown coordinate conversion: {from_system} -> {to_system}, using identity")
        return np.eye(4, dtype=np.float32)

    @staticmethod
    def apply_to_vertices(
        vertices: np.ndarray,
        matrix: np.ndarray
    ) -> np.ndarray:
        """Apply transformation to vertices (N, 3)."""
        if np.allclose(matrix, np.eye(4)):
            return vertices
        # Homogeneous coordinates
        ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([vertices, ones], axis=1)
        transformed = (matrix @ homo.T).T
        return transformed[:, :3].astype(np.float32)

    @staticmethod
    def apply_to_normals(
        normals: np.ndarray,
        matrix: np.ndarray
    ) -> np.ndarray:
        """Apply rotation to normals (ignore translation)."""
        if np.allclose(matrix, np.eye(4)):
            return normals
        R = matrix[:3, :3]
        transformed = (R @ normals.T).T
        # Renormalize
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return (transformed / norms).astype(np.float32)


# =============================================================================
# Matrix Decomposition Utilities
# =============================================================================

def _decompose_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose 4x4 transformation matrix into translation and quaternion.

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        translation: (3,) translation vector
        quaternion: (4,) quaternion [w, x, y, z]
    """
    translation = matrix[:3, 3].copy().astype(np.float32)
    rotation_matrix = matrix[:3, :3].copy()

    # Remove scale from rotation matrix
    scale = np.linalg.norm(rotation_matrix, axis=0)
    scale = np.maximum(scale, 1e-8)
    rotation_matrix = rotation_matrix / scale

    # Shepperd's method for robust matrix to quaternion
    trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s

    quaternion = np.array([w, x, y, z], dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion)

    return translation, quaternion


def _quaternion_slerp_np(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    NumPy implementation of quaternion SLERP.

    Args:
        q0: Start quaternion [w, x, y, z]
        q1: End quaternion [w, x, y, z]
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion [w, x, y, z]
    """
    dot = np.dot(q0, q1)

    # Handle antipodal case
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # Near-linear case
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s0 = np.sin((1 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta

    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


# =============================================================================
# Mesh Extraction
# =============================================================================

def _extract_mesh_from_scene(
    scene,
    coord_matrix: np.ndarray
) -> MeshData:
    """
    Extract and merge all mesh geometry from Assimp scene.

    Args:
        scene: Assimp scene object
        coord_matrix: Coordinate system conversion matrix

    Returns:
        MeshData with merged geometry
    """
    all_vertices = []
    all_faces = []
    all_normals = []
    vertex_offset = 0

    for mesh in scene.meshes:
        # Get vertices
        verts = np.array(mesh.vertices, dtype=np.float32)
        verts = CoordinateSystem.apply_to_vertices(verts, coord_matrix)
        all_vertices.append(verts)

        # Get faces
        faces = np.array([f.indices for f in mesh.faces if len(f.indices) == 3], dtype=np.int32)
        if len(faces) == 0:
            # Handle quads by triangulating
            for f in mesh.faces:
                if len(f.indices) == 4:
                    all_faces.append(np.array([f.indices[0], f.indices[1], f.indices[2]], dtype=np.int32) + vertex_offset)
                    all_faces.append(np.array([f.indices[0], f.indices[2], f.indices[3]], dtype=np.int32) + vertex_offset)
        else:
            faces += vertex_offset
            all_faces.append(faces)

        # Get normals
        if mesh.normals is not None and len(mesh.normals) > 0:
            normals = np.array(mesh.normals, dtype=np.float32)
            normals = CoordinateSystem.apply_to_normals(normals, coord_matrix)
            all_normals.append(normals)

        vertex_offset += len(verts)

    vertices = np.concatenate(all_vertices, axis=0) if all_vertices else np.zeros((0, 3), dtype=np.float32)
    faces = np.concatenate(all_faces, axis=0) if all_faces else np.zeros((0, 3), dtype=np.int32)
    normals = np.concatenate(all_normals, axis=0) if all_normals else None

    return MeshData(vertices=vertices, faces=faces, normals=normals)


# =============================================================================
# Skeleton Extraction
# =============================================================================

def _extract_skeleton_from_scene(
    scene,
    coord_matrix: np.ndarray
) -> SkeletonData:
    """
    Extract skeleton hierarchy from Assimp scene.

    Args:
        scene: Assimp scene object
        coord_matrix: Coordinate system conversion matrix

    Returns:
        SkeletonData with joint_tree dict for KinematicChain
    """
    # Collect all bone names from meshes
    bone_names = set()
    bone_offset_matrices = {}

    for mesh in scene.meshes:
        if hasattr(mesh, 'bones') and mesh.bones:
            for bone in mesh.bones:
                bone_names.add(bone.name)
                # Store the offset (inverse bind) matrix
                offset = np.array(bone.offsetmatrix).T.astype(np.float32)  # Assimp uses column-major
                bone_offset_matrices[bone.name] = offset

    if not bone_names:
        logger.warning("No bones found in scene")
        return SkeletonData(
            joint_tree={},
            joint_names=[],
            bind_matrices=np.eye(4, dtype=np.float32)[np.newaxis]
        )

    # Build hierarchy by walking the scene node tree
    joint_tree = {}
    joint_names = []
    bind_matrices = []

    def _process_node(node, parent_name=None):
        name = node.name if hasattr(node, 'name') else str(node)

        if name in bone_names:
            # Get node's transformation matrix
            matrix = np.array(node.transformation).T.astype(np.float32)  # Column-major to row-major

            # Apply coordinate conversion
            if not np.allclose(coord_matrix, np.eye(4)):
                matrix = coord_matrix @ matrix @ np.linalg.inv(coord_matrix)

            translation, quaternion = _decompose_matrix(matrix)

            # Default rotation axis (Z-axis)
            axis = [0.0, 0.0, 1.0]

            joint_tree[name] = {
                'parent': parent_name if parent_name in bone_names else None,
                'translation': translation.tolist(),
                'quaternion': quaternion.tolist(),
                'axis': axis,
                'children': []
            }

            joint_names.append(name)

            # Store bind matrix
            if name in bone_offset_matrices:
                bind_matrices.append(bone_offset_matrices[name])
            else:
                bind_matrices.append(np.eye(4, dtype=np.float32))

            # Update parent's children list
            if parent_name and parent_name in joint_tree:
                joint_tree[parent_name]['children'].append(name)

            parent_for_children = name
        else:
            parent_for_children = parent_name

        # Process children
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                _process_node(child, parent_for_children)

    _process_node(scene.rootnode)

    # Stack bind matrices
    if bind_matrices:
        bind_matrices = np.stack(bind_matrices, axis=0)
    else:
        bind_matrices = np.eye(4, dtype=np.float32)[np.newaxis]

    return SkeletonData(
        joint_tree=joint_tree,
        joint_names=joint_names,
        bind_matrices=bind_matrices
    )


# =============================================================================
# Skinning Extraction
# =============================================================================

def _extract_skinning_from_scene(
    scene,
    joint_names: List[str],
    total_vertices: int,
    max_bones: int = 4
) -> SkinningData:
    """
    Extract skinning weights from Assimp scene.

    Args:
        scene: Assimp scene object
        joint_names: Ordered list of joint names
        total_vertices: Total number of vertices
        max_bones: Maximum bone influences per vertex

    Returns:
        SkinningData with normalized weights
    """
    # Map bone names to indices
    bone_name_to_idx = {name: i for i, name in enumerate(joint_names)}

    # Initialize with zeros
    bone_weights = np.zeros((total_vertices, max_bones), dtype=np.float32)
    bone_indices = np.zeros((total_vertices, max_bones), dtype=np.int32)

    # Collect all influences per vertex
    vertex_influences: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(total_vertices)}

    vertex_offset = 0
    for mesh in scene.meshes:
        if hasattr(mesh, 'bones') and mesh.bones:
            for bone in mesh.bones:
                if bone.name not in bone_name_to_idx:
                    logger.warning(f"Bone '{bone.name}' not in skeleton, skipping")
                    continue

                bone_idx = bone_name_to_idx[bone.name]

                if hasattr(bone, 'weights') and bone.weights:
                    for weight in bone.weights:
                        vid = weight.vertexid + vertex_offset
                        if vid < total_vertices:
                            vertex_influences[vid].append((bone_idx, weight.weight))

        vertex_offset += len(mesh.vertices)

    # Sort by weight and take top max_bones
    for vid, influences in vertex_influences.items():
        if not influences:
            # Default to first bone with weight 1
            bone_indices[vid, 0] = 0
            bone_weights[vid, 0] = 1.0
            continue

        # Sort descending by weight
        influences.sort(key=lambda x: -x[1])
        influences = influences[:max_bones]

        for slot, (bone_idx, weight) in enumerate(influences):
            bone_indices[vid, slot] = bone_idx
            bone_weights[vid, slot] = weight

    # Normalize weights per vertex
    weight_sums = bone_weights.sum(axis=1, keepdims=True)
    weight_sums = np.maximum(weight_sums, 1e-8)
    bone_weights = bone_weights / weight_sums

    return SkinningData(
        bone_weights=bone_weights,
        bone_indices=bone_indices,
        max_bones_per_vertex=max_bones
    )


# =============================================================================
# Animation Extraction
# =============================================================================

def _sample_rotation_curve(
    keys: List,
    time: float
) -> np.ndarray:
    """
    Sample rotation curve at given time using SLERP.

    Args:
        keys: List of rotation keyframes
        time: Time to sample at

    Returns:
        Quaternion [w, x, y, z]
    """
    if len(keys) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if len(keys) == 1 or time <= keys[0].time:
        q = keys[0].value
        return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

    if time >= keys[-1].time:
        q = keys[-1].value
        return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

    # Find surrounding keys
    for i in range(len(keys) - 1):
        if keys[i].time <= time <= keys[i + 1].time:
            t = (time - keys[i].time) / max(keys[i + 1].time - keys[i].time, 1e-8)
            q0 = keys[i].value
            q1 = keys[i + 1].value

            q0_arr = np.array([q0.w, q0.x, q0.y, q0.z], dtype=np.float32)
            q1_arr = np.array([q1.w, q1.x, q1.y, q1.z], dtype=np.float32)

            return _quaternion_slerp_np(q0_arr, q1_arr, t)

    # Fallback
    q = keys[-1].value
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)


def _sample_position_curve(
    keys: List,
    time: float
) -> np.ndarray:
    """
    Sample position curve at given time using linear interpolation.

    Args:
        keys: List of position keyframes
        time: Time to sample at

    Returns:
        Position (3,)
    """
    if len(keys) == 0:
        return np.zeros(3, dtype=np.float32)

    if len(keys) == 1 or time <= keys[0].time:
        v = keys[0].value
        return np.array([v.x, v.y, v.z], dtype=np.float32)

    if time >= keys[-1].time:
        v = keys[-1].value
        return np.array([v.x, v.y, v.z], dtype=np.float32)

    # Find surrounding keys
    for i in range(len(keys) - 1):
        if keys[i].time <= time <= keys[i + 1].time:
            t = (time - keys[i].time) / max(keys[i + 1].time - keys[i].time, 1e-8)
            v0 = keys[i].value
            v1 = keys[i + 1].value

            p0 = np.array([v0.x, v0.y, v0.z], dtype=np.float32)
            p1 = np.array([v1.x, v1.y, v1.z], dtype=np.float32)

            return (1 - t) * p0 + t * p1

    # Fallback
    v = keys[-1].value
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def _extract_animation_from_scene(
    scene,
    joint_names: List[str],
    sample_fps: float = 30.0,
    coord_matrix: np.ndarray = None
) -> Dict[str, AnimationData]:
    """
    Extract animations from Assimp scene.

    Args:
        scene: Assimp scene object
        joint_names: Ordered list of joint names
        sample_fps: Target FPS for sampling
        coord_matrix: Coordinate conversion matrix

    Returns:
        Dict mapping animation name to AnimationData
    """
    if coord_matrix is None:
        coord_matrix = np.eye(4, dtype=np.float32)

    animations = {}

    if not hasattr(scene, 'animations') or not scene.animations:
        return animations

    # Map joint names to indices
    joint_name_to_idx = {name: i for i, name in enumerate(joint_names)}
    num_joints = len(joint_names)

    for anim in scene.animations:
        name = anim.name if anim.name else 'default'
        ticks_per_second = anim.tickspersecond if anim.tickspersecond > 0 else 24.0
        duration = anim.duration / ticks_per_second

        # Determine number of frames
        num_frames = max(int(duration * sample_fps) + 1, 2)
        keyframe_times = np.linspace(0, duration, num_frames, dtype=np.float32)

        # Initialize output arrays
        keyframe_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
        keyframe_rotations[..., 0] = 1.0  # Identity quaternion default
        root_translations = np.zeros((num_frames, 3), dtype=np.float32)

        # Find root joint (first in list, or one with no parent)
        root_joint_idx = 0

        # Process animation channels
        if hasattr(anim, 'channels') and anim.channels:
            for channel in anim.channels:
                node_name = channel.nodename if hasattr(channel, 'nodename') else None
                if node_name is None:
                    continue

                if node_name not in joint_name_to_idx:
                    # Try without prefix
                    matched = False
                    for jname in joint_names:
                        if jname.endswith(node_name) or node_name.endswith(jname):
                            node_name = jname
                            matched = True
                            break
                    if not matched:
                        continue

                joint_idx = joint_name_to_idx[node_name]

                # Sample rotation keys
                if hasattr(channel, 'rotationkeys') and channel.rotationkeys:
                    for frame_idx, t in enumerate(keyframe_times):
                        tick_time = t * ticks_per_second
                        quat = _sample_rotation_curve(channel.rotationkeys, tick_time)
                        keyframe_rotations[frame_idx, joint_idx] = quat

                # Sample position keys (for root)
                if hasattr(channel, 'positionkeys') and channel.positionkeys:
                    if joint_idx == root_joint_idx:
                        for frame_idx, t in enumerate(keyframe_times):
                            tick_time = t * ticks_per_second
                            pos = _sample_position_curve(channel.positionkeys, tick_time)
                            # Apply coordinate conversion
                            pos = CoordinateSystem.apply_to_vertices(
                                pos.reshape(1, 3), coord_matrix
                            ).squeeze()
                            root_translations[frame_idx] = pos

        animations[name] = AnimationData(
            name=name,
            duration=duration,
            fps=sample_fps,
            keyframe_rotations=keyframe_rotations,
            root_translations=root_translations,
            keyframe_times=keyframe_times
        )

    return animations


# =============================================================================
# glTF Loading Functions (Alternative to pyassimp)
# =============================================================================

def _load_gltf_mesh(gltf_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load mesh data from glTF file using trimesh."""
    import trimesh

    scene = trimesh.load(str(gltf_path))

    if hasattr(scene, 'geometry') and len(scene.geometry) > 0:
        # Merge all geometries
        meshes = []
        for name, geom in scene.geometry.items():
            if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                meshes.append(geom)

        if len(meshes) == 1:
            combined = meshes[0]
        else:
            combined = trimesh.util.concatenate(meshes)

        vertices = combined.vertices.astype(np.float32)
        faces = combined.faces.astype(np.int32)
        normals = combined.vertex_normals.astype(np.float32) if hasattr(combined, 'vertex_normals') else None
    else:
        vertices = scene.vertices.astype(np.float32)
        faces = scene.faces.astype(np.int32)
        normals = scene.vertex_normals.astype(np.float32) if hasattr(scene, 'vertex_normals') else None

    return vertices, faces, normals


def _load_gltf_skeleton(gltf_path: Union[str, Path]) -> Tuple[Dict[str, Dict], List[str], np.ndarray]:
    """Load skeleton data from glTF file.

    Handles Assimp's FBX-to-glTF conversion which creates intermediate nodes
    like `$AssimpFbx$_PreRotation` that contain important bind pose data.
    """
    import json
    from scipy.spatial.transform import Rotation

    with open(gltf_path) as f:
        gltf = json.load(f)

    if not gltf.get('skins'):
        # No skeleton in this file
        return {}, [], np.array([])

    skin = gltf['skins'][0]
    nodes = gltf['nodes']
    joint_indices = set(skin.get('joints', []))

    # Get joint names
    joint_names = []
    for idx in skin.get('joints', []):
        name = nodes[idx].get('name', f'joint_{idx}')
        joint_names.append(name)

    # Build node index to joint name mapping
    joint_idx_to_name = {}
    for idx in joint_indices:
        name = nodes[idx].get('name', f'joint_{idx}')
        joint_idx_to_name[idx] = name

    # Build child to parent mapping by traversing ALL nodes
    child_to_parent = {}
    for node_idx, node in enumerate(nodes):
        if 'children' in node:
            for child_idx in node['children']:
                child_to_parent[child_idx] = node_idx

    def get_node_local_transform(node_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get translation and rotation quaternion (wxyz) for a node."""
        node = nodes[node_idx]
        translation = np.array(node.get('translation', [0, 0, 0]), dtype=np.float32)
        rotation = node.get('rotation', [0, 0, 0, 1])  # glTF uses [x, y, z, w]
        # Convert to [w, x, y, z] format
        quaternion = np.array([rotation[3], rotation[0], rotation[1], rotation[2]], dtype=np.float32)
        return translation, quaternion

    def compose_quaternions(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Compose two quaternions (q1 * q2) in wxyz format."""
        # Convert from wxyz to xyzw for scipy
        r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
        r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
        r_combined = r1 * r2
        q = r_combined.as_quat()  # xyzw
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

    def rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion (wxyz format)."""
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.apply(vec)

    def get_composed_transform_to_parent_joint(node_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """
        Traverse up from a joint node through intermediate $AssimpFbx$ nodes
        until we reach either:
        - Another joint (return composed transform relative to that joint)
        - The root (return composed transform from root)

        Returns: (translation, quaternion_wxyz, parent_joint_name)
        """
        composed_trans = np.zeros(3, dtype=np.float32)
        composed_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity

        current = node_idx
        transforms_to_compose = []

        # Collect transforms from current joint up to parent joint (or root)
        while True:
            trans, quat = get_node_local_transform(current)
            transforms_to_compose.append((trans, quat))

            if current not in child_to_parent:
                # Reached root
                break

            parent_idx = child_to_parent[current]
            if parent_idx in joint_indices:
                # Found parent joint - stop here
                parent_name = joint_idx_to_name[parent_idx]
                break

            # Continue up through intermediate nodes
            current = parent_idx
        else:
            parent_name = None

        # Compose transforms in reverse order (from parent towards joint)
        # T_total = T_n * ... * T_2 * T_1
        for trans, quat in reversed(transforms_to_compose):
            # Compose: new_trans = old_trans + rotate(old_quat, trans)
            composed_trans = composed_trans + rotate_vector(composed_quat, trans)
            # Compose: new_quat = old_quat * quat
            composed_quat = compose_quaternions(composed_quat, quat)

        # Return parent name (or None if root)
        if current not in child_to_parent:
            return composed_trans, composed_quat, None
        else:
            return composed_trans, composed_quat, parent_name

    # Function to find nearest joint ancestor (for parent relationship)
    def find_joint_parent(node_idx: int) -> Optional[str]:
        """Traverse up the tree to find the nearest joint parent."""
        current = node_idx
        visited = set()
        while current in child_to_parent:
            parent_idx = child_to_parent[current]
            if parent_idx in visited:
                break  # Avoid infinite loops
            visited.add(parent_idx)
            if parent_idx in joint_indices:
                return joint_idx_to_name[parent_idx]
            current = parent_idx
        return None

    # Build joint tree with composed transforms
    joint_tree = {}

    for idx in skin.get('joints', []):
        node = nodes[idx]
        name = node.get('name', f'joint_{idx}')

        # Find parent joint
        parent_name = find_joint_parent(idx)

        # Get composed transform including all intermediate $AssimpFbx$ nodes
        translation, quaternion, _ = get_composed_transform_to_parent_joint(idx)

        joint_tree[name] = {
            'parent': parent_name,
            'translation': translation.tolist(),
            'quaternion': quaternion.tolist(),
            'axis': [0, 0, 1],  # Default rotation axis
            'children': []
        }

    # Build children lists based on parent relationships
    for name, info in joint_tree.items():
        parent_name = info.get('parent')
        if parent_name and parent_name in joint_tree:
            joint_tree[parent_name]['children'].append(name)

    # Load inverse bind matrices
    bind_matrices = np.eye(4, dtype=np.float32)[np.newaxis].repeat(len(joint_names), axis=0)
    if 'inverseBindMatrices' in skin:
        accessor_idx = skin['inverseBindMatrices']
        accessors = gltf['accessors']
        buffer_views = gltf['bufferViews']
        buffers = gltf['buffers']

        accessor = accessors[accessor_idx]
        buffer_view = buffer_views[accessor['bufferView']]

        # Load binary data
        buffer_info = buffers[buffer_view['buffer']]
        buffer_path = Path(gltf_path).parent / buffer_info['uri']

        if buffer_path.exists():
            with open(buffer_path, 'rb') as f:
                f.seek(buffer_view.get('byteOffset', 0))
                data = f.read(buffer_view['byteLength'])

            offset = accessor.get('byteOffset', 0)
            mat_data = np.frombuffer(data, dtype=np.float32, offset=offset)
            bind_matrices = mat_data.reshape(-1, 4, 4)

    return joint_tree, joint_names, bind_matrices


def _load_gltf_skinning(
    gltf_path: Union[str, Path],
    joint_names: List[str],
    num_vertices: int,
    max_bones: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Load skinning weights from glTF file."""
    import json

    with open(gltf_path) as f:
        gltf = json.load(f)

    bone_weights = np.zeros((num_vertices, max_bones), dtype=np.float32)
    bone_indices = np.zeros((num_vertices, max_bones), dtype=np.int32)

    if not gltf.get('meshes'):
        # Uniform weights to first bone
        bone_weights[:, 0] = 1.0
        return bone_weights, bone_indices

    # Process each mesh
    accessors = gltf.get('accessors', [])
    buffer_views = gltf.get('bufferViews', [])
    buffers = gltf.get('buffers', [])

    vertex_offset = 0
    for mesh_info in gltf['meshes']:
        for prim in mesh_info.get('primitives', []):
            attrs = prim.get('attributes', {})

            # Get vertex count
            if 'POSITION' in attrs:
                pos_accessor = accessors[attrs['POSITION']]
                vert_count = pos_accessor['count']
            else:
                continue

            # Get joint indices
            joints_data = None
            weights_data = None

            if 'JOINTS_0' in attrs:
                joints_accessor = accessors[attrs['JOINTS_0']]
                joints_bv = buffer_views[joints_accessor['bufferView']]
                buffer_info = buffers[joints_bv['buffer']]
                buffer_path = Path(gltf_path).parent / buffer_info['uri']

                if buffer_path.exists():
                    with open(buffer_path, 'rb') as f:
                        f.seek(joints_bv.get('byteOffset', 0) + joints_accessor.get('byteOffset', 0))
                        # Component type: unsigned byte (5121) or unsigned short (5123)
                        dtype = np.uint8 if joints_accessor.get('componentType') == 5121 else np.uint16
                        joints_data = np.frombuffer(f.read(vert_count * 4 * np.dtype(dtype).itemsize), dtype=dtype)
                        joints_data = joints_data.reshape(vert_count, 4)

            if 'WEIGHTS_0' in attrs:
                weights_accessor = accessors[attrs['WEIGHTS_0']]
                weights_bv = buffer_views[weights_accessor['bufferView']]
                buffer_info = buffers[weights_bv['buffer']]
                buffer_path = Path(gltf_path).parent / buffer_info['uri']

                if buffer_path.exists():
                    with open(buffer_path, 'rb') as f:
                        f.seek(weights_bv.get('byteOffset', 0) + weights_accessor.get('byteOffset', 0))
                        weights_data = np.frombuffer(f.read(vert_count * 4 * 4), dtype=np.float32)
                        weights_data = weights_data.reshape(vert_count, 4)

            if joints_data is not None and weights_data is not None:
                end_idx = min(vertex_offset + vert_count, num_vertices)
                copy_count = end_idx - vertex_offset
                bone_indices[vertex_offset:end_idx, :] = joints_data[:copy_count, :max_bones]
                bone_weights[vertex_offset:end_idx, :] = weights_data[:copy_count, :max_bones]

            vertex_offset += vert_count

    # Normalize weights
    weight_sums = bone_weights.sum(axis=1, keepdims=True)
    weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
    bone_weights = bone_weights / weight_sums

    return bone_weights, bone_indices


def _load_gltf_animation(
    gltf_path: Union[str, Path],
    joint_names: List[str],
    sample_fps: float = 30.0
) -> Dict[str, AnimationData]:
    """Load animation data from glTF file."""
    import json

    with open(gltf_path) as f:
        gltf = json.load(f)

    if not gltf.get('animations'):
        return {}

    animations = {}
    accessors = gltf.get('accessors', [])
    buffer_views = gltf.get('bufferViews', [])
    buffers = gltf.get('buffers', [])
    nodes = gltf.get('nodes', [])

    # Build node name lookup
    node_names = {i: n.get('name', f'node_{i}') for i, n in enumerate(nodes)}

    for anim_info in gltf['animations']:
        anim_name = anim_info.get('name', 'Animation')

        # Collect keyframe times
        all_times = set()
        for sampler in anim_info.get('samplers', []):
            input_accessor = accessors[sampler['input']]
            input_bv = buffer_views[input_accessor['bufferView']]
            buffer_info = buffers[input_bv['buffer']]
            buffer_path = Path(gltf_path).parent / buffer_info['uri']

            if buffer_path.exists():
                with open(buffer_path, 'rb') as f:
                    f.seek(input_bv.get('byteOffset', 0) + input_accessor.get('byteOffset', 0))
                    times = np.frombuffer(f.read(input_accessor['count'] * 4), dtype=np.float32)
                    all_times.update(times.tolist())

        if not all_times:
            continue

        all_times = sorted(all_times)
        duration = max(all_times) - min(all_times)
        num_frames = max(2, int(duration * sample_fps) + 1)

        keyframe_times = np.linspace(min(all_times), max(all_times), num_frames, dtype=np.float32)
        keyframe_rotations = np.zeros((num_frames, len(joint_names), 4), dtype=np.float32)
        keyframe_rotations[:, :, 0] = 1.0  # Identity quaternion [w,x,y,z]
        root_translations = np.zeros((num_frames, 3), dtype=np.float32)

        # Process each channel
        for channel in anim_info.get('channels', []):
            target = channel.get('target', {})
            node_idx = target.get('node')
            path = target.get('path')

            if node_idx is None:
                continue

            node_name = node_names.get(node_idx, '')
            joint_idx = joint_names.index(node_name) if node_name in joint_names else -1

            sampler = anim_info['samplers'][channel['sampler']]

            # Load input times
            input_accessor = accessors[sampler['input']]
            input_bv = buffer_views[input_accessor['bufferView']]
            buffer_info = buffers[input_bv['buffer']]
            buffer_path = Path(gltf_path).parent / buffer_info['uri']

            if not buffer_path.exists():
                continue

            with open(buffer_path, 'rb') as f:
                f.seek(input_bv.get('byteOffset', 0) + input_accessor.get('byteOffset', 0))
                times = np.frombuffer(f.read(input_accessor['count'] * 4), dtype=np.float32)

            # Load output values
            output_accessor = accessors[sampler['output']]
            output_bv = buffer_views[output_accessor['bufferView']]
            buffer_info = buffers[output_bv['buffer']]
            buffer_path = Path(gltf_path).parent / buffer_info['uri']

            with open(buffer_path, 'rb') as f:
                f.seek(output_bv.get('byteOffset', 0) + output_accessor.get('byteOffset', 0))
                if path == 'rotation':
                    values = np.frombuffer(f.read(output_accessor['count'] * 16), dtype=np.float32)
                    values = values.reshape(-1, 4)  # [x, y, z, w] in glTF
                elif path == 'translation':
                    values = np.frombuffer(f.read(output_accessor['count'] * 12), dtype=np.float32)
                    values = values.reshape(-1, 3)
                else:
                    continue

            # Interpolate to keyframe times
            for frame_idx, t in enumerate(keyframe_times):
                # Find surrounding keyframes
                idx = np.searchsorted(times, t)
                if idx == 0:
                    val = values[0]
                elif idx >= len(times):
                    val = values[-1]
                else:
                    t0, t1 = times[idx - 1], times[idx]
                    alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0
                    if path == 'rotation':
                        val = _quaternion_slerp_np(values[idx - 1], values[idx], alpha)
                    else:
                        val = (1 - alpha) * values[idx - 1] + alpha * values[idx]

                if path == 'rotation' and joint_idx >= 0:
                    # Convert [x, y, z, w] to [w, x, y, z]
                    keyframe_rotations[frame_idx, joint_idx] = [val[3], val[0], val[1], val[2]]
                elif path == 'translation' and joint_idx == 0:  # Root
                    root_translations[frame_idx] = val

        animations[anim_name] = AnimationData(
            name=anim_name,
            duration=duration,
            fps=sample_fps,
            keyframe_rotations=keyframe_rotations,
            root_translations=root_translations,
            keyframe_times=keyframe_times
        )

    return animations


def _convert_fbx_to_gltf(fbx_path: Union[str, Path]) -> Optional[Path]:
    """Convert FBX to glTF using assimp command-line tool."""
    import subprocess
    import shutil

    fbx_path = Path(fbx_path)
    gltf_path = fbx_path.with_suffix('.gltf')

    # Try to find assimp command
    assimp_cmd = None
    for cmd in ['/opt/homebrew/opt/assimp@5/bin/assimp', '/opt/homebrew/bin/assimp',
                '/usr/local/bin/assimp', shutil.which('assimp')]:
        if cmd and Path(cmd).exists() if cmd.startswith('/') else shutil.which(cmd):
            assimp_cmd = cmd
            break

    if not assimp_cmd:
        logger.warning("assimp command-line tool not found")
        return None

    try:
        result = subprocess.run(
            [assimp_cmd, 'export', str(fbx_path), str(gltf_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0 and gltf_path.exists():
            logger.info(f"Converted FBX to glTF: {gltf_path}")
            return gltf_path
        else:
            logger.warning(f"FBX to glTF conversion failed: {result.stderr}")
            return None
    except Exception as e:
        logger.warning(f"FBX to glTF conversion error: {e}")
        return None


def load_from_gltf(
    path: Union[str, Path],
    normalize: bool = True,
    max_bones: int = 4
) -> Tuple[MeshData, SkeletonData, SkinningData]:
    """
    Load rigged mesh from glTF file.

    Args:
        path: Path to glTF file
        normalize: Whether to normalize mesh to unit cube
        max_bones: Maximum bone influences per vertex

    Returns:
        Tuple of (MeshData, SkeletonData, SkinningData)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"glTF file not found: {path}")

    logger.info(f"Loading rigged mesh from glTF: {path}")

    # Load mesh
    vertices, faces, normals = _load_gltf_mesh(path)
    logger.info(f"Loaded mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Load skeleton
    joint_tree, joint_names, bind_matrices = _load_gltf_skeleton(path)
    logger.info(f"Loaded skeleton: {len(joint_names)} joints")

    # Load skinning
    bone_weights, bone_indices = _load_gltf_skinning(path, joint_names, vertices.shape[0], max_bones)
    logger.info(f"Loaded skinning: max {max_bones} bones per vertex")

    # Normalize
    if normalize and vertices.shape[0] > 0:
        center = vertices.mean(axis=0)
        vertices = vertices - center
        scale = np.abs(vertices).max()
        if scale > 0:
            vertices = vertices / scale
        logger.info("Normalized mesh to unit cube")

    mesh = MeshData(vertices=vertices, faces=faces, normals=normals)
    skeleton = SkeletonData(joint_tree=joint_tree, joint_names=joint_names, bind_matrices=bind_matrices)
    skinning = SkinningData(bone_weights=bone_weights, bone_indices=bone_indices, max_bones_per_vertex=max_bones)

    return mesh, skeleton, skinning


# =============================================================================
# Main Loading Functions
# =============================================================================

def load_rigged_mesh(
    path: Union[str, Path],
    normalize: bool = True,
    max_bones: int = 4
) -> Tuple[MeshData, SkeletonData, SkinningData]:
    """
    Load rigged mesh from FBX or glTF file.

    Extracts mesh geometry, skeleton hierarchy, and skinning weights.
    Supports FBX files (via pyassimp) and glTF files (native).
    If FBX loading fails with pyassimp, automatically converts to glTF and loads.

    Args:
        path: Path to FBX or glTF file
        normalize: Whether to normalize mesh to unit cube
        max_bones: Maximum bone influences per vertex

    Returns:
        Tuple of (MeshData, SkeletonData, SkinningData)

    Raises:
        FileNotFoundError: If file doesn't exist
        FBXLoadError: If loading fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Handle glTF files directly
    if path.suffix.lower() in ['.gltf', '.glb']:
        return load_from_gltf(path, normalize, max_bones)

    # Check for existing glTF conversion
    gltf_path = path.with_suffix('.gltf')
    if gltf_path.exists():
        logger.info(f"Found existing glTF conversion: {gltf_path}")
        return load_from_gltf(gltf_path, normalize, max_bones)

    logger.info(f"Loading rigged mesh from: {path}")

    # Try pyassimp first
    if _check_assimp_available():
        import pyassimp

        try:
            with pyassimp.load(str(path)) as scene:
                # Detect coordinate system
                source_coords = CoordinateSystem.detect_from_scene(scene)
                coord_matrix = CoordinateSystem.get_conversion_matrix(source_coords, 'opengl')
                logger.info(f"Coordinate system: {source_coords} -> opengl")

                # Extract mesh
                mesh = _extract_mesh_from_scene(scene, coord_matrix)
                logger.info(f"Extracted mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

                # Extract skeleton
                skeleton = _extract_skeleton_from_scene(scene, coord_matrix)
                logger.info(f"Extracted skeleton: {len(skeleton.joint_names)} joints")

                # Extract skinning
                skinning = _extract_skinning_from_scene(
                    scene, skeleton.joint_names, mesh.vertices.shape[0], max_bones
                )
                logger.info(f"Extracted skinning: max {max_bones} bones per vertex")

                # Normalize mesh
                if normalize and mesh.vertices.shape[0] > 0:
                    center = mesh.vertices.mean(axis=0)
                    mesh_centered = mesh.vertices - center
                    scale = np.abs(mesh_centered).max()
                    if scale > 0:
                        mesh_centered = mesh_centered / scale
                    mesh = MeshData(
                        vertices=mesh_centered,
                        faces=mesh.faces,
                        normals=mesh.normals
                    )
                    logger.info("Normalized mesh to unit cube")

                return mesh, skeleton, skinning

        except Exception as e:
            logger.warning(f"pyassimp loading failed: {e}")
            logger.info("Falling back to glTF conversion...")

    # Fallback: convert FBX to glTF and load
    gltf_path = _convert_fbx_to_gltf(path)
    if gltf_path is not None:
        return load_from_gltf(gltf_path, normalize, max_bones)

    raise FBXLoadError(
        f"Failed to load {path}. Neither pyassimp nor glTF conversion worked.\n"
        "Install assimp: brew install assimp@5 (macOS) or apt install assimp-utils (Linux)"
    )


def load_animation(
    path: Union[str, Path],
    skeleton: SkeletonData,
    sample_fps: float = 30.0
) -> Dict[str, AnimationData]:
    """
    Load animation from FBX or glTF file.

    Args:
        path: Path to animation FBX or glTF file
        skeleton: SkeletonData from load_rigged_mesh (for joint name matching)
        sample_fps: Target FPS for animation sampling

    Returns:
        Dict mapping animation name to AnimationData

    Raises:
        FileNotFoundError: If file doesn't exist
        FBXAnimationError: If loading fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Animation file not found: {path}")

    # Handle glTF files directly
    if path.suffix.lower() in ['.gltf', '.glb']:
        logger.info(f"Loading animation from glTF: {path}")
        animations = _load_gltf_animation(path, skeleton.joint_names, sample_fps)
        for name, anim in animations.items():
            logger.info(f"Loaded animation '{name}': {anim.duration:.2f}s, {anim.keyframe_rotations.shape[0]} frames")
        return animations

    # Check for existing glTF conversion
    gltf_path = path.with_suffix('.gltf')
    if gltf_path.exists():
        logger.info(f"Found existing glTF conversion: {gltf_path}")
        animations = _load_gltf_animation(gltf_path, skeleton.joint_names, sample_fps)
        for name, anim in animations.items():
            logger.info(f"Loaded animation '{name}': {anim.duration:.2f}s, {anim.keyframe_rotations.shape[0]} frames")
        return animations

    logger.info(f"Loading animation from: {path}")

    # Try pyassimp first
    if _check_assimp_available():
        import pyassimp

        try:
            with pyassimp.load(str(path)) as scene:
                source_coords = CoordinateSystem.detect_from_scene(scene)
                coord_matrix = CoordinateSystem.get_conversion_matrix(source_coords, 'opengl')

                animations = _extract_animation_from_scene(
                    scene, skeleton.joint_names, sample_fps, coord_matrix
                )

                for name, anim in animations.items():
                    logger.info(f"Loaded animation '{name}': {anim.duration:.2f}s, {anim.keyframe_rotations.shape[0]} frames")

                return animations

        except Exception as e:
            logger.warning(f"pyassimp animation loading failed: {e}")
            logger.info("Falling back to glTF conversion...")

    # Fallback: convert FBX to glTF and load
    gltf_path = _convert_fbx_to_gltf(path)
    if gltf_path is not None:
        animations = _load_gltf_animation(gltf_path, skeleton.joint_names, sample_fps)
        for name, anim in animations.items():
            logger.info(f"Loaded animation '{name}': {anim.duration:.2f}s, {anim.keyframe_rotations.shape[0]} frames")
        return animations

    raise FBXAnimationError(
        f"Failed to load animation from {path}. Neither pyassimp nor glTF conversion worked.\n"
        "Install assimp: brew install assimp@5 (macOS) or apt install assimp-utils (Linux)"
    )


def load_rigged_character(
    mesh_path: Union[str, Path],
    animation_paths: Optional[List[Union[str, Path]]] = None,
    device: Union[str, torch.device] = 'cpu',
    normalize: bool = True,
    max_bones: int = 4,
    sample_fps: float = 30.0
) -> Dict[str, Any]:
    """
    Load a complete rigged character ready for use with pga_inr.

    This high-level function creates all necessary pga_inr objects.

    Args:
        mesh_path: Path to FBX file with mesh and skeleton
        animation_paths: List of paths to animation FBX files
        device: Torch device for tensors
        normalize: Normalize mesh to unit cube
        max_bones: Maximum bone influences per vertex
        sample_fps: Animation sample rate

    Returns:
        Dict containing:
        - 'mesh': trimesh.Trimesh object
        - 'kinematic_chain': KinematicChain instance
        - 'lbs': LinearBlendSkinning instance
        - 'dqs': DualQuaternionSkinning instance
        - 'animations': Dict[str, AnimationData]
        - 'rest_vertices': (V, 3) tensor
        - 'bone_weights': (V, max_bones) tensor
        - 'bone_indices': (V, max_bones) tensor
        - 'joint_names': List[str]
        - 'skeleton_data': SkeletonData
        - 'metadata': Dict with file info

    Example:
        >>> char = load_rigged_character(
        ...     'input/3d_meshes/x_bot_t_pose.fbx',
        ...     animation_paths=['input/animations/walking.fbx']
        ... )
        >>> chain = char['kinematic_chain']
        >>> lbs = char['lbs']
    """
    from ..spacetime import KinematicChain, LinearBlendSkinning, DualQuaternionSkinning
    import trimesh

    if isinstance(device, str):
        device = torch.device(device)

    # Load mesh, skeleton, skinning WITHOUT normalization
    # We normalize manually to apply same transform to skeleton
    mesh_data, skeleton_data, skinning_data = load_rigged_mesh(
        mesh_path, normalize=False, max_bones=max_bones
    )

    # Calculate normalization parameters from raw mesh
    vertices = mesh_data.vertices.copy()
    center = np.zeros(3, dtype=np.float32)
    scale = 1.0

    if normalize and vertices.shape[0] > 0:
        center = vertices.mean(axis=0)
        vertices = vertices - center
        scale = np.abs(vertices).max()
        if scale > 0:
            vertices = vertices / scale
        logger.info(f"Normalization: center={center}, scale={scale}")

    # Apply normalization to skeleton joint translations
    # IMPORTANT: Joint translations are RELATIVE to parent, except for root.
    # The root joint needs both center offset and scale, others just scale.
    normalized_joint_tree = {}
    for name, joint_info in skeleton_data.joint_tree.items():
        trans = np.array(joint_info['translation'], dtype=np.float32)

        if joint_info['parent'] is None:
            # Root joint: apply both centering and scaling
            # This puts the skeleton in the same space as the centered mesh
            if scale > 0:
                trans = (trans - center) / scale
            else:
                trans = trans - center
        else:
            # Non-root: translations are relative, only scale
            if scale > 0:
                trans = trans / scale

        normalized_joint_tree[name] = {
            'parent': joint_info['parent'],
            'translation': trans.tolist(),
            'quaternion': joint_info['quaternion'],
            'axis': joint_info['axis'],
            'children': joint_info['children']
        }

    # Create trimesh object with normalized vertices
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=mesh_data.faces,
        vertex_normals=mesh_data.normals
    )

    # Create KinematicChain with normalized skeleton
    kinematic_chain = None
    lbs = None
    dqs = None

    if skeleton_data.joint_names:
        kinematic_chain = KinematicChain(normalized_joint_tree, device=device)

        # Convert to tensors
        rest_vertices = torch.from_numpy(vertices).float().to(device)
        bone_weights = torch.from_numpy(skinning_data.bone_weights).float().to(device)
        bone_indices = torch.from_numpy(skinning_data.bone_indices).long().to(device)

        # Create skinning modules
        lbs = LinearBlendSkinning(
            kinematic_chain=kinematic_chain,
            rest_vertices=rest_vertices,
            bone_weights=bone_weights,
            bone_indices=bone_indices
        )

        dqs = DualQuaternionSkinning(
            kinematic_chain=kinematic_chain,
            rest_vertices=rest_vertices,
            bone_weights=bone_weights,
            bone_indices=bone_indices
        )
    else:
        rest_vertices = torch.from_numpy(vertices).float().to(device)
        bone_weights = torch.zeros(vertices.shape[0], max_bones, device=device)
        bone_weights[:, 0] = 1.0
        bone_indices = torch.zeros(vertices.shape[0], max_bones, dtype=torch.long, device=device)

    # Load animations with normalized root translations
    # Pre-compensate animation quaternions so they work correctly with FK
    # glTF animations are ABSOLUTE local rotations, but our FK multiplies
    # with rest_quaternion. We need: adjusted = bind_inv * animation
    # so that FK computes: adjusted * bind = animation

    from scipy.spatial.transform import Rotation

    def quat_inverse_wxyz(q):
        """Inverse of quaternion in wxyz format."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_multiply_wxyz(q1, q2):
        """Multiply quaternions in wxyz format: q1 * q2."""
        r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])  # Convert to xyzw
        r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
        r = r1 * r2
        q = r.as_quat()  # xyzw
        return np.array([q[3], q[0], q[1], q[2]])  # wxyz

    # Get bind pose quaternions for each joint (from normalized skeleton)
    bind_quats = {}
    for joint_name in skeleton_data.joint_names:
        if joint_name in normalized_joint_tree:
            bind_quats[joint_name] = np.array(
                normalized_joint_tree[joint_name]['quaternion'], dtype=np.float32
            )
        else:
            bind_quats[joint_name] = np.array([1, 0, 0, 0], dtype=np.float32)

    animations = {}
    if animation_paths and skeleton_data.joint_names:
        for anim_path in animation_paths:
            anim_path = Path(anim_path)
            try:
                anim_dict = load_animation(anim_path, skeleton_data, sample_fps)
                for name, anim in anim_dict.items():
                    # Create unique animation name using filename if collision
                    unique_name = name
                    if unique_name in animations:
                        # Use filename stem as prefix
                        unique_name = f"{anim_path.stem}_{name}"
                    if unique_name in animations:
                        # Add counter as last resort
                        counter = 1
                        while f"{unique_name}_{counter}" in animations:
                            counter += 1
                        unique_name = f"{unique_name}_{counter}"

                    # Convert animation root translations to normalized mesh space
                    # Apply SAME transformation as mesh: (pos - center) / scale
                    root_trans = anim.root_translations.copy()
                    if scale > 0:
                        root_trans = (root_trans - center) / scale
                    else:
                        root_trans = root_trans - center

                    # Get skeleton rest hip position (now in centered space)
                    hip_rest = np.array(normalized_joint_tree['mixamorig:Hips']['translation'])

                    # Make in-place: keep Y (vertical) motion, remove XZ (horizontal) movement
                    # This makes the character animate in place rather than walking across the scene
                    root_trans[:, 0] = hip_rest[0]  # Fix X to rest position
                    root_trans[:, 2] = hip_rest[2]  # Fix Z to rest position
                    # Y is kept as-is (allows up/down bob during walk)

                    # Pre-compensate animation quaternions with inverse bind pose
                    # FK computes: adjusted * bind, we want result = original_animation
                    # Therefore: adjusted = original_animation * bind_inv
                    adjusted_rotations = anim.keyframe_rotations.copy()
                    for j, joint_name in enumerate(skeleton_data.joint_names):
                        bind_q = bind_quats.get(joint_name, np.array([1, 0, 0, 0]))
                        bind_inv = quat_inverse_wxyz(bind_q)
                        for f in range(adjusted_rotations.shape[0]):
                            anim_q = adjusted_rotations[f, j]
                            # Correct order: anim * bind_inv (not bind_inv * anim)
                            adjusted_rotations[f, j] = quat_multiply_wxyz(anim_q, bind_inv)

                    # Convert to torch tensors
                    animations[unique_name] = AnimationData(
                        name=unique_name,
                        duration=anim.duration,
                        fps=anim.fps,
                        keyframe_rotations=torch.from_numpy(adjusted_rotations).float().to(device),
                        root_translations=torch.from_numpy(root_trans).float().to(device),
                        keyframe_times=torch.from_numpy(anim.keyframe_times).float().to(device)
                    )
            except Exception as e:
                logger.warning(f"Failed to load animation {anim_path}: {e}")

    return {
        'mesh': mesh,
        'kinematic_chain': kinematic_chain,
        'lbs': lbs,
        'dqs': dqs,
        'animations': animations,
        'rest_vertices': rest_vertices,
        'bone_weights': bone_weights,
        'bone_indices': bone_indices,
        'joint_names': skeleton_data.joint_names,
        'skeleton_data': skeleton_data,
        'normalization': {'center': center, 'scale': scale},
        'metadata': {
            'mesh_path': str(mesh_path),
            'animation_paths': [str(p) for p in (animation_paths or [])],
            'num_vertices': vertices.shape[0],
            'num_faces': mesh_data.faces.shape[0],
            'num_joints': len(skeleton_data.joint_names),
            'num_animations': len(animations),
        }
    }

"""
Kinematic chains for articulated models.

Provides hierarchical skeleton representations with:
- Forward kinematics
- Inverse kinematics
- Skinning (linear blend skinning, dual quaternion skinning)
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class Joint:
    """
    Single joint in a kinematic chain.

    Represents a local motor (rotation + translation) relative to parent.
    """

    def __init__(
        self,
        name: str,
        parent: Optional['Joint'] = None,
        rest_translation: Optional[torch.Tensor] = None,
        rest_quaternion: Optional[torch.Tensor] = None,
        axis: Optional[torch.Tensor] = None
    ):
        """
        Args:
            name: Joint name
            parent: Parent joint (None for root)
            rest_translation: Rest pose translation (3,)
            rest_quaternion: Rest pose quaternion [w, x, y, z] (4,)
            axis: Rotation axis for single-angle rotations (3,)
        """
        self.name = name
        self.parent = parent
        self.children: List['Joint'] = []

        # Rest pose (bind pose)
        if rest_translation is None:
            rest_translation = torch.zeros(3)
        if rest_quaternion is None:
            rest_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
        if axis is None:
            axis = torch.tensor([0.0, 0.0, 1.0])  # Default: Z-axis

        self.rest_translation = rest_translation
        self.rest_quaternion = rest_quaternion
        self.axis = axis

        # Register as child of parent
        if parent is not None:
            parent.children.append(self)

    @property
    def num_children(self) -> int:
        return len(self.children)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class KinematicChain(nn.Module):
    """
    Articulated kinematic chain (skeleton).

    Computes global transforms via motor composition.
    """

    def __init__(
        self,
        joint_tree: Dict[str, Any],
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            joint_tree: Hierarchical definition of joints
                {
                    'root': {
                        'children': ['spine', 'left_hip', 'right_hip'],
                        'translation': [0, 0, 0],
                        'quaternion': [1, 0, 0, 0]
                    },
                    'spine': {
                        'parent': 'root',
                        'children': ['neck'],
                        'translation': [0, 0.2, 0]
                    },
                    ...
                }
            device: Device for tensors
        """
        super().__init__()

        self.device = device
        self.joints: Dict[str, Joint] = {}
        self.joint_order: List[str] = []

        # Build joint hierarchy
        self._build_hierarchy(joint_tree)

        # Learnable local motors for each joint
        num_joints = len(self.joints)
        self.local_translations = nn.Parameter(torch.zeros(num_joints, 3, device=device))
        self.local_quaternions = nn.Parameter(torch.zeros(num_joints, 4, device=device))
        self.local_quaternions.data[:, 0] = 1.0  # Identity rotation

    def _build_hierarchy(self, joint_tree: Dict[str, Any]):
        """Build joint hierarchy from tree specification."""
        # First pass: create all joints
        for name, spec in joint_tree.items():
            # Support both 'translation'/'quaternion' and 'rest_translation'/'rest_rotation' keys
            trans = torch.tensor(
                spec.get('translation', spec.get('rest_translation', [0, 0, 0])),
                dtype=torch.float32
            )
            quat = torch.tensor(
                spec.get('quaternion', spec.get('rest_rotation', [1, 0, 0, 0])),
                dtype=torch.float32
            )
            axis = None
            if 'axis' in spec:
                axis = torch.tensor(spec['axis'], dtype=torch.float32)
            self.joints[name] = Joint(name, rest_translation=trans, rest_quaternion=quat, axis=axis)
            self.joint_order.append(name)

        # Second pass: connect hierarchy
        for name, spec in joint_tree.items():
            parent_name = spec.get('parent')
            if parent_name is not None:  # Only set parent if it's not None
                self.joints[name].parent = self.joints[parent_name]
                self.joints[parent_name].children.append(self.joints[name])

        # Find root
        self.root = None
        for joint in self.joints.values():
            if joint.is_root:
                self.root = joint
                break

    def get_joint_index(self, name: str) -> int:
        """Get index of joint by name."""
        return self.joint_order.index(name)

    @property
    def joint_names(self) -> List[str]:
        """Get list of joint names in order."""
        return self.joint_order

    def forward_kinematics(
        self,
        local_rotations: Optional[torch.Tensor] = None,
        root_translation: Optional[torch.Tensor] = None,
        root_rotation: Optional[torch.Tensor] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute global motor for each joint.

        M_global[joint] = M_global[parent] * M_local[joint]

        Args:
            local_rotations: Local joint rotations. Can be:
                - (num_joints, 4) quaternions [w, x, y, z]
                - (num_joints,) single angles (uses joint's axis)
                - (batch, num_joints) for batched single angles
                - None to use learnable local_quaternions
            root_translation: Root translation override (3,)
            root_rotation: Root rotation override (4,)

        Returns:
            Dict mapping joint name → (global_translation, global_quaternion)
        """
        from ..utils.quaternion import quaternion_multiply, quaternion_to_matrix, quaternion_from_axis_angle

        if local_rotations is None:
            local_rotations = self.local_quaternions

        # Handle single angle input (convert to quaternions)
        if local_rotations.dim() == 1:
            # (num_joints,) single angles
            local_rotations = local_rotations.unsqueeze(0)

        if local_rotations.shape[-1] != 4:
            # Angles input - convert to quaternions using joint axes
            # local_rotations shape: (..., num_joints)
            angles = local_rotations
            quats = []
            for i, name in enumerate(self.joint_order):
                joint = self.joints[name]
                axis = joint.axis if joint.axis is not None else torch.tensor([0.0, 0.0, 1.0])
                axis = axis.to(self.device).unsqueeze(0)  # (1, 3)
                angle = angles[..., i:i+1].unsqueeze(-1)  # (..., 1, 1)
                quat = quaternion_from_axis_angle(axis, angle).squeeze(-2)  # (..., 4)
                quats.append(quat)
            local_rotations = torch.stack(quats, dim=-2)  # (..., num_joints, 4)
            # Remove batch dimension if it was added
            local_rotations = local_rotations.squeeze(0)

        global_transforms = {}

        def compute_recursive(joint: Joint):
            idx = self.get_joint_index(joint.name)

            # Get local transform
            local_trans = self.local_translations[idx] + joint.rest_translation.to(self.device)
            local_quat = quaternion_multiply(
                F.normalize(local_rotations[idx], dim=-1),
                joint.rest_quaternion.to(self.device)
            )

            if joint.is_root:
                # Apply root overrides
                if root_translation is not None:
                    local_trans = root_translation
                if root_rotation is not None:
                    local_quat = root_rotation

                global_trans = local_trans
                global_quat = local_quat
            else:
                # Compose with parent
                parent_trans, parent_quat = global_transforms[joint.parent.name]

                # Rotate local translation by parent rotation
                R_parent = quaternion_to_matrix(parent_quat)
                rotated_local_trans = R_parent @ local_trans

                global_trans = parent_trans + rotated_local_trans
                global_quat = quaternion_multiply(parent_quat, local_quat)

            global_transforms[joint.name] = (global_trans, F.normalize(global_quat, dim=-1))

            # Recurse to children
            for child in joint.children:
                compute_recursive(child)

        compute_recursive(self.root)

        return global_transforms

    def get_joint_positions(
        self,
        local_rotations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get global positions of all joints.

        Args:
            local_rotations: Local joint rotations

        Returns:
            Joint positions (num_joints, 3)
        """
        transforms = self.forward_kinematics(local_rotations)
        positions = []
        for name in self.joint_order:
            trans, _ = transforms[name]
            positions.append(trans)
        return torch.stack(positions)


class ArticulatedMotor(nn.Module):
    """
    Time-varying articulated skeleton.

    Combines kinematic chain with temporal motor for each joint.
    """

    def __init__(
        self,
        kinematic_chain: KinematicChain,
        num_keyframes: int = 10
    ):
        """
        Args:
            kinematic_chain: Base skeleton
            num_keyframes: Keyframes per joint
        """
        super().__init__()

        self.skeleton = kinematic_chain
        self.num_joints = len(kinematic_chain.joints)
        self.num_keyframes = num_keyframes

        # Learnable keyframe rotations for each joint
        # Shape: (num_keyframes, num_joints, 4)
        self.keyframe_rotations = nn.Parameter(
            torch.zeros(num_keyframes, self.num_joints, 4)
        )
        self.keyframe_rotations.data[..., 0] = 1.0  # Identity

        # Root trajectory
        self.root_translations = nn.Parameter(
            torch.zeros(num_keyframes, 3)
        )

    def forward(
        self,
        t: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get global transforms at time t.

        Args:
            t: Time in [0, 1]

        Returns:
            Dict of global transforms
        """
        from .interpolation import quaternion_slerp

        # Interpolate keyframes
        idx = (t * (self.num_keyframes - 1)).long().clamp(0, self.num_keyframes - 2)
        local_t = t * (self.num_keyframes - 1) - idx.float()

        # Get surrounding keyframes
        rotations_prev = F.normalize(self.keyframe_rotations[idx], dim=-1)
        rotations_next = F.normalize(self.keyframe_rotations[idx + 1], dim=-1)

        # SLERP for each joint
        local_rotations = []
        for j in range(self.num_joints):
            q = quaternion_slerp(rotations_prev[j], rotations_next[j], local_t)
            local_rotations.append(q)
        local_rotations = torch.stack(local_rotations)

        # Interpolate root translation
        root_trans = (
            (1 - local_t) * self.root_translations[idx] +
            local_t * self.root_translations[idx + 1]
        )

        return self.skeleton.forward_kinematics(local_rotations, root_trans)


class LinearBlendSkinning(nn.Module):
    """
    Linear Blend Skinning (LBS) for mesh deformation.

    Transforms mesh vertices based on weighted bone transforms.
    """

    def __init__(
        self,
        kinematic_chain: KinematicChain,
        rest_vertices: torch.Tensor,
        bone_weights: torch.Tensor,
        bone_indices: torch.Tensor
    ):
        """
        Args:
            kinematic_chain: Skeleton
            rest_vertices: Rest pose vertices (V, 3)
            bone_weights: Skinning weights (V, max_bones)
            bone_indices: Bone indices per vertex (V, max_bones)
        """
        super().__init__()

        self.skeleton = kinematic_chain
        self.register_buffer('rest_vertices', rest_vertices)
        self.register_buffer('bone_weights', bone_weights)
        self.register_buffer('bone_indices', bone_indices.long())

        # Compute inverse bind matrices
        rest_transforms = kinematic_chain.forward_kinematics()
        self.register_buffer('inverse_bind_matrices', self._compute_inverse_binds(rest_transforms))

    def _compute_inverse_binds(
        self,
        rest_transforms: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute inverse bind matrices."""
        from ..utils.quaternion import quaternion_to_matrix

        matrices = []
        for name in self.skeleton.joint_order:
            trans, quat = rest_transforms[name]
            R = quaternion_to_matrix(quat)

            # Build 4x4 matrix
            M = torch.eye(4, device=trans.device)
            M[:3, :3] = R
            M[:3, 3] = trans

            # Inverse
            M_inv = torch.inverse(M)
            matrices.append(M_inv)

        return torch.stack(matrices)

    def forward(
        self,
        global_transforms: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Deform vertices using current pose.

        Args:
            global_transforms: Current global transforms from forward kinematics

        Returns:
            Deformed vertices (V, 3)
        """
        from ..utils.quaternion import quaternion_to_matrix

        # Build current transform matrices
        pose_matrices = []
        for name in self.skeleton.joint_order:
            trans, quat = global_transforms[name]
            R = quaternion_to_matrix(quat)

            M = torch.eye(4, device=trans.device)
            M[:3, :3] = R
            M[:3, 3] = trans
            pose_matrices.append(M)
        pose_matrices = torch.stack(pose_matrices)

        # Compute skinning matrices: pose * inverse_bind
        skinning_matrices = pose_matrices @ self.inverse_bind_matrices

        # Apply to vertices
        V = self.rest_vertices.shape[0]
        max_bones = self.bone_weights.shape[1]

        # Homogeneous coordinates
        rest_homo = torch.cat([
            self.rest_vertices,
            torch.ones(V, 1, device=self.rest_vertices.device)
        ], dim=-1)

        # Weighted sum of transformed vertices
        deformed = torch.zeros_like(self.rest_vertices)

        for b in range(max_bones):
            bone_idx = self.bone_indices[:, b]
            weight = self.bone_weights[:, b:b+1]

            # Get skinning matrices for this bone slot
            M = skinning_matrices[bone_idx]  # (V, 4, 4)

            # Transform vertices
            transformed = torch.einsum('vij,vj->vi', M, rest_homo)[:, :3]

            deformed = deformed + weight * transformed

        return deformed


class DualQuaternionSkinning(nn.Module):
    """
    Dual Quaternion Skinning for artifact-free deformation.

    Uses dual quaternions to avoid "candy wrapper" artifacts.
    """

    def __init__(
        self,
        kinematic_chain: KinematicChain,
        rest_vertices: torch.Tensor,
        bone_weights: torch.Tensor,
        bone_indices: torch.Tensor
    ):
        """
        Args:
            kinematic_chain: Skeleton
            rest_vertices: Rest pose vertices (V, 3)
            bone_weights: Skinning weights (V, max_bones)
            bone_indices: Bone indices per vertex (V, max_bones)
        """
        super().__init__()

        self.skeleton = kinematic_chain
        self.register_buffer('rest_vertices', rest_vertices)
        self.register_buffer('bone_weights', bone_weights)
        self.register_buffer('bone_indices', bone_indices.long())

        # Compute inverse bind dual quaternions
        rest_transforms = kinematic_chain.forward_kinematics()
        self.register_buffer('inverse_bind_dq', self._compute_inverse_binds_dq(rest_transforms))

    def _transform_to_dual_quaternion(
        self,
        translation: torch.Tensor,
        quaternion: torch.Tensor
    ) -> torch.Tensor:
        """Convert transform to dual quaternion (8 components)."""
        # Real part: rotation quaternion
        q_r = quaternion

        # Dual part: 0.5 * t * q_r (quaternion multiplication)
        t_quat = torch.cat([torch.zeros_like(translation[:1]), translation])

        # Quaternion product
        from ..utils.quaternion import quaternion_multiply
        q_d = 0.5 * quaternion_multiply(t_quat, q_r)

        return torch.cat([q_r, q_d])

    def _compute_inverse_binds_dq(
        self,
        rest_transforms: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute inverse bind dual quaternions."""
        dqs = []
        for name in self.skeleton.joint_order:
            trans, quat = rest_transforms[name]
            dq = self._transform_to_dual_quaternion(trans, quat)

            # Inverse dual quaternion
            dq_inv = self._dual_quaternion_inverse(dq)
            dqs.append(dq_inv)

        return torch.stack(dqs)

    def _dual_quaternion_inverse(self, dq: torch.Tensor) -> torch.Tensor:
        """Compute inverse of dual quaternion."""
        from ..utils.quaternion import quaternion_conjugate

        q_r = dq[:4]
        q_d = dq[4:]

        # Inverse: conjugate real part, negate dual part
        q_r_inv = quaternion_conjugate(q_r.unsqueeze(0)).squeeze(0)
        q_d_inv = -quaternion_conjugate(q_d.unsqueeze(0)).squeeze(0)

        return torch.cat([q_r_inv, q_d_inv])

    def _dual_quaternion_multiply(
        self,
        dq1: torch.Tensor,
        dq2: torch.Tensor
    ) -> torch.Tensor:
        """Multiply two dual quaternions."""
        from ..utils.quaternion import quaternion_multiply

        q_r1, q_d1 = dq1[:4], dq1[4:]
        q_r2, q_d2 = dq2[:4], dq2[4:]

        # (q_r1 + eps * q_d1) * (q_r2 + eps * q_d2)
        # = q_r1 * q_r2 + eps * (q_r1 * q_d2 + q_d1 * q_r2)
        q_r = quaternion_multiply(q_r1.unsqueeze(0), q_r2.unsqueeze(0)).squeeze(0)
        q_d = (
            quaternion_multiply(q_r1.unsqueeze(0), q_d2.unsqueeze(0)).squeeze(0) +
            quaternion_multiply(q_d1.unsqueeze(0), q_r2.unsqueeze(0)).squeeze(0)
        )

        return torch.cat([q_r, q_d])

    def forward(
        self,
        global_transforms: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Deform vertices using dual quaternion skinning.

        Args:
            global_transforms: Current global transforms

        Returns:
            Deformed vertices (V, 3)
        """
        # Build current dual quaternions
        pose_dqs = []
        for name in self.skeleton.joint_order:
            trans, quat = global_transforms[name]
            dq = self._transform_to_dual_quaternion(trans, quat)
            pose_dqs.append(dq)
        pose_dqs = torch.stack(pose_dqs)

        # Compute skinning dual quaternions
        skinning_dqs = []
        for i, name in enumerate(self.skeleton.joint_order):
            dq = self._dual_quaternion_multiply(pose_dqs[i], self.inverse_bind_dq[i])
            skinning_dqs.append(dq)
        skinning_dqs = torch.stack(skinning_dqs)

        # Apply to vertices
        V = self.rest_vertices.shape[0]
        max_bones = self.bone_weights.shape[1]

        # Weighted blend of dual quaternions
        blended_dq = torch.zeros(V, 8, device=self.rest_vertices.device)

        for b in range(max_bones):
            bone_idx = self.bone_indices[:, b]
            weight = self.bone_weights[:, b:b+1]

            dq = skinning_dqs[bone_idx]

            # Handle antipodal issue (flip if dot product is negative)
            if b > 0:
                dot = (blended_dq[:, :4] * dq[:, :4]).sum(dim=-1, keepdim=True)
                dq = torch.where(dot < 0, -dq, dq)

            blended_dq = blended_dq + weight * dq

        # Normalize
        norm = blended_dq[:, :4].norm(dim=-1, keepdim=True)
        blended_dq = blended_dq / (norm + 1e-8)

        # Apply dual quaternion to vertices
        q_r = blended_dq[:, :4]
        q_d = blended_dq[:, 4:]

        # Transform point: p' = q * p * q^* + t
        # where t = 2 * q_d * q_r^*
        from ..utils.quaternion import quaternion_conjugate, quaternion_multiply

        q_r_conj = quaternion_conjugate(q_r)

        # Translation from dual part
        t = 2 * quaternion_multiply(q_d, q_r_conj)[:, 1:4]

        # Rotate vertices
        p_quat = torch.cat([
            torch.zeros(V, 1, device=self.rest_vertices.device),
            self.rest_vertices
        ], dim=-1)

        rotated = quaternion_multiply(
            quaternion_multiply(q_r, p_quat),
            q_r_conj
        )[:, 1:4]

        # Add translation
        deformed = rotated + t

        return deformed

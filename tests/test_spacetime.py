"""
Tests for spacetime module (4D extensions).
"""

import pytest
import torch
import math

from pga_inr.spacetime.interpolation import (
    quaternion_slerp,
    motor_slerp,
    bezier_motor,
    catmull_rom_motor,
    MotorTrajectory,
)
from pga_inr.spacetime.temporal_motor import (
    TemporalMotor,
    LearnableKeyframes,
    PeriodicMotor,
)
from pga_inr.spacetime.kinematic_chain import (
    Joint,
    KinematicChain,
    LinearBlendSkinning,
    DualQuaternionSkinning,
)
from pga_inr.spacetime.spacetime_inr import (
    Spacetime_PGA_INR,
    DeformableNeuralField,
)
from pga_inr.pga import Motor
from pga_inr.models import PGA_INR_SDF
from pga_inr.utils.quaternion import quaternion_from_axis_angle


class TestQuaternionSlerp:
    """Test quaternion spherical interpolation."""

    def test_slerp_endpoints(self):
        """Test SLERP at t=0 and t=1."""
        q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q1 = torch.tensor([[0.707, 0.707, 0.0, 0.0]])  # 90 deg around X

        result_0 = quaternion_slerp(q0, q1, torch.tensor(0.0))
        result_1 = quaternion_slerp(q0, q1, torch.tensor(1.0))

        assert torch.allclose(result_0, q0, atol=1e-5)
        assert torch.allclose(result_1.abs(), q1.abs(), atol=1e-3)

    def test_slerp_midpoint(self):
        """Test SLERP at t=0.5."""
        q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        # 90 degrees around Z
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        angle = torch.tensor([[math.pi / 2]])
        q1 = quaternion_from_axis_angle(axis, angle)

        result = quaternion_slerp(q0, q1, torch.tensor(0.5))

        # Should be 45 degrees
        expected = quaternion_from_axis_angle(axis, angle / 2)

        # Check by applying to test point
        # Or check angle is half
        dot = (result * expected).sum().abs()
        assert torch.isclose(dot, torch.tensor(1.0), atol=1e-3)

    def test_slerp_normalized(self):
        """Test that SLERP output is normalized."""
        q0 = torch.randn(4)
        q0 = q0 / q0.norm()
        q1 = torch.randn(4)
        q1 = q1 / q1.norm()

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = quaternion_slerp(q0.unsqueeze(0), q1.unsqueeze(0), torch.tensor(t))
            norm = result.norm()
            assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)


class TestMotorSlerp:
    """Test motor spherical interpolation."""

    def test_motor_slerp_endpoints(self):
        """Test motor SLERP at endpoints."""
        trans0 = torch.tensor([[0.0, 0.0, 0.0]])
        trans1 = torch.tensor([[1.0, 1.0, 1.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # motor_slerp takes individual tensors, not Motor objects
        result_trans_0, result_quat_0 = motor_slerp(trans0, quat, trans1, quat, torch.tensor(0.0))
        result_trans_1, result_quat_1 = motor_slerp(trans0, quat, trans1, quat, torch.tensor(1.0))

        assert torch.allclose(result_trans_0, trans0, atol=1e-5)
        assert torch.allclose(result_trans_1, trans1, atol=1e-5)

    def test_motor_slerp_translation_linear(self):
        """Test that translation interpolates linearly."""
        trans0 = torch.tensor([[0.0, 0.0, 0.0]])
        trans1 = torch.tensor([[2.0, 0.0, 0.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        result_trans, result_quat = motor_slerp(trans0, quat, trans1, quat, torch.tensor(0.5))

        expected_trans = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(result_trans, expected_trans, atol=1e-5)


class TestMotorTrajectory:
    """Test motor trajectory interpolation."""

    def test_trajectory_bezier(self):
        """Test Bezier motor trajectory."""
        # MotorTrajectory takes separate translation and quaternion lists
        translations = [
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0]]),
        ]
        quaternions = [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ]

        trajectory = MotorTrajectory(translations, quaternions, method='bezier')

        # Endpoints
        trans_0, quat_0 = trajectory.evaluate(torch.tensor(0.0))
        trans_1, quat_1 = trajectory.evaluate(torch.tensor(1.0))

        assert torch.allclose(trans_0, translations[0], atol=1e-4)
        assert torch.allclose(trans_1, translations[-1], atol=1e-4)

    def test_trajectory_catmull_rom(self):
        """Test Catmull-Rom motor trajectory."""
        translations = [
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0]]),
            torch.tensor([[3.0, 0.0, 0.0]]),
        ]
        quaternions = [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ]

        trajectory = MotorTrajectory(translations, quaternions, method='catmull_rom')

        # Should interpolate through middle control points
        trans_mid, quat_mid = trajectory.evaluate(torch.tensor(0.5))
        assert trans_mid is not None


class TestTemporalMotor:
    """Test time-varying motors."""

    def test_temporal_motor_keyframe(self):
        """Test keyframe-based temporal motor."""
        # TemporalMotor doesn't have device parameter
        motor = TemporalMotor(
            parameterization='keyframe',
            num_keyframes=4,
        )

        t = torch.tensor([0.0, 0.5, 1.0])
        trans, quat = motor(t)

        assert trans.shape == (3, 3)
        assert quat.shape == (3, 4)

    def test_temporal_motor_mlp(self):
        """Test MLP-based temporal motor."""
        motor = TemporalMotor(
            parameterization='mlp',
            hidden_dim=32,
        )

        t = torch.tensor([[0.5]])
        trans, quat = motor(t)

        # MLP returns shape preserving the input structure
        assert trans.shape == (1, 1, 3)
        assert quat.shape == (1, 1, 4)

        # Quaternion should be normalized
        norm = quat.norm(dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


class TestPeriodicMotor:
    """Test periodic motor."""

    def test_periodic_motor_period(self):
        """Test that motion repeats after one period."""
        # PeriodicMotor uses period and num_harmonics, not axis/amplitude/frequency
        motor = PeriodicMotor(
            period=1.0,
            num_harmonics=4,
        )

        t0 = torch.tensor(0.0)
        t1 = torch.tensor(1.0)  # One full period

        trans0, quat0 = motor(t0)
        trans1, quat1 = motor(t1)

        assert torch.allclose(trans0, trans1, atol=1e-5)
        assert torch.allclose(quat0.abs(), quat1.abs(), atol=1e-5)

    def test_periodic_motor_output_shapes(self):
        """Test periodic motor output shapes."""
        motor = PeriodicMotor(period=1.0, num_harmonics=4)

        t = torch.tensor(0.25)
        trans, quat = motor(t)

        assert trans.shape == (3,)
        assert quat.shape == (4,)


class TestLearnableKeyframes:
    """Test learnable keyframe animation."""

    def test_keyframes_learnable(self):
        """Test that keyframes are learnable parameters."""
        keyframes = LearnableKeyframes(num_keyframes=4)

        # Should have gradients
        assert keyframes.translations.requires_grad
        assert keyframes.axis_angles.requires_grad

    def test_keyframes_forward(self):
        """Test keyframes forward pass."""
        keyframes = LearnableKeyframes(num_keyframes=3)

        # Set specific keyframes
        with torch.no_grad():
            keyframes.translations.data[0] = torch.tensor([0.0, 0.0, 0.0])
            keyframes.translations.data[1] = torch.tensor([1.0, 0.0, 0.0])
            keyframes.translations.data[2] = torch.tensor([2.0, 0.0, 0.0])

        # forward() takes time and returns (translation, quaternion)
        trans, quat = keyframes(torch.tensor(0.5))

        # Should return valid shapes
        assert trans.shape[-1] == 3
        assert quat.shape[-1] == 4


class TestKinematicChain:
    """Test kinematic chain."""

    def test_kinematic_chain_creation(self):
        """Test kinematic chain creation."""
        # KinematicChain uses 'translation' and 'quaternion' keys, not 'rest_translation'
        joint_tree = {
            'root': {
                'translation': [0.0, 0.0, 0.0],
                'quaternion': [1.0, 0.0, 0.0, 0.0],
            },
            'child': {
                'parent': 'root',
                'translation': [1.0, 0.0, 0.0],
                'quaternion': [1.0, 0.0, 0.0, 0.0],
            }
        }

        chain = KinematicChain(joint_tree, device=torch.device('cpu'))

        assert len(chain.joints) == 2
        assert 'root' in chain.joints
        assert 'child' in chain.joints

    def test_forward_kinematics_identity(self):
        """Test FK with zero joint angles."""
        joint_tree = {
            'root': {
                'translation': [0.0, 0.0, 0.0],
                'quaternion': [1.0, 0.0, 0.0, 0.0],
            },
            'end': {
                'parent': 'root',
                'translation': [1.0, 0.0, 0.0],
                'quaternion': [1.0, 0.0, 0.0, 0.0],
            }
        }

        chain = KinematicChain(joint_tree, device=torch.device('cpu'))

        # forward_kinematics returns dict of (trans, quat) tuples
        transforms = chain.forward_kinematics()

        # End effector should be at (1, 0, 0) in rest pose
        end_trans, end_quat = transforms['end']
        assert torch.allclose(
            end_trans,
            torch.tensor([1.0, 0.0, 0.0]),
            atol=1e-5
        )


def _create_simple_skeleton():
    """Create a simple 2-bone skeleton for testing."""
    joint_tree = {
        'root': {'children': ['bone1']},
        'bone1': {'parent': 'root', 'children': []}
    }
    return KinematicChain(joint_tree)


def _create_simple_mesh():
    """Create a simple mesh (4 vertices, 2 bones) for skinning tests."""
    # 4 vertices: 2 near root, 2 near bone1
    rest_vertices = torch.tensor([
        [0.0, 0.0, 0.0],   # At root, fully weighted to root
        [0.5, 0.0, 0.0],   # Between, 50% each
        [1.0, 0.0, 0.0],   # At bone1, fully weighted to bone1
        [1.5, 0.0, 0.0],   # Past bone1, fully weighted to bone1
    ])

    # Weights: each vertex affected by max 2 bones
    bone_weights = torch.tensor([
        [1.0, 0.0],  # v0: 100% root
        [0.5, 0.5],  # v1: 50% root, 50% bone1
        [0.0, 1.0],  # v2: 100% bone1
        [0.0, 1.0],  # v3: 100% bone1
    ])

    # Indices: which bone each weight refers to
    # joint_order is ['root', 'bone1'] -> root=0, bone1=1
    bone_indices = torch.tensor([
        [0, 1],  # v0
        [0, 1],  # v1
        [0, 1],  # v2
        [0, 1],  # v3
    ])

    return rest_vertices, bone_weights, bone_indices


class TestSkinning:
    """Test skinning algorithms."""

    def test_lbs_identity(self):
        """Test LBS with identity transforms preserves rest pose."""
        skeleton = _create_simple_skeleton()
        rest_vertices, bone_weights, bone_indices = _create_simple_mesh()

        lbs = LinearBlendSkinning(
            kinematic_chain=skeleton,
            rest_vertices=rest_vertices,
            bone_weights=bone_weights,
            bone_indices=bone_indices
        )

        # Identity pose (same as rest pose)
        identity_transforms = skeleton.forward_kinematics()

        # Apply skinning
        deformed = lbs(identity_transforms)

        # Should match rest pose
        assert torch.allclose(deformed, rest_vertices, atol=1e-5)

    def test_dqs_identity(self):
        """Test DQS with identity transforms preserves rest pose."""
        skeleton = _create_simple_skeleton()
        rest_vertices, bone_weights, bone_indices = _create_simple_mesh()

        dqs = DualQuaternionSkinning(
            kinematic_chain=skeleton,
            rest_vertices=rest_vertices,
            bone_weights=bone_weights,
            bone_indices=bone_indices
        )

        # Identity pose
        identity_transforms = skeleton.forward_kinematics()

        # Apply skinning
        deformed = dqs(identity_transforms)

        # Should match rest pose
        assert torch.allclose(deformed, rest_vertices, atol=1e-5)


class TestSpacetimeINR:
    """Test 4D spacetime neural field."""

    def test_spacetime_inr_output_shape(self):
        """Test spacetime INR output shapes."""
        model = Spacetime_PGA_INR(
            hidden_features=32,
            hidden_layers=2
        )

        batch_size = 2
        num_points = 100
        points = torch.randn(batch_size, num_points, 3)
        times = torch.rand(batch_size, num_points, 1)
        pose = (
            torch.zeros(batch_size, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch_size)
        )

        outputs = model(points, times, pose)

        # Output key is 'density' not 'sdf'
        assert outputs['density'].shape == (batch_size, num_points, 1)
        assert outputs['velocity'].shape == (batch_size, num_points, 3)

    def test_spacetime_inr_without_velocity(self):
        """Test spacetime INR without velocity output."""
        model = Spacetime_PGA_INR(
            hidden_features=32,
            hidden_layers=2,
            output_velocity=False
        )

        points = torch.randn(2, 100, 3)
        times = torch.rand(2, 100, 1)

        outputs = model(points, times)

        assert 'density' in outputs
        assert 'velocity' not in outputs

    def test_deformable_field_output_shape(self):
        """Test deformable field output shapes."""
        canonical = PGA_INR_SDF(hidden_features=32, hidden_layers=2)
        # DeformableNeuralField uses hidden_features and hidden_layers, not deformation_hidden
        deformable = DeformableNeuralField(
            canonical_field=canonical,
            hidden_features=32,
            hidden_layers=2
        )

        points = torch.randn(2, 100, 3)
        times = torch.rand(2, 100, 1)
        pose = (
            torch.zeros(2, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
        )

        outputs = deformable(points, times, pose)

        assert outputs['sdf'].shape == (2, 100, 1)
        assert outputs['deformation'].shape == (2, 100, 3)

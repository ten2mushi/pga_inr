"""
Tests for PGA motor operations.
"""

import pytest
import torch
import math

from pga_inr.pga.motors import (
    Motor,
    rotor_from_axis_angle,
    translator_from_direction,
    motor_log,
    motor_exp,
)
from pga_inr.utils.quaternion import (
    quaternion_from_axis_angle,
    quaternion_multiply,
    quaternion_to_matrix,
)


class TestMotorCreation:
    """Test motor creation methods."""

    def test_identity_motor(self):
        """Test identity motor creation."""
        motor = Motor.identity(batch_size=1, device=torch.device('cpu'))

        assert motor.translation().shape == (1, 3)
        assert motor.rotation_quaternion().shape == (1, 4)

        # Identity has zero translation
        assert torch.allclose(motor.translation(), torch.zeros(1, 3))

        # Identity quaternion is [1, 0, 0, 0]
        expected_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(motor.rotation_quaternion(), expected_quat)

    def test_motor_from_translation_quaternion(self):
        """Test motor from translation and quaternion."""
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        motor = Motor(trans, quat)

        assert torch.allclose(motor.translation(), trans, atol=1e-5)
        assert torch.allclose(motor.rotation_quaternion(), quat, atol=1e-5)

    def test_motor_batched(self):
        """Test batched motor creation."""
        batch_size = 4
        trans = torch.randn(batch_size, 3)
        quat = torch.randn(batch_size, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)  # Normalize

        motor = Motor(trans, quat)

        assert motor.translation().shape == (batch_size, 3)
        assert motor.rotation_quaternion().shape == (batch_size, 4)


class TestMotorTransformations:
    """Test motor transformation operations."""

    def test_apply_identity(self):
        """Applying identity motor should not change points."""
        motor = Motor.identity(batch_size=1, device=torch.device('cpu'))
        points = torch.randn(1, 100, 3)

        transformed = motor.apply(points)

        assert torch.allclose(transformed, points, atol=1e-6)

    def test_apply_pure_translation(self):
        """Test pure translation motor."""
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity rotation

        motor = Motor(trans, quat)
        points = torch.tensor([[[0.0, 0.0, 0.0]]])

        transformed = motor.apply(points)

        expected = torch.tensor([[[1.0, 2.0, 3.0]]])
        assert torch.allclose(transformed, expected, atol=1e-6)

    def test_apply_pure_rotation_90_z(self):
        """Test 90 degree rotation around Z axis."""
        trans = torch.zeros(1, 3)
        # 90 degrees around Z axis
        angle = torch.tensor([[math.pi / 2]])
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        quat = quaternion_from_axis_angle(axis, angle)

        motor = Motor(trans, quat)

        # Point at (1, 0, 0) should go to (0, 1, 0)
        points = torch.tensor([[[1.0, 0.0, 0.0]]])
        transformed = motor.apply(points)

        expected = torch.tensor([[[0.0, 1.0, 0.0]]])
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_apply_combined(self):
        """Test combined rotation and translation."""
        # Rotate 90 degrees around Z, then translate by (1, 0, 0)
        trans = torch.tensor([[1.0, 0.0, 0.0]])
        angle = torch.tensor([[math.pi / 2]])
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        quat = quaternion_from_axis_angle(axis, angle)

        motor = Motor(trans, quat)

        # Point at (1, 0, 0):
        # After rotation: (0, 1, 0)
        # After translation: (1, 1, 0)
        points = torch.tensor([[[1.0, 0.0, 0.0]]])
        transformed = motor.apply(points)

        expected = torch.tensor([[[1.0, 1.0, 0.0]]])
        assert torch.allclose(transformed, expected, atol=1e-5)


class TestMotorComposition:
    """Test motor composition."""

    def test_compose_with_identity(self):
        """Composing with identity should not change motor."""
        identity = Motor.identity(batch_size=1, device=torch.device('cpu'))
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        motor = Motor(trans, quat)

        composed = motor.compose(identity)

        assert torch.allclose(composed.translation(), motor.translation(), atol=1e-6)
        assert torch.allclose(composed.rotation_quaternion(), motor.rotation_quaternion(), atol=1e-6)

    def test_compose_translations(self):
        """Test composing pure translations."""
        trans1 = torch.tensor([[1.0, 0.0, 0.0]])
        trans2 = torch.tensor([[0.0, 1.0, 0.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        motor1 = Motor(trans1, quat)
        motor2 = Motor(trans2, quat)

        composed = motor1.compose(motor2)

        expected_trans = torch.tensor([[1.0, 1.0, 0.0]])
        assert torch.allclose(composed.translation(), expected_trans, atol=1e-6)

    def test_compose_rotations(self):
        """Test composing rotations."""
        trans = torch.zeros(1, 3)
        axis = torch.tensor([[0.0, 0.0, 1.0]])

        # Two 45 degree rotations should equal one 90 degree rotation
        angle45 = torch.tensor([[math.pi / 4]])
        quat45 = quaternion_from_axis_angle(axis, angle45)

        motor45 = Motor(trans, quat45)
        composed = motor45.compose(motor45)

        # Expected: 90 degree rotation
        angle90 = torch.tensor([[math.pi / 2]])
        quat90 = quaternion_from_axis_angle(axis, angle90)

        # Check by applying to a point
        point = torch.tensor([[[1.0, 0.0, 0.0]]])
        result = composed.apply(point)
        expected = Motor(trans, quat90).apply(point)

        assert torch.allclose(result, expected, atol=1e-5)


class TestMotorInverse:
    """Test motor inversion."""

    def test_inverse_identity(self):
        """Identity inverse is identity."""
        identity = Motor.identity(batch_size=1, device=torch.device('cpu'))
        inv = identity.inverse()

        assert torch.allclose(inv.translation(), torch.zeros(1, 3), atol=1e-6)
        assert torch.allclose(inv.rotation_quaternion()[:, 0], torch.ones(1), atol=1e-6)

    def test_inverse_translation(self):
        """Inverse of translation is negative translation."""
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        motor = Motor(trans, quat)

        inv = motor.inverse()
        composed = motor.compose(inv)

        # Should be identity
        points = torch.randn(1, 10, 3)
        transformed = composed.apply(points)

        assert torch.allclose(transformed, points, atol=1e-5)

    def test_inverse_rotation(self):
        """Inverse of rotation undoes rotation."""
        trans = torch.zeros(1, 3)
        angle = torch.tensor([[math.pi / 3]])
        axis = torch.tensor([[1.0, 1.0, 1.0]])
        axis = axis / axis.norm()
        quat = quaternion_from_axis_angle(axis, angle)

        motor = Motor(trans, quat)
        inv = motor.inverse()
        composed = motor.compose(inv)

        # Should be identity
        points = torch.randn(1, 10, 3)
        transformed = composed.apply(points)

        assert torch.allclose(transformed, points, atol=1e-5)


class TestMotorInterpolation:
    """Test motor interpolation."""

    def test_interpolate_endpoints(self):
        """Interpolation at t=0 and t=1 should give endpoints."""
        trans1 = torch.tensor([[0.0, 0.0, 0.0]])
        trans2 = torch.tensor([[1.0, 1.0, 1.0]])
        quat1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        quat2 = quaternion_from_axis_angle(
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[math.pi / 2]])
        )

        motor1 = Motor(trans1, quat1)
        motor2 = Motor(trans2, quat2)

        interp0 = motor1.interpolate(motor2, 0.0)
        interp1 = motor1.interpolate(motor2, 1.0)

        assert torch.allclose(interp0.translation(), motor1.translation(), atol=1e-5)
        assert torch.allclose(interp1.translation(), motor2.translation(), atol=1e-5)

    def test_interpolate_midpoint(self):
        """Interpolation at t=0.5 should be midpoint."""
        trans1 = torch.tensor([[0.0, 0.0, 0.0]])
        trans2 = torch.tensor([[2.0, 0.0, 0.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        motor1 = Motor(trans1, quat)
        motor2 = Motor(trans2, quat)

        interp = motor1.interpolate(motor2, 0.5)

        expected_trans = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(interp.translation(), expected_trans, atol=1e-5)


class TestRotorAndTranslator:
    """Test rotor and translator creation."""

    def test_rotor_from_axis_angle(self):
        """Test rotor creation from axis-angle."""
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        angle = torch.tensor([[math.pi / 2]])

        rotor = rotor_from_axis_angle(axis, angle)

        # Rotor should be a Multivector
        assert hasattr(rotor, 'mv')
        # Check it has the right shape
        assert rotor.mv.shape[-1] == 16

    def test_translator_from_direction(self):
        """Test translator creation."""
        direction = torch.tensor([[1.0, 0.0, 0.0]])
        distance = torch.tensor([[5.0]])

        translator = translator_from_direction(direction, distance)

        # Check it's a Multivector
        assert hasattr(translator, 'mv')
        assert translator.mv.shape[-1] == 16


class TestMotorLogExp:
    """Test logarithm and exponential maps."""

    def test_exp_log_roundtrip(self):
        """Test that exp(log(M)) = M."""
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        angle = torch.tensor([[math.pi / 4]])
        axis = torch.tensor([[0.0, 1.0, 0.0]])
        quat = quaternion_from_axis_angle(axis, angle)

        motor = Motor(trans, quat)

        log_motor = motor_log(motor)
        recovered = motor_exp(log_motor)

        # Should recover original motor - check by applying to points
        points = torch.randn(1, 10, 3)
        result_orig = motor.apply(points)
        result_recovered = recovered.apply(points)

        assert torch.allclose(result_orig, result_recovered, atol=1e-4)

    def test_log_identity(self):
        """Log of identity should be zero."""
        identity = Motor.identity(batch_size=1, device=torch.device('cpu'))
        log_id = motor_log(identity)

        # motor_log returns a tensor of shape (..., 6) representing [omega, v]
        assert torch.allclose(log_id, torch.zeros_like(log_id), atol=1e-6)


class TestToMatrix:
    """Test conversion to homogeneous matrix."""

    def test_to_matrix_identity(self):
        """Identity motor should give identity matrix."""
        motor = Motor.identity(batch_size=1, device=torch.device('cpu'))
        matrix = motor.to_matrix()

        expected = torch.eye(4).unsqueeze(0)
        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_to_matrix_translation(self):
        """Check translation appears in matrix."""
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        motor = Motor(trans, quat)

        matrix = motor.to_matrix()

        # Translation should be in last column
        assert torch.allclose(matrix[0, :3, 3], trans[0], atol=1e-6)

    def test_to_matrix_rotation(self):
        """Check rotation matrix is correct."""
        trans = torch.zeros(1, 3)
        angle = torch.tensor([[math.pi / 2]])
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        quat = quaternion_from_axis_angle(axis, angle)

        motor = Motor(trans, quat)
        matrix = motor.to_matrix()

        # Expected rotation matrix for 90 deg around Z
        expected_rot = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        assert torch.allclose(matrix[0, :3, :3], expected_rot, atol=1e-5)


# =============================================================================
# Additional Tests for Complete Coverage - Yoneda Philosophy
# =============================================================================

class TestMotorProperties:
    """Tests for Motor class properties and methods."""

    def test_device_property(self):
        """Motor returns correct device."""
        motor = Motor.identity(batch_size=1, device=torch.device('cpu'))
        assert motor.device == torch.device('cpu')

    def test_dtype_property(self):
        """Motor returns correct dtype."""
        trans = torch.zeros(1, 3, dtype=torch.float64)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        motor = Motor(trans, quat)
        assert motor.dtype == torch.float64

    def test_shape_property(self):
        """Motor returns correct shape."""
        motor = Motor.identity(batch_size=5)
        assert motor.shape[0] == 5

    def test_to_device(self):
        """Motor can be moved to device."""
        motor = Motor.identity(batch_size=1)
        moved = motor.to(torch.device('cpu'))
        assert moved.device == torch.device('cpu')


class TestMotorBatching:
    """Tests for batched motor operations."""

    def test_batched_apply_matches_individual(self):
        """Batched apply gives same results as individual applies."""
        batch_size = 4
        motors = []
        for i in range(batch_size):
            trans = torch.randn(1, 3)
            quat = torch.randn(1, 4)
            quat = quat / quat.norm(dim=-1, keepdim=True)
            motors.append(Motor(trans, quat))

        # Stack into batched motor
        trans_batch = torch.cat([m.translation() for m in motors], dim=0)
        quat_batch = torch.cat([m.rotation_quaternion() for m in motors], dim=0)
        batched_motor = Motor(trans_batch, quat_batch)

        points = torch.randn(batch_size, 10, 3)
        batched_result = batched_motor.apply(points)

        # Compare with individual results
        for i in range(batch_size):
            individual_result = motors[i].apply(points[i:i+1])
            assert torch.allclose(batched_result[i], individual_result[0], atol=1e-5)

    def test_batched_compose(self):
        """Batched composition works correctly."""
        batch_size = 3
        trans1 = torch.randn(batch_size, 3)
        trans2 = torch.randn(batch_size, 3)
        quat1 = torch.randn(batch_size, 4)
        quat1 = quat1 / quat1.norm(dim=-1, keepdim=True)
        quat2 = torch.randn(batch_size, 4)
        quat2 = quat2 / quat2.norm(dim=-1, keepdim=True)

        motor1 = Motor(trans1, quat1)
        motor2 = Motor(trans2, quat2)

        composed = motor1.compose(motor2)
        assert composed.translation().shape == (batch_size, 3)
        assert composed.rotation_quaternion().shape == (batch_size, 4)

    def test_batched_inverse(self):
        """Batched inverse works correctly."""
        batch_size = 5
        trans = torch.randn(batch_size, 3)
        quat = torch.randn(batch_size, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)

        motor = Motor(trans, quat)
        inv = motor.inverse()

        composed = motor.compose(inv)

        # Should be close to identity for all
        points = torch.randn(batch_size, 10, 3)
        result = composed.apply(points)
        assert torch.allclose(result, points, atol=1e-4)


class TestMotorSpecialCases:
    """Tests for special/edge cases."""

    def test_small_rotation(self):
        """Handle very small rotations."""
        trans = torch.zeros(1, 3)
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        angle = torch.tensor([[1e-10]])
        quat = quaternion_from_axis_angle(axis, angle)

        motor = Motor(trans, quat)
        points = torch.randn(1, 10, 3)
        result = motor.apply(points)

        # Should be nearly identity
        assert torch.allclose(result, points, atol=1e-6)

    def test_180_degree_rotation(self):
        """Handle 180 degree rotation."""
        trans = torch.zeros(1, 3)
        axis = torch.tensor([[1.0, 0.0, 0.0]])
        angle = torch.tensor([[math.pi]])
        quat = quaternion_from_axis_angle(axis, angle)

        motor = Motor(trans, quat)
        points = torch.tensor([[[0.0, 1.0, 0.0]]])
        result = motor.apply(points)

        expected = torch.tensor([[[0.0, -1.0, 0.0]]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_large_translation(self):
        """Handle large translations."""
        trans = torch.tensor([[1e6, 1e6, 1e6]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        motor = Motor(trans, quat)
        points = torch.zeros(1, 10, 3)
        result = motor.apply(points)

        assert torch.allclose(result, trans.unsqueeze(1).expand(1, 10, 3), atol=1.0)

    def test_compose_order_matters(self):
        """Composition order affects result (non-commutativity)."""
        trans1 = torch.tensor([[1.0, 0.0, 0.0]])
        trans2 = torch.zeros(1, 3)
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        angle = torch.tensor([[math.pi / 2]])
        quat_rot = quaternion_from_axis_angle(axis, angle)
        quat_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # Motor 1: translate x by 1
        # Motor 2: rotate 90 deg around z
        motor_trans = Motor(trans1, quat_id)
        motor_rot = Motor(trans2, quat_rot)

        # trans then rot vs rot then trans
        comp1 = motor_rot.compose(motor_trans)  # first trans, then rot
        comp2 = motor_trans.compose(motor_rot)  # first rot, then trans

        # Use a non-origin point to see the difference in order
        # Origin is unaffected by rotation, so we need a different test point
        point = torch.tensor([[[1.0, 0.0, 0.0]]])
        result1 = comp1.apply(point)
        result2 = comp2.apply(point)

        # Results should differ
        assert not torch.allclose(result1, result2, atol=0.1)


class TestMotorGradientFlow:
    """Tests for gradient flow through motor operations."""

    def test_apply_gradient(self):
        """Gradients flow through apply."""
        trans = torch.randn(1, 3, requires_grad=True)
        quat = torch.randn(1, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        quat = quat.clone().requires_grad_(True)

        motor = Motor(trans, quat)
        points = torch.randn(1, 10, 3, requires_grad=True)
        result = motor.apply(points)

        loss = result.sum()
        loss.backward()

        assert trans.grad is not None
        assert points.grad is not None
        assert not torch.isnan(trans.grad).any()
        assert not torch.isnan(points.grad).any()

    def test_compose_gradient(self):
        """Gradients flow through compose."""
        trans1 = torch.randn(1, 3, requires_grad=True)
        trans2 = torch.randn(1, 3, requires_grad=True)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        motor1 = Motor(trans1, quat)
        motor2 = Motor(trans2, quat)

        composed = motor1.compose(motor2)
        # Apply to get gradients flowing
        points = torch.randn(1, 10, 3)
        result = composed.apply(points)
        loss = result.sum()
        loss.backward()

        assert trans1.grad is not None
        assert trans2.grad is not None

    def test_inverse_gradient(self):
        """Gradients flow through inverse."""
        trans = torch.randn(1, 3, requires_grad=True)
        quat = torch.randn(1, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)

        motor = Motor(trans, quat)
        inv = motor.inverse()
        # Apply to get gradients flowing
        points = torch.randn(1, 10, 3)
        result = inv.apply(points)
        loss = result.sum()
        loss.backward()

        assert trans.grad is not None


class TestMotorNumericalStability:
    """Tests for numerical stability."""

    def test_normalize_quaternion_on_creation(self):
        """Motor normalizes quaternion on creation if needed."""
        trans = torch.zeros(1, 3)
        quat_unnormalized = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

        motor = Motor(trans, quat_unnormalized)
        # Check quaternion is normalized
        norm = motor.rotation_quaternion().norm(dim=-1)
        assert torch.isclose(norm[0], torch.tensor(1.0), atol=1e-6)

    def test_repeated_composition_stable(self):
        """Repeated composition remains stable."""
        trans = torch.tensor([[0.01, 0.0, 0.0]])
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        angle = torch.tensor([[0.01]])
        quat = quaternion_from_axis_angle(axis, angle)

        motor = Motor(trans, quat)

        # Compose 100 times
        result = Motor.identity(batch_size=1)
        for _ in range(100):
            result = result.compose(motor)

        # Should not have NaN or Inf
        assert not torch.isnan(result.translation()).any()
        assert not torch.isnan(result.rotation_quaternion()).any()
        assert not torch.isinf(result.translation()).any()
        assert not torch.isinf(result.rotation_quaternion()).any()

        # Quaternion should still be normalized
        norm = result.rotation_quaternion().norm(dim=-1)
        assert torch.isclose(norm[0], torch.tensor(1.0), atol=1e-4)


class TestMotorFromMatrix:
    """Tests for creating Motor from 4x4 matrix."""

    def test_from_matrix_identity(self):
        """Create motor from identity matrix."""
        matrix = torch.eye(4).unsqueeze(0)
        motor = Motor.from_matrix(matrix)

        assert torch.allclose(motor.translation(), torch.zeros(1, 3), atol=1e-6)
        # Quaternion should be identity (w=1, xyz=0)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(motor.rotation_quaternion().abs(), expected.abs(), atol=1e-5)

    def test_from_matrix_roundtrip(self):
        """Motor -> matrix -> Motor roundtrip."""
        trans = torch.randn(1, 3)
        quat = torch.randn(1, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)

        motor_orig = Motor(trans, quat)
        matrix = motor_orig.to_matrix()
        motor_recovered = Motor.from_matrix(matrix)

        # Check by applying to points
        points = torch.randn(1, 10, 3)
        result_orig = motor_orig.apply(points)
        result_recovered = motor_recovered.apply(points)

        assert torch.allclose(result_orig, result_recovered, atol=1e-4)


class TestMotorInterpolationAdvanced:
    """Advanced interpolation tests."""

    def test_interpolation_is_geodesic(self):
        """Interpolation follows geodesic (screw) path.

        For pure translation (no rotation), the interpolation is linear.
        For combined rotation + translation, the path follows a screw motion
        which may not have linear translation component.
        """
        # Test 1: Pure translation interpolation should be linear
        motor1 = Motor.identity(batch_size=1)
        trans2 = torch.tensor([[2.0, 0.0, 0.0]])
        quat2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity rotation
        motor2 = Motor(trans2, quat2)

        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        translations = []
        for t in t_values:
            interp = motor1.interpolate(motor2, t)
            translations.append(interp.translation()[0, 0].item())

        # Pure translation should be linear
        expected = [0.0, 0.5, 1.0, 1.5, 2.0]
        for actual, exp in zip(translations, expected):
            assert abs(actual - exp) < 0.1

        # Test 2: Endpoints are correct for rotation + translation
        trans3 = torch.tensor([[2.0, 0.0, 0.0]])
        quat3 = quaternion_from_axis_angle(
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[math.pi]])
        )
        motor3 = Motor(trans3, quat3)

        # t=0 gives motor1, t=1 gives motor3
        interp_0 = motor1.interpolate(motor3, 0.0)
        interp_1 = motor1.interpolate(motor3, 1.0)

        assert torch.allclose(interp_0.translation(), motor1.translation(), atol=1e-5)
        assert torch.allclose(interp_1.translation(), motor3.translation(), atol=1e-5)

    def test_interpolation_batched(self):
        """Interpolation works with batched motors."""
        batch_size = 4
        motor1 = Motor.identity(batch_size=batch_size)
        trans2 = torch.randn(batch_size, 3)
        quat2 = torch.randn(batch_size, 4)
        quat2 = quat2 / quat2.norm(dim=-1, keepdim=True)
        motor2 = Motor(trans2, quat2)

        interp = motor1.interpolate(motor2, 0.5)
        assert interp.translation().shape == (batch_size, 3)
        assert interp.rotation_quaternion().shape == (batch_size, 4)


class TestToMatrixAndBack:
    """Tests for matrix representations."""

    def test_matrix_is_homogeneous(self):
        """Motor matrix is proper 4x4 homogeneous transform."""
        trans = torch.randn(1, 3)
        quat = torch.randn(1, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        motor = Motor(trans, quat)

        matrix = motor.to_matrix()

        # Bottom row should be [0, 0, 0, 1]
        assert torch.allclose(matrix[0, 3, :3], torch.zeros(3), atol=1e-6)
        assert torch.isclose(matrix[0, 3, 3], torch.tensor(1.0), atol=1e-6)

    def test_matrix_rotation_is_orthogonal(self):
        """Rotation part of matrix is orthogonal."""
        trans = torch.randn(1, 3)
        quat = torch.randn(1, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        motor = Motor(trans, quat)

        matrix = motor.to_matrix()
        R = matrix[0, :3, :3]

        # R^T R should be I
        RtR = R.T @ R
        assert torch.allclose(RtR, torch.eye(3), atol=1e-5)

    def test_batched_to_matrix(self):
        """Batched to_matrix works."""
        batch_size = 5
        trans = torch.randn(batch_size, 3)
        quat = torch.randn(batch_size, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        motor = Motor(trans, quat)

        matrices = motor.to_matrix()
        assert matrices.shape == (batch_size, 4, 4)

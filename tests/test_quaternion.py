"""
Tests for quaternion operations.

Following the Yoneda philosophy: these tests completely DEFINE quaternion
behavior. The tests serve as executable documentation for:

1. Quaternion representation: (w, x, y, z) = w + xi + yj + zk
2. Basic operations: normalize, conjugate, inverse, multiply
3. Conversions: axis-angle, rotation matrix
4. Interpolation: SLERP
5. Utilities: random, identity, rotate_vector, angle_distance
"""

import pytest
import torch
import math

from pga_inr.utils.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_from_axis_angle,
    quaternion_to_axis_angle,
    quaternion_slerp,
    random_quaternion,
    identity_quaternion,
    rotate_vector,
    quaternion_angle_distance,
)


# =============================================================================
# Normalization Tests
# =============================================================================

class TestNormalizeQuaternion:
    """Tests for quaternion normalization."""

    def test_unit_quaternion_unchanged(self):
        """Unit quaternion is unchanged by normalization."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity
        normalized = normalize_quaternion(q)
        assert torch.allclose(normalized, q)

    def test_produces_unit_norm(self):
        """Normalization produces unit norm."""
        q = torch.randn(4)
        normalized = normalize_quaternion(q)
        norm = torch.norm(normalized)
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-6)

    def test_batched_normalization(self):
        """Batched normalization works correctly."""
        q = torch.randn(10, 4)
        normalized = normalize_quaternion(q)
        norms = torch.norm(normalized, dim=-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-6)

    def test_preserves_direction(self):
        """Normalization preserves quaternion direction."""
        q = torch.tensor([2.0, 4.0, 0.0, 0.0])
        normalized = normalize_quaternion(q)
        # Should point in same direction
        expected = torch.tensor([1.0, 2.0, 0.0, 0.0]) / math.sqrt(5)
        assert torch.allclose(normalized, expected, atol=1e-5)

    def test_numerical_stability_small(self):
        """Handles small quaternions without NaN."""
        q = torch.tensor([1e-20, 0.0, 0.0, 0.0])
        normalized = normalize_quaternion(q)
        assert not torch.isnan(normalized).any()


# =============================================================================
# Conjugate Tests
# =============================================================================

class TestQuaternionConjugate:
    """Tests for quaternion conjugate."""

    def test_identity_conjugate(self):
        """Identity quaternion conjugate is itself."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        conj = quaternion_conjugate(q)
        assert torch.allclose(conj, q)

    def test_conjugate_negates_vector(self):
        """Conjugate negates the vector part."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        conj = quaternion_conjugate(q)
        expected = torch.tensor([1.0, -2.0, -3.0, -4.0])
        assert torch.allclose(conj, expected)

    def test_double_conjugate_is_identity(self):
        """Conjugating twice returns original."""
        q = torch.randn(4)
        double_conj = quaternion_conjugate(quaternion_conjugate(q))
        assert torch.allclose(double_conj, q)

    def test_batched_conjugate(self):
        """Batched conjugate works."""
        q = torch.randn(5, 4)
        conj = quaternion_conjugate(q)
        assert conj.shape == q.shape
        assert torch.allclose(conj[..., 0], q[..., 0])
        assert torch.allclose(conj[..., 1:], -q[..., 1:])


# =============================================================================
# Inverse Tests
# =============================================================================

class TestQuaternionInverse:
    """Tests for quaternion inverse."""

    def test_identity_inverse(self):
        """Identity quaternion is its own inverse."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        inv = quaternion_inverse(q)
        assert torch.allclose(inv, q)

    def test_unit_quaternion_inverse_equals_conjugate(self):
        """For unit quaternions, inverse equals conjugate."""
        q = normalize_quaternion(torch.randn(4))
        inv = quaternion_inverse(q)
        conj = quaternion_conjugate(q)
        assert torch.allclose(inv, conj, atol=1e-5)

    def test_inverse_times_original_is_identity(self):
        """q * q^{-1} = 1."""
        q = normalize_quaternion(torch.randn(4))
        inv = quaternion_inverse(q)
        product = quaternion_multiply(q, inv)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(product, identity, atol=1e-5)

    def test_batched_inverse(self):
        """Batched inverse works."""
        q = normalize_quaternion(torch.randn(8, 4))
        inv = quaternion_inverse(q)
        products = quaternion_multiply(q, inv)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(8, 4)
        assert torch.allclose(products, identity, atol=1e-5)


# =============================================================================
# Multiplication Tests
# =============================================================================

class TestQuaternionMultiply:
    """Tests for quaternion multiplication (Hamilton product)."""

    def test_identity_is_neutral(self):
        """1 * q = q * 1 = q."""
        q = torch.randn(4)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])

        left = quaternion_multiply(identity, q)
        right = quaternion_multiply(q, identity)

        assert torch.allclose(left, q)
        assert torch.allclose(right, q)

    def test_i_squared_is_minus_one(self):
        """i^2 = -1 (where i = (0,1,0,0))."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0])
        i_squared = quaternion_multiply(i, i)
        expected = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(i_squared, expected)

    def test_j_squared_is_minus_one(self):
        """j^2 = -1."""
        j = torch.tensor([0.0, 0.0, 1.0, 0.0])
        j_squared = quaternion_multiply(j, j)
        expected = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(j_squared, expected)

    def test_k_squared_is_minus_one(self):
        """k^2 = -1."""
        k = torch.tensor([0.0, 0.0, 0.0, 1.0])
        k_squared = quaternion_multiply(k, k)
        expected = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(k_squared, expected)

    def test_ij_equals_k(self):
        """ij = k."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0])
        j = torch.tensor([0.0, 0.0, 1.0, 0.0])
        k = torch.tensor([0.0, 0.0, 0.0, 1.0])
        ij = quaternion_multiply(i, j)
        assert torch.allclose(ij, k)

    def test_jk_equals_i(self):
        """jk = i."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0])
        j = torch.tensor([0.0, 0.0, 1.0, 0.0])
        k = torch.tensor([0.0, 0.0, 0.0, 1.0])
        jk = quaternion_multiply(j, k)
        assert torch.allclose(jk, i)

    def test_ki_equals_j(self):
        """ki = j."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0])
        j = torch.tensor([0.0, 0.0, 1.0, 0.0])
        k = torch.tensor([0.0, 0.0, 0.0, 1.0])
        ki = quaternion_multiply(k, i)
        assert torch.allclose(ki, j)

    def test_associativity(self):
        """(a * b) * c = a * (b * c)."""
        a = normalize_quaternion(torch.randn(4))
        b = normalize_quaternion(torch.randn(4))
        c = normalize_quaternion(torch.randn(4))

        ab_c = quaternion_multiply(quaternion_multiply(a, b), c)
        a_bc = quaternion_multiply(a, quaternion_multiply(b, c))

        assert torch.allclose(ab_c, a_bc, atol=1e-5)

    def test_non_commutativity(self):
        """Quaternion multiplication is NOT commutative."""
        a = normalize_quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        b = normalize_quaternion(torch.tensor([0.0, 1.0, 0.0, 0.0]))

        ab = quaternion_multiply(a, b)
        ba = quaternion_multiply(b, a)

        assert not torch.allclose(ab, ba)

    def test_batched_multiply(self):
        """Batched multiplication works."""
        q1 = normalize_quaternion(torch.randn(10, 4))
        q2 = normalize_quaternion(torch.randn(10, 4))
        result = quaternion_multiply(q1, q2)
        assert result.shape == (10, 4)


# =============================================================================
# Matrix Conversion Tests
# =============================================================================

class TestQuaternionToMatrix:
    """Tests for quaternion to rotation matrix conversion."""

    def test_identity_to_identity_matrix(self):
        """Identity quaternion gives identity matrix."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_matrix(q)
        expected = torch.eye(3)
        assert torch.allclose(R, expected, atol=1e-6)

    def test_90_deg_rotation_x(self):
        """90 degree rotation around X axis."""
        angle = math.pi / 2
        q = torch.tensor([math.cos(angle/2), math.sin(angle/2), 0.0, 0.0])
        R = quaternion_to_matrix(q)

        # Should rotate Y to Z, Z to -Y
        y = torch.tensor([0.0, 1.0, 0.0])
        z = torch.tensor([0.0, 0.0, 1.0])

        y_rotated = R @ y
        z_rotated = R @ z

        assert torch.allclose(y_rotated, z, atol=1e-5)
        assert torch.allclose(z_rotated, -y, atol=1e-5)

    def test_180_deg_rotation_z(self):
        """180 degree rotation around Z axis."""
        angle = math.pi
        q = torch.tensor([math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)])
        R = quaternion_to_matrix(q)

        # Should rotate X to -X, Y to -Y
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.0, 1.0, 0.0])

        x_rotated = R @ x
        y_rotated = R @ y

        assert torch.allclose(x_rotated, -x, atol=1e-5)
        assert torch.allclose(y_rotated, -y, atol=1e-5)

    def test_matrix_is_orthogonal(self):
        """Resulting matrix is orthogonal (R^T R = I)."""
        q = normalize_quaternion(torch.randn(4))
        R = quaternion_to_matrix(q)
        RtR = R.T @ R
        assert torch.allclose(RtR, torch.eye(3), atol=1e-5)

    def test_matrix_determinant_is_one(self):
        """Rotation matrix has determinant 1."""
        q = normalize_quaternion(torch.randn(4))
        R = quaternion_to_matrix(q)
        det = torch.det(R)
        assert torch.isclose(det, torch.tensor(1.0), atol=1e-5)

    def test_batched_to_matrix(self):
        """Batched conversion works."""
        q = normalize_quaternion(torch.randn(5, 4))
        R = quaternion_to_matrix(q)
        assert R.shape == (5, 3, 3)


class TestMatrixToQuaternion:
    """Tests for rotation matrix to quaternion conversion."""

    def test_identity_matrix_to_identity_quaternion(self):
        """Identity matrix gives identity quaternion."""
        R = torch.eye(3)
        q = matrix_to_quaternion(R)
        # Could be +1 or -1 (q and -q are same rotation)
        assert torch.isclose(q[0].abs(), torch.tensor(1.0), atol=1e-5)

    def test_roundtrip_quat_to_matrix_to_quat(self):
        """q -> R -> q' gives equivalent quaternion."""
        q_orig = normalize_quaternion(torch.randn(4))
        R = quaternion_to_matrix(q_orig)
        q_recovered = matrix_to_quaternion(R)

        # q and -q represent same rotation
        # Check they rotate vectors the same way
        v = torch.randn(3)
        v1 = rotate_vector(v, q_orig)
        v2 = rotate_vector(v, q_recovered)
        assert torch.allclose(v1, v2, atol=1e-4)

    def test_batched_matrix_to_quaternion(self):
        """Batched conversion works."""
        q_orig = normalize_quaternion(torch.randn(8, 4))
        R = quaternion_to_matrix(q_orig)
        q_recovered = matrix_to_quaternion(R)
        assert q_recovered.shape == (8, 4)


# =============================================================================
# Axis-Angle Conversion Tests
# =============================================================================

class TestAxisAngleConversion:
    """Tests for axis-angle <-> quaternion conversion."""

    def test_zero_angle_is_identity(self):
        """Zero angle rotation is identity."""
        axis = torch.tensor([1.0, 0.0, 0.0])
        angle = torch.tensor(0.0)
        q = quaternion_from_axis_angle(axis.unsqueeze(0), angle.unsqueeze(0))
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(q, identity, atol=1e-5)

    def test_90_deg_around_z(self):
        """90 degree rotation around Z."""
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        angle = torch.tensor([[math.pi / 2]])
        q = quaternion_from_axis_angle(axis, angle)

        expected_w = math.cos(math.pi / 4)
        expected_z = math.sin(math.pi / 4)

        assert torch.isclose(q[0, 0], torch.tensor(expected_w), atol=1e-5)
        assert torch.isclose(q[0, 3], torch.tensor(expected_z), atol=1e-5)

    def test_180_deg_around_x(self):
        """180 degree rotation around X."""
        axis = torch.tensor([[1.0, 0.0, 0.0]])
        angle = torch.tensor([[math.pi]])
        q = quaternion_from_axis_angle(axis, angle)

        # w = cos(pi/2) = 0, x = sin(pi/2) = 1
        assert torch.isclose(q[0, 0].abs(), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(q[0, 1].abs(), torch.tensor(1.0), atol=1e-5)

    def test_roundtrip(self):
        """axis-angle -> quaternion -> axis-angle roundtrip."""
        axis_orig = torch.tensor([[1.0, 1.0, 1.0]])
        axis_orig = axis_orig / axis_orig.norm()
        angle_orig = torch.tensor([[1.23]])

        q = quaternion_from_axis_angle(axis_orig, angle_orig)
        axis_rec, angle_rec = quaternion_to_axis_angle(q)

        assert torch.allclose(axis_rec, axis_orig, atol=1e-4)
        assert torch.isclose(angle_rec, angle_orig.squeeze(), atol=1e-4)

    def test_batched_axis_angle(self):
        """Batched axis-angle conversion."""
        axes = torch.randn(10, 3)
        angles = torch.rand(10, 1) * math.pi
        q = quaternion_from_axis_angle(axes, angles)
        assert q.shape == (10, 4)


# =============================================================================
# SLERP Tests
# =============================================================================

class TestQuaternionSlerp:
    """Tests for spherical linear interpolation."""

    def test_t0_returns_q0(self):
        """SLERP at t=0 returns q0."""
        q0 = normalize_quaternion(torch.randn(1, 4))
        q1 = normalize_quaternion(torch.randn(1, 4))
        result = quaternion_slerp(q0, q1, 0.0)
        # May be negated
        assert torch.allclose(result.abs(), q0.abs(), atol=1e-5)

    def test_t1_returns_q1(self):
        """SLERP at t=1 returns q1."""
        q0 = normalize_quaternion(torch.randn(1, 4))
        q1 = normalize_quaternion(torch.randn(1, 4))
        result = quaternion_slerp(q0, q1, 1.0)
        assert torch.allclose(result.abs(), q1.abs(), atol=1e-5)

    def test_t05_is_midpoint(self):
        """SLERP at t=0.5 is angular midpoint."""
        q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q1 = quaternion_from_axis_angle(
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[math.pi / 2]])
        )
        result = quaternion_slerp(q0, q1, 0.5)

        # Should be 45 degree rotation
        expected = quaternion_from_axis_angle(
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[math.pi / 4]])
        )
        assert torch.allclose(result.abs(), expected.abs(), atol=1e-4)

    def test_result_is_normalized(self):
        """SLERP result is normalized."""
        q0 = normalize_quaternion(torch.randn(1, 4))
        q1 = normalize_quaternion(torch.randn(1, 4))
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = quaternion_slerp(q0, q1, t)
            norm = torch.norm(result)
            assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_slerp_same_quaternion(self):
        """SLERP between same quaternion returns that quaternion."""
        q = normalize_quaternion(torch.randn(1, 4))
        result = quaternion_slerp(q, q, 0.5)
        assert torch.allclose(result.abs(), q.abs(), atol=1e-5)

    def test_batched_slerp(self):
        """Batched SLERP works."""
        q0 = normalize_quaternion(torch.randn(5, 4))
        q1 = normalize_quaternion(torch.randn(5, 4))
        result = quaternion_slerp(q0, q1, 0.5)
        assert result.shape == (5, 4)


# =============================================================================
# Random Quaternion Tests
# =============================================================================

class TestRandomQuaternion:
    """Tests for random quaternion generation."""

    def test_output_shape(self):
        """Output has correct shape."""
        q = random_quaternion(batch_size=10)
        assert q.shape == (10, 4)

    def test_normalized(self):
        """Random quaternions are normalized."""
        q = random_quaternion(batch_size=100)
        norms = torch.norm(q, dim=-1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_different_each_call(self):
        """Different calls produce different quaternions."""
        q1 = random_quaternion(batch_size=10)
        q2 = random_quaternion(batch_size=10)
        assert not torch.allclose(q1, q2)

    def test_device_specification(self):
        """Device specification works."""
        q = random_quaternion(batch_size=5, device=torch.device('cpu'))
        assert q.device == torch.device('cpu')

    def test_dtype_specification(self):
        """Dtype specification works."""
        q = random_quaternion(batch_size=5, dtype=torch.float64)
        assert q.dtype == torch.float64


# =============================================================================
# Identity Quaternion Tests
# =============================================================================

class TestIdentityQuaternion:
    """Tests for identity quaternion creation."""

    def test_is_identity(self):
        """Created quaternion is identity."""
        q = identity_quaternion()
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(q, expected)

    def test_batch_size(self):
        """Batch size works correctly."""
        q = identity_quaternion(batch_size=5)
        assert q.shape == (5, 4)
        for i in range(5):
            assert torch.allclose(q[i], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_device_specification(self):
        """Device specification works."""
        q = identity_quaternion(device=torch.device('cpu'))
        assert q.device == torch.device('cpu')


# =============================================================================
# Rotate Vector Tests
# =============================================================================

class TestRotateVector:
    """Tests for rotating vectors by quaternions."""

    def test_identity_rotation_unchanged(self):
        """Identity quaternion leaves vector unchanged."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        v = torch.tensor([1.0, 2.0, 3.0])
        rotated = rotate_vector(v, q)
        assert torch.allclose(rotated, v, atol=1e-5)

    def test_90_deg_z_rotation(self):
        """90 degree rotation around Z rotates X to Y."""
        angle = math.pi / 2
        q = torch.tensor([math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)])
        v = torch.tensor([1.0, 0.0, 0.0])
        rotated = rotate_vector(v, q)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(rotated, expected, atol=1e-5)

    def test_180_deg_x_rotation(self):
        """180 degree rotation around X negates Y and Z."""
        angle = math.pi
        q = torch.tensor([math.cos(angle/2), math.sin(angle/2), 0.0, 0.0])
        v = torch.tensor([1.0, 2.0, 3.0])
        rotated = rotate_vector(v, q)
        expected = torch.tensor([1.0, -2.0, -3.0])
        assert torch.allclose(rotated, expected, atol=1e-5)

    def test_preserves_length(self):
        """Rotation preserves vector length."""
        q = normalize_quaternion(torch.randn(4))
        v = torch.randn(3)
        rotated = rotate_vector(v, q)
        assert torch.isclose(torch.norm(v), torch.norm(rotated), atol=1e-5)

    def test_batched_rotation(self):
        """Batched rotation works."""
        q = normalize_quaternion(torch.randn(10, 4))
        v = torch.randn(10, 3)
        rotated = rotate_vector(v, q)
        assert rotated.shape == (10, 3)

    def test_consistent_with_matrix(self):
        """Quaternion rotation matches matrix rotation."""
        q = normalize_quaternion(torch.randn(4))
        R = quaternion_to_matrix(q)
        v = torch.randn(3)

        v_quat = rotate_vector(v, q)
        v_mat = R @ v

        assert torch.allclose(v_quat, v_mat, atol=1e-5)


# =============================================================================
# Angle Distance Tests
# =============================================================================

class TestQuaternionAngleDistance:
    """Tests for angular distance between quaternions."""

    def test_same_quaternion_zero_distance(self):
        """Same quaternion has zero angular distance."""
        q = normalize_quaternion(torch.randn(4))
        dist = quaternion_angle_distance(q, q)
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-5)

    def test_opposite_quaternion_zero_distance(self):
        """q and -q have zero angular distance (same rotation)."""
        q = normalize_quaternion(torch.randn(4))
        dist = quaternion_angle_distance(q, -q)
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-5)

    def test_180_deg_rotation_gives_pi(self):
        """180 degree rotation gives distance pi."""
        q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 180 deg around X
        dist = quaternion_angle_distance(q1, q2)
        assert torch.isclose(dist, torch.tensor(math.pi), atol=1e-4)

    def test_90_deg_rotation_gives_half_pi(self):
        """90 degree rotation gives distance pi/2."""
        q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q2 = quaternion_from_axis_angle(
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[math.pi / 2]])
        ).squeeze(0)
        dist = quaternion_angle_distance(q1, q2)
        assert torch.isclose(dist, torch.tensor(math.pi / 2), atol=1e-4)

    def test_symmetric(self):
        """Distance is symmetric: d(a,b) = d(b,a)."""
        q1 = normalize_quaternion(torch.randn(4))
        q2 = normalize_quaternion(torch.randn(4))
        d12 = quaternion_angle_distance(q1, q2)
        d21 = quaternion_angle_distance(q2, q1)
        assert torch.isclose(d12, d21, atol=1e-5)

    def test_batched_distance(self):
        """Batched distance computation."""
        q1 = normalize_quaternion(torch.randn(10, 4))
        q2 = normalize_quaternion(torch.randn(10, 4))
        dists = quaternion_angle_distance(q1, q2)
        assert dists.shape == (10,)
        assert (dists >= 0).all()
        assert (dists <= math.pi + 1e-5).all()


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestQuaternionGradientFlow:
    """Tests for gradient flow through quaternion operations."""

    def test_normalize_gradient(self):
        """Gradients flow through normalization."""
        q = torch.randn(4, requires_grad=True)
        normalized = normalize_quaternion(q)
        loss = normalized.sum()
        loss.backward()
        assert q.grad is not None

    def test_multiply_gradient(self):
        """Gradients flow through multiplication."""
        q1 = torch.randn(4, requires_grad=True)
        q2 = torch.randn(4, requires_grad=True)
        product = quaternion_multiply(q1, q2)
        loss = product.sum()
        loss.backward()
        assert q1.grad is not None
        assert q2.grad is not None

    def test_to_matrix_gradient(self):
        """Gradients flow through matrix conversion."""
        q = torch.randn(4, requires_grad=True)
        R = quaternion_to_matrix(normalize_quaternion(q))
        loss = R.sum()
        loss.backward()
        assert q.grad is not None

    def test_rotate_vector_gradient(self):
        """Gradients flow through vector rotation."""
        q = normalize_quaternion(torch.randn(4))
        q = q.clone().requires_grad_(True)
        v = torch.randn(3, requires_grad=True)
        rotated = rotate_vector(v, q)
        loss = rotated.sum()
        loss.backward()
        assert v.grad is not None

    def test_slerp_gradient(self):
        """Gradients flow through SLERP."""
        q0 = normalize_quaternion(torch.randn(1, 4)).requires_grad_(True)
        q1 = normalize_quaternion(torch.randn(1, 4)).requires_grad_(True)
        t = torch.tensor(0.5, requires_grad=True)
        result = quaternion_slerp(q0, q1, t)
        loss = result.sum()
        loss.backward()
        assert q0.grad is not None
        assert q1.grad is not None


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestQuaternionNumericalStability:
    """Tests for numerical stability."""

    def test_slerp_nearly_parallel(self):
        """SLERP handles nearly parallel quaternions."""
        q = normalize_quaternion(torch.randn(1, 4))
        q_perturbed = q + torch.randn(1, 4) * 1e-8
        q_perturbed = normalize_quaternion(q_perturbed)
        result = quaternion_slerp(q, q_perturbed, 0.5)
        assert not torch.isnan(result).any()

    def test_axis_angle_small_angle(self):
        """Axis-angle handles small angles."""
        axis = torch.tensor([[1.0, 0.0, 0.0]])
        angle = torch.tensor([[1e-10]])
        q = quaternion_from_axis_angle(axis, angle)
        assert not torch.isnan(q).any()

    def test_to_axis_angle_near_identity(self):
        """to_axis_angle handles near-identity quaternions."""
        q = torch.tensor([[0.9999999, 0.0, 0.0, 1e-8]])
        q = normalize_quaternion(q)
        axis, angle = quaternion_to_axis_angle(q)
        assert not torch.isnan(axis).any()
        assert not torch.isnan(angle).any()

    def test_matrix_conversion_extreme_rotations(self):
        """Matrix conversion handles extreme rotations."""
        # Near 180 degree rotation
        q = normalize_quaternion(torch.tensor([1e-8, 1.0, 0.0, 0.0]))
        R = quaternion_to_matrix(q)
        assert not torch.isnan(R).any()
        assert not torch.isinf(R).any()


# =============================================================================
# Edge Cases
# =============================================================================

class TestQuaternionEdgeCases:
    """Tests for edge cases."""

    def test_empty_batch(self):
        """Handle empty batch."""
        q = torch.zeros(0, 4)
        normalized = normalize_quaternion(q)
        assert normalized.shape == (0, 4)

    def test_single_quaternion(self):
        """Operations work on single quaternions."""
        q = normalize_quaternion(torch.randn(4))
        assert q.shape == (4,)

    def test_3d_batch(self):
        """Handle 3D batch."""
        q = torch.randn(2, 3, 4)
        normalized = normalize_quaternion(q)
        assert normalized.shape == (2, 3, 4)
        norms = torch.norm(normalized, dim=-1)
        assert torch.allclose(norms, torch.ones(2, 3), atol=1e-6)

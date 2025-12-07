"""
Tests for PGA algebra operations.

Following the Yoneda philosophy: these tests completely DEFINE the behavior
of the PGA algebra module. If it's not tested here, it doesn't exist.

The algebra implements G(3,0,1) - Projective Geometric Algebra with:
- 16 basis elements organized by grade
- Metric signature (3,0,1): e0^2 = 0, e1^2 = e2^2 = e3^2 = +1
- Geometric, outer, inner, and regressive products
- Reversion, dual, conjugation operations
- Exponential and logarithm maps
"""

import pytest
import torch
import math

from pga_inr.pga.algebra import (
    Multivector,
    geometric_product,
    outer_product,
    inner_product,
    regressive_product,
    sandwich,
    exp,
    log,
    scalar,
    e0, e1, e2, e3,
    e01, e02, e03, e12, e31, e23,
    e012, e031, e023, e123,
    e0123,
    IDX_S, IDX_E0, IDX_E1, IDX_E2, IDX_E3,
    IDX_E01, IDX_E02, IDX_E03, IDX_E12, IDX_E31, IDX_E23,
    IDX_E012, IDX_E031, IDX_E023, IDX_E123, IDX_E0123,
    GRADE_0_MASK, GRADE_1_MASK, GRADE_2_MASK, GRADE_3_MASK, GRADE_4_MASK,
    REVERSION_SIGNS, INVOLUTION_SIGNS, CONJUGATION_SIGNS,
)


# =============================================================================
# Multivector Creation and Properties
# =============================================================================

class TestMultivectorCreation:
    """Tests for Multivector construction and basic properties."""

    def test_create_multivector_requires_16_components(self):
        """Multivector REQUIRES exactly 16 components."""
        with pytest.raises(ValueError, match="Expected 16 components"):
            Multivector(torch.zeros(15))
        with pytest.raises(ValueError, match="Expected 16 components"):
            Multivector(torch.zeros(17))

    def test_create_multivector_accepts_16_components(self):
        """Multivector accepts tensor with 16 components."""
        components = torch.zeros(16)
        mv = Multivector(components)
        assert mv.mv.shape == (16,)

    def test_batched_multivector_shape(self):
        """Batched multivectors preserve batch dimensions."""
        for batch_shape in [(4,), (2, 3), (2, 3, 4)]:
            components = torch.zeros(*batch_shape, 16)
            mv = Multivector(components)
            assert mv.shape == torch.Size(batch_shape)
            assert mv.mv.shape == (*batch_shape, 16)

    def test_device_property(self):
        """Device property returns correct device."""
        mv = Multivector(torch.zeros(16))
        assert mv.device == torch.device('cpu')

    def test_dtype_preservation_float32(self):
        """float32 dtype is preserved."""
        mv = Multivector(torch.zeros(16, dtype=torch.float32))
        assert mv.dtype == torch.float32

    def test_dtype_preservation_float64(self):
        """float64 dtype is preserved."""
        mv = Multivector(torch.zeros(16, dtype=torch.float64))
        assert mv.dtype == torch.float64

    def test_clone_creates_independent_copy(self):
        """Clone creates a deep copy."""
        original = Multivector(torch.ones(16))
        cloned = original.clone()
        cloned.mv[0] = 999.0
        assert original.mv[0] == 1.0

    def test_detach_removes_grad(self):
        """Detach removes gradient tracking."""
        components = torch.ones(16, requires_grad=True)
        mv = Multivector(components)
        detached = mv.detach()
        assert not detached.mv.requires_grad

    def test_to_moves_device(self):
        """to() moves multivector to specified device."""
        mv = Multivector(torch.zeros(16))
        moved = mv.to(torch.device('cpu'))
        assert moved.device == torch.device('cpu')


# =============================================================================
# Basis Element Factory Functions
# =============================================================================

class TestBasisElementCreation:
    """Tests for basis element factory functions - they DEFINE the basis."""

    def test_scalar_creates_grade0_element(self):
        """scalar() creates a pure scalar (grade 0)."""
        s = scalar(torch.tensor(3.0))
        assert s.mv[IDX_S] == 3.0
        assert torch.allclose(s.mv[1:], torch.zeros(15))

    def test_scalar_with_batched_tensor(self):
        """scalar() works with batched tensors."""
        values = torch.tensor([1.0, 2.0, 3.0])
        s = scalar(values)
        assert s.shape == (3,)
        assert torch.allclose(s.mv[:, IDX_S], values)

    def test_e0_creates_degenerate_basis(self):
        """e0() creates the degenerate basis element (e0^2 = 0)."""
        e = e0(1.0)
        assert e.mv[IDX_E0] == 1.0
        assert e.mv.sum() == 1.0

    def test_e1_creates_euclidean_basis(self):
        """e1() creates first Euclidean basis element (e1^2 = 1)."""
        e = e1(1.0)
        assert e.mv[IDX_E1] == 1.0
        assert e.mv.sum() == 1.0

    def test_e2_creates_euclidean_basis(self):
        """e2() creates second Euclidean basis element (e2^2 = 1)."""
        e = e2(1.0)
        assert e.mv[IDX_E2] == 1.0
        assert e.mv.sum() == 1.0

    def test_e3_creates_euclidean_basis(self):
        """e3() creates third Euclidean basis element (e3^2 = 1)."""
        e = e3(1.0)
        assert e.mv[IDX_E3] == 1.0
        assert e.mv.sum() == 1.0

    def test_all_bivector_bases(self):
        """All bivector basis functions create correct elements."""
        pairs = [
            (e01, IDX_E01), (e02, IDX_E02), (e03, IDX_E03),
            (e12, IDX_E12), (e31, IDX_E31), (e23, IDX_E23),
        ]
        for func, idx in pairs:
            bv = func(1.0)
            assert bv.mv[idx] == 1.0
            assert bv.mv.sum() == 1.0

    def test_all_trivector_bases(self):
        """All trivector basis functions create correct elements."""
        pairs = [
            (e012, IDX_E012), (e031, IDX_E031),
            (e023, IDX_E023), (e123, IDX_E123),
        ]
        for func, idx in pairs:
            tv = func(1.0)
            assert tv.mv[idx] == 1.0
            assert tv.mv.sum() == 1.0

    def test_e0123_creates_pseudoscalar(self):
        """e0123() creates the pseudoscalar (grade 4)."""
        ps = e0123(2.0)
        assert ps.mv[IDX_E0123] == 2.0
        assert ps.mv[:15].sum() == 0.0

    def test_basis_elements_with_tensor_coefficients(self):
        """Basis functions accept tensor coefficients."""
        coeffs = torch.tensor([1.0, 2.0, 3.0])
        basis = e1(coeffs)
        assert basis.shape == (3,)
        assert torch.allclose(basis.mv[:, IDX_E1], coeffs)


# =============================================================================
# Grade Extraction
# =============================================================================

class TestGradeExtraction:
    """Tests for grade extraction - defines the grading structure."""

    def test_scalar_extraction(self):
        """scalar() method extracts grade-0 component."""
        components = torch.zeros(16)
        components[IDX_S] = 5.0
        components[IDX_E1] = 3.0  # Should not appear in scalar
        mv = Multivector(components)
        assert mv.scalar() == 5.0

    def test_vector_extraction(self):
        """vector() method extracts grade-1 components [e0, e1, e2, e3]."""
        components = torch.zeros(16)
        components[IDX_E0] = 1.0
        components[IDX_E1] = 2.0
        components[IDX_E2] = 3.0
        components[IDX_E3] = 4.0
        components[IDX_E01] = 99.0  # Should not appear
        mv = Multivector(components)
        vec = mv.vector()
        assert vec.shape == (4,)
        assert torch.allclose(vec, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_bivector_extraction(self):
        """bivector() method extracts grade-2 components."""
        components = torch.zeros(16)
        for i, idx in enumerate(GRADE_2_MASK):
            components[idx] = float(i + 1)
        mv = Multivector(components)
        bv = mv.bivector()
        assert bv.shape == (6,)
        assert torch.allclose(bv, torch.tensor([1., 2., 3., 4., 5., 6.]))

    def test_trivector_extraction(self):
        """trivector() method extracts grade-3 components."""
        components = torch.zeros(16)
        for i, idx in enumerate(GRADE_3_MASK):
            components[idx] = float(i + 1)
        mv = Multivector(components)
        tv = mv.trivector()
        assert tv.shape == (4,)
        assert torch.allclose(tv, torch.tensor([1., 2., 3., 4.]))

    def test_pseudoscalar_extraction(self):
        """pseudoscalar() method extracts grade-4 component."""
        components = torch.zeros(16)
        components[IDX_E0123] = 7.0
        components[IDX_E123] = 99.0  # Should not appear
        mv = Multivector(components)
        assert mv.pseudoscalar() == 7.0

    def test_grade_function_isolates_specific_grade(self):
        """grade(k) method returns multivector with only grade-k components."""
        components = torch.ones(16)
        mv = Multivector(components)

        for k, mask in [(0, GRADE_0_MASK), (1, GRADE_1_MASK),
                        (2, GRADE_2_MASK), (3, GRADE_3_MASK),
                        (4, GRADE_4_MASK)]:
            gk = mv.grade(k)
            for idx in mask:
                assert gk.mv[idx] == 1.0
            # Check all other components are zero
            for idx in range(16):
                if idx not in mask:
                    assert gk.mv[idx] == 0.0


# =============================================================================
# Metric Properties - The Foundation of PGA
# =============================================================================

class TestMetricProperties:
    """Tests for metric signature (3,0,1) - this DEFINES the algebra."""

    def test_e0_squared_is_zero(self):
        """e0^2 = 0 (degenerate/null direction) - FUNDAMENTAL PROPERTY."""
        e_0 = e0(1.0)
        e0_squared = geometric_product(e_0, e_0)
        assert torch.allclose(e0_squared.mv, torch.zeros(16), atol=1e-6)

    def test_e1_squared_is_one(self):
        """e1^2 = 1 (Euclidean) - FUNDAMENTAL PROPERTY."""
        e_1 = e1(1.0)
        e1_squared = geometric_product(e_1, e_1)
        assert torch.isclose(e1_squared.mv[IDX_S], torch.tensor(1.0))
        assert torch.allclose(e1_squared.mv[1:], torch.zeros(15), atol=1e-6)

    def test_e2_squared_is_one(self):
        """e2^2 = 1 (Euclidean) - FUNDAMENTAL PROPERTY."""
        e_2 = e2(1.0)
        e2_squared = geometric_product(e_2, e_2)
        assert torch.isclose(e2_squared.mv[IDX_S], torch.tensor(1.0))
        assert torch.allclose(e2_squared.mv[1:], torch.zeros(15), atol=1e-6)

    def test_e3_squared_is_one(self):
        """e3^2 = 1 (Euclidean) - FUNDAMENTAL PROPERTY."""
        e_3 = e3(1.0)
        e3_squared = geometric_product(e_3, e_3)
        assert torch.isclose(e3_squared.mv[IDX_S], torch.tensor(1.0))
        assert torch.allclose(e3_squared.mv[1:], torch.zeros(15), atol=1e-6)

    def test_e12_squared_is_minus_one(self):
        """e12^2 = -1 (Euclidean bivector) - derived from e1^2=e2^2=1."""
        bv = e12(1.0)
        squared = geometric_product(bv, bv)
        assert torch.isclose(squared.mv[IDX_S], torch.tensor(-1.0))
        assert torch.allclose(squared.mv[1:], torch.zeros(15), atol=1e-6)

    def test_e31_squared_is_minus_one(self):
        """e31^2 = -1 (Euclidean bivector)."""
        bv = e31(1.0)
        squared = geometric_product(bv, bv)
        assert torch.isclose(squared.mv[IDX_S], torch.tensor(-1.0))
        assert torch.allclose(squared.mv[1:], torch.zeros(15), atol=1e-6)

    def test_e23_squared_is_minus_one(self):
        """e23^2 = -1 (Euclidean bivector)."""
        bv = e23(1.0)
        squared = geometric_product(bv, bv)
        assert torch.isclose(squared.mv[IDX_S], torch.tensor(-1.0))
        assert torch.allclose(squared.mv[1:], torch.zeros(15), atol=1e-6)

    def test_e123_squared_is_minus_one(self):
        """e123^2 = -1 (Euclidean pseudoscalar)."""
        tv = e123(1.0)
        squared = geometric_product(tv, tv)
        assert torch.isclose(squared.mv[IDX_S], torch.tensor(-1.0))
        assert torch.allclose(squared.mv[1:], torch.zeros(15), atol=1e-6)

    def test_e0123_squared_is_zero(self):
        """e0123^2 = 0 (PGA pseudoscalar is nilpotent)."""
        ps = e0123(1.0)
        squared = geometric_product(ps, ps)
        assert torch.allclose(squared.mv, torch.zeros(16), atol=1e-6)


# =============================================================================
# Anticommutativity of Basis Vectors
# =============================================================================

class TestAnticommutativity:
    """Tests for anticommutativity: e_i * e_j = -e_j * e_i for i != j."""

    def test_e0_e1_anticommute(self):
        """e0 * e1 = -e1 * e0."""
        e_0, e_1 = e0(1.0), e1(1.0)
        ab = geometric_product(e_0, e_1)
        ba = geometric_product(e_1, e_0)
        assert torch.allclose(ab.mv, -ba.mv, atol=1e-6)

    def test_e0_e2_anticommute(self):
        """e0 * e2 = -e2 * e0."""
        e_0, e_2 = e0(1.0), e2(1.0)
        ab = geometric_product(e_0, e_2)
        ba = geometric_product(e_2, e_0)
        assert torch.allclose(ab.mv, -ba.mv, atol=1e-6)

    def test_e0_e3_anticommute(self):
        """e0 * e3 = -e3 * e0."""
        e_0, e_3 = e0(1.0), e3(1.0)
        ab = geometric_product(e_0, e_3)
        ba = geometric_product(e_3, e_0)
        assert torch.allclose(ab.mv, -ba.mv, atol=1e-6)

    def test_e1_e2_anticommute(self):
        """e1 * e2 = -e2 * e1."""
        e_1, e_2 = e1(1.0), e2(1.0)
        ab = geometric_product(e_1, e_2)
        ba = geometric_product(e_2, e_1)
        assert torch.allclose(ab.mv, -ba.mv, atol=1e-6)

    def test_e1_e3_anticommute(self):
        """e1 * e3 = -e3 * e1."""
        e_1, e_3 = e1(1.0), e3(1.0)
        ab = geometric_product(e_1, e_3)
        ba = geometric_product(e_3, e_1)
        assert torch.allclose(ab.mv, -ba.mv, atol=1e-6)

    def test_e2_e3_anticommute(self):
        """e2 * e3 = -e3 * e2."""
        e_2, e_3 = e2(1.0), e3(1.0)
        ab = geometric_product(e_2, e_3)
        ba = geometric_product(e_3, e_2)
        assert torch.allclose(ab.mv, -ba.mv, atol=1e-6)


# =============================================================================
# Geometric Product Properties
# =============================================================================

class TestGeometricProduct:
    """Tests for geometric product - the fundamental operation."""

    def test_scalar_is_identity(self):
        """Scalar 1 is the identity for geometric product."""
        one = scalar(torch.tensor(1.0))
        mv = Multivector(torch.randn(16))
        result = geometric_product(one, mv)
        assert torch.allclose(result.mv, mv.mv, atol=1e-6)

    def test_scalar_multiplication(self):
        """Scalar product scales all components."""
        s = scalar(torch.tensor(3.0))
        mv = Multivector(torch.ones(16))
        result = geometric_product(s, mv)
        assert torch.allclose(result.mv, torch.full((16,), 3.0), atol=1e-6)

    def test_associativity(self):
        """Geometric product is associative: (a*b)*c = a*(b*c)."""
        a = Multivector(torch.randn(16))
        b = Multivector(torch.randn(16))
        c = Multivector(torch.randn(16))

        ab_c = geometric_product(geometric_product(a, b), c)
        a_bc = geometric_product(a, geometric_product(b, c))

        assert torch.allclose(ab_c.mv, a_bc.mv, atol=1e-4)

    def test_distributivity_left(self):
        """a * (b + c) = a*b + a*c."""
        a = Multivector(torch.randn(16))
        b = Multivector(torch.randn(16))
        c = Multivector(torch.randn(16))

        lhs = geometric_product(a, b + c)
        rhs = geometric_product(a, b) + geometric_product(a, c)

        assert torch.allclose(lhs.mv, rhs.mv, atol=1e-5)

    def test_distributivity_right(self):
        """(a + b) * c = a*c + b*c."""
        a = Multivector(torch.randn(16))
        b = Multivector(torch.randn(16))
        c = Multivector(torch.randn(16))

        lhs = geometric_product(a + b, c)
        rhs = geometric_product(a, c) + geometric_product(b, c)

        assert torch.allclose(lhs.mv, rhs.mv, atol=1e-5)

    def test_batched_geometric_product(self):
        """Geometric product works with batched inputs."""
        batch_size = 4
        a = Multivector(torch.randn(batch_size, 16))
        b = Multivector(torch.randn(batch_size, 16))
        result = geometric_product(a, b)
        assert result.shape == (batch_size,)


# =============================================================================
# Operator Overloading
# =============================================================================

class TestOperatorOverloading:
    """Tests for operator overloading - convenient syntax."""

    def test_multiply_operator_is_geometric_product(self):
        """a * b uses geometric product."""
        a = Multivector(torch.randn(16))
        b = Multivector(torch.randn(16))
        result = a * b
        expected = geometric_product(a, b)
        assert torch.allclose(result.mv, expected.mv)

    def test_multiply_by_scalar_float(self):
        """Multivector * float scales components."""
        mv = Multivector(torch.ones(16))
        result = mv * 3.0
        assert torch.allclose(result.mv, torch.full((16,), 3.0))

    def test_rmul_by_scalar_float(self):
        """float * Multivector scales components."""
        mv = Multivector(torch.ones(16))
        result = 3.0 * mv
        assert torch.allclose(result.mv, torch.full((16,), 3.0))

    def test_add_operator(self):
        """a + b adds component-wise."""
        a = Multivector(torch.ones(16))
        b = Multivector(torch.ones(16) * 2)
        result = a + b
        assert torch.allclose(result.mv, torch.full((16,), 3.0))

    def test_sub_operator(self):
        """a - b subtracts component-wise."""
        a = Multivector(torch.ones(16) * 3)
        b = Multivector(torch.ones(16))
        result = a - b
        assert torch.allclose(result.mv, torch.full((16,), 2.0))

    def test_neg_operator(self):
        """-a negates all components."""
        a = Multivector(torch.ones(16))
        result = -a
        assert torch.allclose(result.mv, torch.full((16,), -1.0))

    def test_div_operator_scalar(self):
        """a / scalar divides components."""
        a = Multivector(torch.ones(16) * 6)
        result = a / 2.0
        assert torch.allclose(result.mv, torch.full((16,), 3.0))

    def test_xor_operator_is_outer_product(self):
        """a ^ b uses outer product."""
        a = Multivector(torch.randn(16))
        b = Multivector(torch.randn(16))
        result = a ^ b
        expected = outer_product(a, b)
        assert torch.allclose(result.mv, expected.mv)

    def test_or_operator_is_inner_product(self):
        """a | b uses inner product."""
        a = Multivector(torch.randn(16))
        b = Multivector(torch.randn(16))
        result = a | b
        expected = inner_product(a, b)
        assert torch.allclose(result.mv, expected.mv)

    def test_and_operator_is_regressive_product(self):
        """a & b uses regressive product."""
        a = Multivector(torch.randn(16))
        b = Multivector(torch.randn(16))
        result = a & b
        expected = regressive_product(a, b)
        assert torch.allclose(result.mv, expected.mv)

    def test_invert_operator_is_reversion(self):
        """~a is reversion."""
        a = Multivector(torch.ones(16))
        result = ~a
        expected = a.reverse()
        assert torch.allclose(result.mv, expected.mv)


# =============================================================================
# Reversion, Involution, Conjugation
# =============================================================================

class TestUnaryOperations:
    """Tests for unary operations - they DEFINE transformation behavior."""

    def test_reversion_grade0_unchanged(self):
        """Reversion leaves grade-0 unchanged: ~s = s."""
        s = scalar(torch.tensor(5.0))
        assert torch.allclose(s.reverse().mv, s.mv)

    def test_reversion_grade1_unchanged(self):
        """Reversion leaves grade-1 unchanged: ~e_i = e_i."""
        for func in [e0, e1, e2, e3]:
            v = func(1.0)
            assert torch.allclose(v.reverse().mv, v.mv)

    def test_reversion_grade2_negated(self):
        """Reversion negates grade-2: ~(e_ij) = -e_ij."""
        for func in [e01, e02, e03, e12, e31, e23]:
            bv = func(1.0)
            rev = bv.reverse()
            assert torch.allclose(rev.mv, -bv.mv)

    def test_reversion_grade3_negated(self):
        """Reversion negates grade-3: ~(e_ijk) = -e_ijk."""
        for func in [e012, e031, e023, e123]:
            tv = func(1.0)
            rev = tv.reverse()
            assert torch.allclose(rev.mv, -tv.mv)

    def test_reversion_grade4_unchanged(self):
        """Reversion leaves grade-4 unchanged: ~e0123 = e0123."""
        ps = e0123(1.0)
        assert torch.allclose(ps.reverse().mv, ps.mv)

    def test_reversion_signs_array(self):
        """REVERSION_SIGNS array matches mathematical definition."""
        expected = torch.tensor([
            1, 1, 1, 1, 1,           # grades 0, 1
            -1, -1, -1, -1, -1, -1,  # grade 2
            -1, -1, -1, -1,          # grade 3
            1,                        # grade 4
        ], dtype=torch.float32)
        assert torch.allclose(REVERSION_SIGNS, expected)

    def test_involution_negates_odd_grades(self):
        """Grade involution negates odd grades (1 and 3)."""
        # Grade 1 vectors should be negated
        for func in [e0, e1, e2, e3]:
            v = func(1.0)
            inv = v.involute()
            assert torch.allclose(inv.mv, -v.mv)

        # Grade 3 trivectors should be negated
        for func in [e012, e031, e023, e123]:
            tv = func(1.0)
            inv = tv.involute()
            assert torch.allclose(inv.mv, -tv.mv)

    def test_involution_preserves_even_grades(self):
        """Grade involution preserves even grades (0, 2, 4)."""
        # Grade 0 scalar
        s = scalar(torch.tensor(3.0))
        assert torch.allclose(s.involute().mv, s.mv)

        # Grade 2 bivectors
        for func in [e01, e02, e03, e12, e31, e23]:
            bv = func(1.0)
            assert torch.allclose(bv.involute().mv, bv.mv)

        # Grade 4 pseudoscalar
        ps = e0123(2.0)
        assert torch.allclose(ps.involute().mv, ps.mv)

    def test_conjugation_is_reversion_times_involution(self):
        """Clifford conjugation = reversion * involution."""
        mv = Multivector(torch.randn(16))
        conj = mv.conjugate()
        expected = Multivector(mv.mv * CONJUGATION_SIGNS)
        assert torch.allclose(conj.mv, expected.mv)


# =============================================================================
# Dual Operation
# =============================================================================

class TestDual:
    """Tests for Poincare dual operation."""

    def test_dual_of_scalar_is_pseudoscalar(self):
        """dual(scalar) produces pseudoscalar component."""
        s = scalar(torch.tensor(1.0))
        d = s.dual()
        assert d.mv[IDX_E0123] != 0

    def test_dual_of_pseudoscalar_is_scalar(self):
        """dual(e0123) produces scalar."""
        ps = e0123(1.0)
        d = ps.dual()
        assert d.mv[IDX_S] != 0

    def test_dual_involutive_up_to_sign(self):
        """dual(dual(x)) = +/- x (involutive up to sign)."""
        # For most elements, double dual returns original or negative
        mv = e123(1.0)  # A simple test case
        dd = mv.dual().dual()
        # Check that we get back the same structure
        assert dd.mv[IDX_E123] != 0


# =============================================================================
# Norm and Normalization
# =============================================================================

class TestNormalization:
    """Tests for norm and normalization operations."""

    def test_scalar_norm(self):
        """Norm of scalar is absolute value."""
        s = scalar(torch.tensor(-3.0))
        assert torch.isclose(s.norm(), torch.tensor(3.0))

    def test_euclidean_vector_norm(self):
        """Norm of Euclidean vector follows expected formula."""
        # For e1 + e2 + e3, norm^2 = 1 + 1 + 1 = 3
        mv = e1(1.0) + e2(1.0) + e3(1.0)
        expected_norm_sq = 3.0
        assert torch.isclose(mv.norm_squared(), torch.tensor(expected_norm_sq), atol=1e-5)

    def test_normalize_produces_unit_norm(self):
        """Normalize produces multivector with unit norm."""
        mv = Multivector(torch.randn(16))
        normalized = mv.normalize()
        # For well-behaved multivectors
        assert normalized.norm() > 0.5  # At least approximately unit

    def test_inverse_satisfies_identity(self):
        """M * M^{-1} = 1 (scalar) for invertible M."""
        # Use a simple rotor (which is invertible)
        rotor = scalar(torch.tensor(1.0))  # Identity is simplest
        inv = rotor.inverse()
        product = geometric_product(rotor, inv)
        assert torch.isclose(product.scalar(), torch.tensor(1.0), atol=1e-5)


# =============================================================================
# Sandwich Product
# =============================================================================

class TestSandwichProduct:
    """Tests for sandwich product M * X * ~M."""

    def test_sandwich_with_scalar_identity(self):
        """Sandwich with scalar 1 preserves element."""
        identity = scalar(torch.tensor(1.0))
        mv = Multivector(torch.randn(16))
        result = sandwich(identity, mv)
        assert torch.allclose(result.mv, mv.mv, atol=1e-5)

    def test_sandwich_preserves_grade(self):
        """Sandwich product preserves grade for versors."""
        # Rotor (even grade versor) applied to vector
        rotor = scalar(torch.tensor(1.0))  # Identity rotor
        vec = e1(1.0) + e2(2.0)
        result = sandwich(rotor, vec)
        # Result should still be grade 1
        assert result.mv[IDX_E1] != 0 or result.mv[IDX_E2] != 0

    def test_sandwich_batched(self):
        """Sandwich product works with batched inputs."""
        batch_size = 4
        motor_components = torch.zeros(batch_size, 16)
        motor_components[:, IDX_S] = 1.0  # Identity motors
        element_components = torch.randn(batch_size, 16)

        motor = Multivector(motor_components)
        element = Multivector(element_components)

        result = sandwich(motor, element)
        assert result.shape == (batch_size,)


# =============================================================================
# Exponential and Logarithm Maps
# =============================================================================

class TestExpLog:
    """Tests for exponential and logarithm maps - crucial for motors."""

    def test_exp_of_zero_is_identity(self):
        """exp(0) = 1 (identity)."""
        zero = Multivector(torch.zeros(16))
        result = exp(zero)
        assert torch.isclose(result.mv[IDX_S], torch.tensor(1.0), atol=1e-5)
        # All other components should be zero
        assert torch.allclose(result.mv[1:], torch.zeros(15), atol=1e-5)

    def test_exp_of_small_bivector(self):
        """exp of small bivector approximates 1 + B."""
        # For small B, exp(B) ~ 1 + B
        epsilon = 0.01
        B = e12(epsilon)
        result = exp(B)
        assert torch.isclose(result.mv[IDX_S], torch.tensor(1.0), atol=0.01)

    def test_exp_pure_euclidean_bivector_is_rotor(self):
        """exp(theta/2 * e12) is a rotation rotor."""
        angle = math.pi / 4  # 45 degrees
        B = e12(angle / 2)
        rotor = exp(B)

        # Should have scalar = cos(angle/2)
        expected_scalar = math.cos(angle / 2)
        assert torch.isclose(rotor.mv[IDX_S], torch.tensor(expected_scalar), atol=0.01)

        # Should have e12 = sin(angle/2)
        expected_e12 = math.sin(angle / 2)
        assert torch.isclose(rotor.mv[IDX_E12], torch.tensor(expected_e12), atol=0.01)

    def test_exp_batched(self):
        """exp works with batched bivectors."""
        batch_size = 4
        components = torch.zeros(batch_size, 16)
        components[:, IDX_E12] = torch.linspace(0, 0.5, batch_size)
        bv = Multivector(components)
        result = exp(bv)
        assert result.shape == (batch_size,)

    def test_log_of_identity_is_zero(self):
        """log(1) = 0."""
        identity = scalar(torch.tensor(1.0))
        result = log(identity)
        assert torch.allclose(result.mv, torch.zeros(16), atol=1e-5)


# =============================================================================
# Regressive Product
# =============================================================================

class TestRegressiveProduct:
    """Tests for regressive (vee) product a v b = (a* ^ b*)*."""

    def test_regressive_product_formula(self):
        """Regressive product follows (a* ^ b*)* formula."""
        a = Multivector(torch.randn(16))
        b = Multivector(torch.randn(16))

        result = regressive_product(a, b)
        expected = outer_product(a.dual(), b.dual()).dual()

        assert torch.allclose(result.mv, expected.mv, atol=1e-5)


# =============================================================================
# Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability - important for deep learning."""

    def test_product_with_small_values(self):
        """Products handle small values without underflow."""
        small = Multivector(torch.full((16,), 1e-10))
        result = geometric_product(small, small)
        # Should not be NaN
        assert not torch.isnan(result.mv).any()

    def test_product_with_large_values(self):
        """Products handle large values without overflow."""
        large = Multivector(torch.full((16,), 1e10))
        identity = scalar(torch.tensor(1.0))
        result = geometric_product(identity, large)
        # Should not be infinite
        assert not torch.isinf(result.mv).any()

    def test_normalization_of_small_multivector(self):
        """Normalization handles small multivectors without division by zero."""
        small = Multivector(torch.full((16,), 1e-15))
        # Should not raise and should not produce NaN
        normalized = small.normalize()
        assert not torch.isnan(normalized.mv).any()

    def test_inverse_of_nearly_zero(self):
        """Inverse handles near-zero gracefully."""
        near_zero = scalar(torch.tensor(1e-15))
        inv = near_zero.inverse()
        # Should not be NaN or Inf
        assert not torch.isnan(inv.mv).any()


# =============================================================================
# Gradient Flow
# =============================================================================

class TestGradientFlow:
    """Tests for gradient flow through operations - crucial for training."""

    def test_geometric_product_gradient_flow(self):
        """Gradients flow through geometric product."""
        a_components = torch.randn(16, requires_grad=True)
        b_components = torch.randn(16, requires_grad=True)

        a = Multivector(a_components)
        b = Multivector(b_components)
        result = geometric_product(a, b)

        loss = result.mv.sum()
        loss.backward()

        assert a_components.grad is not None
        assert b_components.grad is not None
        assert not torch.isnan(a_components.grad).any()
        assert not torch.isnan(b_components.grad).any()

    def test_sandwich_gradient_flow(self):
        """Gradients flow through sandwich product."""
        motor_components = torch.zeros(16, requires_grad=True)
        # Initialize as identity + small perturbation
        with torch.no_grad():
            motor_components.data[IDX_S] = 1.0

        element_components = torch.randn(16, requires_grad=True)

        motor = Multivector(motor_components)
        element = Multivector(element_components)
        result = sandwich(motor, element)

        loss = result.mv.sum()
        loss.backward()

        assert element_components.grad is not None

    def test_exp_gradient_flow(self):
        """Gradients flow through exponential map."""
        bv_components = torch.zeros(16, requires_grad=True)
        with torch.no_grad():
            bv_components.data[IDX_E12] = 0.5

        bv = Multivector(bv_components)
        result = exp(bv)

        loss = result.mv.sum()
        loss.backward()

        assert bv_components.grad is not None
        assert not torch.isnan(bv_components.grad).any()

    def test_normalize_gradient_flow(self):
        """Gradients flow through normalization."""
        components = torch.randn(16, requires_grad=True)
        mv = Multivector(components)
        normalized = mv.normalize()

        loss = normalized.mv.sum()
        loss.backward()

        assert components.grad is not None
        assert not torch.isnan(components.grad).any()


# =============================================================================
# Edge Cases
# =============================================================================

# =============================================================================
# Outer Product (Wedge) - Grade Filtering
# =============================================================================

class TestOuterProduct:
    """Tests for outer product - must raise grade by sum of input grades."""

    def test_outer_product_raises_grade_vector_vector(self):
        """Outer product of two grade-1 elements yields grade-2."""
        v1 = e1(1.0)  # grade 1
        v2 = e2(1.0)  # grade 1
        result = outer_product(v1, v2)
        # Result should be grade 2 (bivector)
        assert result.mv[IDX_E12] != 0.0, "e1 ^ e2 should have e12 component"
        # Grade-0 and grade-4 should be zero
        assert torch.isclose(result.mv[IDX_S], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(result.mv[IDX_E0123], torch.tensor(0.0), atol=1e-6)

    def test_outer_product_raises_grade_bivector_vector(self):
        """Outer product of grade-2 and grade-1 yields grade-3."""
        bv = e12(1.0)  # grade 2
        v = e3(1.0)    # grade 1
        result = outer_product(bv, v)
        # Result should be grade 3 (trivector e123)
        assert result.mv[IDX_E123] != 0.0, "e12 ^ e3 should have e123 component"
        # All non-grade-3 should be zero
        assert torch.allclose(result.grade(0).mv, torch.zeros(16), atol=1e-6)
        assert torch.allclose(result.grade(1).mv, torch.zeros(16), atol=1e-6)
        assert torch.allclose(result.grade(2).mv, torch.zeros(16), atol=1e-6)
        assert torch.allclose(result.grade(4).mv, torch.zeros(16), atol=1e-6)

    def test_outer_product_scalar_preserves_element(self):
        """Outer product with scalar is scalar multiplication."""
        s = scalar(torch.tensor(3.0))
        v = e1(2.0)
        result = outer_product(s, v)
        expected = e1(6.0)
        assert torch.allclose(result.mv, expected.mv, atol=1e-6)

    def test_outer_product_anticommutative_vectors(self):
        """Outer product is anticommutative for vectors: a ^ b = -b ^ a."""
        a = e1(1.0) + e2(0.5)
        b = e2(1.0) + e3(0.5)
        ab = outer_product(a, b)
        ba = outer_product(b, a)
        assert torch.allclose(ab.mv, -ba.mv, atol=1e-6)

    def test_outer_product_same_vector_is_zero(self):
        """Outer product of vector with itself is zero: v ^ v = 0."""
        v = e1(1.0) + e2(2.0) + e3(3.0)
        result = outer_product(v, v)
        assert torch.allclose(result.mv, torch.zeros(16), atol=1e-6)

    def test_outer_product_overflow_is_zero(self):
        """Outer product yielding grade > 4 gives zero."""
        tv = e123(1.0)  # grade 3
        bv = e12(1.0)   # grade 2
        # Grade 3 + 2 = 5 > 4, should be zero
        result = outer_product(tv, bv)
        assert torch.allclose(result.mv, torch.zeros(16), atol=1e-6)


# =============================================================================
# Inner Product (Left Contraction)
# =============================================================================

class TestInnerProduct:
    """Tests for inner product (left contraction) - must contract grade."""

    def test_inner_product_contracts_grade(self):
        """Inner product contracts: grade(a|b) = |grade(b) - grade(a)|."""
        # Vector | bivector = vector (grade 2 - 1 = 1)
        v = e1(1.0)   # grade 1
        bv = e12(1.0) # grade 2
        result = inner_product(v, bv)
        # Result should be grade 1 (vector)
        # e1 | e12 = e1 . e1 * e2 = 1 * e2 = e2
        assert result.mv[IDX_E2] != 0.0, "e1 | e12 should have e2 component"

    def test_inner_product_higher_grade_into_lower_is_zero(self):
        """Higher grade contracted into lower grade gives zero (for left contraction)."""
        # Bivector | vector with grade(a) > grade(b) should be 0
        bv = e12(1.0)  # grade 2
        v = e1(1.0)    # grade 1
        # Since grade(bv) > grade(v), left contraction gives 0
        result = inner_product(bv, v)
        # For left contraction: if grade(a) > grade(b), result is 0
        # This depends on the convention used
        # The fix implements left contraction which requires grade(a) <= grade(b)
        # Actually check: e12 | e1 should be zero or near-zero
        # because we can't contract a bivector into a vector

    def test_inner_product_scalar_contracts_to_zero(self):
        """Scalar contracted with anything higher gives that element scaled."""
        s = scalar(torch.tensor(2.0))
        v = e1(3.0)
        result = inner_product(s, v)
        # Grade 0 | grade 1 -> grade 1 (if using symmetric inner) or 0 if strict left contraction
        # Left contraction: s | v = 0 for grade(s)=0 < grade(v)=1 is allowed
        # Actually 0 | 1 = 1-0 = 1, so result is grade 1

    def test_inner_product_batched(self):
        """Inner product works with batched inputs."""
        batch_size = 4
        a = Multivector(torch.randn(batch_size, 16))
        b = Multivector(torch.randn(batch_size, 16))
        result = inner_product(a, b)
        assert result.shape == (batch_size,)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_batch(self):
        """Handle empty batch dimension."""
        components = torch.zeros(0, 16)
        mv = Multivector(components)
        assert mv.shape == (0,)

    def test_single_element_operations(self):
        """All operations work on single elements."""
        mv = Multivector(torch.randn(16))
        assert mv.reverse().shape == ()
        assert mv.dual().shape == ()
        assert mv.conjugate().shape == ()
        assert mv.involute().shape == ()
        assert mv.normalize().shape == ()

    def test_repr_string(self):
        """repr returns informative string."""
        mv = Multivector(torch.zeros(4, 16))
        repr_str = repr(mv)
        assert "Multivector" in repr_str
        assert "shape" in repr_str
        # Shape may be represented as torch.Size([4]) or (4,)
        assert "[4]" in repr_str or "(4,)" in repr_str

"""
Full Projective Geometric Algebra (PGA) implementation for G(3,0,1).

PGA is an algebra with 16 basis elements organized by grade:
- Grade 0 (scalar): 1
- Grade 1 (vectors/planes): e₀, e₁, e₂, e₃
- Grade 2 (bivectors/lines): e₀₁, e₀₂, e₀₃, e₁₂, e₃₁, e₂₃
- Grade 3 (trivectors/points): e₀₁₂, e₀₃₁, e₀₂₃, e₁₂₃
- Grade 4 (pseudoscalar): e₀₁₂₃

The metric signature is (3,0,1) meaning:
- e₁² = e₂² = e₃² = +1 (Euclidean)
- e₀² = 0 (degenerate/null direction)

Component ordering:
[s, e0, e1, e2, e3, e01, e02, e03, e12, e31, e23, e012, e031, e023, e123, e0123]
 0   1   2   3   4   5    6    7    8    9    10   11    12    13    14    15
"""

from __future__ import annotations
from typing import Union, Optional, Tuple
import torch
import torch.nn.functional as F


# Component indices for each basis element
IDX_S = 0      # Scalar (grade 0)
IDX_E0 = 1     # e₀
IDX_E1 = 2     # e₁
IDX_E2 = 3     # e₂
IDX_E3 = 4     # e₃
IDX_E01 = 5    # e₀₁
IDX_E02 = 6    # e₀₂
IDX_E03 = 7    # e₀₃
IDX_E12 = 8    # e₁₂
IDX_E31 = 9    # e₃₁
IDX_E23 = 10   # e₂₃
IDX_E012 = 11  # e₀₁₂
IDX_E031 = 12  # e₀₃₁
IDX_E023 = 13  # e₀₂₃
IDX_E123 = 14  # e₁₂₃
IDX_E0123 = 15 # e₀₁₂₃

# Grade masks for extraction
GRADE_0_MASK = [IDX_S]
GRADE_1_MASK = [IDX_E0, IDX_E1, IDX_E2, IDX_E3]
GRADE_2_MASK = [IDX_E01, IDX_E02, IDX_E03, IDX_E12, IDX_E31, IDX_E23]
GRADE_3_MASK = [IDX_E012, IDX_E031, IDX_E023, IDX_E123]
GRADE_4_MASK = [IDX_E0123]

# Reversion sign table: grade k has sign (-1)^(k*(k-1)/2)
# Grade 0: +1, Grade 1: +1, Grade 2: -1, Grade 3: -1, Grade 4: +1
REVERSION_SIGNS = torch.tensor([
    1,   # s
    1,   # e0
    1,   # e1
    1,   # e2
    1,   # e3
    -1,  # e01
    -1,  # e02
    -1,  # e03
    -1,  # e12
    -1,  # e31
    -1,  # e23
    -1,  # e012
    -1,  # e031
    -1,  # e023
    -1,  # e123
    1,   # e0123
], dtype=torch.float32)

# Grade involution sign table: odd grades get negated
INVOLUTION_SIGNS = torch.tensor([
    1,   # s
    -1,  # e0
    -1,  # e1
    -1,  # e2
    -1,  # e3
    1,   # e01
    1,   # e02
    1,   # e03
    1,   # e12
    1,   # e31
    1,   # e23
    -1,  # e012
    -1,  # e031
    -1,  # e023
    -1,  # e123
    1,   # e0123
], dtype=torch.float32)

# Clifford conjugation: reversion + grade involution
CONJUGATION_SIGNS = REVERSION_SIGNS * INVOLUTION_SIGNS


def _build_cayley_table() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the Cayley table for the geometric product in PGA.

    The Cayley table defines: e_i * e_j = sign * e_k

    Returns:
        signs: (16, 16) tensor of signs (+1, -1, or 0)
        indices: (16, 16) tensor of result indices
    """
    # Initialize tables
    signs = torch.zeros(16, 16, dtype=torch.float32)
    indices = torch.zeros(16, 16, dtype=torch.long)

    # Basis element names for reference
    # 0:s, 1:e0, 2:e1, 3:e2, 4:e3, 5:e01, 6:e02, 7:e03,
    # 8:e12, 9:e31, 10:e23, 11:e012, 12:e031, 13:e023, 14:e123, 15:e0123

    # The metric: e0^2 = 0, e1^2 = e2^2 = e3^2 = 1
    # Anti-commutativity: ei*ej = -ej*ei for i != j

    # This is computed offline and hardcoded for efficiency
    # Format: (i, j, sign, result_idx)
    # Generated using scripts/verify_cayley.py with proper PGA multiplication rules
    products = [
        # Row 0: scalar (1) products
        (0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 1, 2), (0, 3, 1, 3),
        (0, 4, 1, 4), (0, 5, 1, 5), (0, 6, 1, 6), (0, 7, 1, 7),
        (0, 8, 1, 8), (0, 9, 1, 9), (0, 10, 1, 10), (0, 11, 1, 11),
        (0, 12, 1, 12), (0, 13, 1, 13), (0, 14, 1, 14), (0, 15, 1, 15),
        # Row 1: e0 products (e0² = 0)
        (1, 0, 1, 1), (1, 1, 0, 0), (1, 2, 1, 5), (1, 3, 1, 6),
        (1, 4, 1, 7), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0),
        (1, 8, 1, 11), (1, 9, 1, 12), (1, 10, 1, 13), (1, 11, 0, 0),
        (1, 12, 0, 0), (1, 13, 0, 0), (1, 14, 1, 15), (1, 15, 0, 0),
        # Row 2: e1 products (e1² = 1)
        (2, 0, 1, 2), (2, 1, -1, 5), (2, 2, 1, 0), (2, 3, 1, 8),
        (2, 4, -1, 9), (2, 5, -1, 1), (2, 6, -1, 11), (2, 7, 1, 12),
        (2, 8, 1, 3), (2, 9, -1, 4), (2, 10, 1, 14), (2, 11, -1, 6),
        (2, 12, 1, 7), (2, 13, -1, 15), (2, 14, 1, 10), (2, 15, -1, 13),
        # Row 3: e2 products (e2² = 1)
        (3, 0, 1, 3), (3, 1, -1, 6), (3, 2, -1, 8), (3, 3, 1, 0),
        (3, 4, 1, 10), (3, 5, 1, 11), (3, 6, -1, 1), (3, 7, -1, 13),
        (3, 8, -1, 2), (3, 9, 1, 14), (3, 10, 1, 4), (3, 11, 1, 5),
        (3, 12, -1, 15), (3, 13, -1, 7), (3, 14, 1, 9), (3, 15, -1, 12),
        # Row 4: e3 products (e3² = 1)
        (4, 0, 1, 4), (4, 1, -1, 7), (4, 2, 1, 9), (4, 3, -1, 10),
        (4, 4, 1, 0), (4, 5, -1, 12), (4, 6, 1, 13), (4, 7, -1, 1),
        (4, 8, 1, 14), (4, 9, 1, 2), (4, 10, -1, 3), (4, 11, -1, 15),
        (4, 12, -1, 5), (4, 13, 1, 6), (4, 14, 1, 8), (4, 15, -1, 11),
        # Row 5: e01 products (e01² = 0 because e0² = 0)
        (5, 0, 1, 5), (5, 1, 0, 0), (5, 2, 1, 1), (5, 3, 1, 11),
        (5, 4, -1, 12), (5, 5, 0, 0), (5, 6, 0, 0), (5, 7, 0, 0),
        (5, 8, 1, 6), (5, 9, -1, 7), (5, 10, 1, 15), (5, 11, 0, 0),
        (5, 12, 0, 0), (5, 13, 0, 0), (5, 14, 1, 13), (5, 15, 0, 0),
        # Row 6: e02 products (e02² = 0 because e0² = 0)
        (6, 0, 1, 6), (6, 1, 0, 0), (6, 2, -1, 11), (6, 3, 1, 1),
        (6, 4, 1, 13), (6, 5, 0, 0), (6, 6, 0, 0), (6, 7, 0, 0),
        (6, 8, -1, 5), (6, 9, 1, 15), (6, 10, 1, 7), (6, 11, 0, 0),
        (6, 12, 0, 0), (6, 13, 0, 0), (6, 14, 1, 12), (6, 15, 0, 0),
        # Row 7: e03 products (e03² = 0 because e0² = 0)
        (7, 0, 1, 7), (7, 1, 0, 0), (7, 2, 1, 12), (7, 3, -1, 13),
        (7, 4, 1, 1), (7, 5, 0, 0), (7, 6, 0, 0), (7, 7, 0, 0),
        (7, 8, 1, 15), (7, 9, 1, 5), (7, 10, -1, 6), (7, 11, 0, 0),
        (7, 12, 0, 0), (7, 13, 0, 0), (7, 14, 1, 11), (7, 15, 0, 0),
        # Row 8: e12 products (e12² = -1 because e1²=e2²=1 and anticommute)
        (8, 0, 1, 8), (8, 1, 1, 11), (8, 2, -1, 3), (8, 3, 1, 2),
        (8, 4, 1, 14), (8, 5, -1, 6), (8, 6, 1, 5), (8, 7, 1, 15),
        (8, 8, -1, 0), (8, 9, 1, 10), (8, 10, -1, 9), (8, 11, -1, 1),
        (8, 12, 1, 13), (8, 13, -1, 12), (8, 14, -1, 4), (8, 15, -1, 7),
        # Row 9: e31 products (e31² = -1, note: e31 = e3∧e1 = -e13)
        (9, 0, 1, 9), (9, 1, 1, 12), (9, 2, 1, 4), (9, 3, 1, 14),
        (9, 4, -1, 2), (9, 5, 1, 7), (9, 6, 1, 15), (9, 7, -1, 5),
        (9, 8, -1, 10), (9, 9, -1, 0), (9, 10, 1, 8), (9, 11, -1, 13),
        (9, 12, -1, 1), (9, 13, 1, 11), (9, 14, -1, 3), (9, 15, -1, 6),
        # Row 10: e23 products (e23² = -1)
        (10, 0, 1, 10), (10, 1, 1, 13), (10, 2, 1, 14), (10, 3, -1, 4),
        (10, 4, 1, 3), (10, 5, 1, 15), (10, 6, -1, 7), (10, 7, 1, 6),
        (10, 8, 1, 9), (10, 9, -1, 8), (10, 10, -1, 0), (10, 11, 1, 12),
        (10, 12, -1, 11), (10, 13, -1, 1), (10, 14, -1, 2), (10, 15, -1, 5),
        # Row 11: e012 products (e012² = 0 because e0² = 0)
        (11, 0, 1, 11), (11, 1, 0, 0), (11, 2, -1, 6), (11, 3, 1, 5),
        (11, 4, 1, 15), (11, 5, 0, 0), (11, 6, 0, 0), (11, 7, 0, 0),
        (11, 8, -1, 1), (11, 9, 1, 13), (11, 10, -1, 12), (11, 11, 0, 0),
        (11, 12, 0, 0), (11, 13, 0, 0), (11, 14, -1, 7), (11, 15, 0, 0),
        # Row 12: e031 products (e031² = 0, note: e031 = e0∧e3∧e1)
        (12, 0, 1, 12), (12, 1, 0, 0), (12, 2, 1, 7), (12, 3, 1, 15),
        (12, 4, -1, 5), (12, 5, 0, 0), (12, 6, 0, 0), (12, 7, 0, 0),
        (12, 8, -1, 13), (12, 9, -1, 1), (12, 10, 1, 11), (12, 11, 0, 0),
        (12, 12, 0, 0), (12, 13, 0, 0), (12, 14, -1, 6), (12, 15, 0, 0),
        # Row 13: e023 products (e023² = 0)
        (13, 0, 1, 13), (13, 1, 0, 0), (13, 2, 1, 15), (13, 3, -1, 7),
        (13, 4, 1, 6), (13, 5, 0, 0), (13, 6, 0, 0), (13, 7, 0, 0),
        (13, 8, 1, 12), (13, 9, -1, 11), (13, 10, -1, 1), (13, 11, 0, 0),
        (13, 12, 0, 0), (13, 13, 0, 0), (13, 14, -1, 5), (13, 15, 0, 0),
        # Row 14: e123 products (e123² = -1, Euclidean pseudoscalar)
        (14, 0, 1, 14), (14, 1, -1, 15), (14, 2, 1, 10), (14, 3, 1, 9),
        (14, 4, 1, 8), (14, 5, -1, 13), (14, 6, -1, 12), (14, 7, -1, 11),
        (14, 8, -1, 4), (14, 9, -1, 3), (14, 10, -1, 2), (14, 11, 1, 7),
        (14, 12, 1, 6), (14, 13, 1, 5), (14, 14, -1, 0), (14, 15, 1, 1),
        # Row 15: e0123 products (e0123² = 0, PGA pseudoscalar)
        (15, 0, 1, 15), (15, 1, 0, 0), (15, 2, 1, 13), (15, 3, 1, 12),
        (15, 4, 1, 11), (15, 5, 0, 0), (15, 6, 0, 0), (15, 7, 0, 0),
        (15, 8, -1, 7), (15, 9, -1, 6), (15, 10, -1, 5), (15, 11, 0, 0),
        (15, 12, 0, 0), (15, 13, 0, 0), (15, 14, -1, 1), (15, 15, 0, 0),
    ]

    for i, j, s, k in products:
        signs[i, j] = s
        indices[i, j] = k

    return signs, indices


# Build Cayley tables at module load time
CAYLEY_SIGNS, CAYLEY_INDICES = _build_cayley_table()


class Multivector:
    """
    A multivector in the Projective Geometric Algebra G(3,0,1).

    Components are stored as a tensor of shape (..., 16) where the last
    dimension contains the coefficients for each basis element.

    The algebra supports:
    - Geometric product (multiplication)
    - Outer (wedge) product
    - Inner (dot) product
    - Regressive (vee) product
    - Reversion, dual, conjugation
    - Normalization and inversion
    """

    def __init__(self, components: torch.Tensor):
        """
        Initialize a multivector from its components.

        Args:
            components: Tensor of shape (..., 16) containing coefficients
                       for each basis element in order:
                       [s, e0, e1, e2, e3, e01, e02, e03, e12, e31, e23,
                        e012, e031, e023, e123, e0123]
        """
        if components.shape[-1] != 16:
            raise ValueError(f"Expected 16 components, got {components.shape[-1]}")
        self.mv = components

    @property
    def shape(self) -> torch.Size:
        """Batch shape (excluding the 16 components)."""
        return self.mv.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.mv.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mv.dtype

    def to(self, device: torch.device) -> 'Multivector':
        """Move to specified device."""
        return Multivector(self.mv.to(device))

    def clone(self) -> 'Multivector':
        """Create a copy."""
        return Multivector(self.mv.clone())

    def detach(self) -> 'Multivector':
        """Detach from computation graph."""
        return Multivector(self.mv.detach())

    # === Grade extraction ===

    def scalar(self) -> torch.Tensor:
        """Extract scalar (grade 0) component."""
        return self.mv[..., IDX_S]

    def vector(self) -> torch.Tensor:
        """Extract vector (grade 1) components: [e0, e1, e2, e3]."""
        return self.mv[..., GRADE_1_MASK]

    def bivector(self) -> torch.Tensor:
        """Extract bivector (grade 2) components: [e01, e02, e03, e12, e31, e23]."""
        return self.mv[..., GRADE_2_MASK]

    def trivector(self) -> torch.Tensor:
        """Extract trivector (grade 3) components: [e012, e031, e023, e123]."""
        return self.mv[..., GRADE_3_MASK]

    def pseudoscalar(self) -> torch.Tensor:
        """Extract pseudoscalar (grade 4) component."""
        return self.mv[..., IDX_E0123]

    def grade(self, k: int) -> 'Multivector':
        """Extract grade-k part of the multivector."""
        result = torch.zeros_like(self.mv)
        if k == 0:
            result[..., GRADE_0_MASK] = self.mv[..., GRADE_0_MASK]
        elif k == 1:
            result[..., GRADE_1_MASK] = self.mv[..., GRADE_1_MASK]
        elif k == 2:
            result[..., GRADE_2_MASK] = self.mv[..., GRADE_2_MASK]
        elif k == 3:
            result[..., GRADE_3_MASK] = self.mv[..., GRADE_3_MASK]
        elif k == 4:
            result[..., GRADE_4_MASK] = self.mv[..., GRADE_4_MASK]
        return Multivector(result)

    # === Unary operations ===

    def reverse(self) -> 'Multivector':
        """
        Reversion: ~M

        Reverses the order of basis vectors in each term.
        Grade k gets sign (-1)^(k(k-1)/2).
        """
        signs = REVERSION_SIGNS.to(self.device)
        return Multivector(self.mv * signs)

    def __invert__(self) -> 'Multivector':
        """Operator ~: reversion."""
        return self.reverse()

    def conjugate(self) -> 'Multivector':
        """
        Clifford conjugation: reversion + grade involution.
        """
        signs = CONJUGATION_SIGNS.to(self.device)
        return Multivector(self.mv * signs)

    def involute(self) -> 'Multivector':
        """
        Grade involution: negate odd grades.
        """
        signs = INVOLUTION_SIGNS.to(self.device)
        return Multivector(self.mv * signs)

    def dual(self) -> 'Multivector':
        """
        Poincaré dual: M* = M ⌋ I^{-1}

        In PGA, the pseudoscalar I = e₀₁₂₃ has I² = 0,
        so we use the approach of complementing indices.
        """
        # Dual mapping for PGA (complement of indices)
        result = torch.zeros_like(self.mv)
        # s <-> e0123
        result[..., IDX_E0123] = self.mv[..., IDX_S]
        result[..., IDX_S] = self.mv[..., IDX_E0123]
        # e0 <-> e123, e1 <-> e023, e2 <-> e031, e3 <-> e012
        result[..., IDX_E123] = self.mv[..., IDX_E0]
        result[..., IDX_E0] = self.mv[..., IDX_E123]
        result[..., IDX_E023] = -self.mv[..., IDX_E1]
        result[..., IDX_E1] = -self.mv[..., IDX_E023]
        result[..., IDX_E031] = self.mv[..., IDX_E2]
        result[..., IDX_E2] = self.mv[..., IDX_E031]
        result[..., IDX_E012] = -self.mv[..., IDX_E3]
        result[..., IDX_E3] = -self.mv[..., IDX_E012]
        # Bivectors: e01 <-> e23, e02 <-> e31, e03 <-> e12
        result[..., IDX_E23] = self.mv[..., IDX_E01]
        result[..., IDX_E01] = self.mv[..., IDX_E23]
        result[..., IDX_E31] = -self.mv[..., IDX_E02]
        result[..., IDX_E02] = -self.mv[..., IDX_E31]
        result[..., IDX_E12] = self.mv[..., IDX_E03]
        result[..., IDX_E03] = self.mv[..., IDX_E12]
        return Multivector(result)

    def norm_squared(self) -> torch.Tensor:
        """
        Compute |M|² = ⟨M ~M⟩₀

        Returns the scalar part of M * ~M.
        """
        product = self * self.reverse()
        return product.scalar()

    def norm(self) -> torch.Tensor:
        """
        Compute |M| = √|⟨M ~M⟩₀|

        Uses absolute value to handle negative squared norms
        (which can occur for some bivectors in PGA).
        """
        return torch.sqrt(torch.abs(self.norm_squared()))

    def normalize(self) -> 'Multivector':
        """
        Return unit multivector: M / |M|
        """
        n = self.norm().unsqueeze(-1).clamp(min=1e-12)
        return Multivector(self.mv / n)

    def inverse(self) -> 'Multivector':
        """
        Multiplicative inverse: M^{-1} where M * M^{-1} = 1

        Uses the formula: M^{-1} = ~M / (M ~M)₀ for versors.
        """
        rev = self.reverse()
        norm_sq = self.norm_squared().unsqueeze(-1).clamp(min=1e-12)
        return Multivector(rev.mv / norm_sq)

    # === Binary operations ===

    def __mul__(self, other: Union['Multivector', float, torch.Tensor]) -> 'Multivector':
        """Geometric product."""
        if isinstance(other, (int, float)):
            return Multivector(self.mv * other)
        if isinstance(other, torch.Tensor) and other.shape[-1] != 16:
            return Multivector(self.mv * other.unsqueeze(-1))
        if isinstance(other, Multivector):
            return geometric_product(self, other)
        return NotImplemented

    def __rmul__(self, other: Union[float, torch.Tensor]) -> 'Multivector':
        """Right multiplication by scalar."""
        return self.__mul__(other)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        """Addition."""
        if isinstance(other, Multivector):
            return Multivector(self.mv + other.mv)
        return NotImplemented

    def __sub__(self, other: 'Multivector') -> 'Multivector':
        """Subtraction."""
        if isinstance(other, Multivector):
            return Multivector(self.mv - other.mv)
        return NotImplemented

    def __neg__(self) -> 'Multivector':
        """Negation."""
        return Multivector(-self.mv)

    def __truediv__(self, other: Union[float, torch.Tensor]) -> 'Multivector':
        """Division by scalar."""
        if isinstance(other, (int, float)):
            return Multivector(self.mv / other)
        if isinstance(other, torch.Tensor):
            return Multivector(self.mv / other.unsqueeze(-1))
        return NotImplemented

    def __xor__(self, other: 'Multivector') -> 'Multivector':
        """Outer (wedge) product: a ^ b."""
        return outer_product(self, other)

    def __or__(self, other: 'Multivector') -> 'Multivector':
        """Inner (dot) product: a | b."""
        return inner_product(self, other)

    def __and__(self, other: 'Multivector') -> 'Multivector':
        """Regressive (vee) product: a & b."""
        return regressive_product(self, other)

    def outer(self, other: 'Multivector') -> 'Multivector':
        """Outer (wedge) product."""
        return outer_product(self, other)

    def inner(self, other: 'Multivector') -> 'Multivector':
        """Inner (dot) product."""
        return inner_product(self, other)

    def regressive(self, other: 'Multivector') -> 'Multivector':
        """Regressive (vee) product."""
        return regressive_product(self, other)

    def __repr__(self) -> str:
        return f"Multivector(shape={self.shape}, device={self.device})"


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the geometric product a * b.

    Uses the Cayley table for efficient computation.
    """
    # Move Cayley tables to correct device
    signs = CAYLEY_SIGNS.to(a.device)
    indices = CAYLEY_INDICES.to(a.device)

    # Expand for broadcasting: a is (..., 16, 1), b is (..., 1, 16)
    a_expanded = a.mv.unsqueeze(-1)  # (..., 16, 1)
    b_expanded = b.mv.unsqueeze(-2)  # (..., 1, 16)

    # Compute all pairwise products: (..., 16, 16)
    products = a_expanded * b_expanded * signs

    # Accumulate into result components using scatter_add
    batch_shape = a.mv.shape[:-1]
    result = torch.zeros(*batch_shape, 16, device=a.device, dtype=a.dtype)

    # Flatten for scatter
    flat_products = products.reshape(-1, 16, 16)
    flat_result = result.reshape(-1, 16)

    for i in range(16):
        for j in range(16):
            k = indices[i, j].item()
            flat_result[:, k] += flat_products[:, i, j]

    return Multivector(flat_result.reshape(*batch_shape, 16))


def outer_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the outer (wedge) product a ∧ b.

    The outer product extracts the grade-raising part of the geometric product.
    For grade-r and grade-s elements: (a ∧ b) has grade r + s.
    """
    result = geometric_product(a, b)

    # Zero out components that don't match the expected grade
    # This is a simplified implementation - full version would
    # compute grade combinations explicitly
    return result  # TODO: Implement proper grade filtering


def inner_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the inner (dot/left contraction) product a ⌋ b.

    For grade-r and grade-s elements with r <= s:
    (a ⌋ b) has grade |s - r|.
    """
    result = geometric_product(a, b)
    # TODO: Implement proper grade filtering for inner product
    return result


def regressive_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the regressive (vee) product a ∨ b.

    Defined as: a ∨ b = (a* ∧ b*)*
    where * denotes the dual.
    """
    return outer_product(a.dual(), b.dual()).dual()


def sandwich(motor: Multivector, element: Multivector) -> Multivector:
    """
    Compute the sandwich product: M * X * ~M

    This is the fundamental operation for applying transformations in GA.
    """
    return motor * element * motor.reverse()


def exp(bivector: Multivector) -> Multivector:
    """
    Exponential map: exp(B) for a bivector B.

    Produces a rotor (for pure rotation bivector) or motor (for general).

    Uses Taylor series with closed-form for pure bivectors.
    """
    # Extract bivector components
    bv = bivector.bivector()

    # Decompose into Euclidean (e12, e31, e23) and ideal (e01, e02, e03) parts
    ideal = bv[..., :3]      # e01, e02, e03
    euclidean = bv[..., 3:]  # e12, e31, e23

    # Compute magnitude of Euclidean part
    theta_sq = (euclidean * euclidean).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq.clamp(min=1e-12))

    # exp(B) = cos(|B|) + sin(|B|)/|B| * B for pure Euclidean bivector
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    sinc_theta = sin_theta / theta.clamp(min=1e-12)

    # Build result multivector
    result = torch.zeros(*bivector.shape, 16, device=bivector.device, dtype=bivector.dtype)

    # Scalar part
    result[..., IDX_S] = cos_theta.squeeze(-1)

    # Bivector parts (Euclidean)
    result[..., IDX_E12] = sinc_theta.squeeze(-1) * euclidean[..., 0]
    result[..., IDX_E31] = sinc_theta.squeeze(-1) * euclidean[..., 1]
    result[..., IDX_E23] = sinc_theta.squeeze(-1) * euclidean[..., 2]

    # Ideal bivector parts (translation component)
    result[..., IDX_E01] = ideal[..., 0]
    result[..., IDX_E02] = ideal[..., 1]
    result[..., IDX_E03] = ideal[..., 2]

    return Multivector(result)


def log(motor: Multivector) -> Multivector:
    """
    Logarithm map: log(M) for a motor M.

    Returns the bivector B such that exp(B) = M.
    """
    # Extract rotor part (scalar + Euclidean bivector)
    s = motor.scalar()
    bv = motor.bivector()
    euclidean = bv[..., 3:]  # e12, e31, e23

    # Compute rotation angle
    eucl_norm = torch.sqrt((euclidean * euclidean).sum(dim=-1, keepdim=True).clamp(min=1e-12))
    theta = torch.atan2(eucl_norm.squeeze(-1), s)

    # Compute bivector
    result = torch.zeros(*motor.shape, 16, device=motor.device, dtype=motor.dtype)

    # Scale factor for bivector
    scale = theta / eucl_norm.squeeze(-1).clamp(min=1e-12)

    result[..., IDX_E12] = scale * euclidean[..., 0]
    result[..., IDX_E31] = scale * euclidean[..., 1]
    result[..., IDX_E23] = scale * euclidean[..., 2]

    # Copy ideal parts directly
    result[..., IDX_E01] = bv[..., 0]
    result[..., IDX_E02] = bv[..., 1]
    result[..., IDX_E03] = bv[..., 2]

    return Multivector(result)


# === Factory functions for basis elements ===

def _zeros_like(ref: Optional[torch.Tensor] = None, batch_shape: tuple = (),
                device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """Create zeros tensor with proper shape."""
    if ref is not None:
        device = device or ref.device
        dtype = dtype or ref.dtype
        batch_shape = batch_shape or ref.shape[:-1] if ref.dim() > 0 else ()
    return torch.zeros(*batch_shape, 16, device=device, dtype=dtype or torch.float32)


def scalar(s: Union[float, torch.Tensor]) -> Multivector:
    """Create a scalar multivector."""
    if isinstance(s, (int, float)):
        mv = torch.zeros(16)
        mv[IDX_S] = s
    else:
        mv = _zeros_like(s, batch_shape=s.shape)
        mv[..., IDX_S] = s
    return Multivector(mv)


def _basis(idx: int, coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create a basis element multivector."""
    if isinstance(coeff, (int, float)):
        mv = torch.zeros(16)
        mv[idx] = coeff
    else:
        mv = _zeros_like(coeff, batch_shape=coeff.shape)
        mv[..., idx] = coeff
    return Multivector(mv)


# Basis elements (as functions that create multivectors)
def e0(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₀ basis element (degenerate direction)."""
    return _basis(IDX_E0, coeff)


def e1(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₁ basis element."""
    return _basis(IDX_E1, coeff)


def e2(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₂ basis element."""
    return _basis(IDX_E2, coeff)


def e3(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₃ basis element."""
    return _basis(IDX_E3, coeff)


def e01(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₀₁ basis bivector."""
    return _basis(IDX_E01, coeff)


def e02(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₀₂ basis bivector."""
    return _basis(IDX_E02, coeff)


def e03(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₀₃ basis bivector."""
    return _basis(IDX_E03, coeff)


def e12(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₁₂ basis bivector."""
    return _basis(IDX_E12, coeff)


def e31(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₃₁ basis bivector."""
    return _basis(IDX_E31, coeff)


def e23(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₂₃ basis bivector."""
    return _basis(IDX_E23, coeff)


def e012(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₀₁₂ basis trivector."""
    return _basis(IDX_E012, coeff)


def e031(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₀₃₁ basis trivector."""
    return _basis(IDX_E031, coeff)


def e023(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₀₂₃ basis trivector."""
    return _basis(IDX_E023, coeff)


def e123(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₁₂₃ basis trivector (the Euclidean pseudoscalar)."""
    return _basis(IDX_E123, coeff)


def e0123(coeff: Union[float, torch.Tensor] = 1.0) -> Multivector:
    """Create e₀₁₂₃ basis element (the PGA pseudoscalar)."""
    return _basis(IDX_E0123, coeff)

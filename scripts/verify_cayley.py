"""Generate the correct Cayley table for PGA(3,0,1) and find discrepancies."""
import torch

# Basis element definitions
# Index: 0:s, 1:e0, 2:e1, 3:e2, 4:e3, 5:e01, 6:e02, 7:e03,
#        8:e12, 9:e31, 10:e23, 11:e012, 12:e031, 13:e023, 14:e123, 15:e0123

# Map index to blade (tuple of indices with sign)
# e.g., e31 is stored as (3,1) meaning e3∧e1
index_to_blade = {
    0: (),           # scalar
    1: (0,),         # e0
    2: (1,),         # e1
    3: (2,),         # e2
    4: (3,),         # e3
    5: (0, 1),       # e01
    6: (0, 2),       # e02
    7: (0, 3),       # e03
    8: (1, 2),       # e12
    9: (3, 1),       # e31 (NOT e13!)
    10: (2, 3),      # e23
    11: (0, 1, 2),   # e012
    12: (0, 3, 1),   # e031 (NOT e013!)
    13: (0, 2, 3),   # e023
    14: (1, 2, 3),   # e123
    15: (0, 1, 2, 3) # e0123
}

# Metric: e0^2 = 0, e1^2 = e2^2 = e3^2 = 1
metric = {0: 0, 1: 1, 2: 1, 3: 1}

def canonical_blade(blade):
    """Convert blade to canonical form (sorted, with sign)."""
    if len(blade) == 0:
        return (), 1

    # Bubble sort to count swaps
    blade = list(blade)
    sign = 1
    for i in range(len(blade)):
        for j in range(len(blade) - 1 - i):
            if blade[j] > blade[j + 1]:
                blade[j], blade[j + 1] = blade[j + 1], blade[j]
                sign *= -1
    return tuple(blade), sign

def multiply_blades(a, b):
    """Multiply two blades, returning (result_blade, sign).

    Uses bubble sort to bring the combined blade to canonical form,
    contracting pairs of equal indices using the metric.
    Each swap of adjacent elements multiplies the sign by -1.
    """
    # Concatenate blades
    combined = list(a) + list(b)
    sign = 1

    # Bubble sort to canonical form, contracting equal adjacent pairs
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(combined) - 1:
            if combined[i] == combined[i + 1]:
                # Adjacent equal elements: contract using metric
                m = metric[combined[i]]
                if m == 0:
                    return (), 0  # Zero result (e0^2 = 0)
                sign *= m
                # Remove both elements
                combined.pop(i + 1)
                combined.pop(i)
                changed = True
                # Don't increment i, check from same position
            elif combined[i] > combined[i + 1]:
                # Swap adjacent elements: ei*ej = -ej*ei for i != j
                combined[i], combined[i + 1] = combined[i + 1], combined[i]
                sign *= -1
                changed = True
                i += 1
            else:
                i += 1

    return tuple(combined), sign

# Build reverse map: canonical blade -> index
blade_to_index = {}
for idx, blade in index_to_blade.items():
    canonical, sign = canonical_blade(blade)
    if canonical not in blade_to_index:
        blade_to_index[canonical] = (idx, sign)

# Now compute the Cayley table
cayley_signs = torch.zeros(16, 16, dtype=torch.float32)
cayley_indices = torch.zeros(16, 16, dtype=torch.long)

errors = []

from pga_inr.pga.algebra import CAYLEY_SIGNS, CAYLEY_INDICES

for i in range(16):
    for j in range(16):
        blade_i = index_to_blade[i]
        blade_j = index_to_blade[j]
        result_blade, sign = multiply_blades(blade_i, blade_j)

        if sign == 0:
            cayley_signs[i, j] = 0
            cayley_indices[i, j] = 0  # or any index, since sign is 0
        else:
            canonical, canonical_sign = canonical_blade(result_blade)
            if canonical in blade_to_index:
                result_idx, idx_sign = blade_to_index[canonical]
                cayley_signs[i, j] = sign * canonical_sign * idx_sign
                cayley_indices[i, j] = result_idx
            else:
                print(f"Warning: blade {canonical} not in index map")

        # Check against current table
        old_sign = CAYLEY_SIGNS[i, j].item()
        old_idx = CAYLEY_INDICES[i, j].item()
        new_sign = cayley_signs[i, j].item()
        new_idx = cayley_indices[i, j].item()

        if old_sign != new_sign or old_idx != new_idx:
            errors.append((i, j, old_sign, old_idx, new_sign, new_idx))

print(f"Found {len(errors)} discrepancies:")
for i, j, old_s, old_idx, new_s, new_idx in errors:
    blade_i = index_to_blade[i]
    blade_j = index_to_blade[j]
    print(f"  ({i}, {j}): blade{blade_i} * blade{blade_j}")
    print(f"    OLD: sign={old_s}, idx={old_idx}")
    print(f"    NEW: sign={new_s}, idx={new_idx}")

# Generate the corrected products list for the algebra.py file
print("\n\n# Corrected products list for algebra.py:")
print("products = [")
for i in range(16):
    line_items = []
    for j in range(16):
        sign = int(cayley_signs[i, j].item())
        idx = int(cayley_indices[i, j].item())
        line_items.append(f"({i}, {j}, {sign}, {idx})")
    # Print 4 per line for readability
    for k in range(0, 16, 4):
        print("    " + ", ".join(line_items[k:k+4]) + ",")
print("]")

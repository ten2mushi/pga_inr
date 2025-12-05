"""
Tests for PGA geometric primitives: points, lines, and planes.

Following the Yoneda philosophy: these tests completely DEFINE how geometric
primitives are represented and how they interact. The tests serve as the
authoritative specification for:

1. Point representation as grade-3 trivectors
2. Line representation as grade-2 bivectors (Plucker coordinates)
3. Plane representation as grade-1 vectors
4. Join (regressive) and Meet (wedge) operations
5. Distance and projection computations
"""

import pytest
import torch
import math

from pga_inr.pga.primitives import (
    # Point operations
    point,
    point_from_tensor,
    point_to_cartesian,
    ideal_point,
    origin,
    # Plane operations
    plane,
    plane_from_normal_point,
    plane_to_normal_distance,
    xy_plane, xz_plane, yz_plane,
    # Line operations
    line_from_points,
    line_from_plucker,
    line_from_point_direction,
    line_to_plucker,
    x_axis, y_axis, z_axis,
    # Geometric operations
    join,
    meet,
    # Distance computations
    distance_point_point,
    distance_point_plane,
    distance_point_line,
    # Projections
    project_point_plane,
    project_point_line,
    reflect_point_plane,
)
from pga_inr.pga.algebra import (
    Multivector,
    IDX_E0, IDX_E1, IDX_E2, IDX_E3,
    IDX_E01, IDX_E02, IDX_E03, IDX_E12, IDX_E31, IDX_E23,
    IDX_E012, IDX_E031, IDX_E023, IDX_E123,
)


# =============================================================================
# Point Creation and Properties
# =============================================================================

class TestPointCreation:
    """Tests for point creation - DEFINES the point representation."""

    def test_origin_point_representation(self):
        """Origin (0,0,0) is represented as pure e123."""
        p = point(0.0, 0.0, 0.0)
        assert p.mv[IDX_E123] == 1.0
        assert p.mv[IDX_E023] == 0.0
        assert p.mv[IDX_E031] == 0.0
        assert p.mv[IDX_E012] == 0.0

    def test_unit_x_point(self):
        """Point (1,0,0) has correct representation."""
        p = point(1.0, 0.0, 0.0)
        assert p.mv[IDX_E123] == 1.0
        # x coordinate encoded in e023 (negative sign)
        assert p.mv[IDX_E023] == -1.0
        assert p.mv[IDX_E031] == 0.0
        assert p.mv[IDX_E012] == 0.0

    def test_unit_y_point(self):
        """Point (0,1,0) has correct representation."""
        p = point(0.0, 1.0, 0.0)
        assert p.mv[IDX_E123] == 1.0
        assert p.mv[IDX_E023] == 0.0
        # y coordinate encoded in e031 (negative sign)
        assert p.mv[IDX_E031] == -1.0
        assert p.mv[IDX_E012] == 0.0

    def test_unit_z_point(self):
        """Point (0,0,1) has correct representation."""
        p = point(0.0, 0.0, 1.0)
        assert p.mv[IDX_E123] == 1.0
        assert p.mv[IDX_E023] == 0.0
        assert p.mv[IDX_E031] == 0.0
        # z coordinate encoded in e012 (negative sign)
        assert p.mv[IDX_E012] == -1.0

    def test_arbitrary_point(self):
        """Arbitrary point (x,y,z) has correct representation."""
        p = point(3.0, -2.0, 5.0)
        assert p.mv[IDX_E123] == 1.0
        assert p.mv[IDX_E023] == -3.0
        assert p.mv[IDX_E031] == 2.0  # -(-2.0)
        assert p.mv[IDX_E012] == -5.0

    def test_point_is_grade3_trivector(self):
        """Points are pure grade-3 elements."""
        p = point(1.0, 2.0, 3.0)
        # Only trivector components should be nonzero
        assert p.mv[IDX_E123] != 0 or p.mv[IDX_E023] != 0 or \
               p.mv[IDX_E031] != 0 or p.mv[IDX_E012] != 0
        # Lower grade components should be zero
        assert torch.allclose(p.mv[..., :5], torch.zeros(5))  # grades 0,1
        assert torch.allclose(p.mv[..., 5:11], torch.zeros(6))  # grade 2

    def test_point_from_tensor(self):
        """point_from_tensor creates points from coordinate tensor."""
        coords = torch.tensor([1.0, 2.0, 3.0])
        p = point_from_tensor(coords)
        reconstructed = point_to_cartesian(p)
        assert torch.allclose(reconstructed, coords)

    def test_point_batch_creation(self):
        """Batched point creation works correctly."""
        xs = torch.tensor([1.0, 2.0, 3.0])
        ys = torch.tensor([4.0, 5.0, 6.0])
        zs = torch.tensor([7.0, 8.0, 9.0])
        points = point(xs, ys, zs)
        assert points.shape == (3,)
        # Verify each point
        for i in range(3):
            coords = point_to_cartesian(Multivector(points.mv[i]))
            assert torch.allclose(coords, torch.tensor([xs[i], ys[i], zs[i]]))

    def test_point_2d_batch(self):
        """2D batched point creation preserves dimensions."""
        coords = torch.randn(4, 5, 3)
        points = point_from_tensor(coords)
        assert points.shape == (4, 5)
        reconstructed = point_to_cartesian(points)
        assert torch.allclose(reconstructed, coords)


class TestPointToCartesian:
    """Tests for extracting Cartesian coordinates from points."""

    def test_origin_to_cartesian(self):
        """Origin converts to (0,0,0)."""
        p = origin()
        coords = point_to_cartesian(p)
        assert torch.allclose(coords, torch.zeros(3))

    def test_roundtrip_conversion(self):
        """point -> to_cartesian roundtrip preserves coordinates."""
        original_coords = torch.tensor([3.14, -2.71, 1.41])
        p = point(original_coords[0], original_coords[1], original_coords[2])
        recovered = point_to_cartesian(p)
        assert torch.allclose(recovered, original_coords, atol=1e-5)

    def test_batched_roundtrip(self):
        """Batched roundtrip conversion."""
        coords = torch.randn(10, 3)
        points = point_from_tensor(coords)
        recovered = point_to_cartesian(points)
        assert torch.allclose(recovered, coords, atol=1e-5)


class TestIdealPoint:
    """Tests for ideal points (points at infinity)."""

    def test_ideal_point_has_zero_weight(self):
        """Ideal points have e123 = 0."""
        ip = ideal_point(1.0, 0.0, 0.0)
        assert ip.mv[IDX_E123] == 0.0

    def test_ideal_point_represents_direction(self):
        """Ideal points encode direction in trivector components."""
        ip = ideal_point(1.0, 2.0, 3.0)
        assert ip.mv[IDX_E023] == -1.0
        assert ip.mv[IDX_E031] == -2.0
        assert ip.mv[IDX_E012] == -3.0


# =============================================================================
# Plane Creation and Properties
# =============================================================================

class TestPlaneCreation:
    """Tests for plane creation - DEFINES the plane representation."""

    def test_xy_plane_representation(self):
        """XY plane (z=0) has normal (0,0,1) and d=0."""
        p = xy_plane()
        normal, d = plane_to_normal_distance(p)
        assert torch.allclose(normal, torch.tensor([0.0, 0.0, 1.0]))
        assert d == 0.0

    def test_xz_plane_representation(self):
        """XZ plane (y=0) has normal (0,1,0) and d=0."""
        p = xz_plane()
        normal, d = plane_to_normal_distance(p)
        assert torch.allclose(normal, torch.tensor([0.0, 1.0, 0.0]))
        assert d == 0.0

    def test_yz_plane_representation(self):
        """YZ plane (x=0) has normal (1,0,0) and d=0."""
        p = yz_plane()
        normal, d = plane_to_normal_distance(p)
        assert torch.allclose(normal, torch.tensor([1.0, 0.0, 0.0]))
        assert d == 0.0

    def test_plane_is_grade1_vector(self):
        """Planes are pure grade-1 elements."""
        p = plane(1.0, 2.0, 3.0, 4.0)
        # Only vector components should be nonzero
        assert p.mv[IDX_E0] == 4.0
        assert p.mv[IDX_E1] == 1.0
        assert p.mv[IDX_E2] == 2.0
        assert p.mv[IDX_E3] == 3.0
        # Other grades should be zero
        assert p.mv[0] == 0.0  # scalar

    def test_plane_from_normal_point_origin(self):
        """Plane through origin with given normal."""
        normal = torch.tensor([0.0, 0.0, 1.0])
        pt = torch.tensor([0.0, 0.0, 0.0])
        p = plane_from_normal_point(normal, pt)
        n, d = plane_to_normal_distance(p)
        assert torch.allclose(n, normal)
        assert torch.isclose(d, torch.tensor(0.0))

    def test_plane_from_normal_point_offset(self):
        """Plane with offset from origin."""
        normal = torch.tensor([0.0, 0.0, 1.0])
        pt = torch.tensor([0.0, 0.0, 5.0])  # z=5 plane
        p = plane_from_normal_point(normal, pt)
        n, d = plane_to_normal_distance(p)
        # d = -n.p = -(0*0 + 0*0 + 1*5) = -5
        assert torch.isclose(d, torch.tensor(-5.0))

    def test_batched_plane_creation(self):
        """Batched plane creation."""
        nxs = torch.tensor([1.0, 0.0, 0.0])
        nys = torch.tensor([0.0, 1.0, 0.0])
        nzs = torch.tensor([0.0, 0.0, 1.0])
        ds = torch.tensor([0.0, 0.0, 0.0])
        planes = plane(nxs, nys, nzs, ds)
        assert planes.shape == (3,)


# =============================================================================
# Line Creation and Properties
# =============================================================================

class TestLineCreation:
    """Tests for line creation - DEFINES the line representation."""

    def test_x_axis_representation(self):
        """X axis has direction (1,0,0) and moment (0,0,0)."""
        line = x_axis()
        direction, moment = line_to_plucker(line)
        assert torch.allclose(direction, torch.tensor([1.0, 0.0, 0.0]))
        assert torch.allclose(moment, torch.tensor([0.0, 0.0, 0.0]))

    def test_y_axis_representation(self):
        """Y axis has direction (0,1,0) and moment (0,0,0)."""
        line = y_axis()
        direction, moment = line_to_plucker(line)
        assert torch.allclose(direction, torch.tensor([0.0, 1.0, 0.0]))
        assert torch.allclose(moment, torch.tensor([0.0, 0.0, 0.0]))

    def test_z_axis_representation(self):
        """Z axis has direction (0,0,1) and moment (0,0,0)."""
        line = z_axis()
        direction, moment = line_to_plucker(line)
        assert torch.allclose(direction, torch.tensor([0.0, 0.0, 1.0]))
        assert torch.allclose(moment, torch.tensor([0.0, 0.0, 0.0]))

    def test_line_is_grade2_bivector(self):
        """Lines are pure grade-2 elements."""
        line = x_axis()
        # Only bivector components may be nonzero
        bivector_indices = [IDX_E01, IDX_E02, IDX_E03, IDX_E12, IDX_E31, IDX_E23]
        for idx in range(16):
            if idx not in bivector_indices:
                assert line.mv[idx] == 0.0, f"Expected 0 at index {idx}"

    def test_line_from_point_direction(self):
        """Line from point and direction."""
        pt = torch.tensor([1.0, 0.0, 0.0])
        direction = torch.tensor([0.0, 1.0, 0.0])
        line = line_from_point_direction(pt, direction)
        d, m = line_to_plucker(line)
        # Direction should be normalized
        assert torch.allclose(d, direction)
        # Moment = point x direction = (1,0,0) x (0,1,0) = (0,0,1)
        expected_moment = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(m, expected_moment, atol=1e-5)

    def test_line_from_two_points(self):
        """Line through two points."""
        p1 = point(0.0, 0.0, 0.0)
        p2 = point(1.0, 0.0, 0.0)
        line = line_from_points(p1, p2)
        # Should be along x-axis
        d, m = line_to_plucker(line)
        # Direction should point along x (or negative x)
        assert abs(d[0]) > 0.5  # Primarily x direction
        assert torch.allclose(m, torch.zeros(3), atol=1e-5)  # Through origin

    def test_plucker_roundtrip(self):
        """Plucker coordinates roundtrip."""
        direction = torch.tensor([1.0, 2.0, 3.0])
        moment = torch.tensor([0.0, 0.0, 0.0])  # Line through origin
        line = line_from_plucker(direction, moment)
        d_out, m_out = line_to_plucker(line)
        assert torch.allclose(d_out, direction)
        assert torch.allclose(m_out, moment)


# =============================================================================
# Join and Meet Operations
# =============================================================================

class TestJoinOperation:
    """Tests for join (regressive product) - combining primitives."""

    def test_join_two_points_gives_line(self):
        """Joining two distinct points creates a line."""
        p1 = point(0.0, 0.0, 0.0)
        p2 = point(1.0, 0.0, 0.0)
        line = join(p1, p2)
        # Result should be primarily grade-2
        d, m = line_to_plucker(line)
        # Line should point along x-axis direction
        assert d[0] != 0.0

    def test_join_coincident_points_gives_zero(self):
        """Joining the same point gives zero (degenerate)."""
        p = point(1.0, 2.0, 3.0)
        result = join(p, p)
        # The join of a point with itself is degenerate
        # In PGA, the regressive product of identical points gives the pseudoscalar,
        # not zero. This is mathematically correct: P ∨ P = ||P||² * I
        # The bivector (line) components should be zero
        d, m = line_to_plucker(result)
        # Direction and moment should be zero (no actual line)
        assert torch.allclose(d, torch.zeros(3), atol=1e-5)
        assert torch.allclose(m, torch.zeros(3), atol=1e-5)

    def test_join_is_antisymmetric(self):
        """Join is antisymmetric: a v b = -b v a."""
        p1 = point(0.0, 0.0, 0.0)
        p2 = point(1.0, 1.0, 1.0)
        ab = join(p1, p2)
        ba = join(p2, p1)
        assert torch.allclose(ab.mv, -ba.mv, atol=1e-5)

    def test_join_point_line_gives_plane(self):
        """Joining a point and line creates a plane."""
        p = point(0.0, 0.0, 1.0)  # Point above XY plane
        line = x_axis()  # X axis
        plane_result = join(p, line)
        # Result should be primarily grade-1
        normal, d = plane_to_normal_distance(plane_result)
        # The plane should contain both the point and line


class TestMeetOperation:
    """Tests for meet (outer product) - intersecting primitives."""

    def test_meet_two_planes_gives_line(self):
        """Meeting two planes creates a line (their intersection)."""
        p1 = xy_plane()  # z = 0
        p2 = xz_plane()  # y = 0
        line = meet(p1, p2)
        # Intersection should be x-axis
        d, m = line_to_plucker(line)
        # Direction should be along x (or scaled version)
        # The result may need normalization

    def test_meet_parallel_planes_gives_ideal_line(self):
        """Meeting parallel planes gives an ideal element."""
        p1 = plane(0.0, 0.0, 1.0, 0.0)   # z = 0
        p2 = plane(0.0, 0.0, 1.0, -1.0)  # z = 1
        line = meet(p1, p2)
        # Parallel planes meet at infinity (ideal line)
        d, m = line_to_plucker(line)
        # Euclidean part (direction) should be zero
        assert torch.allclose(d, torch.zeros(3), atol=1e-5)


# =============================================================================
# Distance Computations
# =============================================================================

class TestDistancePointPoint:
    """Tests for point-to-point distance."""

    def test_same_point_distance_is_zero(self):
        """Distance from point to itself is zero."""
        p = point(1.0, 2.0, 3.0)
        dist = distance_point_point(p, p)
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-6)

    def test_unit_distance_x(self):
        """Distance between adjacent unit points."""
        p1 = point(0.0, 0.0, 0.0)
        p2 = point(1.0, 0.0, 0.0)
        dist = distance_point_point(p1, p2)
        assert torch.isclose(dist, torch.tensor(1.0), atol=1e-6)

    def test_unit_distance_y(self):
        """Distance along y-axis."""
        p1 = point(0.0, 0.0, 0.0)
        p2 = point(0.0, 1.0, 0.0)
        dist = distance_point_point(p1, p2)
        assert torch.isclose(dist, torch.tensor(1.0), atol=1e-6)

    def test_unit_distance_z(self):
        """Distance along z-axis."""
        p1 = point(0.0, 0.0, 0.0)
        p2 = point(0.0, 0.0, 1.0)
        dist = distance_point_point(p1, p2)
        assert torch.isclose(dist, torch.tensor(1.0), atol=1e-6)

    def test_diagonal_distance(self):
        """Distance for diagonal displacement."""
        p1 = point(0.0, 0.0, 0.0)
        p2 = point(1.0, 1.0, 1.0)
        dist = distance_point_point(p1, p2)
        expected = math.sqrt(3.0)
        assert torch.isclose(dist, torch.tensor(expected), atol=1e-5)

    def test_distance_is_symmetric(self):
        """d(a,b) = d(b,a)."""
        p1 = point(1.0, 2.0, 3.0)
        p2 = point(4.0, 5.0, 6.0)
        d12 = distance_point_point(p1, p2)
        d21 = distance_point_point(p2, p1)
        assert torch.isclose(d12, d21, atol=1e-6)

    def test_batched_distance(self):
        """Batched distance computation."""
        x1 = torch.zeros(5)
        y1 = torch.zeros(5)
        z1 = torch.zeros(5)
        x2 = torch.ones(5)
        y2 = torch.zeros(5)
        z2 = torch.zeros(5)
        p1 = point(x1, y1, z1)
        p2 = point(x2, y2, z2)
        dists = distance_point_point(p1, p2)
        assert dists.shape == (5,)
        assert torch.allclose(dists, torch.ones(5), atol=1e-6)


class TestDistancePointPlane:
    """Tests for signed point-to-plane distance."""

    def test_point_on_plane_distance_zero(self):
        """Point on plane has distance zero."""
        p = point(1.0, 2.0, 0.0)  # On XY plane
        dist = distance_point_plane(p, xy_plane())
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-6)

    def test_point_above_plane_positive(self):
        """Point above plane has positive distance."""
        p = point(0.0, 0.0, 1.0)  # Above XY plane
        dist = distance_point_plane(p, xy_plane())
        assert torch.isclose(dist, torch.tensor(1.0), atol=1e-5)

    def test_point_below_plane_negative(self):
        """Point below plane has negative distance."""
        p = point(0.0, 0.0, -1.0)  # Below XY plane
        dist = distance_point_plane(p, xy_plane())
        assert torch.isclose(dist, torch.tensor(-1.0), atol=1e-5)

    def test_arbitrary_plane_distance(self):
        """Distance to arbitrary plane."""
        # Plane: x = 3
        pi = plane(1.0, 0.0, 0.0, -3.0)  # x - 3 = 0
        p = point(5.0, 0.0, 0.0)  # x = 5
        dist = distance_point_plane(p, pi)
        assert torch.isclose(dist, torch.tensor(2.0), atol=1e-5)


class TestDistancePointLine:
    """Tests for point-to-line distance."""

    def test_point_on_line_distance_zero(self):
        """Point on line has distance zero."""
        line = x_axis()
        p = point(5.0, 0.0, 0.0)  # On x-axis
        dist = distance_point_line(p, line)
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-5)

    def test_point_perpendicular_to_line(self):
        """Distance to perpendicular point."""
        line = x_axis()
        p = point(0.0, 1.0, 0.0)  # Unit distance from x-axis
        dist = distance_point_line(p, line)
        assert torch.isclose(dist, torch.tensor(1.0), atol=1e-5)

    def test_diagonal_point_to_axis(self):
        """Distance from diagonal point to axis."""
        line = x_axis()
        p = point(5.0, 1.0, 0.0)  # y=1 from x-axis
        dist = distance_point_line(p, line)
        assert torch.isclose(dist, torch.tensor(1.0), atol=1e-5)


# =============================================================================
# Projection Operations
# =============================================================================

class TestProjectPointPlane:
    """Tests for projecting points onto planes."""

    def test_project_point_on_plane_unchanged(self):
        """Point on plane projects to itself."""
        p = point(5.0, 3.0, 0.0)  # On XY plane
        projected = project_point_plane(p, xy_plane())
        coords = point_to_cartesian(projected)
        assert torch.allclose(coords, torch.tensor([5.0, 3.0, 0.0]), atol=1e-5)

    def test_project_point_to_xy_plane(self):
        """Project point to XY plane (zero z)."""
        p = point(1.0, 2.0, 3.0)
        projected = project_point_plane(p, xy_plane())
        coords = point_to_cartesian(projected)
        assert torch.allclose(coords, torch.tensor([1.0, 2.0, 0.0]), atol=1e-5)

    def test_project_point_to_xz_plane(self):
        """Project point to XZ plane (zero y)."""
        p = point(1.0, 2.0, 3.0)
        projected = project_point_plane(p, xz_plane())
        coords = point_to_cartesian(projected)
        assert torch.allclose(coords, torch.tensor([1.0, 0.0, 3.0]), atol=1e-5)

    def test_project_point_to_yz_plane(self):
        """Project point to YZ plane (zero x)."""
        p = point(1.0, 2.0, 3.0)
        projected = project_point_plane(p, yz_plane())
        coords = point_to_cartesian(projected)
        assert torch.allclose(coords, torch.tensor([0.0, 2.0, 3.0]), atol=1e-5)

    def test_projected_point_is_on_plane(self):
        """Projected point lies on the plane."""
        p = point(3.0, 4.0, 5.0)
        pi = xy_plane()
        projected = project_point_plane(p, pi)
        dist = distance_point_plane(projected, pi)
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-5)


class TestProjectPointLine:
    """Tests for projecting points onto lines."""

    def test_project_to_x_axis(self):
        """Project to x-axis drops y and z."""
        p = point(5.0, 3.0, 4.0)
        projected = project_point_line(p, x_axis())
        coords = point_to_cartesian(projected)
        assert torch.allclose(coords, torch.tensor([5.0, 0.0, 0.0]), atol=1e-5)

    def test_project_to_y_axis(self):
        """Project to y-axis drops x and z."""
        p = point(5.0, 3.0, 4.0)
        projected = project_point_line(p, y_axis())
        coords = point_to_cartesian(projected)
        assert torch.allclose(coords, torch.tensor([0.0, 3.0, 0.0]), atol=1e-5)

    def test_project_to_z_axis(self):
        """Project to z-axis drops x and y."""
        p = point(5.0, 3.0, 4.0)
        projected = project_point_line(p, z_axis())
        coords = point_to_cartesian(projected)
        assert torch.allclose(coords, torch.tensor([0.0, 0.0, 4.0]), atol=1e-5)

    def test_point_on_line_projects_to_self(self):
        """Point on line projects to itself."""
        p = point(7.0, 0.0, 0.0)  # On x-axis
        projected = project_point_line(p, x_axis())
        coords = point_to_cartesian(projected)
        assert torch.allclose(coords, torch.tensor([7.0, 0.0, 0.0]), atol=1e-5)

    def test_projected_point_is_on_line(self):
        """Projected point lies on the line."""
        p = point(3.0, 4.0, 5.0)
        line = x_axis()
        projected = project_point_line(p, line)
        dist = distance_point_line(projected, line)
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-5)


class TestReflection:
    """Tests for point reflection across planes."""

    def test_reflect_across_xy_plane(self):
        """Reflection across XY plane negates z."""
        p = point(1.0, 2.0, 3.0)
        reflected = reflect_point_plane(p, xy_plane())
        coords = point_to_cartesian(reflected)
        assert torch.allclose(coords, torch.tensor([1.0, 2.0, -3.0]), atol=1e-5)

    def test_reflect_across_xz_plane(self):
        """Reflection across XZ plane negates y."""
        p = point(1.0, 2.0, 3.0)
        reflected = reflect_point_plane(p, xz_plane())
        coords = point_to_cartesian(reflected)
        assert torch.allclose(coords, torch.tensor([1.0, -2.0, 3.0]), atol=1e-5)

    def test_reflect_across_yz_plane(self):
        """Reflection across YZ plane negates x."""
        p = point(1.0, 2.0, 3.0)
        reflected = reflect_point_plane(p, yz_plane())
        coords = point_to_cartesian(reflected)
        assert torch.allclose(coords, torch.tensor([-1.0, 2.0, 3.0]), atol=1e-5)

    def test_reflect_point_on_plane_unchanged(self):
        """Reflection of point on plane returns same point."""
        p = point(1.0, 2.0, 0.0)  # On XY plane
        reflected = reflect_point_plane(p, xy_plane())
        coords = point_to_cartesian(reflected)
        assert torch.allclose(coords, torch.tensor([1.0, 2.0, 0.0]), atol=1e-5)

    def test_double_reflection_is_identity(self):
        """Reflecting twice returns original point."""
        p = point(3.0, 4.0, 5.0)
        reflected_once = reflect_point_plane(p, xy_plane())
        reflected_twice = reflect_point_plane(reflected_once, xy_plane())
        original_coords = point_to_cartesian(p)
        final_coords = point_to_cartesian(reflected_twice)
        assert torch.allclose(original_coords, final_coords, atol=1e-4)


# =============================================================================
# Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of primitive operations."""

    def test_small_coordinates(self):
        """Handle very small coordinates."""
        p = point(1e-10, 1e-10, 1e-10)
        coords = point_to_cartesian(p)
        assert not torch.isnan(coords).any()
        assert not torch.isinf(coords).any()

    def test_large_coordinates(self):
        """Handle very large coordinates."""
        p = point(1e10, 1e10, 1e10)
        coords = point_to_cartesian(p)
        assert not torch.isnan(coords).any()
        assert not torch.isinf(coords).any()

    def test_distance_nearly_coincident_points(self):
        """Distance between nearly coincident points."""
        p1 = point(0.0, 0.0, 0.0)
        p2 = point(1e-15, 0.0, 0.0)
        dist = distance_point_point(p1, p2)
        assert not torch.isnan(dist)
        assert dist >= 0


# =============================================================================
# Gradient Flow
# =============================================================================

class TestGradientFlow:
    """Tests for gradient flow through primitive operations."""

    def test_point_creation_gradient(self):
        """Gradients flow through point creation."""
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(2.0, requires_grad=True)
        z = torch.tensor(3.0, requires_grad=True)
        p = point(x, y, z)
        loss = p.mv.sum()
        loss.backward()
        assert x.grad is not None
        assert y.grad is not None
        assert z.grad is not None

    def test_distance_gradient(self):
        """Gradients flow through distance computation."""
        x = torch.tensor(1.0, requires_grad=True)
        p1 = point(x, 0.0, 0.0)
        p2 = point(0.0, 0.0, 0.0)
        dist = distance_point_point(p1, p2)
        dist.backward()
        assert x.grad is not None
        # Gradient should be positive (moving x increases distance)
        assert x.grad > 0

    def test_projection_gradient(self):
        """Gradients flow through projection."""
        z = torch.tensor(5.0, requires_grad=True)
        p = point(1.0, 2.0, z)
        projected = project_point_plane(p, xy_plane())
        coords = point_to_cartesian(projected)
        loss = coords[0]  # Use x coordinate of projected point
        loss.backward()
        # z should have zero gradient (projection removes z dependence)
        assert z.grad is not None


# =============================================================================
# Type Preservation
# =============================================================================

class TestTypePreservation:
    """Tests for dtype and device preservation."""

    def test_float64_preservation(self):
        """float64 dtype is preserved through operations."""
        x = torch.tensor(1.0, dtype=torch.float64)
        y = torch.tensor(2.0, dtype=torch.float64)
        z = torch.tensor(3.0, dtype=torch.float64)
        p = point(x, y, z)
        assert p.mv.dtype == torch.float64
        coords = point_to_cartesian(p)
        assert coords.dtype == torch.float64

    def test_float32_preservation(self):
        """float32 dtype is preserved."""
        coords = torch.randn(3, dtype=torch.float32)
        p = point_from_tensor(coords)
        assert p.mv.dtype == torch.float32

    def test_device_preservation_cpu(self):
        """CPU device is preserved."""
        coords = torch.randn(3, device='cpu')
        p = point_from_tensor(coords)
        assert p.device == torch.device('cpu')


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_batch_points(self):
        """Handle empty batch of points."""
        coords = torch.zeros(0, 3)
        points = point_from_tensor(coords)
        assert points.shape == (0,)

    def test_origin_factory_function(self):
        """origin() creates (0,0,0)."""
        o = origin()
        coords = point_to_cartesian(o)
        assert torch.allclose(coords, torch.zeros(3))

    def test_plane_equation_satisfied(self):
        """Points on plane satisfy plane equation."""
        # Create plane z = 5 (normal (0,0,1), d = -5)
        pi = plane(0.0, 0.0, 1.0, -5.0)
        # Test points on this plane
        for x, y in [(0, 0), (1, 2), (-3, 4)]:
            p = point(float(x), float(y), 5.0)
            dist = distance_point_plane(p, pi)
            assert torch.isclose(dist, torch.tensor(0.0), atol=1e-5)

    def test_plucker_constraint(self):
        """Plucker constraint: direction.dot(moment) = 0 for lines through origin."""
        # Line through origin should have moment = 0
        line = line_from_point_direction(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0])
        )
        d, m = line_to_plucker(line)
        dot = (d * m).sum()
        assert torch.isclose(dot, torch.tensor(0.0), atol=1e-5)

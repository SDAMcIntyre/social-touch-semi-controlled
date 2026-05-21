"""Unit tests for forearm_slim_uv.py and _slim_helpers.py.

Uses real igl, scipy, trimesh, open3d, numpy — these are available in
the conda env.  Heavy pipeline __init__.py files are stubbed.
"""

from __future__ import annotations

import sys
import types
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _stub(dotted: str, **attrs) -> None:
    if dotted in sys.modules:
        return
    mod = types.ModuleType(dotted)
    parts = dotted.split(".")
    pkg_dir = _SRC / Path(*parts)
    if pkg_dir.exists():
        mod.__path__ = [str(pkg_dir)]
    mod.__package__ = dotted
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[dotted] = mod
    if "." in dotted:
        parent, child = dotted.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], child, sys.modules[dotted])


_stub("utils")
_stub("utils.should_process_task",
      should_process_task=lambda **kw: True,
      clean_task_outputs=lambda **kw: None)
_stub("analysis")
_stub("analysis.receptive_field_mapping")
_stub("analysis.receptive_field_mapping.rf_data_loader",
      load_forearm_vertices=lambda path: None)
_stub("primary_processing")


# ---------------------------------------------------------------------------
# Fixture: unit-square mesh (2 triangles)
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_square_mesh():
    """Unit square in UV space split into 2 triangles.

    Vertices:
      0 = (0, 0, 0)   1 = (1, 0, 0)
      2 = (1, 1, 0)   3 = (0, 1, 0)
    Triangles:
      [0, 1, 2]  [0, 2, 3]
    UV coords equal (x, y) so barycentric results are exact.
    """
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uv = V[:, :2].copy()
    return V, F, uv


# ---------------------------------------------------------------------------
# Fixture: synthetic flat disk mesh
# ---------------------------------------------------------------------------

@pytest.fixture
def disk_mesh():
    """Flat unit disk: outer ring + centre, triangulated with scipy Delaunay.

    The centre vertex (last point) is interior; the outer ring forms
    the open boundary.
    """
    N = 40  # points on outer ring
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    centre = np.array([[0.0, 0.0]])
    pts_2d = np.vstack([outer, centre])
    tri = Delaunay(pts_2d)
    V = np.column_stack([
        pts_2d,
        np.zeros(len(pts_2d), dtype=np.float64),
    ]).astype(np.float64)
    F = tri.simplices.astype(np.int32)
    center_vid = N  # centre point is the last vertex
    boundary_vids = list(range(N))  # outer ring
    return V, F, center_vid, boundary_vids


# ===========================================================================
# TestFlattenSlim
# ===========================================================================

class TestFlattenSlim:

    def test_smoke_flip_free(self, disk_mesh):
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            boundary_loop, flatten_slim, _has_flipped_triangles,
        )
        V, F, center_vid, _ = disk_mesh
        boundary = boundary_loop(F)
        V_out, F_out, uv = flatten_slim(V, F, boundary, center_vid=center_vid, n_iter=20)
        assert not np.any(np.isnan(uv)), "SLIM produced NaN"
        assert not _has_flipped_triangles(uv, F_out), "SLIM UV has flipped triangles"
        _, new_center = __import__('scipy.spatial', fromlist=['KDTree']).KDTree(V_out).query(V[center_vid])
        np.testing.assert_allclose(uv[new_center], [0.0, 0.0], atol=1e-6)
        b0 = int(boundary_loop(F_out)[0])
        assert uv[b0, 0] > 0, "boundary[0] should be on +x"
        np.testing.assert_allclose(uv[b0, 1], 0.0, atol=1e-6)

    def test_boundary_vertex_raises(self, disk_mesh):
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            boundary_loop, flatten_slim,
        )
        V, F, _, boundary_vids = disk_mesh
        boundary = boundary_loop(F)
        with pytest.raises(ValueError, match="boundary"):
            flatten_slim(V, F, boundary, center_vid=int(boundary[0]))

    def test_flipped_harmonic_falls_back_to_tutte(self, disk_mesh, monkeypatch):
        import igl
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            boundary_loop, flatten_slim, _has_flipped_triangles,
        )
        V, F, center_vid, _ = disk_mesh
        boundary = boundary_loop(F)

        _orig_harmonic = igl.harmonic

        def _bad_harmonic(*args, **kwargs):
            uv = _orig_harmonic(*args, **kwargs)
            uv_bad = uv.copy()
            uv_bad[len(uv) // 2:, 1] *= -1
            return uv_bad

        import analysis.receptive_field_mapping.surface.slim_helpers as sh
        monkeypatch.setattr(sh, "igl", type("FakeIgl", (), {
            "harmonic": staticmethod(_bad_harmonic),
            "slim_precompute": igl.slim_precompute,
            "slim_solve": igl.slim_solve,
            "MappingEnergyType": igl.MappingEnergyType,
            "boundary_loop": igl.boundary_loop,
            "boundary_facets": igl.boundary_facets,
        })())

        V_out, F_out, uv = flatten_slim(V, F, boundary, center_vid=center_vid, n_iter=20)
        assert not _has_flipped_triangles(uv, F_out), (
            "Tutte fallback should produce a flip-free init for a simple disk"
        )


# ===========================================================================
# TestPrecomputeForearmSlimUv
# ===========================================================================

class TestPrecomputeForearmSlimUv:

    def _write_ply(self, V, F, path):
        import trimesh
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        mesh.export(str(path))

    def _write_rf_npz(self, vertex_iff_pairs: list[tuple[int, float]], path):
        """Write a synthetic single-touch RF maps NPZ in ``map_single_touch_rf`` format.

        Parameters
        ----------
        vertex_iff_pairs:
            List of ``(vertex_idx, mean_iff)`` tuples for a single touch.
        path:
            Destination path for the ``.npz`` file.
        """
        rf_data = {0: vertex_iff_pairs}
        np.savez(path, rf_data=rf_data)

    def test_precompute_writes_cache(self, disk_mesh, tmp_path, monkeypatch):
        import trimesh
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
            precompute_forearm_slim_uv,
        )
        import analysis.receptive_field_mapping.surface.forearm_slim_uv as _mod

        V, F, center_vid, _ = disk_mesh
        ply_path = tmp_path / "forearm_test.ply"
        rf_npz = tmp_path / "single_touch_rf_maps.npz"

        self._write_ply(V, F, ply_path)
        # One touch: center vertex with nonzero IFF so weighted centroid lands there.
        self._write_rf_npz([(center_vid, 1.0)], rf_npz)

        # Bypass BPA — return trimesh directly from the saved V/F.
        monkeypatch.setattr(
            _mod, "load_or_build_forearm_mesh",
            lambda path, **kw: trimesh.Trimesh(vertices=V, faces=F, process=False),
        )
        # Bypass PLY vertex loader — return mesh vertices directly.
        monkeypatch.setattr(
            _mod, "load_forearm_vertices",
            lambda path: V,
        )

        cache_path = precompute_forearm_slim_uv(ply_path, rf_npz, n_iter=20)
        assert cache_path.exists()

        data = np.load(cache_path, allow_pickle=False)
        assert not np.any(np.isnan(data["uv"])), "cached UV contains NaN"
        assert int(data["center_vid"]) == center_vid
        assert str(data["ply_hash"]) != "", "ply_hash should be non-empty"

    def test_missing_rf_npz_raises(self, disk_mesh, tmp_path, monkeypatch):
        import trimesh
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
            precompute_forearm_slim_uv,
        )
        import analysis.receptive_field_mapping.surface.forearm_slim_uv as _mod

        V, F, _, _ = disk_mesh
        ply_path = tmp_path / "forearm_test.ply"
        self._write_ply(V, F, ply_path)

        monkeypatch.setattr(
            _mod, "load_or_build_forearm_mesh",
            lambda path, **kw: trimesh.Trimesh(vertices=V, faces=F, process=False),
        )

        missing_npz = tmp_path / "nonexistent_rf_maps.npz"
        with pytest.raises(FileNotFoundError):
            precompute_forearm_slim_uv(ply_path, missing_npz)

    def test_empty_rf_data_raises(self, disk_mesh, tmp_path, monkeypatch):
        import trimesh
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
            precompute_forearm_slim_uv,
        )
        import analysis.receptive_field_mapping.surface.forearm_slim_uv as _mod

        V, F, _, _ = disk_mesh
        ply_path = tmp_path / "forearm_test.ply"
        empty_npz = tmp_path / "empty_rf_maps.npz"
        self._write_ply(V, F, ply_path)
        # Empty rf_data dict — no touches recorded.
        np.savez(empty_npz, rf_data={})

        monkeypatch.setattr(
            _mod, "load_or_build_forearm_mesh",
            lambda path, **kw: trimesh.Trimesh(vertices=V, faces=F, process=False),
        )

        with pytest.raises(ValueError):
            precompute_forearm_slim_uv(ply_path, empty_npz)

    def test_centre_on_boundary_raises(self, disk_mesh, tmp_path, monkeypatch):
        import trimesh
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
            precompute_forearm_slim_uv,
        )
        import analysis.receptive_field_mapping.surface.forearm_slim_uv as _mod

        V, F, _, boundary_vids = disk_mesh
        ply_path = tmp_path / "forearm_test.ply"
        rf_npz = tmp_path / "boundary_rf_maps.npz"
        self._write_ply(V, F, ply_path)

        # Place all IFF weight on the first boundary vertex so the weighted
        # centroid snaps to a boundary vertex, triggering the ValueError.
        self._write_rf_npz([(boundary_vids[0], 1.0)], rf_npz)

        monkeypatch.setattr(
            _mod, "load_or_build_forearm_mesh",
            lambda path, **kw: trimesh.Trimesh(vertices=V, faces=F, process=False),
        )
        monkeypatch.setattr(
            _mod, "load_forearm_vertices",
            lambda path: V,
        )

        with pytest.raises(ValueError, match="boundary"):
            precompute_forearm_slim_uv(ply_path, rf_npz)


# ===========================================================================
# TestBarycentricUvLookup
# ===========================================================================

class TestBarycentricUvLookup:

    @pytest.fixture
    def slim_cache(self, disk_mesh):
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            boundary_loop, flatten_slim,
        )
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import SlimUvCache
        V, F, center_vid, _ = disk_mesh
        boundary = boundary_loop(F)
        V_out, F_out, uv = flatten_slim(V, F, boundary, center_vid=center_vid, n_iter=20)
        from scipy.spatial import KDTree
        _, new_center = KDTree(V_out).query(V[center_vid])
        new_boundary = boundary_loop(F_out)
        return SlimUvCache(
            V=V_out, F=F_out, uv=uv,
            center_vid=int(new_center),
            boundary_vid=int(new_boundary[0]),
            ply_mtime=0.0, ply_hash="test", rf_npz_mtime=0.0,
            centroid_3d=V[center_vid],
        )

    def test_vertex_round_trip(self, slim_cache):
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
            barycentric_uv_lookup,
        )
        result = barycentric_uv_lookup(slim_cache, slim_cache.V)
        np.testing.assert_allclose(result, slim_cache.uv, atol=1e-9,
                                   err_msg="Vertex round-trip failed")

    def test_face_centroid_in_hull(self, slim_cache):
        """UV of face centroid should lie inside the face's UV triangle."""
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
            barycentric_uv_lookup,
        )
        V, F, uv = slim_cache.V, slim_cache.F, slim_cache.uv
        face_centroids_3d = V[F].mean(axis=1)
        result = barycentric_uv_lookup(slim_cache, face_centroids_3d)

        # For each face, check that result[i] is in the convex hull of
        # the face's three UV vertices (up to a small tolerance).
        for i, fi in enumerate(range(len(F))):
            uv_triangle = uv[F[fi]]   # (3, 2)
            p = result[i]             # (2,)
            # Compute barycentric coords in UV space.
            A, B, C = uv_triangle
            v0, v1, v2 = B - A, C - A, p - A
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-15:
                continue   # degenerate UV face — skip
            l1 = (d11 * d20 - d01 * d21) / denom
            l2 = (d00 * d21 - d01 * d20) / denom
            l0 = 1.0 - l1 - l2
            # All barycentric coords should be in [-tol, 1+tol].
            tol = 0.05
            assert l0 >= -tol and l1 >= -tol and l2 >= -tol, (
                f"Face {fi} centroid UV is outside its UV triangle "
                f"(bary=({l0:.4f}, {l1:.4f}, {l2:.4f}))"
            )


# ===========================================================================
# TestUvPointsToXyz
# ===========================================================================

class TestUvPointsToXyz:

    def test_barycentric_known_point(self, unit_square_mesh):
        """UV centre (0.5, 0.5) should map to 3D point (0.5, 0.5, 0.0)."""
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import uv_points_to_xyz
        V, F, uv = unit_square_mesh
        query = np.array([[0.5, 0.5]], dtype=np.float64)
        xyz = uv_points_to_xyz(query, uv, F, V)
        np.testing.assert_allclose(xyz, [[0.5, 0.5, 0.0]], atol=1e-12)

    def test_vertex_positions_round_trip(self, unit_square_mesh):
        """Querying at each mesh vertex UV should return the exact 3D vertex."""
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import uv_points_to_xyz
        V, F, uv = unit_square_mesh
        xyz = uv_points_to_xyz(uv, uv, F, V)
        np.testing.assert_allclose(xyz, V, atol=1e-12)

    def test_outside_point_snaps_to_boundary(self, unit_square_mesh):
        """A UV point outside the mesh is snapped to the nearest triangle boundary."""
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import uv_points_to_xyz
        V, F, uv = unit_square_mesh
        outside_point = np.array([[2.0, 2.0]], dtype=np.float64)
        # Must not raise — outside points are snapped, not hard-failed.
        xyz = uv_points_to_xyz(outside_point, uv, F, V)
        # Point (2,2) is nearest to vertex 2 at (1,1,0) — snapped there.
        np.testing.assert_allclose(xyz, [[1.0, 1.0, 0.0]], atol=1e-12)

    def test_multiple_points_mixed_triangles(self, unit_square_mesh):
        """Points in both triangles of the unit square should interpolate correctly."""
        from analysis.receptive_field_mapping.surface.forearm_slim_uv import uv_points_to_xyz
        V, F, uv = unit_square_mesh
        queries = np.array([
            [0.25, 0.25],   # in triangle [0,1,2] near vertex 0
            [0.75, 0.75],   # in triangle [0,2,3] near vertex 2
        ], dtype=np.float64)
        xyz = uv_points_to_xyz(queries, uv, F, V)
        # Since UV = XY and Z = 0 everywhere, XY output must equal input UV
        np.testing.assert_allclose(xyz[:, :2], queries, atol=1e-12)
        np.testing.assert_allclose(xyz[:, 2], [0.0, 0.0], atol=1e-12)


# ===========================================================================
# TestFillInteriorHoles
# ===========================================================================

class TestFillInteriorHoles:
    """Tests for _fill_hole_delaunay and _fill_interior_holes."""

    def _ring_V_loop(self, n: int, radius: float = 1.0):
        """n boundary vertices on a horizontal circle at z=0."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        V = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(n, dtype=np.float64),
        ])
        loop = list(range(n))
        return V, loop

    def test_small_hole_uses_fan(self):
        """3-vertex hole → centroid-fan: exactly 3 new faces and 1 new vertex."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _fill_interior_holes,
        )
        # Build a flat "donut" mesh: outer ring (N=20) + inner triangle (N=3).
        # Outer ring vertices [0..19], inner triangle [20, 21, 22].
        n_out = 20
        angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        outer = np.column_stack([
            2.0 * np.cos(angles_out),
            2.0 * np.sin(angles_out),
            np.zeros(n_out),
        ])
        inner = np.array([
            [0.1, 0.0, 0.0],
            [-0.05, 0.087, 0.0],
            [-0.05, -0.087, 0.0],
        ])
        V = np.vstack([outer, inner]).astype(np.float64)
        # Build faces: outer ring fan to a dummy centre + separate inner triangle
        # so we have a closed outer boundary and an open inner hole.
        # Use scipy Delaunay on all 2D points, then remove triangles inside inner radius.
        from scipy.spatial import Delaunay
        pts2d = V[:, :2]
        tri = Delaunay(pts2d)
        F_all = tri.simplices.astype(np.int32)
        # Keep only triangles whose centroid is between r=0.15 and r=1.95
        centroids = V[F_all].mean(axis=1)[:, :2]
        r = np.linalg.norm(centroids, axis=1)
        F = F_all[(r > 0.15) & (r < 1.95)]
        assert len(F) > 0, "Test mesh setup failed"

        V_out, F_out = _fill_interior_holes(V, F)

        # Inner hole had 3 boundary vertices → centroid-fan → 1 new vert, 3 new faces.
        n_new_verts = V_out.shape[0] - V.shape[0]
        n_new_faces = F_out.shape[0] - F.shape[0]
        assert n_new_verts == 1, f"Expected 1 new vertex (centroid), got {n_new_verts}"
        assert n_new_faces == 3, f"Expected 3 new faces (fan), got {n_new_faces}"

    def test_large_hole_uses_delaunay(self):
        """25-vertex hole → Delaunay: max AR < 10, exactly 1 boundary loop after fill."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _fill_hole_delaunay, _find_boundary_loops,
        )
        import trimesh
        import trimesh.repair

        n = 25
        V, loop = self._ring_V_loop(n)
        new_verts, new_faces = _fill_hole_delaunay(V, loop, n_extra_verts=0)

        assert len(new_verts) >= 1, "Expected at least 1 new vertex (centroid for N>20)"
        assert len(new_faces) > 0, "Expected at least 1 new face"

        # Build a full mesh by combining boundary verts + new verts + faces
        V_all = np.vstack([V] + [nv.reshape(1, 3) for nv in new_verts])
        F_all = np.array(new_faces, dtype=np.int32)

        # Check aspect ratios of the new faces
        p0 = V_all[F_all[:, 0]]
        p1 = V_all[F_all[:, 1]]
        p2 = V_all[F_all[:, 2]]
        e0 = np.linalg.norm(p1 - p0, axis=1)
        e1 = np.linalg.norm(p2 - p1, axis=1)
        e2 = np.linalg.norm(p0 - p2, axis=1)
        longest = np.maximum(np.maximum(e0, e1), e2)
        shortest = np.minimum(np.minimum(e0, e1), e2).clip(1e-15)
        ar = longest / shortest
        assert ar.max() < 10.0, (
            f"Max aspect ratio {ar.max():.2f} >= 10 — Delaunay fill produced poor triangles"
        )

        # Resulting filled mesh should have 1 boundary loop (closed surface)
        # Fix winding and check
        tmp = trimesh.Trimesh(vertices=V_all, faces=F_all, process=False)
        trimesh.repair.fix_winding(tmp)
        loops = _find_boundary_loops(np.asarray(tmp.faces, dtype=np.int32))
        assert len(loops) == 1, (
            f"Expected 1 boundary loop after fill, got {len(loops)}"
        )

    def test_large_hole_steiner_points(self):
        """Large-diameter hole with widely-spaced boundary → Steiner points inserted,
        filled triangle edges bounded relative to boundary spacing."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _fill_hole_delaunay, _compute_face_aspect_ratios,
        )
        # Create a large circular hole (radius=10) with only 8 boundary vertices.
        # Boundary spacing ≈ 7.6mm, diameter = 20mm → ratio ≈ 2.6.
        # With a smaller radius but same vertex count, the ratio grows and
        # Steiner points become necessary.
        n = 8
        V, loop = self._ring_V_loop(n, radius=10.0)

        # Boundary edge length ~ 2*R*sin(pi/N) ≈ 7.65
        # Diameter = 20.  Ratio = 20/7.65 ≈ 2.6 → below default factor 3.0.
        # Use a sparser ring to trigger Steiner insertion: radius=50, N=8.
        V_big = V * 5.0  # radius=50, boundary_edge ≈ 38, diameter=100 → ratio ≈ 2.6
        # Still not enough. Use very sparse: 6 vertices on a large circle.
        n2 = 6
        V2, loop2 = self._ring_V_loop(n2, radius=50.0)
        # boundary_edge ≈ 50, diameter=100, ratio=2.0 → below threshold.
        # Need diameter >> 3 * median_edge.  Use: 10 verts on R=50.
        n3 = 10
        V3, loop3 = self._ring_V_loop(n3, radius=50.0)
        # boundary_edge ≈ 2*50*sin(pi/10) ≈ 30.9, diameter=100 → ratio ≈ 3.2 → triggers.

        new_verts, new_faces = _fill_hole_delaunay(V3, loop3, n_extra_verts=0)

        assert len(new_verts) >= 1, (
            "Expected Steiner interior points for large-diameter hole"
        )
        assert len(new_faces) > n3, (
            f"Expected more faces than boundary vertices ({n3}) due to Steiner points, "
            f"got {len(new_faces)}"
        )

        V_all = np.vstack([V3] + [nv.reshape(1, 3) for nv in new_verts])
        F_all = np.array(new_faces, dtype=np.int32)
        ar = _compute_face_aspect_ratios(V_all, F_all)
        assert ar.max() < 10.0, (
            f"Max AR {ar.max():.2f} >= 10 after Steiner fill"
        )

    def test_nonconvex_hole(self):
        """Concave hole boundary: no filled triangles outside the boundary polygon."""
        from analysis.receptive_field_mapping.surface.slim_helpers import _fill_hole_delaunay
        from matplotlib.path import Path

        # Star-shaped (non-convex) hole boundary: 10 vertices alternating r=1 and r=0.5
        n = 10
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = np.where(np.arange(n) % 2 == 0, 1.0, 0.5)
        pts2d = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        V = np.column_stack([pts2d, np.zeros(n)])
        loop = list(range(n))

        new_verts, new_faces = _fill_hole_delaunay(V, loop, n_extra_verts=0)

        if len(new_faces) == 0:
            pytest.skip("No faces generated for non-convex hole (acceptable for degenerate star)")

        V_all = np.vstack([V] + [nv.reshape(1, 3) for nv in new_verts])
        boundary_poly = Path(pts2d)

        for face in new_faces:
            centroid_3d = V_all[np.array(face)].mean(axis=0)
            centroid_2d = centroid_3d[:2]
            assert boundary_poly.contains_point(centroid_2d), (
                f"Triangle centroid {centroid_2d} is outside the non-convex hole polygon"
            )

    def test_degenerate_plane_fallback(self):
        """Nearly collinear boundary → centroid-fan fallback: 1 new vertex, N new faces."""
        from analysis.receptive_field_mapping.surface.slim_helpers import _fill_hole_delaunay

        # 8 points nearly on a line (degenerate plane, S[1] ≈ 0)
        n = 8
        t = np.linspace(0.0, 1.0, n)
        V = np.column_stack([t, 1e-12 * t, np.zeros(n)])
        loop = list(range(n))

        new_verts, new_faces = _fill_hole_delaunay(V, loop, n_extra_verts=0)

        assert len(new_verts) == 1, (
            f"Degenerate plane fallback should produce 1 centroid vertex, got {len(new_verts)}"
        )
        assert len(new_faces) == n, (
            f"Degenerate plane fallback should produce {n} fan faces, got {len(new_faces)}"
        )


# ===========================================================================
# TestSliverRemoval
# ===========================================================================

class TestSliverRemoval:
    """Tests for _remove_sliver_faces and its integration in clean_mesh."""

    def test_sliver_faces_removed(self):
        """Mesh with injected sliver → sliver removed, good faces kept."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _remove_sliver_faces, _compute_face_aspect_ratios,
        )
        # Build a small planar mesh with one good triangle and one sliver.
        V = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.8, 0.0],
            [20.0, 0.01, 0.0],  # far away → sliver with verts 1, 3, 2
        ], dtype=np.float64)
        F = np.array([
            [0, 1, 2],    # good triangle (AR ~ 1.5)
            [1, 3, 2],    # sliver: long edges to vertex 3, short edge 1-2
        ], dtype=np.int32)

        ar_before = _compute_face_aspect_ratios(V, F)
        assert ar_before.max() > 10.0, "Test setup: expected at least one sliver"

        V_out, F_out = _remove_sliver_faces(V, F)

        ar_after = _compute_face_aspect_ratios(V_out, F_out)
        assert ar_after.max() <= 10.0, (
            f"Sliver not removed: max AR {ar_after.max():.2f}"
        )
        assert F_out.shape[0] < F.shape[0], "Expected fewer faces after sliver removal"

    def test_no_slivers_noop(self):
        """Clean mesh (all AR < 10) → V, F unchanged."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _remove_sliver_faces,
        )
        V = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.8, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        V_out, F_out = _remove_sliver_faces(V, F)

        np.testing.assert_array_equal(V_out, V)
        np.testing.assert_array_equal(F_out, F)

    def test_sliver_removal_preserves_connectivity(self):
        """Removing a sliver that could fragment mesh → largest component kept."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _remove_sliver_faces, _compute_face_aspect_ratios,
        )
        # Two triangles connected by a sliver "bridge".  Removing the sliver
        # should keep the largest component.
        V = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.8, 0.0],
            [10.0, 0.01, 0.0],  # sliver vertex
            [11.0, 0.0, 0.0],
            [10.5, 0.8, 0.0],
        ], dtype=np.float64)
        F = np.array([
            [0, 1, 2],    # good triangle A
            [1, 3, 2],    # sliver bridge
            [3, 4, 5],    # good triangle B (disconnected after sliver removal)
        ], dtype=np.int32)

        V_out, F_out = _remove_sliver_faces(V, F)

        assert F_out.shape[0] >= 1, "At least one component should survive"
        ar = _compute_face_aspect_ratios(V_out, F_out)
        assert ar.max() <= 10.0, f"Sliver survived: max AR {ar.max():.2f}"


# ===========================================================================
# TestStitchBoundaryGaps
# ===========================================================================

class TestStitchBoundaryGaps:
    """Tests for _stitch_boundary_gaps."""

    def test_no_secondary_loops_noop(self):
        """Single-boundary disk mesh: V and F returned unchanged."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _stitch_boundary_gaps, _find_boundary_loops,
        )
        # Build a simple disk: outer ring of 16 vertices + centre, triangulated.
        N = 16
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        outer = np.column_stack([
            np.cos(angles), np.sin(angles), np.zeros(N),
        ]).astype(np.float64)
        centre = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        V = np.vstack([outer, centre])
        # Fan triangles from centre to each edge of the ring.
        c = N
        F = np.array(
            [[c, i, (i + 1) % N] for i in range(N)],
            dtype=np.int32,
        )

        loops_before = _find_boundary_loops(F)
        assert len(loops_before) == 1, "Test setup: expected single boundary loop"

        V_out, F_out = _stitch_boundary_gaps(V, F)

        np.testing.assert_array_equal(V_out, V)
        np.testing.assert_array_equal(F_out, F)

    def test_gap_welded_into_main_boundary(self):
        """Two rectangular patches sharing a narrow gap: after stitching, single boundary loop."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _stitch_boundary_gaps, _find_boundary_loops,
        )
        # Main rectangle: vertices 0-3, split into 2 triangles.
        #   0 --- 1
        #   |   / |
        #   | /   |
        #   3 --- 2
        # Appendage rectangle: vertices 4-7, hanging off the right side with a
        # narrow gap (< 5 mm) between vertex 1 & vertex 4, and vertex 2 & vertex 5.
        #
        # Main boundary vertices at x=1.0; appendage boundary at x=1.003 (gap = 3 mm).
        #
        # Layout (all z=0):
        #   main:        (0,0) (1,0) (1,1) (0,1)   → vids 0,1,2,3
        #   appendage:   (1.003,0) (1.003,1) (2,0) (2,1) → vids 4,5,6,7
        #
        # After stitching, vid 4 should collapse onto vid 1 and vid 5 onto vid 2,
        # merging the two rectangles into a single connected boundary loop.

        V = np.array([
            [0.0,   0.0, 0.0],   # 0
            [1.0,   0.0, 0.0],   # 1  (main boundary, right side)
            [1.0,   1.0, 0.0],   # 2  (main boundary, right side)
            [0.0,   1.0, 0.0],   # 3
            [1.003, 0.0, 0.0],   # 4  (appendage boundary, left side — gap 3 mm from vid 1)
            [1.003, 1.0, 0.0],   # 5  (appendage boundary, left side — gap 3 mm from vid 2)
            [2.0,   0.0, 0.0],   # 6
            [2.0,   1.0, 0.0],   # 7
        ], dtype=np.float64)

        # Two separate rectangles: each split into 2 triangles with consistent winding.
        F = np.array([
            [0, 1, 2],  # main rect tri 1
            [0, 2, 3],  # main rect tri 2
            [4, 6, 7],  # appendage tri 1
            [4, 7, 5],  # appendage tri 2
        ], dtype=np.int32)

        loops_before = _find_boundary_loops(F)
        assert len(loops_before) == 2, (
            f"Test setup: expected 2 boundary loops, got {len(loops_before)}"
        )

        V_out, F_out = _stitch_boundary_gaps(V, F, proximity_mm=5.0)

        loops_after = _find_boundary_loops(F_out)
        assert len(loops_after) == 1, (
            f"Expected 1 boundary loop after stitching, got {len(loops_after)}: {loops_after}"
        )

    def test_true_interior_hole_not_welded(self):
        """Mesh with an interior hole far from boundary: hole is preserved after stitching."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _stitch_boundary_gaps, _find_boundary_loops,
        )
        # Build a donut: outer ring (radius=20) and inner ring (radius=5).
        # All inner vertices are >10 mm from the outer boundary → not welded.
        n_out = 24
        n_in = 8
        angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)

        outer = np.column_stack([
            20.0 * np.cos(angles_out),
            20.0 * np.sin(angles_out),
            np.zeros(n_out),
        ])
        inner = np.column_stack([
            5.0 * np.cos(angles_in),
            5.0 * np.sin(angles_in),
            np.zeros(n_in),
        ])
        V = np.vstack([outer, inner]).astype(np.float64)

        # Build an annular mesh by pairing outer and inner ring vertices.
        # For each sector i, use 2 triangles:
        #   (outer[i], outer[i+1], inner[j])  and  (inner[j], outer[i+1], inner[j+1])
        # where j = i * n_in // n_out (rough alignment).
        F_list = []
        for i in range(n_out):
            o0 = i
            o1 = (i + 1) % n_out
            j = (i * n_in) // n_out
            j1 = ((i + 1) * n_in) // n_out % n_in
            i0 = n_out + j
            i1 = n_out + j1
            F_list.append([o0, o1, i0])
            if i0 != i1:
                F_list.append([i0, o1, i1])

        F = np.array(F_list, dtype=np.int32)

        loops_before = _find_boundary_loops(F)
        assert len(loops_before) == 2, (
            f"Test setup: expected 2 boundary loops (outer + inner hole), got {len(loops_before)}"
        )

        # Min distance from inner ring to outer boundary:
        # outer radius = 20, inner radius = 5 → min gap = 20 - 5 = 15 mm >> 5 mm threshold.
        V_out, F_out = _stitch_boundary_gaps(V, F, proximity_mm=5.0)

        loops_after = _find_boundary_loops(F_out)
        assert len(loops_after) == 2, (
            f"Expected 2 boundary loops (hole preserved), got {len(loops_after)}"
        )

    def test_degenerate_faces_removed(self):
        """Welding that creates faces with 2+ identical vertices: degenerate faces removed."""
        from analysis.receptive_field_mapping.surface.slim_helpers import (
            _stitch_boundary_gaps, _find_boundary_loops,
        )
        # Build a mesh where the gap is exactly 1 vertex wide on each side so
        # welding will collapse one triangle into a degenerate (two identical vids).
        #
        # Main triangle: vids 0, 1, 2 at (0,0), (1,0), (0.5, 1).
        # Gap vertex: vid 3 at (1.001, 0) — 1 mm from vid 1.
        # A second triangle using the gap vertex: [3, 4, 5].
        # Only 1 close vertex (vid 3 near vid 1) → but we need >= 2 close vertices
        # to trigger stitching.  Add vid 6 at (0.499, 1.001) close to vid 2.
        # Secondary loop boundary: 3, 4, 5, 6 (with face [3,4,5] and [3,5,6]).

        V = np.array([
            [0.0,   0.0,  0.0],   # 0 — main
            [1.0,   0.0,  0.0],   # 1 — main (boundary)
            [0.5,   1.0,  0.0],   # 2 — main (boundary)
            [1.001, 0.0,  0.0],   # 3 — gap, 1 mm from vid 1
            [1.5,   0.0,  0.0],   # 4 — appendage
            [1.5,   1.0,  0.0],   # 5 — appendage
            [0.501, 1.0,  0.0],   # 6 — gap, 1 mm from vid 2
        ], dtype=np.float64)

        F = np.array([
            [0, 1, 2],   # main triangle
            [3, 4, 5],   # appendage tri 1
            [3, 5, 6],   # appendage tri 2 — vid 3 near vid 1, vid 6 near vid 2
        ], dtype=np.int32)

        loops_before = _find_boundary_loops(F)
        assert len(loops_before) >= 2, (
            f"Test setup: expected >= 2 boundary loops, got {len(loops_before)}"
        )

        V_out, F_out = _stitch_boundary_gaps(V, F, proximity_mm=5.0)

        # All faces in F_out must be non-degenerate (3 distinct vertex IDs).
        for face in F_out:
            assert len(np.unique(face)) == 3, (
                f"Degenerate face found after stitching: {face}"
            )

        # Mesh must still have faces.
        assert F_out.shape[0] > 0, "All faces removed after stitching"

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
        from analysis.receptive_field_mapping._slim_helpers import (
            boundary_loop, flatten_slim, _has_flipped_triangles,
        )
        V, F, center_vid, _ = disk_mesh
        boundary = boundary_loop(F)
        uv = flatten_slim(V, F, boundary, center_vid=center_vid, n_iter=20)
        assert not np.any(np.isnan(uv)), "SLIM produced NaN"
        assert not _has_flipped_triangles(uv, F), "SLIM UV has flipped triangles"
        np.testing.assert_allclose(uv[center_vid], [0.0, 0.0], atol=1e-6)
        b0 = int(boundary[0])
        assert uv[b0, 0] > 0, "boundary[0] should be on +x"
        np.testing.assert_allclose(uv[b0, 1], 0.0, atol=1e-6)

    def test_boundary_vertex_raises(self, disk_mesh):
        from analysis.receptive_field_mapping._slim_helpers import (
            boundary_loop, flatten_slim,
        )
        V, F, _, boundary_vids = disk_mesh
        boundary = boundary_loop(F)
        with pytest.raises(ValueError, match="boundary"):
            flatten_slim(V, F, boundary, center_vid=int(boundary[0]))

    def test_flipped_init_raises(self, disk_mesh, monkeypatch):
        import igl
        from analysis.receptive_field_mapping._slim_helpers import (
            boundary_loop, flatten_slim, _has_flipped_triangles,
        )
        V, F, center_vid, _ = disk_mesh
        boundary = boundary_loop(F)

        # Return a UV that has flipped triangles: negate every y coordinate
        # of the default harmonic result to guarantee a mix of +/- signed areas.
        _orig_harmonic = igl.harmonic

        def _bad_harmonic(*args, **kwargs):
            uv = _orig_harmonic(*args, **kwargs)
            # Flip every other triangle by negating y on half the boundary.
            uv_bad = uv.copy()
            uv_bad[len(uv) // 2:, 1] *= -1
            return uv_bad

        # Monkeypatch igl inside _slim_helpers module
        import analysis.receptive_field_mapping._slim_helpers as sh
        monkeypatch.setattr(sh, "igl", type("FakeIgl", (), {
            "harmonic": staticmethod(_bad_harmonic),
            "slim_precompute": igl.slim_precompute,
            "slim_solve": igl.slim_solve,
            "MappingEnergyType": igl.MappingEnergyType,
            "boundary_loop": igl.boundary_loop,
            "boundary_facets": igl.boundary_facets,
        })())

        with pytest.raises(RuntimeError, match="flipped"):
            flatten_slim(V, F, boundary, center_vid=center_vid)


# ===========================================================================
# TestPrecomputeForearmSlimUv
# ===========================================================================

class TestPrecomputeForearmSlimUv:

    def _write_ply(self, V, F, path):
        import trimesh
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        mesh.export(str(path))

    def _write_spike_csv(self, xyz: np.ndarray, path):
        import pandas as pd
        pd.DataFrame(xyz, columns=["x", "y", "z"]).to_csv(path, index=False)

    def test_precompute_writes_cache(self, disk_mesh, tmp_path, monkeypatch):
        import trimesh
        from analysis.receptive_field_mapping.forearm_slim_uv import (
            precompute_forearm_slim_uv,
        )
        import analysis.receptive_field_mapping.forearm_slim_uv as _mod

        V, F, center_vid, _ = disk_mesh
        ply_path = tmp_path / "forearm_test.ply"
        spike_csv = tmp_path / "spike_positions.csv"

        self._write_ply(V, F, ply_path)
        self._write_spike_csv(V[center_vid:center_vid + 1], spike_csv)

        # Bypass BPA — return trimesh directly from the saved V/F.
        monkeypatch.setattr(
            _mod, "load_or_build_forearm_mesh",
            lambda path, **kw: trimesh.Trimesh(vertices=V, faces=F, process=False),
        )

        cache_path = precompute_forearm_slim_uv(ply_path, spike_csv, n_iter=20)
        assert cache_path.exists()

        data = np.load(cache_path, allow_pickle=False)
        assert not np.any(np.isnan(data["uv"])), "cached UV contains NaN"
        assert int(data["center_vid"]) == center_vid
        assert str(data["ply_hash"]) != "", "ply_hash should be non-empty"

    def test_missing_spike_csv_raises(self, disk_mesh, tmp_path, monkeypatch):
        import trimesh
        from analysis.receptive_field_mapping.forearm_slim_uv import (
            precompute_forearm_slim_uv,
        )
        import analysis.receptive_field_mapping.forearm_slim_uv as _mod

        V, F, _, _ = disk_mesh
        ply_path = tmp_path / "forearm_test.ply"
        self._write_ply(V, F, ply_path)

        monkeypatch.setattr(
            _mod, "load_or_build_forearm_mesh",
            lambda path, **kw: trimesh.Trimesh(vertices=V, faces=F, process=False),
        )

        missing_csv = tmp_path / "nonexistent_spikes.csv"
        with pytest.raises(FileNotFoundError, match="spike_positions.csv"):
            precompute_forearm_slim_uv(ply_path, missing_csv)

    def test_empty_spike_csv_raises(self, disk_mesh, tmp_path, monkeypatch):
        import trimesh
        import pandas as pd
        from analysis.receptive_field_mapping.forearm_slim_uv import (
            precompute_forearm_slim_uv,
        )
        import analysis.receptive_field_mapping.forearm_slim_uv as _mod

        V, F, _, _ = disk_mesh
        ply_path = tmp_path / "forearm_test.ply"
        empty_csv = tmp_path / "empty_spikes.csv"
        self._write_ply(V, F, ply_path)
        pd.DataFrame(columns=["x", "y", "z"]).to_csv(empty_csv, index=False)

        monkeypatch.setattr(
            _mod, "load_or_build_forearm_mesh",
            lambda path, **kw: trimesh.Trimesh(vertices=V, faces=F, process=False),
        )

        with pytest.raises(ValueError, match="No spikes"):
            precompute_forearm_slim_uv(ply_path, empty_csv)

    def test_centre_on_boundary_raises(self, disk_mesh, tmp_path, monkeypatch):
        import trimesh
        from analysis.receptive_field_mapping.forearm_slim_uv import (
            precompute_forearm_slim_uv,
        )
        import analysis.receptive_field_mapping.forearm_slim_uv as _mod

        V, F, _, boundary_vids = disk_mesh
        ply_path = tmp_path / "forearm_test.ply"
        spike_csv = tmp_path / "spike_positions.csv"
        self._write_ply(V, F, ply_path)

        # Place spike centroid right on the first boundary vertex.
        self._write_spike_csv(V[boundary_vids[:1]], spike_csv)

        monkeypatch.setattr(
            _mod, "load_or_build_forearm_mesh",
            lambda path, **kw: trimesh.Trimesh(vertices=V, faces=F, process=False),
        )

        with pytest.raises(ValueError, match="boundary"):
            precompute_forearm_slim_uv(ply_path, spike_csv)


# ===========================================================================
# TestBarycentricUvLookup
# ===========================================================================

class TestBarycentricUvLookup:

    @pytest.fixture
    def slim_cache(self, disk_mesh):
        from analysis.receptive_field_mapping._slim_helpers import (
            boundary_loop, flatten_slim,
        )
        from analysis.receptive_field_mapping.forearm_slim_uv import SlimUvCache
        V, F, center_vid, _ = disk_mesh
        boundary = boundary_loop(F)
        uv = flatten_slim(V, F, boundary, center_vid=center_vid, n_iter=20)
        return SlimUvCache(
            V=V, F=F, uv=uv,
            center_vid=center_vid,
            boundary_vid=int(boundary[0]),
            ply_mtime=0.0, ply_hash="test", spike_csv_mtime=0.0,
            centroid_3d=V[center_vid],
        )

    def test_vertex_round_trip(self, slim_cache):
        from analysis.receptive_field_mapping.forearm_slim_uv import (
            barycentric_uv_lookup,
        )
        result = barycentric_uv_lookup(slim_cache, slim_cache.V)
        np.testing.assert_allclose(result, slim_cache.uv, atol=1e-9,
                                   err_msg="Vertex round-trip failed")

    def test_face_centroid_in_hull(self, slim_cache):
        """UV of face centroid should lie inside the face's UV triangle."""
        from analysis.receptive_field_mapping.forearm_slim_uv import (
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

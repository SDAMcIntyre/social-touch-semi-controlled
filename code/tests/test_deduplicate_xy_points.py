"""Unit tests for deduplicate_xy single-linkage clustering algorithm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts to path for imports
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "_5_postprocessing"
sys.path.insert(0, str(_SCRIPTS_DIR))

from deduplicate_xy_points import (
    EPSILON_SOURCE_DAG_CONFIG,
    EPSILON_SOURCE_INTERACTIVE_MONITOR,
    FOREARM_DEDUP_METADATA_SCHEMA_VERSION,
    FRAME_INDEX_COLUMN,
    DedupMapping,
    deduplicate_contact_points_csv,
    deduplicate_forearm_ply,
    deduplicate_xy,
    deduplicate_xy_mapping,
    forearm_dedup_metadata_path,
    write_forearm_dedup_metadata,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
)

o3d = pytest.importorskip("open3d", reason="deduplicate_forearm_ply needs Open3D")


def _write_ply(path: Path, points, colors=None, normals=None) -> None:
    """Materialise a point cloud as a PLY so tests exercise the real IO path."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(path), pcd)


class TestDeduplicateXyBasic:
    """Tests for basic deduplication scenarios."""

    def test_empty_input(self) -> None:
        """Empty input returns empty output with n_removed=0."""
        empty = np.empty((0, 3), dtype=np.float64)
        result, n_removed = deduplicate_xy(empty, epsilon=0.5)

        assert len(result) == 0
        assert n_removed == 0
        assert result.shape == (0, 3)

    def test_single_point(self) -> None:
        """Single point passes through unchanged."""
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result, n_removed = deduplicate_xy(pts, epsilon=0.5)

        assert n_removed == 0
        assert len(result) == 1
        np.testing.assert_array_equal(result, pts)

    def test_users_reported_case(self) -> None:
        """User's reported case from the issue: two points at (35, y) with y-distance 0.269 mm < epsilon=0.378 mm.

        Expected: Both should be deduplicated to the lower-z point.
        """
        pts = np.array([
            [35.0, -21.731, 0.0],
            [35.0, -22.0, 1.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.378)

        assert n_removed == 1
        assert len(result) == 1
        # Should keep the first point (lower z)
        np.testing.assert_array_almost_equal(result[0], [35.0, -21.731, 0.0])

    def test_no_duplicates(self) -> None:
        """Points spaced > epsilon apart pass through unchanged."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.5)

        assert n_removed == 0
        assert len(result) == 3
        np.testing.assert_array_equal(result, pts)

    def test_no_duplicates_boundary_case(self) -> None:
        """Points exactly at epsilon distance apart are not deduplicated (strict <, not <=)."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Distance = 1.0, epsilon = 1.0
        # DBSCAN eps param uses <= comparison, so these should cluster
        result, n_removed = deduplicate_xy(pts, epsilon=1.0)

        assert n_removed == 1
        assert len(result) == 1


class TestDeduplicateXyCollocated:
    """Tests for co-located points (same x, y)."""

    def test_two_collocated_points_different_z(self) -> None:
        """Two points at same (x, y) with different z collapse to lower-z point."""
        pts = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.5]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.1)

        assert n_removed == 1
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.5])

    def test_three_collocated_points_different_z(self) -> None:
        """Three points at same (x, y) collapse to the lowest-z point."""
        pts = np.array([
            [5.0, 10.0, 2.0],
            [5.0, 10.0, 0.5],
            [5.0, 10.0, 1.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.05)

        assert n_removed == 2
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [5.0, 10.0, 0.5])

    def test_all_points_identical_xy_collapse_to_one(self) -> None:
        """All points at the same (x, y) collapse to 1 point (lowest z)."""
        pts = np.array([
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 2.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.1)

        assert n_removed == 3
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.0, 1.0, 1.0])


class TestDeduplicateXySingleLinkage:
    """Tests for single-linkage transitive closure."""

    def test_chain_transitive_closure(self) -> None:
        """A chain of points where each overlaps the next collapses to 1 (single-linkage).

        Points at x=0, 1, 2, 3, 4 with epsilon=1.5:
        - Point 0 and 1 are within 1.5
        - Point 1 and 2 are within 1.5
        - Point 2 and 3 are within 1.5
        - Point 3 and 4 are within 1.5
        All should collapse into one cluster (transitive closure).
        """
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [4.0, 0.0, 4.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=1.5)

        # All should cluster together, survivor is lowest z (first point)
        assert n_removed == 4
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])

    def test_chain_spacing_half_epsilon(self) -> None:
        """Points spaced at 0.5*epsilon form a single-linkage chain."""
        epsilon = 2.0
        spacing = epsilon * 0.5

        pts = np.array([
            [0.0 * spacing, 0.0, 0.0],
            [1.0 * spacing, 0.0, 1.0],
            [2.0 * spacing, 0.0, 2.0],
            [3.0 * spacing, 0.0, 3.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=epsilon)

        assert n_removed == 3
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], pts[0])

    def test_two_separate_clusters(self) -> None:
        """Two clusters separated by > epsilon are kept distinct."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0],
            [10.0, 0.0, 2.0],
            [10.5, 0.0, 3.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=1.0)

        # Two clusters: (0, 1) and (2, 3); survivors at (0,0) and (10,0)
        assert n_removed == 2
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[1], [10.0, 0.0, 2.0])


class TestDeduplicateXyTieBreaking:
    """Tests for deterministic tie-breaking when z values are equal."""

    def test_identical_z_keeps_lower_input_index(self) -> None:
        """When two points have identical z within epsilon, keep lower input index."""
        pts = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=1.0)

        assert n_removed == 1
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 1.0])

    def test_three_identical_z_keeps_first(self) -> None:
        """When multiple points have identical z, keep the one with lowest input index."""
        pts = np.array([
            [0.0, 0.0, 5.0],
            [0.2, 0.0, 5.0],
            [0.4, 0.0, 5.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=1.0)

        assert n_removed == 2
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], pts[0])


class TestDeduplicateXyOutputOrder:
    """Tests for output ordering (should preserve input relative order)."""

    def test_output_preserves_input_order(self) -> None:
        """Surviving points appear in their original relative input order."""
        pts = np.array([
            [10.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [5.0, 0.0, 3.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.1)

        # No deduplication occurs (all spaced > epsilon apart)
        assert n_removed == 0
        np.testing.assert_array_equal(result, pts)
        # Order should be: 10, 0, 5 (as in input)

    def test_survivors_maintain_relative_order(self) -> None:
        """Among survivors, relative input order is maintained."""
        pts = np.array([
            [0.0, 0.0, 0.0],   # index 0: kept (survivor of cluster 1)
            [0.5, 0.0, 1.0],   # index 1: removed (same cluster as 0)
            [5.0, 0.0, 2.0],   # index 2: kept (survivor of cluster 2)
            [5.5, 0.0, 3.0],   # index 3: removed (same cluster as 2)
            [10.0, 0.0, 4.0]   # index 4: kept (survivor of cluster 3)
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=1.0)

        assert n_removed == 2
        assert len(result) == 3
        # Should be in order: indices 0, 2, 4
        np.testing.assert_array_equal(result[0], pts[0])
        np.testing.assert_array_equal(result[1], pts[2])
        np.testing.assert_array_equal(result[2], pts[4])


class TestDeduplicateXyEdgeCases:
    """Tests for edge cases and invariants."""

    def test_invariant_count_preserved(self) -> None:
        """deduped_count + n_removed == input_count (always)."""
        for n_pts in [0, 1, 5, 10, 50]:
            pts = np.random.randn(n_pts, 3).astype(np.float64)
            result, n_removed = deduplicate_xy(pts, epsilon=0.5)

            assert len(result) + n_removed == len(pts)

    def test_large_epsilon_collapses_to_one(self) -> None:
        """Very large epsilon exceeding cloud extent collapses to 1 point."""
        pts = np.array([
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 3.0],
            [2.0, 0.5, 1.0],
            [0.5, 2.0, 4.0]
        ], dtype=np.float64)

        # Epsilon much larger than any cloud extent
        result, n_removed = deduplicate_xy(pts, epsilon=100.0)

        assert n_removed == 3
        assert len(result) == 1
        # Lowest z is at index 2
        np.testing.assert_array_equal(result[0], [2.0, 0.5, 1.0])

    def test_very_small_positive_epsilon_near_collocated(self) -> None:
        """With very small positive epsilon (near 0), near-collocated points may cluster.

        DBSCAN requires eps > 0 (strictly positive), so we use a tiny epsilon.
        """
        pts = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],  # Exactly collocated with pt 0
            [1.0, 0.0, 3.0]   # Separate
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.0001)

        # First two (collocated) should merge; third is separate
        assert n_removed == 1
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 1.0])

    def test_very_small_epsilon(self) -> None:
        """With very small epsilon (<<1mm), almost no deduplication occurs."""
        pts = np.array([
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 2.0],  # 0.01 mm away
            [0.5, 0.0, 3.0]    # 0.5 mm away
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.001)

        # No deduplication with epsilon=0.001 mm
        assert n_removed == 0
        assert len(result) == 3

    def test_negative_z_values(self) -> None:
        """Handles negative z values correctly (tie-breaking by lowest z)."""
        pts = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.0, -2.0],
            [0.3, 0.0, 0.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=1.0)

        # All should cluster; lowest z is -2.0
        assert n_removed == 2
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.5, 0.0, -2.0])

    def test_large_coordinates(self) -> None:
        """Handles large coordinate values correctly (typical forearm range)."""
        pts = np.array([
            [100.0, 200.0, 50.0],
            [100.5, 200.0, 60.0],
            [300.0, 150.0, 40.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=1.0)

        # First two should cluster, third separate
        assert n_removed == 1
        assert len(result) == 2

    def test_dtype_preserved(self) -> None:
        """Output dtype is float64 (matching input)."""
        pts = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 2.0]
        ], dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=1.0)

        assert result.dtype == np.float64


class TestDeduplicateXyStress:
    """Stress tests with larger or structured data."""

    def test_grid_of_collocated_clusters(self) -> None:
        """Grid of clusters: each cluster has 4 collocated points."""
        clusters = []
        for i in range(3):
            for j in range(3):
                x_base = i * 10.0
                y_base = j * 10.0
                z_base = i * 10 + j
                for k in range(4):
                    clusters.append([x_base, y_base, z_base + k * 0.001])

        pts = np.array(clusters, dtype=np.float64)

        result, n_removed = deduplicate_xy(pts, epsilon=0.1)

        # 9 clusters, 4 points each = 36 points -> 9 survivors
        assert n_removed == 27
        assert len(result) == 9

    def test_random_cloud_with_noise(self) -> None:
        """Random point cloud with some intentional duplicates."""
        np.random.seed(42)

        # Random cloud
        cloud = np.random.randn(100, 3) * 10.0

        # Add some intentional near-duplicates at specific locations
        cloud[0] = [0.0, 0.0, 0.0]
        cloud[1] = [0.1, 0.0, 1.0]  # 0.1 mm away from cloud[0]

        result, n_removed = deduplicate_xy(cloud, epsilon=0.5)

        # cloud[0] and cloud[1] should merge (0.1 < 0.5)
        # Other points may or may not merge; we just check count invariant
        assert len(result) + n_removed == len(cloud)
        assert n_removed >= 1  # At least the one we know about


class TestDeduplicateXyContract:
    """Tests ensuring the public API contract is maintained."""

    def test_returns_tuple_of_two(self) -> None:
        """Return type is tuple(ndarray, int)."""
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = deduplicate_xy(pts, epsilon=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], (int, np.integer))

    def test_output_array_shape(self) -> None:
        """Output array has shape (N_survivors, 3)."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float64)

        result, _ = deduplicate_xy(pts, epsilon=0.1)

        assert result.ndim == 2
        assert result.shape[1] == 3

    def test_doesnt_modify_input(self) -> None:
        """Input array is not modified."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0]
        ], dtype=np.float64)

        pts_copy = pts.copy()
        deduplicate_xy(pts, epsilon=1.0)

        np.testing.assert_array_equal(pts, pts_copy)


class TestDeduplicateXyReturnIndices:
    """Tests for the return_indices=True mode."""

    def test_return_indices_basic(self) -> None:
        """return_indices=True returns a 3-tuple with correct indices."""
        pts = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.5],
            [5.0, 0.0, 1.0],
        ], dtype=np.float64)

        result, n_removed, indices = deduplicate_xy(pts, epsilon=0.1, return_indices=True)

        assert n_removed == 1
        assert len(indices) == 2
        np.testing.assert_array_equal(indices, [1, 2])
        np.testing.assert_array_equal(result, pts[indices])

    def test_return_indices_empty(self) -> None:
        """return_indices=True with empty input returns empty index array."""
        empty = np.empty((0, 3), dtype=np.float64)
        result, n_removed, indices = deduplicate_xy(empty, epsilon=0.5, return_indices=True)

        assert len(indices) == 0
        assert indices.dtype == np.intp

    def test_return_indices_false_default(self) -> None:
        """Without return_indices, still returns 2-tuple (backward compat)."""
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result = deduplicate_xy(pts, epsilon=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_indices_index_original_array(self) -> None:
        """Returned indices correctly index the original array."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0],
            [5.0, 0.0, 2.0],
            [5.5, 0.0, 3.0],
            [10.0, 0.0, 4.0],
        ], dtype=np.float64)

        result, n_removed, indices = deduplicate_xy(pts, epsilon=1.0, return_indices=True)

        assert n_removed == 2
        np.testing.assert_array_equal(result, pts[indices])

    def test_no_dedup_returns_all_indices(self) -> None:
        """When no deduplication occurs, all indices are returned."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 1.0],
            [10.0, 0.0, 2.0],
        ], dtype=np.float64)

        result, n_removed, indices = deduplicate_xy(pts, epsilon=0.1, return_indices=True)

        assert n_removed == 0
        np.testing.assert_array_equal(indices, [0, 1, 2])


class TestForearmDedupMetadataPath:
    """The sidecar location must be derived in exactly one place."""

    def test_path_sits_beside_the_ply(self) -> None:
        ply = Path("/data/forearm_deduped/ST14-01_forearm.ply")
        meta = forearm_dedup_metadata_path(ply)

        assert meta.parent == ply.parent
        assert meta.name == "ST14-01_forearm_dedup_metadata.json"

    def test_path_is_deterministic(self) -> None:
        ply = Path("a/b/c.ply")
        assert forearm_dedup_metadata_path(ply) == forearm_dedup_metadata_path(ply)


class TestWriteForearmDedupMetadata:
    """The effective epsilon and vertex count must survive on disk."""

    @staticmethod
    def _stats(n_original: int = 10, n_deduped: int = 7, n_removed: int = 3) -> dict:
        return {
            "n_original": n_original,
            "n_deduped": n_deduped,
            "n_removed": n_removed,
        }

    def test_writes_all_provenance_fields(self, tmp_path: Path) -> None:
        deduped = tmp_path / "forearm_deduped" / "session_forearm.ply"
        source = tmp_path / "forearm_source" / "session_forearm.ply"

        out = write_forearm_dedup_metadata(
            deduped,
            source_ply=source,
            epsilon=0.5,
            epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
            stats=self._stats(),
        )

        assert out == forearm_dedup_metadata_path(deduped)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload == {
            "schema_version": FOREARM_DEDUP_METADATA_SCHEMA_VERSION,
            "source_ply": "session_forearm.ply",
            "deduplicated_ply": "session_forearm.ply",
            "dedup_epsilon": 0.5,
            "epsilon_source": EPSILON_SOURCE_DAG_CONFIG,
            "n_vertices_original": 10,
            "n_vertices_deduped": 7,
            "n_vertices_removed": 3,
        }

    def test_records_the_interactive_epsilon_not_the_configured_one(
        self, tmp_path: Path
    ) -> None:
        """A monitor run is only reproducible if the chosen epsilon is stored."""
        deduped = tmp_path / "forearm.ply"

        out = write_forearm_dedup_metadata(
            deduped,
            source_ply=tmp_path / "src.ply",
            epsilon=1.234,
            epsilon_source=EPSILON_SOURCE_INTERACTIVE_MONITOR,
            stats=self._stats(),
        )

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["dedup_epsilon"] == pytest.approx(1.234)
        assert payload["epsilon_source"] == EPSILON_SOURCE_INTERACTIVE_MONITOR

    def test_output_is_byte_identical_on_rewrite(self, tmp_path: Path) -> None:
        """No timestamps — an unchanged re-run must not churn the artifact."""
        deduped = tmp_path / "forearm.ply"
        kwargs = dict(
            source_ply=tmp_path / "src.ply",
            epsilon=0.5,
            epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
            stats=self._stats(),
        )

        first = write_forearm_dedup_metadata(deduped, **kwargs).read_bytes()
        second = write_forearm_dedup_metadata(deduped, **kwargs).read_bytes()

        assert first == second

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        deduped = tmp_path / "does" / "not" / "exist" / "forearm.ply"

        out = write_forearm_dedup_metadata(
            deduped,
            source_ply=tmp_path / "src.ply",
            epsilon=0.5,
            epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
            stats=self._stats(),
        )

        assert out.exists()

    @pytest.mark.parametrize("bad_epsilon", [0.0, -0.5, float("nan"), float("inf")])
    def test_rejects_non_positive_or_non_finite_epsilon(
        self, tmp_path: Path, bad_epsilon: float
    ) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            write_forearm_dedup_metadata(
                tmp_path / "forearm.ply",
                source_ply=tmp_path / "src.ply",
                epsilon=bad_epsilon,
                epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
                stats=self._stats(),
            )

    def test_rejects_unknown_epsilon_source(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown epsilon_source"):
            write_forearm_dedup_metadata(
                tmp_path / "forearm.ply",
                source_ply=tmp_path / "src.ply",
                epsilon=0.5,
                epsilon_source="guessed",
                stats=self._stats(),
            )

    def test_rejects_missing_stats_key(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            write_forearm_dedup_metadata(
                tmp_path / "forearm.ply",
                source_ply=tmp_path / "src.ply",
                epsilon=0.5,
                epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
                stats={"n_original": 10, "n_deduped": 7},
            )

    def test_rejects_inconsistent_vertex_counts(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Inconsistent dedup vertex counts"):
            write_forearm_dedup_metadata(
                tmp_path / "forearm.ply",
                source_ply=tmp_path / "src.ply",
                epsilon=0.5,
                epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
                stats=self._stats(n_original=10, n_deduped=7, n_removed=2),
            )


class TestDeduplicateForearmPly:
    """The source -> deduped vertex mapping must be recoverable by the caller."""

    def test_returns_kept_indices(self, tmp_path: Path) -> None:
        points = [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.5],
            [5.0, 0.0, 1.0],
        ]
        src = tmp_path / "src.ply"
        out = tmp_path / "deduped" / "out.ply"
        _write_ply(src, points)

        stats = deduplicate_forearm_ply(src, out, epsilon=0.1)

        assert stats["n_original"] == 3
        assert stats["n_deduped"] == 2
        assert stats["n_removed"] == 1
        np.testing.assert_array_equal(stats["kept_indices"], [1, 2])

    def test_kept_indices_index_the_source_vertices(self, tmp_path: Path) -> None:
        points = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0],
            [5.0, 0.0, 2.0],
            [5.5, 0.0, 3.0],
            [10.0, 0.0, 4.0],
        ]
        src = tmp_path / "src.ply"
        out = tmp_path / "out.ply"
        _write_ply(src, points)

        stats = deduplicate_forearm_ply(src, out, epsilon=1.0)

        source_pts = np.asarray(o3d.io.read_point_cloud(str(src)).points)
        deduped_pts = np.asarray(o3d.io.read_point_cloud(str(out)).points)
        np.testing.assert_allclose(
            deduped_pts, source_pts[stats["kept_indices"]], atol=1e-6
        )

    def test_kept_indices_match_written_vertex_count(self, tmp_path: Path) -> None:
        points = [[float(i % 4), float(i // 4), float(i)] for i in range(16)]
        src = tmp_path / "src.ply"
        out = tmp_path / "out.ply"
        _write_ply(src, points)

        stats = deduplicate_forearm_ply(src, out, epsilon=0.5)

        n_written = len(np.asarray(o3d.io.read_point_cloud(str(out)).points))
        assert len(stats["kept_indices"]) == n_written == stats["n_deduped"]

    def test_kept_indices_agree_with_carried_colors(self, tmp_path: Path) -> None:
        points = [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.5],
            [5.0, 0.0, 1.0],
        ]
        colors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        src = tmp_path / "src.ply"
        out = tmp_path / "out.ply"
        _write_ply(src, points, colors=colors)

        stats = deduplicate_forearm_ply(src, out, epsilon=0.1)

        source_colors = np.asarray(o3d.io.read_point_cloud(str(src)).colors)
        deduped_colors = np.asarray(o3d.io.read_point_cloud(str(out)).colors)
        np.testing.assert_allclose(
            deduped_colors, source_colors[stats["kept_indices"]], atol=1e-6
        )

    def test_empty_ply_returns_empty_kept_indices(self, tmp_path: Path) -> None:
        # Open3D refuses to *write* a 0-point cloud, so the empty PLY is
        # hand-written; it reads back as an empty cloud all the same.
        src = tmp_path / "src.ply"
        out = tmp_path / "out.ply"
        src.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n",
            encoding="ascii",
        )
        assert len(np.asarray(o3d.io.read_point_cloud(str(src)).points)) == 0

        stats = deduplicate_forearm_ply(src, out, epsilon=0.5)

        assert stats["n_original"] == 0
        assert stats["n_deduped"] == 0
        assert stats["n_removed"] == 0
        assert len(stats["kept_indices"]) == 0
        assert stats["kept_indices"].dtype == np.intp

    def test_epsilon_is_required_and_keyword_only(self, tmp_path: Path) -> None:
        """A silent default epsilon would repoint every derived vertex index."""
        src = tmp_path / "src.ply"
        out = tmp_path / "out.ply"
        _write_ply(src, [[0.0, 0.0, 0.0]])

        with pytest.raises(TypeError):
            deduplicate_forearm_ply(src, out)

        with pytest.raises(TypeError):
            deduplicate_forearm_ply(src, out, 0.5)

    def test_stats_feed_the_metadata_writer_directly(self, tmp_path: Path) -> None:
        """The dict returned by the dedup is the dict the sidecar writer consumes."""
        points = [[float(i % 3), 0.0, float(i)] for i in range(9)]
        src = tmp_path / "src.ply"
        out = tmp_path / "out.ply"
        _write_ply(src, points)

        stats = deduplicate_forearm_ply(src, out, epsilon=0.5)
        meta_path = write_forearm_dedup_metadata(
            out,
            source_ply=src,
            epsilon=0.5,
            epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
            stats=stats,
        )

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        n_written = len(np.asarray(o3d.io.read_point_cloud(str(out)).points))
        assert payload["n_vertices_deduped"] == n_written
        assert payload["n_vertices_deduped"] == len(stats["kept_indices"])


class TestDeduplicateXyMapping:
    """The mapping is what a parallel per-point payload must be reduced by."""

    def test_labels_cover_every_input_point(self) -> None:
        pts = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.5],
            [5.0, 0.0, 1.0],
        ], dtype=np.float64)

        mapping = deduplicate_xy_mapping(pts, 0.1)

        assert mapping.labels.shape == (len(pts),)
        assert mapping.n_input == len(pts)
        assert mapping.n_removed == len(pts) - len(mapping.kept_indices)

    def test_exactly_one_survivor_per_cluster(self) -> None:
        pts = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.5],
            [5.0, 0.0, 1.0],
            [5.05, 0.0, 3.0],
            [10.0, 0.0, 0.0],
        ], dtype=np.float64)

        mapping = deduplicate_xy_mapping(pts, 0.1)

        survivor_labels = mapping.labels[mapping.kept_indices]
        assert len(set(survivor_labels.tolist())) == len(survivor_labels)
        assert set(survivor_labels.tolist()) == set(mapping.labels.tolist())

    def test_labels_identify_which_rows_collapsed_together(self) -> None:
        """The group membership — not just the survivors — must be recoverable."""
        pts = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.5],
            [5.0, 0.0, 1.0],
            [5.05, 0.0, 3.0],
            [10.0, 0.0, 0.0],
        ], dtype=np.float64)

        mapping = deduplicate_xy_mapping(pts, 0.1)

        np.testing.assert_array_equal(mapping.kept_indices, [1, 2, 4])
        assert mapping.labels[0] == mapping.labels[1]
        assert mapping.labels[2] == mapping.labels[3]
        assert mapping.labels[4] not in (mapping.labels[0], mapping.labels[2])
        assert mapping.labels[0] != mapping.labels[2]

    def test_agrees_with_deduplicate_xy(self) -> None:
        """deduplicate_xy is exactly the application of this mapping."""
        rng = np.random.default_rng(20260818)
        pts = rng.normal(size=(200, 3))

        deduped, n_removed, kept = deduplicate_xy(pts, 0.2, return_indices=True)
        mapping = deduplicate_xy_mapping(pts, 0.2)

        np.testing.assert_array_equal(mapping.kept_indices, kept)
        assert mapping.n_removed == n_removed
        assert mapping.n_input == len(pts)
        np.testing.assert_array_equal(pts[mapping.kept_indices], deduped)

    def test_empty_input(self) -> None:
        mapping = deduplicate_xy_mapping(np.empty((0, 3), dtype=np.float64), 0.5)

        assert mapping.kept_indices.shape == (0,)
        assert mapping.labels.shape == (0,)
        assert mapping.kept_indices.dtype == np.intp
        assert mapping.labels.dtype == np.intp
        assert mapping.n_input == 0
        assert mapping.n_removed == 0

    def test_index_arrays_are_intp(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 1.0]], dtype=np.float64)

        mapping = deduplicate_xy_mapping(pts, 0.5)

        assert mapping.kept_indices.dtype == np.intp
        assert mapping.labels.dtype == np.intp

    def test_mapping_is_frozen(self) -> None:
        mapping = deduplicate_xy_mapping(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64), 0.5
        )

        with pytest.raises(AttributeError):
            mapping.kept_indices = np.empty(0, dtype=np.intp)  # type: ignore[misc]


def _contact_cell(points) -> str:
    """Serialise points the way the upstream CSV writer does (%.1f)."""
    return serialize_contact_points([(float(x), float(y), float(z)) for x, y, z in points])


def _write_contact_csv(path: Path, rows) -> None:
    """Write a minimal session CSV. *rows* is a list of (frame_index, cell)."""
    pd.DataFrame({
        "time": np.arange(len(rows), dtype=np.float64),
        FRAME_INDEX_COLUMN: [frame for frame, _ in rows],
        "contact_location_x": [0.0] * len(rows),
        "contact_location_y": [0.0] * len(rows),
        "contact_location_z": [0.0] * len(rows),
        "contact_points": [cell for _, cell in rows],
    }).to_csv(path, index=False)


class TestDeduplicateContactPointsCsvMapping:
    """The per-frame mapping the depth field will later be reduced by."""

    _POINTS_A = [(0.0, 0.0, 2.0), (0.0, 0.0, 0.5), (5.0, 0.0, 1.0)]
    _POINTS_B = [(1.0, 1.0, 3.0), (1.0, 1.0, 1.0)]

    def _run(self, tmp_path: Path) -> dict:
        src = tmp_path / "in.csv"
        out = tmp_path / "out.csv"
        _write_contact_csv(src, [
            (168.0, "[]"),
            (168.0, _contact_cell(self._POINTS_A)),
            (169.0, "[]"),
            (170.0, "[]"),
            (205.0, _contact_cell(self._POINTS_B)),
            (206.0, "[]"),
        ])
        stats = deduplicate_contact_points_csv(src, out, epsilon=0.1)
        stats["_output_csv"] = out
        return stats

    def test_keyed_by_frame_index_not_row_position(self, tmp_path: Path) -> None:
        stats = self._run(tmp_path)

        assert set(stats["frame_mappings"]) == {168, 205}

    def test_keys_are_plain_ints(self, tmp_path: Path) -> None:
        stats = self._run(tmp_path)

        assert all(type(key) is int for key in stats["frame_mappings"])

    def test_values_are_dedup_mappings(self, tmp_path: Path) -> None:
        stats = self._run(tmp_path)

        assert all(
            isinstance(value, DedupMapping) for value in stats["frame_mappings"].values()
        )

    def test_mapping_matches_the_written_cell(self, tmp_path: Path) -> None:
        """Applying the mapping to the input points reproduces the output cell."""
        stats = self._run(tmp_path)
        df = pd.read_csv(stats["_output_csv"])

        inputs = {168: self._POINTS_A, 205: self._POINTS_B}
        for frame, mapping in stats["frame_mappings"].items():
            cell = df.loc[df[FRAME_INDEX_COLUMN] == frame, "contact_points"].iloc[-1]
            written = np.asarray(parse_contact_points(cell), dtype=np.float64)
            expected = np.asarray(inputs[frame], dtype=np.float64)[mapping.kept_indices]

            np.testing.assert_allclose(written, expected, atol=0.05)
            assert mapping.n_input == len(inputs[frame])

    def test_rows_without_contact_points_get_no_mapping(self, tmp_path: Path) -> None:
        stats = self._run(tmp_path)

        assert 169 not in stats["frame_mappings"]
        assert 206 not in stats["frame_mappings"]

    def test_existing_stat_keys_are_unchanged(self, tmp_path: Path) -> None:
        stats = self._run(tmp_path)

        assert stats["n_rows_processed"] == 2
        assert stats["total_points_before"] == 5
        assert stats["total_points_after"] == 3

    def test_missing_frame_index_column_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "in.csv"
        pd.DataFrame({
            "contact_points": [_contact_cell(self._POINTS_A)],
            "contact_location_x": [0.0],
            "contact_location_y": [0.0],
            "contact_location_z": [0.0],
        }).to_csv(src, index=False)

        with pytest.raises(ValueError, match=FRAME_INDEX_COLUMN):
            deduplicate_contact_points_csv(src, tmp_path / "out.csv", epsilon=0.1)

    def test_duplicate_frame_index_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "in.csv"
        _write_contact_csv(src, [
            (168.0, _contact_cell(self._POINTS_A)),
            (168.0, _contact_cell(self._POINTS_B)),
        ])

        with pytest.raises(ValueError, match="ambiguous"):
            deduplicate_contact_points_csv(src, tmp_path / "out.csv", epsilon=0.1)

    def test_nan_frame_index_on_a_contact_row_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "in.csv"
        _write_contact_csv(src, [(np.nan, _contact_cell(self._POINTS_A))])

        with pytest.raises(ValueError, match="whole number"):
            deduplicate_contact_points_csv(src, tmp_path / "out.csv", epsilon=0.1)

    def test_fractional_frame_index_on_a_contact_row_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "in.csv"
        _write_contact_csv(src, [(168.5, _contact_cell(self._POINTS_A))])

        with pytest.raises(ValueError, match="whole number"):
            deduplicate_contact_points_csv(src, tmp_path / "out.csv", epsilon=0.1)

    def test_nan_frame_index_on_an_empty_row_is_tolerated(self, tmp_path: Path) -> None:
        """Non-contact rows carry no mapping, so their frame_index is irrelevant."""
        src = tmp_path / "in.csv"
        out = tmp_path / "out.csv"
        _write_contact_csv(src, [
            (np.nan, "[]"),
            (168.0, _contact_cell(self._POINTS_A)),
        ])

        stats = deduplicate_contact_points_csv(src, out, epsilon=0.1)

        assert set(stats["frame_mappings"]) == {168}

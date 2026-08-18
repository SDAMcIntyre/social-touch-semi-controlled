import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN

from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
)

# --- Deduplicated-forearm provenance sidecar -------------------------------
#
# A vertex index into the deduplicated forearm PLY is only meaningful relative
# to the epsilon that produced that PLY: change epsilon and every vertex is
# renumbered. The DAG config is not part of the mtime-based staleness check
# (utils/should_process_task.py), and under `monitor: true` the epsilon is
# chosen interactively and never reaches the config at all. The sidecar is the
# only place the *effective* epsilon and the resulting vertex count survive, so
# a downstream consumer can prove which PLY a vertex index belongs to.

FOREARM_DEDUP_METADATA_SUFFIX = "_dedup_metadata.json"
FOREARM_DEDUP_METADATA_SCHEMA_VERSION = "1"

#: Epsilon came from the DAG config's ``deduplicate_xy.epsilon`` option.
EPSILON_SOURCE_DAG_CONFIG = "dag_config"
#: Epsilon was chosen by the operator in the interactive monitor viewer.
EPSILON_SOURCE_INTERACTIVE_MONITOR = "interactive_monitor"

EPSILON_SOURCES = frozenset({
    EPSILON_SOURCE_DAG_CONFIG,
    EPSILON_SOURCE_INTERACTIVE_MONITOR,
})

#: CSV column that identifies the Kinect frame a row belongs to. Contact data is
#: keyed by this value, never by row position: the merged CSV is upsampled to the
#: nerve rate, so a row index means nothing outside one particular file, while
#: ``frame_index`` is what every other stage of this pipeline aligns on.
FRAME_INDEX_COLUMN = "frame_index"


@dataclass(frozen=True)
class DedupMapping:
    """How one set of points collapsed under (x, y) deduplication.

    Both arrays describe the *input* points, so a parallel per-point payload
    (e.g. a per-vertex depth field) can be reduced by the very same mapping the
    CSV was reduced by, instead of re-running the clustering against it — which
    would diverge silently wherever two candidates are near-equidistant.

    Attributes:
        kept_indices: 1-D ``intp`` array, ascending, of the input rows that
            survived. ``deduped == points[kept_indices]``.
        labels: 1-D ``intp`` array with one entry per *input* row, giving the
            cluster that row was assigned to. Rows sharing a label collapsed
            into a single survivor — the one whose index is in *kept_indices*.
            Labels are arbitrary ints, dense over ``range(n_clusters)``; there
            is no noise label, because the clustering runs with
            ``min_samples=1``.
    """

    kept_indices: np.ndarray
    labels: np.ndarray

    @property
    def n_input(self) -> int:
        """Number of points the mapping was computed from."""
        return len(self.labels)

    @property
    def n_removed(self) -> int:
        """Number of points that were collapsed into a survivor."""
        return len(self.labels) - len(self.kept_indices)


def deduplicate_xy_mapping(points: np.ndarray, epsilon: float) -> DedupMapping:
    """Compute the (x, y) deduplication mapping for *points*, applying nothing.

    Split out of :func:`deduplicate_xy` so the mapping can be reused by a
    consumer that must reduce a second array the same way. There is exactly one
    implementation of the clustering; :func:`deduplicate_xy` is a thin
    application of what this returns.

    Args:
        points: Array of shape (N, 3) in mm, float64.
        epsilon: Radius in mm. Two points within this distance in (x, y) are
            considered in the same cluster.

    Returns:
        The :class:`DedupMapping` describing which rows survive and which rows
        collapsed together.
    """
    if len(points) == 0:
        return DedupMapping(
            kept_indices=np.empty(0, dtype=np.intp),
            labels=np.empty(0, dtype=np.intp),
        )

    labels = DBSCAN(eps=epsilon, min_samples=1).fit_predict(points[:, :2])

    # Stable sort: primary key = z ascending; secondary key = original index ascending.
    order = np.lexsort((np.arange(len(points)), points[:, 2]))

    seen: set[int] = set()
    kept: list[int] = []
    for i in order:
        lbl = int(labels[i])
        if lbl in seen:
            continue
        seen.add(lbl)
        kept.append(int(i))
    kept.sort()  # restore input order

    return DedupMapping(
        kept_indices=np.array(kept, dtype=np.intp),
        labels=np.asarray(labels, dtype=np.intp),
    )


def deduplicate_xy(
    points: np.ndarray, epsilon: float = 0.35, *, return_indices: bool = False
) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, np.ndarray]:
    """Deduplicate points by (x, y) position using single-linkage clustering.

    Any two points within epsilon (Euclidean distance on x, y only) are merged
    into a single cluster. Within each cluster, the survivor is the point with
    the lowest z value (ties broken by lowest original input index for determinism).

    Args:
        points: Array of shape (N, 3) in mm, float64.
        epsilon: Radius in mm. Two points within this distance in (x, y) are
            considered in the same cluster.
        return_indices: If True, also return the indices of kept points.

    Returns:
        If return_indices is False (default):
            Tuple of (deduped_points, n_removed).
        If return_indices is True:
            Tuple of (deduped_points, n_removed, kept_indices) where
            kept_indices is a 1-D int array such that
            deduped_points == points[kept_indices].

    See :func:`deduplicate_xy_mapping` for the cluster membership itself, which
    this function discards.
    """
    mapping = deduplicate_xy_mapping(points, epsilon)

    deduped = points[mapping.kept_indices]
    n_removed = len(points) - len(deduped)
    assert len(deduped) + n_removed == len(points), (
        f"Deduplication invariant violated: {len(deduped)} + {n_removed} != {len(points)}"
    )
    if return_indices:
        return deduped, n_removed, mapping.kept_indices
    return deduped, n_removed


def _monitor_viewer_process(points: np.ndarray, initial_epsilon: float,
                            slider_range: tuple[float, float],
                            result_queue) -> None:
    """Target for the daemon subprocess that runs the PyVista viewer."""
    import pyvista as pv
    import vtk
    from scipy.spatial import KDTree

    state = {"epsilon": initial_epsilon}
    sel: dict = {"A": None, "B": None}

    deduped, n_removed = deduplicate_xy(points, initial_epsilon)
    tree = KDTree(points)

    pl = pv.Plotter(title="Deduplicate XY — Adjust Epsilon")

    original_cloud = pv.PolyData(points)
    pl.add_mesh(
        original_cloud, color="steelblue", point_size=3,
        render_points_as_spheres=True, opacity=0.3, name="original",
    )

    deduped_cloud = pv.PolyData(deduped)
    pl.add_mesh(
        deduped_cloud, color="tomato", point_size=5,
        render_points_as_spheres=True, name="deduped",
    )

    stats_text = (
        f"Original: {len(points)} | Deduped: {len(deduped)} | "
        f"Removed: {n_removed} ({100 * n_removed / max(len(points), 1):.1f}%) | "
        f"Epsilon: {initial_epsilon:.4f}"
    )
    pl.add_text(stats_text, position="upper_left", font_size=10, name="stats")

    def _on_epsilon_change(value: float) -> None:
        state["epsilon"] = value
        new_deduped, new_removed = deduplicate_xy(points, value)

        new_cloud = pv.PolyData(new_deduped)
        pl.add_mesh(
            new_cloud, color="tomato", point_size=5,
            render_points_as_spheres=True, name="deduped",
        )

        new_stats = (
            f"Original: {len(points)} | Deduped: {len(new_deduped)} | "
            f"Removed: {new_removed} ({100 * new_removed / max(len(points), 1):.1f}%) | "
            f"Epsilon: {value:.4f}"
        )
        pl.add_text(new_stats, position="upper_left", font_size=10, name="stats")

    def _update_pick_text() -> None:
        lines = []
        if sel["A"] is not None:
            a = sel["A"]
            lines.append(f"A: x={a[0]:.3f}  y={a[1]:.3f}  z={a[2]:.3f}")
        if sel["B"] is not None:
            b = sel["B"]
            lines.append(f"B: x={b[0]:.3f}  y={b[1]:.3f}  z={b[2]:.3f}")
        if sel["A"] is not None and sel["B"] is not None:
            d = float(np.linalg.norm(sel["A"] - sel["B"]))
            lines.append(f"dist={d:.3f}")
        pl.add_text("\n".join(lines), position="upper_right", font_size=10, name="pick_info")

    _vtk_picker = vtk.vtkPointPicker()
    _vtk_picker.SetTolerance(0.005)
    _pick_mode = {"target": None}

    def _pick_point() -> None:
        x, y = pl.iren.interactor.GetEventPosition()
        _vtk_picker.Pick(x, y, 0, pl.renderer)
        if _vtk_picker.GetPointId() < 0:
            return
        _, idx = tree.query(np.array(_vtk_picker.GetPickPosition()))
        pt = points[idx]
        label = _pick_mode["target"]
        if label == "A":
            sel["A"] = pt
            pl.add_mesh(pv.Sphere(radius=0.5, center=pt.tolist()),
                        color="red", opacity=0.5, name="sel_A")
        elif label == "B":
            sel["B"] = pt
            pl.add_mesh(pv.Sphere(radius=0.5, center=pt.tolist()),
                        color="limegreen", opacity=0.5, name="sel_B")
        _update_pick_text()
        _pick_mode["target"] = None
        pl.render()

    def _on_click_after_mode(interactor, _event) -> None:
        if _pick_mode["target"] is None:
            return
        _pick_point()

    pl.iren.interactor.AddObserver("LeftButtonPressEvent", _on_click_after_mode, 10.0)

    def _activate_pick_a() -> None:
        _pick_mode["target"] = "A"

    def _activate_pick_b() -> None:
        _pick_mode["target"] = "B"

    pl.add_key_event("a", _activate_pick_a)
    pl.add_key_event("b", _activate_pick_b)

    pl.add_slider_widget(
        callback=_on_epsilon_change,
        rng=slider_range,
        value=initial_epsilon,
        title="Epsilon (bin size mm)",
        style="modern",
        pointa=(0.25, 0.05),
        pointb=(0.25, 0.05),
    )

    pl.add_text(
        "Press A/B then click to pick | Adjust epsilon then close to proceed",
        position="lower_left", font_size=9, color="grey",
    )

    def _view_z() -> None:
        pl.view_xy(negative=True)

    pl.add_key_event("z", _view_z)
    pl.add_checkbox_button_widget(
        callback=lambda state: _view_z(),
        value=False,
        position=(10, 10),
        size=30,
        border_size=2,
        color_on="dodgerblue",
        color_off="grey",
    )
    pl.add_text("View Z", position=(45, 12), font_size=9, name="btn_label_z")

    try:
        pl.show()
    finally:
        pl.close()

    result_queue.put(state["epsilon"])


def monitor_deduplicate_xy_interactive(
    points: np.ndarray,
    initial_epsilon: float = 0.1,
    slider_range: tuple[float, float] = (0.001, 5.0),
) -> float:
    """Open an interactive PyVista viewer to tune epsilon for XY deduplication.

    Shows the forearm pointcloud with original (blue) and deduplicated (red) points.
    The user adjusts epsilon via a slider and sees the effect in real-time.
    Returns the final epsilon value when the user closes the window.

    Runs in a daemon subprocess so aborting the parent pipeline does not freeze.
    """
    import signal
    from multiprocessing import Process, Queue

    result_queue = Queue()
    proc = Process(
        target=_monitor_viewer_process,
        args=(points, initial_epsilon, slider_range, result_queue),
        daemon=True,
    )
    proc.start()

    _aborted = False

    def _on_terminate(signum, frame):
        nonlocal _aborted
        _aborted = True
        if proc.is_alive():
            proc.terminate()

    prev_handler = signal.signal(signal.SIGTERM, _on_terminate)
    try:
        while proc.is_alive():
            proc.join(timeout=0.2)
    finally:
        signal.signal(signal.SIGTERM, prev_handler)

    if _aborted:
        raise SystemExit(1)

    if not result_queue.empty():
        return result_queue.get()
    return initial_epsilon


def forearm_dedup_metadata_path(deduped_ply: Path) -> Path:
    """Return the path of the provenance sidecar for a deduplicated forearm PLY.

    Producers and consumers must both go through this helper so the sidecar is
    always found next to the PLY it describes.

    Args:
        deduped_ply: Path of the deduplicated forearm PLY.

    Returns:
        Path of the JSON sidecar (same directory, same stem).
    """
    return deduped_ply.with_name(deduped_ply.stem + FOREARM_DEDUP_METADATA_SUFFIX)


def write_forearm_dedup_metadata(
    deduped_ply: Path,
    *,
    source_ply: Path,
    epsilon: float,
    epsilon_source: str,
    stats: dict,
) -> Path:
    """Record the effective dedup epsilon and vertex counts beside the deduped PLY.

    Written as a deterministic JSON sidecar (sorted keys, no timestamps) so an
    unchanged re-run produces a byte-identical file.

    Args:
        deduped_ply: Path of the deduplicated forearm PLY this describes.
        source_ply: Path of the PLY that was deduplicated.
        epsilon: The epsilon actually applied — not the configured default.
        epsilon_source: One of EPSILON_SOURCES, saying where that value came from.
        stats: The dict returned by deduplicate_forearm_ply.

    Returns:
        Path of the written sidecar.

    Raises:
        ValueError: If epsilon is not a positive finite value, epsilon_source is
            unknown, stats is missing a required key, or the vertex counts do
            not satisfy n_deduped + n_removed == n_original.
    """
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            f"dedup epsilon must be a positive finite value, got {epsilon!r} "
            f"(writing metadata for {deduped_ply})"
        )

    if epsilon_source not in EPSILON_SOURCES:
        raise ValueError(
            f"Unknown epsilon_source {epsilon_source!r}; expected one of "
            f"{sorted(EPSILON_SOURCES)}"
        )

    required = ("n_original", "n_deduped", "n_removed")
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(
            f"dedup stats is missing required key(s) {missing} "
            f"(writing metadata for {deduped_ply})"
        )

    n_original = int(stats["n_original"])
    n_deduped = int(stats["n_deduped"])
    n_removed = int(stats["n_removed"])
    if n_deduped + n_removed != n_original:
        raise ValueError(
            f"Inconsistent dedup vertex counts: {n_deduped} + {n_removed} "
            f"!= {n_original} (writing metadata for {deduped_ply})"
        )

    metadata = {
        "schema_version": FOREARM_DEDUP_METADATA_SCHEMA_VERSION,
        "source_ply": source_ply.name,
        "deduplicated_ply": deduped_ply.name,
        "dedup_epsilon": epsilon,
        "epsilon_source": epsilon_source,
        "n_vertices_original": n_original,
        "n_vertices_deduped": n_deduped,
        "n_vertices_removed": n_removed,
    }

    metadata_path = forearm_dedup_metadata_path(deduped_ply)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def deduplicate_forearm_ply(input_ply: Path, output_ply: Path, *, epsilon: float) -> dict:
    """Deduplicate a forearm PLY point cloud by (x, y) position.

    Loads the PLY, removes (x, y) duplicates keeping the lowest z (outermost
    surface), and writes the result to output_ply.

    Args:
        input_ply: Path to the input PLY file.
        output_ply: Path to write the deduplicated PLY.
        epsilon: Bin size in mm for (x, y) deduplication. Required — a default
            here would silently repoint every vertex index derived from the
            output PLY (see the sidecar note at the top of this module).

    Returns:
        Dict with keys n_original, n_deduped, n_removed and kept_indices, where
        kept_indices is a 1-D intp array of the surviving source-PLY vertex
        indices in output order: output vertex i is source vertex
        kept_indices[i].
    """
    pcd = o3d.io.read_point_cloud(str(input_ply))
    vertices = np.asarray(pcd.points, dtype=np.float64)

    if len(vertices) == 0:
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_ply), pcd)
        return {
            "n_original": 0,
            "n_deduped": 0,
            "n_removed": 0,
            "kept_indices": np.empty(0, dtype=np.intp),
        }

    deduped, n_removed, kept_indices = deduplicate_xy(vertices, epsilon, return_indices=True)

    deduped_pcd = o3d.geometry.PointCloud()
    deduped_pcd.points = o3d.utility.Vector3dVector(deduped)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        deduped_pcd.colors = o3d.utility.Vector3dVector(colors[kept_indices])

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        deduped_pcd.normals = o3d.utility.Vector3dVector(normals[kept_indices])

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply), deduped_pcd)

    return {
        "n_original": len(vertices),
        "n_deduped": len(deduped),
        "n_removed": n_removed,
        "kept_indices": kept_indices,
    }


def _frame_index_key(value, *, csv_path: Path, row_position: int) -> int:
    """Coerce a ``frame_index`` cell to the integer key the mapping is stored under.

    The column arrives as float64 because rows without a Kinect frame hold NaN,
    so the value must be checked, not merely cast: a NaN or a fractional index
    would otherwise become a silently wrong dict key.
    """
    as_float = float(value)
    if not np.isfinite(as_float) or as_float != int(as_float):
        raise ValueError(
            f"Row {row_position} of {csv_path} carries contact points but its "
            f"{FRAME_INDEX_COLUMN} is {value!r}, which is not a whole number. "
            f"Contact rows must be anchored to a Kinect frame."
        )
    return int(as_float)


def deduplicate_contact_points_csv(
    input_csv: Path, output_csv: Path, epsilon: float = 0.35
) -> dict:
    """Deduplicate contact_points in each row of a session CSV by (x, y) position.

    For each row with non-empty contact_points, removes (x, y) duplicates keeping
    the lowest z per bin, then recomputes contact_location_x/y/z as the mean of
    the remaining points. Rows with empty or null contact_points pass through
    unchanged.

    Args:
        input_csv: Path to the input session CSV.
        output_csv: Path to write the deduplicated CSV.
        epsilon: Bin size in mm for (x, y) deduplication.

    Returns:
        Dict with keys n_rows_processed, total_points_before, total_points_after
        and frame_mappings. ``frame_mappings`` maps ``frame_index`` to the
        :class:`DedupMapping` that was applied to that frame's contact points,
        for the frames that had any — it is what lets a per-point sidecar be
        reduced by this CSV's own clustering rather than by a re-run of it.

    Raises:
        ValueError: If the ``frame_index`` column is absent, if a contact-bearing
            row has a non-integral ``frame_index``, or if two contact-bearing
            rows share one ``frame_index`` (which would make the mapping
            ambiguous).
    """
    df = pd.read_csv(input_csv)

    if FRAME_INDEX_COLUMN not in df.columns:
        raise ValueError(
            f"{input_csv} has no {FRAME_INDEX_COLUMN!r} column; the per-frame "
            f"deduplication mapping cannot be keyed."
        )

    n_rows_processed = 0
    total_points_before = 0
    total_points_after = 0
    frame_mappings: dict[int, DedupMapping] = {}

    new_contact_points = []
    new_location_x = []
    new_location_y = []
    new_location_z = []

    for row_position, (_, row) in enumerate(df.iterrows()):
        raw = row.get("contact_points", None)
        points = parse_contact_points(raw) if raw is not None and str(raw).strip() else []

        if not points:
            new_contact_points.append(raw)
            new_location_x.append(row.get("contact_location_x"))
            new_location_y.append(row.get("contact_location_y"))
            new_location_z.append(row.get("contact_location_z"))
            continue

        pts_array = np.array(points, dtype=np.float64)
        total_points_before += len(pts_array)

        mapping = deduplicate_xy_mapping(pts_array, epsilon)
        deduped_array = pts_array[mapping.kept_indices]

        frame_index = _frame_index_key(
            row[FRAME_INDEX_COLUMN], csv_path=input_csv, row_position=row_position
        )
        if frame_index in frame_mappings:
            raise ValueError(
                f"{input_csv} has two contact-bearing rows with "
                f"{FRAME_INDEX_COLUMN}={frame_index} (second at row {row_position}); "
                f"the per-frame deduplication mapping would be ambiguous."
            )
        frame_mappings[frame_index] = mapping

        total_points_after += len(deduped_array)
        n_rows_processed += 1

        deduped_tuples = [
            (float(p[0]), float(p[1]), float(p[2])) for p in deduped_array
        ]
        new_contact_points.append(serialize_contact_points(deduped_tuples))

        mean_pt = deduped_array.mean(axis=0)
        new_location_x.append(mean_pt[0])
        new_location_y.append(mean_pt[1])
        new_location_z.append(mean_pt[2])

    df["contact_points"] = new_contact_points
    df["contact_location_x"] = new_location_x
    df["contact_location_y"] = new_location_y
    df["contact_location_z"] = new_location_z

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return {
        "n_rows_processed": n_rows_processed,
        "total_points_before": total_points_before,
        "total_points_after": total_points_after,
        "frame_mappings": frame_mappings,
    }

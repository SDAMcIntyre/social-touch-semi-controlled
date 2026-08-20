"""Postprocessing step 2: deduplicate (x, y) duplicates in the forearm PLY and
in every registered block's contact points.

The per-vertex contact depth field
----------------------------------
Deduplication is not a coordinate transform — it *removes rows*.  The sidecar
must therefore lose exactly the rows the CSV lost, and it must lose them by the
CSV's **own** clustering: re-running DBSCAN against the sidecar's float32
coordinates would collapse a different set of points wherever two candidates sit
either side of the epsilon boundary, and would do so silently.
:func:`deduplicate_contact_points_csv` surfaces that clustering as
``frame_mappings``; :func:`deduplicate_contact_depth_field` is the only
consumer, and it must be handed the mapping from the *same* call, so the epsilon
is the one actually applied even when the operator overrode it interactively.

A survivor inherits the deepest penetration of the group that collapsed into it.
That is what keeps the pipeline's strongest cross-artifact invariant true: per
frame, ``max(|signed_depth_mm|)`` still equals the CSV's ``contact_depth``, a
column computed before deduplication and never recomputed after it.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN

from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    read_contact_depth_field,
    write_contact_depth_field_table,
)
from postprocessing.depth_field_stage_io import (
    DEPTH_COLUMN,
    PIPELINE_STAGE_POSTPROCESSING,
    DedupMappingLike,
    apply_dedup_mapping_to_field,
    assert_max_depth_agrees_with_csv,
    assert_row_counts_agree_with_csv,
)

# --- Deduplicated-forearm provenance sidecar -------------------------------
#
# The format lives in ``postprocessing.forearm_dedup_metadata`` because two
# stage scripts need it and stage scripts must not import one another: this one
# writes it, and ``project_contacts_onto_forearm`` reads it to stamp the
# reference-PLY provenance onto the depth field it assigns ``vertex_id`` to.
# Re-exported here so existing importers keep working.
from postprocessing.forearm_dedup_metadata import (  # noqa: E402
    EPSILON_SOURCE_DAG_CONFIG,
    EPSILON_SOURCE_INTERACTIVE_MONITOR,
    EPSILON_SOURCES,
    FOREARM_DEDUP_METADATA_SCHEMA_VERSION,
    FOREARM_DEDUP_METADATA_SUFFIX,
    ForearmDedupMetadata,
    forearm_dedup_metadata_path,
    read_forearm_dedup_metadata,
    write_forearm_dedup_metadata,
)

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


def _assert_depth_only_reduced(
    original: pd.DataFrame, deduped: pd.DataFrame, *, parquet_name: str
) -> None:
    """Raise unless every surviving depth is a value that was already in the frame.

    Deduplication is the one stage allowed to change ``signed_depth_mm``, and it
    is allowed to change it in exactly one way: a survivor may inherit the
    deepest penetration of the group that collapsed into it.  Nothing is
    averaged, interpolated or re-measured, so every value written out must be a
    value that was read in — asserted per frame, because a value borrowed from
    another frame would be just as wrong as an invented one.

    Args:
        original: The field as read, before deduplication.
        deduped: The field after :func:`apply_dedup_mapping_to_field`.
        parquet_name: Name used in the error message.

    Raises:
        ValueError: If the dtype changed, or if any surviving depth is not one
            of the depths its own frame held before deduplication.
    """
    before_dtype = original[DEPTH_COLUMN].to_numpy().dtype
    after_dtype = deduped[DEPTH_COLUMN].to_numpy().dtype
    if before_dtype != after_dtype:
        raise ValueError(
            f"{parquet_name}: deduplication changed the {DEPTH_COLUMN!r} dtype "
            f"({before_dtype} -> {after_dtype}). A depth is a measurement that "
            "is re-attributed, never recomputed."
        )

    for frame, group in deduped.groupby(FRAME_INDEX_COLUMN, sort=False):
        source = original.loc[
            original[FRAME_INDEX_COLUMN] == frame, DEPTH_COLUMN
        ].to_numpy()
        survivors = group[DEPTH_COLUMN].to_numpy()
        if not np.isin(survivors, source).all():
            raise ValueError(
                f"{parquet_name}: frame {int(frame)} carries a "
                f"{DEPTH_COLUMN!r} value that was not measured at that frame. "
                "Deduplication may only re-attribute an existing measurement to "
                "the vertex that now stands for its group."
            )


def deduplicate_contact_depth_field(
    input_parquet: Path,
    output_parquet: Path,
    output_csv: Path,
    frame_mappings: Mapping[int, DedupMappingLike],
) -> Path:
    """Reduce a contact depth field by the mapping its CSV was reduced by.

    The mapping **must** come from the :func:`deduplicate_contact_points_csv`
    call that produced *output_csv*.  That is the only way the epsilon applied
    to the sidecar is guaranteed to be the epsilon actually applied to the CSV —
    which under ``monitor: true`` is chosen interactively and appears in no
    config file.  DBSCAN is never re-run here.

    Deduplication removes rows; it does not move them.  ``coordinate_space`` is
    therefore carried through verbatim along with the rest of the metadata: the
    surviving points are exactly where they were.

    Args:
        input_parquet: The block's depth field as the previous stage left it.
        output_parquet: Destination in ``blocks_deduped/``.
        output_csv: The deduplicated CSV **this stage just wrote** for the same
            block — the only CSV the cross-checks are meaningful against.
        frame_mappings: ``frame_index -> DedupMapping`` from the same CSV run.

    Returns:
        *output_parquet*.

    Raises:
        FileNotFoundError: If *input_parquet* does not exist.
        ValueError: If the mapping does not cover exactly the frames the field
            holds, if a frame's mapping disagrees with its row count, if a
            surviving depth was not one the frame already held, if the written
            field and CSV disagree about a frame's contact-point count, or if
            per-frame ``max(|signed_depth_mm|)`` no longer equals the CSV's
            ``contact_depth``.
    """
    input_parquet = Path(input_parquet)
    if not input_parquet.exists():
        raise FileNotFoundError(
            f"Contact depth field sidecar missing: {input_parquet}. The "
            "per-vertex depth field is a required input of postprocessing; "
            "re-run the previous stage for this block rather than "
            "deduplicating without it."
        )

    table, source_metadata = read_contact_depth_field(input_parquet)
    reduced = apply_dedup_mapping_to_field(table, frame_mappings)
    _assert_depth_only_reduced(table, reduced, parquet_name=input_parquet.name)

    # Verbatim apart from ``pipeline_stage``: nothing moved, so the declared
    # space and the rest of the provenance are unchanged, but the file in
    # ``blocks_deduped/`` was written by postprocessing and must say so rather
    # than repeating the merging stamp its input carried.
    metadata: Dict[str, str] = dict(source_metadata)
    metadata["pipeline_stage"] = PIPELINE_STAGE_POSTPROCESSING

    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(reduced, output_parquet, metadata=metadata)

    assert_row_counts_agree_with_csv(reduced, output_csv)
    assert_max_depth_agrees_with_csv(reduced, output_csv)
    return output_parquet

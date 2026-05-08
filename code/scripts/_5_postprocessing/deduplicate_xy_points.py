from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN

from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
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
    """
    if len(points) == 0:
        if return_indices:
            return points, 0, np.empty(0, dtype=np.intp)
        return points, 0

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

    deduped = points[kept]
    n_removed = len(points) - len(deduped)
    assert len(deduped) + n_removed == len(points), (
        f"Deduplication invariant violated: {len(deduped)} + {n_removed} != {len(points)}"
    )
    if return_indices:
        return deduped, n_removed, np.array(kept, dtype=np.intp)
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


def deduplicate_forearm_ply(input_ply: Path, output_ply: Path, epsilon: float = 0.35) -> dict:
    """Deduplicate a forearm PLY point cloud by (x, y) position.

    Loads the PLY, removes (x, y) duplicates keeping the lowest z (outermost
    surface), and writes the result to output_ply.

    Args:
        input_ply: Path to the input PLY file.
        output_ply: Path to write the deduplicated PLY.
        epsilon: Bin size in mm for (x, y) deduplication.

    Returns:
        Dict with keys n_original, n_deduped, n_removed.
    """
    pcd = o3d.io.read_point_cloud(str(input_ply))
    vertices = np.asarray(pcd.points, dtype=np.float64)

    if len(vertices) == 0:
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_ply), pcd)
        return {"n_original": 0, "n_deduped": 0, "n_removed": 0}

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
    }


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
        Dict with keys n_rows_processed, total_points_before, total_points_after.
    """
    df = pd.read_csv(input_csv)

    n_rows_processed = 0
    total_points_before = 0
    total_points_after = 0

    new_contact_points = []
    new_location_x = []
    new_location_y = []
    new_location_z = []

    for _, row in df.iterrows():
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

        deduped_array, _ = deduplicate_xy(pts_array, epsilon)
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
    }

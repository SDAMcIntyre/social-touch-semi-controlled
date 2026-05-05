#!/usr/bin/env python
"""
Sandbox: PLY point cloud viewer with inter-point distance analysis.

Controls
--------
Left-click   — select point A  (red sphere)
Right-click  — select point B  (green sphere)
q / Esc      — quit

Units follow whatever the PLY was saved in (mm for Kinect clouds).

Requires:  pyvista  scipy
           pip install pyvista scipy
"""
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog

try:
    import pyvista as pv
    import vtk
except ImportError:
    sys.exit("Missing dependency — run:  pip install pyvista")

from scipy.spatial import KDTree


# ──────────────────────────────────────────────────────────── file picker ─────

def _open_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a PLY file",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
    )
    root.destroy()
    return path


# ─────────────────────────────────────────────────────────────── stats ────────

def _print_stats(points: np.ndarray) -> float:
    """Print basic cloud stats to terminal; return median NN distance."""
    n = len(points)
    bb_min, bb_max = points.min(axis=0), points.max(axis=0)

    sample_n = min(10_000, n)
    rng = np.random.default_rng(0)
    sample = points[rng.choice(n, sample_n, replace=False)]
    nn = KDTree(sample).query(sample, k=2)[0][:, 1]

    print("\n=== Point Cloud ===")
    print(f"  Points  : {n:,}")
    print(f"  Bounds  :")
    for i, ax in enumerate("XYZ"):
        lo, hi = bb_min[i], bb_max[i]
        print(f"    {ax} :  {lo:10.3f}  →  {hi:10.3f}   (range {hi - lo:.3f})")
    print(f"  Nearest-neighbour distance  (sample n = {sample_n:,}) :")
    print(
        f"    median {np.median(nn):.4f}"
        f"   mean {np.mean(nn):.4f}"
        f"   std {np.std(nn):.4f}"
        f"   min {np.min(nn):.4f}"
        f"   max {np.max(nn):.4f}"
    )
    print()
    return float(np.median(nn))


# ─────────────────────────────────────────────────────────────── main ─────────

def main() -> None:
    path = _open_file()
    if not path:
        print("No file selected — exiting.")
        sys.exit(0)

    print(f"Loading : {path}")
    cloud = pv.read(path)
    points = np.asarray(cloud.points, dtype=np.float64)

    median_nn = _print_stats(points)
    sphere_r = median_nn * 5.0   # selection-marker radius — scales with cloud density
    tree = KDTree(points)

    # ── scene ─────────────────────────────────────────────────────────────────
    pl = pv.Plotter(title="PLY Viewer — Left-click: A  |  Right-click: B")

    # Use vertex colors when present (common in Kinect scans), else uniform blue
    if "RGB" in cloud.point_data:
        pl.add_mesh(cloud, rgb=True, point_size=3, render_points_as_spheres=True, name="cloud")
    else:
        pl.add_mesh(cloud, color="steelblue", point_size=3, render_points_as_spheres=True, name="cloud")

    pl.add_text(
        "Left-click: A (red)   Right-click: B (green)   q: quit",
        position="upper_left",
        font_size=9,
        color="white",
    )

    # ── selection state ────────────────────────────────────────────────────────
    sel: dict = {"A": None, "B": None}

    def _add_marker(pt: np.ndarray, label: str, color: str) -> None:
        # Passing the same name replaces the previous actor for that label
        pl.add_mesh(
            pv.Sphere(radius=sphere_r, center=pt.tolist()),
            color=color,
            name=f"sel_{label}",
        )

    def _update_distance() -> None:
        if sel["A"] is None or sel["B"] is None:
            return
        d = float(np.linalg.norm(sel["A"] - sel["B"]))
        pl.add_text(
            f"A → B : {d:.4f}",
            position="lower_right",
            font_size=14,
            color="yellow",
            name="dist_text",       # same name = replaces previous label
        )
        print(
            f"  Distance A→B : {d:.4f}"
            f"   A = {np.round(sel['A'], 3)}"
            f"   B = {np.round(sel['B'], 3)}"
        )

    # ── left-click → A  (PyVista built-in picker handles drag-vs-pick) ────────
    def _on_left(pos) -> None:
        _, idx = tree.query(pos)
        pt = points[idx]
        sel["A"] = pt
        _add_marker(pt, "A", "red")
        print(f"  [A] idx = {idx:,}   pos = {np.round(pt, 3)}")
        _update_distance()

    pl.enable_point_picking(
        callback=_on_left,
        show_message=False,
        color="red",
        point_size=8,
        tolerance=0.025,
    )

    # ── right-click → B  (VTK observer; right-click drag still zooms) ─────────
    _vtk_picker = vtk.vtkPointPicker()
    _vtk_picker.SetTolerance(0.005)

    def _on_right_press(interactor, _event) -> None:
        x, y = interactor.GetEventPosition()
        _vtk_picker.Pick(x, y, 0, pl.renderer)
        pid = _vtk_picker.GetPointId()
        if pid < 0:
            return
        # Snap to exact cloud point via KDTree (guards against VTK index drift)
        _, idx = tree.query(np.array(_vtk_picker.GetPickPosition()))
        pt = points[idx]
        sel["B"] = pt
        _add_marker(pt, "B", "limegreen")
        print(f"  [B] idx = {idx:,}   pos = {np.round(pt, 3)}")
        _update_distance()
        pl.render()

    pl.iren.interactor.AddObserver("RightButtonPressEvent", _on_right_press, 1.5)

    pl.show()


if __name__ == "__main__":
    main()

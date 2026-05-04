"""
Diagnostic script for the contact_points column.

Parses the contact_points column from a pipeline CSV at any stage and runs
distance analysis against a forearm PLY to verify that all contact points
lie on the forearm surface.

Phases
------
  Phase 0 — Parsing statistics: row counts, point-count distribution,
             empty/filled breakdown
  Phase 1 — Raw extents: bounding boxes of contacts vs vertices before rotation
  Phase 2 — Overlap check: per-axis overlap between contacts and vertices
  Phase 3 — Distance analysis: histogram of nearest-vertex distances (pre- and
             post-rotation), percentile breakdown, count exceeding 15 mm
  Phase 4 — Per-cell worst summary: top-10 rows by max contact-point distance
  Phase 5 — Mesh quality: vertex count, density, Delaunay edge lengths

Usage
-----
    # GUI mode (file picker dialogs):
    python code/scripts/diagnose_contact_points.py

    # CLI mode:
    python code/scripts/diagnose_contact_points.py <pipeline_csv> <forearm_ply>
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from analysis.receptive_field_mapping.rf_data_loader import (
    load_forearm_vertices,
    parse_contact_points,
)
from analysis.receptive_field_mapping.rf_surface_utils import build_delaunay_mesh
from analysis.receptive_field_mapping.tangent_plane_alignment import (
    compute_tangent_plane_rotation,
)


def _load_contact_points(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Parse contact_points column into a flat point array.

    Returns:
        all_pts  – (N, 3) float64, flat array of all parsed points
        row_idx  – (N,) int64, CSV row index each point came from
        per_row  – list[int] of length len(df), point count per row
    """
    if "contact_points" not in df.columns:
        raise ValueError(
            "CSV has no 'contact_points' column. "
            f"Available columns: {list(df.columns)}"
        )
    raw = df["contact_points"]
    per_row: list[int] = []
    pts_list: list[tuple[float, float, float]] = []
    row_list: list[int] = []

    for i, cell in enumerate(raw):
        parsed = parse_contact_points(cell)
        per_row.append(len(parsed))
        for pt in parsed:
            pts_list.append(pt)
            row_list.append(i)

    all_pts = np.array(pts_list, dtype=np.float64).reshape(-1, 3)
    row_idx = np.array(row_list, dtype=np.int64)
    return all_pts, row_idx, per_row


def _bbox(pts: np.ndarray, label: str) -> None:
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = hi - lo
    print(f"  {label}:")
    print(f"    X: [{lo[0]:+10.3f}, {hi[0]:+10.3f}]  span={span[0]:.3f}")
    print(f"    Y: [{lo[1]:+10.3f}, {hi[1]:+10.3f}]  span={span[1]:.3f}")
    print(f"    Z: [{lo[2]:+10.3f}, {hi[2]:+10.3f}]  span={span[2]:.3f}")


def _overlap(contacts: np.ndarray, vertices: np.ndarray) -> None:
    for ax, name in enumerate("XYZ"):
        c_lo, c_hi = contacts[:, ax].min(), contacts[:, ax].max()
        v_lo, v_hi = vertices[:, ax].min(), vertices[:, ax].max()
        overlap = max(0.0, min(c_hi, v_hi) - max(c_lo, v_lo))
        c_span = c_hi - c_lo
        pct = (overlap / c_span * 100) if c_span > 0 else 0
        print(f"  {name}: contact=[{c_lo:+.3f}, {c_hi:+.3f}]  "
              f"vertex=[{v_lo:+.3f}, {v_hi:+.3f}]  "
              f"overlap={overlap:.3f} ({pct:.1f}% of contact span)")


def _distance_analysis(
    contacts: np.ndarray, vertices: np.ndarray, label: str,
) -> np.ndarray:
    tree = cKDTree(vertices)
    distances, _ = tree.query(contacts)
    print(f"  {label}:")
    pcts = [0, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(distances, pcts)
    for p, v in zip(pcts, vals):
        marker = " <<<" if p == 100 and v > 15.0 else ""
        print(f"    p{p:>3d}: {v:8.3f} mm{marker}")
    n_bad = np.sum(distances > 15.0)
    print(f"    Points > 15mm: {n_bad} / {len(distances)}")
    return distances


def _mesh_quality(
    vertices: np.ndarray, contacts: np.ndarray, max_edge_mm: float,
) -> None:
    print(f"  Vertex count: {len(vertices)}")
    tree = cKDTree(vertices)
    dists_to_centroid, _ = tree.query(contacts.mean(axis=0), k=min(100, len(vertices)))
    print(f"  100 nearest vertices to contact centroid: "
          f"range [{dists_to_centroid.min():.3f}, {dists_to_centroid.max():.3f}] mm")

    mesh = build_delaunay_mesh(vertices, max_edge_mm=max_edge_mm)
    if mesh is None:
        print("  WARNING: build_delaunay_mesh returned None — mesh construction failed")
        return
    print(f"  Mesh faces: {len(mesh.faces)}")
    edges = np.concatenate([
        mesh.vertices[mesh.faces[:, 0]] - mesh.vertices[mesh.faces[:, 1]],
        mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 2]],
        mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]],
    ])
    edge_lengths = np.linalg.norm(edges, axis=1)
    print(f"  Edge lengths: min={edge_lengths.min():.3f}  "
          f"median={np.median(edge_lengths):.3f}  "
          f"max={edge_lengths.max():.3f} mm")
    print(f"  Mesh vertex count (after Delaunay): {len(mesh.vertices)} "
          f"(delta={len(mesh.vertices) - len(vertices)})")


def _worst_cells(
    df: pd.DataFrame,
    distances: np.ndarray,
    row_idx: np.ndarray,
    per_row: list[int],
    contacts: np.ndarray,
    n: int = 10,
) -> None:
    id_cols = [c for c in ("block_order_id", "trial_id", "single_touch_id") if c in df.columns]

    row_max_dist = np.zeros(len(df), dtype=np.float64)
    for pt_i, r in enumerate(row_idx):
        if distances[pt_i] > row_max_dist[r]:
            row_max_dist[r] = distances[pt_i]

    rows_with_points = np.where(np.array(per_row) > 0)[0]
    if len(rows_with_points) == 0:
        print("  No rows with contact points to report.")
        return

    worst_rows = rows_with_points[np.argsort(row_max_dist[rows_with_points])[-n:][::-1]]

    print(f"  Top {len(worst_rows)} worst rows (by max point distance, post-rotation):")
    id_header = "  ".join(f"{c:>20s}" for c in id_cols)
    header = f"    {'Row':>6s}  {'MaxDist':>8s}  {'N pts':>5s}  {'Centroid XYZ':>38s}"
    if id_header:
        header += f"  {id_header}"
    print(header)

    for r in worst_rows:
        mask = row_idx == r
        centroid = contacts[mask].mean(axis=0)
        id_vals = "  ".join(f"{df.iloc[r][c]:>20}" for c in id_cols)
        line = (
            f"    {r:6d}  {row_max_dist[r]:8.3f}  {per_row[r]:5d}  "
            f"({centroid[0]:+10.3f}, {centroid[1]:+10.3f}, {centroid[2]:+10.3f})"
        )
        if id_vals:
            line += f"  {id_vals}"
        print(line)


def main(csv_path: Path, forearm_ply: Path) -> None:
    max_edge_mm = 20.0

    print("=" * 70)
    print("contact_points Diagnostic")
    print(f"  CSV: {csv_path}")
    print(f"  PLY: {forearm_ply}")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} rows from CSV")

    all_pts, row_idx, per_row = _load_contact_points(df)
    n_total_pts = len(all_pts)
    n_rows_with_data = sum(1 for c in per_row if c > 0)
    n_rows_empty = len(per_row) - n_rows_with_data

    vertices = load_forearm_vertices(forearm_ply)
    if vertices is None:
        print("FATAL: load_forearm_vertices returned None")
        return
    print(f"Loaded {len(vertices)} forearm vertices")

    if n_total_pts == 0:
        print("\nFATAL: no contact points parsed — cannot continue analysis.")
        return

    # --- Phase 0: Parsing statistics ---
    print(f"\n{'='*70}")
    print("PHASE 0 — Parsing statistics")
    print("=" * 70)
    print(f"  Total rows:           {len(per_row)}")
    print(f"  Rows with ≥1 point:   {n_rows_with_data}")
    print(f"  Rows empty/NaN:       {n_rows_empty}")
    print(f"  Total points parsed:  {n_total_pts}")
    counts_arr = np.array(per_row, dtype=np.int64)
    nonzero = counts_arr[counts_arr > 0]
    print(f"\n  Points-per-row (non-empty rows):")
    print(f"    min:    {nonzero.min()}")
    print(f"    median: {np.median(nonzero):.1f}")
    print(f"    p95:    {np.percentile(nonzero, 95):.1f}")
    print(f"    max:    {nonzero.max()}")
    print(f"\n  Bucket breakdown (all rows):")
    print(f"    0 pts:   {n_rows_empty}")
    print(f"    1 pt:    {np.sum(counts_arr == 1)}")
    print(f"    2–5 pts: {np.sum((counts_arr >= 2) & (counts_arr <= 5))}")
    print(f"    >5 pts:  {np.sum(counts_arr > 5)}")

    # --- Phase 1: Raw extents ---
    print(f"\n{'='*70}")
    print("PHASE 1 — Raw coordinate extents (before rotation)")
    print("=" * 70)
    _bbox(all_pts, "Contacts")
    _bbox(vertices, "Vertices")
    centroid_c = all_pts.mean(axis=0)
    centroid_v = vertices.mean(axis=0)
    sep = np.linalg.norm(centroid_c - centroid_v)
    print(f"\n  Contact centroid: ({centroid_c[0]:+.3f}, {centroid_c[1]:+.3f}, {centroid_c[2]:+.3f})")
    print(f"  Vertex centroid:  ({centroid_v[0]:+.3f}, {centroid_v[1]:+.3f}, {centroid_v[2]:+.3f})")
    print(f"  Centroid separation: {sep:.3f} mm")

    # --- Phase 2: Overlap ---
    print(f"\n{'='*70}")
    print("PHASE 2 — Per-axis overlap (raw coordinates)")
    print("=" * 70)
    _overlap(all_pts, vertices)

    # --- Phase 3: Distance analysis ---
    print(f"\n{'='*70}")
    print("PHASE 3 — Nearest-vertex distance analysis")
    print("=" * 70)
    print("\n  3a. Pre-rotation (raw coordinates):")
    raw_dists = _distance_analysis(all_pts, vertices, "Raw")

    rotation = compute_tangent_plane_rotation(vertices, all_pts.mean(axis=0))
    if rotation is None:
        print("\nFATAL: compute_tangent_plane_rotation returned None")
        return
    print(f"\n  Rotation matrix:\n{rotation}")

    mesh = build_delaunay_mesh(vertices, max_edge_mm=max_edge_mm)
    if mesh is None:
        print("\nFATAL: build_delaunay_mesh returned None")
        return

    rotated_vertices = (rotation @ mesh.vertices.T).T
    rotated_contacts = (rotation @ all_pts.T).T

    print("\n  3b. Post-rotation (rotated coordinates):")
    post_dists = _distance_analysis(rotated_contacts, rotated_vertices, "Rotated")

    delta = post_dists - raw_dists
    print(f"\n  3c. Rotation impact on distances:")
    print(f"    Mean delta: {delta.mean():+.4f} mm")
    print(f"    Max delta:  {delta.max():+.4f} mm")
    print(f"    Min delta:  {delta.min():+.4f} mm")
    if np.max(np.abs(delta)) > 0.01:
        print("    WARNING: rotation changed distances significantly — "
              "mesh.vertices differ from raw vertices")

    # --- Phase 4: Per-cell worst ---
    print(f"\n{'='*70}")
    print("PHASE 4 — Per-cell worst-distance summary (post-rotation)")
    print("=" * 70)
    _worst_cells(df, post_dists, row_idx, per_row, rotated_contacts)

    # --- Phase 5: Mesh quality ---
    print(f"\n{'='*70}")
    print("PHASE 5 — Mesh quality")
    print("=" * 70)
    _mesh_quality(vertices, all_pts, max_edge_mm)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    n_bad_raw = int(np.sum(raw_dists > 15.0))
    n_bad_rot = int(np.sum(post_dists > 15.0))
    print(f"  Total contact points:              {n_total_pts}")
    print(f"  Points exceeding 15mm (raw):       {n_bad_raw} / {n_total_pts}")
    print(f"  Points exceeding 15mm (rotated):   {n_bad_rot} / {n_total_pts}")
    if sep > 50:
        print(f"  LIKELY CAUSE: centroid separation ({sep:.1f}mm) suggests "
              "contacts and vertices are in different coordinate frames")
    elif n_bad_raw == 0 and n_bad_rot > 0:
        print("  LIKELY CAUSE: Delaunay mesh construction changed vertex positions — "
              "raw distances are fine but rotated-mesh distances are not")
    elif n_bad_raw > 0:
        print(f"  NOTE: {n_bad_raw} points already exceed 15mm before rotation — "
              "issue is in the raw data, not the transform")
    else:
        print("  All contact points are within 15mm of the forearm surface.")


def _ask_file(title: str, filetypes: list) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    chosen = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not chosen:
        return None
    return Path(chosen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose contact_points column: parsing stats and surface distance",
    )
    parser.add_argument("csv_path", nargs="?", type=Path, default=None,
                        help="Path to any pipeline-stage CSV with a contact_points column")
    parser.add_argument("forearm_ply", nargs="?", type=Path, default=None,
                        help="Path to forearm PLY")
    args = parser.parse_args()

    csv_path = args.csv_path
    forearm_ply = args.forearm_ply

    if csv_path is None:
        csv_path = _ask_file(
            "Select pipeline CSV (any stage)",
            [("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if csv_path is None:
            print("No CSV selected — exiting.")
            sys.exit(0)

    if forearm_ply is None:
        forearm_ply = _ask_file(
            "Select forearm PLY",
            [("PLY files", "*.ply"), ("All files", "*.*")],
        )
        if forearm_ply is None:
            print("No PLY selected — exiting.")
            sys.exit(0)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)
    if not forearm_ply.exists():
        print(f"ERROR: PLY not found: {forearm_ply}")
        sys.exit(1)

    main(csv_path, forearm_ply)

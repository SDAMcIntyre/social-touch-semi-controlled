"""
Diagnostic script for the RF Explorer nearest-vertex distance bug.

Reproduces the load_explorer_data logic step-by-step with detailed output
to identify why contact points are far from mesh vertices.

Phases
------
  Phase 1 — Raw extents: bounding boxes of contacts vs vertices before rotation
  Phase 2 — Overlap check: per-axis overlap between contacts and vertices
  Phase 3 — Distance analysis: histogram of nearest-vertex distances (pre- and
             post-rotation), percentile breakdown, worst-frame identification
  Phase 4 — Mesh quality: vertex count, density around contact region, hole
             detection via Delaunay edge lengths

Usage
-----
    # GUI mode (file picker dialogs):
    python code/scripts/diagnose_rf_explorer_distance.py

    # CLI mode:
    python code/scripts/diagnose_rf_explorer_distance.py <series_csv> <forearm_ply>
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

from analysis.receptive_field_mapping.rf_data_loader import load_forearm_vertices
from analysis.receptive_field_mapping.rf_surface_utils import build_delaunay_mesh
from analysis.receptive_field_mapping.tangent_plane_alignment import (
    compute_tangent_plane_rotation,
)


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
    distances, indices = tree.query(contacts)
    print(f"  {label}:")
    pcts = [0, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(distances, pcts)
    for p, v in zip(pcts, vals):
        marker = " <<<" if p == 100 and v > 15.0 else ""
        print(f"    p{p:>3d}: {v:8.3f} mm{marker}")
    n_bad = np.sum(distances > 15.0)
    print(f"    Frames > 15mm: {n_bad} / {len(distances)}")
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


def _worst_frames(
    distances: np.ndarray, contacts: np.ndarray, vertices: np.ndarray,
    n: int = 10,
) -> None:
    tree = cKDTree(vertices)
    _, nearest_idx = tree.query(contacts)
    worst = np.argsort(distances)[-n:][::-1]
    print(f"  Top {len(worst)} worst frames:")
    print(f"    {'Frame':>6s}  {'Dist':>8s}  {'Contact XYZ':>36s}  {'Nearest vertex XYZ':>36s}")
    for i in worst:
        c = contacts[i]
        v = vertices[nearest_idx[i]]
        print(f"    {i:6d}  {distances[i]:8.3f}  "
              f"({c[0]:+10.3f}, {c[1]:+10.3f}, {c[2]:+10.3f})  "
              f"({v[0]:+10.3f}, {v[1]:+10.3f}, {v[2]:+10.3f})")


def main(series_csv: Path, forearm_ply: Path) -> None:
    max_edge_mm = 20.0

    print("=" * 70)
    print("RF Explorer Distance Diagnostic")
    print(f"  CSV: {series_csv}")
    print(f"  PLY: {forearm_ply}")
    print("=" * 70)

    # --- Load ---
    df = pd.read_csv(series_csv)
    mask = df["contact_location_x"].notna()
    n_nan = (~mask).sum()
    df = df[mask].reset_index(drop=True)
    contact_pts = df[
        ["contact_location_x", "contact_location_y", "contact_location_z"]
    ].to_numpy(dtype=np.float64)
    print(f"\nLoaded {len(contact_pts)} contact frames ({n_nan} NaN rows dropped)")

    vertices = load_forearm_vertices(forearm_ply)
    if vertices is None:
        print("FATAL: load_forearm_vertices returned None")
        return
    print(f"Loaded {len(vertices)} forearm vertices")

    # --- Phase 1: Raw extents ---
    print(f"\n{'='*70}")
    print("PHASE 1 — Raw coordinate extents (before rotation)")
    print("=" * 70)
    _bbox(contact_pts, "Contacts")
    _bbox(vertices, "Vertices")
    centroid_c = contact_pts.mean(axis=0)
    centroid_v = vertices.mean(axis=0)
    sep = np.linalg.norm(centroid_c - centroid_v)
    print(f"\n  Contact centroid: ({centroid_c[0]:+.3f}, {centroid_c[1]:+.3f}, {centroid_c[2]:+.3f})")
    print(f"  Vertex centroid:  ({centroid_v[0]:+.3f}, {centroid_v[1]:+.3f}, {centroid_v[2]:+.3f})")
    print(f"  Centroid separation: {sep:.3f} mm")

    # --- Phase 2: Overlap ---
    print(f"\n{'='*70}")
    print("PHASE 2 — Per-axis overlap (raw coordinates)")
    print("=" * 70)
    _overlap(contact_pts, vertices)

    # --- Phase 3: Distance analysis ---
    print(f"\n{'='*70}")
    print("PHASE 3 — Nearest-vertex distance analysis")
    print("=" * 70)
    print("\n  3a. Pre-rotation (raw coordinates):")
    raw_dists = _distance_analysis(contact_pts, vertices, "Raw")

    rotation = compute_tangent_plane_rotation(vertices, contact_pts.mean(axis=0))
    if rotation is None:
        print("\nFATAL: compute_tangent_plane_rotation returned None")
        return
    print(f"\n  Rotation matrix:\n{rotation}")

    mesh = build_delaunay_mesh(vertices, max_edge_mm=max_edge_mm)
    if mesh is None:
        print("\nFATAL: build_delaunay_mesh returned None")
        return

    rotated_vertices = (rotation @ mesh.vertices.T).T
    rotated_contacts = (rotation @ contact_pts.T).T

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

    # --- Phase 4: Mesh quality ---
    print(f"\n{'='*70}")
    print("PHASE 4 — Mesh quality")
    print("=" * 70)
    _mesh_quality(vertices, contact_pts, max_edge_mm)

    # --- Worst frames ---
    print(f"\n{'='*70}")
    print("WORST FRAMES (post-rotation)")
    print("=" * 70)
    _worst_frames(post_dists, rotated_contacts, rotated_vertices)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    n_bad_raw = np.sum(raw_dists > 15.0)
    n_bad_rot = np.sum(post_dists > 15.0)
    print(f"  Frames exceeding 15mm (raw):     {n_bad_raw} / {len(raw_dists)}")
    print(f"  Frames exceeding 15mm (rotated): {n_bad_rot} / {len(post_dists)}")
    if sep > 50:
        print(f"  LIKELY CAUSE: centroid separation ({sep:.1f}mm) suggests "
              "contacts and vertices are in different coordinate frames")
    elif n_bad_raw == 0 and n_bad_rot > 0:
        print("  LIKELY CAUSE: Delaunay mesh construction changed vertex positions — "
              "raw distances are fine but rotated-mesh distances are not")
    elif n_bad_raw > 0:
        print(f"  NOTE: {n_bad_raw} frames already exceed 15mm before rotation — "
              "issue is in the raw data, not the transform")


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
        description="Diagnose RF Explorer nearest-vertex distance failures",
    )
    parser.add_argument("series_csv", nargs="?", type=Path, default=None,
                        help="Path to series-augmented CSV")
    parser.add_argument("forearm_ply", nargs="?", type=Path, default=None,
                        help="Path to forearm PLY")
    args = parser.parse_args()

    series_csv = args.series_csv
    forearm_ply = args.forearm_ply

    if series_csv is None:
        series_csv = _ask_file(
            "Select series-augmented CSV",
            [("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if series_csv is None:
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

    if not series_csv.exists():
        print(f"ERROR: CSV not found: {series_csv}")
        sys.exit(1)
    if not forearm_ply.exists():
        print(f"ERROR: PLY not found: {forearm_ply}")
        sys.exit(1)

    main(series_csv, forearm_ply)

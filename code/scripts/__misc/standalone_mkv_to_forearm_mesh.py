"""Standalone MKV-to-Forearm-Mesh Tool.

Extracts a forearm mesh from a single Kinect MKV frame, fully decoupled
from the existing pipeline (no DAG config, no Prefect flow, no session
configs).  The user navigates the video, picks a frame, draws an ROI,
tunes segmentation parameters interactively, and gets a saved mesh file
-- all in one invocation.

Usage::

    python standalone_mkv_to_forearm_mesh.py path/to/recording.mkv
    python standalone_mkv_to_forearm_mesh.py path/to/recording.mkv --output-dir ./meshes
    python standalone_mkv_to_forearm_mesh.py path/to/recording.mkv --save-intermediates

Steps:
    1. Frame selection   -- browse the MKV colour stream, pick one frame
    2. ROI selection     -- draw a rotated rectangle around the forearm
    3. Forearm extraction -- interactive segmentation (downsample, crop,
                            skin-colour filter, DBSCAN clustering)
    4. Cleaning          -- XY-dedup, keep lowest Z per unique (X, Y)
    5. Normal estimation -- Open3D KDTree hybrid search + tangent-plane
    6. Mesh generation   -- 2.5D Delaunay triangulation via SciPy
"""

# ── Standard library ────────────────────────────────────────────────
import argparse
import os
import sys
import tempfile
from pathlib import Path

# ── CuPy import guard ──────────────────────────────────────────────
# CuPy must be imported BEFORE any preprocessing package imports, or
# NumPy 2.0 dtype-registry crashes occur.
# See: docs/development/knowledge-base/note-cupy-import-order.md
try:
    import cupy  # noqa: F401  -- must precede preprocessing imports
except Exception:
    pass

# ── sys.path setup ──────────────────────────────────────────────────
# Allow running this script directly without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # code/scripts/__misc -> repo root
_SRC_ROOT = _REPO_ROOT / "code" / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ── Third-party ─────────────────────────────────────────────────────
import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import tkinter as tk
from scipy.spatial import Delaunay
import trimesh

# ── Project imports ─────────────────────────────────────────────────
from preprocessing.common.gui.video_frame_selector import VideoFrameSelector
from preprocessing.common.gui.frame_roi_rotatable import FrameROIRotatable
from preprocessing.common.data_access.kinect_mkv_manager import KinectMKV, KinectFrame
from preprocessing.forearm_extraction.arm_segmentation import ArmSegmentation
from preprocessing.forearm_extraction.models.forearm_parameters import (
    RegionOfInterest,
    Point,
)


# ====================================================================
# Copied / adapted helpers (avoid pulling in pipeline dependencies)
# ====================================================================

def _get_3d_cuboid_from_roi(frame: KinectFrame, roi: RegionOfInterest) -> np.ndarray:
    """Convert a 2D ROI into 3D corner points for the box filter.

    Adapted from extract_participant_forearm.py::get_3d_cuboid_from_roi.
    """
    top_left = (roi.top_left_corner.x, roi.top_left_corner.y)
    bottom_right = (roi.bottom_right_corner.x, roi.bottom_right_corner.y)
    p1 = frame.convert_xy_to_xyz(top_left)
    p2 = frame.convert_xy_to_xyz(bottom_right)
    return np.array([p1, p2])


def _roi_from_rotated_rect(roi_data: dict) -> RegionOfInterest:
    """Build a RegionOfInterest (with AABB) from a centre-based ROI dict.

    Adapted from define_extraction_parameters.py (lines 266-289).
    roi_data keys: cx, cy, width, height, angle_deg.
    """
    cx, cy = roi_data["cx"], roi_data["cy"]
    w, h = roi_data["width"], roi_data["height"]
    angle = roi_data["angle_deg"]

    a = np.deg2rad(angle)
    cos_a, sin_a = np.cos(a), np.sin(a)
    hw, hh = w / 2.0, h / 2.0
    local_corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = (rot @ local_corners.T).T + np.array([cx, cy])

    x1, y1 = int(corners[:, 0].min()), int(corners[:, 1].min())
    x2, y2 = int(corners[:, 0].max()), int(corners[:, 1].max())

    return RegionOfInterest(
        top_left_corner=Point(x=x1, y=y1),
        bottom_right_corner=Point(x=x2, y=y2),
        angle_deg=angle,
        center_x=float(cx),
        center_y=float(cy),
        width=float(w),
        height=float(h),
    )


def _clean_xy_duplicates(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Remove duplicate (X, Y) points, keeping the one with the lowest Z.

    Adapted from clean_forearm_pointcloud.py (lines 62-81).
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]})
    df["x_round"] = df["x"].round(6)
    df["y_round"] = df["y"].round(6)

    initial_count = len(df)
    cleaned_indices = (
        df.sort_values("z")
        .drop_duplicates(subset=["x_round", "y_round"], keep="first")
        .index
    )
    cleaned_pcd = pcd.select_by_index(cleaned_indices)
    final_count = len(cleaned_indices)

    print(f"   Cleaning: {initial_count} -> {final_count} points "
          f"(removed {initial_count - final_count} XY-duplicates)")
    return cleaned_pcd


# ====================================================================
# Pipeline steps
# ====================================================================

def step_select_frame(mkv_path: str) -> int:
    """Step 1: Browse MKV colour stream and pick a frame index."""
    print("\n=== Step 1: Frame Selection ===")
    print("Use the slider or arrow keys to browse. Click 'Proceed' to confirm.")

    cap = cv2.VideoCapture(mkv_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {mkv_path}")

    try:
        root = tk.Tk()
        root.withdraw()

        selector = VideoFrameSelector(root, cap, title="Select Frame")
        frame_idx = selector.select_frame()

        root.destroy()
    finally:
        cap.release()

    if frame_idx is None:
        print("Frame selection cancelled.")
        sys.exit(0)

    print(f"   Selected frame: {frame_idx}")
    return frame_idx


def step_select_roi(mkv: KinectMKV, frame_idx: int) -> RegionOfInterest:
    """Step 2: Draw a rotated ROI on the colour frame."""
    print("\n=== Step 2: ROI Selection ===")
    print("Draw a rectangle around the forearm. Confirm with Enter/Space.")

    frame: KinectFrame = mkv[frame_idx]
    color = frame.color
    if color is None:
        raise RuntimeError(f"Frame {frame_idx} has no colour data.")

    roi_ui = FrameROIRotatable(
        color,
        is_rgb=False,  # KinectFrame.color returns BGR
        window_title=f"Draw Forearm ROI -- frame {frame_idx}",
    )
    roi_ui.run()
    roi_data = roi_ui.get_roi_data()

    if roi_data is None:
        print("ROI selection cancelled.")
        sys.exit(0)

    roi = _roi_from_rotated_rect(roi_data)
    print(f"   ROI centre: ({roi_data['cx']:.0f}, {roi_data['cy']:.0f}), "
          f"size: ({roi_data['width']:.0f} x {roi_data['height']:.0f}), "
          f"angle: {roi_data['angle_deg']:.1f}")
    return roi


def step_extract_forearm(
    mkv: KinectMKV,
    frame_idx: int,
    roi: RegionOfInterest,
) -> o3d.geometry.PointCloud:
    """Step 3: Interactive forearm segmentation."""
    print("\n=== Step 3: Forearm Extraction ===")
    print("Tune parameters in the GUI. Click 'Process' then 'Continue'.")

    frame: KinectFrame = mkv[frame_idx]
    pcd = frame.generate_o3d_point_cloud()
    if pcd is None or len(pcd.points) == 0:
        raise RuntimeError(f"Frame {frame_idx} produced an empty point cloud.")

    cuboid_corners = _get_3d_cuboid_from_roi(frame, roi)
    segmenter = ArmSegmentation(interactive=True)

    pcd = segmenter.preprocess(pcd, cuboid_corners)
    pcd = segmenter.extract_arm(pcd)

    n_points = len(pcd.points)
    if n_points == 0:
        raise RuntimeError(
            "Segmentation produced an empty point cloud. "
            "Try adjusting the ROI or segmentation parameters."
        )

    print(f"   Extracted forearm: {n_points} points")
    return pcd


def step_clean(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Step 4: In-memory XY-dedup cleaning."""
    print("\n=== Step 4: Cleaning (XY-dedup) ===")
    return _clean_xy_duplicates(pcd)


def step_estimate_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Step 5: In-memory normal estimation."""
    print("\n=== Step 5: Normal Estimation ===")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
    )
    pcd.orient_normals_consistent_tangent_plane(k=100)
    print(f"   Normals estimated for {len(pcd.points)} points")
    return pcd


def step_generate_mesh(
    pcd: o3d.geometry.PointCloud,
    output_path: Path,
    show: bool = True,
) -> trimesh.Trimesh:
    """Step 6: 2.5D Delaunay mesh generation via temp-PLY bridge.

    Uses a temporary PLY file to feed ``define_forearm_mesh``-equivalent
    logic, keeping the function self-contained (no pipeline imports).
    """
    print("\n=== Step 6: Mesh Generation ===")

    # Write point cloud (with normals + colors) to a temp PLY so the
    # Delaunay logic can read it back with full fidelity.
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_ply_path = tmp.name
    try:
        o3d.io.write_point_cloud(tmp_ply_path, pcd)

        # Re-read to extract arrays (matches define_forearm_mesh flow)
        pcd_loaded = o3d.io.read_point_cloud(tmp_ply_path)
        points = np.asarray(pcd_loaded.points)
        input_normals = np.asarray(pcd_loaded.normals) if pcd_loaded.has_normals() else None
        input_colors = np.asarray(pcd_loaded.colors) if pcd_loaded.has_colors() else None
    finally:
        os.unlink(tmp_ply_path)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point array has unexpected shape {points.shape}; expected (N, 3).")

    # 2.5D Delaunay triangulation on XY projection
    xy_points = points[:, 0:2]
    print("   Performing Delaunay triangulation...")
    tri = Delaunay(xy_points)

    mesh = trimesh.Trimesh(
        vertices=points,
        faces=tri.simplices,
        vertex_colors=input_colors,
    )

    # Normal orientation correction
    if input_normals is not None and len(input_normals) == len(points):
        generated_face_normals = mesh.face_normals
        face_vertex_normals = input_normals[mesh.faces]
        avg_input_normals = face_vertex_normals.mean(axis=1)
        dots = np.einsum("ij,ij->i", generated_face_normals, avg_input_normals)
        if np.sum(dots < 0) > (len(dots) / 2):
            print("   Detected inverted winding. Flipping mesh...")
            mesh.invert()
    else:
        if np.mean(mesh.face_normals[:, 2]) < 0:
            print("   Mesh faces downward (Z-). Flipping to Z+.")
            mesh.invert()

    mesh.fix_normals()

    # Save mesh
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    print(f"   Mesh saved: {output_path}")
    print(f"   Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # Visualization
    if show:
        print("\n--- Mesh Viewer ---")
        print("Press 'F' to flip mesh orientation. Close window to finish.")

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        o3d_mesh.compute_vertex_normals()

        def flip_mesh(vis):
            tris = np.asarray(o3d_mesh.triangles)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])
            o3d_mesh.compute_vertex_normals()
            vis.update_geometry(o3d_mesh)
            return False

        key_to_callback = {70: flip_mesh}  # GLFW key code for 'F'
        o3d.visualization.draw_geometries_with_key_callbacks(
            [o3d_mesh],
            key_to_callback,
            window_name="Forearm Mesh Viewer",
            width=1280,
            height=720,
        )

    return mesh


# ====================================================================
# Main orchestrator
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone tool: extract a forearm mesh from a single Kinect MKV frame. "
            "Launches interactive GUIs for frame selection, ROI drawing, and "
            "segmentation parameter tuning, then automatically cleans, estimates "
            "normals, and generates a triangle mesh."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s recording.mkv\n"
            "  %(prog)s recording.mkv --output-dir ./meshes\n"
            "  %(prog)s recording.mkv --save-intermediates\n"
        ),
    )
    parser.add_argument(
        "mkv_path",
        type=str,
        help="Path to the Kinect MKV recording file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files. Defaults to the MKV file's directory.",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate PLY files (segmented, cleaned, with-normals).",
    )
    args = parser.parse_args()

    mkv_path = Path(args.mkv_path).resolve()
    if not mkv_path.exists():
        raise FileNotFoundError(f"MKV file not found: {mkv_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else mkv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = mkv_path.stem

    print(f"MKV file:   {mkv_path}")
    print(f"Output dir: {output_dir}")

    # Step 1: Frame selection (lightweight cv2.VideoCapture)
    frame_idx = step_select_frame(str(mkv_path))

    # Steps 2-3 require KinectMKV for depth/point-cloud access
    with KinectMKV(str(mkv_path)) as mkv:
        # Step 2: ROI selection
        roi = step_select_roi(mkv, frame_idx)

        # Step 3: Forearm extraction
        pcd = step_extract_forearm(mkv, frame_idx, roi)

    if args.save_intermediates:
        seg_path = output_dir / f"{stem}_frame_{frame_idx:04d}_segmented.ply"
        o3d.io.write_point_cloud(str(seg_path), pcd)
        print(f"   [intermediate] Segmented: {seg_path}")

    # Step 4: Cleaning
    pcd = step_clean(pcd)

    if args.save_intermediates:
        clean_path = output_dir / f"{stem}_frame_{frame_idx:04d}_cleaned.ply"
        o3d.io.write_point_cloud(str(clean_path), pcd)
        print(f"   [intermediate] Cleaned: {clean_path}")

    # Step 5: Normal estimation
    pcd = step_estimate_normals(pcd)

    if args.save_intermediates:
        normals_path = output_dir / f"{stem}_frame_{frame_idx:04d}_with_normals.ply"
        o3d.io.write_point_cloud(str(normals_path), pcd)
        print(f"   [intermediate] With normals: {normals_path}")

    # Step 6: Mesh generation
    mesh_path = output_dir / f"{stem}_frame_{frame_idx:04d}_mesh.obj"
    mesh = step_generate_mesh(pcd, mesh_path, show=True)

    # Summary
    print("\n" + "=" * 50)
    print("Done!")
    print(f"  Frame:    {frame_idx}")
    print(f"  Mesh:     {mesh_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces:    {len(mesh.faces)}")
    if args.save_intermediates:
        print(f"  Intermediates saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import tkinter as tk
from tkinter import messagebox

import utils.path_tools as path_tools
from primary_processing import (
    ForearmConfigFileHandler, ForearmConfig,
    KinectConfigFileHandler, KinectConfig,
)
from preprocessing.forearm_extraction import ForearmFrameParametersFileHandler, ForearmParameters
from _3_preprocessing._3_forearm_extraction import (
    load_saved_parameters,
    select_frame_groups,
    define_rois_for_frame_groups,
    save_forearm_parameters,
    extract_forearm,
    clean_forearm_pointcloud,
    define_normals,
    define_forearm_mesh,
)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

FORCE_PROCESSING = True  # Set False to skip sessions that already have a .SUCCESS flag


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameBatch:
    """All resolved inputs and output paths for processing one frame (or frame group)."""

    # Inputs
    depth_video : Path
    frame_params: ForearmParameters

    # Outputs — one path per processing stage
    raw_ply      : Path
    raw_params   : Path
    cleaned_ply  : Path
    cleaned_meta : Path
    normals_ply  : Path
    normals_meta : Path
    mesh_obj     : Path

    @property
    def description(self) -> str:
        """Human-readable label for log messages."""
        p = self.frame_params
        if p.is_averaged:
            return f"frames {p.frame_ids} (representative: {p.representative_frame_id})"
        return f"frame {p.frame_id}"


# ──────────────────────────────────────────────────────────────────────────────
# I/O RESOLUTION
# ──────────────────────────────────────────────────────────────────────────────

def resolve_rgb_video_paths(config_links: List[str], project_data_root: Path) -> List[Path]:
    """
    Resolves each Kinect config link to its corresponding RGB (.mp4) video path.
    Configs whose video file is missing on disk are skipped with a warning.
    Raises FileNotFoundError if no valid paths can be resolved at all.
    """
    print("🔍 Resolving Kinect configs → RGB video paths...")
    rgb_paths: List[Path] = []

    for link in config_links:
        try:
            config_data   = KinectConfigFileHandler.load_and_resolve_config(link)
            kinect_config = KinectConfig(config_data=config_data, database_path=project_data_root)
            rgb_path      = kinect_config.source_video.with_suffix(".mp4")

            if not rgb_path.exists():
                print(f"  ⚠️  RGB video not found, skipping: {rgb_path}")
                continue

            rgb_paths.append(rgb_path)

        except Exception as exc:
            print(f"  ❌ Failed to load Kinect config '{link}': {exc}")

    if not rgb_paths:
        raise FileNotFoundError("No valid RGB video paths found — cannot define ROI.")

    print(f"  📹 {len(rgb_paths)} RGB video(s) located.\n")
    return rgb_paths


def plan_frame_batch(
    params: ForearmParameters,
    rgb_video_paths: List[Path],
    pointclouds_dir: Path,
) -> Optional[FrameBatch]:
    """
    Resolves all input and output paths for one frame, returning a FrameBatch.

    Returns None (with a warning) if the required source videos cannot be found
    on disk, so the caller can safely skip this frame without crashing.
    """
    # Match the frame's video filename to one of the known RGB paths
    rgb_path = next((p for p in rgb_video_paths if p.name == params.video_filename), None)
    if rgb_path is None:
        print(f"  ⚠️  No source found for '{params.video_filename}' — skipping {params}.")
        return None

    # The depth video shares the same stem as the RGB video, but is an .mkv file
    depth_path = rgb_path.with_suffix(".mkv")
    if not depth_path.exists():
        print(f"  ⚠️  Depth video not found — skipping: {depth_path}")
        return None

    base = _build_output_stem(depth_path.stem, params)

    return FrameBatch(
        depth_video  = depth_path,
        frame_params = params,
        raw_ply      = pointclouds_dir / f"{base}.ply",
        raw_params   = pointclouds_dir / f"{base}_extraction_params.json",
        cleaned_ply  = pointclouds_dir / f"{base}_cleaned.ply",
        cleaned_meta = pointclouds_dir / f"{base}_cleaning_stats.json",
        normals_ply  = pointclouds_dir / f"{base}_with_normals.ply",
        normals_meta = pointclouds_dir / f"{base}_with_normals_metadata.json",
        mesh_obj     = pointclouds_dir / f"{base}_mesh.obj",
    )


def _build_output_stem(video_stem: str, params: ForearmParameters) -> str:
    """Returns a descriptive, zero-padded filename stem for a frame or frame group."""
    if params.is_averaged:
        lo, hi = min(params.frame_ids), max(params.frame_ids)
        return f"{video_stem}_frames_{lo:04d}-{hi:04d}_avg_N{len(params.frame_ids)}"
    return f"{video_stem}_frame_{params.frame_id:04d}"


# ──────────────────────────────────────────────────────────────────────────────
# COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def execute_frame_batch(batch: FrameBatch) -> None:
    """
    Runs the four processing steps for a single FrameBatch in sequence:

        1. Extract  — pull the raw forearm point cloud from the depth video
        2. Clean    — remove noise and outliers
        3. Normals  — estimate surface normals on the cleaned cloud
        4. Mesh     — reconstruct a surface mesh from the oriented cloud
    """
    print(f"  🔎 [1/4] Extracting forearm  ({batch.depth_video.name}, {batch.description})")
    extract_forearm(
        video_path=batch.depth_video, video_config=batch.frame_params,
        output_ply_path=batch.raw_ply, output_params_path=batch.raw_params,
        interactive=True,
    )
    print(f"         → {batch.raw_ply.name}")

    print(f"  🧹 [2/4] Cleaning point cloud")
    clean_forearm_pointcloud(
        input_ply_path=batch.raw_ply, output_ply_path=batch.cleaned_ply,
        output_metadata_path=batch.cleaned_meta,
    )
    print(f"         → {batch.cleaned_ply.name}")

    print(f"  🧠 [3/4] Computing normals")
    define_normals(batch.cleaned_ply, batch.normals_ply, batch.normals_meta)
    print(f"         → {batch.normals_ply.name}")

    print(f"  🕸️  [4/4] Building mesh")
    define_forearm_mesh(source=batch.normals_ply, output_path=batch.mesh_obj, show=True)
    print(f"         → {batch.mesh_obj.name}\n")


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────────────

def run_session(session_config: ForearmConfig, project_data_root: Path) -> None:
    """
    Orchestrates the full forearm extraction pipeline for one recording session:

        1. Resolve source videos from the session's Kinect configs
        2. Prompt the user to select frame groups from those videos
        3. Prompt the user to draw a forearm ROI for each group
        4. Save the resulting parameters to disk
        5. Plan one FrameBatch per defined frame (resolves all paths up front)
        6. Execute each FrameBatch in sequence
    """
    print(f"\n🚀 Session: {session_config.session_id}")

    pointclouds_dir = _setup_output_directories(session_config.session_processed_path)
    rgb_video_paths = resolve_rgb_video_paths(session_config.config_file_links, project_data_root)
    metadata_path   = _build_metadata_path(session_config, pointclouds_dir)

    frame_params_list = _annotate_forearm_roi(rgb_video_paths, metadata_path)

    batches = _plan_all_batches(frame_params_list, rgb_video_paths, pointclouds_dir)
    _execute_all_batches(batches)


def _setup_output_directories(session_output_dir: Path) -> Path:
    """Creates the session output directory and pointclouds subfolder. Returns the subfolder."""
    session_output_dir.mkdir(parents=True, exist_ok=True)

    pointclouds_dir = session_output_dir / "forearm_pointclouds"
    pointclouds_dir.mkdir(exist_ok=True)

    print(f"  📂 Output: {pointclouds_dir}")
    return pointclouds_dir


def _build_metadata_path(session_config: ForearmConfig, pointclouds_dir: Path) -> Path:
    """Returns the path where the ROI annotation metadata JSON will be stored."""
    return pointclouds_dir / f"{session_config.session_id}_arm_roi_metadata.json"


def _annotate_forearm_roi(
    rgb_video_paths: List[Path],
    metadata_path: Path,
) -> List[ForearmParameters]:
    """
    Guides the user through the two interactive annotation steps and persists
    the result to disk.

    Step 1 — select frame groups: the user picks which frames (or groups of
    frames) to annotate. Any previously saved groups are pre-filled so the user
    can review rather than re-do prior work.

    Step 2 — draw ROIs: for each selected group, the representative frame is
    shown and the user draws the forearm region. Previously saved ROIs are
    pre-drawn for the same reason.

    The collected parameters are then saved and returned so the caller can
    immediately plan and execute FrameBatches.

    Args:
        rgb_video_paths: RGB video files available for this session.
        metadata_path: Where to load existing parameters from and save new ones to.

    Returns:
        The saved list of ForearmParameters, ready for FrameBatch planning.

    Raises:
        RuntimeError: If the user cancels group selection or no ROIs are defined.
    """
    print("\n✍️  Please annotate the forearm ROI in the interactive windows.\n")

    saved_parameters = load_saved_parameters(str(metadata_path))

    selected_groups = select_frame_groups(
        rgb_video_paths=[str(p) for p in rgb_video_paths],
        saved_parameters=saved_parameters,
    )
    if selected_groups is None:
        raise RuntimeError("Frame group selection was cancelled — cannot continue.")

    parameters = define_rois_for_frame_groups(selected_groups, saved_parameters)
    if not parameters:
        raise RuntimeError("No ROIs were defined — cannot continue.")

    save_forearm_parameters(parameters, str(metadata_path))
    return ForearmFrameParametersFileHandler.load(str(metadata_path))


def _plan_all_batches(
    frame_params_list: List[ForearmParameters],
    rgb_video_paths: List[Path],
    pointclouds_dir: Path,
) -> List[FrameBatch]:
    """
    Resolves paths for all frames up front, before any processing begins.
    Frames whose source videos cannot be found are dropped here with a warning.
    """
    batches = [
        batch for params in frame_params_list
        if (batch := plan_frame_batch(params, rgb_video_paths, pointclouds_dir)) is not None
    ]
    print(f"  📋 {len(batches)}/{len(frame_params_list)} frame(s) planned successfully.\n")
    return batches


def _execute_all_batches(batches: List[FrameBatch]) -> None:
    """Executes each FrameBatch in sequence, catching and logging per-batch errors."""
    for batch in batches:
        print(f"── {batch.depth_video.name}  |  {batch.description}")
        try:
            execute_frame_batch(batch)
        except Exception as exc:
            print(f"  ❌ Error processing {batch.description}: {exc}\n")


# ──────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def batch_process_all_sessions(configs_forearm_dir: Path, project_data_root: Path) -> None:
    """
    Discovers all *.yaml session configs and runs the pipeline for each one.
    After each session, prompts the user to confirm quality and writes (or
    removes) a .SUCCESS flag accordingly. Sessions with an existing flag are
    skipped unless FORCE_PROCESSING is True.
    """
    session_files = _discover_session_configs(configs_forearm_dir)
    total = len(session_files)

    for idx, session_file in enumerate(session_files, start=1):
        print(f"\n{'─' * 60}")
        print(f"  Session {idx}/{total}: {session_file.name}")
        print(f"{'─' * 60}")

        try:
            session_config: ForearmConfig = ForearmConfigFileHandler.load(session_file)

            if _should_skip_session(session_config):
                continue

            run_session(session_config, project_data_root)
            _handle_session_confirmation(session_config)

        except Exception as exc:
            print(f"  ❌ FATAL — skipping session '{session_file.name}': {exc}")

    print(f"\n🎉 Batch complete — {total} session(s) processed.")


def _discover_session_configs(configs_forearm_dir: Path) -> List[Path]:
    """Returns sorted *.yaml files from the config directory, or raises if none found."""
    if not configs_forearm_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {configs_forearm_dir}")

    session_files = sorted(configs_forearm_dir.glob("*.yaml"))
    if not session_files:
        raise FileNotFoundError(f"No *.yaml session configs found in: {configs_forearm_dir}")

    print(f"Found {len(session_files)} session config(s) in '{configs_forearm_dir}'.")
    return session_files


def _should_skip_session(session_config: ForearmConfig) -> bool:
    """Returns True (and logs a message) if this session already has a .SUCCESS flag."""
    if FORCE_PROCESSING:
        return False

    flag_file = _get_flag_file_path(session_config)
    if flag_file.exists():
        print(f"  ⏭️  .SUCCESS flag found — skipping session '{session_config.session_id}'.\n")
        return True

    return False


def _handle_session_confirmation(session_config: ForearmConfig) -> None:
    """Prompts the user to confirm quality, then writes or removes the .SUCCESS flag."""
    flag_file = _get_flag_file_path(session_config)

    if _ask_user_confirmation():
        flag_file.touch()
        print(f"  ✅ Confirmed — flag written: {flag_file.resolve()}")
    else:
        flag_file.unlink(missing_ok=True)
        print(f"  🧹 Not confirmed — flag removed (session will reprocess next run).")


def _get_flag_file_path(session_config: ForearmConfig) -> Path:
    """Returns the .SUCCESS flag path for a given session."""
    return session_config.session_processed_path / "forearm_pointclouds" / ".SUCCESS"


def _ask_user_confirmation() -> bool:
    """Shows a yes/no dialog and returns True if the user confirms."""
    root = tk.Tk()
    root.withdraw()
    return messagebox.askyesno(
        title="Processing Confirmation",
        message="Was the processing correct?"
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🛠️  Initialising batch processing script...\n")
    try:
        project_root        = Path(__file__).resolve().parents[2]
        configs_forearm_dir = project_root / "configs" / "forearm_configs"
        project_data_root   = path_tools.get_project_data_root()

        print(f"  Project root   : {project_root}")
        print(f"  Data root      : {project_data_root}")
        print(f"  Forearm configs: {configs_forearm_dir}\n")

        batch_process_all_sessions(
            configs_forearm_dir=configs_forearm_dir,
            project_data_root=project_data_root,
        )
    except Exception as exc:
        print(f"❌ Setup / execution error: {exc}")
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import tkinter as tk
from tkinter import messagebox

import utils.path_tools as path_tools
from utils import DagConfigHandler, TaskExecutor
from primary_processing import (
    ForearmConfigFileHandler, ForearmConfig,
    KinectConfigFileHandler, KinectConfig,
)
from preprocessing.forearm_extraction import ForearmFrameParametersFileHandler, ForearmParameters
from preprocessing.forearm_extraction import register_session_forearms
from _3_preprocessing._3_forearm_extraction import (
    load_saved_parameters,
    select_frame_groups,
    define_rois_for_frame_groups,
    save_forearm_parameters,
    extract_forearm,
    curate_forearm_pointcloud,
    clean_forearm_pointcloud,
    define_normals,
    define_forearm_mesh,
)


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
    curated_ply  : Path
    curated_meta : Path
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
# ENVIRONMENT SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_environment():
    project_root      = Path(__file__).resolve().parents[2]
    project_data_root = path_tools.get_project_data_root()
    return project_root, project_data_root


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
        curated_ply  = pointclouds_dir / f"{base}_curated.ply",
        curated_meta = pointclouds_dir / f"{base}_curation_metadata.json",
        cleaned_ply  = pointclouds_dir / f"{base}_cleaned.ply",
        cleaned_meta = pointclouds_dir / f"{base}_cleaning_stats.json",
        normals_ply  = pointclouds_dir / f"{base}_with_normals.ply",
        normals_meta = pointclouds_dir / f"{base}_with_normals_metadata.json",
        mesh_obj     = pointclouds_dir / f"{base}_mesh.obj",
    )


def _build_output_stem(video_stem: str, params: ForearmParameters) -> str:
    """Returns a descriptive, zero-padded filename stem for a frame or frame group."""
    return params.build_output_stem(video_stem)


# ──────────────────────────────────────────────────────────────────────────────
# COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def execute_frame_batch(batch: FrameBatch, dag_handler: DagConfigHandler, interactive: bool = True) -> None:
    """
    Runs the five processing steps for a single FrameBatch in sequence:

        1. Extract  — pull the raw forearm point cloud from the depth video
        2. Curate   — manual interactive removal of artifacts via GUI
        3. Clean    — remove noise and outliers
        4. Normals  — estimate surface normals on the cleaned cloud
        5. Mesh     — reconstruct a surface mesh from the oriented cloud

    Each step is controlled by the DAG config: skipped when its primary output
    already exists and ``force_processing`` is ``false``, or when the step is
    disabled (``enabled: false``) or its upstream dependency was not completed.

    Args:
        batch: All resolved input/output paths for this frame.
        dag_handler: DAG config handler (a per-batch copy is made internally so
            completion state is independent across frames).
        interactive: If True, opens the mesh visualization window after step 5.
            The segmentation GUI (step 1) is always interactive so that HSV
            parameters can be tuned. Set False in batch mode to suppress the
            legacy GLFW mesh viewer and avoid spurious warnings at exit.
    """
    batch_dag = dag_handler.copy()

    # Step 1: Extract
    executor = TaskExecutor('extract_forearm', batch.depth_video.stem, batch_dag, monitor=None)
    with executor:
        if executor.can_run:
            force = batch_dag.get_task_options('extract_forearm').get('force_processing', False)
            print(f"  🔎 [1/5] Extracting forearm  ({batch.depth_video.name}, {batch.description})")
            extract_forearm(
                video_path=batch.depth_video, video_config=batch.frame_params,
                output_ply_path=batch.raw_ply, output_params_path=batch.raw_params,
                interactive=True,
                force_processing=force,
            )

    # Step 2: Curate
    executor = TaskExecutor('curate_forearm', batch.depth_video.stem, batch_dag, monitor=None)
    with executor:
        if executor.can_run:
            force = batch_dag.get_task_options('curate_forearm').get('force_processing', False)
            print(f"  ✂️  [2/5] Curating point cloud (manual artifact removal)")
            curate_forearm_pointcloud(
                input_ply_path=batch.raw_ply,
                output_ply_path=batch.curated_ply,
                output_metadata_path=batch.curated_meta,
                force_processing=force,
            )

    # Step 3: Clean
    executor = TaskExecutor('clean_forearm', batch.depth_video.stem, batch_dag, monitor=None)
    with executor:
        if executor.can_run:
            force = batch_dag.get_task_options('clean_forearm').get('force_processing', False)
            print(f"  🧹 [3/5] Cleaning point cloud")
            clean_input = batch.curated_ply if batch.curated_ply.exists() else batch.raw_ply
            clean_forearm_pointcloud(
                input_ply_path=clean_input, output_ply_path=batch.cleaned_ply,
                output_metadata_path=batch.cleaned_meta,
                force_processing=force,
            )

    # Step 4: Normals
    executor = TaskExecutor('define_normals', batch.depth_video.stem, batch_dag, monitor=None)
    with executor:
        if executor.can_run:
            force = batch_dag.get_task_options('define_normals').get('force_processing', False)
            print(f"  🧠 [4/5] Computing normals")
            define_normals(batch.cleaned_ply, batch.normals_ply, batch.normals_meta, force_processing=force)

    # Step 5: Mesh
    executor = TaskExecutor('build_mesh', batch.depth_video.stem, batch_dag, monitor=None)
    with executor:
        if executor.can_run:
            force = batch_dag.get_task_options('build_mesh').get('force_processing', False)
            print(f"  🕸️  [5/5] Building mesh")
            define_forearm_mesh(source=batch.normals_ply, output_path=batch.mesh_obj, show=interactive, force_processing=force)


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────────────

def run_session(session_config: ForearmConfig, project_data_root: Path, dag_handler: DagConfigHandler) -> None:
    """
    Orchestrates the full forearm extraction pipeline for one recording session:

        1. Resolve source videos from the session's Kinect configs
        2. Prompt the user to select frame groups from those videos
        3. Prompt the user to draw a forearm ROI for each group
        4. Save the resulting parameters to disk
        5. Plan one FrameBatch per defined frame (resolves all paths up front)
        6. Execute each FrameBatch in sequence (per-step control via DAG config)
        7. Register all forearm snapshots (skipped when outputs exist and
           ``force_processing`` is ``false``, or when upstream steps are disabled)
    """
    print(f"\n🚀 Session: {session_config.session_id}")

    pointclouds_dir = _setup_output_directories(session_config.session_processed_path)
    rgb_video_paths = resolve_rgb_video_paths(session_config.config_file_links, project_data_root)
    metadata_path   = _build_metadata_path(session_config, pointclouds_dir)

    force_annotate = dag_handler.get_task_options('extract_forearm').get('force_processing', False)
    frame_params_list = _annotate_forearm_roi(rgb_video_paths, metadata_path, force_processing=force_annotate)

    batches = _plan_all_batches(frame_params_list, rgb_video_paths, pointclouds_dir)
    # Batch mode: disable mesh visualization (no interactive inspection needed)
    # and avoid GLFW context corruption that causes spurious warnings at exit.
    _execute_all_batches(batches, dag_handler, interactive=False)

    # Build a session-level DAG copy to check whether register_forearms should run.
    # Simulate frame-level task completions in dependency order so that
    # depends_on: [build_mesh] propagates correctly when upstream steps are disabled.
    session_dag = dag_handler.copy()
    for task in ['extract_forearm', 'curate_forearm', 'clean_forearm', 'define_normals', 'build_mesh']:
        task_config = session_dag.tasks.get(task, {})
        if not task_config.get('enabled', True):
            break
        deps_met = all(d in session_dag.completed_tasks for d in task_config.get('depends_on', []))
        if deps_met:
            session_dag.completed_tasks.add(task)
        else:
            break

    executor = TaskExecutor('register_forearms', session_config.session_id, session_dag, monitor=None)
    with executor:
        if executor.can_run:
            force = session_dag.get_task_options('register_forearms').get('force_processing', False)
            register_session_forearms(session_config.session_id, pointclouds_dir, metadata_path, force_processing=force)


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
    *,
    force_processing: bool = False,
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

    parameters = define_rois_for_frame_groups(selected_groups, saved_parameters, force_processing=force_processing)
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


def _execute_all_batches(batches: List[FrameBatch], dag_handler: DagConfigHandler, interactive: bool = True) -> None:
    """
    Executes each FrameBatch in sequence, catching and logging per-batch errors.

    Args:
        batches: List of planned FrameBatches to process.
        dag_handler: DAG config handler passed to each batch (a copy is made per batch).
        interactive: Passed through to each batch; False (default) disables
            mesh visualization windows, which prevents GLFW context corruption
            and the spurious warnings it causes at script exit.
    """
    for batch in batches:
        print(f"── {batch.depth_video.name}  |  {batch.description}")
        try:
            execute_frame_batch(batch, dag_handler, interactive=interactive)
        except Exception as exc:
            print(f"  ❌ Error processing {batch.description}: {exc}\n")


# ──────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def batch_process_all_sessions(configs_forearm_dir: Path, project_data_root: Path, dag_handler: DagConfigHandler) -> None:
    """
    Discovers all *.yaml session configs and runs the pipeline for each one.
    After each session, prompts the user to confirm quality and writes (or
    removes) a .SUCCESS flag accordingly. Sessions with an existing flag are
    skipped unless ``force_session_processing`` is ``true`` in the DAG config.
    """
    session_files = _discover_session_configs(configs_forearm_dir, dag_handler)
    total = len(session_files)

    for idx, session_file in enumerate(session_files, start=1):
        print(f"\n{'─' * 60}")
        print(f"  Session {idx}/{total}: {session_file.name}")
        print(f"{'─' * 60}")

        try:
            session_config: ForearmConfig = ForearmConfigFileHandler.load(session_file)

            if _should_skip_session(session_config, dag_handler):
                continue

            run_session(session_config, project_data_root, dag_handler)
            _handle_session_confirmation(session_config)

        except Exception as exc:
            print(f"  ❌ FATAL — skipping session '{session_file.name}': {exc}")

    print(f"\n🎉 Batch complete — {total} session(s) processed.")


def _discover_session_configs(configs_forearm_dir: Path, dag_handler: DagConfigHandler) -> List[Path]:
    """Returns sorted *.yaml files from the config directory.

    If ``forearm_config_files`` is set in the DAG config, only those files are
    returned (include-list model). Otherwise all *.yaml files in the directory
    are returned. Raises if the directory is missing or no files are found.
    """
    if not configs_forearm_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {configs_forearm_dir}")

    included = dag_handler.get_parameter('forearm_config_files') or []
    if included:
        session_files = sorted(
            configs_forearm_dir / name for name in included
            if (configs_forearm_dir / name).exists()
        )
    else:
        session_files = sorted(configs_forearm_dir.glob("*.yaml"))

    if not session_files:
        raise FileNotFoundError(f"No *.yaml session configs found in: {configs_forearm_dir}")

    print(f"Found {len(session_files)} session config(s) in '{configs_forearm_dir}'.")
    return session_files


def _should_skip_session(session_config: ForearmConfig, dag_handler: DagConfigHandler) -> bool:
    """Returns True (and logs a message) if this session already has a .SUCCESS flag."""
    if dag_handler.get_parameter('force_session_processing', True):
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
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--dag-config", type=Path, required=True)
    _args = _parser.parse_args()
    print("🛠️  Initialising batch processing script...\n")
    try:
        project_root, project_data_root = setup_environment()
        dag_config_path = _args.dag_config
        dag_handler = DagConfigHandler(dag_config_path)
        forearm_configs_dir = project_root / "configs" / dag_handler.get_parameter('forearm_configs_directory')

        print(f"  Project root   : {project_root}")
        print(f"  Data root      : {project_data_root}")
        print(f"  Forearm configs: {forearm_configs_dir}\n")

        batch_process_all_sessions(
            configs_forearm_dir=forearm_configs_dir,
            project_data_root=project_data_root,
            dag_handler=dag_handler,
        )
    except Exception as exc:
        print(f"❌ Setup / execution error: {exc}")

"""
view_merged_neural_kinect.py
----------------------------
Workflow script for the NeuralKinectViewer — the high-performance 3D visualization
tool for merged neural + Kinect recordings.

Paths are resolved entirely from the standard KinectConfig system; no hardcoded
paths are required. Point the DAG config to the desired kinect_configs entry.

Usage:
    python code/scripts/view_merged_neural_kinect.py

DAG config: configs/view_merged_neural_kinect_dag.yaml

Features:
  - GPU-cropped point cloud (CuPy, ±crop_half_size_mm AABB around contact centroid)
  - Background MKV pre-loading (512-frame ring buffer)
  - Per-sticker velocity compass widgets
  - Live Nerve_freq / contact_depth / contact_area time-series panel
  - Camera centered on contact region
  - Hot-swap block navigation: all blocks from a batch are presented in ONE
    viewer window; the user switches between them via session / block dropdowns.
  - Contact vertices coloured by penetration depth from the per-vertex contact
    depth field sidecar, on a colour scale fixed over the whole recording.

Everything here is LAZY
-----------------------
Building a block spec must stay cheap: it validates paths, resolves transforms
and hands the viewer *references*.  The heavy artifacts — point clouds, meshes,
hand motion, the merged CSV and the contact depth field — are materialised by
``NeuralKinectViewer._load_block`` for the block currently displayed, and for no
other.  A batch is around a hundred blocks, so a spec builder that read anything
substantial would put the whole dataset between the user and the first pixel.
The depth field travels as a zero-argument loader for exactly that reason.

Artifacts come from ``blocks_filtered/``
----------------------------------------
Both the merged CSV and the contact depth field are resolved from
``3_merged/<session>/blocks_filtered/`` — previously the CSV came from
``blocks_merged/``.  The two directories carry different frame sets (the
neural-quality filter removes whole trials), so pairing across them would leave
frames with contact points and no depth rows.  Behaviour change: trials excluded
for unusable neural quality no longer appear in the scene.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from prefect import flow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication

# ---------------------------------------------------------------------------
# CuPy early-init — MUST happen before any preprocessing import.
#
# Several preprocessing subpackages load compiled C-extensions (Open3D,
# pyk4a, …) whose initialisation mutates NumPy's internal dtype registry.
# If CuPy is imported *after* that mutation, its Cython _dtype init crashes:
#     TypeError: Alias 'bool8' was removed in NumPy 2.0
# Importing CuPy here, while NumPy's state is still pristine, avoids the
# conflict.  The result is cached in sys.modules so later `import cupy`
# calls in any module are free.
# See: docs/bugs/cupy-bool8-import-order.md
# ---------------------------------------------------------------------------
try:
    import cupy  # noqa: F401  — early-init only, used later via _CUPY_AVAILABLE
except Exception:
    pass  # GPU unavailable — viewer falls back to NumPy automatically

import re

import utils.path_tools as path_tools
from utils.pipeline.pipeline_config_manager import DagConfigHandler

from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
)

from merging.contact_depth_field_series import (
    BoundedContactDepthFieldCache,
    ContactDepthFieldLoader,
    make_contact_depth_field_loader,
)
from merging.gui.neural_kinect_scene_viewer import NeuralKinectViewer, NeuralKinectBlockSpec
from preprocessing.forearm_extraction import (
    ForearmCatalog,
    ForearmFrameParametersFileHandler,
    get_forearms_with_fallback,
)


# ---------------------------------------------------------------------------
# Transform-key resolution (mirrors get_forearms_with_fallback key structure)
# ---------------------------------------------------------------------------

def _build_transforms_by_forearm_key(
    transforms: Dict,
    current_video_stem: str,
) -> Dict[int, np.ndarray]:
    """Build ``{forearm_dict_key: transform_4x4}`` matching the key structure
    produced by ``get_forearms_with_fallback`` for *current_video_stem*.

    Same-block snapshots are keyed by their ``representative_frame_id``
    (the integer after ``:`` in the transform key).  The fallback entry at
    key ``0`` mirrors the fallback logic of ``get_forearms_with_fallback``:

    1. Same-block entries exist but none at frame 0 → duplicate the earliest.
    2. No same-block entries → use the latest snapshot from the most recent
       preceding block (walk N-1, N-2, … down to 1, first hit wins).
    """
    block_match = re.match(r"(.*_block-order)(\d+)", current_video_stem)

    if block_match is None:
        # Non-block-order video: single transform keyed at 0
        for key, (matrix, _) in transforms.items():
            if key.rsplit(":", 1)[0] == current_video_stem:
                return {0: matrix}
        return {}

    current_prefix = block_match.group(1)
    current_block = int(block_match.group(2))

    result: Dict[int, np.ndarray] = {}
    by_block: Dict[int, list] = {}  # block_num → [(frame_id, matrix)]

    for key, (matrix, _) in transforms.items():
        key_stem, _, frame_str = key.rpartition(":")
        m = re.match(r"(.*_block-order)(\d+)", key_stem)
        if not m or m.group(1) != current_prefix:
            continue
        try:
            block_num = int(m.group(2))
            frame_id = int(frame_str)
        except ValueError:
            continue
        by_block.setdefault(block_num, []).append((frame_id, matrix))

    # Add same-block transforms keyed by frame_id
    for frame_id, matrix in by_block.get(current_block, []):
        result[frame_id] = matrix

    # Ensure key 0 exists (mirrors get_forearms_with_fallback step 2)
    if 0 not in result:
        if result:
            # Same-block entries exist but none at frame 0 — duplicate earliest
            result[0] = result[min(result)]
        else:
            # No same-block transforms — walk back to the most recent preceding block
            for prev_block in range(current_block - 1, 0, -1):
                if prev_block in by_block:
                    best_frame_id, best_matrix = max(
                        by_block[prev_block], key=lambda x: x[0]
                    )
                    result[0] = best_matrix
                    break

    return result


# ---------------------------------------------------------------------------
# Path resolution (mirrors resolve_filenames from merging_pipeline)
# ---------------------------------------------------------------------------

def resolve_viewer_paths(config: KinectConfig) -> Dict[str, Optional[Path]]:
    """
    Derives all paths required by NeuralKinectViewer from a KinectConfig.

    Naming conventions follow the rest of the pipeline:
      - Sticker XYZ CSV  : video_processed_output_dir/handstickers/<stem>_handstickers_xyz_tracked.csv
      - Hand motion      : video_processed_output_dir/kinematics_analysis/<stem>_handmodel_motion.npz
      - Forearm dir      : session_processed_output_dir/forearm_pointclouds/
      - Forearm metadata : <forearm_dir>/<session_id>_arm_roi_metadata.json
      - Merged CSV       : session_merged_output_dir/blocks_filtered/<session_id>_semicontrolled_<block_id>_merged_data.csv
      - Depth field      : session_merged_output_dir/blocks_filtered/<session_id>_semicontrolled_<block_id>_contact_depth_field.parquet

    ``blocks_filtered/``, not ``blocks_merged/``
    --------------------------------------------
    Both artifacts are resolved from the *same* directory on purpose.  The
    neural-quality filter removes whole trials, so ``blocks_merged/`` and
    ``blocks_filtered/`` carry different frame sets; pairing a merged CSV with a
    filtered depth field would leave frames showing ``contact_points`` and no
    depth rows.  The visible consequence is that trials excluded for unusable
    neural quality no longer appear in the scene — which is the correct
    behaviour for a tool whose purpose is inspecting contact *alongside neural
    data*.

    The depth-field path is returned whether or not it exists; deciding what an
    absent sidecar means is ``resolve_contact_depth_field``'s job, not path
    arithmetic's.
    """
    stem = config.source_video.stem
    forearm_dir = config.session_processed_output_dir / "forearm_pointclouds"

    merged_csv: Optional[Path] = None
    depth_field: Optional[Path] = None
    if config.session_merged_output_dir:
        filtered_dir = config.session_merged_output_dir / "blocks_filtered"
        merged_name = f"{config.session_id}_semicontrolled_{config.block_id}_merged_data.csv"
        candidate = filtered_dir / merged_name
        merged_csv = candidate if candidate.exists() else None
        depth_field = (
            filtered_dir
            / f"{config.session_id}_semicontrolled_{config.block_id}_contact_depth_field.parquet"
        )

    return {
        "recording_name": stem,
        "contact_depth_field_path": depth_field,
        "xyz_csv_path": (
            config.video_processed_output_dir / "handstickers"
            / f"{stem}_handstickers_xyz_tracked.csv"
        ),
        "kinect_mkv_path": config.source_video,
        "forearm_pointcloud_dir": forearm_dir,
        "forearm_metadata_path": forearm_dir / f"{config.session_id}_arm_roi_metadata.json",
        # Passed as a plain filename — ForearmCatalog uses it as a lookup key
        "rgb_video_path": Path(f"{stem}.mp4"),
        "hand_motion_path": (
            config.video_processed_output_dir / "kinematics_analysis"
            / f"{stem}_handmodel_motion.npz"
        ),
        "merged_csv_path": merged_csv,
    }


# ---------------------------------------------------------------------------
# Contact depth field
# ---------------------------------------------------------------------------

def build_contact_depth_field_loader(
    config: KinectConfig,
    paths: Dict[str, Optional[Path]],
    cache: BoundedContactDepthFieldCache,
) -> ContactDepthFieldLoader:
    """Return a zero-argument loader for this block's contact depth field.

    **Nothing is read here.**  Building the loader is a closure over a resolved
    path and must stay free: the batch builds one of these per block — around a
    hundred of them — before the viewer window exists.  Reading the field at
    this point was a measured regression (99 blocks × 10^5-10^6 vertices of
    parquet before the first pixel); the read now happens inside the viewer's
    ``_load_block``, for the one block the user opened.

    The whole recording is still read in one go when the loader *is* called, and
    the **global** colour range computed then — before any frame of that block
    is drawn.  Per-frame autoscaling would make the animation lie about relative
    depth, so the range stays a property of the recording, fixed before its
    first render.

    A missing sidecar remains an explicit ABSENT state, not a silent fallback:
    the resolution message is printed at the moment the block is opened, which
    is the whole point of returning ``None`` rather than quietly leaving the
    contact points flat.  A sidecar that exists but cannot be decoded raises, as
    it should.

    Args:
        config: The block's Kinect config, used only for the log prefix.
        paths: Output of :func:`resolve_viewer_paths`.
        cache: Shared bounded cache — see ``BoundedContactDepthFieldCache``.

    Returns:
        A callable returning the series, or ``None`` for an absent sidecar.
    """
    depth_path = paths["contact_depth_field_path"]
    block_name = config.source_video.name

    def _report(message: str) -> None:
        print(f"[{block_name}] {message}")

    if depth_path is None:
        def _no_merged_output_dir() -> None:
            # Not a fallback: an announced absence.  Reported on every open
            # because every open is a moment the user might expect depth.
            _report(
                "No merged output directory is configured, so no contact depth "
                "field can be resolved — depth colouring DISABLED."
            )
            return None

        return _no_merged_output_dir

    return make_contact_depth_field_loader(depth_path, _report, cache)


# ---------------------------------------------------------------------------
# Block-spec builders
# ---------------------------------------------------------------------------

def _build_plain_spec(
    config: KinectConfig,
    paths: Dict[str, Optional[Path]],
    depth_field_loader: Optional[ContactDepthFieldLoader] = None,
) -> Optional[NeuralKinectBlockSpec]:
    """
    Validate required paths and return a ``NeuralKinectBlockSpec`` for the
    plain (non-transformed) viewer task.  Returns ``None`` when any required
    file is missing (with a printed warning).

    *depth_field_loader* is passed through unchanged and **never called here** —
    building a spec must not read a depth field.
    """
    for key in ("xyz_csv_path", "kinect_mkv_path", "forearm_metadata_path"):
        p = paths[key]
        if not p.exists():
            print(
                f"[{config.source_video.name}] Missing required file '{key}': {p} "
                f"— skipping plain viewer."
            )
            return None

    if paths["merged_csv_path"] is None:
        print(f"[{config.source_video.name}] No merged CSV found — pure-3D mode.")
    else:
        print(f"[{config.source_video.name}] Merged CSV found — neural overlay enabled.")

    return NeuralKinectBlockSpec(
        xyz_csv_path=paths["xyz_csv_path"],
        kinect_mkv_path=paths["kinect_mkv_path"],
        forearm_pointcloud_dir=paths["forearm_pointcloud_dir"],
        forearm_metadata_path=paths["forearm_metadata_path"],
        rgb_video_path=paths["rgb_video_path"],
        hand_motion_path=paths["hand_motion_path"],
        recording_name=paths["recording_name"],
        merged_csv_path=paths["merged_csv_path"],
        registration_transforms_by_forearm_key=None,
        contact_depth_field_loader=depth_field_loader,
    )


def _build_transformed_spec(
    config: KinectConfig,
    paths: Dict[str, Optional[Path]],
    options: Dict,
    depth_field_loader: Optional[ContactDepthFieldLoader] = None,
) -> Optional[NeuralKinectBlockSpec]:
    """
    Build and validate a ``NeuralKinectBlockSpec`` for the transformed-viewer
    task.  Returns ``None`` when:
    - Any required path is missing.
    - No registration transforms file exists.
    - No applicable transform key can be resolved for this block.

    *depth_field_loader* is passed through unchanged and **never called here** —
    building a spec must not read a depth field.
    """
    block_name = config.source_video.name

    for key in ("xyz_csv_path", "kinect_mkv_path", "forearm_metadata_path"):
        p = paths[key]
        if not p.exists():
            print(
                f"[{block_name}] Missing required file '{key}': {p} "
                f"— skipping transformed viewer."
            )
            return None

    forearm_dir = paths["forearm_pointcloud_dir"]
    forearm_params = ForearmFrameParametersFileHandler.load(paths["forearm_metadata_path"])
    catalog = ForearmCatalog(forearm_params, forearm_dir)

    registration_transforms = catalog.load_registration_transforms(config.session_id)
    if registration_transforms is None:
        print(
            f"[{block_name}] No registration transforms found "
            f"('{config.session_id}_registration_transforms.json' missing).  Skipping."
        )
        return None

    transforms_by_forearm_key = _build_transforms_by_forearm_key(
        registration_transforms, config.source_video.stem
    )
    if not transforms_by_forearm_key:
        print(
            f"[{block_name}] Could not resolve any registration transforms for "
            f"stem '{config.source_video.stem}'.  Skipping."
        )
        return None

    for fk, T in sorted(transforms_by_forearm_key.items()):
        R = T[:3, :3]
        cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = np.degrees(np.arccos(cos_angle))
        t = T[:3, 3]
        print(
            f"[{block_name}]   forearm_key={fk}: rotation={rotation_deg:.2f}°  "
            f"translation=({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) mm"
        )

    if paths["merged_csv_path"] is None:
        print(f"[{block_name}] No merged CSV — pure-3D transformed mode.")
    else:
        print(f"[{block_name}] Merged CSV found — neural overlay enabled (transformed mode).")

    return NeuralKinectBlockSpec(
        xyz_csv_path=paths["xyz_csv_path"],
        kinect_mkv_path=paths["kinect_mkv_path"],
        forearm_pointcloud_dir=paths["forearm_pointcloud_dir"],
        forearm_metadata_path=paths["forearm_metadata_path"],
        rgb_video_path=paths["rgb_video_path"],
        hand_motion_path=paths["hand_motion_path"],
        recording_name=paths["recording_name"],
        merged_csv_path=paths["merged_csv_path"],
        registration_transforms_by_forearm_key=transforms_by_forearm_key,
        # The same loader OBJECT is shared with the plain spec, and behind it the
        # same bounded cache, so opening this block in both viewers reads the
        # parquet once.  Registered-frame mode rotates and translates the depth
        # *points* with the ICP matrix the cloud, hand mesh and stickers already
        # use; the depth *values* are invariant under a rigid transform, so
        # nothing is recomputed and no second copy of the field is needed.
        contact_depth_field_loader=depth_field_loader,
    )


# ---------------------------------------------------------------------------
# Batch dispatcher (sequential — one viewer window per task type)
# ---------------------------------------------------------------------------

@flow(name="Run Neural-Kinect Viewer Batch Sequentially", log_prints=True)
def run_batch_sequentially(
    block_files: list[Path],
    project_data_root: Path,
    dag_config_path: Path,
) -> None:
    """
    Collect ``NeuralKinectBlockSpec`` objects for every enabled block, then
    open ONE ``NeuralKinectViewer`` per task type (plain and/or transformed).

    This replaces the old per-block sequential viewer approach: instead of
    opening (and closing) a new window for each block, all blocks are
    presented in a single window where the user navigates via dropdowns.
    """
    dag_handler_template = DagConfigHandler(dag_config_path)

    plain_task = "view_neural_kinect_scene"
    transformed_task = "view_neural_kinect_scene_transformed"

    plain_options = dag_handler_template.get_task_options(plain_task) if dag_handler_template.can_run(plain_task) else {}
    transformed_options = dag_handler_template.get_task_options(transformed_task) if dag_handler_template.can_run(transformed_task) else {}

    plain_blocks: Dict[Tuple[str, str], NeuralKinectBlockSpec] = {}
    transformed_blocks: Dict[Tuple[str, str], NeuralKinectBlockSpec] = {}

    # One cache for the whole batch, deliberately tiny.  Two entries is the
    # smallest size that reproduces what the eager version gave *within* a
    # block — plain and transformed specs share a loader, and stepping to the
    # next block and back re-reads nothing — without letting a batch of ~100
    # blocks accumulate in RAM.  An unbounded memo would re-create the very
    # regression this loader indirection exists to fix, just more slowly.
    depth_field_cache = BoundedContactDepthFieldCache(maxsize=2)

    for block_file in block_files:
        print(f"--- Loading block: {block_file.name} ---")
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            config = KinectConfig(config_data=config_data, database_path=project_data_root)
            paths = resolve_viewer_paths(config)
            key: Tuple[str, str] = (config.session_id, config.block_id)

            # Built once per block and shared by both specs.  This does NOT
            # read the sidecar — the viewer calls the loader when it opens the
            # block.  Sharing one loader object means the field is space-
            # invariant under the registration transform, so the two viewers
            # differ only in where the points are drawn, not in what they are,
            # and the second viewer to ask for a still-cached block pays
            # nothing.
            depth_field_loader = build_contact_depth_field_loader(
                config, paths, depth_field_cache
            )

            if dag_handler_template.can_run(plain_task):
                spec = _build_plain_spec(config, paths, depth_field_loader)
                if spec is not None:
                    plain_blocks[key] = spec
                    print(f"[{block_file.name}] Plain spec built for {key}.")

            if dag_handler_template.can_run(transformed_task):
                spec = _build_transformed_spec(
                    config, paths, transformed_options, depth_field_loader
                )
                if spec is not None:
                    transformed_blocks[key] = spec
                    print(f"[{block_file.name}] Transformed spec built for {key}.")

        except Exception as exc:
            print(f"Failed to build spec for {block_file.name}: {exc}")
            continue

    app = QApplication.instance() or QApplication(sys.argv)

    # --- Plain viewer ---
    if plain_blocks:
        crop_half_size = float(plain_options.get("crop_half_size_mm", 400.0))
        first_sid, first_bid = next(iter(plain_blocks))
        print(
            f"Launching plain NeuralKinectViewer with {len(plain_blocks)} block(s)."
        )
        viewer = NeuralKinectViewer(
            all_blocks=plain_blocks,
            initial_session_id=first_sid,
            initial_block_id=first_bid,
            crop_half_size_mm=crop_half_size,
        )
        viewer.show()
        app.exec_()
        QCoreApplication.processEvents()
        dag_handler_template.mark_completed(plain_task)
    else:
        if dag_handler_template.can_run(plain_task):
            print("No valid blocks found for plain viewer — skipping.")

    # --- Transformed viewer ---
    if transformed_blocks:
        crop_half_size = float(transformed_options.get("crop_half_size_mm", 400.0))
        first_sid, first_bid = next(iter(transformed_blocks))
        print(
            f"Launching transformed NeuralKinectViewer with {len(transformed_blocks)} block(s)."
        )
        viewer = NeuralKinectViewer(
            all_blocks=transformed_blocks,
            initial_session_id=first_sid,
            initial_block_id=first_bid,
            crop_half_size_mm=crop_half_size,
        )
        viewer.show()
        app.exec_()
        QCoreApplication.processEvents()
        dag_handler_template.mark_completed(transformed_task)
    else:
        if dag_handler_template.can_run(transformed_task):
            print("No valid blocks found for transformed viewer — skipping.")

    print("All viewer sessions completed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--dag-config", type=Path, required=True)
    _args = _parser.parse_args()
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    dag_config_path = _args.dag_config

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        entries = main_dag_handler.get_parameter("kinect_configs")
        block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")
    except FileNotFoundError:
        print(f"DAG config not found at '{dag_config_path}'.")
        sys.exit(1)

    print("Launching Neural-Kinect Viewer batch (sequential).")
    run_batch_sequentially(
        block_files=block_files,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path,
    )

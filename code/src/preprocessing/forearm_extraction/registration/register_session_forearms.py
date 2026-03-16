"""Session-level forearm registration orchestration.

Loads all extracted forearm point clouds for a session, aligns them via
ICP to a common reference frame, and persists the unified cloud and
per-snapshot transforms to disk.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import open3d as o3d

from utils.should_process_task import should_process_task
from preprocessing.common import PointCloudDataHandler
from preprocessing.forearm_extraction.data_access.forearm_frame_parameters_filehandler import (
    ForearmFrameParametersFileHandler,
)

from .forearm_registrator import ForearmRegistrator
from .registration_workbench import RegistrationWorkbench

logger = logging.getLogger(__name__)


def register_session_forearms(
    session_id: str,
    pointclouds_dir: Path,
    metadata_path: Path,
    *,
    unification_mode: str = "average",
    canonical_key: Optional[str] = None,
    registration_method: str = "global",
    visualize: bool = True,
    force_processing: bool = False,
) -> Optional[Path]:
    """Register all forearm snapshots of a session to a common reference frame.

    Loads the forearm parameters from *metadata_path*, then for each forearm
    loads the ``_with_normals.ply`` point cloud from *pointclouds_dir*.  If
    only one forearm exists, registration is a no-op.

    Produces two artifacts in *pointclouds_dir*:
        - ``{session_id}_unified_registered.ply``  -- merged, aligned cloud
        - ``{session_id}_registration_transforms.json`` -- per-snapshot 4x4s

    Args:
        session_id: The session identifier (used for output filenames).
        pointclouds_dir: Directory containing the per-forearm PLY files.
        metadata_path: Path to the forearm ROI metadata JSON.
        unification_mode: How to unify the snapshots:

            * ``"reference"`` *(default)* — one cloud is held fixed and all
              others are rigidly aligned to it.
            * ``"average"`` — all clouds are first aligned to an internal
              reference, then every transform is re-centred on the SE(3) mean
              so that no snapshot is privileged.

        canonical_key: Snapshot key (``"<video_stem>:<frame_id>"``) to use as
            the fixed reference for ``"reference"`` mode, or as the internal
            ICP reference for ``"average"`` mode.  If ``None`` (default), the
            snapshot with the lowest ``representative_frame_id`` is selected
            automatically.
        registration_method: ICP strategy to use.  One of ``"vanilla"``
            *(default)*, ``"robust"``, ``"multiscale"``, ``"generalized"``,
            ``"trimmed"``, or ``"global"``.  See
            :class:`~preprocessing.forearm_extraction.registration.forearm_registrator.ForearmRegistrator`
            for details on each method.
        visualize: If ``True``, open the :class:`RegistrationWorkbench` GUI so
            the user can tune parameters interactively and click Accept to
            commit.  The workbench handles mode, canonical key, and method
            selection; the *unification_mode*, *canonical_key*, and
            *registration_method* arguments are ignored in this path.
            Closing the window without clicking Accept returns ``None`` and
            writes no artifacts.  If ``False``, registration runs headlessly
            with the supplied arguments.

    Returns:
        Path to the unified registered PLY, or ``None`` if registration was
        skipped (single forearm, insufficient data, or interactive cancel).

    Raises:
        ValueError: If *unification_mode* is unrecognised, or if a non-``None``
            *canonical_key* is not present in the loaded clouds (headless path
            only).
    """
    unified_ply_path = pointclouds_dir / f"{session_id}_unified_registered.ply"
    transforms_path = pointclouds_dir / f"{session_id}_registration_transforms.json"

    if not should_process_task(
        output_paths=[unified_ply_path, transforms_path],
        input_paths=[metadata_path],
        force=force_processing,
    ):
        return unified_ply_path

    # --- Load forearm parameters ---
    params_list = ForearmFrameParametersFileHandler.load(str(metadata_path))
    if not params_list:
        logger.info("No forearm parameters found — skipping registration.")
        return None

    if len(params_list) == 1:
        logger.info("Single forearm snapshot — registration is a no-op.")
        return None

    # --- Load point clouds keyed by "video_stem:frame_id" ---
    clouds: Dict[str, o3d.geometry.PointCloud] = {}
    for params in params_list:
        video_stem = Path(params.video_filename).stem
        output_stem = params.build_output_stem(video_stem)
        ply_path = pointclouds_dir / f"{output_stem}_with_normals.ply"

        cloud = PointCloudDataHandler.load(str(ply_path))
        if cloud is None:
            logger.warning(
                "Could not load PLY for %s frame %d (%s) — skipping.",
                video_stem,
                params.representative_frame_id,
                ply_path.name,
            )
            continue

        key = f"{video_stem}:{params.representative_frame_id}"
        clouds[key] = cloud

    if len(clouds) < 2:
        logger.warning("Fewer than 2 forearm clouds loaded — skipping registration.")
        return None

    if visualize:
        # --- Interactive path: workbench handles mode/canonical/registration ---
        workbench = RegistrationWorkbench(clouds)
        workbench.show()

        if not workbench.accepted:
            logger.info("Registration workbench cancelled — no artifacts written.")
            return None

        result = workbench.get_result()
        transforms         = result.transforms
        unified_cloud      = result.unified_cloud
        resolved_canonical = result.canonical_key
        unification_mode   = result.mode
        process_parameters = {
            "registration_method":        result.registration_method,
            "max_correspondence_distance": result.max_correspondence_distance,
            "icp_max_iteration":           result.icp_max_iteration,
        }

    else:
        # --- Headless path: validate mode and resolve canonical manually ---
        if unification_mode not in ("reference", "average"):
            raise ValueError(
                f"Unknown unification_mode {unification_mode!r}. "
                "Expected 'reference' or 'average'."
            )

        if canonical_key is not None:
            if canonical_key not in clouds:
                raise ValueError(
                    f"canonical_key {canonical_key!r} not found in loaded clouds. "
                    f"Available keys: {sorted(clouds)}"
                )
            resolved_canonical = canonical_key
            logger.info("Using provided canonical reference: %s", resolved_canonical)
        else:
            # Auto-select: lowest representative_frame_id, tie-break by video_stem.
            resolved_canonical = min(
                clouds.keys(),
                key=lambda k: (int(k.rsplit(":", 1)[1]), k.rsplit(":", 1)[0]),
            )
            logger.info("Auto-selected canonical reference: %s", resolved_canonical)

        registrator = ForearmRegistrator(clouds[resolved_canonical])

        if unification_mode == "reference":
            transforms = registrator.register_all(
                clouds, resolved_canonical, method=registration_method
            )
        else:  # "average"
            transforms = registrator.register_all_to_average(
                clouds, resolved_canonical, method=registration_method
            )

        unified_cloud = registrator.build_unified_cloud(
            clouds, transforms, resolved_canonical
        )
        process_parameters = {
            "registration_method":        registration_method,
            "max_correspondence_distance": 0.10,
            "icp_max_iteration":           200,
        }

    # --- Save artifacts ---
    ForearmRegistrator.save_unified_cloud(unified_cloud, unified_ply_path)
    ForearmRegistrator.save_transforms(
        transforms,
        resolved_canonical,
        transforms_path,
        mode=unification_mode,
        process_parameters=process_parameters,
    )

    logger.info(
        "Registration complete — %d forearm(s) aligned. "
        "Unified cloud: %s, Transforms: %s",
        len(clouds),
        unified_ply_path.name,
        transforms_path.name,
    )

    return unified_ply_path

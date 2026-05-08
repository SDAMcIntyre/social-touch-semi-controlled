"""Postprocessing step 0: Fetch the forearm-of-reference PLY into a canonical location.

Resolves the unified registered forearm PLY for the session (or the single
forearm PLY for single-forearm sessions) and copies it to
``forearm_source/{session_id}_forearm.ply`` under the pipeline output directory.

This makes the forearm a first-class, explicitly-tracked artifact that flows
through every subsequent pipeline stage.
"""
import logging
import shutil
from pathlib import Path
from typing import List

from utils.should_process_task import should_process_task, clean_task_outputs
from primary_processing import KinectConfig

logger = logging.getLogger(__name__)


def _find_forearm_ply(session_configs: List[KinectConfig]) -> Path:
    """Return the unified registered PLY, or the single forearm PLY for single-forearm sessions.

    Args:
        session_configs: KinectConfig objects for the session (one per block).

    Returns:
        Path to the resolved forearm PLY.

    Raises:
        FileNotFoundError: If no forearm PLY is found in ``forearm_pointclouds/``.
    """
    first = session_configs[0]
    forearm_dir = first.session_processed_output_dir / "forearm_pointclouds"
    unified = forearm_dir / f"{first.session_id}_unified_registered.ply"
    if unified.exists():
        return unified

    # Single-forearm fallback: any .ply in the directory
    plies = sorted(forearm_dir.glob("*.ply"))
    if plies:
        logger.info(
            "[%s] No unified registered PLY found; using single forearm PLY: %s",
            first.session_id,
            plies[0].name,
        )
        return plies[0]

    raise FileNotFoundError(
        f"[{first.session_id}] No forearm PLY found in {forearm_dir}. "
        "Expected either a unified registered PLY or a single forearm PLY."
    )


def fetch_forearm_of_reference(
    session_configs: List[KinectConfig],
    output_dir: Path,
    *,
    force_processing: bool = False,
) -> Path:
    """Resolve and copy the session forearm PLY into ``forearm_source/``.

    Locates the forearm-of-reference PLY for the session (preferring
    ``{session_id}_unified_registered.ply``; falling back to any single PLY
    in ``forearm_pointclouds/``) and copies it to
    ``{output_dir}/{session_id}_forearm.ply``.

    Args:
        session_configs: KinectConfig objects for the session (one per block).
        output_dir: Destination directory (``forearm_source/``).
        force_processing: Re-run even if the output is up-to-date.

    Returns:
        Path to the copied forearm PLY in *output_dir*.

    Raises:
        FileNotFoundError: If no forearm PLY is found for the session (fail-fast).
    """
    session_id = session_configs[0].session_id
    source_ply = _find_forearm_ply(session_configs)
    dest_ply = output_dir / f"{session_id}_forearm.ply"

    if not should_process_task(
        input_paths=[source_ply],
        output_paths=[dest_ply],
        force=force_processing,
    ):
        logger.info("[%s] Forearm source up-to-date. Skipping.", session_id)
        return dest_ply
    clean_task_outputs(dest_ply)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_ply, dest_ply)
    logger.info(
        "[%s] Fetched forearm PLY: %s → %s",
        session_id,
        source_ply.name,
        dest_ply,
    )
    return dest_ply

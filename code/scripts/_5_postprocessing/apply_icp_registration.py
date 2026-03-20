"""Postprocessing step 1: Apply ICP registration transforms to merged CSVs.

Reads per-block merged CSVs from ``blocks_merged/``, applies the pre-computed
4x4 ICP registration transform to the spatial columns (overwriting originals),
and writes the results to ``blocks_registered/``.

Single-forearm sessions (no ``registration_transforms.json``) pass through
unchanged with files copied as-is.
"""
import logging
from pathlib import Path
from typing import List

import pandas as pd

from utils.should_process_task import should_process_task, clean_task_outputs
from preprocessing.forearm_extraction import (
    ForearmRegistrator,
    get_transform_schedule,
    transform_spatial_columns_scheduled,
)
from primary_processing import KinectConfig

logger = logging.getLogger(__name__)


def apply_icp_registration(
    input_files: List[Path],
    session_configs: List["KinectConfig"],
    output_dir: Path,
    *,
    force_processing: bool = False,
) -> List[Path]:
    """Apply ICP registration transforms to merged CSVs.

    For multi-forearm sessions, loads ``registration_transforms.json`` from
    ``forearm_pointclouds/`` and applies the appropriate per-block 4x4 rigid
    transform to spatial columns in place.  Single-forearm sessions (no
    transforms file) have their files copied to *output_dir* unchanged.

    Args:
        input_files: Per-block merged CSVs from ``blocks_merged/``.
        session_configs: KinectConfig objects (one per block, same session).
        output_dir: Destination directory (``blocks_registered/``).
        force_processing: Re-run even if outputs are up-to-date.

    Returns:
        List of output CSV paths in *output_dir*.
    """
    # Idempotency check: output filenames mirror input filenames
    expected_outputs = [output_dir / f.name for f in input_files]

    if not should_process_task(
        input_paths=input_files,
        output_paths=expected_outputs,
        force=force_processing,
    ):
        logger.info("[%s] ICP registration up-to-date. Skipping.", output_dir.name)
        return expected_outputs
    clean_task_outputs(expected_outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load registration transforms (keyed by session from first config)
    first_config = session_configs[0]
    forearm_pc_dir = first_config.session_processed_output_dir / "forearm_pointclouds"
    transforms_path = forearm_pc_dir / f"{first_config.session_id}_registration_transforms.json"
    transforms_data = ForearmRegistrator.load_transforms(transforms_path)

    if transforms_data is None:
        logger.info(
            "[%s] No registration transforms found (%s). "
            "Single-forearm session — copying files unchanged.",
            output_dir.name,
            transforms_path.name,
        )
        for input_path in input_files:
            pd.read_csv(input_path).to_csv(output_dir / input_path.name, index=False)
        return expected_outputs

    # Multi-forearm: apply scheduled transforms per block
    output_paths = []
    for input_path, config in zip(input_files, session_configs):
        out_path   = output_dir / input_path.name
        video_stem = f"{config.session_id}_semicontrolled_{config.block_id}".replace("block-order-", "block-order")
        df         = pd.read_csv(input_path)

        max_frame = int(df["frame_index"].dropna().max()) if "frame_index" in df.columns and df["frame_index"].notna().any() else 0

        schedule = get_transform_schedule(transforms_data["transforms"], video_stem, max_frame)

        if not schedule:
            logger.warning(
                "[%s] No applicable transform for '%s' "
                "(no preceding snapshot). Passing through unchanged.",
                output_dir.name,
                video_stem,
            )
            df.to_csv(out_path, index=False)
            output_paths.append(out_path)
            continue

        logger.info(
            "[%s] Applying %d transform segment(s) for '%s'.",
            output_dir.name,
            len(schedule),
            video_stem,
        )
        transform_spatial_columns_scheduled(df, schedule).to_csv(out_path, index=False)
        output_paths.append(out_path)
        logger.info("[%s] Wrote registered CSV: %s", output_dir.name, out_path.name)

    return output_paths

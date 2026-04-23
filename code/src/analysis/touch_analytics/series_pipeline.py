# series_pipeline.py
"""
Stage 2a: series-level transform pipeline.

Loads raw session CSVs, computes per-frame kinematics (velocity, acceleration),
and saves augmented CSVs with the new columns added.

Output layout
-------------
<output_dir>/
  <session_id>_series_augmented.csv
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from utils.should_process_task import should_process_task, clean_task_outputs
from .pipeline_shared import _TqdmLineWrapper, session_id_from_path
from .preparation.loader import load_session_csv
from .preparation.block_id import ensure_block_id_column
from .preparation.grouping import group_touches
from .representation.series_level.kinematics import get_kinematics
from .representation.series_level.pressure import compute_geo_pressure
from .representation.series_level.mechanics import compute_mos_series, MOS_COLUMNS


def run_series_transforms(
    input_items: List[Tuple[Path, Path]],
    transforms: dict,
    output_dir: Path,
    force: bool = False,
) -> List[Path]:
    """
    Compute and persist per-frame kinematics for each session.

    Parameters
    ----------
    input_items
        List of (aggregated_session_csv, database_root_path) tuples.
    transforms
        Dict with transform configs, e.g. ``{'kinematics': {'enabled': True, 'fps': 30.0}}``.
    output_dir
        Directory where augmented CSVs are written.
    force
        Override idempotency checks.

    Returns
    -------
    List of paths to written augmented CSVs.
    """
    transforms = transforms or {}
    kinematics_cfg = transforms.get('kinematics', {})
    kinematics_enabled = kinematics_cfg.get('enabled', True)
    fps = kinematics_cfg.get('fps', 30.0)

    pressure_cfg = transforms.get('pressure', {})
    pressure_enabled = pressure_cfg.get('enabled', True)

    mos_cfg = transforms.get('mechanics_of_solids', {})
    mos_enabled = mos_cfg.get('enabled', False)
    mos_E_kpa = mos_cfg.get('youngs_modulus_kpa', 100.0)
    mos_h_mm = mos_cfg.get('skin_thickness_mm', 1.5)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== series transforms pipeline: {len(input_items)} sessions, "
        f"kinematics={'enabled' if kinematics_enabled else 'disabled'}, "
        f"pressure={'enabled' if pressure_enabled else 'disabled'}, "
        f"mos={'enabled' if mos_enabled else 'disabled'} ===",
        flush=True,
    )

    written: List[Path] = []

    with tqdm(total=len(input_items), desc="series_transforms", unit="session",
              file=_TqdmLineWrapper(sys.stdout)) as progress:
        for input_file, _ in input_items:
            result = _transform_session(
                input_file=input_file,
                output_dir=output_dir,
                kinematics_enabled=kinematics_enabled,
                fps=fps,
                pressure_enabled=pressure_enabled,
                mos_enabled=mos_enabled,
                mos_E_kpa=mos_E_kpa,
                mos_h_mm=mos_h_mm,
                force=force,
            )
            if result is not None:
                written.append(result)
            session_id = session_id_from_path(input_file)
            progress.set_postfix_str(f"series_transforms: {session_id}", refresh=False)
            progress.update(1)

    print(f"=== series transforms pipeline complete: {len(written)} outputs ===", flush=True)
    return written


def _transform_session(
    input_file: Path,
    output_dir: Path,
    kinematics_enabled: bool,
    fps: float,
    pressure_enabled: bool,
    mos_enabled: bool,
    mos_E_kpa: float,
    mos_h_mm: float,
    force: bool,
) -> Path | None:
    session_id = session_id_from_path(input_file)
    output_path = output_dir / f"{session_id}_series_augmented.csv"

    if not kinematics_enabled:
        logging.warning(
            f"series_pipeline: kinematics disabled — skipping {session_id}"
        )
        return None

    if not force and output_path.exists():
        try:
            if not should_process_task(
                input_paths=[input_file],
                output_paths=[output_path],
                force=False,
            ):
                print(f"  [series_transforms] {session_id} — up to date", flush=True)
                return output_path
        except FileNotFoundError:
            pass
    clean_task_outputs(output_path)

    try:
        df = load_session_csv(input_file)
    except Exception as exc:
        logging.error(f"series_pipeline: failed to load {input_file}: {exc}")
        return None

    df = ensure_block_id_column(df)
    groups = group_touches(df)

    vel_series: dict[int, pd.Series] = {}
    accel_series: dict[int, pd.Series] = {}
    pressure_series: dict[int, pd.Series] = {}
    mos_cols_series: dict[int, dict[str, pd.Series]] = {}

    for (_, _, touch_id), group in groups:
        if group.empty or touch_id == 0:
            continue
        vel, accel = get_kinematics(group, fps=fps)
        vel_series[id(group)] = vel
        accel_series[id(group)] = accel
        if pressure_enabled:
            pressure_series[id(group)] = compute_geo_pressure(group)
        if mos_enabled:
            mos_cols_series[id(group)] = compute_mos_series(group, mos_E_kpa, mos_h_mm, fps)

    df = df.copy()
    if vel_series:
        df['velocity_magnitude'] = pd.concat(vel_series.values()).reindex(df.index)
        df['acceleration_magnitude'] = pd.concat(accel_series.values()).reindex(df.index)
    if pressure_series:
        df['geo_pressure'] = pd.concat(pressure_series.values()).reindex(df.index)
    if mos_cols_series:
        for col in MOS_COLUMNS:
            df[col] = pd.concat(
                [group_mos[col] for group_mos in mos_cols_series.values()]
            ).reindex(df.index)

    try:
        df.to_csv(output_path, index=False)
        print(
            f"  [series_transforms] {session_id} — {len(df)} rows → {output_path.name}",
            flush=True,
        )
    except Exception as exc:
        logging.error(f"series_pipeline: failed to save {output_path}: {exc}")
        return None

    return output_path

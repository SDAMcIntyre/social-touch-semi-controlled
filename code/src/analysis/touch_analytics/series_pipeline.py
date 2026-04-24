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
from .preparation.interpolation import interpolate_touch_columns
from .representation.series_level.kinematics import (
    resolve_hand_position,
    compute_velocity_magnitudes,
    compute_acceleration_magnitudes,
    STICKER_INPUT_COLUMNS,
    HAND_POSITION_COLUMNS,
)
from .representation.series_level.pressure import compute_pressure, PRESSURE_INPUT_COLUMNS
from .representation.series_level.mechanics import compute_mos_series, MOS_COLUMNS, MOS_INPUT_COLUMNS


def run_series_transforms(
    input_items: List[Tuple[Path, Path]],
    transforms: dict,
    output_dir: Path,
    force: bool = False,
    preparation_dir: Path | None = None,
) -> List[Path]:
    """
    Compute and persist per-frame series transforms for each session.

    Parameters
    ----------
    input_items
        List of (aggregated_session_csv, database_root_path) tuples.
    transforms
        Dict with transform configs, e.g.::

            {
                'hand_position': {'enabled': True, 'drop_used_inputs': False},
                'hand_velocity': {'enabled': True, 'drop_used_inputs': False},
                'hand_acceleration': {'enabled': True, 'drop_used_inputs': False},
                'pressure': {'enabled': True, 'drop_used_inputs': False},
                'mechanics_of_solids': {'enabled': False, ...},
            }
    output_dir
        Directory where augmented CSVs are written.
    force
        Override idempotency checks.

    Returns
    -------
    List of paths to written augmented CSVs.
    """
    transforms = transforms or {}

    hand_pos_cfg = transforms.get('hand_position', {})
    hand_pos_enabled = hand_pos_cfg.get('enabled', True)
    hand_pos_drop = hand_pos_cfg.get('drop_used_inputs', False)

    hand_vel_cfg = transforms.get('hand_velocity', {})
    hand_vel_enabled = hand_vel_cfg.get('enabled', True)
    hand_vel_drop = hand_vel_cfg.get('drop_used_inputs', False)

    hand_accel_cfg = transforms.get('hand_acceleration', {})
    hand_accel_enabled = hand_accel_cfg.get('enabled', True)
    hand_accel_drop = hand_accel_cfg.get('drop_used_inputs', False)

    pressure_cfg = transforms.get('pressure', {})
    pressure_enabled = pressure_cfg.get('enabled', True)
    pressure_drop = pressure_cfg.get('drop_used_inputs', False)

    mos_cfg = transforms.get('mechanics_of_solids', {})
    mos_enabled = mos_cfg.get('enabled', False)
    mos_drop = mos_cfg.get('drop_used_inputs', False)
    mos_E_kpa = mos_cfg.get('youngs_modulus_kpa', 100.0)
    mos_h_mm = mos_cfg.get('skin_thickness_mm', 1.5)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== series transforms pipeline: {len(input_items)} sessions, "
        f"hand_position={'enabled' if hand_pos_enabled else 'disabled'}, "
        f"hand_velocity={'enabled' if hand_vel_enabled else 'disabled'}, "
        f"hand_acceleration={'enabled' if hand_accel_enabled else 'disabled'}, "
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
                hand_pos_enabled=hand_pos_enabled,
                hand_pos_drop=hand_pos_drop,
                hand_vel_enabled=hand_vel_enabled,
                hand_vel_drop=hand_vel_drop,
                hand_accel_enabled=hand_accel_enabled,
                hand_accel_drop=hand_accel_drop,
                pressure_enabled=pressure_enabled,
                pressure_drop=pressure_drop,
                mos_enabled=mos_enabled,
                mos_drop=mos_drop,
                mos_E_kpa=mos_E_kpa,
                mos_h_mm=mos_h_mm,
                force=force,
                preparation_dir=preparation_dir,
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
    hand_pos_enabled: bool,
    hand_pos_drop: bool,
    hand_vel_enabled: bool,
    hand_vel_drop: bool,
    hand_accel_enabled: bool,
    hand_accel_drop: bool,
    pressure_enabled: bool,
    pressure_drop: bool,
    mos_enabled: bool,
    mos_drop: bool,
    mos_E_kpa: float,
    mos_h_mm: float,
    force: bool,
    preparation_dir: Path | None = None,
) -> Path | None:
    session_id = session_id_from_path(input_file)
    output_path = output_dir / f"{session_id}_series_augmented.csv"

    if preparation_dir is not None:
        source_file = preparation_dir / f"{session_id}_prepared.csv"
        if not source_file.exists():
            raise FileNotFoundError(
                f"series_pipeline: prepared CSV not found for {session_id}: {source_file}. "
                "Run touch_preparation first, or set preparation_dir=None to use inline Stage 1."
            )
    else:
        source_file = input_file

    if not force and output_path.exists():
        try:
            if not should_process_task(
                input_paths=[source_file],
                output_paths=[output_path],
                force=False,
            ):
                print(f"  [series_transforms] {session_id} — up to date", flush=True)
                return output_path
        except FileNotFoundError:
            pass
    clean_task_outputs(output_path)

    try:
        df = load_session_csv(source_file)
    except Exception as exc:
        logging.error(f"series_pipeline: failed to load {source_file}: {exc}")
        return None

    if preparation_dir is None:
        df = ensure_block_id_column(df)
        df = interpolate_touch_columns(df)
    groups = group_touches(df)

    need_velocity = hand_vel_enabled or hand_accel_enabled or mos_enabled

    hand_pos_series: dict[int, pd.DataFrame] = {}
    vel_series: dict[int, pd.Series] = {}
    accel_series: dict[int, pd.Series] = {}
    pressure_series: dict[int, pd.Series] = {}
    mos_cols_series: dict[int, dict[str, pd.Series]] = {}

    for (_, _, touch_id), group in groups:
        if group.empty or touch_id == 0:
            continue

        hand_pos = resolve_hand_position(group)
        hand_pos_series[id(group)] = hand_pos

        if need_velocity:
            vel = compute_velocity_magnitudes(hand_pos)
            accel = compute_acceleration_magnitudes(vel)

            if hand_vel_enabled:
                vel_series[id(group)] = vel
            if hand_accel_enabled:
                accel_series[id(group)] = accel

            if mos_enabled:
                aug_group = pd.concat([group, hand_pos], axis=1).copy()
                aug_group['velocity_magnitude'] = vel
                aug_group['acceleration_magnitude'] = accel
                mos_cols_series[id(group)] = compute_mos_series(aug_group, mos_E_kpa, mos_h_mm)

        if pressure_enabled:
            pressure_series[id(group)] = compute_pressure(group)

    df = df.copy()

    if hand_pos_enabled and hand_pos_series:
        df[HAND_POSITION_COLUMNS] = pd.concat(hand_pos_series.values()).reindex(df.index)
    if vel_series:
        df['velocity_magnitude'] = pd.concat(vel_series.values()).reindex(df.index)
    if accel_series:
        df['acceleration_magnitude'] = pd.concat(accel_series.values()).reindex(df.index)
    if pressure_series:
        df['pressure'] = pd.concat(pressure_series.values()).reindex(df.index)
    if mos_cols_series:
        for col in MOS_COLUMNS:
            df[col] = pd.concat(
                [group_mos[col] for group_mos in mos_cols_series.values()]
            ).reindex(df.index)

    cols_to_drop = []
    if hand_pos_drop:
        cols_to_drop.extend([c for c in STICKER_INPUT_COLUMNS if c in df.columns])
    if hand_vel_drop:
        cols_to_drop.extend([c for c in HAND_POSITION_COLUMNS if c in df.columns])
    if hand_accel_drop and 'velocity_magnitude' in df.columns:
        cols_to_drop.append('velocity_magnitude')
    if pressure_drop:
        cols_to_drop.extend([c for c in PRESSURE_INPUT_COLUMNS if c in df.columns])
    if mos_drop:
        cols_to_drop.extend([c for c in MOS_INPUT_COLUMNS if c in df.columns and c not in cols_to_drop])

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

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

"""Data model and loading functions for the TouchPreparationViewer."""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_bracket_re = re.compile(r'\[([^\[\]]+)\]')

from analysis.pipeline.output_dirs import TOUCH_PREPARE_SESSIONS
from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertices,
    resolve_forearm_ply,
)
from analysis.touch_analytics.pipeline_shared import session_id_from_path
from analysis.touch_analytics.preparation.grouping import group_touches
from analysis.pipeline.shared_constants import (
    NERVE_SPIKE_COL,
    CONTACT_POINTS_COL,
    NERVE_FREQ_COL,
)


SIGNAL_COLUMNS = [
    'contact_depth',
    'contact_area',
    'contact_location_x',
    'contact_location_y',
    'contact_location_z',
    NERVE_FREQ_COL,
    NERVE_SPIKE_COL,
    'trial_id',
    'single_touch_id',
]

_REQUIRED_COLUMNS = [
    'block_order_id',
    'trial_id',
    'single_touch_id',
    'gesture_type',
    'time',
    CONTACT_POINTS_COL,
    'contact_location_x',
    'contact_location_y',
    'contact_location_z',
]


@dataclass
class PreparationTouchData:
    block_order_id: str
    trial_id: int
    single_touch_id: int
    gesture_type: str
    time: np.ndarray                    # (n_frames,) seconds
    signals: dict[str, np.ndarray]      # column_name -> (n_frames,) values
    contact_location: np.ndarray        # (n_frames, 3) — x/y/z RF-centered
    frame_contact_pts: list             # list of (K_i, 3) float64 arrays, one per 1kHz frame


@dataclass
class PreparationViewerData:
    session_id: str
    forearm_vertices: np.ndarray        # (N, 3) from PLY
    block_order_ids: list[str]
    trial_ids_by_block: dict[str, list[int]]
    touches_by_block_trial: dict[tuple, list[PreparationTouchData]]


def load_preparation_viewer_data(
    prepared_csv: Path,
    forearm_ply: Path,
) -> PreparationViewerData:
    """Load a prepared CSV and forearm PLY into a PreparationViewerData.

    Raises ValueError on missing columns, unreadable PLY, or no valid touches.
    """
    df = pd.read_csv(prepared_csv)

    missing_required = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"load_preparation_viewer_data: prepared CSV {prepared_csv} is missing "
            f"required columns: {missing_required}"
        )

    present_signal_cols = [c for c in SIGNAL_COLUMNS if c in df.columns]
    if not present_signal_cols:
        raise ValueError(
            f"load_preparation_viewer_data: prepared CSV {prepared_csv} contains "
            f"none of the expected signal columns: {SIGNAL_COLUMNS}"
        )

    # Keep only rows that belong to a real touch event
    df = df[df['single_touch_id'] > 0].copy()

    forearm_vertices = load_forearm_vertices(forearm_ply)
    if forearm_vertices is None:
        raise ValueError(
            f"load_preparation_viewer_data: could not load forearm PLY: {forearm_ply}"
        )

    groups = group_touches(df)

    touches_by_block_trial: dict[tuple, list[PreparationTouchData]] = {}
    trial_ids_by_block: dict[str, list[int]] = {}

    for (block_order_id, trial_id, single_touch_id), group in groups:
        if single_touch_id == 0:
            continue

        gesture_type = str(group['gesture_type'].iloc[0])
        time = group['time'].to_numpy(dtype=np.float64)
        signals = {
            col: group[col].to_numpy(dtype=np.float64)
            for col in present_signal_cols
        }
        contact_location = group[
            ['contact_location_x', 'contact_location_y', 'contact_location_z']
        ].to_numpy(dtype=np.float64)

        cp_strings = group[CONTACT_POINTS_COL].ffill().fillna('[]').values
        frame_contact_pts: list[np.ndarray] = []
        for s in cp_strings:
            matches = _bracket_re.findall(str(s))
            pts: list[list[float]] = []
            for m in matches:
                parts = m.split()
                if len(parts) == 3:
                    try:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        pass
            frame_contact_pts.append(
                np.array(pts, dtype=np.float64) if pts else np.empty((0, 3), dtype=np.float64)
            )

        touch = PreparationTouchData(
            block_order_id=str(block_order_id),
            trial_id=int(trial_id),
            single_touch_id=int(single_touch_id),
            gesture_type=gesture_type,
            time=time,
            signals=signals,
            contact_location=contact_location,
            frame_contact_pts=frame_contact_pts,
        )

        key = (str(block_order_id), int(trial_id))
        touches_by_block_trial.setdefault(key, []).append(touch)

        if str(block_order_id) not in trial_ids_by_block:
            trial_ids_by_block[str(block_order_id)] = []
        if int(trial_id) not in trial_ids_by_block[str(block_order_id)]:
            trial_ids_by_block[str(block_order_id)].append(int(trial_id))

    if not touches_by_block_trial:
        raise ValueError(
            f"load_preparation_viewer_data: no valid touches found in {prepared_csv}"
        )

    # Sort touches within each (block, trial) by single_touch_id
    for key in touches_by_block_trial:
        touches_by_block_trial[key].sort(key=lambda t: t.single_touch_id)

    # Sort trial lists
    for block_id in trial_ids_by_block:
        trial_ids_by_block[block_id].sort()

    block_order_ids = sorted(trial_ids_by_block.keys(), key=lambda b: int(b))

    session_id = session_id_from_path(prepared_csv)

    return PreparationViewerData(
        session_id=session_id,
        forearm_vertices=forearm_vertices,
        block_order_ids=block_order_ids,
        trial_ids_by_block=trial_ids_by_block,
        touches_by_block_trial=touches_by_block_trial,
    )


def resolve_preparation_paths(
    input_items: list[tuple[Path, Path]],
) -> list[tuple[Path, Path, str]]:
    """Resolve prepared CSV and forearm PLY paths for each (csv_path, database_path) pair.

    Raises ValueError if a prepared CSV or forearm PLY does not exist.
    """
    resolved = []
    for csv_path, database_path in input_items:
        session_id = session_id_from_path(csv_path)

        prepared_csv = database_path / '4_analysed' / TOUCH_PREPARE_SESSIONS / f'{session_id}_prepared.csv'
        if not prepared_csv.exists():
            raise ValueError(
                f"resolve_preparation_paths: prepared CSV does not exist: {prepared_csv}"
            )

        forearm_ply = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply is None:
            raise ValueError(
                f"resolve_preparation_paths: forearm PLY not found for session "
                f"'{session_id}' in {csv_path.parent}"
            )

        resolved.append((prepared_csv, forearm_ply, session_id))

    return resolved


def launch_preparation_viewer(input_items: list[tuple[Path, Path]]) -> None:
    """Resolve paths, load data, and open the TouchPreparationViewer."""
    resolved = resolve_preparation_paths(input_items)

    def _load(args: tuple[Path, Path, str]) -> tuple[str, PreparationViewerData]:
        prepared_csv, forearm_ply, session_id = args
        data = load_preparation_viewer_data(prepared_csv, forearm_ply)
        return session_id, data

    if len(resolved) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(_load, resolved))
    else:
        results = [_load(resolved[0])]

    sessions = [(session_id, data) for session_id, data in results]

    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    from .touch_preparation_viewer import TouchPreparationViewer

    viewer = TouchPreparationViewer(sessions[0][1], sessions=sessions)
    viewer.show()
    app.exec_()

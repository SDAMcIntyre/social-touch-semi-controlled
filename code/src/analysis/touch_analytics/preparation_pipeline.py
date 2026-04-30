# preparation_pipeline.py
"""
Stage 1: preparation pipeline.

Loads raw session CSVs, synthesises the block-ID column, and fills NaN gaps
in touch columns via cubic (or configurable) interpolation. Saves the cleaned
DataFrame as ``<session_id>_prepared.csv``.

Output layout
-------------
<output_dir>/
  <session_id>_prepared.csv
  gesture_type_summary.csv
"""

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from utils.should_process_task import should_process_task, clean_task_outputs
from .pipeline_shared import _TqdmLineWrapper, session_id_from_path
from .preparation.loader import load_session_csv
from .preparation.block_id import ensure_block_id_column
from .preparation.interpolation import interpolate_touch_columns
from .preparation.gesture_type import assign_gesture_type


_DROP_COLUMNS = ['frame_index', 'green_levels', 'time_nerve', 'time_kinect', 'trial_on']
_GESTURE_TYPES = ('tap', 'stroke_proximal', 'stroke_distal')


def _count_gesture_types(df: pd.DataFrame) -> dict[str, int]:
    touch_rows = df.drop_duplicates(subset=['block_order_id', 'trial_id', 'single_touch_id'])
    counts = touch_rows['gesture_type'].value_counts()
    return {g: int(counts.get(g, 0)) for g in _GESTURE_TYPES}


def _write_gesture_type_summary(
    output_dir: Path,
    session_counts: dict[str, dict[str, int]],
) -> None:
    rows = [{'session_id': sid, **counts} for sid, counts in session_counts.items()]
    summary_df = pd.DataFrame(rows, columns=['session_id', *_GESTURE_TYPES])
    summary_path = output_dir / 'gesture_type_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  [preparation] gesture type summary → {summary_path.name}", flush=True)


def _load_cached_counts(output_dir: Path) -> dict[str, dict[str, int]]:
    summary_path = output_dir / 'gesture_type_summary.csv'
    if not summary_path.exists():
        return {}
    try:
        df = pd.read_csv(summary_path)
        return {
            row['session_id']: {g: int(row[g]) for g in _GESTURE_TYPES}
            for _, row in df.iterrows()
        }
    except Exception:
        return {}


def run_preparation(
    input_items: List[Tuple[Path, Path]],
    preparation_cfg: dict,
    output_dir: Path,
    force: bool = False,
) -> List[Path]:
    """
    Stage 1 driver: load, synthesise block IDs, interpolate NaN gaps, and save cleaned CSVs.

    Parameters
    ----------
    input_items
        List of (aggregated_session_csv, database_root_path) tuples.
    preparation_cfg
        Preparation config dict, e.g. ``{'interpolation': {'method': 'cubic'}}``.
    output_dir
        Directory where prepared CSVs are written.
    force
        Override idempotency checks.

    Returns
    -------
    List of paths to written prepared CSVs.
    """
    preparation_cfg = preparation_cfg or {}
    interp_method = preparation_cfg.get('interpolation', {}).get('method', 'cubic')

    output_dir.mkdir(parents=True, exist_ok=True)
    cached_counts = _load_cached_counts(output_dir)

    print(
        f"=== preparation pipeline: {len(input_items)} sessions, "
        f"interpolation.method={interp_method} ===",
        flush=True,
    )

    written: List[Path] = []
    session_counts: dict[str, dict[str, int]] = {}

    with tqdm(total=len(input_items), desc="preparation", unit="session",
              file=_TqdmLineWrapper(sys.stdout)) as progress:
        for input_file, _ in input_items:
            session_id = session_id_from_path(input_file)
            result = _prepare_session(
                input_file=input_file,
                output_dir=output_dir,
                interp_method=interp_method,
                force=force,
                cached_counts=cached_counts,
            )
            if result is not None:
                output_path, counts = result
                written.append(output_path)
                session_counts[session_id] = counts
            progress.set_postfix_str(f"preparation: {session_id}", refresh=False)
            progress.update(1)

    if session_counts:
        _write_gesture_type_summary(output_dir, session_counts)

    print(f"=== preparation pipeline complete: {len(written)} outputs ===", flush=True)
    return written


def _prepare_session(
    input_file: Path,
    output_dir: Path,
    interp_method: str,
    force: bool,
    cached_counts: dict[str, dict[str, int]],
) -> tuple[Path, dict[str, int]] | None:
    session_id = session_id_from_path(input_file)
    output_path = output_dir / f"{session_id}_prepared.csv"

    if not force and output_path.exists():
        try:
            if not should_process_task(
                input_paths=[input_file],
                output_paths=[output_path],
                force=False,
            ):
                counts = cached_counts.get(session_id)
                if counts is None:
                    counts = _count_gesture_types(pd.read_csv(output_path))
                print(
                    f"  [preparation] {session_id} — up to date  "
                    f"tap={counts['tap']}  "
                    f"stroke_proximal={counts['stroke_proximal']}  "
                    f"stroke_distal={counts['stroke_distal']}",
                    flush=True,
                )
                return output_path, counts
        except FileNotFoundError:
            pass
    clean_task_outputs(output_path)

    df = load_session_csv(input_file)
    df = ensure_block_id_column(df)
    df = interpolate_touch_columns(df, method=interp_method)
    df = assign_gesture_type(df)
    df = df[df['single_touch_id'] != 0]
    df = df.drop(columns=_DROP_COLUMNS, errors='ignore')
    df.to_csv(output_path, index=False)

    counts = _count_gesture_types(df)
    print(
        f"  [preparation] {session_id} — {len(df)} rows  "
        f"tap={counts['tap']}  "
        f"stroke_proximal={counts['stroke_proximal']}  "
        f"stroke_distal={counts['stroke_distal']}  "
        f"→ {output_path.name}",
        flush=True,
    )
    return output_path, counts

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
"""

import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from utils.should_process_task import should_process_task, clean_task_outputs
from .pipeline_shared import _TqdmLineWrapper, session_id_from_path
from .preparation.loader import load_session_csv
from .preparation.block_id import ensure_block_id_column
from .preparation.interpolation import interpolate_touch_columns


_DROP_COLUMNS = ['frame_index', 'green_levels', 'time_nerve', 'time_kinect', 'trial_on']


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

    print(
        f"=== preparation pipeline: {len(input_items)} sessions, "
        f"interpolation.method={interp_method} ===",
        flush=True,
    )

    written: List[Path] = []

    with tqdm(total=len(input_items), desc="preparation", unit="session",
              file=_TqdmLineWrapper(sys.stdout)) as progress:
        for input_file, _ in input_items:
            result = _prepare_session(
                input_file=input_file,
                output_dir=output_dir,
                interp_method=interp_method,
                force=force,
            )
            if result is not None:
                written.append(result)
            session_id = session_id_from_path(input_file)
            progress.set_postfix_str(f"preparation: {session_id}", refresh=False)
            progress.update(1)

    print(f"=== preparation pipeline complete: {len(written)} outputs ===", flush=True)
    return written


def _prepare_session(
    input_file: Path,
    output_dir: Path,
    interp_method: str,
    force: bool,
) -> Path | None:
    session_id = session_id_from_path(input_file)
    output_path = output_dir / f"{session_id}_prepared.csv"

    if not force and output_path.exists():
        try:
            if not should_process_task(
                input_paths=[input_file],
                output_paths=[output_path],
                force=False,
            ):
                print(f"  [preparation] {session_id} — up to date", flush=True)
                return output_path
        except FileNotFoundError:
            pass
    clean_task_outputs(output_path)

    df = load_session_csv(input_file)
    df = ensure_block_id_column(df)
    df = interpolate_touch_columns(df, method=interp_method)
    df = df[df['single_touch_id'] != 0]
    df = df.drop(columns=_DROP_COLUMNS, errors='ignore')
    df.to_csv(output_path, index=False)

    print(f"  [preparation] {session_id} — {len(df)} rows → {output_path.name}", flush=True)
    return output_path

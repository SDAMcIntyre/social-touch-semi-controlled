# preparation_pipeline.py
"""
Stage 1: touch data preparation pipeline.

Loads raw session CSVs, fills NaN gaps in touch columns via cubic/linear
interpolation per touch group, and saves prepared CSVs.

Output layout
-------------
<output_dir>/
  <session_id>_prepared.csv
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from utils.should_process_task import should_process_task, clean_task_outputs
from .pipeline_shared import _TqdmLineWrapper, session_id_from_path
from .preparation.loader import load_session_csv
from .preparation.block_id import ensure_block_id_column
from .preparation.interpolation import interpolate_touch_columns


def run_preparation(
    input_items: List[Tuple[Path, Path]],
    config: dict,
    output_dir: Path,
    force: bool = False,
) -> List[Path]:
    """
    Interpolate NaN gaps in touch columns for each session.

    Parameters
    ----------
    input_items
        List of (raw_session_csv, database_root_path) tuples.
    config
        Preparation config dict. Recognised keys:
        - ``interpolation_method`` (str, default ``'cubic'``)
    output_dir
        Directory where prepared CSVs are written.
    force
        Override idempotency checks.

    Returns
    -------
    List of paths to written prepared CSVs.
    """
    config = config or {}
    method = config.get('interpolation_method', 'cubic')

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== preparation pipeline: {len(input_items)} sessions ===", flush=True)

    written: List[Path] = []

    with tqdm(total=len(input_items), desc="preparation", unit="session",
              file=_TqdmLineWrapper(sys.stdout)) as progress:
        for input_file, _ in input_items:
            result = _prepare_session(
                input_file=input_file,
                output_dir=output_dir,
                method=method,
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
    method: str,
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

    try:
        df = load_session_csv(input_file)
    except Exception as exc:
        logging.error(f"preparation_pipeline: failed to load {input_file}: {exc}")
        return None

    df = ensure_block_id_column(df)
    df = interpolate_touch_columns(df, method=method)

    try:
        df.to_csv(output_path, index=False)
        print(
            f"  [preparation] {session_id} — {len(df)} rows → {output_path.name}",
            flush=True,
        )
    except Exception as exc:
        logging.error(f"preparation_pipeline: failed to save {output_path}: {exc}")
        return None

    return output_path

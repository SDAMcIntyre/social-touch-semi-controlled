# session_summary.py
import logging
import re
import pandas as pd
from pathlib import Path
from typing import List

from utils.should_process_task import should_process_task

_BLOCK_ORDER_RE = re.compile(r'_block-order-(\d+)_')


def generate_session_summary(
        input_paths: List[Path],
        output_path: Path,
        force: bool = False,
) -> Path:
    """
    Generate a block-level summary CSV from aggregated session CSVs.

    For each session file, extracts block IDs (via ``source_block_file``) and
    trial IDs (``trial_id`` > 0), then writes one row per block containing:

    - ``session_id``   — derived from the filename prefix before ``_semicontrolled_``
    - ``block_id``     — block order number extracted from ``source_block_file``
    - ``num_trials``   — count of distinct non-zero trial IDs in the block
    - ``trial_ids``    — comma-separated sorted list of those trial IDs

    Includes internal idempotency check via ``should_process_task``.
    """
    if not should_process_task(
        input_paths=input_paths,
        output_paths=[output_path],
        force=force,
    ):
        logging.info(f"Skipping Session Block Summary (up-to-date): {output_path.name}")
        return output_path

    logging.info(f"Generating session block summary from {len(input_paths)} file(s)...")

    rows = []

    for csv_path in input_paths:
        session_id = _extract_session_id(csv_path)

        try:
            df = _load_relevant_columns(csv_path)
        except Exception as exc:
            logging.warning(f"Skipping {csv_path.name}: failed to read — {exc}")
            continue

        if 'source_block_file' not in df.columns:
            logging.warning(f"{csv_path.name}: 'source_block_file' column missing — using 'unknown' block ID.")
            df['_block_id'] = 'unknown'
        else:
            df['_block_id'] = (
                df['source_block_file']
                .astype(str)
                .str.extract(_BLOCK_ORDER_RE, expand=False)
                .fillna('unknown')
            )

        if 'trial_id' not in df.columns:
            logging.warning(f"{csv_path.name}: 'trial_id' column missing — skipping file.")
            continue

        for block_id, block_df in df.groupby('_block_id', sort=True):
            valid_trials = sorted(
                block_df.loc[block_df['trial_id'] > 0, 'trial_id'].unique().tolist()
            )
            rows.append({
                'session_id': session_id,
                'block_id': block_id,
                'num_trials': len(valid_trials),
                'trial_ids': ','.join(str(t) for t in valid_trials),
            })

    summary_df = pd.DataFrame(rows, columns=['session_id', 'block_id', 'num_trials', 'trial_ids'])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary_df.to_csv(output_path, index=False)
        logging.info(f"Saved session block summary to {output_path} ({len(summary_df)} rows)")
    except Exception as exc:
        logging.error(f"Failed to save session block summary: {exc}")
        raise

    return output_path


def _extract_session_id(csv_path: Path) -> str:
    """Return the session ID prefix from the filename, or the stem if the pattern is absent."""
    name = csv_path.name
    if '_semicontrolled_' in name:
        return name.split('_semicontrolled_')[0]
    return csv_path.stem


def _load_relevant_columns(csv_path: Path) -> pd.DataFrame:
    """Load only the columns needed for the summary (memory-efficient)."""
    needed = {'source_block_file', 'trial_id'}
    # Read header first to check which columns exist
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in header if c in needed]
    return pd.read_csv(csv_path, usecols=usecols)

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

from utils.should_process_task import should_process_task, clean_task_outputs

logger = logging.getLogger(__name__)


def extract_unit_and_block_order(filename: str) -> Tuple[str, int]:
    """
    Extracts unit ID and block order from a merged CSV filename.

    Expected patterns: '(ST\\d+-\\d+)' for unit, 'block-order-(\\d+)' for block order.
    Raises ValueError if either pattern is not found.
    """
    unit_match = re.search(r'(ST\d+-\d+)', filename)
    block_match = re.search(r'block-order-(\d+)', filename)

    if not unit_match:
        raise ValueError(f"Cannot extract unit ID from filename: {filename}")
    if not block_match:
        raise ValueError(f"Cannot extract block order from filename: {filename}")

    unit = unit_match.group(1)
    block_order = int(block_match.group(1))  # int() strips leading zeros
    return unit, block_order


def parse_neural_quality_xlsx(xlsx_path: Path) -> Dict[Tuple[str, int], Set[int]]:
    """
    Parses an experimenter-maintained xlsx file to identify Not2Use trial IDs per unit/block.

    Expected xlsx columns: 'unit', 'block_order', '1', '2', ..., '12'.
    Trial columns '1'-'12' contain "Not2Use" (or a substring) for unusable trials.

    Returns:
        Dict mapping (unit, block_order) -> set of Not2Use trial IDs (int).
    """
    df = pd.read_excel(xlsx_path)
    unit_label = 'Unit'
    block_label = 'Block order'

    # Drop rows where key columns are empty (trailing blank rows, formatting
    # artifacts, or partial entries common in hand-maintained Excel files).
    rows_before_clean = len(df)
    df = df.dropna(subset=[unit_label, block_label])
    dropped = rows_before_clean - len(df)
    if dropped:
        logger.debug(f"Dropped {dropped} rows with empty Unit/Block order from {xlsx_path.name}")

    trial_columns = [i for i in range(1, 13)]
    missing = [c for c in [unit_label, block_label] + trial_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"xlsx is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    not2use_map: Dict[Tuple[str, int], Set[int]] = {}

    for _, row in df.iterrows():
        unit = str(row[unit_label]).strip()
        try:
            block_order = int(row[block_label])
        except (ValueError, TypeError):
            logger.warning(f"Skipping row with invalid block_order: {row[block_label]!r}")
            continue

        bad_trials: Set[int] = set()
        for col in trial_columns:
            cell = row[col]
            if isinstance(cell, str) and 'Not2Use' in cell:
                bad_trials.add(int(col))

        if bad_trials:
            not2use_map[(unit, block_order)] = bad_trials

    return not2use_map


def filter_block_by_neural_quality(
    input_csv: Path,
    output_csv: Path,
    xlsx_path: Path,
    *,
    force_processing: bool = False,
    discard_from_first_not2use: bool = True,
) -> Path:
    """
    Filters a single block's merged CSV by removing rows belonging to Not2Use trials.

    Parses the xlsx annotation file, extracts unit and block_order from input_csv's
    filename, and looks up which trial IDs are marked Not2Use for that block.

    NaN trial_id rows (nerve-rate interpolation) are forward-filled to assign them to
    their preceding trial. Rows with trial_id == 0 (inter-trial gaps) are always kept.
    If no Not2Use trials exist for this block the file is copied as-is.

    Args:
        discard_from_first_not2use: When True (default), truncate the block at the
            first row of min(not2use_trials), discarding all subsequent data regardless
            of trial distribution. When False, use the original two-path logic: truncate
            only if Not2Use trials form a contiguous trailing suffix, otherwise remove
            only the individual Not2Use trial rows.

    Returns the output path.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    unit, block_order = extract_unit_and_block_order(input_csv.name)
    not2use_map = parse_neural_quality_xlsx(xlsx_path)
    not2use_trials: Set[int] = not2use_map.get((unit, block_order), set())

    if not not2use_trials:
        if not output_csv.exists() or force_processing:
            shutil.copy2(input_csv, output_csv)
            logger.info(f"Copied (no Not2Use trials): {output_csv.name}")
        else:
            logger.info(f"Already up-to-date (no Not2Use trials): {output_csv.name}")
        return output_csv

    if not should_process_task(
        input_paths=input_csv,
        output_paths=output_csv,
        force=force_processing,
    ):
        logger.info(f"Already up-to-date: {output_csv.name}")
        return output_csv
    clean_task_outputs(output_csv)
    df = pd.read_csv(input_csv)
    if 'trial_id' not in df.columns:
        raise KeyError(f"'trial_id' column missing in {input_csv.name}")

    # Forward-fill to assign NaN nerve-rate rows to their preceding trial.
    # Rows before the first Kinect frame have no trial yet — fill with 0 (kept).
    filled_trial_id = df['trial_id'].ffill().fillna(0).astype(int)

    rows_before = len(df)
    min_not2use = min(not2use_trials)

    if discard_from_first_not2use:
        # Truncate at the first row whose filled trial_id reaches min_not2use,
        # discarding everything from that point onward regardless of trial distribution.
        cutoff_mask = filled_trial_id >= min_not2use
        if cutoff_mask.any():
            cutoff = int(cutoff_mask.values.argmax())
            df_filtered = df.iloc[:cutoff]
        else:
            df_filtered = df
        mode_label = f"truncated from trial {min_not2use} onward"
    else:
        is_trailing_suffix = not2use_trials == set(range(min_not2use, 13))
        if is_trailing_suffix:
            # All remaining trials from min_not2use to 12 are Not2Use: truncate at the
            # first row whose filled trial_id reaches that threshold.
            suffix_mask = filled_trial_id >= min_not2use
            if suffix_mask.any():
                cutoff = int(suffix_mask.values.argmax())
                df_filtered = df.iloc[:cutoff]
            else:
                df_filtered = df
            mode_label = f"trailing-suffix truncated from trial {min_not2use} onward"
        else:
            mask = ~filled_trial_id.isin(not2use_trials)
            df_filtered = df[mask]
            mode_label = f"removed trials {not2use_trials} individually"

    rows_after = len(df_filtered)

    removed = rows_before - rows_after
    logger.info(
        f"{input_csv.name}: {mode_label} "
        f"({not2use_trials} Not2Use) -> removed {removed} rows, {rows_after} remaining"
    )

    unmatched = not2use_trials - set(filled_trial_id.unique())
    if unmatched:
        logger.warning(
            f"Some Not2Use trial IDs not found in data and had no effect: {unmatched}"
        )

    df_filtered.to_csv(output_csv, index=False)
    return output_csv

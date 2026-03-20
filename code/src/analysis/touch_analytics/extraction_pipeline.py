# extraction_pipeline.py
"""
Standalone feature extraction pipeline.

Replaces the extraction logic formerly in unified_pipeline.py.
Runs per-session extraction for all configured profiles and writes
one CSV per (session, profile) pair.

Output layout
-------------
<output_dir>/
  <profile_name>/
    <session>_touch_summary.csv
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from utils.should_process_task import should_process_task, clean_task_outputs
from .feature_extraction import get_extractor
from .pipeline_shared import SHARED_COLUMNS, _TqdmLineWrapper, filter_enabled_profiles, session_id_from_path


def run_feature_extraction(
    input_items: List[Tuple[Path, Path]],
    extraction_profiles: dict,
    output_dir: Path,
    force: bool = False,
) -> dict[str, list[Path]]:
    """
    Run per-session feature extraction for all configured profiles.

    Parameters
    ----------
    input_items
        List of (aggregated_session_csv, database_root_path) tuples.
    extraction_profiles
        Dict mapping profile_name -> profile_config. Profiles with
        ``enabled: false`` are filtered out before calling this function
        (or can be filtered here via the caller).
    output_dir
        Root directory for extraction outputs
        (e.g. ``database / '4_analysed' / 'unified_touches'``).
    force
        Override idempotency checks.

    Returns
    -------
    dict mapping profile_name -> list of session CSV paths written.
    """
    extraction_profiles = filter_enabled_profiles(extraction_profiles)

    per_profile_session_csvs: dict[str, list[Path]] = {p: [] for p in extraction_profiles}

    n_steps = len(input_items) * len(extraction_profiles)

    print(
        f"=== extraction pipeline: {len(input_items)} sessions, "
        f"{len(extraction_profiles)} extractors ===",
        flush=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with tqdm(total=n_steps, desc="extraction", unit="step",
                  file=_TqdmLineWrapper(sys.stdout)) as progress:

            for input_file, _ in input_items:
                session_outputs = _extract_session(
                    input_file=input_file,
                    output_dir=output_dir,
                    extraction_profiles=extraction_profiles,
                    force=force,
                    progress=progress,
                )
                for profile_name, csv_path in session_outputs.items():
                    per_profile_session_csvs[profile_name].append(csv_path)

    total = sum(len(v) for v in per_profile_session_csvs.values())
    print(f"=== extraction pipeline complete: {total} outputs ===", flush=True)
    return per_profile_session_csvs


def _extract_session(
    input_file: Path,
    output_dir: Path,
    extraction_profiles: dict,
    force: bool,
    progress: tqdm = None,
) -> dict[str, Path]:
    """
    Run all enabled extraction profiles on one session CSV.

    Returns
    -------
    dict mapping profile_name -> output CSV path.
    """
    results: dict[str, Path] = {}
    session_id = session_id_from_path(input_file)

    try:
        df = pd.read_csv(input_file)
    except Exception as exc:
        logging.error(f"Failed to load {input_file}: {exc}")
        if progress is not None:
            progress.update(len(extraction_profiles))
        return results

    # Shared preprocessing
    if 'source_block_file' in df.columns:
        df['block_order_id'] = (
            df['source_block_file'].astype(str)
            .str.extract(r'_block-order-(\d+)_', expand=False)
        )
    else:
        df['block_order_id'] = None

    has_nerve_data = 'Nerve_spike' in df.columns
    if not has_nerve_data:
        logging.warning(
            f"'Nerve_spike' column missing in {input_file.name}. "
            "'spike_elicited' will be 0."
        )

    filename = input_file.name
    if '_semicontrolled_' in filename:
        prefix = filename.split('_semicontrolled_')[0]
        csv_stem = f'{prefix}_semicontrolled_touch_summary.csv'
    else:
        csv_stem = f'{input_file.stem}_touch_summary.csv'

    for profile_name, profile_config in extraction_profiles.items():
        method = profile_config.get('method', profile_name)

        profile_dir = output_dir / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        output_path = profile_dir / csv_stem

        # Idempotency
        if not force and output_path.exists():
            try:
                if not should_process_task(
                    input_paths=[input_file],
                    output_paths=[output_path],
                    force=False,
                ):
                    results[profile_name] = output_path
                    if progress is not None:
                        progress.set_postfix_str(f"extract: {session_id}/{profile_name}", refresh=False)
                        progress.update(1)
                    print(f"  [extract] {session_id} / {profile_name} — up to date", flush=True)
                    continue
            except FileNotFoundError:
                pass
        clean_task_outputs(output_path)
        try:
            extractor = get_extractor(method)
        except KeyError as exc:
            logging.error(f"Profile '{profile_name}': {exc}")
            if progress is not None:
                progress.set_postfix_str(f"extract: {session_id}/{profile_name}", refresh=False)
                progress.update(1)
            print(f"  [extract] {session_id} / {profile_name} — error: unknown extractor", flush=True)
            continue

        rows = _extract_all_touches(
            df, extractor, profile_config, has_nerve_data,
            desc=f"{session_id}/{profile_name}",
        )
        summary_df = pd.DataFrame(rows)
        summary_df['session_id'] = session_id

        try:
            summary_df.to_csv(output_path, index=False)
            logging.info(
                f"[{profile_name}] Saved {len(summary_df)} touches → {output_path}"
            )
        except Exception as exc:
            logging.error(f"Failed to save {output_path}: {exc}")
            if progress is not None:
                progress.set_postfix_str(f"extract: {session_id}/{profile_name}", refresh=False)
                progress.update(1)
            print(f"  [extract] {session_id} / {profile_name} — error: save failed", flush=True)
            continue

        results[profile_name] = output_path
        if progress is not None:
            progress.set_postfix_str(f"extract: {session_id}/{profile_name}", refresh=False)
            progress.update(1)
        print(f"  [extract] {session_id} / {profile_name} — {len(summary_df)} touches", flush=True)

    return results


def _extract_all_touches(
    df: pd.DataFrame,
    extractor,
    profile_config: dict,
    has_nerve_data: bool,
    desc: str = "",
) -> list[dict]:
    rows = []
    groups = list(df.groupby(['block_order_id', 'trial_id', 'single_touch_id']))
    for (block_order_id, trial_id, touch_id), group in tqdm(
        groups, desc=desc, leave=False, unit="touch", disable=True
    ):
        if group.empty or touch_id == 0:
            continue

        touch_type = (
            group['type_metadata'].iloc[0]
            if 'type_metadata' in group.columns else 'unknown'
        )

        direction = 'static'
        if touch_type == 'stroke':
            start_y = group['sticker_blue_position_y'].iloc[0]
            end_y = group['sticker_blue_position_y'].iloc[-1]
            direction = 'proximal' if end_y > start_y else 'distal'

        mean_contact_x = (
            group['contact_location_x'].mean()
            if 'contact_location_x' in group.columns else None
        )
        mean_contact_y = (
            group['contact_location_y'].mean()
            if 'contact_location_y' in group.columns else None
        )
        mean_contact_z = (
            group['contact_location_z'].mean()
            if 'contact_location_z' in group.columns else None
        )

        spike_elicited = (
            1 if has_nerve_data and group['Nerve_spike'].max() == 1 else 0
        )

        shared = {
            'block_order_id': block_order_id,
            'trial_id': trial_id,
            'single_touch_id': touch_id,
            'type_metadata': touch_type,
            'direction': direction,
            'mean_contact_x': mean_contact_x,
            'mean_contact_y': mean_contact_y,
            'mean_contact_z': mean_contact_z,
            'spike_elicited': spike_elicited,
        }

        try:
            features = extractor.extract(group, profile_config)
        except Exception as exc:
            logging.warning(
                f"Extractor failed for trial={trial_id} touch={touch_id}: {exc}"
            )
            features = {}

        rows.append({**shared, **features})
    return rows

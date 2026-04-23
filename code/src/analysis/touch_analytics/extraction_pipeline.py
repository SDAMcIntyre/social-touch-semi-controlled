# extraction_pipeline.py
"""
Standalone feature extraction pipeline.

Replaces the extraction logic formerly in unified_pipeline.py.
Runs per-session extraction for all configured features and writes
one CSV per (session, feature) pair under its own folder.

Output layout
-------------
<output_dir>/
  <feature_name>/
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
from .feature_extraction import get_feature_extractor, AGGREGATION_NAMES
from .pipeline_shared import SHARED_COLUMNS, _TqdmLineWrapper, filter_enabled_profiles, session_id_from_path
from .preparation.direction import infer_direction


def _translate_extraction_profiles(extraction_profiles: dict) -> dict:
    """
    Translate old ``extraction_profiles`` format to new ``features`` format.

    Old format (example)::

        extraction_profiles:
          max:
            method: max
          stats:
            method: statistical
            aggregations: [mean, std]
          mos:
            method: mechanics_of_solids
            youngs_modulus_kpa: 100.0

    New format::

        features:
          max:
            enabled: true
          mean:
            enabled: true
          std:
            enabled: true
          mechanics_of_solids:
            enabled: true
            youngs_modulus_kpa: 100.0
    """
    features: dict = {}
    for profile_name, profile_config in extraction_profiles.items():
        enabled = profile_config.get('enabled', True)
        method = profile_config.get('method', profile_name)
        if method == 'max':
            features['max'] = {'enabled': enabled}
        elif method == 'statistical':
            aggregations = profile_config.get('aggregations', ['mean', 'std'])
            for agg in aggregations:
                features[agg] = {'enabled': enabled}
        elif method == 'temporal':
            features['temporal'] = {'enabled': enabled}
        elif method == 'mechanics_of_solids':
            cfg = {k: v for k, v in profile_config.items() if k not in ('method', 'enabled')}
            cfg['enabled'] = enabled
            features['mechanics_of_solids'] = cfg
        else:
            # Unknown method — pass through using the profile name as feature name
            features[method] = profile_config
    return features


def run_feature_extraction(
    input_items: List[Tuple[Path, Path]],
    features: dict,
    output_dir: Path,
    force: bool = False,
) -> dict[str, list[Path]]:
    """
    Run per-session feature extraction for all configured features.

    Each enabled feature extracts to its own subfolder under *output_dir*:
    ``output_dir / feature_name / <session>_touch_summary.csv``.

    Parameters
    ----------
    input_items
        List of (aggregated_session_csv, database_root_path) tuples.
    features
        Dict mapping feature_name -> feature_config. Supports both the new
        ``features`` format and the old ``extraction_profiles`` format
        (backward-compat translation is applied automatically).
    output_dir
        Root directory for extraction outputs
        (e.g. ``database / '4_analysed' / 'unified_touches'``).
    force
        Override idempotency checks.

    Returns
    -------
    dict mapping feature_name -> list of session CSV paths written.
    """
    # Backward-compat: translate old extraction_profiles format if needed.
    # Old format uses a 'method' key inside each entry; new format does not.
    if features and any('method' in v for v in features.values() if isinstance(v, dict)):
        logging.warning(
            "extraction_pipeline: 'extraction_profiles' format detected — "
            "translating to new 'features' format automatically."
        )
        features = _translate_extraction_profiles(features)

    features = filter_enabled_profiles(features)

    per_feature_session_csvs: dict[str, list[Path]] = {f: [] for f in features}

    n_steps = len(input_items) * len(features)

    print(
        f"=== extraction pipeline: {len(input_items)} sessions, "
        f"{len(features)} features ===",
        flush=True,
    )

    if not features:
        logging.warning("extraction_pipeline: no features enabled — producing no output.")
        return per_feature_session_csvs

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with tqdm(total=n_steps, desc="extraction", unit="step",
                  file=_TqdmLineWrapper(sys.stdout)) as progress:

            for input_file, _ in input_items:
                session_outputs = _extract_session(
                    input_file=input_file,
                    output_dir=output_dir,
                    features=features,
                    force=force,
                    progress=progress,
                )
                for feature_name, csv_path in session_outputs.items():
                    per_feature_session_csvs[feature_name].append(csv_path)

    total = sum(len(v) for v in per_feature_session_csvs.values())
    print(f"=== extraction pipeline complete: {total} outputs ===", flush=True)
    return per_feature_session_csvs


def _extract_session(
    input_file: Path,
    output_dir: Path,
    features: dict,
    force: bool,
    progress: tqdm = None,
) -> dict[str, Path]:
    """
    Run all enabled features on one session CSV.

    Returns
    -------
    dict mapping feature_name -> output CSV path.
    """
    results: dict[str, Path] = {}
    session_id = session_id_from_path(input_file)

    try:
        df = pd.read_csv(input_file)
    except Exception as exc:
        logging.error(f"Failed to load {input_file}: {exc}")
        if progress is not None:
            progress.update(len(features))
        return results

    # Shared preprocessing
    if 'block_order_id' not in df.columns:
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

    for feature_name, feature_config in features.items():
        feature_dir = output_dir / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        output_path = feature_dir / csv_stem

        # Idempotency
        if not force and output_path.exists():
            try:
                if not should_process_task(
                    input_paths=[input_file],
                    output_paths=[output_path],
                    force=False,
                ):
                    results[feature_name] = output_path
                    if progress is not None:
                        progress.set_postfix_str(f"extract: {session_id}/{feature_name}", refresh=False)
                        progress.update(1)
                    print(f"  [extract] {session_id} / {feature_name} — up to date", flush=True)
                    continue
            except FileNotFoundError:
                pass
        clean_task_outputs(output_path)

        try:
            extractor = get_feature_extractor(feature_name, feature_config)
        except KeyError as exc:
            logging.error(f"Feature '{feature_name}': {exc}")
            if progress is not None:
                progress.set_postfix_str(f"extract: {session_id}/{feature_name}", refresh=False)
                progress.update(1)
            print(f"  [extract] {session_id} / {feature_name} — error: unknown feature", flush=True)
            continue

        # For aggregation features, inject the aggregation name into the config
        # so that StatisticalExtractor produces the correct columns.
        if feature_name in AGGREGATION_NAMES:
            extract_config = {**feature_config, 'aggregations': [feature_name]}
        else:
            extract_config = feature_config

        rows = _extract_all_touches(
            df, extractor, extract_config, has_nerve_data,
            desc=f"{session_id}/{feature_name}",
        )
        summary_df = pd.DataFrame(rows)
        summary_df['session_id'] = session_id

        try:
            summary_df.to_csv(output_path, index=False)
            logging.info(
                f"[{feature_name}] Saved {len(summary_df)} touches → {output_path}"
            )
        except Exception as exc:
            logging.error(f"Failed to save {output_path}: {exc}")
            if progress is not None:
                progress.set_postfix_str(f"extract: {session_id}/{feature_name}", refresh=False)
                progress.update(1)
            print(f"  [extract] {session_id} / {feature_name} — error: save failed", flush=True)
            continue

        results[feature_name] = output_path
        if progress is not None:
            progress.set_postfix_str(f"extract: {session_id}/{feature_name}", refresh=False)
            progress.update(1)
        print(f"  [extract] {session_id} / {feature_name} — {len(summary_df)} touches", flush=True)

    return results


def _extract_all_touches(
    df: pd.DataFrame,
    extractor,
    feature_config: dict,
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

        direction = infer_direction(group)

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
            features = extractor.extract(group, feature_config)
        except Exception as exc:
            logging.warning(
                f"Extractor failed for trial={trial_id} touch={touch_id}: {exc}"
            )
            features = {}

        rows.append({**shared, **features})
    return rows

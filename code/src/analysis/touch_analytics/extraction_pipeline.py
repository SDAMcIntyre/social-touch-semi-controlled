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
from .preparation.interpolation import interpolate_touch_columns
from .representation.feature_characterization.statistical import _EXCLUDE_FROM_AGGREGATION

TOUCH_KEYS = ['block_order_id', 'trial_id', 'single_touch_id']

_PANDAS_NATIVE_AGGS = frozenset({'mean', 'median', 'std', 'min', 'max'})


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
    series_dir: Path | None = None,
    preparation_dir: Path | None = None,
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
                    series_dir=series_dir,
                    preparation_dir=preparation_dir,
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
    series_dir: Path | None = None,
    preparation_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Run all enabled features on one session CSV.

    Returns
    -------
    dict mapping feature_name -> output CSV path.
    """
    results: dict[str, Path] = {}
    session_id = session_id_from_path(input_file)

    # Determine actual source file: series augmented → prepared → raw (last resort)
    source_file = input_file
    if series_dir is not None:
        candidate = series_dir / f"{session_id}_series_augmented.csv"
        if candidate.exists():
            source_file = candidate
    if source_file is input_file and preparation_dir is not None:
        candidate = preparation_dir / f"{session_id}_prepared.csv"
        if candidate.exists():
            source_file = candidate

    filename = input_file.name
    if '_semicontrolled_' in filename:
        prefix = filename.split('_semicontrolled_')[0]
        csv_stem = f'{prefix}_semicontrolled_touch_summary.csv'
    else:
        csv_stem = f'{input_file.stem}_touch_summary.csv'

    stat_features = {k: v for k, v in features.items() if k in AGGREGATION_NAMES}
    other_features = {k: v for k, v in features.items() if k not in AGGREGATION_NAMES}

    if not force:
        pre_results: dict[str, Path] = {}
        needs_work = False
        for feature_name in features:
            out = output_dir / feature_name / csv_stem
            if not out.exists():
                needs_work = True
                break
            try:
                if should_process_task(input_paths=[source_file],
                                       output_paths=[out], force=False):
                    needs_work = True
                    break
            except FileNotFoundError:
                needs_work = True
                break
            pre_results[feature_name] = out

        if not needs_work:
            for feature_name, out in pre_results.items():
                if progress is not None:
                    progress.set_postfix_str(f"extract: {session_id}/{feature_name}",
                                             refresh=False)
                    progress.update(1)
                print(f"  [extract] {session_id} / {feature_name} — up to date",
                      flush=True)
            return pre_results

    try:
        df = pd.read_csv(source_file)
    except Exception as exc:
        logging.error(f"Failed to load {source_file}: {exc}")
        if progress is not None:
            progress.update(len(features))
        return results

    if source_file is input_file:
        logging.warning(
            f"extraction_pipeline: {session_id} — no prepared or series CSV found; "
            "applying inline interpolation on raw CSV (last resort)."
        )
        df = interpolate_touch_columns(df)

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

    # --- Statistical features: single vectorised groupby pass ---
    stat_to_run: dict[str, dict] = {}
    stat_output_paths: dict[str, Path] = {}

    for feature_name, feature_config in stat_features.items():
        feature_dir = output_dir / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        output_path = feature_dir / csv_stem

        if not force and output_path.exists():
            try:
                if not should_process_task(
                    input_paths=[source_file],
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
        stat_to_run[feature_name] = feature_config
        stat_output_paths[feature_name] = output_path

    if stat_to_run:
        batch_dfs = _extract_statistical_batch(df, list(stat_to_run), has_nerve_data, session_id)
        for feature_name, summary_df in batch_dfs.items():
            output_path = stat_output_paths[feature_name]
            try:
                summary_df.to_csv(output_path, index=False)
                logging.info(f"[{feature_name}] Saved {len(summary_df)} touches → {output_path}")
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

    # --- Non-statistical features: per-touch loop (shared groupby) ---
    touch_groups: list | None = None

    for feature_name, feature_config in other_features.items():
        feature_dir = output_dir / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        output_path = feature_dir / csv_stem

        if not force and output_path.exists():
            try:
                if not should_process_task(
                    input_paths=[source_file],
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

        if touch_groups is None:
            touch_groups = list(df.groupby(TOUCH_KEYS))

        rows = _extract_all_touches(
            df, extractor, feature_config, has_nerve_data,
            desc=f"{session_id}/{feature_name}", _groups=touch_groups,
        )
        summary_df = pd.DataFrame(rows)
        summary_df['session_id'] = session_id

        try:
            summary_df.to_csv(output_path, index=False)
            logging.info(f"[{feature_name}] Saved {len(summary_df)} touches → {output_path}")
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


def _extract_statistical_batch(
    df: pd.DataFrame,
    enabled_aggregations: list[str],
    has_nerve_data: bool,
    session_id: str,
) -> dict[str, pd.DataFrame]:
    """
    Compute all enabled statistical aggregations in one vectorised groupby pass.

    Returns a dict mapping each aggregation name to a per-touch summary DataFrame
    with the same schema produced by the per-touch loop path.
    """
    numeric_cols = [
        col for col in df.columns
        if col not in _EXCLUDE_FROM_AGGREGATION
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    g = df.groupby(TOUCH_KEYS, sort=False)

    # Build the list of native pandas aggs to run in one call.
    # Include min/max as auxiliaries when 'range' is requested.
    native_needed = list(dict.fromkeys(
        [a for a in enabled_aggregations if a in _PANDAS_NATIVE_AGGS]
        + (['min'] if 'range' in enabled_aggregations and 'min' not in enabled_aggregations else [])
        + (['max'] if 'range' in enabled_aggregations and 'max' not in enabled_aggregations else [])
    ))

    if native_needed and numeric_cols:
        agg_df = g[numeric_cols].agg(native_needed)
        agg_df.columns = [f'{col}_{agg}' for col, agg in agg_df.columns]
        agg_df = agg_df.reset_index()
    else:
        agg_df = g.size().reset_index(name='_n').drop(columns=['_n'])

    if 'range' in enabled_aggregations and numeric_cols:
        for col in numeric_cols:
            agg_df[f'{col}_range'] = agg_df[f'{col}_max'] - agg_df[f'{col}_min']

    if 'skewness' in enabled_aggregations and numeric_cols:
        skew_df = g[numeric_cols].skew().reset_index()
        skew_df.columns = TOUCH_KEYS + [f'{col}_skewness' for col in numeric_cols]
        agg_df = agg_df.merge(skew_df, on=TOUCH_KEYS)

    # Shared metadata — all vectorised
    meta_spec: dict = {}
    if 'type_metadata' in df.columns:
        meta_spec['type_metadata'] = 'first'
    if 'gesture_type' in df.columns:
        meta_spec['gesture_type'] = 'first'
    if 'contact_location_x' in df.columns:
        meta_spec['contact_location_x'] = 'mean'
    if 'contact_location_y' in df.columns:
        meta_spec['contact_location_y'] = 'mean'
    if 'contact_location_z' in df.columns:
        meta_spec['contact_location_z'] = 'mean'
    if has_nerve_data and 'Nerve_spike' in df.columns:
        meta_spec['Nerve_spike'] = 'max'

    meta_df = g.agg(meta_spec).reset_index() if meta_spec else g.size().reset_index(name='_n').drop(columns=['_n'])
    meta_df.rename(columns={
        'contact_location_x': 'mean_contact_x',
        'contact_location_y': 'mean_contact_y',
        'contact_location_z': 'mean_contact_z',
    }, inplace=True)
    if 'Nerve_spike' in meta_df.columns:
        meta_df['spike_elicited'] = meta_df['Nerve_spike'].clip(0, 1).fillna(0).astype(int)
        meta_df.drop(columns=['Nerve_spike'], inplace=True)
    else:
        meta_df['spike_elicited'] = 0
    if 'type_metadata' not in meta_df.columns:
        meta_df['type_metadata'] = 'unknown'

    combined = meta_df.merge(agg_df, on=TOUCH_KEYS)
    combined = combined[combined['single_touch_id'] != 0].copy()
    combined['session_id'] = session_id

    result: dict[str, pd.DataFrame] = {}
    for agg in enabled_aggregations:
        agg_cols = [c for c in combined.columns if c.endswith(f'_{agg}')]
        meta_cols = [c for c in SHARED_COLUMNS if c in combined.columns]
        result[agg] = combined[meta_cols + agg_cols].copy()

    return result


def _extract_all_touches(
    df: pd.DataFrame,
    extractor,
    feature_config: dict,
    has_nerve_data: bool,
    desc: str = "",
    *,
    _groups: list | None = None,
) -> list[dict]:
    rows = []
    groups = _groups if _groups is not None else list(df.groupby(TOUCH_KEYS))
    for (block_order_id, trial_id, touch_id), group in tqdm(
        groups, desc=desc, leave=False, unit="touch", disable=True
    ):
        if group.empty or touch_id == 0:
            continue

        touch_type = (
            group['type_metadata'].iloc[0]
            if 'type_metadata' in group.columns else 'unknown'
        )

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
            'gesture_type': group['gesture_type'].iloc[0] if 'gesture_type' in group.columns else None,
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

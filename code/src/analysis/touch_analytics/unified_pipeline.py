# unified_pipeline.py
"""
Unified Touch Analysis Pipeline.

Orchestrates per-session feature extraction and global clustering in a single
flow, replacing the old (summarize_touches_per_session +
analyse_number_single_touches) pair.

Output layout
-------------
4_analysed/unified_touches/
  <profile_name>/
    <session>_touch_summary.csv
    clustering/
      <clusterer_name>/
        pooled_touch_summary_clustered.csv
        cluster_metadata.json
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from utils.should_process_task import should_process_task
from .feature_extraction import get_extractor
from .clustering import get_clusterer

# Columns written by the orchestrator (not the extractor)
_SHARED_COLUMNS = [
    'block_order_id', 'trial_id', 'single_touch_id',
    'type_metadata', 'direction',
    'mean_contact_x', 'mean_contact_y', 'mean_contact_z',
    'spike_elicited',
]

# Default config used when the YAML section is missing
_DEFAULT_OPTIONS: dict = {
    'force_processing': False,
    'extraction_profiles': {
        'max': {'method': 'max'},
    },
    'clustering_profiles': {
        'kmeans': {'method': 'kmeans', 'min_touches_per_cluster': 30},
    },
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_unified_touch_analysis(
    input_items: List[Tuple[Path, Path]],
    options: dict,
    output_dir: Path,
    force: bool = False,
) -> List[Path]:
    """
    Run extraction (per session) then clustering (global, per profile).

    Parameters
    ----------
    input_items
        List of (aggregated_session_csv, database_root_path) tuples.
    options
        Task options dict from YAML (merged with _DEFAULT_OPTIONS for missing keys).
    output_dir
        Root directory for all unified touch outputs (e.g. ``database / '4_analysed' / 'unified_touches'``).
    force
        Override idempotency checks.

    Returns
    -------
    List of output file paths written.
    """
    opts = {**_DEFAULT_OPTIONS, **options}
    extraction_profiles: dict = opts.get('extraction_profiles', _DEFAULT_OPTIONS['extraction_profiles'])
    clustering_profiles: dict = opts.get('clustering_profiles', _DEFAULT_OPTIONS['clustering_profiles'])
    force = opts.get('force_processing', False) or force

    all_outputs: List[Path] = []

    # Per-session extraction for every profile
    # Keys: profile_name → list of session CSV paths
    per_profile_session_csvs: dict[str, list[Path]] = {p: [] for p in extraction_profiles}

    for input_file, _ in input_items:
        session_outputs = _extract_session(
            input_file=input_file,
            output_dir=output_dir,
            extraction_profiles=extraction_profiles,
            force=force,
        )
        for profile_name, csv_path in session_outputs.items():
            per_profile_session_csvs[profile_name].append(csv_path)
            all_outputs.append(csv_path)

    # Global clustering per extraction profile × clustering profile
    for profile_name, session_csvs in per_profile_session_csvs.items():
        if not session_csvs:
            continue
        cluster_outputs = _cluster_profile(
            profile_name=profile_name,
            session_csvs=session_csvs,
            output_dir=output_dir,
            clustering_profiles=clustering_profiles,
            force=force,
        )
        all_outputs.extend(cluster_outputs)

    return all_outputs


# ---------------------------------------------------------------------------
# Per-session extraction
# ---------------------------------------------------------------------------

def _extract_session(
    input_file: Path,
    output_dir: Path,
    extraction_profiles: dict,
    force: bool,
) -> dict[str, Path]:
    """
    Run all enabled extraction profiles on one session CSV.

    Returns
    -------
    dict mapping profile_name → output CSV path (may point to an up-to-date
    file that was not rewritten).
    """
    results: dict[str, Path] = {}

    try:
        df = pd.read_csv(input_file)
    except Exception as exc:
        logging.error(f"Failed to load {input_file}: {exc}")
        return results

    # Shared preprocessing (block_order_id, nerve flag)
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

    # Determine output filename stem once
    filename = input_file.name
    if '_semicontrolled_' in filename:
        prefix = filename.split('_semicontrolled_')[0]
        csv_stem = f'{prefix}_semicontrolled_touch_summary.csv'
    else:
        csv_stem = f'{input_file.stem}_touch_summary.csv'

    for profile_name, profile_config in extraction_profiles.items():
        method = profile_config.get('method', profile_name)

        # Output paths
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
                    continue
            except FileNotFoundError:
                pass  # input missing — will error below

        # Extract
        try:
            extractor = get_extractor(method)
        except KeyError as exc:
            logging.error(f"Profile '{profile_name}': {exc}")
            continue

        rows = _extract_all_touches(df, extractor, profile_config, has_nerve_data)
        summary_df = pd.DataFrame(rows)

        try:
            summary_df.to_csv(output_path, index=False)
            logging.info(
                f"[{profile_name}] Saved {len(summary_df)} touches → {output_path}"
            )
        except Exception as exc:
            logging.error(f"Failed to save {output_path}: {exc}")
            continue

        results[profile_name] = output_path

    return results


def _extract_all_touches(
    df: pd.DataFrame,
    extractor,
    profile_config: dict,
    has_nerve_data: bool,
) -> list[dict]:
    rows = []
    for (block_order_id, trial_id, touch_id), group in df.groupby(
        ['block_order_id', 'trial_id', 'single_touch_id']
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


# ---------------------------------------------------------------------------
# Global clustering
# ---------------------------------------------------------------------------

def _cluster_profile(
    profile_name: str,
    session_csvs: list[Path],
    output_dir: Path,
    clustering_profiles: dict,
    force: bool,
) -> list[Path]:
    """Pool all session CSVs for *profile_name* and run each clusterer."""
    outputs: list[Path] = []

    # Pool
    dfs = []
    for p in session_csvs:
        if p.exists():
            try:
                dfs.append(pd.read_csv(p))
            except Exception as exc:
                logging.warning(f"Could not read {p}: {exc}")
    if not dfs:
        logging.warning(f"[{profile_name}] No session CSVs to cluster.")
        return outputs

    pooled = pd.concat(dfs, ignore_index=True)

    for clusterer_name, clusterer_config in clustering_profiles.items():
        method = clusterer_config.get('method', clusterer_name)
        out_dir = output_dir / profile_name / 'clustering' / clusterer_name
        out_dir.mkdir(parents=True, exist_ok=True)

        pooled_csv = out_dir / 'pooled_touch_summary_clustered.csv'
        metadata_json = out_dir / 'cluster_metadata.json'

        # Idempotency
        if not force and pooled_csv.exists():
            try:
                if not should_process_task(
                    input_paths=session_csvs,
                    output_paths=[pooled_csv],
                    force=False,
                ):
                    outputs.append(pooled_csv)
                    continue
            except FileNotFoundError:
                pass

        try:
            clusterer = get_clusterer(method)
        except KeyError as exc:
            logging.error(f"Clustering profile '{clusterer_name}': {exc}")
            continue

        # Feature columns only (drop shared/categorical)
        feature_cols = [
            c for c in pooled.columns
            if c not in _SHARED_COLUMNS
            and pd.api.types.is_numeric_dtype(pooled[c])
        ]
        if not feature_cols:
            logging.warning(
                f"[{profile_name}/{clusterer_name}] No numeric feature columns found."
            )
            continue

        feature_df = pooled[feature_cols].dropna()
        valid_idx = feature_df.index

        try:
            labels, metadata = clusterer.fit_predict(feature_df, clusterer_config)
        except Exception as exc:
            logging.error(
                f"[{profile_name}/{clusterer_name}] Clustering failed: {exc}"
            )
            continue

        result_df = pooled.loc[valid_idx].copy()
        result_df['cluster_label'] = labels

        # Session coverage
        sessions = result_df.get('trial_id', pd.Series(dtype=str)).unique().tolist()
        metadata['session_coverage'] = len(sessions)

        try:
            result_df.to_csv(pooled_csv, index=False)
            with open(metadata_json, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logging.info(
                f"[{profile_name}/{clusterer_name}] Clustered {len(result_df)} touches "
                f"→ {pooled_csv}"
            )
            outputs.append(pooled_csv)
        except Exception as exc:
            logging.error(f"Failed to save clustering output: {exc}")

    return outputs

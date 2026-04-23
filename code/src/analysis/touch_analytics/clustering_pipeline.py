# clustering_pipeline.py
"""
Standalone clustering pipeline.

Discovers per-session feature CSVs on disk (written by extraction_pipeline.py),
merges them per feature combination, and runs the cross-product of
(enabled feature_combinations) x (enabled clustering_profiles).

Output layout
-------------
<output_dir>/
  <combination_name>/
    <clusterer_name>/
      pooled_touch_summary_clustered.csv
      cluster_metadata.json
      heatmaps/
        <session>_touch_density.png
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.should_process_task import should_process_task, clean_task_outputs
from .clustering import get_clusterer
from .reporting import VisualReportingStrategy
from .pipeline_shared import (
    SHARED_COLUMNS,
    _TqdmLineWrapper,
    filter_enabled_profiles,
    session_id_from_path,
)

# Columns that uniquely identify a single touch across feature CSVs
_TOUCH_ID_COLS = ['block_order_id', 'trial_id', 'single_touch_id', 'session_id']


def _translate_extraction_profiles_to_combinations(extraction_profiles: dict) -> dict:
    """
    Translate old ``extraction_profiles`` format to ``feature_combinations``.

    Each old profile becomes a combination whose feature list contains only
    the profile's own name::

        extraction_profiles:
          max:
            method: max
          stats:
            method: statistical

    becomes::

        feature_combinations:
          max:
            enabled: true
            features: [max]
          stats:
            enabled: true
            features: [stats]
    """
    combinations: dict = {}
    for profile_name, profile_config in extraction_profiles.items():
        enabled = profile_config.get('enabled', True)
        combinations[profile_name] = {
            'enabled': enabled,
            'features': [profile_name],
        }
    return combinations


def _merge_feature_csvs(feature_names: list[str], extraction_dir: Path) -> pd.DataFrame:
    """
    Load and merge per-session CSVs from multiple feature folders.

    For each session present in **all** requested feature folders, the CSVs are
    joined on touch-identity columns (block_order_id, trial_id, single_touch_id,
    session_id). Sessions missing from any feature folder are skipped with a
    warning. All merged sessions are then pooled into a single DataFrame.

    Parameters
    ----------
    feature_names
        Ordered list of feature names (e.g. ``['max', 'mean']``).
    extraction_dir
        Root of the extraction output tree
        (e.g. ``database / '4_analysed' / 'touch_features'``).

    Returns
    -------
    pd.DataFrame
        Pooled DataFrame with shared columns from the first feature plus
        feature-specific columns from all features merged on touch identity.
        Empty DataFrame if no common sessions are found.
    """
    # --- Load all session CSVs per feature --------------------------------
    feature_sessions: dict[str, dict[str, pd.DataFrame]] = {}
    for feature_name in feature_names:
        feature_dir = extraction_dir / feature_name
        if not feature_dir.exists():
            logging.warning(
                f"_merge_feature_csvs: feature folder not found: {feature_dir} — "
                f"run touch_feature_extraction with feature '{feature_name}' enabled."
            )
            feature_sessions[feature_name] = {}
            continue
        sessions: dict[str, pd.DataFrame] = {}
        for csv in sorted(feature_dir.glob('*_touch_summary.csv')):
            try:
                sessions[csv.stem] = pd.read_csv(csv)
            except Exception as exc:
                logging.warning(f"Could not read {csv}: {exc}")
        feature_sessions[feature_name] = sessions

    # --- Intersect sessions present in ALL feature folders ----------------
    non_empty = [set(v.keys()) for v in feature_sessions.values() if v]
    if not non_empty:
        logging.warning("_merge_feature_csvs: no session CSVs found in any feature folder.")
        return pd.DataFrame()

    common_stems = set.intersection(*non_empty)
    if not common_stems:
        logging.warning(
            "_merge_feature_csvs: no sessions are present in all feature folders — "
            f"features requested: {feature_names}"
        )
        return pd.DataFrame()

    # Warn about sessions present in some but not all feature folders
    all_stems = set.union(*non_empty)
    incomplete = all_stems - common_stems
    if incomplete:
        logging.warning(
            f"_merge_feature_csvs: {len(incomplete)} session(s) skipped — "
            f"not present in all feature folders: {sorted(incomplete)}"
        )

    # --- Merge per session ------------------------------------------------
    all_sessions: list[pd.DataFrame] = []
    for stem in sorted(common_stems):
        first_df = feature_sessions[feature_names[0]][stem].copy()
        merged = first_df

        for feature_name in feature_names[1:]:
            other_df = feature_sessions[feature_name][stem]
            # Only carry feature-specific columns from subsequent CSVs
            feature_only_cols = [c for c in other_df.columns if c not in SHARED_COLUMNS]
            merge_keys = [k for k in _TOUCH_ID_COLS if k in merged.columns and k in other_df.columns]
            subset = other_df[merge_keys + feature_only_cols]
            merged = merged.merge(subset, on=merge_keys, how='inner')
            # Warn on unexpected duplicate columns
            dup_cols = [c for c in feature_only_cols if c in first_df.columns]
            if dup_cols:
                logging.warning(
                    f"_merge_feature_csvs: duplicate columns detected when merging "
                    f"feature '{feature_name}': {dup_cols}"
                )

        all_sessions.append(merged)

    if not all_sessions:
        return pd.DataFrame()
    return pd.concat(all_sessions, ignore_index=True)


def run_clustering(
    output_dir: Path,
    feature_combinations: dict,
    clustering_profiles: dict,
    force: bool = False,
    extraction_dir: Path = None,
) -> dict[str, List[Path]]:
    """
    Merge feature CSVs per combination and run all configured clusterers.

    Executes the cross-product: each enabled ``feature_combination`` x each
    enabled ``clustering_profile`` produces one output directory.

    Parameters
    ----------
    output_dir
        Root directory for clustering outputs
        (e.g. ``database / '4_analysed' / 'touch_clusters'``).
    feature_combinations
        Dict mapping combination_name -> combination_config.
        Each config must contain ``features: [feature_name, ...]`` listing which
        feature folders to merge. Supports backward-compat ``extraction_profiles``
        format (each profile becomes a single-feature combination automatically).
    clustering_profiles
        Dict mapping clusterer_name -> clusterer_config. Profiles with
        ``enabled: false`` are skipped.
    force
        Override idempotency checks.
    extraction_dir
        Root of the extraction output tree. Defaults to *output_dir* when not
        provided (backward-compatible).

    Returns
    -------
    Dict mapping ``"<combination_name>/<clusterer_name>"`` -> list of
    pooled-clustered CSV paths written.
    """
    # Backward-compat: old format used extraction_profiles without a 'features' list
    if feature_combinations and not any(
        'features' in v for v in feature_combinations.values() if isinstance(v, dict)
    ):
        logging.warning(
            "clustering_pipeline: 'extraction_profiles' format detected — "
            "translating to new 'feature_combinations' format automatically."
        )
        feature_combinations = _translate_extraction_profiles_to_combinations(feature_combinations)

    feature_combinations = filter_enabled_profiles(feature_combinations)
    clustering_profiles = filter_enabled_profiles(clustering_profiles)

    src_dir = extraction_dir if extraction_dir is not None else output_dir

    results: dict[str, List[Path]] = {}

    n_combinations = len(feature_combinations)
    n_clusterers = len(clustering_profiles)
    print(
        f"=== clustering pipeline: {n_combinations} feature combination(s), "
        f"{n_clusterers} clusterer(s) ===",
        flush=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for combination_name, combination_config in feature_combinations.items():
            feature_names: list[str] = combination_config.get('features', [])
            if not feature_names:
                logging.warning(
                    f"[{combination_name}] No features listed in combination config — skipping."
                )
                continue

            print(
                f"  [cluster] combination '{combination_name}' — "
                f"features: {feature_names}",
                flush=True,
            )

            pooled = _merge_feature_csvs(feature_names, src_dir)
            if pooled.empty:
                logging.warning(
                    f"[{combination_name}] No data after merging feature CSVs — skipping."
                )
                continue

            print(
                f"  [cluster] combination '{combination_name}' — "
                f"{len(pooled)} touches from {pooled['session_id'].nunique() if 'session_id' in pooled.columns else '?'} session(s)",
                flush=True,
            )

            cluster_outputs = _cluster_combination(
                combination_name=combination_name,
                pooled=pooled,
                output_dir=output_dir,
                clustering_profiles=clustering_profiles,
                force=force,
            )
            for clusterer_name, paths in cluster_outputs.items():
                key = f"{combination_name}/{clusterer_name}"
                results[key] = paths

    total = sum(len(v) for v in results.values())
    print(f"=== clustering pipeline complete: {total} output(s) ===", flush=True)
    return results


def _cluster_combination(
    combination_name: str,
    pooled: pd.DataFrame,
    output_dir: Path,
    clustering_profiles: dict,
    force: bool,
) -> dict[str, List[Path]]:
    """Run each clusterer on the already-pooled *pooled* DataFrame."""
    outputs: dict[str, List[Path]] = {}

    for clusterer_name, clusterer_config in clustering_profiles.items():
        method = clusterer_config.get('method', clusterer_name)
        out_dir = output_dir / combination_name / clusterer_name
        out_dir.mkdir(parents=True, exist_ok=True)

        pooled_csv = out_dir / 'pooled_touch_summary_clustered.csv'
        metadata_json = out_dir / 'cluster_metadata.json'

        # Idempotency: use session CSVs as conceptual inputs — skip if up to date
        if not force and pooled_csv.exists():
            try:
                if not should_process_task(
                    input_paths=[pooled_csv],  # approximate: use output mtime
                    output_paths=[pooled_csv],
                    force=False,
                ):
                    outputs.setdefault(clusterer_name, []).append(pooled_csv)
                    print(
                        f"  [cluster] {combination_name} / {clusterer_name} — up to date",
                        flush=True,
                    )
                    continue
            except FileNotFoundError:
                pass
        clean_task_outputs([pooled_csv, metadata_json])

        try:
            clusterer = get_clusterer(method)
        except KeyError as exc:
            logging.error(f"Clustering profile '{clusterer_name}': {exc}")
            print(
                f"  [cluster] {combination_name} / {clusterer_name} — error: unknown clusterer",
                flush=True,
            )
            continue

        # Feature columns only (drop shared/categorical)
        feature_cols = [
            c for c in pooled.columns
            if c not in SHARED_COLUMNS
            and pd.api.types.is_numeric_dtype(pooled[c])
        ]
        if not feature_cols:
            logging.warning(
                f"[{combination_name}/{clusterer_name}] No numeric feature columns found."
            )
            print(
                f"  [cluster] {combination_name} / {clusterer_name} — error: no numeric features",
                flush=True,
            )
            continue

        feature_df = pooled[feature_cols].dropna()
        valid_idx = feature_df.index

        # Inject sensor labels for clusterers that need them (e.g. HierarchicalClusterer)
        sensor_col = clusterer_config.get('sensor_col')
        if sensor_col and sensor_col in pooled.columns:
            clusterer_config = {**clusterer_config, '_sensor_labels': pooled.loc[valid_idx, sensor_col].values}

        type_col = clusterer_config.get('type_col')
        if type_col and type_col in pooled.columns:
            clusterer_config = {
                **clusterer_config,
                '_type_labels': pooled.loc[valid_idx, type_col].values,
                '_direction_labels': (
                    pooled.loc[valid_idx, 'direction'].values
                    if 'direction' in pooled.columns else None
                ),
            }

        try:
            labels, metadata = clusterer.fit_predict(feature_df, clusterer_config)
        except Exception as exc:
            logging.error(f"[{combination_name}/{clusterer_name}] Clustering failed: {exc}")
            print(
                f"  [cluster] {combination_name} / {clusterer_name} — error: clustering failed",
                flush=True,
            )
            continue

        result_df = pooled.loc[valid_idx].copy()
        result_df['cluster_label'] = labels

        extra_cols = metadata.pop('extra_columns', None)
        if extra_cols:
            for col_name, col_values in extra_cols.items():
                result_df[col_name] = col_values

        # Session coverage
        sessions = result_df.get('session_id', pd.Series(dtype=str)).unique().tolist()
        metadata['session_coverage'] = len(sessions)

        try:
            result_df.to_csv(pooled_csv, index=False)
            with open(metadata_json, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logging.info(
                f"[{combination_name}/{clusterer_name}] Clustered {len(result_df)} touches "
                f"→ {pooled_csv}"
            )
            outputs.setdefault(clusterer_name, []).append(pooled_csv)
        except Exception as exc:
            logging.error(f"Failed to save clustering output: {exc}")
            print(
                f"  [cluster] {combination_name} / {clusterer_name} — error: save failed",
                flush=True,
            )
            continue

        if 'k' in metadata:
            cluster_desc = f"{metadata['k']} clusters, {len(result_df)} samples"
        elif 'n_bins' in metadata:
            cluster_desc = f"{metadata['n_bins']} bins, {len(result_df)} samples"
        else:
            cluster_desc = f"{len(result_df)} samples"
        print(f"  [cluster] {combination_name} / {clusterer_name} — {cluster_desc}", flush=True)

        if len(feature_cols) >= 2 and 'session_id' in result_df.columns:
            n_heatmap_sessions = result_df['session_id'].nunique()
            _generate_session_heatmaps(result_df, out_dir, feature_cols)
            print(
                f"  [heatmap] {combination_name} / {clusterer_name} — {n_heatmap_sessions} sessions",
                flush=True,
            )
        else:
            print(f"  [heatmap] {combination_name} / {clusterer_name} — skipped", flush=True)

    return outputs


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _select_highest_variance_feature(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Return the feature column with the highest variance across *df*."""
    return df[feature_cols].var().idxmax()


def _generate_session_heatmaps(
    result_df: pd.DataFrame,
    out_dir: Path,
    feature_cols: list[str],
) -> None:
    """
    Generate one touch-density heatmap PNG per session in *result_df*.

    Skips silently when fewer than 2 feature columns are available.
    """
    if len(feature_cols) < 2:
        logging.warning("_generate_session_heatmaps: need >= 2 features, skipping.")
        return
    if 'session_id' not in result_df.columns:
        logging.warning("_generate_session_heatmaps: 'session_id' column missing, skipping.")
        return

    velocity_cols = [c for c in feature_cols if 'velocity' in c.lower()]
    if velocity_cols:
        x_col = velocity_cols[0]
    else:
        x_col = _select_highest_variance_feature(result_df, feature_cols)
    y_cols = [c for c in feature_cols if c != x_col]

    num_bins = 20
    all_cols = [x_col] + y_cols
    global_edges: dict[str, np.ndarray] = {}
    for col in all_cols:
        series = result_df[col].dropna()
        vmin, vmax = series.min(), series.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
            global_edges[col] = np.linspace(0, 1, num_bins + 1)
        elif vmin > 0:
            global_edges[col] = np.geomspace(vmin, vmax, num_bins + 1)
        else:
            global_edges[col] = np.linspace(vmin, vmax, num_bins + 1)

    global_max_count = 0
    for session_id in result_df['session_id'].unique():
        session_df = result_df[result_df['session_id'] == session_id]
        for i_type in ('tap', 'stroke'):
            subset = session_df[session_df['type_metadata'] == i_type]
            if subset.empty:
                continue
            x_binned = pd.cut(subset[x_col], bins=global_edges[x_col], include_lowest=True)
            for y_col in y_cols:
                y_binned = pd.cut(subset[y_col], bins=global_edges[y_col], include_lowest=True)
                matrix = pd.crosstab(y_binned, x_binned, dropna=False)
                cur_max = int(matrix.max().max())
                if cur_max > global_max_count:
                    global_max_count = cur_max

    reporter = VisualReportingStrategy(out_dir / 'heatmaps')

    for session_id in result_df['session_id'].unique():
        session_df = result_df[result_df['session_id'] == session_id]
        try:
            reporter.generate_touch_density_heatmap(
                df=session_df,
                x_col=x_col,
                y_cols=y_cols,
                type_col='type_metadata',
                num_bins=num_bins,
                log_axis=True,
                title_suffix=str(session_id),
                filename=f"{session_id}_touch_density.png",
                global_edges=global_edges,
                global_max_count=global_max_count,
            )
        except Exception as exc:
            logging.error(f"Heatmap generation failed for session {session_id}: {exc}")

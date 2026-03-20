# clustering_pipeline.py
"""
Standalone clustering pipeline.

Replaces the clustering logic formerly in unified_pipeline.py.
Discovers per-session extraction CSVs on disk (written by extraction_pipeline.py),
pools them per extraction profile, and runs each configured clusterer.

Output layout
-------------
<output_dir>/
  <extraction_profile>/
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


def run_clustering(
    output_dir: Path,
    extraction_profiles: dict,
    clustering_profiles: dict,
    force: bool = False,
    extraction_dir: Path = None,
) -> dict[str, List[Path]]:
    """
    Discover extraction CSVs on disk and run all configured clusterers.

    Parameters
    ----------
    output_dir
        Root directory for clustering outputs
        (e.g. ``database / '4_analysed' / 'touch_clusters'``).
    extraction_dir
        Root directory where extraction CSVs were written
        (e.g. ``database / '4_analysed' / 'touch_features'``).
        Defaults to *output_dir* when not provided (backward-compatible).
        Extraction CSVs are expected at ``extraction_dir/<profile>/*_touch_summary.csv``.
    extraction_profiles
        Dict mapping profile_name -> profile_config. Only the names are used
        for directory scanning; profiles with ``enabled: false`` are skipped.
    clustering_profiles
        Dict mapping clusterer_name -> clusterer_config. Profiles with
        ``enabled: false`` are skipped.
    force
        Override idempotency checks.

    Returns
    -------
    Dict mapping ``"<extraction_profile>/<clusterer_name>"`` -> list of
    pooled-clustered CSV paths written.
    """
    extraction_profiles = filter_enabled_profiles(extraction_profiles)
    clustering_profiles = filter_enabled_profiles(clustering_profiles)

    src_dir = extraction_dir if extraction_dir is not None else output_dir

    results: dict[str, List[Path]] = {}

    print(
        f"=== clustering pipeline: {len(extraction_profiles)} extraction profile(s), "
        f"{len(clustering_profiles)} clusterer(s) ===",
        flush=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for profile_name in extraction_profiles:
            profile_dir = src_dir / profile_name
            session_csvs = sorted(profile_dir.glob("*_touch_summary.csv"))

            if not session_csvs:
                logging.warning(
                    f"[{profile_name}] No extraction CSVs found in {profile_dir}. "
                    "Run touch_feature_extraction first."
                )
                continue

            print(
                f"  [cluster] {profile_name} — {len(session_csvs)} session CSV(s) found",
                flush=True,
            )

            cluster_outputs = _cluster_profile(
                profile_name=profile_name,
                session_csvs=session_csvs,
                output_dir=output_dir,
                clustering_profiles=clustering_profiles,
                force=force,
            )
            for clusterer_name, paths in cluster_outputs.items():
                key = f"{profile_name}/{clusterer_name}"
                results[key] = paths

    total = sum(len(v) for v in results.values())
    print(f"=== clustering pipeline complete: {total} output(s) ===", flush=True)
    return results


def _cluster_profile(
    profile_name: str,
    session_csvs: List[Path],
    output_dir: Path,
    clustering_profiles: dict,
    force: bool,
) -> dict[str, List[Path]]:
    """Pool all session CSVs for *profile_name* and run each clusterer."""
    outputs: dict[str, List[Path]] = {}

    # Pool all sessions
    dfs = []
    for p in session_csvs:
        if p.exists():
            try:
                session_df = pd.read_csv(p)
                session_df['session_id'] = session_id_from_path(p)
                dfs.append(session_df)
            except Exception as exc:
                logging.warning(f"Could not read {p}: {exc}")
    if not dfs:
        logging.warning(f"[{profile_name}] No session CSVs could be loaded.")
        return outputs

    pooled = pd.concat(dfs, ignore_index=True)

    for clusterer_name, clusterer_config in clustering_profiles.items():
        method = clusterer_config.get('method', clusterer_name)
        out_dir = output_dir / profile_name / clusterer_name
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
                    outputs.setdefault(clusterer_name, []).append(pooled_csv)
                    print(
                        f"  [cluster] {profile_name} / {clusterer_name} — up to date",
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
                f"  [cluster] {profile_name} / {clusterer_name} — error: unknown clusterer",
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
                f"[{profile_name}/{clusterer_name}] No numeric feature columns found."
            )
            print(
                f"  [cluster] {profile_name} / {clusterer_name} — error: no numeric features",
                flush=True,
            )
            continue

        feature_df = pooled[feature_cols].dropna()
        valid_idx = feature_df.index

        # Inject sensor labels for clusterers that need them (e.g. HierarchicalClusterer)
        sensor_col = clusterer_config.get('sensor_col')
        if sensor_col and sensor_col in pooled.columns:
            clusterer_config = {**clusterer_config, '_sensor_labels': pooled.loc[valid_idx, sensor_col].values}

        try:
            labels, metadata = clusterer.fit_predict(feature_df, clusterer_config)
        except Exception as exc:
            logging.error(f"[{profile_name}/{clusterer_name}] Clustering failed: {exc}")
            print(
                f"  [cluster] {profile_name} / {clusterer_name} — error: clustering failed",
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
                f"[{profile_name}/{clusterer_name}] Clustered {len(result_df)} touches "
                f"→ {pooled_csv}"
            )
            outputs.setdefault(clusterer_name, []).append(pooled_csv)
        except Exception as exc:
            logging.error(f"Failed to save clustering output: {exc}")
            print(
                f"  [cluster] {profile_name} / {clusterer_name} — error: save failed",
                flush=True,
            )
            continue

        if 'k' in metadata:
            cluster_desc = f"{metadata['k']} clusters, {len(result_df)} samples"
        elif 'n_bins' in metadata:
            cluster_desc = f"{metadata['n_bins']} bins, {len(result_df)} samples"
        else:
            cluster_desc = f"{len(result_df)} samples"
        print(f"  [cluster] {profile_name} / {clusterer_name} — {cluster_desc}", flush=True)

        if len(feature_cols) >= 2 and 'session_id' in result_df.columns:
            n_heatmap_sessions = result_df['session_id'].nunique()
            _generate_session_heatmaps(result_df, out_dir, feature_cols)
            print(
                f"  [heatmap] {profile_name} / {clusterer_name} — {n_heatmap_sessions} sessions",
                flush=True,
            )
        else:
            print(f"  [heatmap] {profile_name} / {clusterer_name} — skipped", flush=True)

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

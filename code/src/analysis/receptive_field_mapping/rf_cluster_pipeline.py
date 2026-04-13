"""Cluster-based receptive field mapping pipeline.

Reads clustered touch summaries from touch_clustering, traces each touch back
to its session's aggregated CSV to extract spike-filtered contact points (with
proper 30Hz-to-1kHz forward-fill), counts spikes per contact point, and saves
per-cluster spike_counts.csv files. Renders per-session 3D forearm heatmap PNGs.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from analysis.receptive_field_mapping.rf_cluster_visualizer import render_forearm_heatmap
from analysis.receptive_field_mapping.rf_data_loader import (
    parse_contact_points,
    resolve_forearm_ply,
)
from analysis.touch_analytics.pipeline_shared import (
    filter_enabled_profiles,
    session_id_from_path,
)
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

# Columns needed from aggregated CSVs
_KEY_COLS = ['block_order_id', 'trial_id', 'single_touch_id']
_DATA_COLS = ['contact_points', 'Nerve_spike']
_NEEDED_COLS = _KEY_COLS + _DATA_COLS


def _resolve_session_paths(
    input_items: List[Tuple[Path, Path]],
) -> Dict[str, Path]:
    """Map session_id -> session_merged_output_dir from input_items.

    Parameters
    ----------
    input_items:
        List of (aggregated_csv_path, database_path) tuples from the DAG runner.

    Returns
    -------
    dict mapping session_id to its session_merged_output_dir (parent of aggregated CSV).
    """
    session_map: Dict[str, Path] = {}
    for csv_path, _ in input_items:
        session_id = session_id_from_path(csv_path)
        session_map[session_id] = csv_path.parent
    return session_map


def _group_touches_by_cluster(
    clustered_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Group clustered touch summary by cluster_label.

    Parameters
    ----------
    clustered_df:
        DataFrame from pooled_touch_summary_clustered.csv, must contain
        cluster_label and the 4-column touch key columns.

    Returns
    -------
    dict mapping cluster_label (str) to a sub-DataFrame with unique 4-col keys.
    """
    key_cols = ['session_id', 'block_order_id', 'trial_id', 'single_touch_id']
    available_keys = [c for c in key_cols if c in clustered_df.columns]

    groups: Dict[str, pd.DataFrame] = {}
    for cluster_label, group_df in clustered_df.groupby('cluster_label'):
        label_str = str(cluster_label)
        groups[label_str] = group_df[available_keys].drop_duplicates().reset_index(drop=True)
    return groups


def _build_session_touch_map(
    cluster_df: pd.DataFrame,
) -> Dict[str, List[Tuple]]:
    """Group touch keys by session_id.

    Parameters
    ----------
    cluster_df:
        Sub-DataFrame for one cluster with columns including session_id,
        block_order_id, trial_id, single_touch_id.

    Returns
    -------
    dict mapping session_id -> list of (block_order_id, trial_id, single_touch_id) tuples.
    """
    session_map: Dict[str, List[Tuple]] = {}
    for session_id, grp in cluster_df.groupby('session_id'):
        keys = list(
            grp[['block_order_id', 'trial_id', 'single_touch_id']]
            .itertuples(index=False, name=None)
        )
        session_map[str(session_id)] = keys
    return session_map


def _extract_spike_contact_points(
    aggregated_csv_path: Path,
    touch_keys: List[Tuple],
) -> Counter:
    """Extract spike-associated contact points from one session's aggregated CSV.

    Reads the aggregated CSV, filters to the specified touches, forward-fills
    contact_points within each touch group (30Hz -> 1kHz alignment), filters
    rows where Nerve_spike == 1, parses contact_points, and returns a Counter
    of (x, y, z) -> spike_count.

    Parameters
    ----------
    aggregated_csv_path:
        Path to the session's *_semicontrolled_aggregated_session.csv.
    touch_keys:
        List of (block_order_id, trial_id, single_touch_id) tuples for this cluster.

    Returns
    -------
    Counter mapping (x, y, z) tuple -> spike count.
    """
    spike_counter: Counter = Counter()

    if not aggregated_csv_path.exists():
        logger.warning("Aggregated CSV not found, skipping: %s", aggregated_csv_path)
        return spike_counter

    try:
        header_df = pd.read_csv(aggregated_csv_path, nrows=0)
        available = set(header_df.columns)
        missing = [c for c in _NEEDED_COLS if c not in available]
        if missing:
            logger.warning(
                "Skipping %s: missing columns %s", aggregated_csv_path.name, missing
            )
            return spike_counter

        df = pd.read_csv(aggregated_csv_path, usecols=_NEEDED_COLS)
    except Exception:
        logger.exception("Error reading %s", aggregated_csv_path.name)
        return spike_counter

    if df.empty:
        return spike_counter

    # Merge-based filter: much faster than row-wise apply for large DataFrames
    keys_df = pd.DataFrame(
        touch_keys, columns=['block_order_id', 'trial_id', 'single_touch_id']
    ).drop_duplicates()
    touch_df = df.merge(keys_df, on=['block_order_id', 'trial_id', 'single_touch_id'], how='inner').copy()

    if touch_df.empty:
        return spike_counter

    # Forward-fill contact_points within each touch group (30Hz -> 1kHz)
    # Must be per-touch to avoid leaking contact location between touches.
    touch_df['contact_points'] = touch_df.groupby(
        ['block_order_id', 'trial_id', 'single_touch_id']
    )['contact_points'].ffill()

    # Filter to spike rows and parse contact points
    spike_rows = touch_df[touch_df['Nerve_spike'] == 1]
    for cp_str in spike_rows['contact_points']:
        parsed = parse_contact_points(cp_str)
        for pt in parsed:
            spike_counter[pt] += 1

    return spike_counter


def _aggregate_spike_counts(
    session_counters: List[Counter],
) -> pd.DataFrame:
    """Aggregate spike counters from multiple sessions into a DataFrame.

    Parameters
    ----------
    session_counters:
        List of Counter objects, one per session.

    Returns
    -------
    DataFrame with columns (x, y, z, spike_count), sorted by spike_count descending.
    """
    total: Counter = Counter()
    for c in session_counters:
        total.update(c)

    if not total:
        return pd.DataFrame(columns=['x', 'y', 'z', 'spike_count'])

    rows = [
        {'x': pt[0], 'y': pt[1], 'z': pt[2], 'spike_count': count}
        for pt, count in total.items()
    ]
    df = pd.DataFrame(rows).sort_values('spike_count', ascending=False).reset_index(drop=True)
    return df


def run_cluster_rf_mapping(
    clustering_dir: Path,
    input_items: List[Tuple[Path, Path]],
    output_dir: Path,
    feature_combinations: dict,
    clustering_profiles: dict,
    force: bool = False,
) -> List[Path]:
    """Orchestrate cluster-based receptive field mapping.

    For each enabled (feature_combination, clusterer) pair:
    1. Reads pooled_touch_summary_clustered.csv
    2. Groups by cluster_label
    3. For each cluster: traces touches to aggregated CSVs, forward-fills
       contact_points, filters by Nerve_spike==1, counts spikes per (x,y,z),
       saves spike_counts.csv
    4. For each (cluster, session): renders 3D forearm heatmap PNG
    5. Saves rf_cluster_summary.json as the idempotency sentinel

    Parameters
    ----------
    clustering_dir:
        Root of touch_clusters output:
        ``database_path / '4_analysed' / 'touch_clusters'``
    input_items:
        List of (aggregated_csv_path, database_path) tuples.
    output_dir:
        Root of output:
        ``database_path / '4_analysed' / 'receptive_field_maps_clustered'``
    feature_combinations:
        Dict of feature combination configs (from DAG options).
    clustering_profiles:
        Dict of clustering profile configs (from DAG options).
    force:
        If True, reprocess even if outputs are up-to-date.

    Returns
    -------
    List of paths to produced spike_counts.csv files.
    """

    session_dir_map = _resolve_session_paths(input_items)
    enabled_combinations = filter_enabled_profiles(feature_combinations)
    enabled_clusterers = filter_enabled_profiles(clustering_profiles)

    produced: List[Path] = []

    for combo_name in enabled_combinations:
        for clusterer_name in enabled_clusterers:
            print(f"[RF Cluster Mapping] {combo_name}/{clusterer_name}...")
            clustered_csv = (
                clustering_dir / combo_name / clusterer_name
                / 'pooled_touch_summary_clustered.csv'
            )
            if not clustered_csv.exists():
                print(f"  Clustered CSV not found, skipping.")
                logger.warning(
                    "Clustered CSV not found, skipping (%s/%s): %s",
                    combo_name, clusterer_name, clustered_csv,
                )
                continue

            base_output = output_dir / combo_name / clusterer_name
            summary_json = base_output / 'rf_cluster_summary.json'

            if not should_process_task(
                input_paths=[clustered_csv],
                output_paths=[summary_json],
                force=force,
            ):
                print(f"  Up-to-date, skipping.")
                produced.append(summary_json)
                continue

            try:
                clustered_df = pd.read_csv(clustered_csv)
            except Exception:
                logger.exception("Failed to read %s", clustered_csv)
                continue

            if clustered_df.empty:
                print(f"  Clustered CSV is empty, skipping.")
                logger.warning("Clustered CSV is empty: %s", clustered_csv)
                continue

            cluster_groups = _group_touches_by_cluster(clustered_df)
            n_clusters = len(cluster_groups)
            print(f"  {n_clusters} clusters found.")
            summary_data: dict = {}

            for cluster_idx, (cluster_label, cluster_df) in enumerate(cluster_groups.items(), start=1):
                n_touches = len(cluster_df)
                print(f"  Cluster {cluster_idx}/{n_clusters} (label={cluster_label}, {n_touches} touches)...")
                cluster_out = base_output / f'cluster_{cluster_label}'
                cluster_out.mkdir(parents=True, exist_ok=True)

                session_touch_map = _build_session_touch_map(cluster_df)
                session_counters: List[Counter] = []
                session_spike_dfs: Dict[str, pd.DataFrame] = {}

                for session_id, touch_keys in session_touch_map.items():
                    print(f"    {session_id}: loading {len(touch_keys)} touches...")
                    if session_id not in session_dir_map:
                        print(f"    -> not in input_items, skipping.")
                        logger.warning(
                            "Session '%s' in cluster %s not found in input_items. Skipping.",
                            session_id, cluster_label,
                        )
                        continue

                    session_dir = session_dir_map[session_id]
                    agg_csvs = list(session_dir.glob('*_semicontrolled_aggregated_session.csv'))
                    if not agg_csvs:
                        print(f"    -> no aggregated CSV found, skipping.")
                        logger.warning(
                            "No aggregated CSV in %s. Skipping session %s.",
                            session_dir, session_id,
                        )
                        continue

                    session_counter = _extract_spike_contact_points(agg_csvs[0], touch_keys)
                    n_spikes = sum(session_counter.values())
                    print(f"    -> {n_spikes} spikes at {len(session_counter)} contact points.")
                    session_counters.append(session_counter)

                    if session_counter:
                        session_spike_dfs[session_id] = _aggregate_spike_counts([session_counter])

                # Save pooled spike_counts.csv for this cluster
                pooled_df = _aggregate_spike_counts(session_counters)
                spike_counts_csv = cluster_out / 'spike_counts.csv'
                pooled_df.to_csv(spike_counts_csv, index=False)
                produced.append(spike_counts_csv)

                total_spikes = int(pooled_df['spike_count'].sum()) if not pooled_df.empty else 0
                if pooled_df.empty:
                    print(f"  -> No spikes found for cluster {cluster_label}. spike_counts.csv is empty.")
                    logger.warning(
                        "No spikes found for cluster %s (%s/%s). spike_counts.csv is empty.",
                        cluster_label, combo_name, clusterer_name,
                    )
                else:
                    print(
                        f"  -> spike_counts.csv: {total_spikes} total spikes,"
                        f" {len(pooled_df)} unique contact points."
                    )

                # Render per-session 3D forearm heatmaps
                for session_id, spike_df in session_spike_dfs.items():
                    if spike_df.empty:
                        continue
                    session_dir = session_dir_map[session_id]
                    forearm_ply = resolve_forearm_ply(session_dir, session_id)
                    png_path = cluster_out / f'{session_id}_rf_heatmap.png'
                    print(f"    Rendering heatmap: {session_id}...")
                    try:
                        render_forearm_heatmap(
                            forearm_ply_path=forearm_ply,
                            spike_counts_df=spike_df,
                            output_path=png_path,
                            session_id=session_id,
                            cluster_label=cluster_label,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to render heatmap for session %s, cluster %s",
                            session_id, cluster_label,
                        )

                summary_data[cluster_label] = {
                    'n_sessions': len(session_touch_map),
                    'n_touches': n_touches,
                    'total_spike_points': len(pooled_df),
                    'total_spikes': total_spikes,
                }

            # Save rf_cluster_summary.json (idempotency sentinel)
            base_output.mkdir(parents=True, exist_ok=True)
            with open(summary_json, 'w') as f:
                json.dump(
                    {
                        'feature_combination': combo_name,
                        'clusterer': clusterer_name,
                        'clusters': summary_data,
                    },
                    f,
                    indent=2,
                )

            print(f"[RF Cluster Mapping] {combo_name}/{clusterer_name}: done ({n_clusters} clusters).")
            logger.info(
                "[%s/%s] RF cluster mapping complete: %d clusters.",
                combo_name, clusterer_name, n_clusters,
            )

    return produced

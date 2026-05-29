"""Cluster-based receptive field mapping — extraction pipeline.

Reads clustered touch summaries from stimulus_cluster_touches, traces each touch back
to its session's aggregated CSV to extract spike-filtered contact points (with
proper 30Hz-to-1kHz forward-fill), counts spikes per contact point, and saves
per-cluster spike_counts.csv files. Renders per-session 3D forearm heatmap PNGs.
"""

import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertices,
    parse_contact_points,
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.rf_extraction_io import (
    save_cluster_session_data,
    save_extraction_summary,
    save_forearm_vertices,
    save_neuron_cluster_touches,
    save_neuron_contacts,
    save_neuron_touches,
    save_sessions_metadata,
    load_extraction_summary,
)
from analysis.touch_analytics.clustering_pipeline import DATA_TYPE_TO_COLUMNS
from analysis.pipeline.shared_constants import GESTURE_TYPES, NERVE_SPIKE_COL, CONTACT_POINTS_COL
from analysis.pipeline.output_dirs import TOUCH_COMPUTE_SERIES
from analysis.touch_analytics.pipeline_shared import (
    SHARED_COLUMNS,
    filter_enabled_profiles,
    session_id_from_path,
)
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

# Columns needed from aggregated CSVs
_KEY_COLS = ['block_order_id', 'trial_id', 'single_touch_id']
_DATA_COLS = [CONTACT_POINTS_COL, NERVE_SPIKE_COL]
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


def _unique_mm(points_xyz: np.ndarray, mm: float = 1.0) -> np.ndarray:
    """Round points to mm-grid and return unique rows."""
    if len(points_xyz) == 0:
        return points_xyz
    rounded = np.round(points_xyz / mm) * mm
    return np.unique(rounded, axis=0)


def _load_session_aggregated(agg_csv: Path) -> pd.DataFrame:
    """Load the aggregated CSV with required columns, ffill contact_points per touch group.

    Raises ValueError if required columns are missing or the file is empty.
    """
    header_df = pd.read_csv(agg_csv, nrows=0)
    available = set(header_df.columns)
    missing = [c for c in _NEEDED_COLS if c not in available]
    if missing:
        raise ValueError(f"_load_session_aggregated: {agg_csv.name} missing columns {missing}")

    df = pd.read_csv(agg_csv, usecols=_NEEDED_COLS)
    if df.empty:
        raise ValueError(f"_load_session_aggregated: {agg_csv.name} is empty")

    df[CONTACT_POINTS_COL] = df.groupby(
        ['block_order_id', 'trial_id', 'single_touch_id']
    )[CONTACT_POINTS_COL].ffill()

    return df


def _parse_contacts_for_keys(
    df_session: pd.DataFrame,
    touch_keys: Optional[List[Tuple]],
    dedup_mm: float = 1.0,
) -> Tuple[Counter, defaultdict, np.ndarray]:
    """Filter df_session to touch_keys (or all rows if None), then return:
      - spike_counter: Counter mapping xyz_tuple -> spike count (Nerve_spike==1 rows).
      - unique_touch_counter: defaultdict mapping xyz_tuple -> set of touch keys.
      - all_contacts_xyz: (N, 3) mm-rounded unique cloud across ALL filtered rows.
    """
    spike_counter: Counter = Counter()
    unique_touch_counter: defaultdict = defaultdict(set)

    if touch_keys is not None:
        keys_df = pd.DataFrame(
            touch_keys, columns=['block_order_id', 'trial_id', 'single_touch_id']
        ).drop_duplicates()
        touch_df = df_session.merge(
            keys_df, on=['block_order_id', 'trial_id', 'single_touch_id'], how='inner'
        )
    else:
        touch_df = df_session

    if touch_df.empty:
        return spike_counter, unique_touch_counter, np.empty((0, 3))

    # Single pass: collect mm-rounded hull points and spike counts simultaneously.
    # contact_points ffill was applied at load time (per touch group).
    all_pts_mm: set = set()
    for row in touch_df.itertuples(index=False):
        cp_str = row.contact_points
        touch_key = (row.block_order_id, row.trial_id, row.single_touch_id)
        parsed = parse_contact_points(cp_str)
        for pt in parsed:
            all_pts_mm.add(tuple(round(v / dedup_mm) * dedup_mm for v in pt))
            if row.Nerve_spike == 1:
                spike_counter[pt] += 1
                unique_touch_counter[pt].add(touch_key)

    if all_pts_mm:
        all_contacts_xyz = np.array(sorted(all_pts_mm), dtype=float)
    else:
        all_contacts_xyz = np.empty((0, 3))

    return spike_counter, unique_touch_counter, all_contacts_xyz


def _aggregate_spike_counts(
    session_counters: List[Tuple[Counter, defaultdict]],
) -> pd.DataFrame:
    """Aggregate spike counters from multiple sessions into a DataFrame.

    Parameters
    ----------
    session_counters:
        List of (spike_counter, unique_touch_counter) pairs, one per session.
        spike_counter: Counter mapping (x, y, z) -> spike count.
        unique_touch_counter: DefaultDict mapping (x, y, z) -> set of unique touch keys.

    Returns
    -------
    DataFrame with columns (x, y, z, spike_count, unique_touch_spike_count),
    sorted by spike_count descending.
    """
    total_spike: Counter = Counter()
    total_unique: defaultdict = defaultdict(set)

    for spike_c, unique_c in session_counters:
        total_spike.update(spike_c)
        for pt, touch_set in unique_c.items():
            total_unique[pt].update(touch_set)

    if not total_spike:
        return pd.DataFrame(columns=['x', 'y', 'z', 'spike_count', 'unique_touch_spike_count'])

    rows = [
        {
            'x': pt[0],
            'y': pt[1],
            'z': pt[2],
            'spike_count': count,
            'unique_touch_spike_count': len(total_unique[pt]),
        }
        for pt, count in total_spike.items()
    ]
    df = pd.DataFrame(rows).sort_values('spike_count', ascending=False).reset_index(drop=True)
    return df


def _format_cluster_folder(cluster_label: str) -> str:
    try:
        n = int(cluster_label)
        return f'cluster_{n:02d}' if n >= 0 else 'cluster_noise'
    except (ValueError, TypeError):
        return f'cluster_{cluster_label}'


def _build_cluster_description(
    clustered_df: pd.DataFrame,
    cluster_label: str,
    metadata_json_path: Path = None,
    neuron_touches: Dict[str, int] = None,
    neuron_cluster_touches: Dict[str, int] = None,
    cluster_features: Optional[dict] = None,
) -> dict:
    rows = clustered_df[clustered_df['cluster_label'].astype(str) == str(cluster_label)]
    desc: dict = {'cluster_label': cluster_label, 'n_touches': int(len(rows))}

    if neuron_touches is not None:
        desc['neuron_touches'] = neuron_touches
    if neuron_cluster_touches is not None:
        desc['neuron_cluster_touches'] = neuron_cluster_touches

    if 'type_metadata' in rows.columns:
        vc = rows['type_metadata'].value_counts(normalize=True)
        desc['type_distribution'] = {str(k): round(float(v), 3) for k, v in vc.items()}

    if 'gesture_type' in rows.columns:
        vc = rows['gesture_type'].value_counts(normalize=True)
        desc['gesture_type_distribution'] = {str(k): round(float(v), 3) for k, v in vc.items()}

    feature_cols = [
        c for c in rows.columns
        if c not in SHARED_COLUMNS
        and c != 'cluster_label'
        and not c.startswith('bin_')
        and pd.api.types.is_numeric_dtype(rows[c])
    ]
    if feature_cols:
        ranges = {}
        for col in feature_cols:
            vals = rows[col].dropna()
            if len(vals) > 0:
                ranges[col] = {
                    'min': round(float(vals.min()), 2),
                    'max': round(float(vals.max()), 2),
                    'mean': round(float(vals.mean()), 2),
                }
        desc['feature_ranges'] = ranges

    if cluster_features is not None and feature_cols:
        display_ranges: dict = {}
        for data_type, aggregations in cluster_features.items():
            if data_type == 'location':
                continue
            base_cols = DATA_TYPE_TO_COLUMNS.get(data_type, [data_type])
            for agg in aggregations:
                type_min = float('inf')
                type_max = float('-inf')
                found = False
                for base in base_cols:
                    col = f'{base}_{agg}'
                    if col in ranges:
                        type_min = min(type_min, ranges[col]['min'])
                        type_max = max(type_max, ranges[col]['max'])
                        found = True
                if found:
                    label = data_type if len(aggregations) == 1 else f'{data_type}_{agg}'
                    display_ranges[label] = {
                        'min': round(type_min, 2),
                        'max': round(type_max, 2),
                    }
        if display_ranges:
            desc['display_ranges'] = display_ranges

    if metadata_json_path and metadata_json_path.exists():
        try:
            with open(metadata_json_path) as f:
                meta = json.load(f)
            algo = meta.get('algorithm', 'unknown')

            if algo == 'type_stratified':
                label_str = str(cluster_label)
                sep = label_str.rfind('_')
                if sep != -1:
                    type_key = label_str[:sep]
                    idx_str = label_str[sep + 1:]
                    per_type_meta = meta.get('per_type', {}).get(type_key, {})
                    pf = per_type_meta.get('primary_feature')
                    if pf and per_type_meta.get('bin_edges'):
                        edges = per_type_meta['bin_edges'].get(pf, [])
                        try:
                            idx = int(idx_str)
                            if 0 <= idx < len(edges) - 1:
                                desc['bin_range'] = {
                                    'feature': pf,
                                    'low': round(edges[idx], 2),
                                    'high': round(edges[idx + 1], 2),
                                }
                        except (ValueError, IndexError):
                            pass
                    if pf:
                        desc['primary_feature'] = pf
            else:
                if meta.get('primary_feature'):
                    desc['primary_feature'] = meta['primary_feature']
                if algo == 'binning' and meta.get('bin_edges'):
                    pf = meta.get('primary_feature')
                    edges = meta['bin_edges'].get(pf, [])
                    try:
                        idx = int(cluster_label)
                        if 0 <= idx < len(edges) - 1:
                            desc['bin_range'] = {
                                'feature': pf,
                                'low': round(edges[idx], 2),
                                'high': round(edges[idx + 1], 2),
                            }
                    except (ValueError, IndexError):
                        pass

            generation = {'algorithm': algo}
            if algo == 'binning':
                generation['n_bins'] = meta.get('n_bins')
                generation['bin_method'] = meta.get('bin_method')
                generation['primary_feature'] = meta.get('primary_feature')
            elif algo == 'kmeans':
                generation['k'] = meta.get('k')
            elif algo == 'dbscan':
                generation['eps'] = meta.get('eps_used')
                generation['min_samples'] = meta.get('min_samples')
            elif algo == 'hierarchical':
                generation['k'] = meta.get('k')
            elif algo == 'type_stratified':
                generation['base_algorithm'] = meta.get('base_algorithm')
                label_str = str(cluster_label)
                sep = label_str.rfind('_')
                if sep != -1:
                    type_key = label_str[:sep]
                    generation['type'] = type_key
                    per_type = meta.get('per_type', {}).get(type_key, {})
                    base = meta.get('base_algorithm')
                    if base == 'binning':
                        generation['n_bins'] = per_type.get('n_bins')
                        generation['primary_feature'] = per_type.get('primary_feature')
                    elif base == 'kmeans':
                        generation['k'] = per_type.get('k')
            elif algo == 'gmm':
                generation['k'] = meta.get('k')
                generation['covariance_type'] = meta.get('covariance_type')
                retained = meta.get('reduction', {}).get('retained_columns', [])
                if retained:
                    generation['features'] = retained
            desc['generation_params'] = generation
        except Exception:
            pass

    return desc


def _build_pairs(
    cluster_groups,
    cluster_group_defs,
    feature_combinations,
    clustering_profiles,
    caller: str = "_build_pairs",
) -> List[Tuple[str, str]]:
    """Resolve (combo_name, clusterer_name) pairs from either schema."""
    pairs: List[Tuple[str, str]] = []
    if cluster_groups is not None:
        if cluster_group_defs is None:
            raise ValueError(
                f"{caller}: 'cluster_group_defs' must be provided when using 'cluster_groups'."
            )
        missing = [g for g in cluster_groups if g not in cluster_group_defs]
        if missing:
            raise ValueError(
                f"{caller}: group name(s) {missing} not found in cluster_group_defs."
            )
        for group_name in cluster_groups:
            group_spec = cluster_group_defs[group_name]
            for clusterer_name in filter_enabled_profiles(group_spec.get('clustering_methods', {})):
                pairs.append((group_name, clusterer_name))
    else:
        logger.warning(
            "%s: 'feature_combinations' is deprecated — migrate to 'cluster_groups'.", caller
        )
        enabled_combinations = filter_enabled_profiles(feature_combinations)
        enabled_clusterers = filter_enabled_profiles(clustering_profiles or {})
        for combo_name in enabled_combinations:
            for clusterer_name in enabled_clusterers:
                pairs.append((combo_name, clusterer_name))
    return pairs


def _resolve_explorer_session_paths(
    input_items: List[Tuple[Path, Path]],
) -> List[Tuple[str, Path, Path]]:
    """Return ``(session_id, series_csv_path, forearm_ply_path)`` for each item."""
    result = []
    for csv_path, database_path in input_items:
        session_id = session_id_from_path(csv_path)
        series_csv_path = (
            database_path / '4_analysed' / TOUCH_COMPUTE_SERIES
            / f'{session_id}_series_augmented.csv'
        )
        if not series_csv_path.exists():
            raise ValueError(
                f"_resolve_explorer_session_paths: series-augmented CSV not found "
                f"for session '{session_id}': {series_csv_path}"
            )
        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise ValueError(
                f"_resolve_explorer_session_paths: forearm PLY not found "
                f"for session '{session_id}' in {csv_path.parent}"
            )
        result.append((session_id, series_csv_path, forearm_ply_path))
    return result


def run_cluster_rf_extraction(
    clustering_dir: Path,
    input_items: List[Tuple[Path, Path]],
    output_dir: Path,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    force: bool = False,
) -> List[Path]:
    """Extract spike data from aggregated CSVs grouped by cluster labels.

    Reads pooled_touch_summary_clustered.csv, traces each touch to its
    session's aggregated CSV, forward-fills contact_points, counts spikes per
    (x, y, z) point, and writes intermediate artifacts to output_dir.
    Does not render heatmaps or compute RF metrics.

    Parameters
    ----------
    clustering_dir:
        Root of touch_clusters output.
    input_items:
        List of (aggregated_csv_path, database_path) tuples.
    output_dir:
        Root of RF cluster artifact directory.
    cluster_groups:
        List of group names (new schema). Must accompany *cluster_group_defs*.
    cluster_group_defs:
        Dict mapping group_name -> group_spec.
    feature_combinations:
        **DEPRECATED.**
    clustering_profiles:
        **DEPRECATED.**
    force:
        If True, reprocess even if outputs are up-to-date.

    Returns
    -------
    List of paths to produced spike_counts.csv files.
    """
    if cluster_groups is None and feature_combinations is None:
        raise ValueError(
            "run_cluster_rf_extraction: either 'cluster_groups' or 'feature_combinations' must be provided."
        )

    session_dir_map = _resolve_session_paths(input_items)
    pairs = _build_pairs(
        cluster_groups, cluster_group_defs, feature_combinations, clustering_profiles,
        caller="run_cluster_rf_extraction",
    )
    produced: List[Path] = []

    for combo_name, clusterer_name in pairs:
        group_spec = (cluster_group_defs or {}).get(combo_name, {})
        per_type = group_spec.get('per_type_clustering', False)
        if per_type:
            runs = [
                (
                    gt,
                    clustering_dir / combo_name / clusterer_name / gt
                    / 'pooled_touch_summary_clustered.csv',
                    output_dir / combo_name / clusterer_name / gt,
                    clustering_dir / combo_name / clusterer_name / gt / 'cluster_metadata.json',
                )
                for gt in GESTURE_TYPES
            ]
        else:
            runs = [(
                None,
                clustering_dir / combo_name / clusterer_name
                / 'pooled_touch_summary_clustered.csv',
                output_dir / combo_name / clusterer_name,
                clustering_dir / combo_name / clusterer_name / 'cluster_metadata.json',
            )]

        for gesture_type, clustered_csv, base_output, cluster_metadata_path in runs:
            _type_label = f"/{gesture_type}" if gesture_type else ""
            print(f"[RF Extraction] {combo_name}/{clusterer_name}{_type_label}...")
            if not clustered_csv.exists():
                print(f"  Clustered CSV not found, skipping.")
                logger.warning(
                    "Clustered CSV not found, skipping (%s/%s%s): %s",
                    combo_name, clusterer_name, _type_label, clustered_csv,
                )
                continue

            extraction_json = base_output / 'extraction_summary.json'

            if not should_process_task(
                input_paths=[clustered_csv],
                output_paths=[extraction_json],
                force=force,
            ):
                print(f"  Extraction up-to-date, skipping.")
                for cluster_dir in sorted(base_output.glob('cluster_*')):
                    sc = cluster_dir / 'spike_counts.csv'
                    if sc.exists():
                        produced.append(sc)
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

            clustered_by_label = _group_touches_by_cluster(clustered_df)
            n_clusters = len(clustered_by_label)
            print(f"  {n_clusters} clusters found.")

            neuron_touches: Dict[str, int] = {
                str(k): int(v)
                for k, v in clustered_df.groupby('session_id').size().items()
            }

            session_dfs: Dict[str, pd.DataFrame] = {}
            neuron_contacts_xyz: Dict[str, np.ndarray] = {}
            sessions_metadata: Dict[str, dict] = {}

            for sid in list(neuron_touches.keys()):
                if sid not in session_dir_map:
                    logger.warning(
                        "Session '%s' in clustered_df not found in input_items. Skipping.", sid
                    )
                    continue
                _session_dir = session_dir_map[sid]
                _agg_csvs = list(_session_dir.glob('*_semicontrolled_aggregated_session.csv'))
                if not _agg_csvs:
                    logger.warning(
                        "No aggregated CSV in %s. Skipping session %s.", _session_dir, sid
                    )
                    continue
                try:
                    _session_df = _load_session_aggregated(_agg_csvs[0])
                except ValueError as _exc:
                    logger.warning("Skipping session %s: %s", sid, _exc)
                    continue
                except Exception:
                    logger.exception("Error reading %s", _agg_csvs[0].name)
                    continue

                session_dfs[sid] = _session_df

                _all_session_keys = list(
                    clustered_df[clustered_df['session_id'].astype(str) == sid][_KEY_COLS]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                _, _, _neuron_xyz = _parse_contacts_for_keys(_session_df, touch_keys=_all_session_keys)
                if len(_neuron_xyz) == 0:
                    raise ValueError(
                        f"Session '{sid}': aggregated CSV yields zero parseable contact_points "
                        f"across all {len(_all_session_keys)} touches. Data is malformed."
                    )
                neuron_contacts_xyz[sid] = _neuron_xyz
                save_neuron_contacts(base_output, sid, _neuron_xyz)

                _ply_path = resolve_forearm_ply(_session_dir, sid)
                sessions_metadata[sid] = {'forearm_ply': str(_ply_path) if _ply_path else None}
                if _ply_path is not None:
                    try:
                        _verts = load_forearm_vertices(_ply_path)
                        if _verts is not None:
                            save_forearm_vertices(base_output, sid, _verts)
                    except Exception:
                        logger.warning("Failed to cache forearm vertices for session %s", sid)

            save_neuron_touches(base_output, neuron_touches)
            save_sessions_metadata(base_output, sessions_metadata)

            summary_data: dict = {}

            for cluster_idx, (cluster_label, cluster_df) in enumerate(clustered_by_label.items(), start=1):
                n_touches = len(cluster_df)
                print(f"  Cluster {cluster_idx}/{n_clusters} (label={cluster_label}, {n_touches} touches)...")
                cluster_out = base_output / _format_cluster_folder(cluster_label)
                cluster_out.mkdir(parents=True, exist_ok=True)

                neuron_cluster_touches: Dict[str, int] = {
                    str(k): int(v)
                    for k, v in (
                        clustered_df[clustered_df['cluster_label'].astype(str) == cluster_label]
                        .groupby('session_id')
                        .size()
                        .items()
                    )
                }

                _group_features = None
                if cluster_group_defs and combo_name in cluster_group_defs:
                    _group_features = cluster_group_defs[combo_name].get('features')
                cluster_desc = _build_cluster_description(
                    clustered_df,
                    cluster_label,
                    cluster_metadata_path,
                    neuron_touches=neuron_touches,
                    neuron_cluster_touches=neuron_cluster_touches,
                    cluster_features=_group_features,
                )
                with open(cluster_out / 'cluster_description.json', 'w') as _f:
                    json.dump(cluster_desc, _f, indent=2)

                session_touch_map = _build_session_touch_map(cluster_df)
                session_counters: List[Tuple[Counter, defaultdict]] = []

                for session_id, touch_keys in session_touch_map.items():
                    print(f"    {session_id}: loading {len(touch_keys)} touches...")
                    if session_id not in session_dfs:
                        print(f"    -> not loaded, skipping.")
                        logger.warning(
                            "Session '%s' in cluster %s: no loaded session df, skipping.",
                            session_id, cluster_label,
                        )
                        continue

                    spike_counter, session_unique_counter, cluster_xyz = _parse_contacts_for_keys(
                        session_dfs[session_id], touch_keys
                    )
                    n_spikes = sum(spike_counter.values())
                    print(f"    -> {n_spikes} spikes at {len(spike_counter)} contact points.")
                    session_counters.append((spike_counter, session_unique_counter))

                    if spike_counter:
                        session_spike_df = _aggregate_spike_counts(
                            [(spike_counter, session_unique_counter)]
                        )
                        save_cluster_session_data(cluster_out, session_id, session_spike_df, cluster_xyz)

                save_neuron_cluster_touches(cluster_out, neuron_cluster_touches)

                pooled_df = _aggregate_spike_counts(session_counters)
                spike_counts_csv = cluster_out / 'spike_counts.csv'
                pooled_df.to_csv(spike_counts_csv, index=False)
                produced.append(spike_counts_csv)

                total_spikes = int(pooled_df['spike_count'].sum()) if not pooled_df.empty else 0
                if pooled_df.empty:
                    print(f"  -> No spikes found for cluster {cluster_label}. spike_counts.csv is empty.")
                    logger.warning(
                        "No spikes found for cluster %s (%s/%s%s). spike_counts.csv is empty.",
                        cluster_label, combo_name, clusterer_name, _type_label,
                    )
                else:
                    print(
                        f"  -> spike_counts.csv: {total_spikes} total spikes,"
                        f" {len(pooled_df)} unique contact points."
                    )

                n_unique_touch_points = int((pooled_df['unique_touch_spike_count'] > 0).sum()) if not pooled_df.empty else 0
                summary_data[cluster_label] = {
                    'n_sessions': len(session_touch_map),
                    'n_touches': n_touches,
                    'total_spike_points': len(pooled_df),
                    'total_spikes': total_spikes,
                    'n_unique_touch_points': n_unique_touch_points,
                }

            # Sentinel written LAST (checked FIRST on next run)
            base_output.mkdir(parents=True, exist_ok=True)
            save_extraction_summary(base_output, combo_name, clusterer_name, summary_data)

            print(f"[RF Extraction] {combo_name}/{clusterer_name}{_type_label}: done ({n_clusters} clusters).")
            logger.info(
                "[%s/%s%s] RF extraction complete: %d clusters.",
                combo_name, clusterer_name, _type_label, n_clusters,
            )

    return produced


def run_cluster_rf_mapping(
    clustering_dir: Path,
    input_items: List[Tuple[Path, Path]],
    output_dir: Path,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    force: bool = False,
    projection_method: Optional[str] = None,
    disjoint_mask_distance_mm: float = 8.0,
) -> List[Path]:
    """Backward-compatible wrapper: calls extraction, metrics, and visualization sequentially.

    Calls ``run_cluster_rf_extraction()``, ``run_cluster_rf_metrics_computation()``,
    and ``run_cluster_rf_visualization()``, each with its own idempotency sentinel.
    Callers that previously used this function are unaffected.

    Returns the list of spike_counts.csv paths produced by the extraction step.
    """
    from .rf_cluster_metrics_pipeline import run_cluster_rf_metrics_computation
    from .rf_cluster_visualization_pipeline import run_cluster_rf_visualization

    result = run_cluster_rf_extraction(
        clustering_dir=clustering_dir,
        input_items=input_items,
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        force=force,
    )
    run_cluster_rf_metrics_computation(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        projection_method=projection_method,
        force=force,
    )
    run_cluster_rf_visualization(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        projection_method=projection_method,
        disjoint_mask_distance_mm=disjoint_mask_distance_mm,
        force=force,
    )
    return result

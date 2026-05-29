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
from .clustering.base import ClusteringContext
from .clustering.feature_space_renderer import render_gmm_feature_space
from .clustering.cartesian_binning_renderer import render_cartesian_bin_partition
from .reporting import VisualReportingStrategy
from .pipeline_shared import (
    SHARED_COLUMNS,
    _TqdmLineWrapper,
    filter_enabled_profiles,
    session_id_from_path,
)
from .reduction import ReductionPipeline
from .evaluation import compute_internal_metrics, bootstrap_stability
from analysis.pipeline.shared_constants import (
    GESTURE_TYPES,  # re-exported for backward compatibility
    TOUCH_ID_COLS_WITH_SESSION,
)

# Columns that uniquely identify a single touch across feature CSVs
# (backward-compat alias — use TOUCH_ID_COLS_WITH_SESSION from shared_constants)
_TOUCH_ID_COLS = list(TOUCH_ID_COLS_WITH_SESSION)

# Maps cluster-group data-type names to the column name prefix(es) written by
# StatisticalExtractor.  Column names match the raw input columns directly
# (e.g. contact_depth_mean, hand_velocity_x_max).
# Empty list means the type uses its own selection logic (location).
DATA_TYPE_TO_COLUMNS = {
    'contact_area':           ['contact_area'],
    'contact_depth':          ['contact_depth'],
    'hand_velocity':           ['hand_velocity_x', 'hand_velocity_y', 'hand_velocity_z'],
    'hand_velocity_amplitude': ['hand_velocity_amplitude'],
    'hand_acceleration':       ['hand_acceleration_x', 'hand_acceleration_y', 'hand_acceleration_z'],
    'pressure':               ['pressure'],
    'hand_position':          ['hand_position_x', 'hand_position_y', 'hand_position_z'],
    'mos_strain':             ['mos_strain'],
    'mos_stress_kpa':         ['mos_stress_kpa'],
    'mos_strain_rate':        ['mos_strain_rate'],
    'mos_elastic_energy_mj':  ['mos_elastic_energy_mj'],
    'mos_impulse_mns':        ['mos_impulse_mns'],
    'mechanics_of_solids':    ['mos_strain', 'mos_stress_kpa', 'mos_strain_rate',
                               'mos_elastic_energy_mj', 'mos_impulse_mns'],
    'location':               [],
}

# location:mean uses these shared columns (always present in every per-aggregation CSV)
_LOCATION_SHARED_COLS = ['mean_contact_x', 'mean_contact_y', 'mean_contact_z']
# location:non-mean uses these as base column names; extractor writes e.g. contact_location_x_std
_LOCATION_BASE_COLS = ['contact_location_x', 'contact_location_y', 'contact_location_z']


def _resolve_required_feature_folders(group_spec: dict) -> list[str]:
    """Return folder names for the given cluster group spec.

    ``folder_names`` is the list of per-aggregation folder names to load from
    ``touch_features/<folder>/``.  ``location: [mean]`` does NOT add ``mean`` to
    the folder list because mean location values are already present in every
    per-aggregation CSV as shared columns.
    """
    features: dict = group_spec.get('features', {})
    folders: list[str] = []

    for data_type, aggregations in features.items():
        if data_type == 'location':
            non_mean_aggs = [a for a in aggregations if a != 'mean']
            folders.extend(non_mean_aggs)
        else:
            folders.extend(aggregations)

    seen: set[str] = set()
    deduped: list[str] = []
    for f in folders:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def _select_feature_columns(
    merged_df: pd.DataFrame,
    group_spec: dict,
) -> list[str]:
    """Return the list of feature column names to pass to the clusterer.

    Selects ``{base_col}_{agg}`` columns from *merged_df* per the group spec.
    Emits a warning for any expected column that is missing.
    """
    features: dict = group_spec.get('features', {})
    selected: list[str] = []

    for data_type, aggregations in features.items():
        if data_type == 'location':
            for agg in aggregations:
                if agg == 'mean':
                    for col in _LOCATION_SHARED_COLS:
                        if col in merged_df.columns:
                            selected.append(col)
                        else:
                            logging.warning(
                                f"cluster_groups: expected location[mean] column '{col}' not found — skipping."
                            )
                else:
                    for base in _LOCATION_BASE_COLS:
                        col = f'{base}_{agg}'
                        if col in merged_df.columns:
                            selected.append(col)
                        else:
                            logging.warning(
                                f"cluster_groups: expected location[{agg}] column '{col}' not found — skipping."
                            )
            continue

        base_cols = DATA_TYPE_TO_COLUMNS.get(data_type)
        if base_cols is None:
            logging.warning(
                f"cluster_groups: unknown data type '{data_type}' — skipping."
            )
            continue

        for base in base_cols:
            for agg in aggregations:
                col = f'{base}_{agg}'
                if col in merged_df.columns:
                    selected.append(col)
                else:
                    logging.warning(
                        f"cluster_groups: expected column '{col}' not found — skipping."
                    )

    return selected


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
                f"run stimulus_extract_features with feature '{feature_name}' enabled."
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
    cluster_groups: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    force: bool = False,
    extraction_dir: Path = None,
    reduction: dict = None,
    evaluation: dict = None,
) -> dict[str, List[Path]]:
    """
    Merge feature CSVs per group and run each group's clustering profiles.

    Parameters
    ----------
    output_dir
        Root directory for clustering outputs
        (e.g. ``database / '4_analysed' / 'touch_clusters'``).
    cluster_groups
        Dict mapping group_name -> group_spec.  Each spec must contain:
        ``features: {data_type: [agg, ...]}`` and
        ``clustering_methods: {clusterer_name: config}``.
    feature_combinations
        **DEPRECATED.** Old format: dict of combination_name -> config with
        ``features: [folder_name, ...]``.  Pass with *clustering_profiles*.
        A deprecation warning is emitted and the old code path runs unchanged.
    clustering_profiles
        **DEPRECATED.** Companion to *feature_combinations*.
    force
        Override idempotency checks.
    extraction_dir
        Root of the extraction output tree. Defaults to *output_dir* when not
        provided (backward-compatible).
    reduction
        Task-level reduction config injected into each group/combination as a
        default (a per-group key wins).
    evaluation
        Task-level evaluation config injected into each group/combination as a
        default (a per-group key wins).

    Returns
    -------
    Dict mapping ``"<group_name>/<clusterer_name>"`` -> list of
    pooled-clustered CSV paths written.
    """
    if cluster_groups is None and feature_combinations is None:
        raise ValueError(
            "run_clustering: either 'cluster_groups' or 'feature_combinations' must be provided."
        )

    src_dir = extraction_dir if extraction_dir is not None else output_dir
    results: dict[str, List[Path]] = {}

    # ---- New code path: cluster_groups ----------------------------------------
    if cluster_groups is not None:
        cluster_groups = filter_enabled_profiles(cluster_groups)
        n_groups = len(cluster_groups)
        print(
            f"=== clustering pipeline: {n_groups} cluster group(s) ===",
            flush=True,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            for group_name, group_spec in cluster_groups.items():
                group_spec = {**group_spec}
                if reduction is not None and "reduction" not in group_spec:
                    group_spec["reduction"] = reduction
                if evaluation is not None and "evaluation" not in group_spec:
                    group_spec["evaluation"] = evaluation

                feature_names_to_load = _resolve_required_feature_folders(group_spec)

                if not feature_names_to_load:
                    raise ValueError(
                        f"[{group_name}] No feature folders resolved from group spec. "
                        "A group with only 'location: [mean]' cannot be loaded — "
                        "add at least one non-mean-location feature or aggregation."
                    )

                print(
                    f"  [cluster] group '{group_name}' — folders: {feature_names_to_load}",
                    flush=True,
                )

                pooled = _merge_feature_csvs(feature_names_to_load, src_dir)
                if pooled.empty:
                    logging.warning(
                        f"[{group_name}] No data after merging feature CSVs — skipping."
                    )
                    continue

                print(
                    f"  [cluster] group '{group_name}' — "
                    f"{len(pooled)} touches from "
                    f"{pooled['session_id'].nunique() if 'session_id' in pooled.columns else '?'} "
                    "session(s)",
                    flush=True,
                )

                feature_cols = _select_feature_columns(pooled, group_spec)

                group_clustering_methods = filter_enabled_profiles(
                    group_spec.get('clustering_methods', {})
                )
                if not group_clustering_methods:
                    logging.warning(
                        f"[{group_name}] No clustering methods enabled — skipping."
                    )
                    continue

                cluster_outputs = _cluster_combination(
                    combination_name=group_name,
                    combination_config=group_spec,
                    pooled=pooled,
                    output_dir=output_dir,
                    clustering_profiles=group_clustering_methods,
                    force=force,
                    feature_cols_override=feature_cols,
                )
                for clusterer_name, paths in cluster_outputs.items():
                    results[f"{group_name}/{clusterer_name}"] = paths

        total = sum(len(v) for v in results.values())
        print(f"=== clustering pipeline complete: {total} output(s) ===", flush=True)
        return results

    # ---- Deprecated code path: feature_combinations --------------------------
    logging.warning(
        "clustering_pipeline: 'feature_combinations' is deprecated — "
        "migrate to 'cluster_groups' with per-group 'clustering_methods'."
    )

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
    clustering_profiles = filter_enabled_profiles(clustering_profiles or {})

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
            combination_config = {**combination_config}
            if reduction is not None and "reduction" not in combination_config:
                combination_config["reduction"] = reduction
            if evaluation is not None and "evaluation" not in combination_config:
                combination_config["evaluation"] = evaluation

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
                f"{len(pooled)} touches from "
                f"{pooled['session_id'].nunique() if 'session_id' in pooled.columns else '?'} session(s)",
                flush=True,
            )

            cluster_outputs = _cluster_combination(
                combination_name=combination_name,
                combination_config=combination_config,
                pooled=pooled,
                output_dir=output_dir,
                clustering_profiles=clustering_profiles,
                force=force,
            )
            for clusterer_name, paths in cluster_outputs.items():
                results[f"{combination_name}/{clusterer_name}"] = paths

    total = sum(len(v) for v in results.values())
    print(f"=== clustering pipeline complete: {total} output(s) ===", flush=True)
    return results


def _run_reduction_and_clusterers(
    combination_name: str,
    combination_config: dict,
    pooled: pd.DataFrame,
    out_base: Path,
    clustering_profiles: dict,
    force: bool,
    feature_cols: list[str],
    gesture_type: str | None,
) -> dict[str, List[Path]]:
    """Run reduction + every clusterer on *pooled*, writing outputs under *out_base*.

    Parameters
    ----------
    combination_name
        Group/combination name used in log messages.
    combination_config
        Full group/combination config dict (reduction, evaluation, visualization …).
    pooled
        Feature DataFrame to cluster (may be the full pool or a type-filtered subset).
    out_base
        ``output_dir / combination_name``; each clusterer appends its own name,
        then optionally *gesture_type*.
    clustering_profiles
        Enabled clusterer name → config dict.
    force
        Override idempotency checks.
    feature_cols
        Feature column names selected from the full pooled DataFrame.
    gesture_type
        When ``None``, outputs go to ``out_base / clusterer_name``.
        When a string, outputs go to ``out_base / clusterer_name / gesture_type``.
    """
    outputs: dict[str, List[Path]] = {}

    feature_cols = [c for c in feature_cols if c in pooled.columns]

    feature_df_full = pooled[feature_cols].dropna() if feature_cols else pd.DataFrame()
    valid_idx = feature_df_full.index

    reduction_meta: dict = {}
    X_scaled: np.ndarray = np.empty((0, 0))

    if not feature_cols:
        logging.warning(
            f"[{combination_name}] No numeric feature columns found — "
            "all clusterers will be skipped."
        )
    else:
        rp = ReductionPipeline()
        X_scaled, reduction_meta = rp.fit_transform(feature_df_full, combination_config)

    for clusterer_name, clusterer_config in clustering_profiles.items():
        method = clusterer_config.get('method', clusterer_name)
        if gesture_type is None:
            out_dir = out_base / clusterer_name
        else:
            out_dir = out_base / clusterer_name / gesture_type
        out_dir.mkdir(parents=True, exist_ok=True)

        pooled_csv = out_dir / 'pooled_touch_summary_clustered.csv'
        metadata_json = out_dir / 'cluster_metadata.json'
        feature_space_png = out_dir / 'feature_space.png'

        # Idempotency: use session CSVs as conceptual inputs — skip if up to date
        _IMAGE_GENERATING_METHODS = {'gmm', 'cartesian_binning'}
        if not force and pooled_csv.exists():
            try:
                if not should_process_task(
                    input_paths=[pooled_csv],  # approximate: use output mtime
                    output_paths=[pooled_csv],
                    force=False,
                ):
                    if method in _IMAGE_GENERATING_METHODS and not feature_space_png.exists():
                        pass  # fall through: re-run to generate the missing feature-space image
                    else:
                        outputs.setdefault(clusterer_name, []).append(pooled_csv)
                        print(
                            f"  [cluster] {combination_name} / {clusterer_name} — up to date",
                            flush=True,
                        )
                        continue
            except FileNotFoundError:
                pass
        clean_task_outputs([pooled_csv, metadata_json, feature_space_png])

        if not feature_cols:
            print(
                f"  [cluster] {combination_name} / {clusterer_name} — error: no numeric features",
                flush=True,
            )
            continue

        try:
            clusterer = get_clusterer(method)
        except KeyError as exc:
            logging.error(f"Clustering profile '{clusterer_name}': {exc}")
            print(
                f"  [cluster] {combination_name} / {clusterer_name} — error: unknown clusterer",
                flush=True,
            )
            continue

        # Wrap scaled array back into a DataFrame so clusterers that read
        # .values still work, and pass the same retained-column names.
        retained_cols = reduction_meta.get("retained_columns", feature_cols)
        scaled_df = pd.DataFrame(X_scaled, index=feature_df_full.index, columns=retained_cols)

        # Build ClusteringContext with runtime arrays for clusterers that need them
        sensor_col = clusterer_config.get('sensor_col')
        context = ClusteringContext(
            sensor_labels=pooled.loc[valid_idx, sensor_col].to_numpy() if sensor_col and sensor_col in pooled.columns else None,
            gesture_type_labels=pooled.loc[valid_idx, 'gesture_type'].to_numpy() if 'gesture_type' in pooled.columns else None,
        )

        if method == 'gmm':
            max_k = min(
                clusterer_config.get('max_components', 15),
                len(scaled_df) // clusterer_config.get('min_touches_per_component', 30),
            )
            print(
                f"  [gmm] starting: {combination_name} / {clusterer_name}"
                f"  ({len(scaled_df)} touches, {len(retained_cols)} features,"
                f" max_k={max_k})",
                flush=True,
            )

        try:
            labels, metadata = clusterer.fit_predict(scaled_df, clusterer_config, context)
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

        # --- Internal quality metrics --------------------------------------
        try:
            internal = compute_internal_metrics(X_scaled, labels)
            metadata['internal_metrics'] = internal
        except Exception as exc:
            _sessions = result_df['session_id'].unique().tolist() if 'session_id' in result_df.columns else 'N/A'
            _trials = sorted(result_df['trial_id'].unique().tolist()) if 'trial_id' in result_df.columns else 'N/A'
            _blocks = sorted(result_df['block_order_id'].unique().tolist()) if 'block_order_id' in result_df.columns else 'N/A'
            logging.warning(
                f"[{combination_name}/{clusterer_name}] Internal metrics failed: {exc} "
                f"| file={pooled_csv} | n_touches={len(result_df)} "
                f"| sessions={_sessions} | trials={_trials} | blocks={_blocks}"
            )

        # --- Stability (bootstrap) -----------------------------------------
        eval_cfg: dict = combination_config.get('evaluation', {}) if combination_config else {}
        stability_cfg: dict = eval_cfg.get('stability', {})
        n_rounds: int = stability_cfg.get('n_rounds', 20)
        subsample_fraction: float = stability_cfg.get('subsample_fraction', 0.8)
        try:
            stability = bootstrap_stability(
                clusterer,
                X_scaled,
                clusterer_config,
                n_rounds=n_rounds,
                subsample_fraction=subsample_fraction,
                context=context,
            )
            metadata['stability'] = stability
        except Exception as exc:
            _sessions = result_df['session_id'].unique().tolist() if 'session_id' in result_df.columns else 'N/A'
            _trials = sorted(result_df['trial_id'].unique().tolist()) if 'trial_id' in result_df.columns else 'N/A'
            _blocks = sorted(result_df['block_order_id'].unique().tolist()) if 'block_order_id' in result_df.columns else 'N/A'
            logging.warning(
                f"[{combination_name}/{clusterer_name}] Stability estimation failed: {exc} "
                f"| file={pooled_csv} | n_touches={len(result_df)} "
                f"| sessions={_sessions} | trials={_trials} | blocks={_blocks}"
            )

        # --- Reduction metadata -------------------------------------------
        metadata['reduction'] = reduction_meta

        # --- GMM: inject scaler params for feature-space visualisation ------
        if metadata.get('algorithm') == 'gmm':
            scaler_mean = reduction_meta.get('scaler_mean')
            scaler_scale = reduction_meta.get('scaler_scale')
            if scaler_mean is None or scaler_scale is None:
                raise ValueError(
                    f"[{combination_name}/{clusterer_name}] GMM feature-space "
                    "visualisation requires StandardScaler, but scaler="
                    f"'{reduction_meta.get('scaler')}' does not provide per-axis "
                    "mean/scale. Set `reduction.scaler: standard` in the cluster "
                    "group config."
                )
            metadata['scaler_mean'] = scaler_mean
            metadata['scaler_scale'] = scaler_scale
            metadata['retained_columns'] = reduction_meta['retained_columns']

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

        # --- GMM: render feature-space PNG ----------------------------------
        if metadata.get('algorithm') == 'gmm':
            vis_cfg: dict = (combination_config or {}).get('visualization', {})
            x_feat: str = vis_cfg.get('x_feature', 'pressure_mean')
            retained = metadata['retained_columns']
            default_y = next(
                (c for c in retained if c.startswith('hand_velocity_') and c.endswith('_mean')),
                retained[1] if len(retained) > 1 else retained[0],
            )
            y_feat: str = vis_cfg.get('y_feature', default_y)
            for feat_name in (x_feat, y_feat):
                if feat_name not in retained:
                    raise KeyError(
                        f"[{combination_name}/{clusterer_name}] visualization feature "
                        f"'{feat_name}' not in retained_columns {retained}."
                    )
            try:
                render_gmm_feature_space(
                    result_df=result_df,
                    metadata=metadata,
                    x_feature=x_feat,
                    y_feature=y_feat,
                    output_path=feature_space_png,
                )
                print(
                    f"  [feature_space] {combination_name} / {clusterer_name} — "
                    f"{x_feat} × {y_feat}",
                    flush=True,
                )
            except Exception as exc:
                logging.error(
                    f"[{combination_name}/{clusterer_name}] Feature-space render failed: {exc}"
                )

        # --- Cartesian binning: render feature-space PNG ---------------------
        elif metadata.get('algorithm') == 'cartesian_binning':
            binned = metadata.get('binned_features', [])
            if len(binned) >= 2:
                vis_cfg: dict = (combination_config or {}).get('visualization', {})
                default_x = 'pressure_mean' if 'pressure_mean' in binned else binned[0]
                x_feat: str = vis_cfg.get('x_feature', default_x)
                default_y = next(
                    (c for c in binned if 'velocity' in c),
                    binned[1],
                )
                y_feat: str = vis_cfg.get('y_feature', default_y)
                for feat_name in (x_feat, y_feat):
                    if feat_name not in metadata['bin_edges']:
                        raise KeyError(
                            f"[{combination_name}/{clusterer_name}] visualization feature "
                            f"'{feat_name}' not in bin_edges {list(metadata['bin_edges'])}."
                        )
                try:
                    render_cartesian_bin_partition(
                        result_df=result_df,
                        metadata=metadata,
                        x_feature=x_feat,
                        y_feature=y_feat,
                        output_path=feature_space_png,
                    )
                    print(
                        f"  [feature_space] {combination_name} / {clusterer_name} — "
                        f"{x_feat} × {y_feat}",
                        flush=True,
                    )
                except Exception as exc:
                    logging.error(
                        f"[{combination_name}/{clusterer_name}] Cartesian bin render failed: {exc}"
                    )
                    print(
                        f"  [feature_space] {combination_name} / {clusterer_name} — "
                        f"ERROR: {exc}",
                        flush=True,
                    )

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


def _cluster_combination(
    combination_name: str,
    combination_config: dict,
    pooled: pd.DataFrame,
    output_dir: Path,
    clustering_profiles: dict,
    force: bool,
    feature_cols_override: list[str] | None = None,
) -> dict[str, List[Path]]:
    """Run each clusterer on the already-pooled *pooled* DataFrame.

    Parameters
    ----------
    feature_cols_override
        When provided, use this column list instead of auto-discovering all
        numeric non-shared columns.  Used by the cluster_groups code path to
        restrict features to exactly those requested in the group spec.
    """
    if feature_cols_override is not None:
        feature_cols = [c for c in feature_cols_override if c in pooled.columns]
    else:
        feature_cols = [
            c for c in pooled.columns
            if c not in SHARED_COLUMNS
            and pd.api.types.is_numeric_dtype(pooled[c])
        ]

    per_type: bool = combination_config.get('per_type_clustering', False)

    if per_type:
        outputs: dict[str, List[Path]] = {}
        for gesture_type in GESTURE_TYPES:
            type_pooled = pooled[pooled['gesture_type'] == gesture_type]
            if type_pooled.empty:
                logging.warning(
                    f"[{combination_name}] No touches for type '{gesture_type}' — skipping."
                )
                continue
            type_outputs = _run_reduction_and_clusterers(
                combination_name=combination_name,
                combination_config=combination_config,
                pooled=type_pooled,
                out_base=output_dir / combination_name,
                clustering_profiles=clustering_profiles,
                force=force,
                feature_cols=feature_cols,
                gesture_type=gesture_type,
            )
            for clusterer_name, paths in type_outputs.items():
                outputs.setdefault(clusterer_name, []).extend(paths)
        return outputs

    return _run_reduction_and_clusterers(
        combination_name=combination_name,
        combination_config=combination_config,
        pooled=pooled,
        out_base=output_dir / combination_name,
        clustering_profiles=clustering_profiles,
        force=force,
        feature_cols=feature_cols,
        gesture_type=None,
    )


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
        for i_type in ('tap', 'stroke_proximal', 'stroke_distal'):
            subset = session_df[session_df['gesture_type'] == i_type]
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
                type_col='gesture_type',
                num_bins=num_bins,
                log_axis=True,
                title_suffix=str(session_id),
                filename=f"{session_id}_touch_density.png",
                global_edges=global_edges,
                global_max_count=global_max_count,
            )
        except Exception as exc:
            logging.error(f"Heatmap generation failed for session {session_id}: {exc}")

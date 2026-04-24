# comparing_pipeline.py
"""
Standalone comparing pipeline (Stage 3).

Discovers clustered CSVs and metadata written by clustering_pipeline.py,
applies coverage checks, runs enabled comparison strategies, and writes
per-stratum results plus a dispersion-weighted synthesis report.

Output layout
-------------
<output_dir>/
  <group_name>/
    <clusterer_name>/
      <strategy>_results.json    (per-stratum test results)
      synthesis_report.json      (dispersion-weighted global summary)
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

from .comparing import get_comparator
from .comparing.synthesis import synthesize_across_strata
from .pipeline_shared import filter_enabled_profiles


def _translate_extraction_profiles_to_combinations(extraction_profiles: dict) -> dict:
    """
    Translate old ``extraction_profiles`` format to ``feature_combinations``.

    Each old profile becomes a combination named after itself, used purely
    for directory discovery (the comparing pipeline does not re-merge features).
    """
    combinations: dict = {}
    for profile_name, profile_config in extraction_profiles.items():
        enabled = profile_config.get('enabled', True)
        combinations[profile_name] = {
            'enabled': enabled,
            'features': [profile_name],
        }
    return combinations


def _compare_one_clusterer(
    combo_label: str,
    clusterer_name: str,
    cluster_src: Path,
    output_dir: Path,
    comparing_profiles: dict,
    min_instances_per_sensor: int,
    min_sensor_types: int,
    force: bool,
    outputs: list,
) -> None:
    """Run all comparing strategies for one (combo_label, clusterer_name) pair."""
    cluster_dir = cluster_src / combo_label / clusterer_name
    pooled_csv = cluster_dir / 'pooled_touch_summary_clustered.csv'
    metadata_json = cluster_dir / 'cluster_metadata.json'

    if not pooled_csv.exists():
        logging.warning(
            f"[{combo_label}/{clusterer_name}] Clustered CSV not found: {pooled_csv}. "
            "Run touch_clustering first."
        )
        return

    comparisons_dir = output_dir / combo_label / clusterer_name
    synthesis_path = comparisons_dir / 'synthesis_report.json'

    if not force and synthesis_path.exists():
        print(
            f"  [compare] {combo_label} / {clusterer_name} — up to date",
            flush=True,
        )
        outputs.append(synthesis_path)
        return

    comparisons_dir.mkdir(parents=True, exist_ok=True)

    try:
        pooled_df = pd.read_csv(pooled_csv)
    except Exception as exc:
        logging.error(f"Failed to load {pooled_csv}: {exc}")
        return

    dispersion_map: dict[int, float] = {}
    if metadata_json.exists():
        try:
            with open(metadata_json) as f:
                meta = json.load(f)
            dispersion_map = {
                int(k): float(v)
                for k, v in meta.get('dispersion_per_stratum', {}).items()
            }
        except Exception as exc:
            logging.warning(f"Could not load cluster metadata: {exc}")

    if 'cluster_label' not in pooled_df.columns:
        logging.error(
            f"[{combo_label}/{clusterer_name}] 'cluster_label' column missing."
        )
        return

    all_results = []
    strategy_results: dict[str, list] = {p: [] for p in comparing_profiles}

    valid_strata = [
        lbl for lbl in pooled_df['cluster_label'].unique()
        if lbl >= 0
    ]

    for stratum_label in sorted(valid_strata):
        stratum_df = pooled_df[pooled_df['cluster_label'] == stratum_label].copy()
        dispersion = dispersion_map.get(stratum_label, 0.0)

        sensor_col_candidates = ['session_id']
        sensor_col = next(
            (c for c in sensor_col_candidates if c in stratum_df.columns),
            None,
        )
        if sensor_col is None:
            logging.warning(
                f"[{combo_label}/{clusterer_name}] stratum {stratum_label}: "
                "no sensor column found — skipping."
            )
            continue

        if not _satisfies_coverage(
            stratum_df, sensor_col, min_instances_per_sensor, min_sensor_types
        ):
            logging.info(
                f"  [compare] {combo_label}/{clusterer_name} stratum "
                f"{stratum_label} — non-exploitable (coverage)"
            )
            continue

        for profile_name, profile_config in comparing_profiles.items():
            method = profile_config.get('method', profile_name)
            meas_col = profile_config.get('measurement_col', 'spike_elicited')
            s_col = profile_config.get('sensor_col', sensor_col)

            if meas_col not in stratum_df.columns:
                logging.warning(
                    f"[{combo_label}/{clusterer_name}] "
                    f"measurement column '{meas_col}' missing — skipping."
                )
                continue

            try:
                comparator = get_comparator(method)
                result = comparator.compare(
                    stratum_df=stratum_df,
                    sensor_col=s_col,
                    measurement_col=meas_col,
                    config=profile_config,
                )
                result.stratum_dispersion = dispersion
            except Exception as exc:
                logging.error(
                    f"[{combo_label}/{clusterer_name}] strategy '{profile_name}' "
                    f"stratum {stratum_label} failed: {exc}"
                )
                continue

            strategy_results[profile_name].append(result)
            all_results.append(result)

    for profile_name, results in strategy_results.items():
        if not results:
            continue
        out_path = comparisons_dir / f"{profile_name}_results.json"
        try:
            with open(out_path, 'w') as f:
                json.dump(
                    [_result_to_dict(r) for r in results],
                    f, indent=2, default=str,
                )
            outputs.append(out_path)
        except Exception as exc:
            logging.error(f"Failed to write {out_path}: {exc}")

    if all_results:
        try:
            synthesis_df = synthesize_across_strata(all_results)
            synthesis_records = synthesis_df.to_dict(orient='records')
            with open(synthesis_path, 'w') as f:
                json.dump(synthesis_records, f, indent=2, default=str)
            outputs.append(synthesis_path)
            print(
                f"  [compare] {combo_label} / {clusterer_name} — "
                f"{len(all_results)} result(s), synthesis written",
                flush=True,
            )
        except Exception as exc:
            logging.error(f"Failed to write synthesis report: {exc}")
    else:
        print(
            f"  [compare] {combo_label} / {clusterer_name} — "
            "no exploitable strata",
            flush=True,
        )


def run_comparing(
    output_dir: Path,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    comparing_profiles: dict = None,
    min_instances_per_sensor: int = 5,
    min_sensor_types: int = 2,
    force: bool = False,
    clustering_dir: Path = None,
) -> List[Path]:
    """
    Discover clustered CSVs and run all configured comparison strategies.

    Parameters
    ----------
    output_dir
        Root directory for comparison outputs.
    cluster_groups
        List of group names to compare (new schema).  Must accompany
        *cluster_group_defs* which supplies the full group specs.
    cluster_group_defs
        Dict mapping group_name -> group_spec (from touch_clustering config).
        Required when *cluster_groups* is provided.
    feature_combinations
        **DEPRECATED.** Dict of combination_name -> config for directory discovery.
    clustering_profiles
        **DEPRECATED.** Companion to *feature_combinations*.
    comparing_profiles
        Dict mapping strategy_name -> strategy_config.
    min_instances_per_sensor
        Minimum observations per sensor per stratum.
    min_sensor_types
        Minimum distinct sensors per stratum.
    force
        Rewrite existing outputs.
    clustering_dir
        Root where clustering outputs live; defaults to *output_dir*.

    Returns
    -------
    List of output JSON paths written.
    """
    if cluster_groups is None and feature_combinations is None:
        raise ValueError(
            "run_comparing: either 'cluster_groups' or 'feature_combinations' must be provided."
        )

    comparing_profiles = filter_enabled_profiles(comparing_profiles or {})
    cluster_src = clustering_dir if clustering_dir is not None else output_dir
    outputs: List[Path] = []

    ctx = dict(
        cluster_src=cluster_src,
        output_dir=output_dir,
        comparing_profiles=comparing_profiles,
        min_instances_per_sensor=min_instances_per_sensor,
        min_sensor_types=min_sensor_types,
        force=force,
        outputs=outputs,
    )

    # ---- New code path: cluster_groups ----------------------------------------
    if cluster_groups is not None:
        if cluster_group_defs is None:
            raise ValueError(
                "run_comparing: 'cluster_group_defs' must be provided when using 'cluster_groups'."
            )
        missing = [g for g in cluster_groups if g not in cluster_group_defs]
        if missing:
            raise ValueError(
                f"run_comparing: group name(s) {missing} not found in cluster_group_defs."
            )

        print(
            f"=== comparing pipeline: {len(cluster_groups)} group(s), "
            f"{len(comparing_profiles)} strategy(ies) ===",
            flush=True,
        )

        for group_name in cluster_groups:
            group_spec = cluster_group_defs[group_name]
            group_clustering_methods = filter_enabled_profiles(
                group_spec.get('clustering_methods', {})
            )
            for clusterer_name in group_clustering_methods:
                _compare_one_clusterer(group_name, clusterer_name, **ctx)

        total = len(outputs)
        print(f"=== comparing pipeline complete: {total} output(s) ===", flush=True)
        return outputs

    # ---- Deprecated code path: feature_combinations --------------------------
    logging.warning(
        "comparing_pipeline: 'feature_combinations' is deprecated — "
        "migrate to 'cluster_groups' with 'cluster_group_defs'."
    )

    if feature_combinations and not any(
        'features' in v for v in feature_combinations.values() if isinstance(v, dict)
    ):
        logging.warning(
            "comparing_pipeline: 'extraction_profiles' format detected — "
            "translating to new 'feature_combinations' format automatically."
        )
        feature_combinations = _translate_extraction_profiles_to_combinations(feature_combinations)

    feature_combinations = filter_enabled_profiles(feature_combinations)
    clustering_profiles = filter_enabled_profiles(clustering_profiles or {})

    print(
        f"=== comparing pipeline: {len(feature_combinations)} combination(s), "
        f"{len(clustering_profiles)} clustering profile(s), "
        f"{len(comparing_profiles)} strategy(ies) ===",
        flush=True,
    )

    for combination_name in feature_combinations:
        for clusterer_name in clustering_profiles:
            _compare_one_clusterer(combination_name, clusterer_name, **ctx)

    total = len(outputs)
    print(f"=== comparing pipeline complete: {total} output(s) ===", flush=True)
    return outputs


def _satisfies_coverage(
    df: pd.DataFrame,
    sensor_col: str,
    min_instances_per_sensor: int,
    min_sensor_types: int,
) -> bool:
    """Return True if the stratum passes the coverage constraint."""
    counts = df[sensor_col].value_counts()
    valid = (counts >= min_instances_per_sensor).sum()
    return int(valid) >= min_sensor_types


def _result_to_dict(result) -> dict:
    d = asdict(result)
    for key in ('test_statistic', 'p_value', 'effect_size'):
        val = d.get(key)
        if val is not None:
            try:
                import math
                if math.isnan(val):
                    d[key] = None
            except (TypeError, ValueError):
                pass
    return d

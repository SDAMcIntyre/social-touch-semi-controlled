# comparing_pipeline.py
"""
Standalone comparing pipeline (Stage 3).

Discovers clustered CSVs and metadata written by clustering_pipeline.py,
applies coverage checks, runs enabled comparison strategies, and writes
per-stratum results plus a dispersion-weighted synthesis report.

Output layout
-------------
<output_dir>/
  <extraction_profile>/
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


def run_comparing(
    output_dir: Path,
    extraction_profiles: dict,
    clustering_profiles: dict,
    comparing_profiles: dict,
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
        Root directory for comparison outputs
        (e.g. ``database / '4_analysed' / 'touch_comparisons'``).
    clustering_dir
        Root directory where clustering outputs were written
        (e.g. ``database / '4_analysed' / 'touch_clusters'``).
        Defaults to *output_dir* when not provided (backward-compatible).
    extraction_profiles
        Only names are used for directory discovery; profiles with
        ``enabled: false`` are skipped.
    clustering_profiles
        Only names are used for directory discovery; profiles with
        ``enabled: false`` are skipped.
    comparing_profiles
        Dict mapping strategy_name -> strategy_config. Profiles with
        ``enabled: false`` are skipped.
    min_instances_per_sensor
        Minimum observations per sensor per stratum to be exploitable.
    min_sensor_types
        Minimum distinct sensors per stratum to be exploitable.
    force
        Rewrite existing comparison outputs.

    Returns
    -------
    List of output JSON paths written.
    """
    extraction_profiles = filter_enabled_profiles(extraction_profiles)
    clustering_profiles = filter_enabled_profiles(clustering_profiles)
    comparing_profiles = filter_enabled_profiles(comparing_profiles)

    cluster_src = clustering_dir if clustering_dir is not None else output_dir

    outputs: List[Path] = []

    print(
        f"=== comparing pipeline: {len(extraction_profiles)} extraction profile(s), "
        f"{len(clustering_profiles)} clustering profile(s), "
        f"{len(comparing_profiles)} strategy(ies) ===",
        flush=True,
    )

    for extraction_name in extraction_profiles:
        for clusterer_name in clustering_profiles:
            cluster_dir = cluster_src / extraction_name / clusterer_name
            pooled_csv = cluster_dir / 'pooled_touch_summary_clustered.csv'
            metadata_json = cluster_dir / 'cluster_metadata.json'

            if not pooled_csv.exists():
                logging.warning(
                    f"[{extraction_name}/{clusterer_name}] Clustered CSV not found: {pooled_csv}. "
                    "Run touch_clustering first."
                )
                continue

            comparisons_dir = output_dir / extraction_name / clusterer_name

            # Idempotency: skip if synthesis report exists and not force
            synthesis_path = comparisons_dir / 'synthesis_report.json'
            if not force and synthesis_path.exists():
                print(
                    f"  [compare] {extraction_name} / {clusterer_name} — up to date",
                    flush=True,
                )
                outputs.append(synthesis_path)
                continue

            comparisons_dir.mkdir(parents=True, exist_ok=True)

            try:
                pooled_df = pd.read_csv(pooled_csv)
            except Exception as exc:
                logging.error(f"Failed to load {pooled_csv}: {exc}")
                continue

            # Load dispersion per stratum from metadata if available
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
                    f"[{extraction_name}/{clusterer_name}] 'cluster_label' column missing."
                )
                continue

            all_results = []
            strategy_results: dict[str, list] = {p: [] for p in comparing_profiles}

            valid_strata = [
                lbl for lbl in pooled_df['cluster_label'].unique()
                if lbl >= 0
            ]

            for stratum_label in sorted(valid_strata):
                stratum_df = pooled_df[pooled_df['cluster_label'] == stratum_label].copy()
                dispersion = dispersion_map.get(stratum_label, 0.0)

                # Coverage check
                sensor_col_candidates = ['session_id']
                sensor_col = next(
                    (c for c in sensor_col_candidates if c in stratum_df.columns),
                    None,
                )
                if sensor_col is None:
                    logging.warning(
                        f"[{extraction_name}/{clusterer_name}] stratum {stratum_label}: "
                        "no sensor column found — skipping."
                    )
                    continue

                if not _satisfies_coverage(
                    stratum_df, sensor_col, min_instances_per_sensor, min_sensor_types
                ):
                    logging.info(
                        f"  [compare] {extraction_name}/{clusterer_name} stratum "
                        f"{stratum_label} — non-exploitable (coverage)"
                    )
                    continue

                for profile_name, profile_config in comparing_profiles.items():
                    method = profile_config.get('method', profile_name)
                    meas_col = profile_config.get('measurement_col', 'spike_elicited')
                    s_col = profile_config.get('sensor_col', sensor_col)

                    if meas_col not in stratum_df.columns:
                        logging.warning(
                            f"[{extraction_name}/{clusterer_name}] "
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
                            f"[{extraction_name}/{clusterer_name}] strategy '{profile_name}' "
                            f"stratum {stratum_label} failed: {exc}"
                        )
                        continue

                    strategy_results[profile_name].append(result)
                    all_results.append(result)

            # Write per-strategy JSON
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

            # Write synthesis report
            if all_results:
                try:
                    synthesis_df = synthesize_across_strata(all_results)
                    synthesis_records = synthesis_df.to_dict(orient='records')
                    with open(synthesis_path, 'w') as f:
                        json.dump(synthesis_records, f, indent=2, default=str)
                    outputs.append(synthesis_path)
                    print(
                        f"  [compare] {extraction_name} / {clusterer_name} — "
                        f"{len(all_results)} result(s), synthesis written",
                        flush=True,
                    )
                except Exception as exc:
                    logging.error(f"Failed to write synthesis report: {exc}")
            else:
                print(
                    f"  [compare] {extraction_name} / {clusterer_name} — "
                    "no exploitable strata",
                    flush=True,
                )

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
    # Convert nan floats to None for clean JSON
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

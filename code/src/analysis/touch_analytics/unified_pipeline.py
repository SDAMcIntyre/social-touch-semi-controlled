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
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class _TqdmLineWrapper:
    """Routes tqdm output through print()-compatible newline-terminated writes.

    tqdm normally writes `\\r<bar>` (no newline) to stderr. In the GUI subprocess
    pipe, these bytes accumulate and prefix the next print() call's newline,
    causing ProcessOutputReader to route the milestone through cr_line_received
    (replace_last_line) instead of line_received (append_line).

    This wrapper:
      - strips the leading `\\r` so bar lines do not trigger replace_last_line
      - appends `\\n` so each tqdm write is a complete, independently-readable line
      - writes to sys.stdout (same stream as print()) for deterministic ordering
    """
    def __init__(self, stream):
        self._stream = stream

    def write(self, s: str) -> int:
        s = s.lstrip('\r')
        if s and not s.endswith('\n'):
            s += '\n'
        return self._stream.write(s)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False


from utils.should_process_task import should_process_task
from .feature_extraction import get_extractor
from .clustering import get_clusterer
from .reporting import VisualReportingStrategy

# Columns written by the orchestrator (not the extractor)
_SHARED_COLUMNS = [
    'block_order_id', 'trial_id', 'single_touch_id',
    'type_metadata', 'direction',
    'mean_contact_x', 'mean_contact_y', 'mean_contact_z',
    'spike_elicited',
    'session_id',
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
    extraction_profiles = {k: v for k, v in extraction_profiles.items() if v.get('enabled', True)}
    clustering_profiles = {k: v for k, v in clustering_profiles.items() if v.get('enabled', True)}
    force = opts.get('force_processing', False) or force

    all_outputs: List[Path] = []

    # Per-session extraction for every profile
    # Keys: profile_name → list of session CSV paths
    per_profile_session_csvs: dict[str, list[Path]] = {p: [] for p in extraction_profiles}

    n_steps = (
        len(input_items) * len(extraction_profiles)
        + 2 * len(extraction_profiles) * len(clustering_profiles)
    )

    print(
        f"=== unified pipeline: {len(input_items)} sessions, "
        f"{len(extraction_profiles)} extractors, "
        f"{len(clustering_profiles)} clusterers ===",
        flush=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with tqdm(total=n_steps, desc="unified pipeline", unit="step",
                  file=_TqdmLineWrapper(sys.stdout)) as progress:

            for input_file, _ in input_items:
                session_outputs = _extract_session(
                    input_file=input_file,
                    output_dir=output_dir,
                    extraction_profiles=extraction_profiles,
                    force=force,
                    progress=progress,
                )
                for profile_name, csv_path in session_outputs.items():
                    per_profile_session_csvs[profile_name].append(csv_path)
                    all_outputs.append(csv_path)

            # Global clustering per extraction profile × clustering profile
            for profile_name, session_csvs in per_profile_session_csvs.items():
                if not session_csvs:
                    progress.update(2 * len(clustering_profiles))
                    continue
                cluster_outputs = _cluster_profile(
                    profile_name=profile_name,
                    session_csvs=session_csvs,
                    output_dir=output_dir,
                    clustering_profiles=clustering_profiles,
                    force=force,
                    progress=progress,
                )
                all_outputs.extend(cluster_outputs)

    print(f"=== unified pipeline complete: {len(all_outputs)} outputs ===", flush=True)
    return all_outputs


# ---------------------------------------------------------------------------
# Per-session extraction
# ---------------------------------------------------------------------------

def _extract_session(
    input_file: Path,
    output_dir: Path,
    extraction_profiles: dict,
    force: bool,
    progress: tqdm = None,
) -> dict[str, Path]:
    """
    Run all enabled extraction profiles on one session CSV.

    Returns
    -------
    dict mapping profile_name → output CSV path (may point to an up-to-date
    file that was not rewritten).
    """
    results: dict[str, Path] = {}
    session_id = _session_id_from_path(input_file)

    try:
        df = pd.read_csv(input_file)
    except Exception as exc:
        logging.error(f"Failed to load {input_file}: {exc}")
        if progress is not None:
            progress.update(len(extraction_profiles))
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
                    if progress is not None:
                        progress.set_postfix_str(f"extract: {session_id}/{profile_name}", refresh=False)
                        progress.update(1)
                    print(f"  [extract] {session_id} / {profile_name} — up to date", flush=True)
                    continue
            except FileNotFoundError:
                pass  # input missing — will error below

        # Extract
        try:
            extractor = get_extractor(method)
        except KeyError as exc:
            logging.error(f"Profile '{profile_name}': {exc}")
            if progress is not None:
                progress.set_postfix_str(f"extract: {session_id}/{profile_name}", refresh=False)
                progress.update(1)
            print(f"  [extract] {session_id} / {profile_name} — error: unknown extractor", flush=True)
            continue

        rows = _extract_all_touches(
            df, extractor, profile_config, has_nerve_data,
            desc=f"{session_id}/{profile_name}",
        )
        summary_df = pd.DataFrame(rows)

        try:
            summary_df.to_csv(output_path, index=False)
            logging.info(
                f"[{profile_name}] Saved {len(summary_df)} touches → {output_path}"
            )
        except Exception as exc:
            logging.error(f"Failed to save {output_path}: {exc}")
            if progress is not None:
                progress.set_postfix_str(f"extract: {session_id}/{profile_name}", refresh=False)
                progress.update(1)
            print(f"  [extract] {session_id} / {profile_name} — error: save failed", flush=True)
            continue

        results[profile_name] = output_path
        if progress is not None:
            progress.set_postfix_str(f"extract: {session_id}/{profile_name}", refresh=False)
            progress.update(1)
        print(f"  [extract] {session_id} / {profile_name} — {len(summary_df)} touches", flush=True)

    return results


def _extract_all_touches(
    df: pd.DataFrame,
    extractor,
    profile_config: dict,
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
    progress: tqdm = None,
) -> list[Path]:
    """Pool all session CSVs for *profile_name* and run each clusterer."""
    outputs: list[Path] = []

    # Pool
    dfs = []
    for p in session_csvs:
        if p.exists():
            try:
                session_df = pd.read_csv(p)
                session_df['session_id'] = _session_id_from_path(p)
                dfs.append(session_df)
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
                    if progress is not None:
                        progress.set_postfix_str(f"cluster: {profile_name}/{clusterer_name}", refresh=False)
                        progress.update(2)
                    print(f"  [cluster] {profile_name} / {clusterer_name} — up to date", flush=True)
                    continue
            except FileNotFoundError:
                pass

        try:
            clusterer = get_clusterer(method)
        except KeyError as exc:
            logging.error(f"Clustering profile '{clusterer_name}': {exc}")
            if progress is not None:
                progress.set_postfix_str(f"cluster: {profile_name}/{clusterer_name}", refresh=False)
                progress.update(2)
            print(f"  [cluster] {profile_name} / {clusterer_name} — error: unknown clusterer", flush=True)
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
            if progress is not None:
                progress.set_postfix_str(f"cluster: {profile_name}/{clusterer_name}", refresh=False)
                progress.update(2)
            print(f"  [cluster] {profile_name} / {clusterer_name} — error: no numeric features", flush=True)
            continue

        feature_df = pooled[feature_cols].dropna()
        valid_idx = feature_df.index

        try:
            labels, metadata = clusterer.fit_predict(feature_df, clusterer_config)
        except Exception as exc:
            logging.error(
                f"[{profile_name}/{clusterer_name}] Clustering failed: {exc}"
            )
            if progress is not None:
                progress.set_postfix_str(f"cluster: {profile_name}/{clusterer_name}", refresh=False)
                progress.update(2)
            print(f"  [cluster] {profile_name} / {clusterer_name} — error: clustering failed", flush=True)
            continue

        result_df = pooled.loc[valid_idx].copy()
        result_df['cluster_label'] = labels

        extra_cols = metadata.pop('extra_columns', None)
        if extra_cols:
            for col_name, col_values in extra_cols.items():
                result_df[col_name] = col_values

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
            if progress is not None:
                progress.set_postfix_str(f"cluster: {profile_name}/{clusterer_name}", refresh=False)
                progress.update(2)
            print(f"  [cluster] {profile_name} / {clusterer_name} — error: save failed", flush=True)
            continue

        if 'k' in metadata:
            cluster_desc = f"{metadata['k']} clusters, {len(result_df)} samples"
        elif 'n_bins' in metadata:
            cluster_desc = f"{metadata['n_bins']} bins, {len(result_df)} samples"
        else:
            cluster_desc = f"{len(result_df)} samples"
        if progress is not None:
            progress.set_postfix_str(f"cluster: {profile_name}/{clusterer_name}", refresh=False)
            progress.update(1)
        print(f"  [cluster] {profile_name} / {clusterer_name} — {cluster_desc}", flush=True)

        if len(feature_cols) >= 2 and 'session_id' in result_df.columns:
            n_heatmap_sessions = result_df['session_id'].nunique()
            _generate_session_heatmaps(result_df, out_dir, feature_cols)
            if progress is not None:
                progress.set_postfix_str(f"heatmap: {profile_name}/{clusterer_name}", refresh=False)
                progress.update(1)
            print(f"  [heatmap] {profile_name} / {clusterer_name} — {n_heatmap_sessions} sessions", flush=True)
        else:
            if progress is not None:
                progress.set_postfix_str(f"heatmap: {profile_name}/{clusterer_name}", refresh=False)
                progress.update(1)
            print(f"  [heatmap] {profile_name} / {clusterer_name} — skipped", flush=True)

    return outputs


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _session_id_from_path(p: Path) -> str:
    """Return the session ID prefix from a touch-summary CSV filename."""
    name = p.name
    if '_semicontrolled_' in name:
        return name.split('_semicontrolled_')[0]
    return p.stem


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

    Skips silently when fewer than 2 feature columns are available (N×2 layout
    requires at least one X and one Y axis).
    """
    if len(feature_cols) < 2:
        logging.warning("_generate_session_heatmaps: need ≥ 2 features, skipping.")
        return
    if 'session_id' not in result_df.columns:
        logging.warning("_generate_session_heatmaps: 'session_id' column missing, skipping.")
        return

    # Prefer velocity column on X-axis (aligns with matrix_generation.py convention);
    # fall back to highest-variance feature when no velocity column is present.
    velocity_cols = [c for c in feature_cols if 'velocity' in c.lower()]
    if velocity_cols:
        x_col = velocity_cols[0]
    else:
        x_col = _select_highest_variance_feature(result_df, feature_cols)
    y_cols = [c for c in feature_cols if c != x_col]

    # Pre-compute global bin edges from the full pooled DataFrame so that every
    # per-session heatmap shares identical axis ranges and color scaling.
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

    # Pre-compute the cross-session global max touch count for shared LogNorm.
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

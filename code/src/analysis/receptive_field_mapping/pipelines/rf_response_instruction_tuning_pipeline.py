"""Pipeline orchestrator for response instruction-tuning bar charts.

Groups single touches by designed metadata instruction levels and renders:

- per-session bar charts (IFF +/- STD per instruction level)
- all-sessions overlay dot-plots (jittered dots per session per level)

Output layout::

    4_analysed/stimulus_response_instruction_tuning/iff_{metric}/
        response_instruction_tuning_sentinel.json
        {category_col}/
            {gesture_subset}/
                {session_id}_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning.png
                {session_id}_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning.csv
                overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning_by_type.png
                overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning_by_session.png
                overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import STIMULUS_RESPONSE_INSTRUCTION_TUNING
from analysis.pipeline.shared_constants import (
    TOUCH_ID_COLS,
    session_id_from_path,
)
from analysis.receptive_field_mapping.pipelines.rf_response_tuning_pipeline import (
    RESPONSE_METRICS,
    _filter_gesture,
    _write_sentinel,
    _load_session_metadata_df,
)
from analysis.receptive_field_mapping.pipelines.rf_touch_feature_radar_pipeline import (
    _find_feature_csv,
)
from analysis.receptive_field_mapping.rendering.rf_response_instruction_tuning_renderer import (
    CategoryResult,
    _group_by_category,
    render_session_instruction_tuning,
    render_overlay_instruction_tuning,
)
from analysis.receptive_field_mapping.rendering.neuron_type_colors import (
    build_session_color_scheme,
    SessionColorScheme,
)
from analysis.receptive_field_mapping.rendering.rf_cross_neuron_renderer import (
    render_cross_neuron_comparison_multicondition,
)

logger = logging.getLogger(__name__)

_GESTURE_SUBSETS = ["all", "tap", "stroke", "stroke_proximal", "stroke_distal"]
_MIN_ROWS = 5


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------


def _write_category_csv(
    category_result: CategoryResult,
    session_id: str,
    category_col: str,
    gesture_subset: str,
    iff_metric: str,
    out_path: Path,
) -> None:
    """Write per-category statistics to a CSV file.

    Columns written:
        ``category_label``, ``count``, ``iff_mean``, ``iff_std``,
        ``session_id``, ``category_column``, ``gesture_subset``, ``iff_metric``

    Parameters
    ----------
    category_result:
        Grouped statistics from ``_group_by_category``.
    session_id:
        Session identifier stored as a scalar repeated per row.
    category_col:
        Metadata column name stored as a scalar repeated per row.
    gesture_subset:
        Gesture subset label stored as a scalar repeated per row.
    iff_metric:
        Metric token (e.g. ``"iff_mean"`` or ``"iff_max"``) stored as a
        scalar repeated per row.
    out_path:
        Absolute path where the CSV will be saved.  Parent directories are
        created automatically.
    """
    n = len(category_result.category_labels)
    rows = {
        "category_label": category_result.category_labels,
        "count": category_result.counts,
        "iff_mean": category_result.mean_iff,
        "iff_std": category_result.std_iff,
        "session_id": [session_id] * n,
        "category_column": [category_col] * n,
        "gesture_subset": [gesture_subset] * n,
        "iff_metric": [iff_metric] * n,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Global count max helper
# ---------------------------------------------------------------------------


def _compute_global_category_count_max(
    session_dfs: list[pd.DataFrame],
    category_col: str,
    global_levels: list[str],
) -> int:
    """Return the maximum per-category count across all sessions for annotation positioning.

    Iterates all session DataFrames, groups each by *category_col*, and
    returns the single largest per-level count found across all sessions.

    Parameters
    ----------
    session_dfs:
        All per-session DataFrames (already merged with metadata).
    category_col:
        Metadata column name to group by.
    global_levels:
        Ordered list of all known category levels.

    Returns
    -------
    int
        Maximum per-category touch count.  Returns 1 if no data is found to
        prevent a zero result.
    """
    global_max = 0
    for df in session_dfs:
        if category_col not in df.columns:
            continue
        for level in global_levels:
            count = int((df[category_col] == level).sum())
            if count > global_max:
                global_max = count
    return max(global_max, 1)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_response_instruction_tuning(
    session_config_paths: list[tuple[Path, Path]],
    options: dict,
    output_base_dir: Path,
) -> None:
    """Render instruction-tuning bar chart PNGs for all sessions and response metrics.

    Two-pass pipeline:

    1. **Load** — for each session, find the ``mean`` aggregation CSV under
       ``stimulus_extract_features/``, load the metadata CSV, and merge both
       on ``TOUCH_ID_COLS``.  When the response column is absent from the mean
       CSV, load its dedicated aggregation folder and inner-merge.
    2. **Render** — for each metric × category column × gesture subset, call
       ``render_session_instruction_tuning`` (per session) and
       ``render_overlay_instruction_tuning`` (cross-session), using globally
       consistent Y-axis limits.

    Idempotency is provided by a per-metric
    ``response_instruction_tuning_sentinel.json`` inside each
    ``iff_{metric_subdir}/`` subfolder.  When the sentinel exists and
    ``force_processing`` is ``False`` the metric is skipped.

    Parameters
    ----------
    session_config_paths:
        List of ``(csv_path, database_path)`` tuples.
    options:
        Task options from the DAG config.  Recognised keys:

        - ``tuning_categories`` — list of metadata column names
          (e.g. ``['contact_area_metadata', 'speed_metadata']``).
        - ``iff_metric`` — ``"mean"``, ``"max"``, or ``"both"``
          (default ``"mean"``).
        - ``clip_percentile`` — percentile used for IFF ylim computation
          (default ``1.0``).
        - ``force_processing`` — re-render even when sentinel exists
          (default ``False``).
        - ``metadata_dir`` — subdirectory under ``4_analysed/`` for metadata
          CSVs (default ``"touch_prepare_sessions"``).
        - ``metadata_filename`` — filename template for the metadata CSV;
          must contain ``{session_id}``
          (default ``"{session_id}_prepared.csv"``).
    output_base_dir:
        Root output directory, e.g.
        ``database_path / '4_analysed' / STIMULUS_RESPONSE_INSTRUCTION_TUNING``.

    Raises
    ------
    ValueError
        If ``tuning_categories`` is empty, if ``iff_metric`` is unknown, if
        the metadata CSV is missing, if required columns are absent, or if the
        inner merge drops rows.
    """
    tuning_categories: list[str] = list(options.get("tuning_categories") or [])
    if not tuning_categories:
        raise ValueError(
            "run_response_instruction_tuning: 'tuning_categories' is empty. "
            "Specify at least one metadata column in the DAG config "
            "(e.g. 'contact_area_metadata')."
        )

    clip_percentile: float = float(options.get("clip_percentile", 1.0))
    force_processing: bool = bool(options.get("force_processing", False))
    metadata_dir: str = str(options.get("metadata_dir", "touch_prepare_sessions"))
    metadata_filename: str = str(
        options.get("metadata_filename", "{session_id}_prepared.csv")
    )

    iff_metric_token = str(options.get("iff_metric", "mean"))
    if iff_metric_token == "both":
        metric_specs = [RESPONSE_METRICS["iff_mean"], RESPONSE_METRICS["iff_max"]]
    elif iff_metric_token == "mean":
        metric_specs = [RESPONSE_METRICS["iff_mean"]]
    elif iff_metric_token == "max":
        metric_specs = [RESPONSE_METRICS["iff_max"]]
    else:
        raise ValueError(
            f"run_response_instruction_tuning: unknown iff_metric '{iff_metric_token}'. "
            f"Valid values: 'mean', 'max', 'both'."
        )

    for response_col, agg_folder, _bin_agg, response_ylabel, metric_subdir in metric_specs:

        metric_dir = output_base_dir / metric_subdir
        sentinel = metric_dir / "response_instruction_tuning_sentinel.json"

        if sentinel.exists() and not force_processing:
            logger.info(
                "[Response Instruction Tuning] %s — up-to-date, skipping (sentinel: %s).",
                metric_subdir,
                sentinel,
            )
            continue

        # =====================================================================
        # Pass 1 — Load all sessions for this metric
        # =====================================================================
        session_data: list[dict] = []

        for csv_path, database_path in session_config_paths:
            csv_path = Path(csv_path)
            database_path = Path(database_path)
            session_id = session_id_from_path(csv_path)

            # --- Load mean feature CSV ---
            mean_csv = _find_feature_csv(database_path, "mean", session_id)
            df = pd.read_csv(mean_csv)

            # --- Ensure response_col is present ---
            if response_col not in df.columns:
                agg_csv = _find_feature_csv(database_path, agg_folder, session_id)
                df_agg = pd.read_csv(agg_csv)

                touch_id_cols = list(TOUCH_ID_COLS)
                missing_in_mean = [c for c in touch_id_cols if c not in df.columns]
                if missing_in_mean:
                    raise ValueError(
                        f"[Response Instruction Tuning] {session_id}: TOUCH_ID_COLS "
                        f"columns {missing_in_mean} not found in mean CSV {mean_csv}."
                    )
                missing_in_agg = [c for c in touch_id_cols if c not in df_agg.columns]
                if missing_in_agg:
                    raise ValueError(
                        f"[Response Instruction Tuning] {session_id}: TOUCH_ID_COLS "
                        f"columns {missing_in_agg} not found in '{agg_folder}' CSV "
                        f"{agg_csv}."
                    )
                if response_col not in df_agg.columns:
                    raise ValueError(
                        f"[Response Instruction Tuning] {session_id}: column "
                        f"'{response_col}' not found in '{agg_folder}' CSV {agg_csv}. "
                        f"Available columns: {sorted(df_agg.columns)}. "
                        f"Ensure 'stimulus_extract_features' with aggregation "
                        f"'{agg_folder}' has run."
                    )

                n_mean_rows = len(df)
                df = df.merge(
                    df_agg[touch_id_cols + [response_col]],
                    on=touch_id_cols,
                    how="inner",
                )
                n_merged_rows = len(df)
                if n_merged_rows != n_mean_rows:
                    raise ValueError(
                        f"[Response Instruction Tuning] {session_id}: inner merge on "
                        f"TOUCH_ID_COLS yielded {n_merged_rows} rows but the mean CSV "
                        f"had {n_mean_rows} rows. Touch IDs must be identical across "
                        f"aggregation CSVs — check that 'stimulus_extract_features' "
                        f"(mean and '{agg_folder}') was run on the same data."
                    )

            # --- Load metadata CSV and merge ---
            meta_df = _load_session_metadata_df(
                database_path,
                metadata_dir,
                metadata_filename,
                session_id,
                tuning_categories,
            )

            touch_id_cols = list(TOUCH_ID_COLS)
            meta_deduped = (
                meta_df[touch_id_cols + tuning_categories]
                .groupby(touch_id_cols)[tuning_categories]
                .first()
                .reset_index()
            )
            overlap_cols = [c for c in tuning_categories if c in df.columns]
            if overlap_cols:
                df = df.drop(columns=overlap_cols)
            n_before = len(df)
            df = df.merge(meta_deduped, on=touch_id_cols, how="inner")
            n_after = len(df)
            if n_after != n_before:
                raise ValueError(
                    f"[Response Instruction Tuning] {session_id}: inner merge with "
                    f"metadata dropped rows ({n_before} → {n_after}). "
                    f"Touch IDs in the metadata CSV must cover every touch in the "
                    f"feature CSV."
                )

            session_data.append({
                "session_id": session_id,
                "df": df,
            })

        if not session_data:
            logger.info("[Response Instruction Tuning] No sessions to process.")
            continue

        # =====================================================================
        # Resolve global category levels per tuning column
        # =====================================================================
        pooled = pd.concat([entry["df"] for entry in session_data], ignore_index=True)

        global_levels: dict[str, list] = {}
        for cat_col in tuning_categories:
            if cat_col not in pooled.columns:
                raise ValueError(
                    f"[Response Instruction Tuning] Category column '{cat_col}' is not "
                    f"present in the pooled DataFrame after merging metadata. "
                    f"Available columns: {sorted(pooled.columns.tolist())[:20]}"
                )
            raw_values = pooled[cat_col].dropna().unique()
            try:
                numeric_vals = [float(v) for v in raw_values]
                ordered = [v for _, v in sorted(zip(numeric_vals, raw_values))]
            except (ValueError, TypeError):
                ordered = sorted(str(v) for v in raw_values)
            global_levels[cat_col] = ordered

        # =====================================================================
        # Compute global IFF ylim
        # =====================================================================
        all_iff_vals: list[float] = []
        for entry in session_data:
            all_iff_vals.extend(
                entry["df"][response_col].dropna().tolist()
            )
        if all_iff_vals:
            arr = np.array(all_iff_vals)
            ymin = max(0.0, float(np.percentile(arr, clip_percentile)))
            ymax = float(np.percentile(arr, 100.0 - clip_percentile))
            if ymax <= ymin:
                ymax = ymin + 1.0
            iff_ylim = (ymin, ymax)
        else:
            iff_ylim = (0.0, 1.0)

        session_ids = [entry["session_id"] for entry in session_data]
        neuron_summary_xlsx_str: str | None = options.get("neuron_summary_xlsx") or None
        if not neuron_summary_xlsx_str:
            raise ValueError(
                "run_response_instruction_tuning: 'neuron_summary_xlsx' is not set in the task options. "
                "Set configs/analyse_workflow_processing_dag.yaml parameters.neuron_summary_xlsx "
                "to the path of MNG-DataSummary.xlsx (absolute, or relative to the database root)."
            )
        xlsx_path = Path(neuron_summary_xlsx_str)
        if not xlsx_path.is_absolute():
            xlsx_path = database_path / xlsx_path
        if not xlsx_path.is_file():
            raise FileNotFoundError(
                f"run_response_instruction_tuning: neuron_summary_xlsx not found: {xlsx_path}"
            )
        scheme: SessionColorScheme = build_session_color_scheme(session_ids, xlsx_path)

        # =====================================================================
        # Pass 2 — Render
        # =====================================================================
        for category_col in tuning_categories:
            levels = global_levels[category_col]

            for gesture_subset in _GESTURE_SUBSETS:
                overlay_session_data: dict[str, CategoryResult] = {}
                overlay_csv_dfs: list[pd.DataFrame] = []

                for entry in session_data:
                    session_id = entry["session_id"]
                    df = entry["df"]

                    filtered = _filter_gesture(df, gesture_subset)
                    if len(filtered) < _MIN_ROWS:
                        logger.warning(
                            "[Response Instruction Tuning] %s / %s / %s / %s: only %d "
                            "row(s) after gesture filter — skipping (need ≥%d).",
                            metric_subdir,
                            category_col,
                            gesture_subset,
                            session_id,
                            len(filtered),
                            _MIN_ROWS,
                        )
                        continue

                    cat_result = _group_by_category(
                        filtered, category_col, response_col, levels
                    )

                    out_path = (
                        metric_dir
                        / category_col
                        / gesture_subset
                        / f"{session_id}_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning.png"
                    )
                    render_session_instruction_tuning(
                        category_result=cat_result,
                        category_col=category_col,
                        iff_col_name=response_col,
                        session_id=session_id,
                        gesture_subset=gesture_subset,
                        out_path=out_path,
                        iff_ylim=iff_ylim,
                        iff_ylabel=response_ylabel,
                        bar_color=scheme.session_color[session_id],
                    )
                    print(
                        f"[Response Instruction Tuning] {metric_subdir} / {category_col} / "
                        f"{gesture_subset} / {session_id}: saved {out_path.name}",
                        flush=True,
                    )

                    csv_path = out_path.with_suffix(".csv")
                    _write_category_csv(
                        cat_result,
                        session_id,
                        category_col,
                        gesture_subset,
                        metric_subdir,
                        csv_path,
                    )

                    overlay_session_data[session_id] = cat_result
                    overlay_csv_dfs.append(pd.read_csv(csv_path))

                if len(overlay_session_data) < 2:
                    logger.warning(
                        "[Response Instruction Tuning] %s / %s / %s: fewer than 2 sessions "
                        "have ≥%d rows — skipping overlay.",
                        metric_subdir,
                        category_col,
                        gesture_subset,
                        _MIN_ROWS,
                    )
                    continue

                overlay_path_by_type = (
                    metric_dir
                    / category_col
                    / gesture_subset
                    / f"overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning_by_type.png"
                )
                render_overlay_instruction_tuning(
                    session_category_data=overlay_session_data,
                    category_col=category_col,
                    gesture_subset=gesture_subset,
                    out_path=overlay_path_by_type,
                    iff_ylim=iff_ylim,
                    session_colors={sid: scheme.session_color[sid] for sid in overlay_session_data},
                    iff_ylabel=response_ylabel,
                    legend_mode="by_type",
                    session_neuron_types=scheme.session_neuron_type,
                    type_colors=scheme.type_color,
                )
                overlay_path_by_session = (
                    metric_dir
                    / category_col
                    / gesture_subset
                    / f"overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning_by_session.png"
                )
                render_overlay_instruction_tuning(
                    session_category_data=overlay_session_data,
                    category_col=category_col,
                    gesture_subset=gesture_subset,
                    out_path=overlay_path_by_session,
                    iff_ylim=iff_ylim,
                    session_colors={sid: scheme.session_color[sid] for sid in overlay_session_data},
                    iff_ylabel=response_ylabel,
                    legend_mode="by_session",
                    session_neuron_types=scheme.session_neuron_type,
                    type_colors=scheme.type_color,
                )
                overlay_csv_path = overlay_path_by_type.parent / f"overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning.csv"
                pd.concat(overlay_csv_dfs, ignore_index=True).to_csv(
                    overlay_csv_path, index=False
                )
                print(
                    f"[Response Instruction Tuning] {metric_subdir} / {category_col} / "
                    f"{gesture_subset}: saved {overlay_path_by_type.name} + "
                    f"{overlay_path_by_session.name}",
                    flush=True,
                )

                cross_neuron_data: dict[str, dict[str, list[float]]] = {}
                for sid, cat_result in overlay_session_data.items():
                    neuron_type = scheme.session_neuron_type.get(sid)
                    if neuron_type is None:
                        continue
                    if neuron_type not in cross_neuron_data:
                        cross_neuron_data[neuron_type] = {}
                    for level_label, mean_val in zip(cat_result.category_labels, cat_result.mean_iff):
                        if not np.isfinite(mean_val):
                            continue
                        if level_label not in cross_neuron_data[neuron_type]:
                            cross_neuron_data[neuron_type][level_label] = []
                        cross_neuron_data[neuron_type][level_label].append(float(mean_val))

                if cross_neuron_data:
                    cross_neuron_path = (
                        metric_dir
                        / category_col
                        / gesture_subset
                        / f"cross_neuron_{category_col}_{gesture_subset}_{metric_subdir}.png"
                    )
                    render_cross_neuron_comparison_multicondition(
                        neuron_data=cross_neuron_data,
                        metric_name=f"{metric_subdir} | {category_col}",
                        condition_labels=levels,
                        output_path=cross_neuron_path,
                        ylabel=response_ylabel,
                        title=f"Cross-neuron | {category_col} | {gesture_subset} | {metric_subdir}",
                    )
                    print(
                        f"[Response Instruction Tuning] {metric_subdir} / {category_col} / "
                        f"{gesture_subset}: saved {cross_neuron_path.name}",
                        flush=True,
                    )

        # =====================================================================
        # Write per-metric sentinel
        # =====================================================================
        _write_sentinel(
            sentinel,
            n_sessions=len(session_data),
            n_features=len(tuning_categories),
        )
        print(
            f"[Response Instruction Tuning] {metric_subdir} — done. "
            f"{len(tuning_categories)} category column(s), "
            f"{len(session_data)} session(s). Sentinel written.",
            flush=True,
        )

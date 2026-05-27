"""Pipeline orchestrator for cross-session touch feature comparison plots.

Pools per-session touch feature summary CSVs into a single DataFrame and
delegates per-feature and per-gesture rendering to
``rf_stimulus_session_comparison_renderer``.

Output layout per group::

    4_analysed/stimulus_compare_sessions/{group_name}/
        pooled_feature_summary.csv
        session_comparison_sentinel.json
        {gesture_type}/
            {feature_col}_session_comparison.png
        {gesture_type}_feature_grid.png
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import STIMULUS_COMPARE_SESSIONS
from analysis.pipeline.shared_constants import (
    GESTURE_TYPES,
    TOUCH_ID_COLS_WITH_SESSION,
    session_id_from_path,
)
from analysis.receptive_field_mapping.pipelines.rf_touch_feature_radar_pipeline import (
    _find_feature_csv,
    _load_and_merge_feature_csvs,
    _resolve_radar_columns,
    _resolve_required_aggregation_folders,
    _DISPLAY_NAMES,
)
from analysis.receptive_field_mapping.rendering.rf_stimulus_session_comparison_renderer import (
    assign_session_colors,
    render_feature_session_comparison,
    render_feature_summary_grid,
)
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Y-axis limit computation
# ---------------------------------------------------------------------------


def _compute_ylims(
    df: pd.DataFrame,
    cols: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute per-feature y-axis limits with a 5% margin.

    For positive-only features the lower bound is anchored at 0.0.  If a
    column contains no finite values a ``ValueError`` is raised immediately
    (fail-fast — a column with no data indicates a pipeline configuration
    error that must be surfaced, not silently ignored).

    Parameters
    ----------
    df:
        Pooled feature DataFrame (all sessions concatenated).
    cols:
        Feature column names to compute limits for.

    Returns
    -------
    dict
        Mapping ``{col: (lo, hi)}`` for each column in *cols*.

    Raises
    ------
    ValueError
        If any column has no finite values.
    """
    ylims: dict[str, tuple[float, float]] = {}
    for col in cols:
        finite = df[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(finite) == 0:
            raise ValueError(
                f"_compute_ylims: column '{col}' contains no finite values in the "
                f"pooled DataFrame. Ensure 'stimulus_extract_features' has produced "
                f"valid data for this feature across all sessions."
            )
        lo = float(finite.min())
        hi = float(finite.max())
        if lo == hi:
            pad = abs(lo) * 0.05 if lo != 0.0 else 1.0
            lo_out = lo - pad
            hi_out = hi + pad
        else:
            margin = 0.05 * (hi - lo)
            lo_out = lo - margin if lo < 0.0 else 0.0
            hi_out = hi + margin
        ylims[col] = (lo_out, hi_out)
    return ylims


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------


def _write_sentinel(sentinel: Path, group_name: str, n_sessions: int) -> None:
    """Write a JSON sentinel recording completion metadata for *group_name*."""
    data = {
        "group_name": group_name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_sessions": n_sessions,
    }
    with open(sentinel, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def run_stimulus_session_comparison(
    session_configs: list[tuple[Path, Path]],
    comparison_groups: dict,
    output_dir: Path,
    plot_type: Literal["box_strip", "violin", "bar_error"] = "box_strip",
    force_processing: bool = False,
) -> None:
    """Render cross-session touch feature comparison plots for each enabled group.

    For each enabled group the pipeline runs in two passes:

    1. **Load** — iterate over all sessions, load and merge feature CSVs,
       assign ``session_id``, concatenate into a pooled DataFrame.
    2. **Render** — for each gesture subset (``all`` + each gesture type in
       ``GESTURE_TYPES``) and each resolved feature column, call the renderer.
       A summary grid PNG is also written per gesture subset.

    Idempotency is provided by a per-group sentinel JSON file.  When the
    sentinel exists and ``force_processing`` is ``False`` the group is skipped
    entirely.

    Parameters
    ----------
    session_configs:
        List of ``(csv_path, database_path)`` tuples, following the same
        convention as other pipelines in this package.
    comparison_groups:
        Dict mapping group name -> group spec.  Each spec must have an
        ``enabled`` bool and a ``features`` dict (data type -> list of
        aggregation methods).  Groups with ``enabled: false`` are skipped.
    output_dir:
        Root output directory for this task
        (e.g. ``database_path / '4_analysed' / STIMULUS_COMPARE_SESSIONS``).
    plot_type:
        Visual style for each feature plot: ``'box_strip'`` (box + jittered
        dots, default), ``'violin'``, or ``'bar_error'`` (mean ± std).
    force_processing:
        If ``True``, reprocess groups even when the sentinel file is present.

    Raises
    ------
    ValueError
        If *comparison_groups* is empty, if a group's ``features`` spec is
        empty or missing, if ``gesture_type`` column is absent from a loaded
        DataFrame, or if any feature column has no finite values.
    """
    if not comparison_groups:
        raise ValueError(
            "run_stimulus_session_comparison: 'comparison_groups' is empty. "
            "Define at least one comparison group in the DAG config."
        )

    for group_name, group_spec in comparison_groups.items():
        if not group_spec.get("enabled", True):
            logger.info(
                "[Stimulus Compare Sessions] Group '%s': disabled — skipping.",
                group_name,
            )
            continue

        features_spec: dict = group_spec.get("features", {})
        if not features_spec:
            raise ValueError(
                f"[Stimulus Compare Sessions] Group '{group_name}': 'features' is "
                f"empty or missing. Define at least one data type in the group spec."
            )

        sentinel = output_dir / group_name / "session_comparison_sentinel.json"

        if sentinel.exists() and not force_processing:
            logger.info(
                "[Stimulus Compare Sessions] Group %s: up-to-date — skipping.",
                group_name,
            )
            continue

        agg_folders = _resolve_required_aggregation_folders(features_spec)
        if not agg_folders:
            raise ValueError(
                f"[Stimulus Compare Sessions] Group '{group_name}': no aggregation "
                f"folders resolved from features spec. "
                f"A group with only 'location: [mean]' is not supported — "
                f"add at least one non-mean-location feature or aggregation."
            )

        print(
            f"[Stimulus Compare Sessions] Group '{group_name}' — folders: {agg_folders}",
            flush=True,
        )

        # =================================================================
        # Pass 1 — Load all sessions
        # =================================================================
        session_data: list[dict] = []
        resolved_cols: list[str] | None = None
        display_labels: list[str] | None = None

        for csv_path, database_path in session_configs:
            csv_path = Path(csv_path)
            database_path = Path(database_path)
            session_id = session_id_from_path(csv_path)

            df = _load_and_merge_feature_csvs(database_path, agg_folders, session_id)

            if "gesture_type" not in df.columns:
                raise ValueError(
                    f"[Stimulus Compare Sessions] {session_id}: 'gesture_type' column "
                    f"not found in merged feature DataFrame for group '{group_name}'. "
                    f"Available columns: {list(df.columns)}"
                )

            cols, labels = _resolve_radar_columns(df, features_spec)
            if resolved_cols is None:
                resolved_cols, display_labels = cols, labels

            df["session_id"] = session_id
            session_data.append({
                "session_id": session_id,
                "df": df,
                "cols": cols,
                "labels": labels,
                "feature_csv_path": _find_feature_csv(
                    database_path, agg_folders[0], session_id
                ),
            })

        if not session_data:
            logger.info(
                "[Stimulus Compare Sessions] Group '%s': no sessions to process.",
                group_name,
            )
            continue

        # Pool all sessions into one DataFrame
        pooled = pd.concat([s["df"] for s in session_data], ignore_index=True)

        # =================================================================
        # Write pooled CSV
        # =================================================================
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        pooled.to_csv(group_dir / "pooled_feature_summary.csv", index=False)
        logger.info(
            "[Stimulus Compare Sessions] Group '%s': wrote pooled_feature_summary.csv "
            "(%d rows, %d sessions).",
            group_name, len(pooled), len(session_data),
        )

        # =================================================================
        # Compute global y-limits per feature column
        # =================================================================
        ylims = _compute_ylims(pooled, resolved_cols)

        session_ids = [s["session_id"] for s in session_data]
        colors = assign_session_colors(session_ids)

        # =================================================================
        # Pass 2 — Render
        # =================================================================
        gesture_subsets = ["all"] + list(GESTURE_TYPES)

        for gesture in gesture_subsets:
            gesture_dir = group_dir / gesture
            gesture_dir.mkdir(parents=True, exist_ok=True)

            for col, label in zip(resolved_cols, display_labels):
                out_png = gesture_dir / f"{col}_session_comparison.png"
                render_feature_session_comparison(
                    df=pooled,
                    feature_col=col,
                    display_label=label,
                    session_ids=session_ids,
                    colors=colors,
                    plot_type=plot_type,
                    output_path=out_png,
                    gesture_type=gesture,
                    ylim=ylims[col],
                )
                print(
                    f"[Stimulus Compare Sessions] {group_name} / {gesture} / {col}: "
                    f"saved {out_png.name}",
                    flush=True,
                )

            # Summary grid for this gesture subset
            grid_png = group_dir / f"{gesture}_feature_grid.png"
            render_feature_summary_grid(
                df=pooled,
                feature_cols=resolved_cols,
                display_labels=display_labels,
                session_ids=session_ids,
                colors=colors,
                plot_type=plot_type,
                output_path=grid_png,
                gesture_type=gesture,
            )
            print(
                f"[Stimulus Compare Sessions] {group_name} / {gesture}: "
                f"saved {grid_png.name}",
                flush=True,
            )

        # =================================================================
        # Write sentinel
        # =================================================================
        _write_sentinel(sentinel, group_name, n_sessions=len(session_data))
        print(
            f"[Stimulus Compare Sessions] {group_name}: done — sentinel written.",
            flush=True,
        )

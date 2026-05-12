import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from utils.should_process_task import should_process_task

matplotlib.use('Agg')

logger = logging.getLogger(__name__)

IFF_METRICS = [
    "touch_count",
    "max_iff",
    "mean_iff",
    "median_iff",
    "std_iff",
    "iff_range",
    "n_active_vertices",
    "hypsometric_integral",
    "coefficient_of_variation",
    "iff_skewness",
    "iff_kurtosis",
    "gini_coefficient",
    "shannon_entropy",
    "perimeter_mm",
    "circularity",
    "eccentricity",
    "hotspot_area_mm2",
    "threshold_area_mm2",
    "convex_hull_area_mm2",
]

DEVIATION_METRICS = [
    "deviation_area_ratio",
    "deviation_hotspot_area_ratio",
    "deviation_mean_iff_ratio",
    "deviation_peak_iff_ratio",
    "deviation_centroid_shift_mm",
    "deviation_centroid_shift_normalized",
    "deviation_vertex_overlap_jaccard",
    "deviation_vertex_containment",
]

_DEVIATION_COLORMAP_CONFIG: dict[str, tuple[str, float | None]] = {
    "deviation_area_ratio": ("RdBu_r", 1.0),
    "deviation_hotspot_area_ratio": ("RdBu_r", 1.0),
    "deviation_mean_iff_ratio": ("RdBu_r", 1.0),
    "deviation_peak_iff_ratio": ("RdBu_r", 1.0),
    "deviation_centroid_shift_mm": ("Reds", None),
    "deviation_centroid_shift_normalized": ("Reds", None),
    "deviation_vertex_overlap_jaccard": ("RdYlGn", None),
    "deviation_vertex_containment": ("RdYlGn", None),
}

_DEVIATION_FIXED_RANGE: dict[str, tuple[float, float]] = {
    "deviation_vertex_overlap_jaccard": (0.0, 1.0),
    "deviation_vertex_containment": (0.0, 1.0),
}

_SHARED_SCALE: dict[str, bool] = {
    "touch_count": False,
}

# touch_count colormap: one discrete color per integer count.
#   1–4  : red    (too few touches, unreliable)
#   5–9  : orange (marginal)
#   10–50: blue→green gradient via 'winter' (good range, clipped above 50)
_TOUCH_COUNT_VMAX = 50
_n_gradient = _TOUCH_COUNT_VMAX - 10 + 1  # 41 steps for counts 10…50

_touch_count_colors = (
    [(1.0, 0.0, 0.0, 1.0)] * 4
    + [(1.0, 0.5, 0.0, 1.0)] * 5
    + list(matplotlib.cm.get_cmap("winter")(np.linspace(0.0, 1.0, _n_gradient)))
)
_TOUCH_COUNT_CMAP = ListedColormap(_touch_count_colors, name="touch_count")
# One boundary per integer: [1, 2, …, 51]; values ≥ 51 are clipped to green.
_TOUCH_COUNT_NORM = BoundaryNorm(
    boundaries=list(range(1, _TOUCH_COUNT_VMAX + 2)),
    ncolors=len(_touch_count_colors),
    clip=True,
)


def _get_shared_range(
    metric_name: str,
    global_metric_ranges: dict[str, tuple[float, float]],
) -> tuple[float | None, float | None]:
    if _SHARED_SCALE.get(metric_name, True):
        return global_metric_ranges[metric_name]
    return None, None


def render_grid_metric_heatmap(
    df: pd.DataFrame,
    metric_name: str,
    x_feature_col: str,
    y_feature_col: str,
    output_path: Path,
    session_id: str,
    gesture_type: str,
    vmin: float | None = None,
    vmax: float | None = None,
    title_suffix: str = "",
    colormap_name: str | None = None,
    center_value: float | None = None,
) -> None:
    pivoted = df.pivot(index=y_feature_col, columns=x_feature_col, values=metric_name)

    if metric_name == "touch_count":
        # Zero means no data — convert to NaN so set_bad renders it as black.
        pivoted = pivoted.where(pivoted > 0)

    n_x = len(pivoted.columns)
    n_y = len(pivoted.index)
    cell_size = 0.65
    fig_w = max(6.0, n_x * cell_size + 3.0)
    fig_h = max(4.0, n_y * cell_size + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    if metric_name == "touch_count":
        cmap = _TOUCH_COUNT_CMAP
        mesh_kwargs: dict = {"norm": _TOUCH_COUNT_NORM}
    elif center_value is not None:
        cmap = plt.get_cmap(colormap_name or "RdBu_r").copy()
        finite_vals = pivoted.values[np.isfinite(pivoted.values)]
        if len(finite_vals) == 0:
            mesh_kwargs = {"vmin": vmin, "vmax": vmax}
        else:
            data_vmin = float(vmin) if vmin is not None else float(np.nanmin(finite_vals))
            data_vmax = float(vmax) if vmax is not None else float(np.nanmax(finite_vals))
            _eps = 1e-9
            if data_vmin >= center_value:
                data_vmin = center_value - _eps
            if data_vmax <= center_value:
                data_vmax = center_value + _eps
            mesh_kwargs = {
                "norm": matplotlib.colors.TwoSlopeNorm(
                    vcenter=center_value, vmin=data_vmin, vmax=data_vmax
                )
            }
    else:
        cmap = plt.get_cmap(colormap_name or "viridis").copy()
        mesh_kwargs = {"vmin": vmin, "vmax": vmax}

    cmap.set_bad(color="black")

    im = ax.pcolormesh(
        pivoted.columns.values,
        pivoted.index.values,
        pivoted.values,
        cmap=cmap,
        shading="auto",
        **mesh_kwargs,
    )

    plt.colorbar(im, ax=ax, label=metric_name.replace("_", " "))

    x_label = x_feature_col.removesuffix("_center").replace("_", " ")
    y_label = y_feature_col.removesuffix("_center").replace("_", " ")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = f"{session_id} / {gesture_type} / {metric_name}"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap: %s", output_path)


def run_population_rf_grid_metrics_visualization(
    input_items: list,
    output_dir: Path,
    force: bool = False,
    extracted_features: list | None = None,
    metrics_base_dir: Path | None = None,
) -> None:
    if not input_items:
        raise ValueError(
            "run_population_rf_grid_metrics_visualization: input_items is empty"
        )

    sentinel = output_dir / "heatmaps_summary.json"

    if metrics_base_dir is None:
        metrics_base_dir = output_dir.parent / "population_rf_grid_metrics"

    csv_paths = []
    for item in input_items:
        session_id: str = item["session_id"]
        session_metrics_dir = metrics_base_dir / session_id
        session_csvs = sorted(session_metrics_dir.glob("population_rf_grid_metrics_*.csv"))
        csv_paths.extend(session_csvs)

    if not should_process_task(
        input_paths=csv_paths if csv_paths else [metrics_base_dir],
        output_paths=[sentinel],
        force=force,
    ):
        return

    metrics_to_render = list(extracted_features) if extracted_features else list(IFF_METRICS)

    all_session_data: list[dict] = []
    for item in input_items:
        session_id: str = item["session_id"]
        session_metrics_dir = metrics_base_dir / session_id

        csv_paths_for_session = sorted(
            session_metrics_dir.glob("population_rf_grid_metrics_*.csv")
        )

        for csv_path in csv_paths_for_session:
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"run_population_rf_grid_metrics_visualization: CSV not found: {csv_path}"
                )

            df = pd.read_csv(csv_path)

            center_cols = [c for c in df.columns if c.endswith("_center")]
            if len(center_cols) != 2:
                raise ValueError(
                    f"run_population_rf_grid_metrics_visualization: expected exactly 2 "
                    f"'_center' columns in {csv_path}, found {len(center_cols)}: {center_cols}"
                )

            gesture_types = df["gesture_type"].unique()
            if len(gesture_types) != 1:
                raise ValueError(
                    f"run_population_rf_grid_metrics_visualization: expected exactly 1 "
                    f"gesture_type per CSV in {csv_path}, found {list(gesture_types)}"
                )

            all_session_data.append({
                "session_id": session_id,
                "df": df,
                "x_feature_col": center_cols[0],
                "y_feature_col": center_cols[1],
                "gesture_type": str(gesture_types[0]),
            })

    has_deviation_columns = any(
        any(col.startswith("deviation_") for col in entry["df"].columns)
        for entry in all_session_data
    )
    if has_deviation_columns:
        for m in DEVIATION_METRICS:
            if m not in metrics_to_render:
                metrics_to_render.append(m)

    session_total_touches: dict[str, int] = {}
    for entry in all_session_data:
        sid = entry["session_id"]
        session_total_touches[sid] = (
            session_total_touches.get(sid, 0)
            + int(entry["df"]["touch_count"].sum())
        )

    global_metric_ranges: dict[str, tuple[float, float]] = {}
    for metric_name in metrics_to_render:
        if metric_name in _DEVIATION_FIXED_RANGE:
            global_metric_ranges[metric_name] = _DEVIATION_FIXED_RANGE[metric_name]
            continue
        all_entries_with_col = [
            entry for entry in all_session_data if metric_name in entry["df"].columns
        ]
        if not all_entries_with_col:
            continue
        vals = pd.concat(
            [entry["df"][metric_name] for entry in all_entries_with_col],
            ignore_index=True,
        )
        metric_min = float(np.nanmin(vals.values))
        metric_max = float(np.nanmax(vals.values))
        global_metric_ranges[metric_name] = (metric_min, metric_max)

    session_ids_seen: set[str] = set()

    for entry in all_session_data:
        session_id = entry["session_id"]
        df = entry["df"]
        x_feature_col = entry["x_feature_col"]
        y_feature_col = entry["y_feature_col"]
        gesture_type = entry["gesture_type"]

        session_ids_seen.add(session_id)

        for metric_name in metrics_to_render:
            if metric_name not in df.columns:
                if metric_name in DEVIATION_METRICS:
                    logger.debug(
                        "Skipping deviation metric '%s' — not present in %s",
                        metric_name, session_id,
                    )
                    continue
                raise ValueError(
                    f"run_population_rf_grid_metrics_visualization: metric "
                    f"'{metric_name}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

            vmin, vmax = _get_shared_range(metric_name, global_metric_ranges)

            if gesture_type == "all_gestures":
                output_path = output_dir / metric_name / "all_gestures" / f"{session_id}.png"
            else:
                safe_gesture = gesture_type.replace(" ", "_")
                output_path = output_dir / metric_name / f"{session_id}_{safe_gesture}.png"

            suffix = ""
            if metric_name == "touch_count":
                suffix = f"session total: {session_total_touches[session_id]}"

            colormap_name: str | None = None
            center_value: float | None = None
            if metric_name in _DEVIATION_COLORMAP_CONFIG:
                colormap_name, center_value = _DEVIATION_COLORMAP_CONFIG[metric_name]
            if metric_name in _DEVIATION_FIXED_RANGE:
                vmin, vmax = _DEVIATION_FIXED_RANGE[metric_name]

            render_grid_metric_heatmap(
                df=df,
                metric_name=metric_name,
                x_feature_col=x_feature_col,
                y_feature_col=y_feature_col,
                output_path=output_path,
                session_id=session_id,
                gesture_type=gesture_type,
                vmin=vmin,
                vmax=vmax,
                title_suffix=suffix,
                colormap_name=colormap_name,
                center_value=center_value,
            )

    session_count = len(session_ids_seen)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sentinel, "w", encoding="utf-8") as fh:
        json.dump({"status": "complete", "session_count": session_count}, fh, indent=4)
    logger.info("Wrote sentinel: %s", sentinel)

import json
import logging
from pathlib import Path

import matplotlib
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
) -> None:
    pivoted = df.pivot(index=y_feature_col, columns=x_feature_col, values=metric_name)

    if metric_name == "touch_count":
        # Zero means no data — convert to NaN so set_bad renders it as black.
        pivoted = pivoted.where(pivoted > 0)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    if metric_name == "touch_count":
        cmap = _TOUCH_COUNT_CMAP
        mesh_kwargs: dict = {"norm": _TOUCH_COUNT_NORM}
    else:
        cmap = plt.get_cmap("viridis").copy()
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


def _render_all_gestures_heatmaps(
    gesture_dfs: list[pd.DataFrame],
    session_id: str,
    x_feature_col: str,
    y_feature_col: str,
    output_dir: Path,
    global_metric_ranges: dict[str, tuple[float, float]],
    metrics_to_render: list[str],
    session_total_touches: int = 0,
) -> None:
    grid_cols = [x_feature_col, y_feature_col]
    stacked = pd.concat(gesture_dfs, ignore_index=True)
    combined = stacked.groupby(grid_cols, sort=False)[list(metrics_to_render)].sum().reset_index()

    for metric_name in metrics_to_render:
        vmin, vmax = _get_shared_range(metric_name, global_metric_ranges)
        output_path = output_dir / metric_name / "all_gestures" / f"{session_id}.png"

        suffix = ""
        if metric_name == "touch_count":
            suffix = f"total count: {session_total_touches}"

        render_grid_metric_heatmap(
            df=combined,
            metric_name=metric_name,
            x_feature_col=x_feature_col,
            y_feature_col=y_feature_col,
            output_path=output_path,
            session_id=session_id,
            gesture_type="all gestures",
            vmin=vmin,
            vmax=vmax,
            title_suffix=suffix,
        )


def run_population_rf_grid_metrics_visualization(
    input_items: list,
    output_dir: Path,
    force: bool = False,
    extracted_features: list | None = None,
) -> None:
    if not input_items:
        raise ValueError(
            "run_population_rf_grid_metrics_visualization: input_items is empty"
        )

    sentinel = output_dir / "heatmaps_summary.json"

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

    metrics_to_render = extracted_features if extracted_features else IFF_METRICS

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

    session_total_touches: dict[str, int] = {}
    for entry in all_session_data:
        sid = entry["session_id"]
        session_total_touches[sid] = (
            session_total_touches.get(sid, 0)
            + int(entry["df"]["touch_count"].sum())
        )

    global_metric_ranges: dict[str, tuple[float, float]] = {}
    for metric_name in metrics_to_render:
        vals = pd.concat(
            [entry["df"][metric_name] for entry in all_session_data],
            ignore_index=True,
        )
        metric_min = float(np.nanmin(vals.values))
        metric_max = float(np.nanmax(vals.values))
        global_metric_ranges[metric_name] = (metric_min, metric_max)

    session_count = 0
    prev_session_id = None
    session_gesture_dfs: list[pd.DataFrame] = []
    session_x_col: str = ""
    session_y_col: str = ""

    for entry in all_session_data:
        session_id = entry["session_id"]
        df = entry["df"]
        x_feature_col = entry["x_feature_col"]
        y_feature_col = entry["y_feature_col"]
        gesture_type = entry["gesture_type"]

        if prev_session_id is not None and session_id != prev_session_id:
            _render_all_gestures_heatmaps(
                session_gesture_dfs, prev_session_id,
                session_x_col, session_y_col,
                output_dir, global_metric_ranges,
                metrics_to_render=metrics_to_render,
                session_total_touches=session_total_touches[prev_session_id],
            )
            session_gesture_dfs = []
            session_count += 1

        prev_session_id = session_id
        session_x_col = x_feature_col
        session_y_col = y_feature_col
        session_gesture_dfs.append(df)

        for metric_name in metrics_to_render:
            if metric_name not in df.columns:
                raise ValueError(
                    f"run_population_rf_grid_metrics_visualization: metric "
                    f"'{metric_name}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

            vmin, vmax = _get_shared_range(metric_name, global_metric_ranges)
            safe_gesture = gesture_type.replace(" ", "_")
            output_path = output_dir / metric_name / f"{session_id}_{safe_gesture}.png"

            suffix = ""
            if metric_name == "touch_count":
                suffix = f"session total: {session_total_touches[session_id]}"

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
            )

    if prev_session_id is not None:
        _render_all_gestures_heatmaps(
            session_gesture_dfs, prev_session_id,
            session_x_col, session_y_col,
            output_dir, global_metric_ranges,
            metrics_to_render=metrics_to_render,
            session_total_touches=session_total_touches[prev_session_id],
        )
        session_count += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sentinel, "w", encoding="utf-8") as fh:
        json.dump({"status": "complete", "session_count": session_count}, fh, indent=4)
    logger.info("Wrote sentinel: %s", sentinel)

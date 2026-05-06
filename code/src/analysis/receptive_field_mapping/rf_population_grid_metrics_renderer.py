import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from utils.should_process_task import should_process_task

matplotlib.use('Agg')

logger = logging.getLogger(__name__)

IFF_METRICS = [
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
]


def render_grid_metric_heatmap(
    df: pd.DataFrame,
    metric_name: str,
    x_feature_col: str,
    y_feature_col: str,
    output_path: Path,
    session_id: str,
    gesture_type: str,
) -> None:
    pivoted = df.pivot(index=y_feature_col, columns=x_feature_col, values=metric_name)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(
        pivoted.columns.values,
        pivoted.index.values,
        pivoted.values,
        cmap=cmap,
        shading="auto",
    )

    plt.colorbar(im, ax=ax, label=metric_name.replace("_", " "))

    x_label = x_feature_col.removesuffix("_center").replace("_", " ")
    y_label = y_feature_col.removesuffix("_center").replace("_", " ")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{session_id} / {gesture_type} / {metric_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", output_path)


def run_population_rf_grid_metrics_visualization(
    input_items: list,
    output_dir: Path,
    force: bool = False,
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

    session_count = 0
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

            x_feature_col = center_cols[0]
            y_feature_col = center_cols[1]

            gesture_types = df["gesture_type"].unique()
            if len(gesture_types) != 1:
                raise ValueError(
                    f"run_population_rf_grid_metrics_visualization: expected exactly 1 "
                    f"gesture_type per CSV in {csv_path}, found {list(gesture_types)}"
                )
            gesture_type = str(gesture_types[0])

            for metric_name in IFF_METRICS:
                if metric_name not in df.columns:
                    raise ValueError(
                        f"run_population_rf_grid_metrics_visualization: metric "
                        f"'{metric_name}' not found in {csv_path}. "
                        f"Available columns: {list(df.columns)}"
                    )

                safe_gesture = gesture_type.replace(" ", "_")
                output_path = output_dir / metric_name / f"{session_id}_{safe_gesture}.png"

                render_grid_metric_heatmap(
                    df=df,
                    metric_name=metric_name,
                    x_feature_col=x_feature_col,
                    y_feature_col=y_feature_col,
                    output_path=output_path,
                    session_id=session_id,
                    gesture_type=gesture_type,
                )

        session_count += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sentinel, "w", encoding="utf-8") as fh:
        json.dump({"status": "complete", "session_count": session_count}, fh, indent=4)
    logger.info("Wrote sentinel: %s", sentinel)

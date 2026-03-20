# matrix_generation.py
import logging
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Import local modules
from .touch_config import DISCRETIZATION_CONFIG, get_discretization_config, KINEMATIC_SIGNALS
from .reporting import VisualReportingStrategy
from utils.should_process_task import should_process_task, clean_task_outputs

def generate_touch_summary_matrix(
    input_files: List[Path], 
    output_file: Path, 
    config: Optional[Dict] = None,
    show: bool = True,
    log_scale: bool = True,
    log_axis: bool = True,
    force: bool = False
) -> Path:
    """
    Aggregates multiple single-touch analysis CSVs into a single matrix (COUNT aggregation).
    """
    if not should_process_task(
        input_paths=input_files,
        output_paths=[output_file],
        force=force,
    ):
        logging.info(f"Skipping Touch Summary Matrix (up-to-date): {output_file.name}")
        return output_file
    clean_task_outputs(output_file)
    return _generate_matrix_internal(
        input_files, output_file, config, show, log_scale, log_axis, mode="count"
    )

def generate_ap_efficacy_matrix(
    input_files: List[Path], 
    output_file: Path, 
    config: Optional[Dict] = None,
    show: bool = True,
    force: bool = False
) -> Path:
    """
    Generates a matrix showing the RATIO (0.0 to 1.0) of single touches
    that elicited an action potential per condition. (MEAN aggregation).
    """
    if not should_process_task(
        input_paths=input_files,
        output_paths=[output_file],
        force=force,
    ):
        logging.info(f"Skipping AP Efficacy Matrix (up-to-date): {output_file.name}")
        return output_file
    clean_task_outputs(output_file)
    # Note: Log scale is False for efficacy (ratios 0-1 don't work well with log colors)
    return _generate_matrix_internal(
        input_files, output_file, config, show, log_scale=False, log_axis=True, mode="efficacy"
    )

def _detect_kinematic_aggregations(columns: List[str]) -> List[str]:
    """Return sorted aggregation suffixes found in columns matching {signal}_{agg}."""
    col_set = set(columns)
    found: set[str] = set()
    for col in col_set:
        for sig in KINEMATIC_SIGNALS:
            prefix = f'{sig}_'
            if col.startswith(prefix):
                agg = col[len(prefix):]
                if agg:
                    found.add(agg)
    return sorted(found)


def _generate_matrix_internal(
    input_files: List[Path],
    output_file: Path,
    config: Optional[Dict],
    show: bool,
    log_scale: bool,
    log_axis: bool,
    mode: str
) -> Path:
    """
    Internal shared logic for matrix generation.

    Detects kinematic aggregations present in the data and generates one full
    set of outputs (matrix CSV + heatmaps) per aggregation.  When a single
    aggregation is present the output path is unchanged (backward compatible).
    Multiple aggregations produce files named ``{stem}_{agg}{suffix}``.
    """
    if not input_files:
        logging.warning(f"No input files provided for {mode} matrix generation.")
        return output_file

    logging.info(f"Generating {mode} matrix for {len(input_files)} files...")

    # 1. Load and Tag Data
    all_data = []
    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            if mode == "efficacy" and 'spike_elicited' not in df.columns:
                continue
            match = re.search(r'(ST\d+-\d+)', file_path.name)
            file_id = match.group(1) if match else file_path.stem
            df['source_file_id'] = file_id
            all_data.append(df)
        except Exception as e:
            logging.error(f"Error loading {file_path.name}: {e}")

    if not all_data:
        logging.warning("No valid data found to generate matrix.")
        return output_file

    full_df = pd.concat(all_data, ignore_index=True)

    # 2. Detect aggregations; fall back to [None] when no kinematic columns present
    aggregations = _detect_kinematic_aggregations(list(full_df.columns))
    agg_list: List[Optional[str]] = aggregations if aggregations else [None]
    multi = len(agg_list) > 1

    # 3. Per-aggregation: visualizations + matrix
    for agg in agg_list:
        agg_output_file = (
            output_file.parent / f"{output_file.stem}_{agg}{output_file.suffix}"
            if multi else output_file
        )

        # Resolve config for this aggregation
        if config is not None:
            agg_config = config
        elif agg is not None:
            agg_config = get_discretization_config(agg)
        else:
            agg_config = {'continuous_vars': {}, 'categorical_vars': ['type_metadata', 'direction']}

        # Visualizations (skipped when no kinematic aggregation is available)
        if show and not full_df.empty and agg is not None:
            viz_dir = output_file.parent / agg if multi else output_file.parent
            _run_advanced_visualizations(full_df, viz_dir, log_scale, log_axis, mode, agg)

        # Variable Encoding
        legend_data = []
        hierarchy_order = ['type_metadata']
        if agg is not None:
            hierarchy_order += [f'velocity_{agg}', f'depth_{agg}', f'area_{agg}']

        for col, params in agg_config['continuous_vars'].items():
            if col not in full_df.columns:
                continue
            new_col_name = f"{col}_code"
            try:
                if params['method'] == 'qcut':
                    cat_series = pd.qcut(full_df[col], q=params['q'], duplicates='drop')
                elif params['method'] == 'cut':
                    cat_series = pd.cut(full_df[col], bins=params['bins'], duplicates='drop')
                else:
                    continue
                full_df[new_col_name] = cat_series.cat.codes
                for idx, interval in enumerate(cat_series.cat.categories):
                    legend_data.append({"Dimension": col, "Code": idx, "Value/Range": str(interval)})
            except Exception:
                full_df[new_col_name] = -1

        for col in agg_config['categorical_vars']:
            if col in full_df.columns:
                new_col_name = f"{col}_code"
                cat_series = full_df[col].fillna('unknown').astype('category')
                full_df[new_col_name] = cat_series.cat.codes
                for idx, label in enumerate(cat_series.cat.categories):
                    legend_data.append({"Dimension": col, "Code": idx, "Value/Range": str(label)})

        # Hierarchical Matrix
        pivot_cols = [
            full_df[f"{col}_code"]
            for col in hierarchy_order
            if f"{col}_code" in full_df.columns
        ]
        if not pivot_cols:
            continue

        if mode == "count":
            matrix_df = pd.crosstab(
                index=full_df['source_file_id'],
                columns=pivot_cols,
                rownames=['source_file_id'],
                colnames=hierarchy_order,
            )
        else:  # efficacy
            matrix_df = pd.pivot_table(
                full_df,
                values='spike_elicited',
                index='source_file_id',
                columns=pivot_cols,
                aggfunc='mean',
                fill_value=0,
            )

        agg_output_file.parent.mkdir(parents=True, exist_ok=True)
        matrix_df.to_csv(agg_output_file)
        pd.DataFrame(legend_data).to_csv(
            agg_output_file.parent / f"{agg_output_file.stem}_legend.csv", index=False
        )
        logging.info(f"Saved {mode} matrix to {agg_output_file}")

    return output_file

def _run_advanced_visualizations(full_df, output_dir, log_scale, log_axis, mode, aggregation: str):
    """
    Runs heatmap generation for a given kinematic *aggregation* suffix.
    Supports both count and efficacy modes. Skips silently if the required
    columns are not present in *full_df*.
    """
    logging.info(f"Generating advanced visualizations (Mode: {mode}, Agg: {aggregation})...")
    heat_x = f'velocity_{aggregation}'
    heat_y1 = f'depth_{aggregation}'
    heat_y2 = f'area_{aggregation}'
    facet_type = 'type_metadata'
    pop_col = 'source_file_id'

    value_col = None
    if mode == "efficacy":
        value_col = "spike_elicited"

    required_cols = [heat_x, heat_y1, heat_y2, facet_type]
    if value_col:
        required_cols.append(value_col)

    if all(c in full_df.columns for c in required_cols):
        viz_strategy = VisualReportingStrategy(output_dir=output_dir)
        viz_strategy.generate_population_heatmaps(
            full_df,
            x_col=heat_x,
            y_col_1=heat_y1,
            y_col_2=heat_y2,
            pop_col=pop_col,
            type_col=facet_type,
            log_scale=log_scale,
            log_axis=log_axis,
            mode=mode,
            value_col=value_col,
        )
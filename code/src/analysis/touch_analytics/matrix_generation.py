# matrix_generation.py
import logging
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Import local modules
from .touch_config import DISCRETIZATION_CONFIG
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
    """
    if not input_files:
        logging.warning(f"No input files provided for {mode} matrix generation.")
        return output_file

    if config is None:
        config = DISCRETIZATION_CONFIG

    logging.info(f"Generating {mode} matrix for {len(input_files)} files...")
    
    all_data = []

    # 1. Load and Tag Data
    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # Validation for efficacy mode
            if mode == "efficacy" and 'spike_elicited' not in df.columns:
                continue

            # Extract ST ID
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

    # --- VISUALIZATION ---
    # Modified: Visualizations now run for both "count" and "efficacy" modes
    if show and not full_df.empty:
        _run_advanced_visualizations(full_df, output_file.parent, log_scale, log_axis, mode)
    # ---------------------

    # 2. Variable Encoding
    legend_data = []
    hierarchy_order = ['type_metadata', 'max_velocity', 'max_depth', 'max_contact_area']
    
    # Process Continuous Vars
    for col, params in config['continuous_vars'].items():
        if col not in full_df.columns: continue
        new_col_name = f"{col}_code"
        try:
            if params['method'] == 'qcut':
                cat_series = pd.qcut(full_df[col], q=params['q'], duplicates='drop')
            elif params['method'] == 'cut':
                cat_series = pd.cut(full_df[col], bins=params['bins'], duplicates='drop')
            else: continue

            full_df[new_col_name] = cat_series.cat.codes
            for idx, interval in enumerate(cat_series.cat.categories):
                legend_data.append({"Dimension": col, "Code": idx, "Value/Range": str(interval)})
        except Exception:
            full_df[new_col_name] = -1

    # Process Categorical Vars
    for col in config['categorical_vars']:
        if col in full_df.columns:
            new_col_name = f"{col}_code"
            cat_series = full_df[col].fillna('unknown').astype('category')
            full_df[new_col_name] = cat_series.cat.codes
            for idx, label in enumerate(cat_series.cat.categories):
                legend_data.append({"Dimension": col, "Code": idx, "Value/Range": str(label)})

    # 3. Create Hierarchical Matrix
    pivot_cols = []
    for col in hierarchy_order:
        code_col = f"{col}_code"
        if code_col in full_df.columns:
            pivot_cols.append(full_df[code_col])

    if not pivot_cols:
        return output_file

    if mode == "count":
        # Crosstab counts occurrences
        matrix_df = pd.crosstab(
            index=full_df['source_file_id'], 
            columns=pivot_cols,
            rownames=['source_file_id'],
            colnames=hierarchy_order
        )
    elif mode == "efficacy":
        # Pivot Table averages the boolean outcome (Ratio)
        matrix_df = pd.pivot_table(
            full_df,
            values='spike_elicited',
            index='source_file_id',
            columns=pivot_cols,
            aggfunc='mean',
            fill_value=0
        )

    # 4. Save Outputs
    output_file.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(output_file)
    pd.DataFrame(legend_data).to_csv(output_file.parent / f"{output_file.stem}_legend.csv", index=False)
    
    logging.info(f"Saved {mode} matrix to {output_file}")

    return output_file

def _run_advanced_visualizations(full_df, output_dir, log_scale, log_axis, mode):
    """
    Runs heatmap generation. Now supports both count and efficacy modes.
    """
    logging.info(f"Generating advanced visualizations (Mode: {mode})...")
    heat_x = 'max_velocity'
    heat_y1 = 'max_depth'
    heat_y2 = 'max_contact_area'
    facet_type = 'type_metadata'
    pop_col = 'source_file_id'
    
    # Define value column for aggregation based on mode
    value_col = None
    if mode == "efficacy":
        value_col = "spike_elicited"
    
    # Check for necessary columns
    required_cols = [heat_x, heat_y1, heat_y2, facet_type]
    if value_col:
        required_cols.append(value_col)

    cols_exist = all(c in full_df.columns for c in required_cols)
    
    if cols_exist:
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
            value_col=value_col
        )
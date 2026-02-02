# matrix_generation.py
import logging
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Import local modules
from .touch_config import DISCRETIZATION_CONFIG
from .reporting import TableContext, GreatTablesStrategy, VisualReportingStrategy

def generate_touch_summary_matrix(
    input_files: List[Path], 
    output_file: Path, 
    config: Optional[Dict] = None,
    show: bool = True,
    log_scale: bool = True,
    log_axis: bool = True,
    heatmap_only: bool = True
) -> Path:
    """
    Aggregates multiple single-touch analysis CSVs into a single matrix.
    
    Args:
        input_files: List of paths to the analyzed summary CSVs.
        output_file: Path where the resulting matrix CSV should be saved.
        config: Configuration dict. Defaults to DISCRETIZATION_CONFIG.
        show: If True, renders the HTML report and Plots.
        log_scale: If True (default), heatmaps use a logarithmic color scale (Z-axis).
        log_axis: If True (default), heatmap spatial axes use logarithmic binning (X/Y-axis).
        heatmap_only: If True, generates only heatmaps (skips Parallel Coords and HTML report).
    """
    if not input_files:
        logging.warning("No input files provided for matrix generation.")
        return output_file

    if config is None:
        config = DISCRETIZATION_CONFIG

    logging.info(f"Generating summary matrix for {len(input_files)} files...")
    
    all_data = []

    # 1. Load and Tag Data
    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # Extract ST ID
            match = re.search(r'(ST\d+-\d+)', file_path.name)
            if match:
                file_id = match.group(1)
            else:
                file_id = file_path.stem
                
            df['source_file_id'] = file_id
            all_data.append(df)
            
        except Exception as e:
            logging.error(f"Error loading {file_path.name} for matrix: {e}")

    if not all_data:
        logging.warning("No valid data found to generate matrix.")
        return output_file

    full_df = pd.concat(all_data, ignore_index=True)

    # --- ADVANCED VISUALIZATION INTEGRATION ---
    # We generate plots using the raw continuous data before discretization.
    if show and not full_df.empty:
        logging.info("Generating advanced visualizations...")
        
        # Define dimension mapping based on Touch Data domain
        heat_x = 'max_velocity'
        heat_y1 = 'max_depth'
        heat_y2 = 'max_contact_area' # Added as the secondary Y-axis variable
        facet_type = 'type_metadata'
        pop_col = 'source_file_id'
        
        # Parallel Coords Variables: Contact Area, Depth, Velocity (as requested)
        parallel_cols = ['max_contact_area', 'max_depth', 'max_velocity']
        
        # Check column existence
        cols_exist = all(c in full_df.columns for c in [heat_x, heat_y1, heat_y2, facet_type] + parallel_cols)
        
        if cols_exist:
            viz_strategy = VisualReportingStrategy(output_dir=output_file.parent)
            
            # 1. Per-Population Faceted Heatmaps (2x2 Grid)
            # This is generated if show=True, regardless of heatmap_only
            viz_strategy.generate_population_heatmaps(
                full_df, 
                x_col=heat_x, 
                y_col_1=heat_y1,
                y_col_2=heat_y2,
                pop_col=pop_col,
                type_col=facet_type,
                log_scale=log_scale,
                log_axis=log_axis
            )
            
            # 2. Parallel Coordinates (Split View: Tap vs Stroke)
            # SKIPPED if heatmap_only is True
            if not heatmap_only:
                viz_strategy.generate_parallel_coordinates(
                    full_df, 
                    cols=parallel_cols, 
                    pop_col=pop_col,
                    type_col=facet_type
                )
            else:
                logging.info("Skipping Parallel Coordinates (heatmap_only=True)")
        else:
            logging.warning("Skipping visualizations: Required columns missing in source data.")
    # ------------------------------------------

    # 2. Variable Encoding and Legend Generation
    legend_data = []
    
    # Hierarchy: Type -> Velocity -> Depth -> Contact area
    hierarchy_order = ['type_metadata', 'max_velocity', 'max_depth', 'max_contact_area']
    
    missing_cols = [c for c in hierarchy_order if c not in full_df.columns]
    if missing_cols:
        logging.error(f"Critical columns for hierarchy missing: {missing_cols}")
        return output_file

    # A. Process Continuous Variables
    for col, params in config['continuous_vars'].items():
        if col not in full_df.columns:
            continue
            
        new_col_name = f"{col}_code"
        
        try:
            if params['method'] == 'qcut':
                cat_series = pd.qcut(
                    full_df[col], 
                    q=params['q'], 
                    duplicates='drop'
                )
            elif params['method'] == 'cut':
                cat_series = pd.cut(
                    full_df[col], 
                    bins=params['bins'], 
                    duplicates='drop'
                )
            else:
                continue

            full_df[new_col_name] = cat_series.cat.codes
            
            categories = cat_series.cat.categories
            for idx, interval in enumerate(categories):
                legend_data.append({
                    "Dimension": col,
                    "Code": idx,
                    "Value/Range": str(interval)
                })

        except Exception as e:
            logging.warning(f"Discretization failed for {col}: {e}")
            full_df[new_col_name] = -1

    # B. Process Categorical Variables
    for col in config['categorical_vars']:
        if col in full_df.columns:
            new_col_name = f"{col}_code"
            
            cat_series = full_df[col].fillna('unknown').astype('category')
            full_df[new_col_name] = cat_series.cat.codes
            
            for idx, label in enumerate(cat_series.cat.categories):
                legend_data.append({
                    "Dimension": col,
                    "Code": idx,
                    "Value/Range": str(label)
                })

    # 3. Create Hierarchical Matrix
    pivot_cols = []
    for col in hierarchy_order:
        code_col = f"{col}_code"
        if code_col in full_df.columns:
            pivot_cols.append(full_df[code_col])
        else:
            logging.warning(f"Code column {code_col} missing during pivot creation.")

    if not pivot_cols:
        logging.error("No valid columns found for pivoting.")
        return output_file

    matrix_df = pd.crosstab(
        index=full_df['source_file_id'], 
        columns=pivot_cols,
        rownames=['source_file_id'],
        colnames=hierarchy_order
    )

    # 4. Save Outputs
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    matrix_df.to_csv(output_file)
    logging.info(f"Summary matrix saved to {output_file} with shape {matrix_df.shape}")
    
    legend_df = pd.DataFrame(legend_data)
    legend_path = output_file.parent / f"{output_file.stem}_legend.csv"
    legend_df.to_csv(legend_path, index=False)
    logging.info(f"Matrix legend saved to {legend_path}")

    # 5. Advanced Visualization Strategies (HTML Report)
    # SKIPPED if heatmap_only is True
    if show and not matrix_df.empty and not heatmap_only:
        display_df = matrix_df.copy()
        
        if isinstance(display_df.columns, pd.MultiIndex):
            display_df.columns = [
                ' | '.join(map(str, col)).strip() 
                for col in display_df.columns.values
            ]
        
        render_df = display_df.reset_index()
        title = "Batch Analysis Matrix (Hierarchical Data)"
        
        logging.info("Generating HTML Report...")
        html_output_path = output_file.with_suffix(".html")
        report_ctx = TableContext(GreatTablesStrategy(html_output_path))
        report_ctx.execute_render(render_df, title)
    elif heatmap_only:
        logging.info("Skipping HTML Report (heatmap_only=True)")

    return output_file
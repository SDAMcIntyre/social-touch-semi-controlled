# matrix_generation.py
import logging
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Import local modules
from .touch_config import DISCRETIZATION_CONFIG
from .reporting import TableContext, GreatTablesStrategy

def generate_touch_summary_matrix(
    input_files: List[Path], 
    output_file: Path, 
    config: Optional[Dict] = None,
    show: bool = True
) -> Path:
    """
    Aggregates multiple single-touch analysis CSVs into a single matrix.
    
    Features:
    - Encodes variables into integer codes (0, 1, 2...) based on config.
    - Generates a hierarchical MultiIndex column structure:
      Type -> Velocity -> Depth -> Contact Area.
    - Produces a secondary 'legend' CSV mapping codes to their real values.
    - Generates HTML reports via GreatTables (Flattening columns for compatibility).

    Args:
        input_files: List of paths to the analyzed summary CSVs.
        output_file: Path where the resulting matrix CSV should be saved.
        config: Configuration dict. Defaults to DISCRETIZATION_CONFIG.
        show: If True, renders the HTML report.
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

    # 2. Variable Encoding and Legend Generation
    legend_data = []
    
    # Hierarchy: Type -> Velocity -> Depth -> Contact area
    hierarchy_order = ['type_metadata', 'max_velocity', 'max_depth', 'max_contact_area']
    
    # Check for missing columns strictly required for the hierarchy
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

            # Assign integer codes
            full_df[new_col_name] = cat_series.cat.codes
            
            # Extract mapping: Code -> Interval
            categories = cat_series.cat.categories
            for idx, interval in enumerate(categories):
                legend_data.append({
                    "Dimension": col,
                    "Code": idx,
                    "Value/Range": str(interval)
                })

        except Exception as e:
            logging.warning(f"Discretization failed for {col}: {e}")
            full_df[new_col_name] = -1 # Error code

    # B. Process Categorical Variables
    for col in config['categorical_vars']:
        if col in full_df.columns:
            new_col_name = f"{col}_code"
            
            # Convert to category type to get deterministic codes
            cat_series = full_df[col].fillna('unknown').astype('category')
            full_df[new_col_name] = cat_series.cat.codes
            
            # Extract mapping
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

    # Create Matrix with MultiIndex columns
    matrix_df = pd.crosstab(
        index=full_df['source_file_id'], 
        columns=pivot_cols,
        rownames=['source_file_id'],
        colnames=hierarchy_order
    )

    # 4. Save Outputs
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save Matrix (Retains hierarchical MultiIndex in CSV)
    matrix_df.to_csv(output_file)
    logging.info(f"Summary matrix saved to {output_file} with shape {matrix_df.shape}")
    
    # Save Legend
    legend_df = pd.DataFrame(legend_data)
    legend_path = output_file.parent / f"{output_file.stem}_legend.csv"
    legend_df.to_csv(legend_path, index=False)
    logging.info(f"Matrix legend saved to {legend_path}")

    # 5. Advanced Visualization Strategies (HTML Report)
    if show and not matrix_df.empty:
        # ARCHITECTURAL FIX: GreatTables does not support MultiIndex columns.
        # We create a display-specific copy where we flatten the column levels
        # into a single string (e.g., "0 | 1 | 2 | 0") for rendering only.
        
        display_df = matrix_df.copy()
        
        if isinstance(display_df.columns, pd.MultiIndex):
            # Flatten tuples to "Code | Code | Code"
            display_df.columns = [
                ' | '.join(map(str, col)).strip() 
                for col in display_df.columns.values
            ]
        
        # Reset index to make 'source_file_id' a visible column in the HTML table
        render_df = display_df.reset_index()
        title = "Batch Analysis Matrix (Hierarchical Data)"
        
        logging.info("Generating HTML Report...")
        html_output_path = output_file.with_suffix(".html")
        report_ctx = TableContext(GreatTablesStrategy(html_output_path))
        report_ctx.execute_render(render_df, title)

    return output_file
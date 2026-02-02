# reporting.py
import logging
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import Protocol, Union, List, Optional, Tuple, Dict

class TableRenderer(Protocol):
    """
    Protocol defining the contract for table rendering strategies.
    """
    def render(self, df: pd.DataFrame, title: str = "Data Table") -> None:
        ...

class GreatTablesStrategy:
    """
    Renders tables to static HTML using 'great_tables' (GT).
    Optimized for publication-quality reporting.
    """
    def __init__(self, output_file: Union[str, Path]):
        self.output_file = Path(output_file)
        try:
            from great_tables import GT, style, loc
            self.GT = GT
            self.style = style
            self.loc = loc
            self.available = True
        except ImportError:
            logging.warning("Library 'great_tables' not found. HTML report generation disabled.")
            self.available = False

    def render(self, df: pd.DataFrame, title: str = "Data Table") -> None:
        if not self.available:
            return

        try:
            gt_tbl = (
                self.GT(df)
                .tab_header(
                    title=title,
                    subtitle=f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            )
            
            # Apply heatmapping to numeric columns if present
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                gt_tbl = gt_tbl.data_color(
                    columns=numeric_cols,
                    palette=["#ffffff", "#e6f2ff", "#004d99"], # White to Blue
                )

            html_content = gt_tbl.as_raw_html()
            
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            abs_path = self.output_file.resolve()
            logging.info(f"Report generated at: {abs_path}")
            webbrowser.open(f"file://{abs_path}")
            
        except Exception as e:
            logging.error(f"Failed to render HTML report: {e}")

class VisualReportingStrategy:
    """
    Renders static statistical visualizations (Heatmaps, Parallel Coordinates)
    using Matplotlib and Seaborn. Enforces global scaling for comparability.
    """
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set non-interactive backend for headless environments
        plt.switch_backend('Agg') 

    def _save_plot(self, filename: str):
        filepath = self.output_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"Saved visualization: {filepath}")

    def _get_global_edges(self, series: pd.Series, num_bins: int = 10, log_axis: bool = False) -> np.ndarray:
        """
        Calculates fixed bin edges based on the global range of a series.
        Supports both Linear and Geometric (Log) progression.
        """
        if series.empty:
            return np.linspace(0, 1, num_bins + 1)

        if log_axis:
            # For geometric progression, we need strictly positive values
            # Filter valid data
            pos_series = series[series > 0]
            
            if pos_series.empty:
                logging.warning(f"Log axis requested but no positive data found. Falling back to linear.")
                vmin, vmax = series.min(), series.max()
            else:
                vmin = pos_series.min()
                vmax = series.max() # Use overall max
                
                # Check bounds
                if vmin >= vmax:
                    # Not enough spread for log space, or single value
                    vmin = vmin if vmin > 0 else 0.001
                    vmax = vmin * 10 
                    
                try:
                    return np.geomspace(vmin, vmax, num_bins + 1)
                except Exception as e:
                    logging.warning(f"Geomspace calculation failed ({e}). Falling back to linear.")
                    vmin, vmax = series.min(), series.max()
        else:
            vmin = series.min()
            vmax = series.max()

        # Linear Fallback / Default
        if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
            return np.linspace(0, 1, num_bins + 1)
        
        return np.linspace(vmin, vmax, num_bins + 1)

    def generate_population_heatmaps(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_col_1: str, 
        y_col_2: str,
        pop_col: str,
        type_col: str,
        log_scale: bool = True,
        log_axis: bool = True
    ):
        """
        Generates a 2x2 heatmap grid PER POPULATION (source file).
        Includes point counts per heatmap and aggregate sum in the title.
        
        Args:
            log_scale: Toggles LogNorm (color scaling).
            log_axis: Toggles Geometric Binning (spatial axis scaling).
        """
        color_scale_name = "Log" if log_scale else "Linear"
        axis_scale_name = "Log" if log_axis else "Linear"
        
        logging.info(f"Generating Per-Population Heatmaps (Color: {color_scale_name}, Axis: {axis_scale_name})...")
        
        unique_pops = df[pop_col].unique()
        
        # 1. Calculate Global Bin Edges
        num_bins = 20
        x_edges = self._get_global_edges(df[x_col], num_bins, log_axis=log_axis)
        y1_edges = self._get_global_edges(df[y_col_1], num_bins, log_axis=log_axis)
        y2_edges = self._get_global_edges(df[y_col_2], num_bins, log_axis=log_axis)

        x_cats = pd.cut(df[x_col], bins=x_edges, include_lowest=True).cat.categories
        y1_cats = pd.cut(df[y_col_1], bins=y1_edges, include_lowest=True).cat.categories
        y2_cats = pd.cut(df[y_col_2], bins=y2_edges, include_lowest=True).cat.categories

        # 2. Pre-compute Matrices
        global_max_density = 0
        # Cache stores: (pop_id, interaction, y_var) -> (Matrix, Count)
        plot_cache: Dict[Tuple, Tuple[pd.DataFrame, int]] = {} 
        
        # Helper to create an empty zero-filled dataframe with correct structure
        def create_empty_matrix(y_categories):
            return pd.DataFrame(0, index=y_categories, columns=x_cats)

        logging.info("Pre-computing heatmap matrices for global normalization...")
        
        for pop_id in unique_pops:
            pop_df = df[df[pop_col] == pop_id]
            
            for interaction_type in ['tap', 'stroke']:
                subset = pop_df[pop_df[type_col] == interaction_type]
                
                configs = [
                    (y_col_1, y1_edges, y1_cats),
                    (y_col_2, y2_edges, y2_cats)
                ]

                for y_var, y_edges, y_cats in configs:
                    try:
                        current_matrix = None
                        current_count = 0

                        if subset.empty:
                            current_matrix = create_empty_matrix(y_cats)
                            current_count = 0
                        else:
                            # Bin the data
                            x_binned = pd.cut(subset[x_col], bins=x_edges, include_lowest=True)
                            y_binned = pd.cut(subset[y_var], bins=y_edges, include_lowest=True)

                            # Create Crosstab
                            current_matrix = pd.crosstab(y_binned, x_binned, dropna=False)
                            # Ensure full grid exists
                            current_matrix = current_matrix.reindex(index=y_cats, columns=x_cats, fill_value=0)
                            # Calculate count
                            current_count = int(current_matrix.sum().sum())

                        current_max = current_matrix.max().max()
                        if current_max > global_max_density:
                            global_max_density = current_max
                        
                        plot_cache[(pop_id, interaction_type, y_var)] = (current_matrix, current_count)

                    except Exception as e:
                        logging.warning(f"Binning error for {pop_id}/{interaction_type}: {e}")
                        # Fallback to empty matrix structure to ensure plotting logic works
                        plot_cache[(pop_id, interaction_type, y_var)] = (create_empty_matrix(y_cats), 0)

        if global_max_density == 0:
            global_max_density = 1

        # 3. Render Plots
        logging.info(f"Rendering heatmaps (Global Max: {global_max_density})...")

        for pop_id in unique_pops:
            try:
                # Calculate Grand Total for this population before plotting
                # We need to access the counts stored in the cache
                pop_counts = []
                for i_type in ['tap', 'stroke']:
                    for y_v in [y_col_1, y_col_2]:
                        _, c = plot_cache.get((pop_id, i_type, y_v), (None, 0))
                        pop_counts.append(c)
                
                grand_total = sum(pop_counts)

                fig, axes = plt.subplots(2, 2, figsize=(18, 14))
                fig.suptitle(f'Population Analysis: {pop_id} (Axis: {axis_scale_name} | Color: {color_scale_name}) | Total Points: {grand_total}', fontsize=16)
                
                plot_configs = [
                    (0, 0, 'tap', y_col_1, y1_cats, f"Tap: {x_col} vs {y_col_1}"),
                    (0, 1, 'tap', y_col_2, y2_cats, f"Tap: {x_col} vs {y_col_2}"),
                    (1, 0, 'stroke', y_col_1, y1_cats, f"Stroke: {x_col} vs {y_col_1}"),
                    (1, 1, 'stroke', y_col_2, y2_cats, f"Stroke: {x_col} vs {y_col_2}")
                ]

                for row, col, i_type, y_var, y_cats, base_title in plot_configs:
                    ax = axes[row, col]
                    matrix, count = plot_cache.get((pop_id, i_type, y_var))

                    # Logic to ensure consistency:
                    # 1. Determine if empty.
                    # 2. Even if empty, we PLOT the heatmap (to generate the colorbar/layout).
                    # 3. If empty, we MASK the heatmap and add text.
                    
                    is_empty = (matrix.sum().sum() == 0)
                    
                    # Setup Mask: True means "don't show this cell"
                    if is_empty:
                        mask = np.ones_like(matrix) # Mask everything
                        # For LogNorm, we cannot plot 0s without error, even if masked.
                        # We temporarily fill with a safe value (1).
                        plot_data = matrix.replace(0, 1) if log_scale else matrix
                    else:
                        mask = None
                        plot_data = matrix

                    # Configure Heatmap
                    heatmap_kwargs = {
                        'cmap': 'magma',
                        'cbar': True, # ALWAYS True to reserve layout space
                        'ax': ax,
                        'mask': mask
                    }

                    if log_scale:
                        heatmap_kwargs['norm'] = LogNorm(vmin=1, vmax=global_max_density)
                    else:
                        heatmap_kwargs['vmin'] = 0
                        heatmap_kwargs['vmax'] = global_max_density

                    # Render
                    
                    sns.heatmap(plot_data, **heatmap_kwargs)
                    
                    # Post-Plot Decoration
                    # Append count to the individual subplot title
                    ax.set_title(f"{base_title} (n={count})")
                    ax.invert_yaxis()
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_var)

                    # Overlay "No Data" if needed
                    if is_empty:
                        ax.text(0.5, 0.5, "No Data", 
                                ha='center', va='center', 
                                transform=ax.transAxes,
                                fontsize=12, color='gray')

                    # Format Labels
                    def format_labels(cats):
                        return [f"{c.mid:.2f}" for c in cats]

                    ax.set_xticklabels(format_labels(x_cats), rotation=45, ha='right')
                    ax.set_yticklabels(format_labels(y_cats), rotation=0)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                safe_name = str(pop_id).replace(" ", "_").replace("/", "-")
                self._save_plot(f"heatmap_{safe_name}.png")

            except Exception as e:
                logging.error(f"Failed to render heatmap for {pop_id}: {e}")

    def generate_parallel_coordinates(self, df: pd.DataFrame, cols: List[str], pop_col: str, type_col: str):
        """
        Generates two parallel coordinates plots (Tap vs Stroke).
        """
        logging.info("Generating Parallel Coordinates Plots (Tap/Stroke)...")
        try:
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols or pop_col not in df.columns or type_col not in df.columns:
                logging.warning("Missing columns for parallel coordinates.")
                return

            # Global Normalization (Min-Max)
            data_norm = df.copy()
            data_norm = data_norm.dropna(subset=valid_cols + [pop_col, type_col])
            
            for col in valid_cols:
                min_val = data_norm[col].min()
                max_val = data_norm[col].max()
                if max_val - min_val != 0:
                    data_norm[col] = (data_norm[col] - min_val) / (max_val - min_val)
                else:
                    data_norm[col] = 0.0

            interaction_types = ['tap', 'stroke']
            
            for i_type in interaction_types:
                subset = data_norm[data_norm[type_col] == i_type]
                
                if subset.empty:
                    logging.info(f"No data for Parallel Coordinates: {i_type}")
                    continue
                    
                plt.figure(figsize=(14, 7))
                
                parallel_coordinates(
                    subset[valid_cols + [pop_col]], 
                    class_column=pop_col, 
                    colormap='viridis', 
                    alpha=0.6,
                    linewidth=1.5
                )
                
                plt.title(f"Parallel Coordinates: {i_type.capitalize()} (Color: Population)")
                plt.ylabel("Normalized Value (Global 0-1)")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Population")
                
                self._save_plot(f"parallel_coordinates_{i_type}.png")

        except Exception as e:
            logging.error(f"Failed to generate parallel coordinates: {e}")

class TableContext:
    """
    Context manager for executing the selected rendering strategy.
    """
    def __init__(self, strategy: TableRenderer) -> None:
        self._strategy = strategy

    def execute_render(self, df: pd.DataFrame, title: str) -> None:
        self._strategy.render(df, title)
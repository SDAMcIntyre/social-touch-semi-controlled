# reporting.py
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import Union, Tuple, Dict, Optional

class VisualReportingStrategy:
    """
    Renders static statistical visualizations (Heatmaps only)
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
        log_axis: bool = True,
        mode: str = "count",
        value_col: Optional[str] = None
    ):
        """
        Generates a 2x2 heatmap grid PER POPULATION (source file).
        
        Args:
            log_scale: Toggles LogNorm (color scaling).
            log_axis: Toggles Geometric Binning (spatial axis scaling).
            mode: "count" (frequency) or "efficacy" (ratio).
            value_col: Column to average if mode is "efficacy".
        """
        color_scale_name = "Log" if log_scale else "Linear"
        axis_scale_name = "Log" if log_axis else "Linear"
        
        logging.info(f"Generating Per-Population Heatmaps [{mode.upper()}] (Color: {color_scale_name}, Axis: {axis_scale_name})...")
        
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
        
        # Cache stores: (pop_id, interaction, y_var) -> (ValueMatrix, Count)
        plot_cache: Dict[Tuple, Tuple[pd.DataFrame, int]] = {} 
        
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
                            fill_val = 0 if mode == "count" else np.nan
                            current_matrix = pd.DataFrame(fill_val, index=y_cats, columns=x_cats)
                            current_count = 0
                        else:
                            x_binned = pd.cut(subset[x_col], bins=x_edges, include_lowest=True)
                            y_binned = pd.cut(subset[y_var], bins=y_edges, include_lowest=True)

                            count_matrix_raw = pd.crosstab(y_binned, x_binned, dropna=False)
                            current_count = int(count_matrix_raw.sum().sum())

                            if mode == "efficacy" and value_col:
                                mean_series = subset.groupby([y_binned, x_binned], observed=False)[value_col].mean()
                                current_matrix = mean_series.unstack(fill_value=np.nan)
                                current_matrix = current_matrix.round(2)
                            else:
                                current_matrix = count_matrix_raw

                            fill_val = 0 if mode == "count" else np.nan
                            current_matrix = current_matrix.reindex(index=y_cats, columns=x_cats, fill_value=fill_val)

                        current_max = current_matrix.max().max()
                        if pd.notna(current_max) and current_max > global_max_density:
                            global_max_density = current_max
                        
                        plot_cache[(pop_id, interaction_type, y_var)] = (current_matrix, current_count)

                    except Exception as e:
                        logging.warning(f"Binning error for {pop_id}/{interaction_type}: {e}")
                        fill_val = 0 if mode == "count" else np.nan
                        plot_cache[(pop_id, interaction_type, y_var)] = (pd.DataFrame(fill_val, index=y_cats, columns=x_cats), 0)

        # Set Scale defaults
        if mode == "efficacy":
            global_max_density = 1.0 
        elif global_max_density == 0:
            global_max_density = 1

        # 3. Render Plots
        logging.info(f"Rendering heatmaps (Global Max: {global_max_density})...")

        for pop_id in unique_pops:
            try:
                pop_counts = []
                for i_type in ['tap', 'stroke']:
                    for y_v in [y_col_1, y_col_2]:
                        _, c = plot_cache.get((pop_id, i_type, y_v), (None, 0))
                        pop_counts.append(c)
                
                grand_total = sum(pop_counts)

                fig, axes = plt.subplots(2, 2, figsize=(18, 14))
                title_metric = "Ratio" if mode == "efficacy" else "Count"
                fig.suptitle(f'Population Analysis: {pop_id} (Axis: {axis_scale_name} | Color: {color_scale_name}) | Total Points: {grand_total}', fontsize=16)
                
                plot_configs = [
                    (0, 0, 'tap', y_col_1, y1_cats, f"Tap: {x_col} vs {y_col_1}"),
                    (0, 1, 'stroke', y_col_1, y1_cats, f"Stroke: {x_col} vs {y_col_1}"),
                    (1, 0, 'tap', y_col_2, y2_cats, f"Tap: {x_col} vs {y_col_2}"),
                    (1, 1, 'stroke', y_col_2, y2_cats, f"Stroke: {x_col} vs {y_col_2}")
                ]

                for row, col, i_type, y_var, y_cats, base_title in plot_configs:
                    ax = axes[row, col]
                    matrix, count = plot_cache.get((pop_id, i_type, y_var))

                    if mode == "efficacy":
                        is_empty = matrix.isna().all().all()
                    else:
                        is_empty = (matrix.sum().sum() == 0)
                    
                    if is_empty:
                        mask = np.ones_like(matrix)
                        plot_data = matrix
                    else:
                        mask = None
                        plot_data = matrix

                    heatmap_kwargs = {
                        'cmap': 'magma',
                        'cbar': True, 
                        'ax': ax,
                        'mask': mask
                    }

                    if mode == "efficacy":
                        heatmap_kwargs['vmin'] = 0.0
                        heatmap_kwargs['vmax'] = 1.0
                    else:
                        if log_scale:
                            safe_data = plot_data.replace(0, 1)
                            heatmap_kwargs['norm'] = LogNorm(vmin=1, vmax=global_max_density)
                            if not is_empty:
                                plot_data = safe_data
                        else:
                            heatmap_kwargs['vmin'] = 0
                            heatmap_kwargs['vmax'] = global_max_density

                    # Render
                    sns.heatmap(plot_data, **heatmap_kwargs)
                    
                    # ---------------------------------------------------------
                    # MODIFICATION: Force Max Value on Colorbar
                    # ---------------------------------------------------------
                    try:
                        cbar = ax.collections[0].colorbar
                        # Get current ticks generated by matplotlib
                        current_ticks = cbar.get_ticks()
                        
                        # Determine the upper bound we want to enforce
                        target_max = 1.0 if mode == "efficacy" else global_max_density
                        
                        # Filter existing ticks that might be out of bounds (ghost ticks)
                        # and ensure we don't have duplicates close to target_max
                        new_ticks = [t for t in current_ticks if t < target_max]
                        
                        # Add the global max explicitly
                        new_ticks.append(target_max)
                        
                        # Remove values < 1 for Count/Log mode to avoid log(0) issues or clutter
                        if mode == "count" and log_scale:
                            new_ticks = [t for t in new_ticks if t >= 1]
                        elif mode == "count":
                            new_ticks = [t for t in new_ticks if t >= 0]
                            
                        # Apply new ticks
                        cbar.set_ticks(new_ticks)
                        
                        # Format labels
                        if mode == "count":
                            # Integers for count
                            cbar.set_ticklabels([f"{int(t)}" for t in new_ticks])
                        else:
                            # Floats for efficacy
                            cbar.set_ticklabels([f"{t:.2f}" for t in new_ticks])
                            
                    except Exception as e:
                        logging.warning(f"Could not adjust colorbar ticks: {e}")
                    # ---------------------------------------------------------

                    ax.set_title(f"{base_title} ({title_metric}, n={count})")
                    ax.invert_yaxis()
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_var)

                    if is_empty:
                        ax.text(0.5, 0.5, "No Data", 
                                ha='center', va='center', 
                                transform=ax.transAxes,
                                fontsize=12, color='gray')

                    def format_labels(cats):
                        return [f"{c.mid:.2f}" for c in cats]

                    ax.set_xticklabels(format_labels(x_cats), rotation=45, ha='right')
                    ax.set_yticklabels(format_labels(y_cats), rotation=0)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                safe_name = str(pop_id).replace(" ", "_").replace("/", "-")
                self._save_plot(f"heatmap_{mode}_{safe_name}.png")

            except Exception as e:
                logging.error(f"Failed to render heatmap for {pop_id}: {e}")

    def generate_touch_density_heatmap(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_cols: list,
        type_col: str = 'gesture_type',
        num_bins: int = 20,
        log_axis: bool = False,
        title_suffix: str = '',
        filename: str = 'touch_density.png',
        global_edges: Optional[Dict[str, np.ndarray]] = None,
        global_max_count: Optional[int] = None,
    ):
        """
        Render an N×3 per-session touch density heatmap and save to disk.

        Layout: N rows (one per entry in *y_cols*) × 3 columns (tap | stroke_proximal | stroke_distal).
        The shared X-axis is *x_col* (typically the highest-variance feature).
        Color encodes touch count with LogNorm scaling; bins with zero touches
        are masked.

        Args:
            df:           DataFrame for a single session.
            x_col:        Feature to use on the shared X-axis.
            y_cols:       Remaining feature columns (one row per entry).
            type_col:     Column that distinguishes tap vs. stroke touches.
            num_bins:     Number of bins per axis.
            log_axis:     Use geometric (log) bin spacing when True.
            title_suffix: Appended to the figure suptitle (e.g. session ID).
            filename:     Output filename passed to ``_save_plot``.
        """
        n_rows = len(y_cols)
        if n_rows == 0:
            logging.warning("generate_touch_density_heatmap: no y_cols provided, skipping.")
            return

        # 1. Bin edges — use pre-computed global edges when provided, otherwise
        #    compute from this session's data subset (backward-compatible fallback).
        if global_edges is not None and x_col in global_edges:
            x_edges = global_edges[x_col]
        else:
            x_edges = self._get_global_edges(df[x_col], num_bins, log_axis=log_axis)
        x_cats = pd.cut(df[x_col], bins=x_edges, include_lowest=True).cat.categories

        y_edges_list = []
        y_cats_list = []
        for y_col in y_cols:
            if global_edges is not None and y_col in global_edges:
                edges = global_edges[y_col]
            else:
                edges = self._get_global_edges(df[y_col], num_bins, log_axis=log_axis)
            cats = pd.cut(df[y_col], bins=edges, include_lowest=True).cat.categories
            y_edges_list.append(edges)
            y_cats_list.append(cats)

        interaction_types = ['tap', 'stroke_proximal', 'stroke_distal']

        # 2. Pre-compute count matrices and determine global max for shared LogNorm
        plot_cache: Dict[Tuple, Tuple[pd.DataFrame, int]] = {}
        global_max = 0

        for i_type in interaction_types:
            subset = df[df[type_col] == i_type]
            for y_idx, (y_col, y_edges, y_cats) in enumerate(
                zip(y_cols, y_edges_list, y_cats_list)
            ):
                if subset.empty:
                    matrix = pd.DataFrame(0, index=y_cats, columns=x_cats)
                    count = 0
                else:
                    x_binned = pd.cut(subset[x_col], bins=x_edges, include_lowest=True)
                    y_binned = pd.cut(subset[y_col], bins=y_edges, include_lowest=True)
                    matrix = pd.crosstab(y_binned, x_binned, dropna=False)
                    matrix = matrix.reindex(index=y_cats, columns=x_cats, fill_value=0)
                    count = int(matrix.sum().sum())

                cur_max = matrix.max().max()
                if pd.notna(cur_max) and cur_max > global_max:
                    global_max = cur_max

                plot_cache[(i_type, y_idx)] = (matrix, count)

        if global_max_count is not None:
            # Override with caller-supplied cross-session maximum for consistent color scaling
            global_max = global_max_count
        if global_max == 0:
            global_max = 1

        # 3. Render
        n_types = len(interaction_types)
        fig, axes = plt.subplots(n_rows, n_types, figsize=(7 * n_types, 5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        total_points = sum(
            plot_cache[(i_type, 0)][1]
            for i_type in interaction_types
            if (i_type, 0) in plot_cache
        )
        base_title = f"Touch Density — {title_suffix}" if title_suffix else "Touch Density"
        title = f"{base_title} | Total Points: {total_points}"
        fig.suptitle(title, fontsize=14)

        def fmt_cats(cats):
            return [f"{c.mid:.2f}" for c in cats]

        for y_idx, (y_col, y_cats) in enumerate(zip(y_cols, y_cats_list)):
            for col_idx, i_type in enumerate(interaction_types):
                ax = axes[y_idx, col_idx]
                matrix, count = plot_cache[(i_type, y_idx)]
                is_empty = (matrix.sum().sum() == 0)

                plot_data = matrix.copy()
                if not is_empty:
                    plot_data = plot_data.replace(0, 1)
                    mask = matrix == 0
                else:
                    mask = np.ones(matrix.shape, dtype=bool)

                ax.set_facecolor('white')
                sns.heatmap(
                    plot_data,
                    mask=mask,
                    cmap='magma',
                    norm=LogNorm(vmin=1, vmax=global_max),
                    cbar=True,
                    ax=ax,
                )
                ax.set_title(f"{i_type.capitalize()} (n={count})")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.invert_yaxis()
                ax.set_xticklabels(fmt_cats(x_cats), rotation=45, ha='right')
                ax.set_yticklabels(fmt_cats(y_cats), rotation=0)

                if is_empty:
                    ax.text(
                        0.5, 0.5, "No Data",
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=12, color='gray',
                    )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(filename)
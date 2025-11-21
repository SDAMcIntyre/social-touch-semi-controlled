import numpy as np
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Assuming these are available in the local context based on the provided snippet
from .calibration_pca_engine import PCACalibrationEngine, CalibrationResult

logger = logging.getLogger(__name__)

# --- 4. Visualization / Monitoring ---

class CalibrationVisualizer:
    
    @staticmethod
    def _plot_temporal_3d(ax, data, t_vals, title, labels, goal_text=None):
        """
        Shared helper function to plot 3D trajectory with time-based color.
        Refactored out of visualize() to allow reuse in sequential monitoring.
        """
        # 1. Connect dots with a gradient line to show trajectory
        points = data.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(t_vals.min(), t_vals.max())
        lc = Line3DCollection(segments, cmap='rainbow', norm=norm)
        
        lc.set_array(t_vals[:-1])
        lc.set_linewidth(1)
        lc.set_alpha(0.5)
        
        ax.add_collection(lc)
        
        # 2. Scatter points with Rainbow gradient over time
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=t_vals, cmap='rainbow', alpha=0.6, s=0.5)
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        
        cbar = plt.colorbar(lc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Time', fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        if goal_text:
            ax.text2D(0.05, 0.95, goal_text, transform=ax.transAxes, fontsize=8)

    @staticmethod
    def visualize_sequence(tapping_segments: list, stroking_segments: list, 
                           calib: CalibrationResult):
        """
        Visualizes individual segments sequentially.
        Blocking call: User must close the window to proceed to the next segment.
        """
        plt.style.use('ggplot')
        MAX_POINTS_PER_PLOT = 1500

        logger.info("Starting Sequential Visualization. Close the plot window to view the next segment.")

        # --- 1. Visualize Tapping Segments ---
        for i, raw_seg in enumerate(tapping_segments):
            # Apply Transform
            seg_step1 = PCACalibrationEngine.apply_step1_transform(raw_seg, calib)
            
            # Downsample if necessary
            stride = max(1, len(raw_seg) // MAX_POINTS_PER_PLOT)
            t_vals = np.linspace(0, 1, len(raw_seg))

            # Setup Figure: 1 Row, 2 Columns (Raw vs Aligned)
            fig = plt.figure(figsize=(12, 6))
            fig.canvas.manager.set_window_title(f"Tapping Segment {i+1}/{len(tapping_segments)}")
            
            # Plot Raw
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            CalibrationVisualizer._plot_temporal_3d(
                ax1, raw_seg[::stride], t_vals[::stride], 
                "Raw Input", ("X", "Y", "Z")
            )
            
            # Plot Aligned
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            CalibrationVisualizer._plot_temporal_3d(
                ax2, seg_step1[::stride], t_vals[::stride], 
                "Z-Aligned (Step 1)", ("X'", "Y'", "Z'"),
                goal_text="Goal: Flat Z"
            )
            
            plt.tight_layout()
            plt.show() # Blocking

        # --- 2. Visualize Stroking Segments ---
        for i, raw_seg in enumerate(stroking_segments):
            # Apply Transforms
            seg_step1 = PCACalibrationEngine.apply_step1_transform(raw_seg, calib)
            seg_final = PCACalibrationEngine.apply_full_transform(raw_seg, calib)
            
            # Downsample
            stride = max(1, len(raw_seg) // MAX_POINTS_PER_PLOT)
            t_vals = np.linspace(0, 1, len(raw_seg))

            # Setup Figure: 1 Row, 3 Columns
            fig = plt.figure(figsize=(16, 6))
            fig.canvas.manager.set_window_title(f"Stroking Segment {i+1}/{len(stroking_segments)}")

            # Plot Raw
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            CalibrationVisualizer._plot_temporal_3d(
                ax1, raw_seg[::stride], t_vals[::stride], 
                "Raw Input", ("X", "Y", "Z")
            )

            # Plot Step 1
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            CalibrationVisualizer._plot_temporal_3d(
                ax2, seg_step1[::stride], t_vals[::stride], 
                "Z-Aligned (Step 1)", ("X'", "Y'", "Z'")
            )

            # Plot Final
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            CalibrationVisualizer._plot_temporal_3d(
                ax3, seg_final[::stride], t_vals[::stride], 
                "Final XY-Aligned (Step 2)", ("X''", "Y''", "Z''"),
                goal_text="Goal: Aligned to XY Plane"
            )

            plt.tight_layout()
            plt.show() # Blocking

    @staticmethod
    def visualize(tapping_raw: np.ndarray, stroking_raw: np.ndarray, 
                  calib: CalibrationResult, output_name: str):
        
        # --- Performance Configuration ---
        MAX_POINTS_PER_PLOT = 1500
        
        # Generate intermediate and final data
        tap_step1 = PCACalibrationEngine.apply_step1_transform(tapping_raw, calib)
        str_step1 = PCACalibrationEngine.apply_step1_transform(stroking_raw, calib)
        str_final = PCACalibrationEngine.apply_full_transform(stroking_raw, calib)

        plt.style.use('ggplot')
        
        t_tap = np.linspace(0, 1, len(tapping_raw))
        t_str = np.linspace(0, 1, len(stroking_raw))

        stride_tap = max(1, len(tapping_raw) // MAX_POINTS_PER_PLOT)
        stride_str = max(1, len(stroking_raw) // MAX_POINTS_PER_PLOT)

        if stride_tap > 1:
            logger.info(f"Downsampling tapping data by factor {stride_tap}")
        if stride_str > 1:
            logger.info(f"Downsampling stroking data by factor {stride_str}")

        # --- Tapping Data Visualization ---
        
        # Window 1: Raw Tapping
        fig_tap_raw = plt.figure(figsize=(8, 6))
        try:
            fig_tap_raw.canvas.manager.set_window_title(f"1. Raw Tapping: {output_name}")
        except AttributeError: pass
        ax1 = fig_tap_raw.add_subplot(1, 1, 1, projection='3d')
        CalibrationVisualizer._plot_temporal_3d(
            ax1, tapping_raw[::stride_tap], t_tap[::stride_tap], 
            "1. Raw Tapping Data", ("X", "Y", "Z")
        )

        # Window 2: Tapping After PCA 1
        fig_tap_s1 = plt.figure(figsize=(8, 6))
        try:
            fig_tap_s1.canvas.manager.set_window_title(f"2. Tapping Z-Aligned: {output_name}")
        except AttributeError: pass
        ax2 = fig_tap_s1.add_subplot(1, 1, 1, projection='3d')
        CalibrationVisualizer._plot_temporal_3d(
            ax2, tap_step1[::stride_tap], t_tap[::stride_tap], 
            "2. After PCA 1 (Z-Alignment)", ("X'", "Y'", "Z'"),
            goal_text="Goal: Variation aligned to Z-axis"
        )

        # --- Stroking Data Visualization ---

        # Window 3: Raw Stroking
        fig_str_raw = plt.figure(figsize=(8, 6))
        try:
            fig_str_raw.canvas.manager.set_window_title(f"3. Raw Stroking: {output_name}")
        except AttributeError: pass
        ax3 = fig_str_raw.add_subplot(1, 1, 1, projection='3d')
        CalibrationVisualizer._plot_temporal_3d(
            ax3, stroking_raw[::stride_str], t_str[::stride_str], 
            "1. Raw Stroking Data", ("X", "Y", "Z")
        )

        # Window 4: Stroking After PCA 1
        fig_str_s1 = plt.figure(figsize=(8, 6))
        try:
            fig_str_s1.canvas.manager.set_window_title(f"4. Stroking Z-Aligned: {output_name}")
        except AttributeError: pass
        ax4 = fig_str_s1.add_subplot(1, 1, 1, projection='3d')
        CalibrationVisualizer._plot_temporal_3d(
            ax4, str_step1[::stride_str], t_str[::stride_str], 
            "2. After PCA 1 (Inherited Z-Align)", ("X'", "Y'", "Z'")
        )

        # Window 5: Stroking Final
        fig_str_final = plt.figure(figsize=(8, 6))
        try:
            fig_str_final.canvas.manager.set_window_title(f"5. Stroking Final: {output_name}")
        except AttributeError: pass
        ax5 = fig_str_final.add_subplot(1, 1, 1, projection='3d')
        CalibrationVisualizer._plot_temporal_3d(
            ax5, str_final[::stride_str], t_str[::stride_str], 
            "3. After PCA 2 (XY-Alignment)", ("X''", "Y''", "Z''")
        )
        
        # --- Window 6: Final Time Series Verification ---
        fig_ts = plt.figure(figsize=(10, 6))
        try:
            fig_ts.canvas.manager.set_window_title(f"6. Time Series Verification: {output_name}")
        except AttributeError: pass
            
        ax_z = fig_ts.add_subplot(1, 1, 1)
        ax_z.plot(tapping_raw[::stride_tap, 2], color='gray', alpha=0.5, label='Raw Z', linewidth=1)
        ax_z.plot(tap_step1[::stride_tap, 2], color='red', alpha=0.8, label='Aligned Z (Step 1)', linewidth=1)
        ax_z.set_title("Z-Axis Flattening Verification (Tapping)")
        ax_z.legend()

        logger.info("Displaying 6 separate calibration windows. Close all to continue...")
        plt.show()
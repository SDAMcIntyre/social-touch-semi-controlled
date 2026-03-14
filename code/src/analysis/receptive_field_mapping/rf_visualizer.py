"""Open3D visualization of per-group receptive field clusters."""

import logging
from typing import Optional

import numpy as np
import open3d as o3d
from matplotlib.colors import hsv_to_rgb

from .rf_mapping_config import RFMapResult
from utils.gui.visualize_point_cloud_comparison import visualize_point_cloud_comparison

logger = logging.getLogger(__name__)

# Base hues evenly spaced around the color wheel (10 distinguishable colors).
_BASE_HUES = [i / 10.0 for i in range(10)]


class RFVisualizer:
    """Static-method helper for visualizing receptive field mapping results."""

    @staticmethod
    def visualize_rf_map(
        rf_result: RFMapResult,
        forearm_pcd: Optional[o3d.geometry.PointCloud] = None,
    ) -> None:
        """Launch a split-screen Open3D viewer for one group's RF clusters.

        Left panel: forearm reference point cloud (empty placeholder if *None*).
        Right panel: RF cluster points colored by cluster membership, with
        brightness scaled by selectivity.

        Args:
            rf_result: The mapping result for a single group.
            forearm_pcd: Optional forearm reference geometry for the left panel.
        """
        if not rf_result.clusters:
            logger.info(
                "No clusters found for group '%s'; skipping visualization.",
                rf_result.group_label,
            )
            return

        # --- Left panel ---
        if forearm_pcd is not None:
            left_geometry = forearm_pcd
        else:
            left_geometry = o3d.geometry.PointCloud()
            logger.info("No forearm point cloud provided; left panel will be empty.")

        # --- Right panel ---
        rf_pcd = RFVisualizer._build_rf_point_cloud(rf_result)

        total_points = sum(c.point_count for c in rf_result.clusters)
        visualize_point_cloud_comparison(
            pcd_left=left_geometry,
            pcd_right=rf_pcd,
            title=f"RF Map — {rf_result.group_label}",
            left_label="Forearm Reference",
            right_label=f"RF Clusters ({len(rf_result.clusters)} clusters, {total_points} pts)",
        )

    @staticmethod
    def save_rf_map_image(
        rf_result: RFMapResult,
        output_dir,
        forearm_pcd: Optional[o3d.geometry.PointCloud] = None,
    ) -> Optional["Path"]:
        """Save a static 2-panel PNG (XY top-down and XZ side-view) for one group.

        Renders forearm reference as a subtle grey background when *forearm_pcd*
        is provided.  Cluster colours and brightness follow the same HSV scheme
        as the interactive viewer.

        Args:
            rf_result: The mapping result for a single group.
            output_dir: Directory where the PNG will be written.
            forearm_pcd: Optional forearm reference geometry drawn as background.

        Returns:
            Path to the saved PNG, or ``None`` if there are no clusters to render.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path

        if not rf_result.clusters:
            logger.info(
                "No clusters for group '%s'; skipping image export.",
                rf_result.group_label,
            )
            return None

        # Build per-point colour arrays — same logic as _build_rf_point_cloud.
        all_points = []
        all_colors = []
        for idx, cluster in enumerate(rf_result.clusters):
            hue = _BASE_HUES[idx % len(_BASE_HUES)]
            saturation = 0.8
            scores = cluster.selectivity_scores
            max_sel = scores.max() if len(scores) > 0 else 1.0
            if max_sel <= 0:
                max_sel = 1.0
            for pt_idx in range(len(cluster.points)):
                sel = scores[pt_idx]
                value = 0.3 + 0.7 * (sel / max_sel)
                all_colors.append(hsv_to_rgb([hue, saturation, value]))
            all_points.append(cluster.points)

        points_np = np.vstack(all_points)
        colors_np = np.asarray(all_colors, dtype=np.float64)

        projections = [
            (0, 1, "X (mm)", "Y (mm)", "XY — top-down"),
            (0, 2, "X (mm)", "Z (mm)", "XZ — side view"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (xi, yi, xlabel, ylabel, title) in zip(axes, projections):
            if forearm_pcd is not None:
                fg_pts = np.asarray(forearm_pcd.points)
                if len(fg_pts) > 0:
                    ax.scatter(
                        fg_pts[:, xi], fg_pts[:, yi],
                        c="lightgray", s=0.5, alpha=0.3, rasterized=True, zorder=1,
                    )
            ax.scatter(points_np[:, xi], points_np[:, yi], c=colors_np, s=8, zorder=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.grid(True, linewidth=0.4, alpha=0.5)

        fig.suptitle(f"RF Map — {rf_result.group_label}", fontsize=12)
        fig.tight_layout()

        safe_label = rf_result.group_label.replace("/", "_").replace(" ", "_")
        out_path = Path(output_dir) / f"rf_map_{safe_label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved RF map image: %s", out_path)
        return out_path

    @staticmethod
    def _build_rf_point_cloud(rf_result: RFMapResult) -> o3d.geometry.PointCloud:
        """Build a colored Open3D point cloud from the clusters in *rf_result*.

        Each cluster receives a distinct base hue. Per-point brightness (HSV
        value channel) is scaled by selectivity: higher selectivity produces a
        brighter point.

        Returns:
            An ``o3d.geometry.PointCloud`` with colors assigned.  Returns an
            empty point cloud when *rf_result* contains no clusters.
        """
        pcd = o3d.geometry.PointCloud()

        if not rf_result.clusters:
            return pcd

        all_points = []
        all_colors = []

        for idx, cluster in enumerate(rf_result.clusters):
            hue = _BASE_HUES[idx % len(_BASE_HUES)]
            saturation = 0.8

            # Determine per-point brightness from selectivity scores.
            scores = cluster.selectivity_scores
            max_sel = scores.max() if len(scores) > 0 else 1.0
            if max_sel <= 0:
                max_sel = 1.0  # avoid division by zero

            for pt_idx in range(len(cluster.points)):
                sel = scores[pt_idx]
                value = 0.3 + 0.7 * (sel / max_sel)
                rgb = hsv_to_rgb([hue, saturation, value])
                all_colors.append(rgb)

            all_points.append(cluster.points)

        points_np = np.vstack(all_points)
        colors_np = np.asarray(all_colors, dtype=np.float64)

        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

        return pcd

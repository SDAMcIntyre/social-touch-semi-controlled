import logging
import warnings
from typing import List

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


class FrameDepthAverager:
    """Averages depth point clouds across multiple frames from a KinectMKV."""

    @staticmethod
    def average(
        mkv,
        frame_ids: List[int],
    ) -> o3d.geometry.PointCloud:
        """
        Computes a per-pixel mean of transformed_depth_point_cloud and color across
        the given frames.

        Uses running accumulators (one frame at a time) so memory stays O(H×W×3)
        regardless of how many frames are averaged.  Color frames that are None are
        skipped silently; depth frames that are None are also skipped.

        Args:
            mkv: An open KinectMKV context (supports __getitem__).
            frame_ids: Frame indices to average (will be sorted internally).

        Returns:
            An o3d.geometry.PointCloud with averaged XYZ points and averaged colors.
            Pixels with no valid depth in any frame are excluded.

        Raises:
            ValueError: If no frames yielded valid depth data.
        """
        sorted_ids = sorted(frame_ids)
        running_sum: np.ndarray | None = None
        valid_count: np.ndarray | None = None
        running_color_sum: np.ndarray | None = None
        valid_color_frames: int = 0

        for fid in sorted_ids:
            try:
                frame = mkv[fid]
            except (IndexError, ValueError) as exc:
                logger.warning(
                    "FrameDepthAverager: frame %d could not be loaded (%s); skipping.", fid, exc
                )
                continue

            xyz = frame.transformed_depth_point_cloud
            if xyz is None:
                logger.warning(
                    "FrameDepthAverager: frame %d has no depth point cloud; skipping.", fid
                )
                continue

            # mask shape: (H, W)  — True where depth is valid
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mask = (xyz[:, :, 2] > 0) & ~np.isnan(xyz).any(axis=2)

            if running_sum is None:
                running_sum = np.zeros_like(xyz, dtype=np.float64)
                valid_count = np.zeros(xyz.shape[:2], dtype=np.int32)

            running_sum += xyz * mask[:, :, np.newaxis]
            valid_count += mask.astype(np.int32)

            # Accumulate color from the same frame (None frames are skipped)
            color = frame.color
            if color is not None:
                if running_color_sum is None:
                    running_color_sum = np.zeros((*color.shape[:2], 3), dtype=np.float64)
                running_color_sum += color.astype(np.float64)
                valid_color_frames += 1

        if running_sum is None:
            raise ValueError(
                "FrameDepthAverager: no valid frames were loaded; cannot produce a point cloud."
            )

        # Per-pixel mean; pixels with no valid observations stay at 0 and are
        # excluded by final_mask rather than included as zero points.
        avg_xyz = running_sum / np.maximum(valid_count[:, :, np.newaxis], 1)
        final_mask = valid_count >= 1  # (H, W) bool

        points_xyz = avg_xyz[final_mask]

        if running_color_sum is not None:
            avg_color = running_color_sum / valid_color_frames  # float64, range 0–255
            points_rgb = avg_color[final_mask][:, ::-1] / 255.0  # BGR → RGB
        else:
            logger.warning(
                "FrameDepthAverager: no valid color frames found among %d frame(s); "
                "using white colors.",
                len(sorted_ids),
            )
            points_rgb = np.ones((len(points_xyz), 3), dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(points_rgb)
        return pcd

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
        color_frame_id: int,
    ) -> o3d.geometry.PointCloud:
        """
        Computes a per-pixel mean of transformed_depth_point_cloud across the given frames.

        Uses a running accumulator (one frame at a time) so memory stays O(H×W×3)
        regardless of how many frames are averaged.

        Args:
            mkv: An open KinectMKV context (supports __getitem__).
            frame_ids: Frame indices to average (will be sorted internally).
            color_frame_id: Frame whose color image is used for the output point cloud.

        Returns:
            An o3d.geometry.PointCloud with averaged XYZ points and color from
            color_frame_id.  Pixels with no valid depth in any frame are excluded.

        Raises:
            ValueError: If no frames yielded valid depth data.
        """
        sorted_ids = sorted(frame_ids)
        running_sum: np.ndarray | None = None
        valid_count: np.ndarray | None = None

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

        if running_sum is None:
            raise ValueError(
                "FrameDepthAverager: no valid frames were loaded; cannot produce a point cloud."
            )

        # Per-pixel mean; pixels with no valid observations stay at 0 and are
        # excluded by final_mask rather than included as zero points.
        avg_xyz = running_sum / np.maximum(valid_count[:, :, np.newaxis], 1)
        final_mask = valid_count >= 1  # (H, W) bool

        # Fetch color from the representative frame
        try:
            color_frame = mkv[color_frame_id]
            color_img = color_frame.color
        except (IndexError, ValueError) as exc:
            logger.warning(
                "FrameDepthAverager: color_frame_id %d could not be loaded (%s); "
                "using white colors.",
                color_frame_id,
                exc,
            )
            color_img = None

        points_xyz = avg_xyz[final_mask]

        if color_img is not None:
            points_rgb = color_img[final_mask][:, ::-1] / 255.0  # BGR → RGB
        else:
            logger.warning(
                "FrameDepthAverager: color_frame_id %d returned no color image; "
                "using white colors.",
                color_frame_id,
            )
            points_rgb = np.ones((len(points_xyz), 3), dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(points_rgb)
        return pcd

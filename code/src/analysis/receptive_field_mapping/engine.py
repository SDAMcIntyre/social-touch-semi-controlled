"""Core selectivity scoring and DBSCAN clustering engine for RF mapping."""

import logging
from collections import Counter
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .config import (
    GroupedSpatialData,
    RFCluster,
    RFMapResult,
    SelectivityDBSCANConfig,
)

logger = logging.getLogger(__name__)


class RFMappingEngine:
    """Pure computational engine for receptive field mapping."""

    @staticmethod
    def compute_selectivity(
        spike_counts: Counter,
        total_counts: Counter,
    ) -> Dict[Tuple[float, float, float], float]:
        """Compute per-point selectivity as spike_count / total_touch_count.

        Args:
            spike_counts: Counter mapping (x, y, z) tuples to spike-frame counts.
            total_counts: Counter mapping (x, y, z) tuples to total-frame counts.

        Returns:
            Dict mapping each point to its selectivity score in [0, 1].
        """
        selectivity = {}
        for point, total in total_counts.items():
            if total > 0:
                selectivity[point] = spike_counts.get(point, 0) / total
        return selectivity

    @staticmethod
    def cluster_receptive_field(
        selectivity_scores: Dict[Tuple[float, float, float], float],
        config: SelectivityDBSCANConfig,
        group_label: str,
        touch_count: int = 0,
    ) -> RFMapResult:
        """Filter by selectivity threshold and cluster survivors with DBSCAN.

        Args:
            selectivity_scores: Per-point selectivity from compute_selectivity().
            config: Algorithm parameters.
            group_label: Label identifying this group.
            touch_count: Number of touches in this group (for metadata).

        Returns:
            RFMapResult with clusters, scores, and stats.
        """
        # Filter points above threshold
        above = {
            pt: score
            for pt, score in selectivity_scores.items()
            if score >= config.selectivity_threshold
        }

        total_evaluated = len(selectivity_scores)
        n_above = len(above)

        if n_above == 0:
            logger.warning(
                "[%s] No points above selectivity threshold %.2f (evaluated %d points).",
                group_label,
                config.selectivity_threshold,
                total_evaluated,
            )
            return RFMapResult(
                group_label=group_label,
                clusters=[],
                all_selectivity_scores=selectivity_scores,
                total_points_evaluated=total_evaluated,
                points_above_threshold=0,
                touch_count=touch_count,
            )

        # Prepare arrays for DBSCAN
        points_list = list(above.keys())
        coords = np.array(points_list)  # (N, 3)
        scores = np.array([above[pt] for pt in points_list])

        # Run DBSCAN
        db = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
        labels = db.fit_predict(coords)

        # Build clusters, discard noise (label == -1) and micro-clusters
        clusters = []
        unique_labels = set(labels)
        unique_labels.discard(-1)

        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_points = coords[mask]
            cluster_scores = scores[mask]

            if len(cluster_points) < config.min_cluster_points:
                logger.info(
                    "[%s] Discarding micro-cluster %d with %d points (min=%d).",
                    group_label,
                    cluster_id,
                    len(cluster_points),
                    config.min_cluster_points,
                )
                continue

            clusters.append(
                RFCluster(
                    cluster_id=cluster_id,
                    points=cluster_points,
                    selectivity_scores=cluster_scores,
                    mean_selectivity=float(cluster_scores.mean()),
                    point_count=len(cluster_points),
                )
            )

        n_noise = int(np.sum(labels == -1))
        logger.info(
            "[%s] DBSCAN: %d clusters found, %d noise points, %d points above threshold.",
            group_label,
            len(clusters),
            n_noise,
            n_above,
        )

        return RFMapResult(
            group_label=group_label,
            clusters=clusters,
            all_selectivity_scores=selectivity_scores,
            total_points_evaluated=total_evaluated,
            points_above_threshold=n_above,
            touch_count=touch_count,
        )

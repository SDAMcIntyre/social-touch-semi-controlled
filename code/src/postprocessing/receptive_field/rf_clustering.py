"""Threshold-and-cluster a spatial score field with DBSCAN.

Domain-agnostic clustering helpers used by the postprocessing stage that
re-centres a session's coordinate origin on the receptive field. The module
knows nothing about spikes, neurons, CSV columns, files or the DAG: it takes
plain numeric containers and returns plain result dataclasses.

Two computations are provided, both static methods on :class:`RFMappingEngine`:

``compute_selectivity(numerator_counts, denominator_counts)``
    Per-point ratio ``numerator / denominator`` for every point present in
    *denominator_counts* with a strictly positive denominator. Points whose
    denominator is zero are dropped from the output entirely rather than
    yielding a zero or a NaN, so "no observations here" stays distinguishable
    from "observed, scored zero". Points that appear only in the numerator are
    ignored. The result is in ``[0, 1]`` whenever the numerator is bounded by
    the denominator.

``cluster_receptive_field(scores, config, ...)``
    Keeps the points whose score is ``>= config.selectivity_threshold``, runs
    :class:`sklearn.cluster.DBSCAN` over their raw 3D coordinates, drops the
    noise label ``-1``, drops clusters smaller than
    ``config.min_cluster_points``, and returns the survivors in ascending
    DBSCAN-label order inside an :class:`RFMapResult`. When nothing clears the
    threshold it returns a result with an empty cluster list — an expected
    outcome that the caller is responsible for interpreting, not an error.

Note what this module does **not** do: it computes no centroid and applies no
weighting. Reducing a cluster to a single representative point is the caller's
job, and the postprocessing caller does it with a different weighting than the
selectivity scores computed here.

Units: coordinates are consumed as-is and are never rescaled or normalised. The
postprocessing stage supplies millimetres, so the ``dbscan_eps = 5.0`` default
is a 5 mm neighbourhood radius.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


@dataclass
class SelectivityDBSCANConfig:
    """Algorithm parameters for selectivity scoring and DBSCAN clustering."""
    selectivity_threshold: float = 0.3
    dbscan_eps: float = 5.0
    dbscan_min_samples: int = 3
    min_cluster_points: int = 5


@dataclass
class RFCluster:
    """A single spatially coherent receptive field cluster."""
    cluster_id: int
    points: np.ndarray  # (N, 3) array of 3D coordinates
    selectivity_scores: np.ndarray  # (N,) per-point selectivity
    mean_selectivity: float
    point_count: int


@dataclass
class RFMapResult:
    """Complete RF mapping result for one group."""
    group_label: str
    clusters: List[RFCluster]
    all_selectivity_scores: Dict[Tuple[float, float, float], float]
    total_points_evaluated: int
    points_above_threshold: int
    touch_count: int


@dataclass
class GroupedSpatialData:
    """Accumulated spatial data for one group label."""
    group_label: str
    spike_counts: Counter = field(default_factory=Counter)
    total_counts: Counter = field(default_factory=Counter)
    touch_count: int = 0


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

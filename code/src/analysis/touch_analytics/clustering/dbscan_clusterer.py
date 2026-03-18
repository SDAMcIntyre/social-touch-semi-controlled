# clustering/dbscan_clusterer.py
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .base import TouchClusterer

_DEFAULT_MIN_PER_CLUSTER = 30


class DBSCANClusterer(TouchClusterer):
    """
    DBSCAN with optional automatic eps selection via the k-distance knee method.

    config keys:
      eps                  : float | 'auto'  (default 'auto')
      min_touches_per_cluster : int           (default 30, used as min_samples)
    """

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
    ) -> Tuple[np.ndarray, dict]:
        min_samples = config.get('min_touches_per_cluster', _DEFAULT_MIN_PER_CLUSTER)
        eps_config = config.get('eps', 'auto')

        scaler = StandardScaler()
        X = scaler.fit_transform(feature_df.values)

        if eps_config == 'auto':
            eps = _auto_eps(X, min_samples)
            logging.info(f"DBSCAN auto-eps selected eps={eps:.4f}")
        else:
            eps = float(eps_config)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        noise_count = int((labels == -1).sum())
        unique_clusters = sorted(set(labels) - {-1})
        sizes = {str(c): int((labels == c).sum()) for c in unique_clusters}

        metadata = {
            'algorithm': 'dbscan',
            'params': config,
            'eps_used': eps,
            'eps_config': eps_config,
            'min_samples': min_samples,
            'k': len(unique_clusters),
            'cluster_sizes': sizes,
            'noise_count': noise_count,
        }
        return labels, metadata


def _auto_eps(X: np.ndarray, min_samples: int) -> float:
    """
    Estimate eps via the k-distance knee heuristic:
    fit NearestNeighbors with k=min_samples, sort the k-th distances,
    and pick the point of maximum curvature as eps.
    """
    k = min(min_samples, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    # Curvature via second derivative
    second_deriv = np.diff(k_distances, n=2)
    knee_idx = int(np.argmax(second_deriv)) + 1  # offset for two diffs
    return float(k_distances[knee_idx])

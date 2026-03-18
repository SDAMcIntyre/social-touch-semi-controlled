# clustering/kmeans_clusterer.py
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import TouchClusterer

_DEFAULT_MIN_PER_CLUSTER = 30


class KMeansClusterer(TouchClusterer):
    """
    Adaptive K-means: starts at k_max = total_touches // min_touches_per_cluster,
    then decrements k until every cluster has >= min_touches_per_cluster touches.
    Floor is k=2.  Features are z-scored with StandardScaler before clustering.
    """

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
    ) -> Tuple[np.ndarray, dict]:
        min_per_cluster = config.get('min_touches_per_cluster', _DEFAULT_MIN_PER_CLUSTER)
        n = len(feature_df)

        if n < 2 * min_per_cluster:
            logging.warning(
                f"KMeans: only {n} touches, need at least {2 * min_per_cluster} "
                f"for min_touches_per_cluster={min_per_cluster}. Assigning all to cluster 0."
            )
            labels = np.zeros(n, dtype=int)
            return labels, _build_metadata('kmeans', config, k=1, labels=labels)

        scaler = StandardScaler()
        X = scaler.fit_transform(feature_df.values)

        k_max = max(2, n // min_per_cluster)
        k = k_max

        while k >= 2:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            sizes = np.bincount(labels)
            if sizes.min() >= min_per_cluster:
                break
            k -= 1
        else:
            # k fell below 2 — just assign everything to one cluster
            labels = np.zeros(n, dtype=int)
            k = 1

        metadata = _build_metadata('kmeans', config, k=k, labels=labels)
        metadata['k_max_tried'] = k_max
        return labels, metadata


def _build_metadata(algorithm: str, config: dict, k: int, labels: np.ndarray) -> dict:
    sizes = np.bincount(labels[labels >= 0]).tolist()
    return {
        'algorithm': algorithm,
        'params': config,
        'k': k,
        'cluster_sizes': sizes,
    }

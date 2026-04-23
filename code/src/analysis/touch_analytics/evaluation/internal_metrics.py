# evaluation/internal_metrics.py
"""
Internal clustering quality metrics.

All metrics are computed on the already-scaled feature matrix *X* using the
cluster labels produced by a clusterer.  Edge cases (too few samples, too few
clusters) return ``None`` so callers can include them in JSON output without
crashing.
"""

import logging
from typing import Optional

import numpy as np


def compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute standard internal clustering quality metrics.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Already-scaled feature matrix used for clustering.
    labels : np.ndarray, shape (n_samples,)
        Cluster labels.  Noise points (label == -1) are excluded before
        computing metrics.

    Returns
    -------
    dict with keys:
        ``silhouette_score``, ``davies_bouldin_score``,
        ``calinski_harabasz_score`` — float or ``None`` when unavailable.
    """
    result: dict = {
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
    }

    # Exclude noise points (DBSCAN label == -1)
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    n_samples = len(X_valid)
    n_clusters = len(set(labels_valid))

    if n_samples < 2:
        logging.debug(
            f"compute_internal_metrics: only {n_samples} valid sample(s) — "
            "skipping all metrics."
        )
        return result

    if n_clusters < 2:
        logging.debug(
            f"compute_internal_metrics: only {n_clusters} cluster(s) — "
            "skipping all metrics (need >= 2)."
        )
        return result

    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        result["silhouette_score"] = float(
            silhouette_score(X_valid, labels_valid)
        )
    except Exception as exc:
        logging.warning(f"compute_internal_metrics: silhouette_score failed: {exc}")

    try:
        result["davies_bouldin_score"] = float(
            davies_bouldin_score(X_valid, labels_valid)
        )
    except Exception as exc:
        logging.warning(f"compute_internal_metrics: davies_bouldin_score failed: {exc}")

    try:
        result["calinski_harabasz_score"] = float(
            calinski_harabasz_score(X_valid, labels_valid)
        )
    except Exception as exc:
        logging.warning(
            f"compute_internal_metrics: calinski_harabasz_score failed: {exc}"
        )

    return result

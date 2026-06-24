# evaluation/stability.py
"""
Bootstrap stability estimation for clustering results.

Repeatedly subsamples the feature matrix, re-runs the clusterer, and
measures label agreement between consecutive runs via Adjusted Rand Index.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ..clustering.base import ClusteringContext


def bootstrap_stability(
    clusterer,
    X: np.ndarray,
    config: dict,
    n_rounds: int = 20,
    subsample_fraction: float = 0.8,
    context: Optional[ClusteringContext] = None,
) -> dict:
    """
    Estimate clustering stability via bootstrapped subsampling.

    For each round, a random subsample of rows is drawn (without replacement),
    the clusterer is run on that subsample, and the Adjusted Rand Index (ARI)
    is computed between adjacent pairs of bootstrap runs (round i vs round i-1)
    using only the rows that appear in both subsamples.

    Parameters
    ----------
    clusterer : TouchClusterer
        A clusterer instance with ``fit_predict(feature_df, config, context)``
        signature.  The clusterer receives an **already-scaled** DataFrame (or
        array wrapped as a DataFrame) — it must NOT apply its own scaler.
    X : np.ndarray, shape (n_samples, n_features)
        Already-scaled feature matrix.
    config : dict
        Full combination config dict passed through to ``clusterer.fit_predict``.
    n_rounds : int
        Number of bootstrap rounds.
    subsample_fraction : float
        Fraction of rows to draw per round (without replacement).
    context : ClusteringContext or None
        Runtime label arrays to pass through to ``clusterer.fit_predict``.
        When ``None`` an empty ``ClusteringContext()`` is used.

    Returns
    -------
    dict with keys:
        ``method`` — ``"bootstrap"``.
        ``n_rounds`` — actual number of rounds executed.
        ``bootstrap_ari`` — mean ARI across adjacent round pairs, or ``1.0``
            when every round produced a single cluster (perfectly stable),
            or ``None`` when ARI could not be computed.
    """
    _context = context if context is not None else ClusteringContext()

    n_samples = len(X)
    sub_size = max(2, int(n_samples * subsample_fraction))

    rng = np.random.default_rng(seed=42)
    round_results: list[tuple[np.ndarray, np.ndarray]] = []  # (indices, labels)

    for _ in range(n_rounds):
        indices = rng.choice(n_samples, size=sub_size, replace=False)
        X_sub = X[indices]
        sub_df = pd.DataFrame(X_sub)

        try:
            labels_sub, _ = clusterer.fit_predict(sub_df, config, _context)
        except Exception as exc:
            logging.warning(f"bootstrap_stability: fit_predict failed on subsample: {exc}")
            continue

        unique_labels = set(labels_sub[labels_sub >= 0])
        if len(unique_labels) <= 1:
            # Single cluster or all noise — trivially stable; skip ARI calculation
            round_results.append((indices, labels_sub))
            continue

        round_results.append((indices, labels_sub))

    if len(round_results) < 2:
        logging.warning(
            "bootstrap_stability: fewer than 2 successful rounds — "
            "returning bootstrap_ari=None."
        )
        return {
            "method": "bootstrap",
            "n_rounds": len(round_results),
            "bootstrap_ari": None,
        }

    # Compute ARI between adjacent pairs using shared indices
    ari_values: list[float] = []
    for i in range(len(round_results) - 1):
        idx_a, labels_a = round_results[i]
        idx_b, labels_b = round_results[i + 1]

        # Find rows present in both subsamples
        shared_idx_a = np.intersect1d(idx_a, idx_b)
        if len(shared_idx_a) < 2:
            continue

        # Map shared original indices → positions within each subsample
        pos_a = np.searchsorted(np.sort(idx_a), shared_idx_a)
        # np.searchsorted requires sorted array — use index mapping instead
        a_inv = {orig: pos for pos, orig in enumerate(idx_a)}
        b_inv = {orig: pos for pos, orig in enumerate(idx_b)}

        lab_a = np.array([labels_a[a_inv[idx]] for idx in shared_idx_a])
        lab_b = np.array([labels_b[b_inv[idx]] for idx in shared_idx_a])

        # If either set collapsed to a single cluster, treat as stable
        if len(set(lab_a[lab_a >= 0])) <= 1 or len(set(lab_b[lab_b >= 0])) <= 1:
            ari_values.append(1.0)
            continue

        try:
            ari = adjusted_rand_score(lab_a, lab_b)
            ari_values.append(float(ari))
        except Exception as exc:
            logging.warning(f"bootstrap_stability: ARI computation failed: {exc}")

    if not ari_values:
        mean_ari = None
    else:
        mean_ari = float(np.mean(ari_values))

    return {
        "method": "bootstrap",
        "n_rounds": len(round_results),
        "bootstrap_ari": mean_ari,
    }

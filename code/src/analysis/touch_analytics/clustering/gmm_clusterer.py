# clustering/gmm_clusterer.py
import logging
from typing import ClassVar, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .base import ClusteringContext, TouchClusterer

_DEFAULT_MAX_COMPONENTS = 15
_DEFAULT_COVARIANCE_TYPE = "full"
_DEFAULT_N_INIT = 10
_DEFAULT_MIN_TOUCHES_PER_COMPONENT = 30


class GMMClusterer(TouchClusterer):
    """
    GMM with BIC-based automatic K selection (sklearn.mixture.GaussianMixture).

    Config keys
    -----------
    max_components : int
        Upper bound for the BIC sweep (default 15). Clamped to
        ``n // min_touches_per_component`` so no component can be trivially
        underpopulated.
    covariance_type : str
        One of ``'full'``, ``'tied'``, ``'diag'``, ``'spherical'``
        (default ``'full'``).
    n_init : int
        Random initialisations per K value (default 10). Reduces sensitivity
        to bad local optima at the cost of runtime.
    min_touches_per_component : int
        Minimum touches a component must contain. K candidates that produce a
        component smaller than this are skipped during best-K selection;
        K=1 is always accepted as a last resort (default 30).

    Returns
    -------
    labels : np.ndarray
        Hard cluster labels (argmax of posterior probabilities).
    metadata : dict
        Keys include:

        * ``'extra_columns'`` — maps ``'gmm_prob_<i>'`` names to per-row
          probability arrays (one column per component). Popped by the
          pipeline before JSON serialisation; appended to the output CSV.
        * ``'means'`` — list of shape K × D (plain Python lists, JSON-safe).
        * ``'covariances'`` — list of shape K × D × D. All covariance types
          (``'full'``, ``'tied'``, ``'diag'``, ``'spherical'``) are expanded
          to per-component D × D matrices before storage.
        * ``'feature_columns'`` — list of D column names corresponding to the
          columns in *feature_df* (the already-scaled input).

    Notes
    -----
    All fitted models from the BIC sweep are cached in memory; the best-K
    model is never re-fitted. Bootstrap stability re-runs the full BIC sweep
    per subsample, which is slower than K-Means bootstrapping — reduce
    ``n_rounds`` or ``max_components`` if runtime is a concern.
    """

    PATH: ClassVar[Literal["A", "B"]] = "B"

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
        context: ClusteringContext,
    ) -> Tuple[np.ndarray, dict]:
        max_components: int = config.get('max_components', _DEFAULT_MAX_COMPONENTS)
        covariance_type: str = config.get('covariance_type', _DEFAULT_COVARIANCE_TYPE)
        n_init: int = config.get('n_init', _DEFAULT_N_INIT)
        min_touches: int = config.get('min_touches_per_component', _DEFAULT_MIN_TOUCHES_PER_COMPONENT)

        n = len(feature_df)

        if n < 2 * min_touches:
            logging.warning(
                f"GMMClusterer: only {n} touches, need at least {2 * min_touches} "
                f"for min_touches_per_component={min_touches}. "
                "Assigning all to cluster 0."
            )
            labels = np.zeros(n, dtype=int)
            feature_columns = feature_df.columns.tolist()
            col_means = feature_df.mean().tolist()
            col_cov = np.diag(feature_df.var().fillna(1.0).values).tolist()
            return labels, {
                'algorithm': 'gmm',
                'params': config,
                'k': 1,
                'cluster_sizes': [n],
                'bic_scores': {},
                'covariance_type': covariance_type,
                'component_weights': [],
                'convergence': True,
                'means': [col_means],
                'covariances': [col_cov],
                'feature_columns': feature_columns,
                'extra_columns': {},
            }

        X = feature_df.values
        k_max = max(1, min(max_components, n // min_touches))

        print(
            f"  [gmm] BIC sweep k=1..{k_max}  n={n}  D={X.shape[1]}"
            f"  cov={covariance_type}  n_init={n_init}",
            flush=True,
        )

        # BIC sweep — cache all fitted models to avoid re-fitting the winner
        models: dict[int, GaussianMixture] = {}
        bic_scores: dict[int, float] = {}
        for k in range(1, k_max + 1):
            gm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                n_init=n_init,
                random_state=42,
            )
            gm.fit(X)
            if not gm.converged_:
                raise RuntimeError(
                    f"GMMClusterer: GaussianMixture with n_components={k} "
                    "did not converge. Try increasing max_iter, reducing "
                    "max_components, or switching covariance_type to 'diag'."
                )
            models[k] = gm
            bic_scores[k] = float(gm.bic(X))
            print(f"  [gmm] k={k:>2}/{k_max}  BIC={bic_scores[k]:>12.2f}", flush=True)

        # Select best K: lowest BIC first, skipping K where any component
        # falls below min_touches. K=1 is always accepted as a last resort.
        best_k = 1
        skipped: list[int] = []
        for k in sorted(bic_scores, key=lambda k: bic_scores[k]):
            trial_labels = models[k].predict(X)
            sizes = np.bincount(trial_labels, minlength=k)
            if sizes.min() >= min_touches or k == 1:
                best_k = k
                break
            skipped.append(k)

        if skipped:
            print(
                f"  [gmm] skipped k={skipped} (component < min_touches={min_touches})",
                flush=True,
            )

        best_model = models[best_k]
        labels = best_model.predict(X).astype(int)
        probs = best_model.predict_proba(X)  # shape (n, best_k)
        sizes_preview = np.bincount(labels, minlength=best_k).tolist()
        print(
            f"  [gmm] selected k={best_k}  BIC={bic_scores[best_k]:.2f}"
            f"  sizes={sizes_preview}",
            flush=True,
        )

        extra_columns: dict[str, np.ndarray] = {
            f'gmm_prob_{i}': probs[:, i] for i in range(best_k)
        }
        sizes = np.bincount(labels, minlength=best_k).tolist()

        feature_columns = feature_df.columns.tolist()
        D = len(feature_columns)
        means_list: list = best_model.means_.tolist()  # K × D

        raw_cov = best_model.covariances_
        if covariance_type == 'full':
            covs_list = raw_cov.tolist()  # K × D × D
        elif covariance_type == 'tied':
            # single shared D×D matrix → broadcast to K copies
            covs_list = np.broadcast_to(raw_cov[np.newaxis], (best_k, D, D)).tolist()
        elif covariance_type == 'diag':
            # K × D per-axis variances → K × D × D diagonal matrices
            covs_list = [np.diag(raw_cov[k]).tolist() for k in range(best_k)]
        elif covariance_type == 'spherical':
            # K scalar variances → K × D × D scaled identity matrices
            covs_list = [(np.eye(D) * raw_cov[k]).tolist() for k in range(best_k)]
        else:
            raise ValueError(
                f"GMMClusterer: unrecognised covariance_type='{covariance_type}'. "
                "Expected one of: 'full', 'tied', 'diag', 'spherical'."
            )

        metadata = {
            'algorithm': 'gmm',
            'params': config,
            'k': best_k,
            'cluster_sizes': sizes,
            'bic_scores': {str(k): v for k, v in bic_scores.items()},
            'covariance_type': covariance_type,
            'component_weights': best_model.weights_.tolist(),
            'convergence': True,
            'means': means_list,
            'covariances': covs_list,
            'feature_columns': feature_columns,
            'extra_columns': extra_columns,
        }
        return labels, metadata

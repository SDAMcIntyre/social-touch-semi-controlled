# reduction/pipeline.py
"""
ReductionPipeline: optional variance filter → scaling → optional PCA decomposition.

Used by the clustering orchestrator to pre-process feature DataFrames before
passing data to clusterers.  Clusterers receive already-scaled data and must
NOT apply their own scaler.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .scaling import get_scaler


class ReductionPipeline:
    """
    Stateless transformer: fit-and-transform a feature DataFrame in one call.

    Steps (each optional, controlled by *config*):

    1. **Variance filter** — drop columns whose variance is below a threshold.
    2. **Scaling** — apply a scaler (default ``"standard"``).
    3. **Decomposition** — apply PCA to reduce dimensionality.

    Usage::

        rp = ReductionPipeline()
        X, meta = rp.fit_transform(feature_df, config)
    """

    def fit_transform(
        self,
        feature_df: pd.DataFrame,
        config: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply reduction steps to *feature_df* and return the scaled array.

        Parameters
        ----------
        feature_df : pd.DataFrame
            One row per touch; all columns are numeric features.
        config : dict
            Full combination config dict.  The ``"reduction"`` sub-key is read;
            all other keys are ignored.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features_out)
            Scaled (and optionally reduced) feature matrix.
        metadata : dict
            Keys:

            * ``dropped_columns`` — list of column names removed by variance filter.
            * ``retained_columns`` — list of column names kept after filtering.
            * ``scaler`` — scaler name used.
            * ``decomposition`` — dict with PCA info, or ``None``.
        """
        reduction_cfg: dict = config.get("reduction", {}) if config else {}

        scaler_name: str = reduction_cfg.get("scaler", "standard")
        variance_threshold = reduction_cfg.get("variance_filter", None)
        decomposition_cfg = reduction_cfg.get("decomposition", None)

        columns = list(feature_df.columns)
        X = feature_df.values.copy().astype(float)

        # ------------------------------------------------------------------ #
        # Step 1: Variance filter
        # ------------------------------------------------------------------ #
        dropped_columns: list[str] = []
        retained_columns: list[str] = list(columns)

        if variance_threshold is not None:
            variances = np.var(X, axis=0)
            keep_mask = variances >= float(variance_threshold)
            dropped_columns = [col for col, keep in zip(columns, keep_mask) if not keep]
            retained_columns = [col for col, keep in zip(columns, keep_mask) if keep]

            if dropped_columns:
                logging.info(
                    f"ReductionPipeline: variance filter (threshold={variance_threshold}) "
                    f"dropped {len(dropped_columns)} column(s): {dropped_columns}"
                )
            X = X[:, keep_mask]

        if X.shape[1] == 0:
            logging.warning(
                "ReductionPipeline: all columns were dropped by variance filter — "
                "returning empty array."
            )
            meta = {
                "dropped_columns": dropped_columns,
                "retained_columns": retained_columns,
                "scaler": scaler_name,
                "scaler_mean": None,
                "scaler_scale": None,
                "decomposition": None,
            }
            return X, meta

        # ------------------------------------------------------------------ #
        # Step 2: Scaling
        # ------------------------------------------------------------------ #
        scaler = get_scaler(scaler_name)
        scaler_mean_list = None
        scaler_scale_list = None
        if scaler is not None:
            X = scaler.fit_transform(X)
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                scaler_mean_list = scaler.mean_.tolist()
                scaler_scale_list = scaler.scale_.tolist()

        # ------------------------------------------------------------------ #
        # Step 3: Optional PCA decomposition
        # ------------------------------------------------------------------ #
        decomposition_meta = None
        if decomposition_cfg is not None:
            n_components = decomposition_cfg.get("n_components", None)
            pca = PCA(n_components=n_components, random_state=42)
            X = pca.fit_transform(X)
            decomposition_meta = {
                "method": "pca",
                "n_components_requested": n_components,
                "n_components_fitted": pca.n_components_,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
            logging.info(
                f"ReductionPipeline: PCA reduced to {pca.n_components_} components "
                f"(explained variance: "
                f"{sum(pca.explained_variance_ratio_):.3f})"
            )

        metadata = {
            "dropped_columns": dropped_columns,
            "retained_columns": retained_columns,
            "scaler": scaler_name,
            "scaler_mean": scaler_mean_list,
            "scaler_scale": scaler_scale_list,
            "decomposition": decomposition_meta,
        }
        return X, metadata

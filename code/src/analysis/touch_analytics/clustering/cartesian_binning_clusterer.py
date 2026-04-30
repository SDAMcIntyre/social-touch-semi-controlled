# clustering/cartesian_binning_clusterer.py
import logging
from typing import ClassVar, Literal, Tuple

import numpy as np
import pandas as pd

from .base import ClusteringContext, TouchClusterer
from .outlier_detection import OUTLIER_BIN_LOW, detect_outlier_bounds

_DEFAULT_N_BINS = 5

_VALID_OUTLIER_METHODS = {"iqr", "mad", "percentile", "tukey"}


class CartesianBinningClusterer(TouchClusterer):
    """
    Bins each numeric feature column independently and emits one cluster per
    occupied combination of bin indices.

    Config keys
    -----------
    n_bins : int
        Default number of bins for every feature column (default 5).
    n_bins_per_feature : dict[str, int]
        Per-column overrides.  Any key not present in *feature_df* raises
        ``ValueError`` immediately (fail-fast).
    bin_method : str
        ``"equal_width"`` (default) uses ``pd.cut``; ``"equal_frequency"``
        uses ``pd.qcut``.
    outlier_method : str or None
        Per-feature outlier detection applied before binning.  One of
        ``"iqr"``, ``"mad"``, ``"percentile"``, ``"tukey"``, or ``None``
        (disabled, the default).  Detected outliers are placed in dedicated
        low (``-2``) / high (``n_bins``) bins instead of the main bin range.
    outlier_params : dict
        Technique-specific parameter overrides (e.g. ``{"k": 2.0}`` for IQR).

    Returns
    -------
    labels : np.ndarray, shape (n_touches,)
        Dense integer cluster ids ``0..K-1`` over occupied bin combinations.
    metadata : dict
        Includes ``"extra_columns"`` — a mapping of ``"bin_<col>"`` names to
        per-row integer bin arrays written as DataFrame columns by the pipeline.
    """

    PATH: ClassVar[Literal["A", "B"]] = "B"

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
        context: ClusteringContext,
    ) -> Tuple[np.ndarray, dict]:
        n_bins_default: int = config.get('n_bins', _DEFAULT_N_BINS)
        n_bins_per_feature: dict[str, int] = config.get('n_bins_per_feature', {})
        bin_method: str = config.get('bin_method', 'equal_width')
        outlier_method: str | None = config.get('outlier_method', None)
        outlier_params: dict = config.get('outlier_params', {})

        if n_bins_default < 2:
            raise ValueError(
                f"CartesianBinningClusterer: n_bins must be >= 2, got {n_bins_default}."
            )

        if outlier_method is not None and outlier_method not in _VALID_OUTLIER_METHODS:
            raise ValueError(
                f"CartesianBinningClusterer: outlier_method must be one of "
                f"{sorted(_VALID_OUTLIER_METHODS)} or None, got '{outlier_method}'."
            )

        unknown_keys = [k for k in n_bins_per_feature if k not in feature_df.columns]
        if unknown_keys:
            raise ValueError(
                f"CartesianBinningClusterer: n_bins_per_feature contains keys not found in "
                f"feature_df.columns: {unknown_keys}. "
                f"Available columns: {list(feature_df.columns)}."
            )

        extra_columns: dict[str, np.ndarray] = {}
        bin_edges: dict[str, list[float]] = {}
        n_bins_resolved: dict[str, int] = {}
        bin_arrays: list[np.ndarray] = []
        binned_features: list[str] = []
        outlier_info: dict[str, dict] = {}

        for col in feature_df.columns:
            series = feature_df[col]
            col_n_bins = n_bins_per_feature.get(col, n_bins_default)

            if series.nunique() < 2:
                logging.warning(
                    f"CartesianBinningClusterer: column '{col}' is constant — skipping."
                )
                continue

            # --- Outlier detection (before binning) ---------------------------
            if outlier_method is not None:
                low_bound, high_bound = detect_outlier_bounds(
                    series, outlier_method, outlier_params,
                )
                low_mask = (series < low_bound).to_numpy()
                high_mask = (series > high_bound).to_numpy()
                n_low = int(low_mask.sum())
                n_high = int(high_mask.sum())
                binning_series = series.copy()
                binning_series.iloc[low_mask | high_mask] = np.nan
            else:
                low_mask = high_mask = None
                n_low = n_high = 0
                binning_series = series

            # --- Binning (on non-outlier data) --------------------------------
            if bin_method == 'equal_frequency':
                bins, edges = pd.qcut(binning_series, q=col_n_bins, labels=False, duplicates='drop', retbins=True)
            else:
                bins, edges = pd.cut(binning_series, bins=col_n_bins, labels=False, retbins=True)

            bin_array = bins.to_numpy(dtype=float)
            bin_array = np.where(np.isnan(bin_array), -1, bin_array).astype(int)

            # --- Assign outlier bin indices -----------------------------------
            if outlier_method is not None:
                positive_bins = bin_array[bin_array >= 0]
                actual_n_bins = int(positive_bins.max()) + 1 if len(positive_bins) > 0 else col_n_bins
                if n_low > 0:
                    bin_array[low_mask] = OUTLIER_BIN_LOW
                if n_high > 0:
                    bin_array[high_mask] = actual_n_bins
                # Restore original NaN positions (not outliers)
                original_nan = series.isna().to_numpy()
                bin_array[original_nan] = -1

            if len(np.unique(bin_array)) < 2:
                logging.warning(
                    f"CartesianBinningClusterer: column '{col}' produced fewer than 2 distinct "
                    f"bin indices after binning — skipping."
                )
                continue

            bin_edges[col] = edges.tolist()
            n_bins_resolved[col] = col_n_bins
            extra_columns[f'bin_{col}'] = bin_array
            bin_arrays.append(bin_array)
            binned_features.append(col)

            if outlier_method is not None and (n_low > 0 or n_high > 0):
                outlier_info[col] = {
                    'method': outlier_method,
                    'params': outlier_params,
                    'lower_bound': float(low_bound),
                    'upper_bound': float(high_bound),
                    'n_low_outliers': n_low,
                    'n_high_outliers': n_high,
                }

        if not bin_arrays:
            logging.warning(
                "CartesianBinningClusterer: all columns are constant; assigning label 0."
            )
            labels = np.zeros(len(feature_df), dtype=int)
            metadata = {
                'algorithm': 'cartesian_binning',
                'params': config,
                'n_bins_default': n_bins_default,
                'n_bins_per_feature': n_bins_resolved,
                'bin_method': bin_method,
                'binned_features': [],
                'bin_edges': {},
                'n_clusters': 1,
                'cluster_combinations': np.zeros((1, 0), dtype=int).tolist(),
                'extra_columns': extra_columns,
                'outlier_method': outlier_method,
                'outlier_info': outlier_info,
            }
            return labels, metadata

        B = np.column_stack(bin_arrays)
        unique_combos, labels = np.unique(B, axis=0, return_inverse=True)

        K = len(unique_combos)
        if K > 1000:
            logging.warning(
                f"CartesianBinningClusterer: produced {K} clusters (> 1000). "
                f"Consider reducing n_bins or the number of binned features."
            )

        metadata = {
            'algorithm': 'cartesian_binning',
            'params': config,
            'n_bins_default': n_bins_default,
            'n_bins_per_feature': n_bins_resolved,
            'bin_method': bin_method,
            'binned_features': binned_features,
            'bin_edges': bin_edges,
            'n_clusters': K,
            'cluster_combinations': unique_combos.tolist(),
            'extra_columns': extra_columns,
            'outlier_method': outlier_method,
            'outlier_info': outlier_info,
        }
        return labels, metadata

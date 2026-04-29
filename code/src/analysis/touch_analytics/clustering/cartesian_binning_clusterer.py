# clustering/cartesian_binning_clusterer.py
import logging
from typing import ClassVar, Literal, Tuple

import numpy as np
import pandas as pd

from .base import ClusteringContext, TouchClusterer

_DEFAULT_N_BINS = 5


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

        if n_bins_default < 2:
            raise ValueError(
                f"CartesianBinningClusterer: n_bins must be >= 2, got {n_bins_default}."
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

        for col in feature_df.columns:
            series = feature_df[col]
            col_n_bins = n_bins_per_feature.get(col, n_bins_default)

            if series.nunique() < 2:
                logging.warning(
                    f"CartesianBinningClusterer: column '{col}' is constant — skipping."
                )
                continue

            if bin_method == 'equal_frequency':
                bins, edges = pd.qcut(series, q=col_n_bins, labels=False, duplicates='drop', retbins=True)
            else:
                bins, edges = pd.cut(series, bins=col_n_bins, labels=False, retbins=True)

            bin_array = bins.to_numpy(dtype=float)
            bin_array = np.where(np.isnan(bin_array), -1, bin_array).astype(int)

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
        }
        return labels, metadata

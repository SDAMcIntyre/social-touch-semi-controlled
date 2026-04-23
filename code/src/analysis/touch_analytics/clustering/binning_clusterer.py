# clustering/binning_clusterer.py
import logging
from typing import ClassVar, Literal, Tuple

import numpy as np
import pandas as pd

from .base import ClusteringContext, TouchClusterer

_DEFAULT_N_BINS = 20


class BinningClusterer(TouchClusterer):
    """
    Bins each numeric feature column independently into N fixed-width or
    equal-frequency intervals.

    Config keys
    -----------
    n_bins : int
        Number of bins per feature column (default 20).
    bin_method : str
        ``"equal_width"`` (default) uses ``pd.cut``; ``"equal_frequency"``
        uses ``pd.qcut``.

    Returns
    -------
    labels : np.ndarray
        Bin indices of the highest-variance feature column.  Used as
        ``cluster_label`` for backward compatibility with heatmap generation.
    metadata : dict
        Includes ``"extra_columns"`` — a mapping of ``"bin_<col>"`` names to
        per-row integer bin arrays.  The pipeline pops this key before JSON
        serialisation and writes the arrays as DataFrame columns instead.
    """

    PATH: ClassVar[Literal["A", "B"]] = "B"

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
        context: ClusteringContext,
    ) -> Tuple[np.ndarray, dict]:
        n_bins: int = config.get('n_bins', _DEFAULT_N_BINS)
        bin_method: str = config.get('bin_method', 'equal_width')

        extra_columns: dict[str, np.ndarray] = {}
        bin_edges: dict[str, list[float]] = {}
        primary_col: str | None = None
        primary_variance: float = -1.0

        for col in feature_df.columns:
            series = feature_df[col]

            if series.nunique() < 2:
                logging.warning(
                    f"BinningClusterer: column '{col}' is constant — skipping."
                )
                continue

            if bin_method == 'equal_frequency':
                bins, edges = pd.qcut(series, q=n_bins, labels=False, duplicates='drop', retbins=True)
            else:
                bins, edges = pd.cut(series, bins=n_bins, labels=False, retbins=True)

            bin_edges[col] = edges.tolist()
            bin_array = bins.to_numpy(dtype=float)
            bin_array = np.where(np.isnan(bin_array), -1, bin_array).astype(int)
            extra_columns[f'bin_{col}'] = bin_array

            var = float(series.var())
            if var > primary_variance:
                primary_variance = var
                primary_col = col

        if primary_col is None:
            # All columns were constant — return zeros
            logging.warning(
                "BinningClusterer: all columns are constant; assigning label 0."
            )
            labels = np.zeros(len(feature_df), dtype=int)
        else:
            labels = extra_columns[f'bin_{primary_col}'].copy()

        binned_features = [col for col in feature_df.columns if f'bin_{col}' in extra_columns]
        metadata = {
            'algorithm': 'binning',
            'params': config,
            'n_bins': n_bins,
            'bin_method': bin_method,
            'primary_feature': primary_col,
            'binned_features': binned_features,
            'bin_edges': bin_edges,
            'extra_columns': extra_columns,
        }
        return labels, metadata

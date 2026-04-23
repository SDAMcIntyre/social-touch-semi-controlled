# clustering/type_stratified_clusterer.py
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .base import TouchClusterer

_GROUPS = ['tap', 'stroke_proximal', 'stroke_distal']

_logger = logging.getLogger(__name__)


class TypeStratifiedClusterer(TouchClusterer):
    """
    Wrapper clusterer that splits touches by type, clusters each group
    independently via a delegate base clusterer, and merges results with
    type-prefixed string labels (e.g. ``tap_00``, ``stroke_proximal_01``).

    Type and direction labels are injected into ``config`` by the calling
    pipeline under the ``_type_labels`` and ``_direction_labels`` keys — the
    same convention used by HierarchicalClusterer for ``_sensor_labels``.

    Config keys
    -----------
    base_method : str
        Name of the delegate clusterer (must be in CLUSTERER_REGISTRY).
    _type_labels : np.ndarray
        Runtime-injected array (length = len(feature_df)) with type strings
        (e.g. ``'tap'``, ``'stroke'``).
    _direction_labels : np.ndarray or None
        Runtime-injected array with direction strings (e.g. ``'proximal'``,
        ``'distal'``).  Optional.

    Returns
    -------
    labels : np.ndarray of object dtype
        String cluster label per touch.
    metadata : dict
        Nested per-type metadata plus merged ``extra_columns``.
    """

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
    ) -> Tuple[np.ndarray, dict]:
        from . import get_clusterer

        if '_type_labels' not in config:
            raise ValueError(
                "TypeStratifiedClusterer requires '_type_labels' in config. "
                "Ensure 'type_col' is set in the clustering profile YAML and "
                "the column exists in the pooled DataFrame so the pipeline "
                "injects '_type_labels' before calling fit_predict."
            )

        type_labels = config['_type_labels']
        direction_labels = config.get('_direction_labels')

        group_keys = _build_group_keys(type_labels, direction_labels, len(feature_df))

        labels = np.empty(len(feature_df), dtype=object)
        per_type_meta: dict[str, dict] = {}
        extra_columns_parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

        for group in _GROUPS:
            mask = group_keys == group
            if not mask.any():
                _logger.warning(
                    "TypeStratifiedClusterer: group '%s' has zero touches — skipping.",
                    group,
                )
                continue

            indices = np.where(mask)[0]
            base_clusterer = get_clusterer(config['base_method'])
            base_labels, base_meta = base_clusterer.fit_predict(feature_df.iloc[indices], config)

            string_labels = np.array(
                [f"{group}_{int(lbl):02d}" for lbl in base_labels],
                dtype=object,
            )
            labels[indices] = string_labels
            per_type_meta[group] = base_meta

            for col_name, col_array in base_meta.get('extra_columns', {}).items():
                if col_name not in extra_columns_parts:
                    extra_columns_parts[col_name] = []
                extra_columns_parts[col_name].append((indices, col_array))

        merged_extra_columns = _merge_extra_columns(extra_columns_parts, len(feature_df))

        metadata = {
            'algorithm': 'type_stratified',
            'base_algorithm': config['base_method'],
            'params': {k: v for k, v in config.items() if not k.startswith('_')},
            'per_type': per_type_meta,
            'extra_columns': merged_extra_columns,
        }
        return labels, metadata


def _build_group_keys(
    type_labels: np.ndarray,
    direction_labels,
    n: int,
) -> np.ndarray:
    keys = np.empty(n, dtype=object)
    for i in range(n):
        t = type_labels[i]
        if isinstance(t, str) and 'tap' in t.lower():
            keys[i] = 'tap'
        else:
            if direction_labels is not None and not pd.isna(direction_labels[i]):
                d = direction_labels[i]
                if isinstance(d, str) and 'proximal' in d.lower():
                    keys[i] = 'stroke_proximal'
                else:
                    keys[i] = 'stroke_distal'
            else:
                keys[i] = 'stroke_distal'
    return keys


def _merge_extra_columns(
    parts: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    n: int,
) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for col_name, index_array_pairs in parts.items():
        out = np.empty(n, dtype=object)
        for indices, col_array in index_array_pairs:
            out[indices] = col_array
        merged[col_name] = out
    return merged

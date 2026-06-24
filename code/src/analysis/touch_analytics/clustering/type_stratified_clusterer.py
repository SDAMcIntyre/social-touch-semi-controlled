# clustering/type_stratified_clusterer.py
import logging
from typing import ClassVar, Literal, Tuple

import numpy as np
import pandas as pd

from .base import ClusteringContext, TouchClusterer

_GROUPS = ['tap', 'stroke_proximal', 'stroke_distal']

_logger = logging.getLogger(__name__)


class TypeStratifiedClusterer(TouchClusterer):
    """
    Wrapper clusterer that splits touches by gesture type, clusters each group
    independently via a delegate base clusterer, and merges results with
    type-prefixed string labels (e.g. ``tap_00``, ``stroke_proximal_01``).

    Gesture type labels are provided via ``context.gesture_type_labels``.

    Config keys
    -----------
    base_method : str
        Name of the delegate clusterer (must be in CLUSTERER_REGISTRY).

    Returns
    -------
    labels : np.ndarray of object dtype
        String cluster label per touch.
    metadata : dict
        Nested per-type metadata plus merged ``extra_columns``.
    """

    PATH: ClassVar[Literal["A", "B"]] = "B"

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
        context: ClusteringContext,
    ) -> Tuple[np.ndarray, dict]:
        from . import get_clusterer

        if context.gesture_type_labels is None:
            raise ValueError(
                "TypeStratifiedClusterer requires context.gesture_type_labels. "
                "Ensure the 'gesture_type' column exists in the pooled DataFrame and "
                "the pipeline populates ClusteringContext.gesture_type_labels."
            )

        group_keys = context.gesture_type_labels
        invalid = set(group_keys) - set(_GROUPS)
        if invalid:
            raise ValueError(
                f"TypeStratifiedClusterer: invalid gesture_type values {invalid}; "
                f"expected values from {_GROUPS}"
            )

        labels = np.empty(len(feature_df), dtype=object)
        per_type_meta: dict[str, dict] = {}
        extra_columns_parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

        for group in _GROUPS:
            mask = group_keys == group
            if not mask.any():
                _logger.warning(
                    "TypeStratifiedClusterer: group '%s' has zero touches — skipping. "
                    "(n_touches=%d, groups_present=%s)",
                    group,
                    len(feature_df),
                    np.unique(group_keys).tolist(),
                )
                continue

            indices = np.where(mask)[0]
            base_clusterer = get_clusterer(config['base_method'])
            base_labels, base_meta = base_clusterer.fit_predict(feature_df.iloc[indices], config, context)

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

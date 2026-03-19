# clustering/hierarchical_clusterer.py
"""
Hierarchical clusterer using Ward's linkage with an adaptive non-uniform
dendrogram cut driven by a coverage constraint.

Algorithm (from docs/design/inter_neuron_agreement_methodology.md):
  1. Z-score all features with StandardScaler.
  2. Compute Ward's linkage matrix.
  3. Recursively cut the dendrogram top-down:
     - If both child subtrees independently satisfy the coverage constraint,
       descend into both (finer strata).
     - Otherwise, declare the current node a stratum (if it satisfies the
       constraint itself).
     - If the current node also fails the constraint, mark its instances as
       non-exploitable (label -1).
  4. Tag each valid stratum with its within-cluster variance (dispersion).

Coverage constraint
-------------------
A node satisfies coverage when:
  len(sensors with count >= min_instances_per_sensor) >= min_sensor_types

Sensor label injection
----------------------
The caller (clustering_pipeline.py) injects sensor labels at runtime via
``config['_sensor_labels']`` (a numpy array aligned with feature_df rows).
The ``_`` prefix signals a runtime key — not YAML-configured. Existing
clusterers that receive this key simply ignore it.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import TouchClusterer

_DEFAULT_MIN_INSTANCES = 5
_DEFAULT_MIN_SENSOR_TYPES = 2


class HierarchicalClusterer(TouchClusterer):
    """
    Ward's linkage + adaptive non-uniform dendrogram cut with coverage constraint.
    """

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
    ) -> Tuple[np.ndarray, dict]:
        from scipy.cluster.hierarchy import linkage

        min_instances = config.get('min_instances_per_sensor', _DEFAULT_MIN_INSTANCES)
        min_sensors = config.get('min_sensor_types', _DEFAULT_MIN_SENSOR_TYPES)
        sensor_labels = config.get('_sensor_labels')  # runtime-injected, may be None

        n = len(feature_df)

        if n < 2:
            labels = np.full(n, -1, dtype=int)
            return labels, _build_metadata(config, k=0, labels=labels,
                                           dispersion_per_stratum={},
                                           non_exploitable_count=n)

        if n > 5000:
            logging.warning(
                f"HierarchicalClusterer: n={n} is large. "
                "Ward linkage has O(n^2) memory cost — consider subsampling."
            )

        scaler = StandardScaler()
        X = scaler.fit_transform(feature_df.values)

        Z = linkage(X, method='ward')

        labels = np.full(n, -1, dtype=int)
        stratum_id = [0]
        non_exploitable_count = [0]
        dispersion_per_stratum: dict[int, float] = {}

        def _get_leaves(node_id: int) -> list[int]:
            """Iteratively collect all leaf indices under *node_id*."""
            stack = [node_id]
            leaves = []
            while stack:
                node = stack.pop()
                if node < n:
                    leaves.append(node)
                else:
                    i = node - n
                    stack.append(int(Z[i, 0]))
                    stack.append(int(Z[i, 1]))
            return leaves

        def _satisfies_coverage(indices: np.ndarray) -> bool:
            if sensor_labels is None:
                return True
            sensors = sensor_labels[indices]
            unique, counts = np.unique(sensors, return_counts=True)
            valid = (counts >= min_instances).sum()
            return int(valid) >= min_sensors

        def _assign_stratum(indices: np.ndarray) -> None:
            disp = float(np.var(X[indices], axis=0).mean()) if len(indices) > 1 else 0.0
            sid = stratum_id[0]
            labels[indices] = sid
            dispersion_per_stratum[sid] = disp
            stratum_id[0] += 1

        def _cut(node_id: int) -> None:
            indices = np.array(_get_leaves(node_id), dtype=int)

            # Leaf node
            if node_id < n:
                if _satisfies_coverage(indices):
                    _assign_stratum(indices)
                else:
                    non_exploitable_count[0] += len(indices)
                return

            # Internal node: try to split
            i = node_id - n
            left = int(Z[i, 0])
            right = int(Z[i, 1])
            left_idx = np.array(_get_leaves(left), dtype=int)
            right_idx = np.array(_get_leaves(right), dtype=int)

            if _satisfies_coverage(left_idx) and _satisfies_coverage(right_idx):
                _cut(left)
                _cut(right)
            elif _satisfies_coverage(indices):
                _assign_stratum(indices)
            else:
                non_exploitable_count[0] += len(indices)

        root = 2 * n - 2
        _cut(root)

        k = stratum_id[0]
        return labels, _build_metadata(
            config, k=k, labels=labels,
            dispersion_per_stratum=dispersion_per_stratum,
            non_exploitable_count=non_exploitable_count[0],
        )


def _build_metadata(
    config: dict,
    k: int,
    labels: np.ndarray,
    dispersion_per_stratum: dict,
    non_exploitable_count: int,
) -> dict:
    valid_labels = labels[labels >= 0]
    sizes = np.bincount(valid_labels, minlength=k).tolist() if k > 0 else []
    return {
        'algorithm': 'hierarchical',
        'params': {key: val for key, val in config.items() if not key.startswith('_')},
        'k': k,
        'cluster_sizes': sizes,
        'dispersion_per_stratum': dispersion_per_stratum,
        'non_exploitable_count': non_exploitable_count,
    }

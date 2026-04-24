# clustering/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Tuple
import numpy as np
import pandas as pd


@dataclass
class ClusteringContext:
    """
    Runtime labels passed alongside the feature matrix to clusterers.

    All arrays must be aligned with the rows of *feature_df* (same length,
    same order).  Fields default to ``None`` when not available.
    """

    sensor_labels: np.ndarray | None = field(default=None)
    type_labels: np.ndarray | None = field(default=None)
    direction_labels: np.ndarray | None = field(default=None)


class TouchClusterer(ABC):
    PATH: ClassVar[Literal["A", "B"]] = "B"

    @abstractmethod
    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
        context: ClusteringContext,
    ) -> Tuple[np.ndarray, dict]:
        """
        Cluster rows of *feature_df* and return labels + metadata.

        Parameters
        ----------
        feature_df : pd.DataFrame
            One row per touch; columns are numeric features used for clustering.
            Non-feature columns (trial_id, single_touch_id, …) should be
            excluded by the caller before passing here.
        config : dict
            Clustering-profile options from YAML.
        context : ClusteringContext
            Runtime arrays (sensor_labels, type_labels, direction_labels)
            aligned with the rows of *feature_df*.

        Returns
        -------
        labels : np.ndarray, shape (n_touches,)
            Cluster label per touch. DBSCAN noise points are -1.
        metadata : dict
            Algorithm name, parameters, k, per-cluster sizes, etc.
        """

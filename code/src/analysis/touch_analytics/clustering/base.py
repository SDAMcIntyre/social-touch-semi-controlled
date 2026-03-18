# clustering/base.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import pandas as pd


class TouchClusterer(ABC):
    @abstractmethod
    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        config: dict,
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

        Returns
        -------
        labels : np.ndarray, shape (n_touches,)
            Cluster label per touch. DBSCAN noise points are -1.
        metadata : dict
            Algorithm name, parameters, k, per-cluster sizes, etc.
        """

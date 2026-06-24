# representation/feature_characterization/base.py
from abc import ABC, abstractmethod
import pandas as pd


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        """
        Extract method-specific kinematic features from one touch group.

        Parameters
        ----------
        group : pd.DataFrame
            All frames for a single (trial_id, single_touch_id) pair.
        config : dict
            Profile-level options from YAML (may be empty for default extractors).

        Returns
        -------
        dict
            Feature key-value pairs to be merged into the output row.
            Must NOT include shared columns (trial_id, single_touch_id,
            block_order_id, type_metadata, direction, spike_elicited,
            mean_contact_x/y/z) — those are added by the orchestrator.
        """

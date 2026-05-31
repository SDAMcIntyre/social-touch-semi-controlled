# representation/feature_characterization/spike_count.py
import pandas as pd
from .base import FeatureExtractor
from analysis.pipeline.shared_constants import NERVE_SPIKE_COL


class SpikeCountExtractor(FeatureExtractor):
    """
    Counts the total number of nerve spikes within a touch group.

    Output column: Nerve_spike_count (int)
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        if NERVE_SPIKE_COL not in group.columns:
            raise ValueError(
                f"SpikeCountExtractor requires a '{NERVE_SPIKE_COL}' column in the group "
                f"DataFrame, but it was not found. Ensure the session CSV includes spike data."
            )
        return {'Nerve_spike_count': int(group[NERVE_SPIKE_COL].sum())}

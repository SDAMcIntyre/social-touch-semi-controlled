# feature_extraction/__init__.py
# Shim module — re-exports everything from the canonical location so that
# existing callers (e.g. feature_combination_dialog.py:19) continue to work:
#
#   from analysis.touch_analytics.feature_extraction import AGGREGATION_NAMES, EXTRACTOR_REGISTRY
#
# The authoritative implementations now live under:
#   representation/feature_characterization/
from ..representation.feature_characterization import (
    FeatureExtractor,
    StatisticalExtractor,
    TemporalExtractor,
    MechanicsOfSolidsExtractor,
    PressureExtractor,
    TouchCategoryExtractor,
    AGGREGATION_NAMES,
    EXTRACTOR_REGISTRY,
    get_feature_extractor,
    get_extractor,
)

__all__ = [
    'FeatureExtractor',
    'StatisticalExtractor',
    'TemporalExtractor',
    'MechanicsOfSolidsExtractor',
    'PressureExtractor',
    'TouchCategoryExtractor',
    'AGGREGATION_NAMES',
    'EXTRACTOR_REGISTRY',
    'get_feature_extractor',
    'get_extractor',
]

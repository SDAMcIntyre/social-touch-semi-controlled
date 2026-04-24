# representation/feature_characterization/__init__.py
import warnings
from .base import FeatureExtractor
from .statistical import StatisticalExtractor
from .touch_category import TouchCategoryExtractor

# Kinematic aggregations: each is a first-class feature name handled by StatisticalExtractor
AGGREGATION_NAMES = frozenset({'max', 'min', 'mean', 'median', 'std', 'range', 'skewness'})

EXTRACTOR_REGISTRY: dict[str, type[FeatureExtractor]] = {
    'statistical': StatisticalExtractor,
    'touch_category': TouchCategoryExtractor,
}


def get_feature_extractor(feature_name: str, feature_config: dict) -> FeatureExtractor:
    """
    Return a fresh extractor instance for *feature_name*.

    For aggregation names (max, min, mean, median, std, range, skewness),
    returns a StatisticalExtractor. The extraction pipeline is responsible for
    injecting ``aggregations: [feature_name]`` into the config before calling
    ``extract()``.

    For named feature types (statistical, touch_category), instantiates the
    corresponding extractor from EXTRACTOR_REGISTRY.

    Raises
    ------
    KeyError
        If *feature_name* is not a known aggregation and not in EXTRACTOR_REGISTRY.
    """
    if feature_name in AGGREGATION_NAMES:
        return StatisticalExtractor()
    if feature_name in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[feature_name]()
    raise KeyError(
        f"Unknown feature '{feature_name}'. "
        f"Aggregation names: {sorted(AGGREGATION_NAMES)}, "
        f"Other features: {sorted(EXTRACTOR_REGISTRY)}"
    )


def get_extractor(method: str) -> FeatureExtractor:
    """Deprecated: use get_feature_extractor() instead."""
    warnings.warn(
        "get_extractor() is deprecated; use get_feature_extractor() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if method not in EXTRACTOR_REGISTRY and method not in AGGREGATION_NAMES:
        raise KeyError(
            f"Unknown extraction method '{method}'. "
            f"Available: {sorted(AGGREGATION_NAMES | set(EXTRACTOR_REGISTRY))}"
        )
    return get_feature_extractor(method, {})


__all__ = [
    'FeatureExtractor',
    'StatisticalExtractor',
    'TouchCategoryExtractor',
    'AGGREGATION_NAMES',
    'EXTRACTOR_REGISTRY',
    'get_feature_extractor',
    'get_extractor',
]

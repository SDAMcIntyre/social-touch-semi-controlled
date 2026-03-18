# feature_extraction/__init__.py
from .base import FeatureExtractor
from .max_extractor import MaxExtractor
from .statistical_extractor import StatisticalExtractor
from .temporal_extractor import TemporalExtractor
from .mos_extractor import MechanicsOfSolidsExtractor

EXTRACTOR_REGISTRY: dict[str, type[FeatureExtractor]] = {
    'max': MaxExtractor,
    'statistical': StatisticalExtractor,
    'temporal': TemporalExtractor,
    'mechanics_of_solids': MechanicsOfSolidsExtractor,
}


def get_extractor(method: str) -> FeatureExtractor:
    """
    Return a fresh extractor instance for *method*.

    Raises
    ------
    KeyError
        If *method* is not in EXTRACTOR_REGISTRY.
    """
    if method not in EXTRACTOR_REGISTRY:
        raise KeyError(
            f"Unknown extraction method '{method}'. "
            f"Available: {sorted(EXTRACTOR_REGISTRY)}"
        )
    return EXTRACTOR_REGISTRY[method]()


__all__ = [
    'FeatureExtractor',
    'MaxExtractor',
    'StatisticalExtractor',
    'TemporalExtractor',
    'MechanicsOfSolidsExtractor',
    'EXTRACTOR_REGISTRY',
    'get_extractor',
]

# comparing/__init__.py
from .base import ComparisonStrategy, ComparisonResult
from .bias_comparator import BiasComparator
from .precision_comparator import PrecisionComparator
from .distribution_comparator import DistributionComparator

COMPARATOR_REGISTRY: dict[str, type[ComparisonStrategy]] = {
    'bias': BiasComparator,
    'precision': PrecisionComparator,
    'distribution': DistributionComparator,
}


def get_comparator(method: str) -> ComparisonStrategy:
    """
    Return a fresh comparator instance for *method*.

    Raises
    ------
    KeyError
        If *method* is not in COMPARATOR_REGISTRY.
    """
    if method not in COMPARATOR_REGISTRY:
        raise KeyError(
            f"Unknown comparison method '{method}'. "
            f"Available: {sorted(COMPARATOR_REGISTRY)}"
        )
    return COMPARATOR_REGISTRY[method]()


__all__ = [
    'ComparisonStrategy',
    'ComparisonResult',
    'BiasComparator',
    'PrecisionComparator',
    'DistributionComparator',
    'COMPARATOR_REGISTRY',
    'get_comparator',
]

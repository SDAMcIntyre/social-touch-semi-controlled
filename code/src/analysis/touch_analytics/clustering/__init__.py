# clustering/__init__.py
from .base import TouchClusterer
from .kmeans_clusterer import KMeansClusterer
from .dbscan_clusterer import DBSCANClusterer
from .binning_clusterer import BinningClusterer
from .hierarchical_clusterer import HierarchicalClusterer

CLUSTERER_REGISTRY: dict[str, type[TouchClusterer]] = {
    'kmeans': KMeansClusterer,
    'dbscan': DBSCANClusterer,
    'binning': BinningClusterer,
    'hierarchical': HierarchicalClusterer,
}


def get_clusterer(method: str) -> TouchClusterer:
    """
    Return a fresh clusterer instance for *method*.

    Raises
    ------
    KeyError
        If *method* is not in CLUSTERER_REGISTRY.
    """
    if method not in CLUSTERER_REGISTRY:
        raise KeyError(
            f"Unknown clustering method '{method}'. "
            f"Available: {sorted(CLUSTERER_REGISTRY)}"
        )
    return CLUSTERER_REGISTRY[method]()


__all__ = [
    'TouchClusterer',
    'KMeansClusterer',
    'DBSCANClusterer',
    'BinningClusterer',
    'HierarchicalClusterer',
    'CLUSTERER_REGISTRY',
    'get_clusterer',
]

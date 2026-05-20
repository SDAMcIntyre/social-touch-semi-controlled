"""Configuration and result types for receptive field mapping via selectivity + DBSCAN."""

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from analysis.pipeline.shared_constants import NERVE_SPIKE_COL, CONTACT_POINTS_COL


@dataclass
class SelectivityDBSCANConfig:
    """Algorithm parameters for selectivity scoring and DBSCAN clustering."""
    selectivity_threshold: float = 0.3
    dbscan_eps: float = 5.0
    dbscan_min_samples: int = 3
    min_cluster_points: int = 5


@dataclass
class RFMappingColumnConfig:
    """Column name mapping for input CSVs."""
    touch_id: str = "single_touch_id"
    trial_id: str = "trial_id"
    points: str = CONTACT_POINTS_COL
    spike: str = NERVE_SPIKE_COL


@dataclass
class RFMappingConfig:
    """Combined configuration for RF mapping pipeline."""
    algorithm: SelectivityDBSCANConfig = field(default_factory=SelectivityDBSCANConfig)
    columns: RFMappingColumnConfig = field(default_factory=RFMappingColumnConfig)


@dataclass
class RFCluster:
    """A single spatially coherent receptive field cluster."""
    cluster_id: int
    points: np.ndarray  # (N, 3) array of 3D coordinates
    selectivity_scores: np.ndarray  # (N,) per-point selectivity
    mean_selectivity: float
    point_count: int


@dataclass
class RFMapResult:
    """Complete RF mapping result for one group."""
    group_label: str
    clusters: List[RFCluster]
    all_selectivity_scores: Dict[Tuple[float, float, float], float]
    total_points_evaluated: int
    points_above_threshold: int
    touch_count: int


@dataclass
class GroupedSpatialData:
    """Accumulated spatial data for one group label."""
    group_label: str
    spike_counts: Counter = field(default_factory=Counter)
    total_counts: Counter = field(default_factory=Counter)
    touch_count: int = 0

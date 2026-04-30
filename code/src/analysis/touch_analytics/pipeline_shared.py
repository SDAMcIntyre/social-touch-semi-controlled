# pipeline_shared.py
"""
Shared utilities for the split touch analysis pipeline.

Contains constants and helpers used by extraction_pipeline.py,
clustering_pipeline.py, and comparing_pipeline.py.
"""

import sys
from pathlib import Path


# Columns written by the orchestrator (not the extractor)
SHARED_COLUMNS = [
    'block_order_id', 'trial_id', 'single_touch_id',
    'type_metadata', 'gesture_type',
    'mean_contact_x', 'mean_contact_y', 'mean_contact_z',
    'spike_elicited',
    'session_id',
]

# Default options used when a YAML section is missing
DEFAULT_EXTRACTION_OPTIONS: dict = {
    'force_processing': False,
    'features': {
        'max': {'enabled': True},
    },
}

DEFAULT_FEATURE_COMBINATIONS: dict = {
    'basic': {'enabled': True, 'features': ['max']},
}

DEFAULT_CLUSTERING_OPTIONS: dict = {
    'force_processing': False,
    'feature_combinations': DEFAULT_FEATURE_COMBINATIONS,
    'clustering_profiles': {
        'kmeans': {'method': 'kmeans', 'min_touches_per_cluster': 30},
    },
}

DEFAULT_COMPARING_OPTIONS: dict = {
    'force_processing': False,
    'feature_combinations': DEFAULT_FEATURE_COMBINATIONS,
    'comparing_profiles': {
        'bias': {'method': 'bias', 'measurement_col': 'spike_elicited', 'sensor_col': 'session_id'},
    },
    'min_instances_per_sensor': 5,
    'min_sensor_types': 2,
}

DEFAULT_PREPARATION_OPTIONS: dict = {
    'force_processing': False,
    'interpolation_method': 'cubic',
}


class _TqdmLineWrapper:
    """Routes tqdm output through print()-compatible newline-terminated writes.

    tqdm normally writes ``\\r<bar>`` (no newline) to stderr. In the GUI subprocess
    pipe, these bytes accumulate and prefix the next print() call's newline,
    causing ProcessOutputReader to route the milestone through cr_line_received
    (replace_last_line) instead of line_received (append_line).

    This wrapper:
      - strips the leading ``\\r`` so bar lines do not trigger replace_last_line
      - appends ``\\n`` so each tqdm write is a complete, independently-readable line
      - writes to sys.stdout (same stream as print()) for deterministic ordering
    """
    def __init__(self, stream):
        self._stream = stream

    def write(self, s: str) -> int:
        s = s.lstrip('\r')
        if s and not s.endswith('\n'):
            s += '\n'
        return self._stream.write(s)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False


def filter_enabled_profiles(profiles: dict) -> dict:
    """Return only profiles without ``enabled: false``."""
    return {k: v for k, v in profiles.items() if v.get('enabled', True)}


def session_id_from_path(p: Path) -> str:
    """Return the session ID prefix from a touch-summary CSV filename."""
    name = p.name
    if '_semicontrolled_' in name:
        return name.split('_semicontrolled_')[0]
    return p.stem

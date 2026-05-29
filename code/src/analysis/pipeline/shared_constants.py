"""Shared domain constants for the analysis pipeline.

Single source of truth for constants and utility functions used across both
``touch_analytics/`` and ``receptive_field_mapping/`` sub-systems.

Do NOT add constants that are local to a single sub-system here.  These are
only values that are referenced in two or more packages, or that represent
fundamental domain concepts (column names, gesture vocabulary).
"""

from __future__ import annotations

import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Gesture taxonomy
# ---------------------------------------------------------------------------

#: Canonical ordered tuple of gesture type labels produced by preparation.
GESTURE_TYPES: tuple[str, ...] = ('tap', 'stroke_proximal', 'stroke_distal')


# ---------------------------------------------------------------------------
# Touch identity columns
# ---------------------------------------------------------------------------

#: Columns that uniquely identify a single touch within a session.
TOUCH_ID_COLS: tuple[str, ...] = ('block_order_id', 'trial_id', 'single_touch_id')

#: Touch identity columns plus ``session_id`` for cross-session datasets.
TOUCH_ID_COLS_WITH_SESSION: tuple[str, ...] = (*TOUCH_ID_COLS, 'session_id')


# ---------------------------------------------------------------------------
# Neural data column names
# ---------------------------------------------------------------------------

#: Column name for the binary nerve-spike marker (1 kHz).
NERVE_SPIKE_COL: str = 'Nerve_spike'

#: Column name for the contact-points geometry column.
CONTACT_POINTS_COL: str = 'contact_points'

#: Column name for the instantaneous firing frequency.
NERVE_FREQ_COL: str = 'Nerve_freq'


# ---------------------------------------------------------------------------
# Location columns
# ---------------------------------------------------------------------------

#: Per-touch mean contact location columns (written by extraction pipeline).
LOCATION_SHARED_COLS: tuple[str, ...] = (
    'mean_contact_x',
    'mean_contact_y',
    'mean_contact_z',
)

#: Per-frame contact location columns (raw 30 Hz Kinect samples).
LOCATION_BASE_COLS: tuple[str, ...] = (
    'contact_location_x',
    'contact_location_y',
    'contact_location_z',
)


# ---------------------------------------------------------------------------
# Neuron modes
# ---------------------------------------------------------------------------

#: Valid neuron response modes for RF mapping (IFF or spike count).
NEURON_MODES: tuple[str, ...] = ('iff', 'spike')


# ---------------------------------------------------------------------------
# IFF metric selection
# ---------------------------------------------------------------------------

#: Valid IFF aggregation metrics for single-touch RF maps.
IFF_METRICS: tuple[str, ...] = ('mean', 'max')


def single_touch_npz_filename(iff_metric: str) -> str:
    """Return the NPZ filename for the given IFF metric.

    Parameters
    ----------
    iff_metric:
        One of ``IFF_METRICS`` (``'mean'`` or ``'max'``).

    Returns
    -------
    str
        Filename of the form ``"single_touch_rf_maps_{iff_metric}.npz"``.

    Raises
    ------
    ValueError
        If *iff_metric* is not in ``IFF_METRICS``.
    """
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"Invalid iff_metric {iff_metric!r}. Expected one of {IFF_METRICS}."
        )
    return f"single_touch_rf_maps_{iff_metric}.npz"


def resolve_single_touch_npz(session_dir: Path, iff_metric: str) -> Path:
    """Resolve the single-touch RF maps NPZ path for the given IFF metric.

    For ``iff_metric='mean'``, falls back to the legacy
    ``"single_touch_rf_maps.npz"`` filename with a deprecation warning if the
    new ``"single_touch_rf_maps_mean.npz"`` does not exist.

    For ``iff_metric='max'``, only ``"single_touch_rf_maps_max.npz"`` is
    considered — no legacy fallback exists.

    Parameters
    ----------
    session_dir:
        Directory that should contain the NPZ file (typically the per-session
        output folder for ``spatial_map_single_touch``).
    iff_metric:
        One of ``IFF_METRICS`` (``'mean'`` or ``'max'``).

    Returns
    -------
    Path
        Absolute path to the NPZ file.

    Raises
    ------
    ValueError
        If *iff_metric* is not in ``IFF_METRICS``.
    FileNotFoundError
        If no suitable NPZ file is found in *session_dir*.
    """
    canonical = session_dir / single_touch_npz_filename(iff_metric)
    if canonical.exists():
        return canonical

    if iff_metric == 'mean':
        legacy = session_dir / 'single_touch_rf_maps.npz'
        if legacy.exists():
            warnings.warn(
                f"Legacy NPZ file found at {legacy}. "
                "Re-run 'spatial_map_single_touch' to produce the new "
                "'single_touch_rf_maps_mean.npz' file and retire this path.",
                DeprecationWarning,
                stacklevel=2,
            )
            return legacy

    raise FileNotFoundError(
        f"Single-touch RF maps NPZ not found for iff_metric={iff_metric!r} "
        f"in {session_dir}. Expected: {canonical}"
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def session_id_from_path(p: Path) -> str:
    """Return the session ID prefix from a touch-summary CSV filename.

    For filenames containing ``_semicontrolled_``, returns the stem prefix
    before that marker.  Otherwise returns the full stem.

    Parameters
    ----------
    p:
        Path to a session CSV file.

    Returns
    -------
    str
        Session identifier string (e.g. ``'P01_semicontrolled'`` prefix).
    """
    name = p.name
    if '_semicontrolled_' in name:
        return name.split('_semicontrolled_')[0]
    return p.stem


def filter_enabled_profiles(profiles: dict) -> dict:
    """Return only profiles without ``enabled: false``.

    Profiles that omit the ``enabled`` key are treated as enabled.

    Parameters
    ----------
    profiles:
        Mapping of profile name → profile config dict.

    Returns
    -------
    dict
        Filtered subset of *profiles* with ``enabled != False``.
    """
    return {k: v for k, v in profiles.items() if v.get('enabled', True)}

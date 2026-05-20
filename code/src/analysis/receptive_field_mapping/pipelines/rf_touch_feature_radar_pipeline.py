"""Pipeline orchestrator for per-session touch feature radar plots.

Discovers feature extraction CSVs, computes per-gesture statistics with
session-wide min-max normalization, and delegates rendering to
``rf_touch_feature_radar_renderer``.

Outputs per session (in ``4_analysed/touch_feature_radar/{session_id}/``):
  - ``{session_id}_radar_{gesture_name}.png``  — one per gesture type present
  - ``{session_id}_radar_composite.png``        — overlay when ≥2 gesture types present
  - ``radar_sentinel.json``                     — idempotency sentinel
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.shared_constants import GESTURE_TYPES, session_id_from_path
from analysis.receptive_field_mapping.rendering.rf_touch_feature_radar_renderer import (
    GESTURE_COLORS,
    render_gesture_radar,
    render_gesture_radar_composite,
)
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature axis constants
# ---------------------------------------------------------------------------

#: Canonical base feature column names (without aggregation suffix).
#: These are the 9 scalar features from DATA_TYPE_TO_COLUMNS in
#: clustering_pipeline.py (single-column types only).
RADAR_FEATURE_COLUMNS: list[str] = [
    "contact_area",
    "contact_depth",
    "hand_velocity_amplitude",
    "pressure",
    "mos_strain",
    "mos_stress_kpa",
    "mos_strain_rate",
    "mos_elastic_energy_mj",
    "mos_impulse_mns",
]

#: Shortened display labels for radar axis readability (1-to-1 with RADAR_FEATURE_COLUMNS).
RADAR_DISPLAY_LABELS: list[str] = [
    "Area",
    "Depth",
    "Velocity",
    "Pressure",
    "Strain",
    "Stress",
    "Strain Rate",
    "Elastic E.",
    "Impulse",
]

#: Aggregation suffix candidates to try in order when resolving a feature column.
_SUFFIX_CANDIDATES: tuple[str, ...] = ("_mean", "_median", "_max", "_min", "_std")

# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------


def _find_feature_csv(db_path: Path, aggregation: str, session_id: str) -> Path:
    """Return the touch-summary CSV for *session_id* under *aggregation*.

    Parameters
    ----------
    db_path:
        Root database path (the directory that contains ``4_analysed/``).
    aggregation:
        Aggregation folder name (e.g. ``"mean_during_iff"``).
    session_id:
        Session identifier prefix (e.g. ``"P01_semicontrolled"``).

    Returns
    -------
    Path
        Resolved path to the single matching CSV.

    Raises
    ------
    ValueError
        If no CSV is found, or if more than one CSV matches the pattern.
    """
    feature_dir = db_path / "4_analysed" / "touch_features" / aggregation
    candidates = sorted(feature_dir.glob(f"{session_id}_*_touch_summary.csv"))
    if not candidates:
        raise ValueError(
            f"No touch feature CSV found for session '{session_id}' under "
            f"{feature_dir}. "
            f"Ensure 'touch_feature_extraction' with aggregation '{aggregation}' has run."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple feature CSVs found for session '{session_id}': {candidates}. "
            f"Expected exactly one."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def _resolve_feature_columns(df: pd.DataFrame, csv_path: Path) -> list[str]:
    """Return the actual DataFrame column name for each entry in RADAR_FEATURE_COLUMNS.

    For each base feature name, the first matching suffixed column found in
    *df* is used.  Suffix candidates are tried in the order defined by
    ``_SUFFIX_CANDIDATES``.

    Parameters
    ----------
    df:
        Feature summary DataFrame.
    csv_path:
        Path to the CSV (used in error messages only).

    Returns
    -------
    list[str]
        Resolved column names, one per entry in RADAR_FEATURE_COLUMNS.

    Raises
    ------
    ValueError
        If no suffixed variant of a base feature name exists in *df*.
    """
    resolved: list[str] = []
    for base in RADAR_FEATURE_COLUMNS:
        matched: str | None = None
        for suffix in _SUFFIX_CANDIDATES:
            candidate = base + suffix
            if candidate in df.columns:
                matched = candidate
                break
        if matched is None:
            raise ValueError(
                f"Feature '{base}' not found in {csv_path}. "
                f"Tried columns: {[base + s for s in _SUFFIX_CANDIDATES]}. "
                f"Available columns: {list(df.columns)}"
            )
        resolved.append(matched)
    return resolved


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def _compute_session_stats(df: pd.DataFrame, resolved_cols: list[str]) -> dict:
    """Compute session-wide min-max normalised statistics per gesture type.

    Normalization is computed across **all** rows (session-wide), so all
    gesture-type subsets share the same [0, 1] scale and are directly
    comparable.

    Parameters
    ----------
    df:
        Feature summary DataFrame (all rows for the session).
    resolved_cols:
        Resolved column names (one per feature axis), matching the order of
        ``RADAR_FEATURE_COLUMNS``.

    Returns
    -------
    dict
        Mapping of gesture name → ``{"medians": np.ndarray, "q25": np.ndarray,
        "q75": np.ndarray}``.  Includes an ``"all"`` key for all rows combined.
        Gesture types with fewer than 2 rows are omitted (logged at INFO level).
    """
    # --- Session-wide min-max normalisation ---
    raw = df[resolved_cols].to_numpy(dtype=float)  # shape: (n_touches, n_features)
    col_min = np.nanmin(raw, axis=0)
    col_max = np.nanmax(raw, axis=0)
    col_range = col_max - col_min

    # Zero-variance guard: if max == min, set normalized value to 0.0
    safe_range = np.where(col_range == 0.0, 1.0, col_range)
    normalized = np.where(
        col_range == 0.0,
        0.0,
        (raw - col_min) / safe_range,
    )

    results: dict = {}

    subsets: list[tuple[str, pd.Series | None]] = [("all", None)] + [
        (gtype, df["gesture_type"] == gtype) for gtype in GESTURE_TYPES
    ]

    for name, mask in subsets:
        if mask is None:
            rows = normalized
        else:
            row_indices = mask.to_numpy()
            rows = normalized[row_indices]

        n_rows = len(rows)
        if n_rows < 2:
            logger.info(
                "[Touch Feature Radar] Gesture '%s' has %d row(s) — need ≥2 to compute "
                "statistics; skipping.",
                name, n_rows,
            )
            continue

        medians = np.nanmedian(rows, axis=0)
        q25 = np.nanpercentile(rows, 25, axis=0)
        q75 = np.nanpercentile(rows, 75, axis=0)

        results[name] = {
            "medians": medians,
            "q25": q25,
            "q75": q75,
        }

    return results


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------


def _write_sentinel(sentinel: Path, session_id: str, produced: list[Path]) -> None:
    data = {
        "session_id": session_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_pngs": len(produced),
        "pngs": [str(p) for p in produced],
    }
    with open(sentinel, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def run_touch_feature_radar(
    session_configs: list,
    aggregation: str = "mean_during_iff",
    force_processing: bool = False,
) -> None:
    """Render per-session touch feature radar plots.

    For each session, produces one radar PNG per gesture type present (≥2
    touches) plus a composite overlay PNG when ≥2 gesture types are available.

    Outputs are written to ``{db}/4_analysed/touch_feature_radar/{session_id}/``.

    Parameters
    ----------
    session_configs:
        List of ``(aggregated_csv_path, db_path)`` tuples, following the same
        convention as other pipelines in this package.
    aggregation:
        Aggregation folder name for the feature CSV (default: ``"mean_during_iff"``).
    force_processing:
        If ``True``, reprocess sessions even when the sentinel file is fresh.
    """
    for csv_path, database_path in session_configs:
        csv_path = Path(csv_path)
        database_path = Path(database_path)

        session_id = session_id_from_path(csv_path)
        output_dir = database_path / "4_analysed" / "touch_feature_radar" / session_id
        sentinel = output_dir / "radar_sentinel.json"

        # --- Idempotency check ---
        feature_csv = _find_feature_csv(database_path, aggregation, session_id)
        if not should_process_task(
            output_paths=sentinel,
            input_paths=feature_csv,
            force=force_processing,
        ):
            logger.info(
                "[Touch Feature Radar] %s: outputs up-to-date — skipping.", session_id
            )
            continue

        print(f"[Touch Feature Radar] {session_id}: processing...")

        # --- Load feature CSV ---
        df = pd.read_csv(feature_csv)

        # --- Validate gesture_type column ---
        if "gesture_type" not in df.columns:
            raise ValueError(
                f"[Touch Feature Radar] {session_id}: 'gesture_type' column not found "
                f"in {feature_csv}. Available columns: {list(df.columns)}"
            )

        # --- Resolve feature columns ---
        resolved_cols = _resolve_feature_columns(df, feature_csv)

        # --- Compute per-gesture statistics ---
        stats = _compute_session_stats(df, resolved_cols)

        if not stats:
            logger.info(
                "[Touch Feature Radar] %s: no gesture type has ≥2 touches — no PNGs produced.",
                session_id,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_sentinel(sentinel, session_id, produced=[])
            continue

        # --- Render outputs ---
        output_dir.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []

        # Individual per-gesture radar plots
        for gesture_name, gstats in stats.items():
            color = GESTURE_COLORS.get(gesture_name, "#ffffff")
            title = f"{session_id} | {gesture_name} | touch feature radar"
            out_path = output_dir / f"{session_id}_radar_{gesture_name}.png"

            render_gesture_radar(
                labels=RADAR_DISPLAY_LABELS,
                medians=gstats["medians"],
                q25=gstats["q25"],
                q75=gstats["q75"],
                title=title,
                color=color,
                out_path=out_path,
            )
            produced.append(out_path)
            print(f"[Touch Feature Radar] {session_id}: saved {out_path.name}")

        # Composite overlay when ≥2 gesture types (excluding "all")
        non_all_gesture_names = [g for g in stats if g != "all"]
        if len(non_all_gesture_names) >= 2:
            composite_stats: dict[str, dict] = {}
            for gesture_name, gstats in stats.items():
                if gesture_name == "all":
                    continue
                composite_stats[gesture_name] = {
                    "medians": gstats["medians"],
                    "q25": gstats["q25"],
                    "q75": gstats["q75"],
                    "color": GESTURE_COLORS.get(gesture_name, "#ffffff"),
                }

            composite_path = output_dir / f"{session_id}_radar_composite.png"
            render_gesture_radar_composite(
                labels=RADAR_DISPLAY_LABELS,
                gesture_stats=composite_stats,
                title=f"{session_id} | all gesture types | touch feature radar",
                out_path=composite_path,
            )
            produced.append(composite_path)
            print(f"[Touch Feature Radar] {session_id}: saved {composite_path.name}")

        # --- Write sentinel ---
        _write_sentinel(sentinel, session_id, produced=produced)
        print(
            f"[Touch Feature Radar] {session_id}: done — {len(produced)} PNG(s) written."
        )

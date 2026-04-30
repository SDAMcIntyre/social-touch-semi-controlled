"""
Diagnostic script for stroke proximal/distal classification imbalance.

Investigates why gesture_type_summary.csv shows a ~20x imbalance
(stroke_proximal << stroke_distal) by running three phases:

  Phase 1 — Summary triage: per-session ratio analysis
  Phase 2 — Delta-x distribution: histogram + axis dominance check
  Phase 5 — Interpolation boundary: raw Kinect samples vs interpolated endpoints

Usage
-----
    # GUI mode (directory picker dialog):
    python code/scripts/diagnose_stroke_direction.py

    # CLI mode:
    python code/scripts/diagnose_stroke_direction.py <database_root>

The script derives both paths from the database root:
    - 4_analysed/preparation/  (prepared CSVs + gesture_type_summary.csv)
    - 3_merged/                (raw aggregated session CSVs for Phase 5)
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import pandas as pd

GROUP_KEYS = ['block_order_id', 'trial_id', 'single_touch_id']
GESTURE_TYPES = ('tap', 'stroke_proximal', 'stroke_distal')
GROUP_FFILL_COLUMNS = [
    'trial_id', 'single_touch_id', 'type_metadata',
    'speed_metadata', 'contact_area_metadata', 'force_metadata',
]


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def phase1_summary_triage(preparation_dir: Path) -> pd.DataFrame:
    """Read gesture_type_summary.csv and report per-session imbalance."""
    print("\n" + "=" * 70)
    print("PHASE 1: Summary Triage")
    print("=" * 70)

    summary_path = preparation_dir / "gesture_type_summary.csv"
    if not summary_path.exists():
        print(f"  ERROR: {summary_path} not found.")
        sys.exit(1)

    df = pd.read_csv(summary_path)
    print(f"\n  Loaded {len(df)} sessions from {summary_path.name}\n")

    df['stroke_total'] = df['stroke_proximal'] + df['stroke_distal']
    df['pct_proximal'] = (
        df['stroke_proximal'] / df['stroke_total'].replace(0, np.nan) * 100
    ).round(1)
    df['ratio_d_to_p'] = (
        df['stroke_distal'] / df['stroke_proximal'].replace(0, np.nan)
    ).round(1)

    cols = [
        'session_id', 'tap', 'stroke_proximal', 'stroke_distal',
        'pct_proximal', 'ratio_d_to_p',
    ]
    print(df[cols].to_string(index=False))

    total_p = int(df['stroke_proximal'].sum())
    total_d = int(df['stroke_distal'].sum())
    print(f"\n  Global totals: proximal={total_p}  distal={total_d}  "
          f"ratio={total_d / max(total_p, 1):.1f}x")

    balanced = df[df['pct_proximal'].between(30, 70)]
    inverted = df[df['pct_proximal'] > 70]
    biased = len(df) - len(balanced) - len(inverted)
    print(f"\n  Sessions balanced (30-70 % proximal): {len(balanced)}")
    print(f"  Sessions inverted (proximal > 70 %):  {len(inverted)}")
    print(f"  Sessions biased distal (< 30 %):      {biased}")

    if len(balanced) == 0 and len(inverted) == 0:
        print("\n  >> DIAGNOSTIC: ALL sessions are biased distal — systematic cause.")
        print("     Proceed to Phase 2.")
    elif len(balanced) + len(inverted) > 0:
        print("\n  >> DIAGNOSTIC: Imbalance varies across sessions — may be session-specific.")
        print("     Check PCA sign consistency (Phase 3).")

    return df


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def phase2_delta_x_distribution(preparation_dir: Path) -> pd.DataFrame:
    """Compute delta_x for every stroke group and analyse the distribution."""
    print("\n" + "=" * 70)
    print("PHASE 2: Delta-x Distribution")
    print("=" * 70)

    prepared_csvs = sorted(preparation_dir.glob("*_prepared.csv"))
    print(f"\n  Found {len(prepared_csvs)} prepared CSVs")

    records = []
    for csv_path in prepared_csvs:
        session_id = csv_path.stem.replace("_prepared", "")
        df = pd.read_csv(csv_path)

        strokes = df[df['gesture_type'].isin(['stroke_proximal', 'stroke_distal'])]
        for key, group in strokes.groupby(GROUP_KEYS, sort=False):
            if group.empty:
                continue
            x = group['contact_location_x']
            y = group['contact_location_y']
            records.append({
                'session_id': session_id,
                'block_order_id': key[0],
                'trial_id': key[1],
                'single_touch_id': key[2],
                'start_x': x.iloc[0],
                'end_x': x.iloc[-1],
                'delta_x': x.iloc[-1] - x.iloc[0],
                'x_range': x.max() - x.min(),
                'y_range': y.max() - y.min(),
                'n_rows': len(group),
                'gesture_type': group['gesture_type'].iloc[0],
            })

    results = pd.DataFrame(records)
    print(f"  Total stroke groups: {len(results)}")

    n_pos = int((results['delta_x'] > 0).sum())
    n_neg = int((results['delta_x'] < 0).sum())
    n_zero = int((results['delta_x'] == 0).sum())
    print(f"  Δx > 0 (proximal): {n_pos}")
    print(f"  Δx < 0 (distal):   {n_neg}")
    print(f"  Δx = 0 (distal):   {n_zero}")

    print(f"\n  --- Δx statistics (mm) ---")
    print(f"  {results['delta_x'].describe().to_string()}")

    print(f"\n  --- Axis dominance ---")
    mean_xr = results['x_range'].mean()
    mean_yr = results['y_range'].mean()
    print(f"  Mean X-range: {mean_xr:.2f} mm")
    print(f"  Mean Y-range: {mean_yr:.2f} mm")
    print(f"  Ratio X/Y:    {mean_xr / max(mean_yr, 0.001):.2f}")

    if mean_xr < mean_yr:
        print("  >> WARNING: Y-range > X-range — strokes have more motion in Y than X.")
        print("     The PCA X-axis may NOT be aligned with the proximal-distal direction.")
    else:
        print("  >> X-range dominates — PCA X-axis appears aligned with stroke direction.")

    median_dx = results['delta_x'].median()
    if abs(median_dx) < 1.0:
        print(f"\n  >> DIAGNOSTIC: Median Δx = {median_dx:.3f} mm (near zero).")
        print("     Stroke displacement in X is tiny — classification is noise-driven.")
        print("     Likely cause: wrong axis or weak PCA alignment.")
    elif (results['delta_x'] < 0).mean() > 0.8:
        print(f"\n  >> DIAGNOSTIC: {(results['delta_x'] < 0).mean()*100:.0f}% of strokes have Δx < 0.")
        print("     PCA X-axis likely points distal → proximal (sign is flipped).")
        print("     Fix: negate X in PCA engine or swap label logic.")
    elif (results['delta_x'] > 0).mean() > 0.8:
        print(f"\n  >> DIAGNOSTIC: {(results['delta_x'] > 0).mean()*100:.0f}% of strokes have Δx > 0.")
        print("     Classification rule assigns these as proximal, which contradicts the")
        print("     observed imbalance. Double-check gesture_type_summary.csv.")
    else:
        print(f"\n  >> DIAGNOSTIC: Δx is spread across both signs.")
        print("     No single systematic cause — investigate per-session variation.")

    # save diagnostic CSV
    out_path = preparation_dir / '_diagnostic_stroke_delta_x.csv'
    results.to_csv(out_path, index=False)
    print(f"\n  Saved per-stroke diagnostics → {out_path.name}")

    # attempt plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(results['delta_x'], bins=80, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', ls='--', lw=2, label='threshold (0)')
        axes[0].set_xlabel('Δx = end_x − start_x (mm)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Δx distribution (all strokes)')
        axes[0].legend()

        axes[1].scatter(results['x_range'], results['y_range'], alpha=0.3, s=10)
        lim = max(results['x_range'].max(), results['y_range'].max()) * 1.05
        axes[1].plot([0, lim], [0, lim], 'r--', label='x = y')
        axes[1].set_xlabel('X range (mm)')
        axes[1].set_ylabel('Y range (mm)')
        axes[1].set_title('X-range vs Y-range per stroke')
        axes[1].legend()

        session_ids = results['session_id'].unique()
        for i, sid in enumerate(session_ids):
            sess = results[results['session_id'] == sid]
            axes[2].scatter([i] * len(sess), sess['delta_x'], alpha=0.3, s=10)
        axes[2].set_xticks(range(len(session_ids)))
        axes[2].set_xticklabels(session_ids, rotation=90, fontsize=7)
        axes[2].axhline(0, color='red', ls='--')
        axes[2].set_ylabel('Δx (mm)')
        axes[2].set_title('Δx per session')

        plt.tight_layout()
        plot_path = preparation_dir / '_diagnostic_delta_x_distribution.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot → {plot_path.name}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")

    return results


# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------

def phase5_interpolation_boundary(raw_dir: Path, preparation_dir: Path) -> None:
    """Compare raw Kinect-sample Δx vs post-interpolation Δx."""
    print("\n" + "=" * 70)
    print("PHASE 5: Interpolation Boundary Effect")
    print("=" * 70)

    raw_csvs = sorted(raw_dir.glob("*_semicontrolled_aggregated_session.csv"))
    if not raw_csvs:
        raw_csvs = sorted(raw_dir.rglob("*_semicontrolled_aggregated_session.csv"))
    print(f"\n  Found {len(raw_csvs)} raw aggregated CSVs")

    if not raw_csvs:
        print("  ERROR: no aggregated session CSVs found. Skipping Phase 5.")
        return

    records = []
    for csv_path in raw_csvs:
        session_id = csv_path.stem.replace("_semicontrolled_aggregated_session", "")
        df = pd.read_csv(csv_path, low_memory=False)

        if 'contact_location_x' not in df.columns:
            print(f"  SKIP {session_id}: no contact_location_x column")
            continue

        for col in GROUP_FFILL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].ffill()

        stroke_mask = (df.get('type_metadata') == 'stroke') & (df.get('single_touch_id', 0) != 0)
        if not stroke_mask.any():
            continue

        stroke_df = df[stroke_mask]

        for (bid, tid, sid), group in stroke_df.groupby(GROUP_KEYS, sort=False):
            if group.empty:
                continue
            x = group['contact_location_x']
            valid = x.dropna()
            if len(valid) < 2:
                continue

            records.append({
                'session_id': session_id,
                'block_order_id': bid,
                'trial_id': tid,
                'single_touch_id': sid,
                'n_total_rows': len(x),
                'n_valid_x': len(valid),
                'first_valid_x': valid.iloc[0],
                'last_valid_x': valid.iloc[-1],
                'raw_delta_x': valid.iloc[-1] - valid.iloc[0],
                'iloc0_is_nan': pd.isna(x.iloc[0]),
                'iloc_last_is_nan': pd.isna(x.iloc[-1]),
                'rows_before_first_valid': valid.index[0] - group.index[0],
                'rows_after_last_valid': group.index[-1] - valid.index[-1],
            })

    if not records:
        print("  No stroke groups found in raw data.")
        return

    raw_df = pd.DataFrame(records)
    raw_df['raw_proximal'] = raw_df['raw_delta_x'] > 0

    n_prox = int(raw_df['raw_proximal'].sum())
    n_dist = int((~raw_df['raw_proximal']).sum())
    print(f"\n  Total stroke groups (raw): {len(raw_df)}")
    print(f"  Raw proximal (Δx > 0): {n_prox}")
    print(f"  Raw distal   (Δx ≤ 0): {n_dist}")
    print(f"  Raw ratio distal/proximal: {n_dist / max(n_prox, 1):.1f}x")

    pct_iloc0_nan = raw_df['iloc0_is_nan'].mean() * 100
    pct_last_nan = raw_df['iloc_last_is_nan'].mean() * 100
    print(f"\n  Groups where iloc[0] is NaN (pre-interp): {pct_iloc0_nan:.1f}%")
    print(f"  Groups where iloc[-1] is NaN (pre-interp): {pct_last_nan:.1f}%")

    print(f"\n  Rows between group boundary and first/last valid Kinect sample:")
    print(f"    Before first valid: median={raw_df['rows_before_first_valid'].median():.0f}  "
          f"max={raw_df['rows_before_first_valid'].max()}")
    print(f"    After last valid:   median={raw_df['rows_after_last_valid'].median():.0f}  "
          f"max={raw_df['rows_after_last_valid'].max()}")

    if n_dist / max(n_prox, 1) < 3:
        print("\n  >> DIAGNOSTIC: Raw Kinect-sample direction is roughly balanced.")
        print("     The interpolation step is likely distorting boundary values.")
        print("     Fix: use first/last non-NaN contact_location_x instead of iloc[0]/iloc[-1].")
    else:
        print("\n  >> DIAGNOSTIC: Raw data also shows strong distal bias.")
        print("     Problem is upstream of preparation — likely PCA axis/sign.")

    out_path = preparation_dir / '_diagnostic_interpolation_boundary.csv'
    raw_df.to_csv(out_path, index=False)
    print(f"\n  Saved raw boundary diagnostics → {out_path.name}")


# ---------------------------------------------------------------------------
# GUI directory picker
# ---------------------------------------------------------------------------

_PREPARATION_SUFFIX = Path("4_analysed") / "preparation"
_MERGED_SUFFIX = Path("3_merged")


def _resolve_database_root(path: Path) -> Path:
    if path.parts[-2:] == _PREPARATION_SUFFIX.parts:
        return path.parent.parent
    return path


def _preparation_dir(database_root: Path) -> Path:
    return database_root / _PREPARATION_SUFFIX


def _merged_dir(database_root: Path) -> Path:
    return database_root / _MERGED_SUFFIX


def _ask_directory(title: str) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    chosen = filedialog.askdirectory(title=title)
    root.destroy()
    if not chosen:
        return None
    return Path(chosen)


def _resolve_paths_gui() -> tuple[Path, Path | None]:
    selected = _ask_directory("Select database root directory")
    if selected is None:
        print("No directory selected — exiting.")
        sys.exit(0)

    database_root = _resolve_database_root(selected)
    preparation = _preparation_dir(database_root)
    merged = _merged_dir(database_root)

    if not preparation.is_dir():
        print(f"ERROR: {preparation} does not exist.")
        sys.exit(1)

    raw_dir = merged if merged.is_dir() else None
    return preparation, raw_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_diagnostic(preparation_dir: Path, raw_dir: Path | None) -> None:
    phase1_summary_triage(preparation_dir)
    phase2_delta_x_distribution(preparation_dir)

    if raw_dir is not None:
        phase5_interpolation_boundary(raw_dir, preparation_dir)
    else:
        print("\n  (Phase 5 skipped)")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("Next steps based on results:")
    print("  - If PCA sign flip: inspect pca-xyz_transformation-matrices.json")
    print("  - If wrong axis: check PCA calibration alignment")
    print("  - If interpolation boundary: fix classify_gesture_type to use")
    print("    first/last non-NaN values instead of iloc[0]/iloc[-1]")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose stroke proximal/distal classification imbalance",
    )
    parser.add_argument(
        'database_root',
        nargs='?',
        type=Path,
        default=None,
        help="Database root directory (omit to use GUI picker). "
             "Derives 4_analysed/preparation/ and 3_merged/ automatically.",
    )
    args = parser.parse_args()

    if args.database_root is None:
        preparation_dir, raw_dir = _resolve_paths_gui()
    else:
        database_root = _resolve_database_root(args.database_root)
        preparation_dir = _preparation_dir(database_root)
        raw_dir = _merged_dir(database_root)
        if not raw_dir.is_dir():
            raw_dir = None

    if not preparation_dir.is_dir():
        print(f"ERROR: {preparation_dir} is not a directory.")
        sys.exit(1)

    _run_diagnostic(preparation_dir, raw_dir)


if __name__ == '__main__':
    main()

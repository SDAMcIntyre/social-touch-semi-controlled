"""Print min/max of every numeric touch feature across all sessions.

Usage (from repo root):
    conda activate social-touch-env
    python code/scripts/inspect_touch_feature_ranges.py
    python code/scripts/inspect_touch_feature_ranges.py --percentile 90
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import path_tools

TOUCH_KEYS = {"block_order_id", "trial_id", "single_touch_id"}
SHARED_NON_NUMERIC = {
    "type_metadata", "gesture_type", "session_id",
    "mean_contact_x", "mean_contact_y", "mean_contact_z",
    "spike_elicited",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print min/max/percentile of touch features.")
    parser.add_argument(
        "--percentile", type=float, default=99,
        help="Percentile range to show (default: 99 → P0.5 and P99.5)",
    )
    args = parser.parse_args()

    pct = args.percentile
    p_lo = (100 - pct) / 2
    p_hi = 100 - p_lo

    project_data_root = path_tools.get_project_data_root()
    if project_data_root is None:
        print("ERROR: could not resolve project data root.")
        sys.exit(1)

    touch_features_dir = project_data_root / "4_analysed" / "touch_features"
    if not touch_features_dir.is_dir():
        print(f"ERROR: touch_features directory not found: {touch_features_dir}")
        sys.exit(1)

    csvs = sorted(touch_features_dir.rglob("*_touch_summary.csv"))
    if not csvs:
        print(f"ERROR: no *_touch_summary.csv files found under {touch_features_dir}")
        sys.exit(1)

    print(f"Found {len(csvs)} touch-feature CSVs under {touch_features_dir}\n")

    col_values: dict[str, list[np.ndarray]] = {}

    skip_cols = TOUCH_KEYS | SHARED_NON_NUMERIC

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  WARN: could not read {csv_path.name}: {exc}")
            continue

        numeric_cols = [
            c for c in df.columns
            if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        for col in numeric_cols:
            vals = df[col].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue
            col_values.setdefault(col, []).append(vals)

    if not col_values:
        print("No numeric feature columns found.")
        sys.exit(1)

    stats: dict[str, tuple[float, float, float, float, int]] = {}
    for col, parts in col_values.items():
        all_vals = np.concatenate(parts)
        stats[col] = (
            float(np.min(all_vals)),
            float(np.percentile(all_vals, p_lo)),
            float(np.percentile(all_vals, p_hi)),
            float(np.max(all_vals)),
            len(all_vals),
        )

    families: dict[str, list[str]] = {}
    for col in sorted(stats):
        for suffix in ("_mean_during_iff", "_mean_before_iff", "_mean", "_median",
                        "_std", "_min", "_max", "_range", "_skewness"):
            if col.endswith(suffix):
                families.setdefault(suffix.lstrip("_"), []).append(col)
                break
        else:
            families.setdefault("other", []).append(col)

    p_lo_label = f"P{p_lo:g}"
    p_hi_label = f"P{p_hi:g}"
    print(f"{'Feature':<55s}  {'Min':>12s}  {p_lo_label:>12s}  {p_hi_label:>12s}  {'Max':>12s}  {'N':>7s}")
    print("-" * 118)

    for family in sorted(families):
        cols = families[family]
        print(f"\n  [{family}]")
        for col in sorted(cols):
            lo, plo, phi, hi, n = stats[col]
            print(f"  {col:<53s}  {lo:>12.4f}  {plo:>12.4f}  {phi:>12.4f}  {hi:>12.4f}  {n:>7d}")

    print()


if __name__ == "__main__":
    main()

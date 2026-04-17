"""
Aggregate Kinect RGB ↔ depth offset measurements collected with the
``KinectRgbDepthViewer`` audit tool.

The script loads every CSV under a measurement root (default
``F:/_tmp/kinect-measurement-offsets``), filters out misclick/off-image
fallback rows (|Δ| > 100 px), and reports a single-bin near-field
tolerance band (500–800 mm — the operating range of the downstream
pipelines that consume paired RGB + depth).

Outputs:
    - markdown table on stdout (per-session and pooled near-field stats),
    - 2-panel PNG written next to the input root,
    - one-line decision string with the headline band.

Far-range rows (z > 800 mm) are reported as a single line "for
reference" and do not influence the band — by design, since the
downstream consumers do not operate at those distances.

Usage
-----
    python analyse_rgb_depth_offset_dataset.py
    python analyse_rgb_depth_offset_dataset.py --root <dir> --out <png>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles default to cp1252 — reconfigure stdout/stderr so that
# the Δ glyph in the markdown output does not crash the script.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(r"F:\_tmp\kinect-measurement-offsets")
NEAR_FIELD_MIN_MM = 500.0
NEAR_FIELD_MAX_MM = 800.0
OUTLIER_DELTA_PX = 100.0


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["source_csv"] = path.name
    df["phase"] = path.parent.name  # phase-1 / phase-2
    df["session"] = path.stem.replace("_kinect-measurement-offsets", "").replace(
        "_phase-2", ""
    )
    return df


def load_all(root: Path) -> pd.DataFrame:
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under {root}")
    frames = [_load_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    # Phase-1 schema lacks the median/std columns; align dtypes for delta_mag.
    df["delta_mag"] = pd.to_numeric(df["delta_mag"], errors="coerce")
    df["delta_u"] = pd.to_numeric(df["delta_u"], errors="coerce")
    df["delta_v"] = pd.to_numeric(df["delta_v"], errors="coerce")
    df["z_mm"] = pd.to_numeric(df["z_mm"], errors="coerce")
    return df


def drop_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask_outlier = df["delta_mag"].abs() > OUTLIER_DELTA_PX
    return df.loc[~mask_outlier].copy(), df.loc[mask_outlier].copy()


def _sign_consistency(series: pd.Series) -> float:
    """Return % of rows whose sign matches the modal sign (excluding zeros)."""
    signs = np.sign(series.dropna())
    signs = signs[signs != 0]
    if len(signs) == 0:
        return float("nan")
    pos = float((signs > 0).sum())
    neg = float((signs < 0).sum())
    return 100.0 * max(pos, neg) / len(signs)


def _stats_block(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    delta = df["delta_mag"]
    q1, q3 = delta.quantile(0.25), delta.quantile(0.75)
    near_edge = (
        int(df["near_edge"].astype(str).str.lower().eq("true").sum())
        if "near_edge" in df.columns
        else 0
    )
    return {
        "n": int(len(df)),
        "median_abs": float(delta.median()),
        "iqr_lo": float(q1),
        "iqr_hi": float(q3),
        "max_abs": float(delta.max()),
        "median_du": float(df["delta_u"].median()),
        "median_dv": float(df["delta_v"].median()),
        "du_sign_consistency_pct": _sign_consistency(df["delta_u"]),
        "near_edge_count": near_edge,
    }


def print_markdown_table(by_session: pd.DataFrame, pooled: dict) -> None:
    print()
    print("## Near-field band (500–800 mm) — per session")
    print()
    print(
        "| session | phase | n | median |Δ| (px) | IQR (px) | max |Δ| (px) | "
        "median Δu (px) | Δu sign-consistency (%) | near_edge=True |"
    )
    print(
        "|---------|-------|---|----------------|----------|--------------|"
        "----------------|-------------------------|----------------|"
    )
    for (session, phase), grp in by_session.groupby(["session", "phase"], sort=True):
        s = _stats_block(grp)
        if s["n"] == 0:
            continue
        print(
            f"| {session} | {phase} | {s['n']} | {s['median_abs']:.2f} | "
            f"[{s['iqr_lo']:.2f}, {s['iqr_hi']:.2f}] | {s['max_abs']:.2f} | "
            f"{s['median_du']:+.2f} | {s['du_sign_consistency_pct']:.0f} | "
            f"{s['near_edge_count']} |"
        )
    print()
    print("## Near-field band (500–800 mm) — pooled")
    print()
    print(
        f"- n = {pooled['n']}\n"
        f"- median |Δ| = {pooled['median_abs']:.2f} px\n"
        f"- IQR |Δ| = [{pooled['iqr_lo']:.2f}, {pooled['iqr_hi']:.2f}] px\n"
        f"- worst |Δ| = {pooled['max_abs']:.2f} px\n"
        f"- median Δu = {pooled['median_du']:+.2f} px\n"
        f"- median Δv = {pooled['median_dv']:+.2f} px\n"
        f"- Δu sign-consistency = {pooled['du_sign_consistency_pct']:.0f} %\n"
        f"- rows with near_edge=True = {pooled['near_edge_count']}"
    )


def save_plot(near: pd.DataFrame, pooled: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8))

    # Top: histogram of |Δ| with median + IQR + max overlaid.
    ax_top.hist(near["delta_mag"], bins=20, color="#4477aa", edgecolor="white")
    ax_top.axvline(pooled["median_abs"], color="black", lw=2, label="median")
    ax_top.axvspan(
        pooled["iqr_lo"], pooled["iqr_hi"], color="black", alpha=0.1, label="IQR"
    )
    ax_top.axvline(
        pooled["max_abs"], color="firebrick", lw=1.5, ls="--", label="worst"
    )
    ax_top.set_xlabel("|Δ| (px)")
    ax_top.set_ylabel("count")
    ax_top.set_title(
        f"Near-field |Δ| distribution (n={pooled['n']}, "
        f"500–800 mm, |Δ|≤{OUTLIER_DELTA_PX:.0f} px)"
    )
    ax_top.legend(loc="upper right")

    # Bottom: Δu vs u, colour by v, in the near-field bin.
    sc = ax_bot.scatter(
        near["u_rgb"], near["delta_u"], c=near["v_rgb"], cmap="viridis", s=30
    )
    ax_bot.axhline(0, color="grey", lw=0.5)
    ax_bot.set_xlabel("u_rgb (px)")
    ax_bot.set_ylabel("Δu (px)")
    ax_bot.set_title("Δu vs image-x position (near-field), colour = v_rgb")
    fig.colorbar(sc, ax=ax_bot, label="v_rgb (px)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <root>/aggregate_near_field.png).",
    )
    args = p.parse_args()

    root: Path = args.root
    out_png: Path = args.out or (root / "aggregate_near_field.png")

    df_raw = load_all(root)
    print(f"Loaded {len(df_raw)} rows from {root} "
          f"({df_raw['phase'].nunique()} phase dirs, "
          f"{df_raw['source_csv'].nunique()} CSVs).")

    df, dropped = drop_outliers(df_raw)
    if not dropped.empty:
        print(f"\nDropped {len(dropped)} row(s) with |Δ| > {OUTLIER_DELTA_PX:.0f} px:")
        for _, row in dropped.iterrows():
            print(
                f"  - {row['source_csv']} frame={row['frame']} "
                f"u_rgb={row['u_rgb']} v_rgb={row['v_rgb']} "
                f"delta_mag={row['delta_mag']:.1f}"
            )

    # Per-phase row counts (sanity check).
    print("\nRows by phase (after outlier drop):")
    for phase, n in df.groupby("phase").size().items():
        print(f"  {phase}: {n}")

    # Near-field filter.
    near_mask = (df["z_mm"] >= NEAR_FIELD_MIN_MM) & (df["z_mm"] <= NEAR_FIELD_MAX_MM)
    near = df.loc[near_mask].copy()
    far = df.loc[~near_mask & df["z_mm"].notna()].copy()

    pooled = _stats_block(near)
    print_markdown_table(near, pooled)

    print(
        f"\nFor reference (NOT used for the band): "
        f"{len(far)} row(s) at z outside "
        f"[{NEAR_FIELD_MIN_MM:.0f}, {NEAR_FIELD_MAX_MM:.0f}] mm."
    )

    save_plot(near, pooled, out_png)
    print(f"\nWrote plot → {out_png}")

    print(
        f"\nnear_field_band: median={pooled['median_abs']:.2f}px, "
        f"IQR=[{pooled['iqr_lo']:.2f},{pooled['iqr_hi']:.2f}]px, "
        f"worst={pooled['max_abs']:.2f}px, "
        f"delta_u_sign_consistency={pooled['du_sign_consistency_pct']:.0f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

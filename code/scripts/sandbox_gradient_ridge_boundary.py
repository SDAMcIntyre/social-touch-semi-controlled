"""Proof-of-concept sandbox for the gradient-ridge RF boundary method.

The gradient-ridge boundary (``rf_gradient_boundary.compute_gradient_ridge``)
recently replaced the Laplacian inflection boundary as the canonical RF
boundary, but its results are not satisfying.  This standalone script
isolates that method on a *single, hardcoded* session so the researcher can
see — step by step, in 2D and 3D — what the method is sensitive to and how
that compares to where the response-field signal actually lives.

It does **not** touch the pipeline.  It loads the already-computed population
response-field heatmap (``grid_z``) from the session's
``*_population_response_fields.npz`` and re-runs the gradient method *live*,
so every tunable (Gaussian sigma, number of rays, Savitzky-Golay window) can
be changed at the top of ``main()`` and the effect inspected immediately.

What each figure shows
----------------------
  Fig 1  Pipeline steps (2D): raw heatmap → smoothed → Laplacian →
         gradient magnitude → both contours overlaid on the heatmap.
  Fig 2  Radial-profiling diagnostic — the heart of the method.  Rays cast
         from the peak, and, for a few angles, the |grad z| profile (what the
         method maximises) next to the normalised response profile (where the
         signal of interest actually decays).  This reveals that the ridge
         sits on the *steepest slope*, not at the RF edge.
  Fig 3  Sigma sensitivity sweep — gradient-ridge contour for several smoothing
         sigmas, plus boundary area vs sigma.
  Fig 4  3D response surface with both contours drawn at their true height —
         shows the ridge clinging to the flank of the bell.
  Fig 5  3D forearm mesh coloured by the heatmap, with the gradient-ridge
         contour back-projected onto the surface (real anatomical view).

Display
-------
By default (``USE_PLT_SHOW = True``) every figure stays open at the end and you
navigate between the windows with the keyboard: ←/→ (or n/p) flips to the
previous/next figure, q closes them all.  This needs a real GUI backend, so run
it from a plain terminal.  Under VS Code's debugger set ``USE_PLT_SHOW = False``
— debugpy forces matplotlib to the Agg backend and ``plt.show`` hangs; in that
mode each figure is saved as a PNG and opened with ``os.startfile`` instead.
Either way the PNGs are written to the output folder.

Usage
-----
    python code/scripts/sandbox_gradient_ridge_boundary.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

# --- Make the analysis package importable without installing -----------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import matplotlib.pyplot as plt
# NOTE: the analysis package imports below call matplotlib.use("Agg") at import
# time, so the interactive backend cannot be selected here — it is re-asserted at
# runtime in main() via _ensure_interactive_backend(), which must run AFTER them.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (registers 3d)
from scipy.ndimage import map_coordinates

from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_inflection_boundary,
    compute_laplacian_arrays,
    contour_pixels_to_uv,
    find_peak_location,
)
from analysis.receptive_field_mapping.metrics.rf_gradient_boundary import (
    _extract_ridge_via_radial_profiling,
    compute_gradient_magnitude,
    compute_gradient_ridge,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import uv_points_to_xyz


# =============================================================================
# Display helper
# =============================================================================

# True  → keep every figure open and navigate between them with the arrow keys
#         at the end (interactive; needs a real GUI backend, e.g. a terminal run).
# False → save each PNG and open it with os.startfile (debugger-safe; VS Code's
#         debugpy forces the Agg backend and plt.show hangs — set this False there).
USE_PLT_SHOW = True

# Figures kept alive for interactive navigation: list of (figure, title) pairs.
_FIGS: list[tuple["plt.Figure", str]] = []

# Set in main(): True only when USE_PLT_SHOW and an interactive backend was secured.
_INTERACTIVE = False


def _ensure_interactive_backend() -> bool:
    """Force an interactive matplotlib backend, returning True on success.

    The analysis package imports call ``matplotlib.use("Agg")`` at import time,
    so we must switch back to a GUI backend at runtime (after those imports).
    Tries Qt then Tk; returns False if none is available (e.g. under debugpy),
    in which case the caller falls back to saving + opening PNGs.
    """
    for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            plt.switch_backend(backend)
            print(f"  interactive backend: {backend}")
            return True
        except Exception:
            continue
    print("  WARNING: no interactive backend available — opening saved PNGs instead.")
    return False


def _emit(fig: "plt.Figure", out_dir: Path, name: str) -> None:
    """Save the figure, then either keep it open for navigation or open the PNG."""
    path = out_dir / name
    fig.savefig(path, dpi=140)
    print(f"  saved {path.name}")
    if _INTERACTIVE:
        # Keep the figure alive; main() wires navigation and calls plt.show().
        _FIGS.append((fig, path.stem))
    else:
        plt.close(fig)
        try:
            os.startfile(str(path))  # noqa: B606  (Windows-only, intentional)
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"  (could not auto-open {path.name}: {exc})")


def _wire_navigation() -> None:
    """Let the user flip between all open figure windows with the keyboard.

    ←/→ (or n/p) raises the previous/next figure; q closes them all. Window
    titles are numbered so the current position is always visible.
    """
    figs = [f for f, _ in _FIGS]
    n = len(figs)
    if n == 0:
        return

    for i, (fig, title) in enumerate(_FIGS):
        try:
            fig.canvas.manager.set_window_title(
                f"[{i + 1}/{n}] {title}   (←/→ or n/p to navigate, q to close)"
            )
        except Exception:
            pass

    def _raise(idx: int) -> None:
        fig = figs[idx % n]
        mgr = getattr(fig.canvas, "manager", None)
        win = getattr(mgr, "window", None)
        if win is None:
            return
        for attr in ("activateWindow", "raise_", "lift", "focus_force"):
            method = getattr(win, attr, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    def _on_key(event) -> None:
        if event.canvas.figure not in figs:
            return
        cur = figs.index(event.canvas.figure)
        if event.key in ("right", "n", "pagedown"):
            _raise(cur + 1)
        elif event.key in ("left", "p", "pageup"):
            _raise(cur - 1)
        elif event.key == "q":
            plt.close("all")

    for fig in figs:
        fig.canvas.mpl_connect("key_press_event", _on_key)


# =============================================================================
# Data loading
# =============================================================================


def load_grid(npz_path: Path, gtype: str):
    """Load the interpolated heatmap grid + mesh + stored boundaries for one gesture.

    Returns a dict with grid_u/grid_v/grid_z, the forearm mesh arrays, the
    stored inflection_sigma the pipeline used, and the stored gradient/inflection
    contours (for reference comparison).
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)

    available = list(d["gesture_types"])
    if gtype not in available:
        raise ValueError(
            f"gesture '{gtype}' not in NPZ. Available: {available}"
        )

    def _get(stem: str):
        key = f"{stem}_{gtype}"
        if key not in d:
            raise KeyError(f"missing NPZ key '{key}'")
        return d[key]

    out = {
        "grid_u": _get("grid_u"),
        "grid_v": _get("grid_v"),
        "grid_z": _get("grid_z"),
        "forearm_uv": d["forearm_uv"],
        "forearm_faces": d["forearm_faces"],
        "forearm_V": d["forearm_V"],
        "pipeline_sigma": float(d["inflection_sigma"]),
        "pipeline_boundary_method": str(d["boundary_method"]) if "boundary_method" in d else "?",
        "gesture_types": available,
        "session_id": str(d["session_id"]),
    }
    # Stored reference contours (whatever the pipeline produced).
    out["stored_gradient_contour_uv"] = (
        d[f"gradient_contour_uv_{gtype}"] if f"gradient_contour_uv_{gtype}" in d else None
    )
    out["stored_inflection_contour_uv"] = (
        d[f"inflection_contour_uv_{gtype}"] if f"inflection_contour_uv_{gtype}" in d else None
    )
    return out


# =============================================================================
# Faithful radial-profile sampler (mirrors _extract_ridge_via_radial_profiling)
# =============================================================================


def sample_ray(field: np.ndarray, peak_rc, angle: float):
    """Sample *field* along a single ray from peak_rc at *angle*.

    Mirrors the sampling used inside ``_extract_ridge_via_radial_profiling``
    so the diagnostic faithfully reflects what the method sees.

    Returns (radii, values) — both 1D, values bilinearly interpolated with
    NaN cells treated as 0 (as the method does).
    """
    n_rows, n_cols = field.shape
    peak_r, peak_c = peak_rc
    max_radius = int(math.ceil(math.hypot(n_rows, n_cols)))
    filled = np.where(np.isnan(field), 0.0, field)

    radii = np.arange(1, max_radius, dtype=np.float64)
    rows = peak_r + radii * math.cos(angle)
    cols = peak_c + radii * math.sin(angle)
    valid = (rows >= 0) & (rows < n_rows - 1) & (cols >= 0) & (cols < n_cols - 1)
    radii, rows, cols = radii[valid], rows[valid], cols[valid]
    if radii.size == 0:
        return np.array([]), np.array([])
    values = map_coordinates(filled, np.array([rows, cols]), order=1, mode="nearest")
    return radii, values


def contour_height(grid_z: np.ndarray, contour_rc: np.ndarray) -> np.ndarray:
    """Bilinearly sample grid_z at contour (row, col) points (NaN→nanmean fill)."""
    filled = np.where(np.isnan(grid_z), np.nanmean(grid_z), grid_z)
    return map_coordinates(
        filled, np.array([contour_rc[:, 0], contour_rc[:, 1]]), order=1, mode="nearest"
    )


# =============================================================================
# Figures
# =============================================================================


def fig1_pipeline_steps(g, smoothed, laplacian, grad_mag, peak_rc,
                        grad_contour_rc, infl_contour_rc, out_dir):
    """Five-panel 2D walk-through of the gradient-ridge pipeline."""
    grid_z = g["grid_z"]
    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (ax_raw, ax_sm, ax_lap), (ax_grad, ax_overlay, ax_text) = axes

    ax_raw.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")
    ax_raw.plot(peak_rc[1], peak_rc[0], "rx", ms=11, mew=2)
    ax_raw.set_title("1. Raw heatmap grid_z (peak ✕)")

    ax_sm.imshow(np.ma.masked_invalid(smoothed), cmap=jet, origin="upper")
    ax_sm.set_title("2. NaN-aware Gaussian smoothed")

    vabs = max(float(np.nanmax(np.abs(laplacian))), 1e-12)
    ax_lap.imshow(laplacian, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="upper")
    ax_lap.contour(np.where(np.isnan(laplacian), 0.0, laplacian),
                   levels=[0.0], colors="black", linewidths=0.6)
    ax_lap.set_title("3. Laplacian (zero-crossing = inflection)")

    vmax_g = max(float(np.nanmax(grad_mag)), 1e-12)
    ax_grad.imshow(np.ma.masked_invalid(grad_mag), cmap="inferno",
                   vmin=0, vmax=vmax_g, origin="upper")
    if grad_contour_rc is not None:
        ax_grad.plot(grad_contour_rc[:, 1], grad_contour_rc[:, 0], "-", color="lime", lw=1.6)
    ax_grad.plot(peak_rc[1], peak_rc[0], "cx", ms=11, mew=2)
    ax_grad.set_title("4. |∇z| gradient magnitude + ridge\n(THIS is what the method maximises)")

    ax_overlay.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")
    # Expectation reference: iso-contours of the response itself.
    zf = np.where(np.isnan(grid_z), np.nanmin(grid_z), grid_z)
    ax_overlay.contour(zf, levels=6, colors="white", linewidths=0.5, alpha=0.6)
    if grad_contour_rc is not None:
        ax_overlay.plot(grad_contour_rc[:, 1], grad_contour_rc[:, 0], "-",
                        color="lime", lw=2.0, label="gradient ridge (new)")
    if infl_contour_rc is not None:
        ax_overlay.plot(infl_contour_rc[:, 1], infl_contour_rc[:, 0], "-",
                        color="red", lw=2.0, label="Laplacian inflection (old)")
    ax_overlay.plot(peak_rc[1], peak_rc[0], "kx", ms=11, mew=2)
    ax_overlay.legend(loc="upper right", fontsize=8)
    ax_overlay.set_title("5. Both boundaries vs response iso-contours")

    # Text panel: quantitative comparison.
    ax_text.axis("off")
    lines = ["Method sensitivity summary", "-" * 30]
    peak_val = float(np.nanmax(grid_z))
    lines.append(f"peak grid_z value : {peak_val:.3f}")
    for label, c_rc in (("gradient ridge", grad_contour_rc),
                        ("inflection", infl_contour_rc)):
        if c_rc is None:
            lines.append(f"{label:>14}: <none>")
            continue
        h = contour_height(grid_z, c_rc)
        r = np.hypot(c_rc[:, 0] - peak_rc[0], c_rc[:, 1] - peak_rc[1])
        lines.append(
            f"{label:>14}: mean r={r.mean():5.1f}px  "
            f"height={h.mean():.3f} ({100*h.mean()/peak_val:4.0f}% of peak)"
        )
    lines += ["", "If the ridge sits at a high % of peak,",
              "it hugs the hotspot, not the RF edge."]
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                 family="monospace", fontsize=10, transform=ax_text.transAxes)

    fig.suptitle(f"{g['session_id']} — gradient-ridge pipeline steps", fontsize=13)
    fig.tight_layout()
    _emit(fig, out_dir, "fig1_pipeline_steps.png")


def fig2_radial_profiling(g, grad_mag, peak_rc, grad_contour_rc, n_angles,
                          out_dir, n_show=6):
    """Show the rays and, for a few angles, |grad z| profile vs response profile."""
    grid_z = g["grid_z"]
    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")
    peak_val = float(np.nanmax(grid_z))

    fig = plt.figure(figsize=(16, 8))
    ax_map = fig.add_subplot(1, 2, 1)
    ax_prof = fig.add_subplot(1, 2, 2)

    ax_map.imshow(np.ma.masked_invalid(grad_mag), cmap="inferno", origin="upper",
                  vmin=0, vmax=max(float(np.nanmax(grad_mag)), 1e-12))
    if grad_contour_rc is not None:
        closed = np.vstack([grad_contour_rc, grad_contour_rc[:1]])
        ax_map.plot(closed[:, 1], closed[:, 0], "-", color="lime", lw=1.8)
    ax_map.plot(peak_rc[1], peak_rc[0], "cx", ms=12, mew=2)

    show_angles = np.linspace(0, 2 * np.pi, n_show, endpoint=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))
    for angle, col in zip(show_angles, colors):
        radii, gvals = sample_ray(grad_mag, peak_rc, angle)
        if radii.size == 0:
            continue
        # Draw the ray on the map.
        rr = peak_rc[0] + radii * math.cos(angle)
        cc = peak_rc[1] + radii * math.sin(angle)
        ax_map.plot(cc, rr, "-", color=col, lw=0.9, alpha=0.8)

        # Profiles: gradient magnitude (solid) vs normalised response (dashed).
        _, zvals = sample_ray(grid_z, peak_rc, angle)
        gnorm = gvals / max(gvals.max(), 1e-12)
        znorm = zvals / max(peak_val, 1e-12)
        deg = int(round(math.degrees(angle)))
        ax_prof.plot(radii, gnorm, "-", color=col, lw=1.3, label=f"|∇z| {deg}°")
        ax_prof.plot(radii[: len(znorm)], znorm, "--", color=col, lw=1.0, alpha=0.7)
        # Mark the chosen ridge radius (argmax of |grad z|).
        idx = int(np.argmax(gvals))
        ax_prof.plot(radii[idx], gnorm[idx], "o", color=col, ms=6)

    ax_map.set_title(f"Rays from peak ({n_angles} used in method, {n_show} shown)")
    ax_prof.set_xlabel("radius from peak (px)")
    ax_prof.set_ylabel("normalised value")
    ax_prof.set_title("Solid = |∇z| (method target, ● = chosen radius)\n"
                      "Dashed = response grid_z (where signal actually decays)")
    ax_prof.legend(fontsize=7, ncol=2)
    ax_prof.grid(alpha=0.3)

    fig.suptitle(f"{g['session_id']} — radial-profiling: what the method is sensitive to",
                 fontsize=13)
    fig.tight_layout()
    _emit(fig, out_dir, "fig2_radial_profiling.png")


def fig3_sigma_sweep(g, sigmas, n_angles, savgol_window, out_dir):
    """Gradient-ridge contour for several smoothing sigmas + area-vs-sigma."""
    grid_u, grid_v, grid_z = g["grid_u"], g["grid_v"], g["grid_z"]
    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, (ax_map, ax_area) = plt.subplots(1, 2, figsize=(15, 6.5))
    ax_map.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")

    colors = plt.cm.viridis(np.linspace(0, 1, len(sigmas)))
    areas = []
    for sigma, col in zip(sigmas, colors):
        smoothed, _ = compute_laplacian_arrays(grid_z, sigma)
        b = compute_gradient_ridge(grid_u, grid_v, grid_z, smoothed,
                                   n_angles=n_angles, savgol_window=savgol_window)
        if b is None:
            areas.append(np.nan)
            continue
        areas.append(b.area_uv)
        # Convert UV contour back to pixel space for the imshow overlay.
        n_rows, n_cols = grid_z.shape
        u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
        v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])
        rr = (b.contour_uv[:, 0] - u_min) / (u_max - u_min) * (n_rows - 1)
        cc = (b.contour_uv[:, 1] - v_min) / (v_max - v_min) * (n_cols - 1)
        rr = np.append(rr, rr[0]); cc = np.append(cc, cc[0])
        ax_map.plot(cc, rr, "-", color=col, lw=1.6, label=f"σ={sigma:g}")

    ax_map.legend(fontsize=8, loc="upper right")
    ax_map.set_title("Gradient-ridge contour vs Gaussian σ")

    ax_area.plot(sigmas, areas, "o-", color="green")
    ax_area.set_xlabel("Gaussian σ")
    ax_area.set_ylabel("boundary area (UV²)")
    ax_area.set_title("Boundary area vs σ (parameter sensitivity)")
    ax_area.grid(alpha=0.3)

    fig.suptitle(f"{g['session_id']} — σ sensitivity sweep", fontsize=13)
    fig.tight_layout()
    _emit(fig, out_dir, "fig3_sigma_sweep.png")


def fig4_surface_3d(g, grad_contour_rc, infl_contour_rc, out_dir):
    """3D response surface with both contours drawn at their true height."""
    grid_u, grid_v, grid_z = g["grid_u"], g["grid_v"], g["grid_z"]
    Z = np.ma.masked_invalid(grid_z)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(grid_v, grid_u, np.where(np.isnan(grid_z), np.nan, grid_z),
                    cmap="jet", linewidth=0, antialiased=True, alpha=0.85)

    for c_rc, col, lbl in ((grad_contour_rc, "lime", "gradient ridge"),
                           (infl_contour_rc, "red", "inflection")):
        if c_rc is None:
            continue
        uv = contour_pixels_to_uv(c_rc, grid_u, grid_v)
        h = contour_height(grid_z, c_rc)
        closed_u = np.append(uv[:, 0], uv[0, 0])
        closed_v = np.append(uv[:, 1], uv[0, 1])
        closed_h = np.append(h, h[0])
        ax.plot(closed_v, closed_u, closed_h, "-", color=col, lw=3, label=lbl)

    ax.set_xlabel("V"); ax.set_ylabel("U"); ax.set_zlabel("response (IFF)")
    ax.set_title(f"{g['session_id']} — 3D response surface + boundaries\n"
                 "(does the green ridge cling to the flank instead of the base?)")
    ax.legend()
    fig.tight_layout()
    _emit(fig, out_dir, "fig4_surface_3d.png")


def fig5_forearm_3d(g, grad_contour_uv, infl_contour_uv, out_dir):
    """3D forearm mesh coloured by heatmap with the contour back-projected."""
    forearm_uv = g["forearm_uv"]
    faces = g["forearm_faces"]
    V = g["forearm_V"]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Light wireframe-ish surface via trisurf of the forearm mesh.
    ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=faces,
                    color="lightgrey", alpha=0.35, linewidth=0, shade=True)

    for c_uv, col, lbl in ((grad_contour_uv, "lime", "gradient ridge"),
                           (infl_contour_uv, "red", "inflection")):
        if c_uv is None:
            continue
        xyz = uv_points_to_xyz(c_uv, forearm_uv, faces, V)
        xyz = np.vstack([xyz, xyz[:1]])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-", color=col, lw=3, label=lbl)

    ax.set_title(f"{g['session_id']} — boundaries back-projected on the forearm")
    ax.legend()
    try:
        ax.set_box_aspect((np.ptp(V[:, 0]), np.ptp(V[:, 1]), np.ptp(V[:, 2])))
    except Exception:
        pass
    fig.tight_layout()
    _emit(fig, out_dir, "fig5_forearm_3d.png")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # ----------------------- HARDCODED INPUT (edit me) -----------------------
    ST13_03 = "2022-06-14_ST13-03/2022-06-14_ST13-03_population_response_fields.npz"
    ST14_01 = "2022-06-15_ST14-01/2022-06-15_ST14-01_population_response_fields.npz"
    ST14_02 = "2022-06-15_ST14-02/2022-06-15_ST14-02_population_response_fields.npz"
    ST14_04 = "2022-06-15_ST14-04/2022-06-15_ST14-04_population_response_fields.npz"

    NPZ_PATH = Path(
        "F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/"
        "semi-controlled/4_analysed/spatial_extract_boundaries/iff_mean/" +
        ST14_04
    )    
    
    
    GESTURE = "all"        # one of: all, stroke, tap, stroke_proximal, stroke_distal

    # ----------------------- TUNABLE METHOD PARAMETERS -----------------------
    GAUSSIAN_SIGMA = 3.0   # smoothing before gradient (pipeline used inflection_sigma)
    N_ANGLES = 360         # radial rays
    SAVGOL_WINDOW = 31     # contour smoothing window (None to disable)
    SIGMA_SWEEP = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    DRAW_FOREARM_3D = False  # Fig 5 (back-projection); set False to skip if slow

    out_dir = Path(__file__).resolve().parent / "_sandbox_gradient_ridge_out"
    out_dir.mkdir(exist_ok=True)

    # Secure an interactive backend now (after the analysis imports forced Agg).
    global _INTERACTIVE
    if USE_PLT_SHOW:
        _INTERACTIVE = _ensure_interactive_backend()

    print(f"Loading {NPZ_PATH.name}  (gesture='{GESTURE}')")
    g = load_grid(NPZ_PATH, GESTURE)
    print(f"  session={g['session_id']}  pipeline sigma={g['pipeline_sigma']}  "
          f"pipeline method={g['pipeline_boundary_method']}")
    print(f"  available gestures: {g['gesture_types']}")
    grid_u, grid_v, grid_z = g["grid_u"], g["grid_v"], g["grid_z"]

    # --- Re-run the methods LIVE with the chosen parameters ------------------
    peak_rc = find_peak_location(grid_z)
    if peak_rc is None:
        raise RuntimeError("grid_z is all-NaN — nothing to analyse.")
    print(f"  peak at (row,col)={peak_rc}, peak value={np.nanmax(grid_z):.3f}")

    smoothed, laplacian = compute_laplacian_arrays(grid_z, GAUSSIAN_SIGMA)
    grad_mag = compute_gradient_magnitude(smoothed, np.isnan(grid_z))

    grad_contour_rc = _extract_ridge_via_radial_profiling(
        grad_mag, peak_rc, n_angles=N_ANGLES, savgol_window=SAVGOL_WINDOW,
    )
    grad_boundary = compute_gradient_ridge(
        grid_u, grid_v, grid_z, smoothed,
        n_angles=N_ANGLES, savgol_window=SAVGOL_WINDOW,
    )
    infl_boundary = compute_inflection_boundary(grid_u, grid_v, grid_z, GAUSSIAN_SIGMA)

    # Inflection contour in pixel space (from its UV contour) for overlays.
    infl_contour_rc = None
    if infl_boundary is not None:
        uv = infl_boundary.contour_uv
        n_rows, n_cols = grid_z.shape
        u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
        v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])
        rr = (uv[:, 0] - u_min) / (u_max - u_min) * (n_rows - 1)
        cc = (uv[:, 1] - v_min) / (v_max - v_min) * (n_cols - 1)
        infl_contour_rc = np.column_stack([rr, cc])

    print("Rendering figures...")
    fig1_pipeline_steps(g, smoothed, laplacian, grad_mag, peak_rc,
                        grad_contour_rc, infl_contour_rc, out_dir)
    fig2_radial_profiling(g, grad_mag, peak_rc, grad_contour_rc, N_ANGLES, out_dir)
    fig3_sigma_sweep(g, SIGMA_SWEEP, N_ANGLES, SAVGOL_WINDOW, out_dir)
    fig4_surface_3d(g, grad_contour_rc, infl_contour_rc, out_dir)
    if DRAW_FOREARM_3D:
        fig5_forearm_3d(
            g,
            grad_boundary.contour_uv if grad_boundary is not None else None,
            infl_boundary.contour_uv if infl_boundary is not None else None,
            out_dir,
        )

    print(f"Done. Figures in: {out_dir}")
    if _INTERACTIVE:
        _wire_navigation()
        print("Interactive: Left/Right (or n/p) to navigate between figures, q to close all.")
        plt.show()


if __name__ == "__main__":
    main()

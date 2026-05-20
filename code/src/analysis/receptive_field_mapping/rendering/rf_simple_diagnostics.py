"""Step-by-step diagnostic figures for the RF simple pipeline.

Each function takes intermediate numpy/pandas data produced at a specific
pipeline step and returns a matplotlib Figure.  The orchestrator ``run_diagnostics``
calls all four and handles save/show/close.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from analysis.receptive_field_mapping.data.touch_population_data import PopulationData

_BG = "#1a1a1a"
_CMAP = "RdYlBu_r"


def _apply_dark_style(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(_BG)
    for ax in np.asarray(axes).flat:
        ax.set_facecolor(_BG)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")


# ---------------------------------------------------------------------------
# Step 1 — Population Data
# ---------------------------------------------------------------------------

def diagnose_population_data(
    population_data: PopulationData,
    spike_mask: np.ndarray,
    rotation_matrix: np.ndarray = None,
) -> plt.Figure:
    """1×3 diagnostic figure for the loaded PopulationData.

    Parameters
    ----------
    population_data:
        Loaded ``PopulationData`` dataclass.
    spike_mask:
        Boolean array of length T (one entry per touch) — True for spike touches.
    rotation_matrix:
        (3, 3) camera rotation from ``camera_settings_to_rotation()``.  When provided,
        the scatter view matches the RF camera settings viewer orientation.
    """
    fv = population_data.forearm_vertices       # (V, 3)
    cp_idx = population_data.cp_vertex_idx      # (C,) — vertex index per contact pt
    cp_touch = population_data.cp_touch_idx     # (C,) — owning touch index
    gesture_types = population_data.gesture_types  # (T,)

    # cp_idx can have millions of entries (one per contact-point-frame at 1 kHz).
    # Subsample the index arrays *before* fancy-indexing into fv to avoid allocating
    # a (C, 3) array that can be hundreds of MB.
    _MAX_SCATTER = 20_000

    _all_step = max(1, len(cp_idx) // _MAX_SCATTER)
    all_contact_plot = fv[cp_idx[::_all_step]]   # (≤20k, 3)

    spike_touch_indices = np.where(spike_mask)[0]
    spike_cp_idx = cp_idx[np.isin(cp_touch, spike_touch_indices)]
    _spike_step = max(1, len(spike_cp_idx) // _MAX_SCATTER)
    spike_contact_plot = fv[spike_cp_idx[::_spike_step]]  # (≤20k, 3)

    if rotation_matrix is not None:
        all_contact_plot = all_contact_plot @ rotation_matrix.T
        spike_contact_plot = spike_contact_plot @ rotation_matrix.T

    T = len(gesture_types)
    n_spike_touches = int(spike_mask.sum())
    unique_spike_vtx = int(np.unique(spike_cp_idx).shape[0]) if len(spike_cp_idx) > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _apply_dark_style(fig, axes)

    # --- left: camera view ---
    ax = axes[0]
    ax.scatter(all_contact_plot[:, 0], all_contact_plot[:, 1],
               c="steelblue", alpha=0.3, s=2, linewidths=0, label="all contacts")
    if len(spike_contact_plot) > 0:
        ax.scatter(spike_contact_plot[:, 0], spike_contact_plot[:, 1],
                   c="red", alpha=0.7, s=4, linewidths=0, label="spike contacts")
    ax.set_xlabel("horiz. (mm)", color="white")
    ax.set_ylabel("vert. (mm)", color="white")
    ax.set_title("Camera view — all contacts + spike contacts", color="white")
    ax.set_aspect("equal")
    ax.legend(facecolor="#2a2a2a", labelcolor="white", markerscale=2)

    # --- center: histogram of contact points per touch ---
    ax = axes[1]
    cp_counts_per_touch = np.bincount(cp_touch, minlength=T)
    spike_counts_per_touch = cp_counts_per_touch[spike_mask]
    ax.hist(cp_counts_per_touch, bins=50, color="steelblue", alpha=0.7, label="all touches")
    if len(spike_counts_per_touch) > 0:
        ax.hist(spike_counts_per_touch, bins=50, color="red", alpha=0.7, label="spike touches")
    ax.set_xlabel("Contact points per touch", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Contact points per touch", color="white")
    ax.legend(facecolor="#2a2a2a", labelcolor="white")

    # --- right: summary text ---
    ax = axes[2]
    ax.axis("off")
    unique_gestures, gesture_counts = np.unique(gesture_types, return_counts=True)
    gesture_lines = "\n".join(
        f"  {g}: {c}" for g, c in zip(unique_gestures, gesture_counts)
    )
    text_lines = [
        f"Total touches (T):        {T}",
        f"Total contact pts (C):    {len(cp_idx)}",
        f"Spike touches:            {n_spike_touches}",
        f"Unique spike vertices:    {unique_spike_vtx}",
        f"Forearm vertices (V):     {len(fv)}",
        "",
        "Gesture breakdown:",
        gesture_lines,
    ]
    ax.text(
        0.05, 0.95, "\n".join(text_lines),
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10, color="white",
        fontfamily="monospace",
    )
    ax.set_title("Summary statistics", color="white")

    fig.suptitle("Step 1 — Population Data", color="white", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Step 2 — Spike Extraction
# ---------------------------------------------------------------------------

def diagnose_spike_extraction(
    forearm_vertices: np.ndarray,
    spike_vertex_indices: np.ndarray,
    rotation_matrix: np.ndarray = None,
) -> plt.Figure:
    """1×2 diagnostic figure for spike vertex extraction.

    Parameters
    ----------
    forearm_vertices:
        (V, 3) forearm mesh vertices.
    spike_vertex_indices:
        1-D integer array of vertex indices — may contain repeats.
    rotation_matrix:
        (3, 3) camera rotation.  When provided, the scatter view matches the
        RF camera settings viewer orientation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _apply_dark_style(fig, axes)

    if len(spike_vertex_indices) > 0:
        occurrence_counts = np.bincount(spike_vertex_indices, minlength=len(forearm_vertices))
        unique_vtx = np.unique(spike_vertex_indices)
        spike_xyz = forearm_vertices[unique_vtx]
        vtx_occ = occurrence_counts[unique_vtx]
        sizes = 10 + vtx_occ * 2.0
    else:
        spike_xyz = np.empty((0, 3))
        vtx_occ = np.empty(0)
        sizes = np.empty(0)

    _fv_step = max(1, len(forearm_vertices) // 20_000)
    fv_plot = forearm_vertices[::_fv_step]

    if rotation_matrix is not None:
        fv_plot = fv_plot @ rotation_matrix.T
        if len(spike_xyz) > 0:
            spike_xyz = spike_xyz @ rotation_matrix.T

    # --- left: camera view ---
    ax = axes[0]
    ax.scatter(fv_plot[:, 0], fv_plot[:, 1],
               c="#555555", alpha=0.2, s=1, linewidths=0, label="forearm")
    if len(spike_xyz) > 0:
        ax.scatter(spike_xyz[:, 0], spike_xyz[:, 1],
                   c="red", alpha=0.8, s=sizes, linewidths=0, label="spike vertices")
    ax.set_xlabel("horiz. (mm)", color="white")
    ax.set_ylabel("vert. (mm)", color="white")
    ax.set_title("Camera view — spike vertices (sized by occurrence)", color="white")
    ax.set_aspect("equal")
    ax.legend(facecolor="#2a2a2a", labelcolor="white", markerscale=1.5)

    # --- right: vertex reuse histogram ---
    ax = axes[1]
    if len(spike_vertex_indices) > 0:
        reuse_counts = occurrence_counts[occurrence_counts > 0]
        ax.hist(reuse_counts, bins=30, color="steelblue", alpha=0.8)
        ax.set_xlabel("Times each vertex appears", color="white")
        ax.set_ylabel("Number of vertices", color="white")
        ax.set_title("Spike vertex reuse distribution", color="white")
    else:
        ax.text(0.5, 0.5, "No spike vertices", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=12)
        ax.set_title("Spike vertex reuse distribution", color="white")

    fig.suptitle("Step 2 — Spike Extraction", color="white", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Step 3 — Aggregation
# ---------------------------------------------------------------------------

def diagnose_aggregation(
    spike_counts_df: pd.DataFrame,
    forearm_vertices: np.ndarray,
    spike_xyz: np.ndarray,
    neuron_contacts_xyz: np.ndarray,
    rotation_matrix: np.ndarray = None,
) -> plt.Figure:
    """1×3 diagnostic figure for spike count aggregation.

    Parameters
    ----------
    spike_counts_df:
        DataFrame with columns x, y, z, spike_count (one row per unique spike vertex).
    forearm_vertices:
        (V, 3) forearm mesh vertices.
    spike_xyz:
        (N_spikes, 3) XYZ of all spike contact occurrences (with repeats).
    neuron_contacts_xyz:
        (M, 3) XYZ of all contact vertices used to compute projection centroid.
    rotation_matrix:
        (3, 3) camera rotation.  When provided, the scatter view matches the
        RF camera settings viewer orientation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _apply_dark_style(fig, axes)

    has_spikes = len(spike_counts_df) > 0

    sc = spike_counts_df["spike_count"].values if has_spikes else np.array([1.0])
    norm = mcolors.LogNorm(vmin=max(sc.min(), 1), vmax=sc.max()) if has_spikes else None

    if has_spikes and rotation_matrix is not None:
        spike_pos = spike_counts_df[["x", "y", "z"]].to_numpy() @ rotation_matrix.T
    elif has_spikes:
        spike_pos = spike_counts_df[["x", "y", "z"]].to_numpy()
    else:
        spike_pos = np.empty((0, 3))

    # --- left: camera view colored by spike_count ---
    ax = axes[0]
    if has_spikes:
        sc_plot = ax.scatter(
            spike_pos[:, 0], spike_pos[:, 1],
            c=spike_counts_df["spike_count"], cmap=_CMAP, norm=norm,
            s=8, alpha=0.9, linewidths=0,
        )
        plt.colorbar(sc_plot, ax=ax, label="spike_count").ax.yaxis.label.set_color("white")
    else:
        ax.text(0.5, 0.5, "No spikes", transform=ax.transAxes,
                ha="center", va="center", color="white")
    ax.set_xlabel("horiz. (mm)", color="white")
    ax.set_ylabel("vert. (mm)", color="white")
    ax.set_title("Camera view — spike count per vertex", color="white")
    ax.set_aspect("equal")

    # --- center: centroid comparison ---
    _fv_step = max(1, len(forearm_vertices) // 20_000)
    fv_sub = forearm_vertices[::_fv_step]
    if rotation_matrix is not None:
        fv_sub = fv_sub @ rotation_matrix.T
    ax = axes[1]
    ax.scatter(fv_sub[:, 0], fv_sub[:, 1],
               c="#555555", alpha=0.15, s=1, linewidths=0, label="forearm")

    proj_centroid = neuron_contacts_xyz.mean(axis=0) if len(neuron_contacts_xyz) > 0 else np.zeros(3)
    spike_centroid = spike_xyz.mean(axis=0) if len(spike_xyz) > 0 else np.zeros(3)
    mesh_centroid = forearm_vertices.mean(axis=0)
    if rotation_matrix is not None:
        proj_centroid = proj_centroid @ rotation_matrix.T
        spike_centroid = spike_centroid @ rotation_matrix.T
        mesh_centroid = mesh_centroid @ rotation_matrix.T

    ax.plot(proj_centroid[0], proj_centroid[1], "r*", markersize=14, label="proj centroid")
    ax.plot(spike_centroid[0], spike_centroid[1], "g*", markersize=14, label="spike centroid")
    ax.plot(mesh_centroid[0], mesh_centroid[1], "b*", markersize=14, label="mesh centroid")

    d_proj_spike = float(np.linalg.norm(proj_centroid - spike_centroid))
    d_proj_mesh = float(np.linalg.norm(proj_centroid - mesh_centroid))
    d_spike_mesh = float(np.linalg.norm(spike_centroid - mesh_centroid))

    annotation = (
        f"proj↔spike: {d_proj_spike:.1f} mm\n"
        f"proj↔mesh:  {d_proj_mesh:.1f} mm\n"
        f"spike↔mesh: {d_spike_mesh:.1f} mm"
    )
    ax.text(
        0.02, 0.98, annotation,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9, color="white",
        fontfamily="monospace",
        bbox=dict(facecolor="#2a2a2a", alpha=0.7, edgecolor="none"),
    )

    ax.set_xlabel("horiz. (mm)", color="white")
    ax.set_ylabel("vert. (mm)", color="white")
    ax.set_title("Camera view — centroid comparison", color="white")
    ax.set_aspect("equal")
    ax.legend(facecolor="#2a2a2a", labelcolor="white", markerscale=0.8)

    # --- right: spike_count distribution ---
    ax = axes[2]
    if has_spikes:
        ax.hist(spike_counts_df["spike_count"], bins=30, color="steelblue", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("spike_count (log scale)", color="white")
        ax.set_ylabel("Number of vertices", color="white")
        ax.set_title("Spike count distribution", color="white")
    else:
        ax.text(0.5, 0.5, "No spikes", transform=ax.transAxes,
                ha="center", va="center", color="white")
        ax.set_title("Spike count distribution", color="white")

    fig.suptitle("Step 3 — Aggregation", color="white", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Step 4 — 2D Projection
# ---------------------------------------------------------------------------

def diagnose_projection(
    forearm_vertices: np.ndarray,
    spike_counts_df: pd.DataFrame,
    uv_spikes: np.ndarray,
    projection_metadata: dict,
) -> plt.Figure:
    """1×3 diagnostic figure for the 2D cylindrical projection (spike vertices only).

    Parameters
    ----------
    forearm_vertices:
        (V, 3) forearm mesh vertices (used for camera-view background in panel 1).
    spike_counts_df:
        DataFrame with columns x, y, z, spike_count.
    uv_spikes:
        (K, 2) UV coordinates for spike vertices — centred on the spike centroid.
    projection_metadata:
        Dict with keys: centroid, rotation_matrix, axis_direction, mean_radius.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _apply_dark_style(fig, axes)

    has_spikes = len(spike_counts_df) > 0
    sc = spike_counts_df["spike_count"].values if has_spikes else np.array([1.0])
    norm = mcolors.LogNorm(vmin=max(sc.min(), 1), vmax=sc.max()) if has_spikes else None

    R = projection_metadata.get("rotation_matrix")
    if R is not None and np.allclose(R, np.eye(3)):
        R = None

    _fv_step = max(1, len(forearm_vertices) // 20_000)
    fv_plot = forearm_vertices[::_fv_step]
    if has_spikes:
        spike_pos = spike_counts_df[["x", "y", "z"]].to_numpy()
    else:
        spike_pos = np.empty((0, 3))

    if R is not None:
        fv_plot = fv_plot @ R.T
        if has_spikes:
            spike_pos = spike_pos @ R.T

    # --- panel 1: camera view — forearm + spikes (reference, matches steps 1–3) ---
    ax = axes[0]
    ax.scatter(fv_plot[:, 0], fv_plot[:, 1],
               c="#555555", alpha=0.2, s=1, linewidths=0, label="forearm (subsampled)")
    if has_spikes:
        sp = ax.scatter(
            spike_pos[:, 0], spike_pos[:, 1],
            c=spike_counts_df["spike_count"], cmap=_CMAP, norm=norm,
            s=6, alpha=0.9, linewidths=0, label="spike vertices",
        )
        plt.colorbar(sp, ax=ax, label="spike_count").ax.yaxis.label.set_color("white")
    ax.set_xlabel("horiz. (mm)", color="white")
    ax.set_ylabel("vert. (mm)", color="white")
    ax.set_title("Camera view — forearm + spikes", color="white")
    ax.set_aspect("equal")
    ax.legend(facecolor="#2a2a2a", labelcolor="white", markerscale=2)

    # --- panel 2: UV — spike vertices only ---
    ax = axes[1]
    if has_spikes:
        sp = ax.scatter(
            uv_spikes[:, 0], uv_spikes[:, 1],
            c=spike_counts_df["spike_count"], cmap=_CMAP, norm=norm,
            s=10, alpha=0.9, linewidths=0,
        )
        plt.colorbar(sp, ax=ax, label="spike_count").ax.yaxis.label.set_color("white")
    else:
        ax.text(0.5, 0.5, "No spikes", transform=ax.transAxes,
                ha="center", va="center", color="white")
    ax.set_xlabel("u (mm)", color="white")
    ax.set_ylabel("v (mm)", color="white")
    ax.set_title("UV — spike vertices only", color="white")

    # --- panel 3: metadata text ---
    ax = axes[2]
    ax.axis("off")

    centroid = projection_metadata.get("centroid", np.zeros(3))
    R = projection_metadata.get("rotation_matrix", np.eye(3))
    axis_dir = projection_metadata.get("axis_direction", np.zeros(3))
    mean_radius = projection_metadata.get("mean_radius", float("nan"))

    def _fmt_vec(v):
        return f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"

    text_lines = [
        f"Method:        cylindrical_unwrap",
        f"Spike centroid: {_fmt_vec(centroid)}",
        "",
        "Rotation matrix (R):",
        f"  R[0]: {_fmt_vec(R[0])}",
        f"  R[1]: {_fmt_vec(R[1])}",
        f"  R[2]: {_fmt_vec(R[2])}",
        "",
        f"Axis direction: {_fmt_vec(axis_dir)}",
        f"Mean radius:    {mean_radius:.2f} mm",
        "",
        f"Forearm verts:  {len(forearm_vertices)}",
        f"Spike verts:    {len(uv_spikes)}",
    ]

    ax.text(
        0.05, 0.95, "\n".join(text_lines),
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9, color="white",
        fontfamily="monospace",
    )
    ax.set_title("Projection metadata", color="white")

    fig.suptitle("Step 4 — Camera view + UV projection (spikes)", color="white", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_diagnostics(
    population_data: PopulationData,
    spike_mask: np.ndarray,
    forearm_vertices: np.ndarray,
    spike_vertex_indices: np.ndarray,
    spike_counts_df: pd.DataFrame,
    spike_xyz: np.ndarray,
    neuron_contacts_xyz: np.ndarray,
    uv_spikes: np.ndarray,
    projection_metadata: dict,
    output_dir: Path,
    save: bool = True,
    show: bool = False,
) -> None:
    """Generate and optionally save/show all 4 diagnostic figures.

    Parameters
    ----------
    population_data:
        Loaded ``PopulationData`` from Step 1.
    spike_mask:
        Boolean array of length T — True for spike touches.
    forearm_vertices:
        (V, 3) forearm mesh vertices.
    spike_vertex_indices:
        1-D integer array of vertex indices with repeats (Step 2 output).
    spike_counts_df:
        Aggregated per-vertex spike count DataFrame (Step 3 output).
    spike_xyz:
        (N_spikes, 3) XYZ of all spike contacts with repeats.
    neuron_contacts_xyz:
        (M, 3) all contact vertices (used for step 3 centroid comparison).
    uv_spikes:
        (K, 2) UV for spike vertices — centred on the spike centroid.
    projection_metadata:
        Dict with keys: centroid, rotation_matrix, axis_direction, mean_radius.
    output_dir:
        Session output directory.  Figures are saved under ``output_dir / "diagnostics"``.
    save:
        If True, create the ``diagnostics/`` subfolder and save PNGs.
    show:
        If True, display each figure interactively.  When the matplotlib backend
        is interactive (non-Agg), calls ``plt.show()``.  When the backend is Agg
        (e.g. set by the heatmap renderer), opens the saved PNG with the OS viewer
        instead (requires ``save=True``).
    """
    diag_dir = output_dir / "diagnostics"
    if save:
        diag_dir.mkdir(parents=True, exist_ok=True)

    R = projection_metadata.get("rotation_matrix")
    if R is not None and np.allclose(R, np.eye(3)):
        R = None  # identity — no rotation to apply

    steps = [
        (
            "step1_population_data.png",
            lambda: diagnose_population_data(population_data, spike_mask, R),
        ),
        (
            "step2_spike_extraction.png",
            lambda: diagnose_spike_extraction(forearm_vertices, spike_vertex_indices, R),
        ),
        (
            "step3_aggregation.png",
            lambda: diagnose_aggregation(
                spike_counts_df, forearm_vertices, spike_xyz, neuron_contacts_xyz, R
            ),
        ),
        (
            "step4_projection.png",
            lambda: diagnose_projection(
                forearm_vertices, spike_counts_df, uv_spikes, projection_metadata
            ),
        ),
    ]

    _backend_is_interactive = matplotlib.get_backend().lower() != 'agg'

    for filename, build_fig in steps:
        print(f"  [diag] {filename}...", flush=True)
        fig = build_fig()
        saved_path = None
        if save:
            saved_path = diag_dir / filename
            fig.savefig(saved_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        if show:
            if _backend_is_interactive:
                plt.show()
            elif saved_path is not None:
                os.startfile(str(saved_path))
        plt.close(fig)

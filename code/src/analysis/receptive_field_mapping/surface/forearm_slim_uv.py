"""
forearm_slim_uv.py
==================

Per-session SLIM UV precompute and cache I/O for the production RF mapping
pipeline.

The two public functions are:

precompute_forearm_slim_uv
    Build a SLIM UV map for a single session and write it to a ``.npz`` cache
    next to the forearm PLY.  Raises loudly on any failure — no silent
    fallbacks.

load_slim_uv_cache
    Load the ``.npz`` cache and optionally verify that the source PLY and
    single-touch RF maps NPZ have not changed since the cache was written.

The cached data is described by :class:`SlimUvCache`.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertices,
    load_forearm_vertex_colors,
)
from .rf_surface_utils import load_or_build_forearm_mesh, _VALID_MESH_METHODS
from .slim_helpers import clean_mesh, boundary_loop, flatten_slim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache schema
# ---------------------------------------------------------------------------

@dataclass
class SlimUvCache:
    V: np.ndarray          # (N_v, 3) float64 — cleaned mesh vertices
    F: np.ndarray          # (N_f, 3) int32   — face indices
    uv: np.ndarray         # (N_v, 2) float64 — SLIM UV per vertex
    center_vid: int        # interior vertex at UV origin
    boundary_vid: int      # boundary anchor vertex (boundary[0])
    ply_mtime: float       # PLY mtime at cache-write time
    ply_hash: str          # SHA-256 of first 4 kB of PLY
    rf_npz_mtime: float    # single-touch RF maps NPZ mtime at cache-write time
    centroid_3d: np.ndarray  # (3,) float64 — IFF-weighted centroid
    mesh_method: str = "bpa"  # mesh construction method used ("bpa" or "delaunay")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ply_hash(ply_path: Path) -> str:
    """SHA-256 hex digest of the first 4 kB of the PLY file."""
    with open(ply_path, 'rb') as f:
        data = f.read(4096)
    return hashlib.sha256(data).hexdigest()


def _transfer_ply_colors(
    ply_vertices: np.ndarray,
    ply_colors_uint8: np.ndarray,
    mesh_vertices: np.ndarray,
) -> np.ndarray:
    """Map PLY RGB colours to mesh vertices via KDTree nearest-neighbour.

    Returns (N_mesh, 4) float64 RGBA in [0, 1] suitable for matplotlib.
    """
    tree = KDTree(ply_vertices)
    _, indices = tree.query(mesh_vertices)
    rgb = ply_colors_uint8[indices].astype(np.float64) / 255.0
    return np.column_stack([rgb, np.ones(len(rgb), dtype=np.float64)])


# ---------------------------------------------------------------------------
# Interactive step-by-step viewer
# ---------------------------------------------------------------------------

def _launch_slim_steps_viewer(
    *,
    session_id: str,
    V_raw: np.ndarray,
    F_raw: np.ndarray,
    clean_diag: dict,
    V_clean: np.ndarray,
    F_clean: np.ndarray,
    centroid_3d: np.ndarray,
    bloop_pre: np.ndarray,
    slim_diag: dict,
    V_final: np.ndarray | None,
    F_final: np.ndarray | None,
    uv_final: np.ndarray | None,
    raw_mesh_colors: np.ndarray | None,
    clean_mesh_colors: np.ndarray | None,
    mesh_method: str = "bpa",
) -> None:
    """Build the step list and open a blocking SlimUvStepsViewer.

    When ``slim_diag`` contains ``tutte_fail_uv`` (flatten_slim raised), the
    viewer is built in *failure* mode: cleaning + boundary steps as usual,
    followed by flipped-triangle diagnostics on the failing Tutte init. In
    that case ``V_final``/``F_final``/``uv_final`` are ignored and may be
    ``None``.
    """
    import sys
    from PyQt5.QtWidgets import QApplication
    from analysis.receptive_field_mapping.gui.slim_uv_steps_viewer import (
        SlimStep,
        SlimUvStepsViewer,
    )
    from .slim_helpers import compute_face_distortion, _identify_flipped_triangles

    failure_mode = "tutte_fail_uv" in slim_diag

    steps: list[SlimStep] = []

    # 1. Raw mesh.
    steps.append(SlimStep(
        label=f"Raw mesh ({mesh_method.upper()})",
        info=f"{V_raw.shape[0]} verts, {F_raw.shape[0]} faces",
        V=V_raw,
        F=F_raw,
        vertex_colors=raw_mesh_colors,
    ))

    # 2-8. Clean sub-steps from clean_diag.
    for entry in clean_diag["clean_steps"]:
        if len(entry) == 5:
            V_step, F_step, label, info, overlay = entry
        else:
            V_step, F_step, label, info = entry
            overlay = None
        # Transfer skin colours to this snapshot's vertices via nearest-neighbour.
        if clean_mesh_colors is not None:
            tree = KDTree(V_clean)
            _, idx = tree.query(V_step)
            step_colors = clean_mesh_colors[idx]
        else:
            step_colors = None
        overlay_points = dict(overlay) if overlay else {}
        overlay_colors = {k: "red" for k in overlay_points}
        steps.append(SlimStep(
            label=label,
            info=info,
            V=V_step,
            F=F_step,
            vertex_colors=step_colors,
            overlay_points=overlay_points,
            overlay_colors=overlay_colors,
        ))

    # 9. Centroid + boundary overlay on cleaned mesh.
    boundary_points = V_clean[bloop_pre]
    steps.append(SlimStep(
        label="Centroid + boundary overlay",
        info=(
            f"centroid_3d=({centroid_3d[0]:.1f}, {centroid_3d[1]:.1f}, "
            f"{centroid_3d[2]:.1f}); boundary={len(bloop_pre)} verts"
        ),
        V=V_clean,
        F=F_clean,
        vertex_colors=clean_mesh_colors,
        overlay_points={
            "boundary": boundary_points,
            "centroid": centroid_3d.reshape(1, 3),
        },
        overlay_colors={
            "boundary": "gold",
            "centroid": "red",
        },
    ))

    if failure_mode:
        # 10-11: Failure-diagnostic steps. Show which triangles flipped during
        # the Tutte init that exceeded the trim-retry budget.
        V_fail = slim_diag["tutte_fail_V"]
        F_fail = slim_diag["tutte_fail_F"]
        uv_fail = slim_diag["tutte_fail_uv"]
        flipped_mask = np.asarray(slim_diag["tutte_fail_flipped"], dtype=bool)
        n_flipped = int(slim_diag.get("tutte_fail_n_flipped", int(flipped_mask.sum())))
        fail_round = int(slim_diag.get("tutte_fail_round", 0))
        face_scalars = flipped_mask.astype(np.float64)

        # 10. Failing Tutte init in UV space.
        steps.append(SlimStep(
            label=f"Tutte init (FAILED: {n_flipped} flipped, round {fail_round + 1})",
            info=(
                f"{V_fail.shape[0]} verts, {F_fail.shape[0]} faces; "
                f"red = flipped, white = valid"
            ),
            V=uv_fail,
            F=F_fail,
            face_scalars=face_scalars,
            cmap="Reds",
            clim=(0.0, 1.0),
            scalar_bar_title="flipped",
        ))

        # 11. Same flipped faces shown on the 3D mesh — locates the anatomy
        # responsible for the failure (e.g. thumb junction, BPA seam).
        steps.append(SlimStep(
            label=f"Flipped triangles in 3D ({n_flipped} face(s))",
            info=(
                f"{V_fail.shape[0]} verts, {F_fail.shape[0]} faces; "
                f"red = flipped in UV"
            ),
            V=V_fail,
            F=F_fail,
            face_scalars=face_scalars,
            cmap="Reds",
            clim=(0.0, 1.0),
            scalar_bar_title="flipped",
        ))
    else:
        # 10-13: UV-space steps.  Render UV as 3D with z=0.
        init_uv = slim_diag["init_uv"]
        V_trim = slim_diag["V_trimmed"]
        F_trim = slim_diag["F_trimmed"]
        init_method = slim_diag.get("init_method", "?")
        trimmed = slim_diag.get("trimmed", False)
        trim_label = " (mesh trimmed)" if trimmed else ""

        # Colours for the (possibly trimmed) mesh used by UV-init step.
        if clean_mesh_colors is not None and trimmed:
            tree = KDTree(V_clean)
            _, idx = tree.query(V_trim)
            trim_colors = clean_mesh_colors[idx]
        else:
            trim_colors = clean_mesh_colors

        # 10. UV initialisation.
        steps.append(SlimStep(
            label=f"UV initialisation ({init_method}){trim_label}",
            info=f"{V_trim.shape[0]} verts, {F_trim.shape[0]} faces",
            V=init_uv,
            F=F_trim,
            vertex_colors=trim_colors,
        ))

        # 11. SLIM optimised UV — overlay residual flipped triangles (should
        # be zero on a clean run; visible red flecks indicate SLIM did not
        # fully repair what the init left behind).
        if clean_mesh_colors is not None:
            tree = KDTree(V_clean)
            _, idx = tree.query(V_final)
            final_colors = clean_mesh_colors[idx]
        else:
            final_colors = None
        final_flipped = _identify_flipped_triangles(uv_final, F_final)
        n_final_flipped = int(final_flipped.sum())
        if n_final_flipped > 0:
            steps.append(SlimStep(
                label=f"SLIM optimised UV ({n_final_flipped} flipped)",
                info=(
                    f"{V_final.shape[0]} verts, {F_final.shape[0]} faces; "
                    f"red = flipped"
                ),
                V=uv_final,
                F=F_final,
                face_scalars=final_flipped.astype(np.float64),
                cmap="Reds",
                clim=(0.0, 1.0),
                scalar_bar_title="flipped",
            ))
        else:
            steps.append(SlimStep(
                label="SLIM optimised UV",
                info=f"{V_final.shape[0]} verts, {F_final.shape[0]} faces",
                V=uv_final,
                F=F_final,
                vertex_colors=final_colors,
            ))

        # 12-13. Distortion overlays.
        conformal, area = compute_face_distortion(V_final, F_final, uv_final)
        steps.append(SlimStep(
            label="Conformal distortion (σ_max / σ_min)",
            info=(
                f"median={float(np.median(conformal)):.3f}, "
                f"max={float(conformal.max()):.3f}"
            ),
            V=uv_final,
            F=F_final,
            face_scalars=conformal,
            cmap="YlOrRd",
            clim=(1.0, float(np.percentile(conformal, 99))),
            scalar_bar_title="σ_max/σ_min",
        ))
        steps.append(SlimStep(
            label="Area distortion (log2 ratio)",
            info=(
                f"abs-median={float(np.median(np.abs(area))):.3f}, "
                f"range=[{float(area.min()):.2f}, {float(area.max()):.2f}]"
            ),
            V=uv_final,
            F=F_final,
            face_scalars=area,
            cmap="RdBu_r",
            clim=(-float(np.abs(area).max()), float(np.abs(area).max())),
            scalar_bar_title="log2(area)",
        ))

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = SlimUvStepsViewer(session_id=session_id, steps=steps)
    viewer.show()
    app.exec_()


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------

def precompute_forearm_slim_uv(
    forearm_ply_path: Path,
    rf_maps_npz: Path,
    cache_path: Path | None = None,
    n_iter: int = 40,
    save_diagnostics: bool = False,
    camera_settings_dir: Path | None = None,
    interactive: bool = False,
    clean_steps: dict | None = None,
    mesh_method: str = "bpa",
    max_edge_mm: float | None = None,
) -> Path:
    """Build SLIM UV cache for a single session.  Raises on any failure.

    Parameters
    ----------
    forearm_ply_path:
        Path to the forearm point-cloud PLY file.
    rf_maps_npz:
        Path to the ``single_touch_rf_maps.npz`` produced by
        ``map_single_touch_rf``.  Its ``rf_data`` key must contain at least
        one touch with at least one contacted vertex carrying a nonzero IFF.
    cache_path:
        Destination path for the ``.npz`` cache.  Defaults to
        ``<forearm_ply_stem>_slim_uv.npz`` in the same directory.
    n_iter:
        Number of SLIM iterations (default 40).
    mesh_method:
        Mesh construction method: ``"bpa"`` (Ball Pivoting Algorithm, default)
        or ``"delaunay"`` (2.5D Delaunay triangulation).  Raises
        :class:`ValueError` for unknown values.
    max_edge_mm:
        Maximum edge length (mm) for Delaunay edge filtering.  Only used when
        ``mesh_method="delaunay"``.  When ``None``, defaults to
        ``3.0 * avg_nn`` (auto-computed from point density).

    Returns
    -------
    Path
        The path to the written ``.npz`` cache file.

    Raises
    ------
    ValueError
        If ``mesh_method`` is not a recognised value.
    ValueError
        If ``load_or_build_forearm_mesh`` returns ``None`` (degenerate PLY).
    FileNotFoundError
        If ``rf_maps_npz`` does not exist.
    ValueError
        If ``rf_maps_npz`` contains no touches (empty ``rf_data`` dict).
    ValueError
        If all aggregated per-vertex IFF values are zero (cannot compute
        weighted centroid).
    ValueError
        If the IFF-weighted centroid maps to a boundary vertex.
    RuntimeError
        If the cotangent-harmonic initialisation has flipped triangles
        (propagated from ``flatten_slim``).
    """
    # 1. Default cache path.
    if cache_path is None:
        cache_path = forearm_ply_path.with_name(
            forearm_ply_path.stem + "_slim_uv.npz"
        )

    # 1b. Validate mesh_method.
    if mesh_method not in _VALID_MESH_METHODS:
        raise ValueError(
            f"Unknown mesh_method {mesh_method!r}. "
            f"Valid options are: {_VALID_MESH_METHODS}"
        )

    # 2. Build mesh.
    raw_mesh = load_or_build_forearm_mesh(
        forearm_ply_path,
        mesh_method=mesh_method,
        max_edge_mm=max_edge_mm,
    )
    if raw_mesh is None:
        raise ValueError(
            f"load_or_build_forearm_mesh returned None for {forearm_ply_path}. "
            "The PLY may be empty, too sparse, or not a forearm segmentation."
        )
    V_raw = np.asarray(raw_mesh.vertices, dtype=np.float64)
    F_raw = np.asarray(raw_mesh.faces, dtype=np.int32)

    # 3. Clean mesh.
    clean_diag: dict | None = {"clean_steps": []} if interactive else None
    V, F = clean_mesh(raw_mesh, diagnostics=clean_diag, clean_steps=clean_steps)
    logger.info(
        "Cleaned mesh: %d vertices, %d faces (source: %s)",
        V.shape[0], F.shape[0], forearm_ply_path.name,
    )

    # 4. Load single-touch RF maps NPZ.
    if not rf_maps_npz.exists():
        raise FileNotFoundError(
            f"Single-touch RF maps NPZ not found: {rf_maps_npz}\n"
            "Run 'map_single_touch_rf' before 'precompute_forearm_slim_uv'."
        )

    npz = np.load(rf_maps_npz, allow_pickle=True)
    rf_data: dict = npz["rf_data"].item()

    if not rf_data:
        raise ValueError(
            f"No touches in {rf_maps_npz} — cannot determine forearm "
            "hotspot centroid."
        )

    # 5. Load raw PLY vertices to resolve NPZ vertex indices to 3D positions.
    raw_verts = load_forearm_vertices(forearm_ply_path)
    if raw_verts is None:
        raise ValueError(
            f"load_forearm_vertices returned None for {forearm_ply_path}. "
            "The PLY may be empty or unreadable."
        )
    n_verts = len(raw_verts)

    # 5b. Load PLY vertex colours and transfer to raw / cleaned mesh vertices.
    ply_colors_uint8 = load_forearm_vertex_colors(forearm_ply_path)
    if ply_colors_uint8 is not None:
        raw_mesh_colors = _transfer_ply_colors(raw_verts, ply_colors_uint8, V_raw)
        clean_mesh_colors = _transfer_ply_colors(raw_verts, ply_colors_uint8, V)
    else:
        raw_mesh_colors = None
        clean_mesh_colors = None

    # 6. Aggregate per-vertex mean IFF across all touches.
    #    Pattern mirrors touch_population_explorer.py:654-684.
    iff_sum = np.zeros(n_verts, dtype=np.float64)
    touch_count = np.zeros(n_verts, dtype=np.int64)

    for pairs in rf_data.values():
        for vertex_idx, mean_iff in pairs:
            idx = int(vertex_idx)
            if idx < 0 or idx >= n_verts:
                # Out-of-bounds index — skip silently (safety guard).
                continue
            iff_sum[idx] += float(mean_iff)
            touch_count[idx] += 1

    contacted_mask = touch_count > 0
    contacted_indices = np.where(contacted_mask)[0]

    if len(contacted_indices) == 0:
        raise ValueError(
            f"No contacted vertices found in {rf_maps_npz} — cannot determine "
            "forearm hotspot centroid."
        )

    per_vertex_mean_iff = iff_sum[contacted_indices] / touch_count[contacted_indices]

    # 7. Compute IFF-weighted 3D centroid.
    total_weight = per_vertex_mean_iff.sum()
    if total_weight == 0.0:
        raise ValueError(
            f"All aggregated per-vertex IFF values are zero in {rf_maps_npz}. "
            "Cannot compute IFF-weighted centroid — check that the neuron was "
            "responding during the recorded touches."
        )

    contacted_positions = raw_verts[contacted_indices]
    centroid_3d = np.average(contacted_positions, weights=per_vertex_mean_iff, axis=0)

    # 8. KDTree → nearest cleaned-mesh vertex.
    tree = KDTree(V)
    _, center_vid = tree.query(centroid_3d)
    center_vid = int(center_vid)

    # 9. Boundary loop.
    bloop = boundary_loop(F)

    # 10. Fail-fast if centroid maps to a boundary vertex.
    boundary_set = set(int(v) for v in bloop)
    if center_vid in boundary_set:
        raise ValueError(
            f"IFF-weighted centroid maps to mesh boundary vertex {center_vid}. "
            "The forearm mesh boundary does not cover the neuron hotspot — "
            "consider re-extracting the forearm PLY with a larger skin region."
        )

    # 11. SLIM flattening.
    logger.info(
        "Running SLIM (n_iter=%d, center_vid=%d) for %s ...",
        n_iter, center_vid, forearm_ply_path.name,
    )
    V_clean = V.copy()
    F_clean = F.copy()
    center_vid_pre = center_vid
    bloop_pre = bloop.copy()
    slim_diag: dict | None = {} if (save_diagnostics or interactive) else None
    try:
        V, F, uv = flatten_slim(V, F, bloop, center_vid=center_vid, n_iter=n_iter, diagnostics=slim_diag)
    except RuntimeError as exc:
        if interactive and slim_diag is not None and "tutte_fail_uv" in slim_diag:
            logger.warning(
                "flatten_slim failed; launching diagnostic viewer before re-raising: %s",
                exc,
            )
            _launch_slim_steps_viewer(
                session_id=cache_path.parent.name,
                V_raw=V_raw,
                F_raw=F_raw,
                clean_diag=clean_diag,
                V_clean=V_clean,
                F_clean=F_clean,
                centroid_3d=centroid_3d,
                bloop_pre=bloop_pre,
                slim_diag=slim_diag,
                V_final=None,
                F_final=None,
                uv_final=None,
                raw_mesh_colors=raw_mesh_colors,
                clean_mesh_colors=clean_mesh_colors,
                mesh_method=mesh_method,
            )
        raise

    # 12. Re-derive center_vid and boundary after potential mesh trimming inside
    #     flatten_slim.  In the common (no-trim) case these are unchanged.
    _, center_vid = KDTree(V).query(centroid_3d)
    center_vid = int(center_vid)
    bloop = boundary_loop(F)

    # 12b. Recompute colours for the (possibly trimmed) final mesh.
    if ply_colors_uint8 is not None:
        final_mesh_colors = _transfer_ply_colors(raw_verts, ply_colors_uint8, V)
    else:
        final_mesh_colors = None

    # 13. Collect provenance.
    ply_mtime = forearm_ply_path.stat().st_mtime
    rf_npz_mtime = rf_maps_npz.stat().st_mtime
    phash = _ply_hash(forearm_ply_path)

    # 14. Write cache.
    np.savez(
        cache_path,
        V=V.astype(np.float64),
        F=F.astype(np.int32),
        uv=uv.astype(np.float64),
        center_vid=np.int32(center_vid),
        boundary_vid=np.int32(int(bloop[0])),
        ply_mtime=np.float64(ply_mtime),
        ply_hash=np.array(phash, dtype='U64'),
        rf_npz_mtime=np.float64(rf_npz_mtime),
        centroid_3d=centroid_3d.astype(np.float64),
        mesh_method=np.array(mesh_method, dtype='U16'),
    )

    logger.info("SLIM UV cache written → %s", cache_path)

    # 15. Load RF camera settings (used by both QC and diagnostic figures).
    out_dir = cache_path.parent
    session_id = out_dir.name
    cam_settings: dict | None = None
    if camera_settings_dir is not None:
        settings_path = camera_settings_dir / "rf_camera_settings.json"
        try:
            with open(settings_path) as _f:
                all_cameras: dict = json.load(_f)
            cam_settings = all_cameras.get(session_id)
        except Exception:
            cam_settings = None

    # 16. Save QC figures (300 DPI) next to the cache for visual verification.
    from .slim_qc_figures import save_slim_qc_figures, save_slim_diagnostic_figures
    qc_path, dist_path = save_slim_qc_figures(V, F, uv, center_vid, cache_path,
                                              vertex_colors=final_mesh_colors,
                                              cam_settings=cam_settings)
    logger.info("QC figures written → %s, %s", qc_path, dist_path)

    # 17. Save step-by-step diagnostic figures when requested.
    if save_diagnostics:
        diag_paths = save_slim_diagnostic_figures(
            out_dir=out_dir,
            session_id=session_id,
            V_raw=V_raw,
            F_raw=F_raw,
            V=V_clean,
            F=F_clean,
            center_vid=center_vid_pre,
            boundary=bloop_pre,
            slim_diag=slim_diag,
            uv_final=uv,
            cam_settings=cam_settings,
            raw_mesh_colors=raw_mesh_colors,
            clean_mesh_colors=clean_mesh_colors,
        )
        logger.info("Saved %d diagnostic figures to %s", len(diag_paths), out_dir / "diagnostics")

    # 18. Launch interactive step-by-step viewer when requested.
    if interactive:
        _launch_slim_steps_viewer(
            session_id=session_id,
            V_raw=V_raw,
            F_raw=F_raw,
            clean_diag=clean_diag,
            V_clean=V_clean,
            F_clean=F_clean,
            centroid_3d=centroid_3d,
            bloop_pre=bloop_pre,
            slim_diag=slim_diag,
            V_final=V,
            F_final=F,
            uv_final=uv,
            raw_mesh_colors=raw_mesh_colors,
            clean_mesh_colors=clean_mesh_colors,
            mesh_method=mesh_method,
        )

    # 19. Return cache path.
    return cache_path


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_slim_uv_cache(
    cache_path: Path,
    *,
    forearm_ply_path: Path | None = None,
    rf_maps_npz: Path | None = None,
) -> SlimUvCache:
    """Load cached UV + mesh data.

    Parameters
    ----------
    cache_path:
        Path to the ``.npz`` cache file.
    forearm_ply_path:
        When provided, verify that the PLY mtime and hash match the cached
        values.  Raises :class:`RuntimeError` if stale.
    rf_maps_npz:
        When provided, verify that the single-touch RF maps NPZ mtime matches
        the cached value.  Raises :class:`RuntimeError` if stale.

    Returns
    -------
    SlimUvCache

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    RuntimeError
        If the cache was written with the old ``spike_csv_mtime`` schema
        (pre-IFF-weighted centroid).  Delete the cache and re-run
        ``precompute_forearm_slim_uv`` to rebuild it.
    RuntimeError
        If mtime/hash staleness check fails (inputs changed since cache).
    """
    # 1. Guard: cache must exist.
    if not cache_path.exists():
        raise FileNotFoundError(
            f"SLIM UV cache not found: {cache_path}\n"
            "Run 'precompute_forearm_slim_uv' first."
        )

    # 2. Load arrays.
    data = np.load(cache_path, allow_pickle=False)

    # 3. Old-cache guard: reject caches written before the IFF-weighted centroid
    #    migration.  A cache with spike_csv_mtime but without rf_npz_mtime was
    #    produced by the old CSV-based code.
    if "spike_csv_mtime" in data and "rf_npz_mtime" not in data:
        raise RuntimeError(
            f"Old-format SLIM UV cache detected at {cache_path}: contains "
            "'spike_csv_mtime' but not 'rf_npz_mtime'. "
            "Delete the cache and re-run 'precompute_forearm_slim_uv' to "
            "rebuild it with the IFF-weighted centroid schema."
        )

    # 4. Build dataclass.
    cache = SlimUvCache(
        V=data['V'],
        F=data['F'],
        uv=data['uv'],
        center_vid=int(data['center_vid']),
        boundary_vid=int(data['boundary_vid']),
        ply_mtime=float(data['ply_mtime']),
        ply_hash=str(data['ply_hash']),
        rf_npz_mtime=float(data['rf_npz_mtime']),
        centroid_3d=data['centroid_3d'],
        mesh_method=str(data.get('mesh_method', np.array("bpa"))),
    )

    # 5. Optional PLY staleness check.
    if forearm_ply_path is not None:
        current_mtime = forearm_ply_path.stat().st_mtime
        if current_mtime > cache.ply_mtime + 1e-3:
            raise RuntimeError(
                f"SLIM UV cache is stale: forearm PLY has been modified since the "
                f"cache was written.\n"
                f"  PLY mtime:   {current_mtime}\n"
                f"  Cache mtime: {cache.ply_mtime}\n"
                f"  Cache path:  {cache_path}\n"
                "Re-run 'precompute_forearm_slim_uv' to rebuild the cache."
            )
        current_hash = _ply_hash(forearm_ply_path)
        if current_hash != cache.ply_hash:
            raise RuntimeError(
                f"SLIM UV cache is stale: forearm PLY content has changed "
                f"(hash mismatch) since the cache was written.\n"
                f"  Current hash: {current_hash}\n"
                f"  Cached hash:  {cache.ply_hash}\n"
                f"  Cache path:   {cache_path}\n"
                "Re-run 'precompute_forearm_slim_uv' to rebuild the cache."
            )

    # 6. Optional RF maps NPZ staleness check.
    if rf_maps_npz is not None:
        current_mtime = rf_maps_npz.stat().st_mtime
        if current_mtime > cache.rf_npz_mtime + 1e-3:
            raise RuntimeError(
                f"SLIM UV cache is stale: single-touch RF maps NPZ has been "
                f"modified since the cache was written.\n"
                f"  NPZ mtime:   {current_mtime}\n"
                f"  Cache mtime: {cache.rf_npz_mtime}\n"
                f"  Cache path:  {cache_path}\n"
                "Re-run 'precompute_forearm_slim_uv' to rebuild the cache."
            )

    return cache


# ---------------------------------------------------------------------------
# UV → 3D world mapping
# ---------------------------------------------------------------------------

def uv_points_to_xyz(
    uv_points: np.ndarray,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
) -> np.ndarray:
    """Map UV points to 3D world (mm) via barycentric interpolation on the SLIM mesh.

    Uses matplotlib.tri.Triangulation + get_trifinder() to locate the containing
    triangle for each UV point, then interpolates forearm_V with barycentric weights.
    Points that fall epsilon-outside the mesh boundary (e.g. contour vertices from
    an interpolation grid that slightly overshoots the triangulation) are snapped to
    the nearest triangle boundary via clamped barycentric coords; a warning is logged.
    """
    import logging
    import matplotlib.tri as mtri

    uv_points = np.asarray(uv_points, dtype=np.float64)
    forearm_uv = np.asarray(forearm_uv, dtype=np.float64)
    forearm_V = np.asarray(forearm_V, dtype=np.float64)

    triang = mtri.Triangulation(forearm_uv[:, 0], forearm_uv[:, 1], forearm_faces)
    trifinder = triang.get_trifinder()

    tri_idx = trifinder(uv_points[:, 0], uv_points[:, 1])
    outside = tri_idx < 0
    if np.any(outside):
        n_out = int(outside.sum())
        logging.getLogger(__name__).warning(
            "uv_points_to_xyz: %d/%d UV point(s) lie outside the SLIM mesh "
            "and will be snapped to the nearest triangle boundary.",
            n_out, len(uv_points),
        )
        # For each outside point, assign the nearest triangle by centroid distance.
        tri_centroids = forearm_uv[forearm_faces].mean(axis=1)  # (n_F, 2)
        diffs = tri_centroids[np.newaxis, :, :] - uv_points[outside, np.newaxis, :]
        tri_idx[outside] = (diffs ** 2).sum(axis=-1).argmin(axis=-1)

    face_vids = forearm_faces[tri_idx]  # (N, 3)
    A = forearm_uv[face_vids[:, 0]]
    B = forearm_uv[face_vids[:, 1]]
    C = forearm_uv[face_vids[:, 2]]

    v0 = B - A
    v1 = C - A
    v2 = uv_points - A

    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)

    denom = d00 * d11 - d01 * d01
    if np.any(np.abs(denom) < 1e-30):
        raise ValueError(
            "uv_points_to_xyz: degenerate triangle(s) in SLIM mesh (near-zero "
            "UV area). The SLIM UV map may be invalid."
        )

    lam1 = (d11 * d20 - d01 * d21) / denom
    lam2 = (d00 * d21 - d01 * d20) / denom
    lam0 = 1.0 - lam1 - lam2

    # Snapped points land outside their assigned triangle; clamp to the nearest
    # edge/vertex by zeroing negative weights and renormalising.
    if np.any(outside):
        lams = np.stack([lam0, lam1, lam2], axis=1)
        lams[outside] = np.clip(lams[outside], 0.0, None)
        lams[outside] /= lams[outside].sum(axis=1, keepdims=True)
        lam0, lam1, lam2 = lams[:, 0], lams[:, 1], lams[:, 2]

    V0 = forearm_V[face_vids[:, 0]]
    V1 = forearm_V[face_vids[:, 1]]
    V2 = forearm_V[face_vids[:, 2]]

    xyz = (
        lam0[:, np.newaxis] * V0
        + lam1[:, np.newaxis] * V1
        + lam2[:, np.newaxis] * V2
    )
    return xyz.astype(np.float64)


# ---------------------------------------------------------------------------
# Barycentric UV lookup
# ---------------------------------------------------------------------------

def barycentric_uv_lookup(
    cache: 'SlimUvCache',
    query_points_3d: np.ndarray,
) -> np.ndarray:
    """For each query point, find the nearest face and barycentric-interpolate UV.

    Parameters
    ----------
    cache:
        Loaded SLIM UV cache from ``load_slim_uv_cache()``.
    query_points_3d:
        (N, 3) array of 3D query positions in mesh coordinates.

    Returns
    -------
    np.ndarray
        (N, 2) float64 UV coordinates.
    """
    V = cache.V
    F = cache.F
    uv = cache.uv
    P = np.asarray(query_points_3d, dtype=np.float64)

    # Face centroids — KDTree finds the nearest face for each query point.
    face_centroids = V[F].mean(axis=1)          # (M, 3)
    tree = KDTree(face_centroids)
    _, face_idx = tree.query(P)                  # (N,)

    # Face vertices
    A = V[F[face_idx, 0]]                        # (N, 3)
    B = V[F[face_idx, 1]]                        # (N, 3)
    C = V[F[face_idx, 2]]                        # (N, 3)

    # Face normals (unnormalized)
    n = np.cross(B - A, C - A)                   # (N, 3)
    nn = np.einsum('ij,ij->i', n, n)             # (N,)
    degenerate = nn < 1e-20
    nn_safe = np.where(degenerate, 1.0, nn)
    n_hat = n / np.sqrt(nn_safe)[:, np.newaxis]  # (N, 3)

    # Project query points onto the face plane.
    PA = P - A
    d_plane = np.einsum('ij,ij->i', PA, n_hat)
    P_proj = P - d_plane[:, np.newaxis] * n_hat  # (N, 3)

    # Barycentric coordinates (Cramer's rule on the face plane).
    v0 = B - A
    v1 = C - A
    v2 = P_proj - A

    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)

    denom = d00 * d11 - d01 * d01
    bad = degenerate | (np.abs(denom) < 1e-20)
    denom_safe = np.where(bad, 1.0, denom)

    lam1 = (d11 * d20 - d01 * d21) / denom_safe
    lam2 = (d00 * d21 - d01 * d20) / denom_safe
    lam0 = 1.0 - lam1 - lam2

    # Interpolate UV.
    uv0 = uv[F[face_idx, 0]]                     # (N, 2)
    uv1 = uv[F[face_idx, 1]]                     # (N, 2)
    uv2 = uv[F[face_idx, 2]]                     # (N, 2)

    result = (
        lam0[:, np.newaxis] * uv0
        + lam1[:, np.newaxis] * uv1
        + lam2[:, np.newaxis] * uv2
    )

    # For degenerate faces, fall back to the face centroid UV.
    face_centroid_uv = (uv0 + uv1 + uv2) / 3.0
    result = np.where(bad[:, np.newaxis], face_centroid_uv, result)

    return result.astype(np.float64)

"""Interactive workbench GUI for exploring forearm registration parameters.

Provides a top parameter panel (unification mode, canonical key, registration
method) and dual synchronized 3D viewports side-by-side:

* **Left viewport** - unified white cloud + original red snapshot (before ICP)
* **Right viewport** - unified white cloud + ICP-transformed green snapshot (after ICP)

Standalone test::

    python registration_workbench.py
"""

import dataclasses
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

try:
    from .forearm_registrator import ForearmRegistrator
except:
    from forearm_registrator import ForearmRegistrator


@dataclasses.dataclass(frozen=True)
class RegistrationResult:
    """Immutable result returned by :meth:`RegistrationWorkbench.get_result`.

    All fields are captured at the moment the user clicks Accept, so they
    remain valid even after the GUI window has been destroyed.

    Attributes:
        mode: Unification mode used (``"reference"`` or ``"average"``).
        canonical_key: Snapshot key used as the ICP reference.
        registration_method: Name of the ICP method used (e.g. ``"vanilla"``).
        max_correspondence_distance: ICP distance threshold (metres).
        icp_max_iteration: Maximum ICP iterations used.
        transforms: Per-snapshot ``{key: (4x4_matrix, fitness)}`` mapping.
        unified_cloud: The unified registered point cloud.
    """

    mode: str
    canonical_key: str
    registration_method: str
    max_correspondence_distance: float
    icp_max_iteration: int
    transforms: Dict[str, Tuple[np.ndarray, float]]
    unified_cloud: "o3d.geometry.PointCloud"


_C_UNIFIED     = [1.0, 1.0, 1.0]        # white
_C_ORIGINAL    = [0.90, 0.20, 0.20]     # red   - before ICP
_C_TRANSFORMED = [0.20, 0.80, 0.30]     # green - after ICP

_UNIFICATION_MODES    = ["reference", "average"]
_REGISTRATION_METHODS = ["vanilla", "robust", "multiscale", "generalized", "trimmed", "global"]

# Widest possible fitness label text: "[canonical - identity transform]" (32 chars).
# Using a same-width placeholder of non-breaking spaces guarantees that
# gui.Horiz.calc_preferred_size() always returns a non-zero width for the label,
# preventing the container from collapsing it to ~0 px when the display is "empty".
_FITNESS_LABEL_EMPTY = "\u00a0" * 70 # 34 ≥ len("[canonical - identity transform]")


def _geo_base(key: str) -> str:
    return f"base_{key.replace(':', '_')}"


def _geo_xf(key: str) -> str:
    return f"xf_{key.replace(':', '_')}"


def _apply_pose_to_camera(
    widget: gui.SceneWidget,
    model_matrix: np.ndarray,
) -> None:
    """Apply *model_matrix* to *widget*'s camera via look_at decomposition."""
    eye     = model_matrix[:3, 3]
    R       = model_matrix[:3, :3]
    up      = R[:, 1]
    forward = -R[:, 2]
    center  = eye + forward
    widget.scene.camera.look_at(center, eye, up)


class RegistrationWorkbench:
    """Interactive GUI for exploring forearm ICP registration parameters.

    The window layout (all children are direct Window children per the
    Open3D SceneWidget constraint documented in note-open3d-scenewidget-layout.md):

    .. code-block:: text

        Window
          +-- top_panel      (gui.Vert)   - mode/canonical/method dropdowns + Process button
          +-- snapshot_panel (gui.Horiz)  - snapshot selector dropdown + fitness label
          +-- scene_left     (SceneWidget) - unified white + original red
          +-- scene_right    (SceneWidget) - unified white + transformed green

    Args:
        clouds: Mapping ``{snapshot_key: point_cloud}`` of forearm snapshots.
    """

    def __init__(
        self,
        clouds: Dict[str, o3d.geometry.PointCloud],
        existing_state: Optional[dict] = None,
    ) -> None:
        self._clouds       = clouds
        self._snapshot_keys = sorted(clouds.keys())
        self._existing_state: Optional[dict] = existing_state

        # Set after processing
        self._transforms:    Optional[Dict[str, Tuple[np.ndarray, float]]] = None
        self._unified_cloud: Optional[o3d.geometry.PointCloud]              = None
        self._canonical_key: Optional[str]                                  = None

        # Camera sync state
        self._saved_camera:        Optional[np.ndarray] = None
        self._camera_initialized:  bool                 = False
        self._left_matrix:         np.ndarray           = np.eye(4)
        self._right_matrix:        np.ndarray           = np.eye(4)

        # Currently visible geometry names (for show/hide toggling)
        self._shown_base: Optional[str] = None
        self._shown_xf:   Optional[str] = None

        # Suppress snapshot callback during dropdown repopulation
        self._rebuilding_snapshot: bool = False

        # Accept state and cached parameters (widgets may be invalid after close).
        # Seed from existing_state when provided so that _build_gui can read them.
        _es = existing_state or {}
        self._accepted:      bool  = False
        self._last_mode:     str   = _es.get("mode", "average")
        self._last_method:   str   = _es.get("parameters", {}).get("registration_method", "global")
        self._last_max_dist: float = _es.get("parameters", {}).get("max_correspondence_distance", 2.00)
        self._last_max_iter: int   = _es.get("parameters", {}).get("icp_max_iteration", 500)

        # GUI handles (all set by _build_gui)
        self._win:                  Optional[gui.Window]      = None
        self._top_panel:            Optional[gui.Vert]        = None
        self._snapshot_panel:       Optional[gui.Horiz]       = None
        self._scene_left:           Optional[gui.SceneWidget] = None
        self._scene_right:          Optional[gui.SceneWidget] = None
        self._combo_mode:       Optional[gui.Combobox]    = None
        self._lbl_canonical:    Optional[gui.Label]       = None
        self._combo_canonical:  Optional[gui.Combobox]    = None
        self._combo_method:     Optional[gui.Combobox]    = None
        self._combo_snapshot:   Optional[gui.Combobox]    = None
        self._edit_max_dist:    Optional[gui.NumberEdit]  = None
        self._edit_max_iter:    Optional[gui.NumberEdit]  = None
        self._btn_accept:       Optional[gui.Button]      = None
        self._lbl_status:       Optional[gui.Label]       = None
        self._lbl_fitness:      Optional[gui.Label]       = None
        self._em:               float                     = 0.0
        self._layout_ctx:       Optional[gui.LayoutContext] = None

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_gui(self, app: gui.Application) -> None:
        try:
            screen = app.get_monitor_bounds(0)
            win_w, win_h = screen.width, screen.height
        except Exception:
            win_w, win_h = 1400, 900

        self._win = app.create_window("Registration Workbench", win_w, win_h)
        em = self._em = self._win.theme.font_size

        # --- Top panel: single row of parameter controls ---
        self._top_panel = gui.Vert(
            int(0.5 * em), gui.Margins(em, int(0.5 * em), em, int(0.5 * em))
        )

        # Single control row:
        #   Mode  [gap]  Canonical (ref only)  [gap]  Method  [gap]
        #   MaxDist  MaxIter  [gap]  Process  ?  status
        row = gui.Horiz(int(0.5 * em), gui.Margins(0, 0, 0, 0))

        row.add_child(gui.Label("Mode:"))
        self._combo_mode = gui.Combobox()
        for m in _UNIFICATION_MODES:
            self._combo_mode.add_item(m)
        _mode_idx = _UNIFICATION_MODES.index(self._last_mode) if self._last_mode in _UNIFICATION_MODES else _UNIFICATION_MODES.index("average")
        self._combo_mode.selected_index = _mode_idx
        self._combo_mode.set_on_selection_changed(self._on_mode_changed)
        row.add_child(self._combo_mode)

        row.add_fixed(int(em * 1.5))
        self._lbl_canonical = gui.Label("Canonical (ref only):")
        row.add_child(self._lbl_canonical)
        self._combo_canonical = gui.Combobox()
        for k in self._snapshot_keys:
            self._combo_canonical.add_item(k)
        _existing_canonical = (self._existing_state or {}).get("canonical_key")
        if _existing_canonical is not None and _existing_canonical in self._snapshot_keys:
            self._combo_canonical.selected_index = self._snapshot_keys.index(_existing_canonical)
        else:
            self._combo_canonical.selected_index = 0
        row.add_child(self._combo_canonical)
        row.add_fixed(int(em * 1.5))
        row.add_child(gui.Label("Method:"))
        self._combo_method = gui.Combobox()
        for meth in _REGISTRATION_METHODS:
            self._combo_method.add_item(meth)
        _method_idx = _REGISTRATION_METHODS.index(self._last_method) if self._last_method in _REGISTRATION_METHODS else len(_REGISTRATION_METHODS) - 1
        self._combo_method.selected_index = _method_idx
        row.add_child(self._combo_method)

        row.add_fixed(int(em * 1.5))
        row.add_child(gui.Label("Max dist (m):"))
        self._edit_max_dist = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._edit_max_dist.double_value = self._last_max_dist
        self._edit_max_dist.set_limits(0.001, 5.000)
        row.add_child(self._edit_max_dist)

        row.add_fixed(em)
        row.add_child(gui.Label("Max iter:"))
        self._edit_max_iter = gui.NumberEdit(gui.NumberEdit.INT)
        self._edit_max_iter.int_value = self._last_max_iter
        self._edit_max_iter.set_limits(10, 2000)
        row.add_child(self._edit_max_iter)

        row.add_fixed(int(em * 1.5))
        btn = gui.Button("Process")
        btn.horizontal_padding_em = 1.0
        btn.vertical_padding_em = 0.25
        btn.background_color = gui.Color(0.15, 0.60, 0.20, 1.0)
        btn.set_on_clicked(self._on_process_clicked)
        row.add_child(btn)

        row.add_fixed(int(0.5 * em))
        btn_help = gui.Button("?")
        btn_help.horizontal_padding_em = 0.5
        btn_help.vertical_padding_em = 0.25
        btn_help.set_on_clicked(self._on_help_clicked)
        row.add_child(btn_help)

        row.add_fixed(int(0.5 * em))
        self._btn_accept = gui.Button("Accept")
        self._btn_accept.horizontal_padding_em = 1.0
        self._btn_accept.vertical_padding_em = 0.25
        self._btn_accept.background_color = gui.Color(0.10, 0.40, 0.80, 1.0)
        self._btn_accept.enabled = False
        self._btn_accept.set_on_clicked(self._on_accept_clicked)
        row.add_child(self._btn_accept)

        row.add_fixed(em)
        self._lbl_status = gui.Label("")
        row.add_child(self._lbl_status)

        self._top_panel.add_child(row)

        # --- Snapshot panel: selector + fitness label ---
        self._snapshot_panel = gui.Horiz(
            em, gui.Margins(em, int(0.25 * em), em, int(0.25 * em))
        )
        self._snapshot_panel.add_child(gui.Label("Snapshot:"))
        self._combo_snapshot = gui.Combobox()
        self._combo_snapshot.add_item("(process first)" + " " * 150)
        self._combo_snapshot.selected_index = 0
        self._combo_snapshot.set_on_selection_changed(self._on_snapshot_changed)
        self._snapshot_panel.add_child(self._combo_snapshot)

        self._lbl_fitness = gui.Label(_FITNESS_LABEL_EMPTY)
        self._snapshot_panel.add_stretch()
        self._snapshot_panel.add_child(self._lbl_fitness)
        self._snapshot_panel.add_fixed(em)

        # --- Scene widgets - must be direct window children (see KB note) ---
        self._scene_left = gui.SceneWidget()
        self._scene_left.scene = rendering.Open3DScene(self._win.renderer)
        self._scene_left.scene.set_background([0.12, 0.12, 0.12, 1.0])

        self._scene_right = gui.SceneWidget()
        self._scene_right.scene = rendering.Open3DScene(self._win.renderer)
        self._scene_right.scene.set_background([0.12, 0.12, 0.12, 1.0])

        self._win.set_on_layout(self._on_layout)
        self._win.add_child(self._top_panel)
        self._win.add_child(self._snapshot_panel)
        self._win.add_child(self._scene_left)
        self._win.add_child(self._scene_right)

    def _on_layout(self, ctx: gui.LayoutContext) -> None:
        """Explicitly size all 4 direct window children."""
        self._layout_ctx = ctx  # cache so we can force re-layout on canonical visibility changes
        r  = self._win.content_rect
        c  = gui.Widget.Constraints()
        em = self._em

        top_h  = self._top_panel.calc_preferred_size(ctx, c).height
        snap_h = self._snapshot_panel.calc_preferred_size(ctx, c).height

        self._top_panel.frame      = gui.Rect(r.x, r.y,         r.width, top_h)
        self._snapshot_panel.frame = gui.Rect(r.x, r.y + top_h, r.width, snap_h)

        scene_y = r.y + top_h + snap_h
        scene_h = max(1, r.height - top_h - snap_h)
        half_w  = r.width // 2

        self._scene_left.frame  = gui.Rect(r.x,          scene_y, half_w,           scene_h)
        self._scene_right.frame = gui.Rect(r.x + half_w, scene_y, r.width - half_w, scene_h)

    @staticmethod
    def _make_mat() -> rendering.MaterialRecord:
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0
        return mat

    # ------------------------------------------------------------------
    # Help dialog
    # ------------------------------------------------------------------

    _HELP_TEXT = """\
REGISTRATION WORKBENCH - Parameter Reference

---------------------------------------------------
PARAMETER CONTROLS (top row)
---------------------------------------------------

Mode
  reference - The canonical snapshot is held fixed (identity
              transform). All other snapshots are aligned to it.
  average   - All snapshots are aligned to the canonical, then every
              transform is re-centred on the SE(3) group mean so that
              no single snapshot is privileged.

Canonical
  The snapshot used as the fixed reference frame for ICP. Defaults to
  the first key (lowest representative_frame_id). The canonical always
  receives an identity transform (fitness = 1.0).

Method
  vanilla      - Point-to-plane ICP, equal weights. Best for small
                 shifts with > 95 % overlap.
  robust       - Point-to-plane ICP + Tukey loss kernel. Use when
                 fringe artefacts bias the vanilla result.
  multiscale   - Coarse-to-fine ICP at 3 resolution levels. Use for
                 larger shifts or multi-cm displacement.
  generalized  - Generalized ICP (GICP), models local surface
                 covariance. Best for dense, noisy clouds.
  trimmed      - Removes the farthest source points before ICP.
                 Use when partial overlap is known (arm shift).
  global       - FPFH + RANSAC global alignment followed by ICP
                 refinement. Use when identity initialisation fails
                 (large displacements).

Max correspondence dist (m)
  ICP distance threshold. Source points farther than this from their
  nearest target point are excluded from each ICP iteration.
  Default: 0.10 m (10 cm). Increase for larger arm shifts; decrease
  to reject false correspondences on well-aligned clouds.

Max ICP iterations
  Upper bound on ICP iterations per registration. Convergence
  typically occurs in 50-80 iterations for forearm-scale clouds.
  Default: 200. Increase if fitness is still rising at termination.

---------------------------------------------------
BUTTONS
---------------------------------------------------

Process (green)
  Runs ICP registration with the current parameters. Populates both
  viewports with the unified white cloud. The snapshot dropdown is
  then updated with fitness scores for each snapshot.

? (this dialog)
  Shows this help text.

Accept (blue)
  Saves the results of the last successful Process run and closes the
  workbench. Only enabled after at least one successful Process run.
  Closing the window without clicking Accept discards the results and
  causes the calling pipeline step to return None (no artifacts written).

---------------------------------------------------
SNAPSHOT SELECTOR (second row)
---------------------------------------------------

Snapshot dropdown
  "- none -"  - Both viewports show only the unified white cloud.
  Any other entry - Overlays the selected snapshot in both viewports:
    Left viewport:  white (unified) + red   (original, before ICP)
    Right viewport: white (unified) + green (transformed, after ICP)

  [ref] prefix indicates the canonical snapshot (identity transform).
  fit=  shows the ICP fitness score (fraction of source points with a
        correspondence within max_correspondence_distance).
  Fitness >= 0.9 is considered reliable.

---------------------------------------------------
VIEWPORTS
---------------------------------------------------

Both viewports are synchronised: rotating or zooming in one
automatically mirrors the same camera movement in the other.
The camera view is preserved when switching snapshots or
re-running Process with different parameters.\
"""

    def _on_help_clicked(self) -> None:
        self._win.show_message_box("Registration Workbench - Help", self._HELP_TEXT)

    # ------------------------------------------------------------------
    # Accept / result
    # ------------------------------------------------------------------

    def _on_accept_clicked(self) -> None:
        self._accepted = True
        self._win.close()

    @property
    def accepted(self) -> bool:
        """``True`` if the user clicked Accept before closing the window."""
        return self._accepted

    def get_result(self) -> Optional[RegistrationResult]:
        """Return the last accepted registration result, or ``None`` if cancelled.

        All values are read from cached fields so the method is safe to call
        after the GUI window has been destroyed.
        """
        if not self._accepted or self._transforms is None:
            return None
        return RegistrationResult(
            mode=self._last_mode,
            canonical_key=self._canonical_key,
            registration_method=self._last_method,
            max_correspondence_distance=self._last_max_dist,
            icp_max_iteration=self._last_max_iter,
            transforms=self._transforms,
            unified_cloud=self._unified_cloud,
        )

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _on_mode_changed(self, text: str, index: int) -> None:
        self._win.post_redraw()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _on_process_clicked(self) -> None:
        """Run ICP with the current dropdown values and refresh viewports."""
        if not self._snapshot_keys:
            return

        mode     = self._combo_mode.selected_text
        can_key  = self._combo_canonical.selected_text
        method   = self._combo_method.selected_text
        max_dist = self._edit_max_dist.double_value
        max_iter = int(self._edit_max_iter.int_value)

        self._lbl_status.text = "Processing..."
        self._win.post_redraw()

        registrator = ForearmRegistrator(
            self._clouds[can_key],
            max_correspondence_distance=max_dist,
            icp_max_iteration=max_iter,
        )

        if mode == "reference":
            transforms = registrator.register_all(
                self._clouds, can_key, method=method
            )
        else:  # "average"
            transforms = registrator.register_all_to_average(
                self._clouds, can_key, method=method
            )

        unified = registrator.build_unified_cloud(self._clouds, transforms, can_key)

        self._transforms    = transforms
        self._unified_cloud = unified
        self._canonical_key = can_key

        # Cache widget values before they could be destroyed (needed by get_result)
        self._last_mode     = mode
        self._last_method   = method
        self._last_max_dist = max_dist
        self._last_max_iter = max_iter

        self._rebuild_geometries()
        self._repopulate_snapshot_dropdown()
        self._lbl_status.text = ""
        self._btn_accept.enabled = True

    def _rebuild_geometries(self) -> None:
        """Clear both scenes and re-upload all geometries; preserve camera."""
        # Save camera before clearing (clear_geometry may reset it)
        if self._camera_initialized:
            self._saved_camera = self._scene_left.scene.camera.get_model_matrix()

        self._scene_left.scene.clear_geometry()
        self._scene_right.scene.clear_geometry()
        self._shown_base = None
        self._shown_xf   = None

        mat = self._make_mat()

        # Unified cloud (white) - always visible in both viewports
        unified_display = o3d.geometry.PointCloud(self._unified_cloud)
        unified_display.paint_uniform_color(_C_UNIFIED)
        self._scene_left.scene.add_geometry("unified",  unified_display, mat)
        self._scene_right.scene.add_geometry("unified", unified_display, mat)

        # Per-snapshot variants: base (red) in left, transformed (green) in right
        # All hidden initially; toggled by _on_snapshot_changed via show_geometry.
        for key in self._snapshot_keys:
            T, _ = self._transforms[key]

            base = o3d.geometry.PointCloud(self._clouds[key])
            base.paint_uniform_color(_C_ORIGINAL)
            self._scene_left.scene.add_geometry(_geo_base(key), base, mat)
            self._scene_left.scene.show_geometry(_geo_base(key), False)

            xf = o3d.geometry.PointCloud(self._clouds[key])
            xf.transform(T)
            xf.paint_uniform_color(_C_TRANSFORMED)
            self._scene_right.scene.add_geometry(_geo_xf(key), xf, mat)
            self._scene_right.scene.show_geometry(_geo_xf(key), False)

        # Camera: first use → setup from bounding box; subsequent → restore
        bbox = unified_display.get_axis_aligned_bounding_box()
        if not self._camera_initialized or self._saved_camera is None:
            self._scene_left.setup_camera(60.0, bbox, bbox.get_center())
            self._scene_right.setup_camera(60.0, bbox, bbox.get_center())
            self._camera_initialized = True
        else:
            _apply_pose_to_camera(self._scene_left,  self._saved_camera)
            _apply_pose_to_camera(self._scene_right, self._saved_camera)

        self._left_matrix  = self._scene_left.scene.camera.get_model_matrix()
        self._right_matrix = self._left_matrix.copy()

        self._win.post_redraw()

    def _repopulate_snapshot_dropdown(self) -> None:
        """Rebuild the snapshot combobox with keys and fitness scores.

        A blank "- none -" placeholder is inserted at index 0 so that the
        initial state shows only the white unified cloud in both viewports.
        The user selects a specific snapshot to add the red/green overlay.
        """
        # Remember the currently selected snapshot key so we can restore it.
        prev_index = self._combo_snapshot.selected_index
        prev_key: Optional[str] = None
        if prev_index > 0:
            snap_idx = prev_index - 1
            if 0 <= snap_idx < len(self._snapshot_keys):
                prev_key = self._snapshot_keys[snap_idx]

        self._rebuilding_snapshot = True
        try:
            while self._combo_snapshot.number_of_items > 0:
                self._combo_snapshot.remove_item(0)
            self._combo_snapshot.add_item("- none -")
            mode = self._combo_mode.selected_text
            for key in self._snapshot_keys:
                _, fitness = self._transforms[key]
                prefix = "[ref] " if (mode == "reference" and key == self._canonical_key) else ""
                self._combo_snapshot.add_item(f"{prefix}{key}  fit={fitness:.3f}")

            # Restore previous selection if it still exists, otherwise reset.
            if prev_key is not None and prev_key in self._snapshot_keys:
                self._combo_snapshot.selected_index = self._snapshot_keys.index(prev_key) + 1
            else:
                self._combo_snapshot.selected_index = 0
        finally:
            self._rebuilding_snapshot = False

        # Refresh the overlay to match the (possibly restored) selection.
        idx = self._combo_snapshot.selected_index
        if idx == 0:
            self._clear_snapshot_overlays()
        else:
            self._on_snapshot_changed(self._combo_snapshot.selected_text, idx)

    def _clear_snapshot_overlays(self) -> None:
        """Hide any currently visible snapshot overlay in both viewports."""
        if self._shown_base is not None:
            self._scene_left.scene.show_geometry(self._shown_base, False)
            self._shown_base = None
        if self._shown_xf is not None:
            self._scene_right.scene.show_geometry(self._shown_xf, False)
            self._shown_xf = None
        self._lbl_fitness.text = _FITNESS_LABEL_EMPTY
        self._win.post_redraw()

    def _on_snapshot_changed(self, text: str, index: int) -> None:
        """Toggle geometry visibility to show the selected snapshot."""
        if self._rebuilding_snapshot or self._transforms is None:
            return

        # Index 0 is the "- none -" placeholder: show only the unified cloud.
        if index == 0:
            self._clear_snapshot_overlays()
            return

        snap_index = index - 1  # offset by the placeholder at position 0
        if snap_index < 0 or snap_index >= len(self._snapshot_keys):
            return

        key = self._snapshot_keys[snap_index]
        _, fitness = self._transforms[key]

        # Unified cloud (white) is always visible in both viewports.
        self._scene_left.scene.show_geometry("unified", True)
        self._scene_right.scene.show_geometry("unified", True)

        # Left viewport: unified (white) + original (red, before ICP)
        if self._shown_base is not None:
            self._scene_left.scene.show_geometry(self._shown_base, False)
        geo_b = _geo_base(key)
        self._scene_left.scene.show_geometry(geo_b, True)
        self._shown_base = geo_b

        # Right viewport: unified (white) + transformed (green, after ICP)
        if self._shown_xf is not None:
            self._scene_right.scene.show_geometry(self._shown_xf, False)
        geo_x = _geo_xf(key)
        self._scene_right.scene.show_geometry(geo_x, True)
        self._shown_xf = geo_x

        is_reference_canonical = (
            key == self._canonical_key
            and self._combo_mode.selected_text == "reference"
        )
        self._lbl_fitness.text = (
            "[canonical - identity transform]"
            if is_reference_canonical
            else f"fitness={fitness:.4f}"
        )
        self._win.post_redraw()

    # ------------------------------------------------------------------
    # Camera synchronization (polling loop)
    # ------------------------------------------------------------------

    def _sync_loop(self) -> None:
        """Mirror camera changes between viewports.  Runs on the main thread."""
        if self._scene_left.frame.height <= 0 or self._scene_right.frame.height <= 0:
            gui.Application.instance.post_to_main_thread(self._win, self._sync_loop)
            return

        cur_left  = self._scene_left.scene.camera.get_model_matrix()
        cur_right = self._scene_right.scene.camera.get_model_matrix()

        if not np.allclose(cur_left, self._left_matrix, atol=1e-6):
            _apply_pose_to_camera(self._scene_right, cur_left)
            h = self._scene_right.frame.height
            if h > 0:
                self._scene_right.scene.camera.set_projection(
                    self._scene_left.scene.camera.get_field_of_view(),
                    self._scene_right.frame.width / h,
                    0.1, 1000.0,
                    rendering.Camera.FovType.Vertical,
                )
            self._scene_right.force_redraw()
            self._left_matrix  = cur_left
            self._right_matrix = self._scene_right.scene.camera.get_model_matrix()

        elif not np.allclose(cur_right, self._right_matrix, atol=1e-6):
            _apply_pose_to_camera(self._scene_left, cur_right)
            h = self._scene_left.frame.height
            if h > 0:
                self._scene_left.scene.camera.set_projection(
                    self._scene_right.scene.camera.get_field_of_view(),
                    self._scene_left.frame.width / h,
                    0.1, 1000.0,
                    rendering.Camera.FovType.Vertical,
                )
            self._scene_left.force_redraw()
            self._right_matrix = cur_right
            self._left_matrix  = self._scene_left.scene.camera.get_model_matrix()

        gui.Application.instance.post_to_main_thread(self._win, self._sync_loop)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _maximize_os_window(self) -> None:
        """Ask the OS window manager to maximize the window after first layout."""
        # Windows (including WSL2 running a native Windows Python)
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, "Registration Workbench")
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
                return
        except Exception:
            pass
        # Linux / WSL2-WSLg via X11 (requires: sudo apt install wmctrl)
        try:
            import subprocess
            subprocess.Popen(
                ["wmctrl", "-r", "Registration Workbench",
                 "-b", "add,maximized_vert,maximized_horz"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def show(self) -> None:
        """Build the workbench GUI and run the event loop (blocks until closed)."""
        app = gui.Application.instance
        app.initialize()
        self._build_gui(app)
        gui.Application.instance.post_to_main_thread(self._win, self._sync_loop)
        gui.Application.instance.post_to_main_thread(self._win, self._maximize_os_window)
        app.run()


# ------------------------------------------------------------------
# __main__ - standalone test with fake Gaussian point clouds
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def _make_plane_cloud(
        offset_xyz=(0.0, 0.0, 0.0),
        tilt_deg=0.0,
        nx: int = 80,
        ny: int = 25,
        length: float = 0.30,
        width: float = 0.09,
        z_noise: float = 0.002,
    ) -> o3d.geometry.PointCloud:
        """Rectangular flat plane - forearm-shaped (long × narrow)."""
        xs = np.linspace(-length / 2, length / 2, nx)
        ys = np.linspace(-width / 2,  width / 2,  ny)
        xx, yy = np.meshgrid(xs, ys)
        zz = rng.standard_normal(xx.shape) * z_noise

        pts = np.column_stack([
            xx.ravel() + offset_xyz[0],
            yy.ravel() + offset_xyz[1],
            zz.ravel() + offset_xyz[2],
        ])

        # Apply a small tilt around the Y axis (simulate arm rotation)
        if tilt_deg != 0.0:
            a = np.deg2rad(tilt_deg)
            R = np.array([[np.cos(a), 0, np.sin(a)],
                          [0,         1, 0         ],
                          [-np.sin(a), 0, np.cos(a)]])
            pts = pts @ R.T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        return pcd

    fake_clouds = {
        "session:0": _make_plane_cloud(offset_xyz=( 0.000,  0.000,  0.000), tilt_deg= 0.0),
        "session:1": _make_plane_cloud(offset_xyz=( 0.015,  0.005,  0.040), tilt_deg= 2.0),
        "session:2": _make_plane_cloud(offset_xyz=(-0.010,  0.008, -0.035), tilt_deg=-1.5),
    }

    RegistrationWorkbench(fake_clouds).show()

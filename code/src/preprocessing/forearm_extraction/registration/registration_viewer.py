"""Interactive GUI for inspecting forearm registration quality."""

from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def show_registration_viewer(
    clouds: Dict[str, o3d.geometry.PointCloud],
    transforms: Dict[str, Tuple[np.ndarray, float]],
    unified_cloud: o3d.geometry.PointCloud,
    canonical_key: str,
) -> None:
    """Interactive GUI for inspecting registration quality.

    Shows the merged unified cloud and lets the user select any snapshot from
    a side-panel list to compare its original position (red) against the
    ICP-aligned position (green), both overlaid on the dimmed unified cloud
    (grey).

    The window blocks until closed.
    """

    _C_UNIFIED    = [1.0, 1.0, 1.0]       # white
    _C_ORIGINAL   = [0.90, 0.20, 0.20]    # red   — before ICP
    _C_TRANSFORMED = [0.20, 0.80, 0.30]   # green — after ICP

    # Build a display copy of the unified cloud once.
    unified_display = o3d.geometry.PointCloud(unified_cloud)
    unified_display.paint_uniform_color(_C_UNIFIED)

    # ------------------------------------------------------------------
    # Pre-compute coloured snapshot geometries for every key.
    # Both the base (red) and the transformed (green) variant are built
    # up-front so they can be uploaded to the GPU once and then toggled
    # via show_geometry / hide_geometry — avoiding add/remove races.
    # ------------------------------------------------------------------
    snapshot_keys = sorted(transforms.keys())

    def _geo_base(key: str) -> str:
        return f"base_{key.replace(':', '_')}"

    def _geo_xf(key: str) -> str:
        return f"xf_{key.replace(':', '_')}"

    snap_base: Dict[str, o3d.geometry.PointCloud] = {}
    snap_xf:   Dict[str, o3d.geometry.PointCloud] = {}
    for key in snapshot_keys:
        T, _ = transforms[key]

        base = o3d.geometry.PointCloud(clouds[key])
        base.paint_uniform_color(_C_ORIGINAL)
        snap_base[key] = base

        xf = o3d.geometry.PointCloud(clouds[key])
        xf.transform(T)
        xf.paint_uniform_color(_C_TRANSFORMED)
        snap_xf[key] = xf

    app = gui.Application.instance
    app.initialize()
    win = app.create_window("Forearm Registration Viewer", 1400, 900)
    em = win.theme.font_size

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    scene.scene.set_background([0.12, 0.12, 0.12, 1.0])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0

    # ------------------------------------------------------------------
    # Visibility management.
    # "unified" is always visible.
    # Exactly one snapshot geometry (or none) is visible at a time.
    # _shown tracks which geometry name is currently shown so we can
    # hide it before showing another.
    # ------------------------------------------------------------------
    _shown: Dict[str, Optional[str]] = {"name": None}

    def _set_snapshot_visible(key: Optional[str], show_transformed: bool) -> None:
        """Make exactly one snapshot visible (or none). Hides the previous."""
        if _shown["name"] is not None:
            scene.scene.show_geometry(_shown["name"], False)
            _shown["name"] = None
        if key is not None:
            name = _geo_xf(key) if show_transformed else _geo_base(key)
            scene.scene.show_geometry(name, True)
            _shown["name"] = name
        win.post_redraw()

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------

    panel = gui.Vert(int(0.5 * em), gui.Margins(em, em, em, em))

    panel.add_child(gui.Label("Forearm Registration Viewer"))
    panel.add_child(gui.Label(""))

    status = gui.Label("Showing: unified cloud")
    panel.add_child(status)
    panel.add_child(gui.Label(""))

    legend = gui.Vert(0, gui.Margins(0, 0, 0, 0))
    legend.add_child(gui.Label("Legend:"))
    legend.add_child(gui.Label("  white = unified cloud"))
    legend.add_child(gui.Label("  red   = original (before ICP)"))
    legend.add_child(gui.Label("  green = transformed (after ICP)"))
    panel.add_child(legend)
    panel.add_child(gui.Label(""))

    def _show_unified(*, reset_camera: bool = False):
        _set_snapshot_visible(None, False)
        if reset_camera:
            bounds = unified_display.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())
        status.text = "Showing: unified cloud"
        win.post_redraw()

    # Toggle: original vs transformed snapshot
    chk_transformed = gui.Checkbox("Show transformed (green)")
    chk_transformed.checked = False  # default: show original (red)
    panel.add_child(chk_transformed)
    panel.add_child(gui.Label(""))
    panel.add_child(gui.Label("Snapshots (select to inspect):"))

    _selected: Dict[str, object] = {"key": None}
    _viewer_ready: Dict[str, bool] = {"ready": False}

    def _show_snapshot(key: str) -> None:
        _, fitness = transforms[key]
        show_transformed = chk_transformed.checked
        _set_snapshot_visible(key, show_transformed)
        if key == canonical_key:
            status.text = f"{key}  [canonical — identity transform]"
        else:
            which = "green = after ICP" if show_transformed else "red = before ICP"
            status.text = f"{key}   fit={fitness:.4f}   {which}"

    def _on_toggle(checked):
        if _selected["key"] is not None:
            _show_snapshot(_selected["key"])

    chk_transformed.set_on_checked(_on_toggle)

    # --- Radio-style checkboxes: one snapshot at a time ---
    snap_checkboxes: Dict[str, gui.Checkbox] = {}
    snap_list = gui.Vert(int(0.25 * em), gui.Margins(0, 0, 0, 0))

    for key in snapshot_keys:
        _, fitness = transforms[key]
        prefix = "[ref] " if key == canonical_key else ""
        label = f"{prefix}{key}  fit={fitness:.3f}"
        cb = gui.Checkbox(label)
        cb.checked = False

        def _make_handler(k):
            def _on_snap_checked(checked):
                if not _viewer_ready["ready"]:
                    return
                if checked:
                    # Uncheck all others
                    for other_key, other_cb in snap_checkboxes.items():
                        if other_key != k:
                            other_cb.checked = False
                    _selected["key"] = k
                    _show_snapshot(k)
                else:
                    # User unchecked — show unified
                    _selected["key"] = None
                    _show_unified()
                    status.text = "Showing: unified cloud"
            return _on_snap_checked

        cb.set_on_checked(_make_handler(key))
        snap_checkboxes[key] = cb
        snap_list.add_child(cb)

    panel.add_child(snap_list)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    PANEL_W = 360

    def _on_layout(_ctx):
        r = win.content_rect
        panel.frame = gui.Rect(r.x, r.y, PANEL_W, r.height)
        scene.frame = gui.Rect(r.x + PANEL_W, r.y, r.width - PANEL_W, r.height)
        if not _viewer_ready["ready"]:
            _viewer_ready["ready"] = True
            # Unified cloud — always visible.
            scene.scene.add_geometry("unified", unified_display, mat)
            # All snapshot geometries — uploaded once, all hidden initially.
            for key in snapshot_keys:
                scene.scene.add_geometry(_geo_base(key), snap_base[key], mat)
                scene.scene.show_geometry(_geo_base(key), False)
                scene.scene.add_geometry(_geo_xf(key), snap_xf[key], mat)
                scene.scene.show_geometry(_geo_xf(key), False)
            _show_unified(reset_camera=True)

    win.set_on_layout(_on_layout)
    win.add_child(panel)
    win.add_child(scene)

    app.run()

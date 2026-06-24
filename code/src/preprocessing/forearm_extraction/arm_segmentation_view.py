"""
arm_segmentation_view.py
------------------------
All display/interactive code for ArmSegmentation, extracted from
arm_segmentation.py.

Module-level imports are restricted to numpy, colorsys, and typing.
open3d.visualization.gui and open3d.visualization.rendering are imported
inside _display_filament() only so that batch mode never pulls in the
GUI/OpenGL stack.
"""
import colorsys
from typing import Callable, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GLFW key constants (used by the legacy VisualizerWithKeyCallback path)
# ---------------------------------------------------------------------------

_GLFW_KEY_UP    = 265
_GLFW_KEY_DOWN  = 264
_GLFW_KEY_LEFT  = 263
_GLFW_KEY_RIGHT = 262
_GLFW_KEY_ESC   = 256
_GLFW_KEY_ENTER = 257


# ---------------------------------------------------------------------------
# Filament availability probe (cached after first call)
# ---------------------------------------------------------------------------

_filament_available: bool | None = None


def _probe_filament() -> bool:
    """Try to create a 1×1 Filament window in a subprocess.

    Returns True if the subprocess exits with code 0, False otherwise.
    The result is *not* cached here; caching happens in
    ``_is_filament_available()``.
    """
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import open3d.visualization.gui as gui; "
             "gui.Application.instance.initialize(); "
             "w = gui.Application.instance.create_window('probe', 1, 1); "
             "gui.Application.instance.quit()"],
            timeout=10,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _is_filament_available() -> bool:
    """Return True if the Filament OpenGL backend is usable.

    Runs the probe on the first call and caches the result.  Subsequent
    calls return immediately.  Issues a ``RuntimeWarning`` if unavailable.
    """
    global _filament_available
    if _filament_available is None:
        _filament_available = _probe_filament()
        if not _filament_available:
            import warnings
            warnings.warn(
                "Filament OpenGL context creation failed — falling back to "
                "legacy GLFW Visualizer. Parameter tuning via keyboard only.",
                RuntimeWarning,
                stacklevel=3,
            )
    return _filament_available


# ---------------------------------------------------------------------------
# Screen size helper
# ---------------------------------------------------------------------------

def _get_screen_size() -> Tuple[int, int]:
    """Returns (width, height) of the primary screen, falling back to 1920×1080."""
    try:
        import tkinter as _tk
        _r = _tk.Tk()
        _r.withdraw()
        w, h = _r.winfo_screenwidth(), _r.winfo_screenheight()
        _r.destroy()
        return w, h
    except Exception:
        return 1920, 1080


# ---------------------------------------------------------------------------
# Widget factories
# ---------------------------------------------------------------------------

def _make_slider(cfg: dict, initial_value):
    """Creates a gui.Slider from a slider-config dict and an initial value."""
    import open3d.visualization.gui as gui
    slider_type = gui.Slider.INT if cfg.get('type') == 'int' else gui.Slider.DOUBLE
    s = gui.Slider(slider_type)
    s.set_limits(cfg['min'], cfg['max'])
    if slider_type == gui.Slider.INT:
        s.int_value = int(round(initial_value))
    else:
        s.double_value = float(initial_value)
    return s


def _make_range_slider_row(
        label_text: str,
        lo_cfg: dict,
        hi_cfg: dict,
        lo_init,
        hi_init,
        em: float,
) -> Tuple:
    """Creates a coupled lo/hi range control enforcing lo ≤ hi.

    Builds a vertical group labelled *label_text* containing two paired-slider
    rows.  Dragging lo above hi clamps hi upward; dragging hi below lo clamps
    lo downward.  Direct text entry is similarly clamped.

    Returns ``(container, get_range_fn)`` where
    ``get_range_fn() -> [lo_value, hi_value]``.
    """
    import open3d.visualization.gui as gui
    type_str = lo_cfg.get('type', 'float')
    lo_slider = _make_slider(lo_cfg, lo_init)
    lo_text   = gui.NumberEdit(gui.NumberEdit.DOUBLE)
    lo_text.double_value = float(lo_init)
    lo_text.set_preferred_width(4 * em)

    hi_slider = _make_slider(hi_cfg, hi_init)
    hi_text   = gui.NumberEdit(gui.NumberEdit.DOUBLE)
    hi_text.double_value = float(hi_init)
    hi_text.set_preferred_width(4 * em)

    syncing = [False]

    def _get_lo_raw():
        return lo_slider.int_value if type_str == 'int' else lo_slider.double_value

    def _get_hi_raw():
        return hi_slider.int_value if type_str == 'int' else hi_slider.double_value

    def _set_lo(v):
        if type_str == 'int':
            lo_slider.int_value = int(round(v))
        else:
            lo_slider.double_value = v
        lo_text.double_value = float(v)

    def _set_hi(v):
        if type_str == 'int':
            hi_slider.int_value = int(round(v))
        else:
            hi_slider.double_value = v
        hi_text.double_value = float(v)

    def _on_lo_slider(new_v):
        if syncing[0]:
            return
        syncing[0] = True
        lo_text.double_value = float(int(round(new_v)) if type_str == 'int' else new_v)
        if new_v > _get_hi_raw():
            _set_hi(new_v)
        syncing[0] = False

    def _on_hi_slider(new_v):
        if syncing[0]:
            return
        syncing[0] = True
        hi_text.double_value = float(int(round(new_v)) if type_str == 'int' else new_v)
        if new_v < _get_lo_raw():
            _set_lo(new_v)
        syncing[0] = False

    def _on_lo_text(new_v):
        if syncing[0]:
            return
        syncing[0] = True
        clamped = max(lo_cfg['min'], min(lo_cfg['max'], new_v))
        _set_lo(clamped)
        if clamped > _get_hi_raw():
            _set_hi(clamped)
        syncing[0] = False

    def _on_hi_text(new_v):
        if syncing[0]:
            return
        syncing[0] = True
        clamped = max(hi_cfg['min'], min(hi_cfg['max'], new_v))
        _set_hi(clamped)
        if clamped < _get_lo_raw():
            _set_lo(clamped)
        syncing[0] = False

    lo_slider.set_on_value_changed(_on_lo_slider)
    hi_slider.set_on_value_changed(_on_hi_slider)
    lo_text.set_on_value_changed(_on_lo_text)
    hi_text.set_on_value_changed(_on_hi_text)

    lo_row = gui.Horiz(0.25 * em)
    lo_row.add_child(gui.Label(f"  {lo_cfg.get('label', 'lo')}:"))
    lo_row.add_child(lo_slider)
    lo_row.add_child(lo_text)

    hi_row = gui.Horiz(0.25 * em)
    hi_row.add_child(gui.Label(f"  {hi_cfg.get('label', 'hi')}:"))
    hi_row.add_child(hi_slider)
    hi_row.add_child(hi_text)

    container = gui.Vert(0.25 * em)
    container.add_child(gui.Label(label_text))
    container.add_child(lo_row)
    container.add_child(hi_row)

    def get_range() -> list:
        lo = lo_slider.int_value if type_str == 'int' else lo_slider.double_value
        hi = hi_slider.int_value if type_str == 'int' else hi_slider.double_value
        return [lo, hi]

    return container, get_range


def _render_hue_wheel(size_px: int) -> np.ndarray:
    """Renders a full hue wheel as an RGB uint8 image of *size_px* × *size_px* pixels.

    Layout:
    - Ring occupying radii [32 %, 48 %] × size_px, coloured HSV(h, 1, 1).
    - Inner disc (r < 32 %) in dark grey — provides contrast for the handles.
    - Exterior corners (r > 48 %) in black.

    Hue 0° (red) is placed at the top (12 o'clock), increasing clockwise.
    """
    cx = cy = size_px / 2.0
    ring_outer = size_px * 0.48
    ring_inner = size_px * 0.32

    ys, xs = np.mgrid[0:size_px, 0:size_px]
    dx = (xs + 0.5) - cx
    dy = (ys + 0.5) - cy
    r = np.sqrt(dx * dx + dy * dy)

    # Hue angle: atan2 with -dy gives standard-math angle (CCW, 0 = right).
    # Subtracting from 90° flips to CW with 0° at top.
    theta_deg = np.degrees(np.arctan2(-dy, dx))
    hue_deg   = (90.0 - theta_deg) % 360.0
    hue_norm  = hue_deg / 360.0

    # Vectorised HSV(h, 1, 1) → RGB.
    h6 = hue_norm * 6.0
    i  = h6.astype(int) % 6
    f  = h6 - np.floor(h6)
    one  = np.ones_like(f)
    zero = np.zeros_like(f)
    R = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [one,     1 - f,  zero,   zero,   f,      one])
    G = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [f,      one,    one,    1 - f,  zero,   zero])
    B = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [zero,   zero,   f,      one,    one,    1 - f])

    ring_mask  = (r >= ring_inner) & (r <= ring_outer)
    inner_mask = r < ring_inner

    img = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    img[ring_mask,  0] = (R[ring_mask]  * 255).astype(np.uint8)
    img[ring_mask,  1] = (G[ring_mask]  * 255).astype(np.uint8)
    img[ring_mask,  2] = (B[ring_mask]  * 255).astype(np.uint8)
    img[inner_mask]    = 45  # dark grey inner disc
    return img


def _render_hue_arc_overlay(
        base_img: np.ndarray,
        h_start: float,
        h_end: float,
        handle_radius: int = 8,
) -> np.ndarray:
    """Returns a copy of *base_img* (RGB uint8) with the selected arc at full
    saturation and the rest of the ring darkened, plus two circular drag
    handles drawn on the mid-ring track.

    The selected arc is the clockwise arc from *h_start* to *h_end*; wrap-around
    ranges (e.g. 330°–30°) are fully supported.
    h_start handle → white fill; h_end handle → light-grey fill.
    """
    img = base_img.copy()
    size_px = img.shape[0]
    cx = cy = size_px / 2.0
    ring_outer    = size_px * 0.48
    ring_inner    = size_px * 0.32
    handle_track_r = (ring_inner + ring_outer) / 2.0

    ys, xs = np.mgrid[0:size_px, 0:size_px]
    dx = (xs + 0.5) - cx
    dy = (ys + 0.5) - cy
    r  = np.sqrt(dx * dx + dy * dy)

    theta_deg = np.degrees(np.arctan2(-dy, dx))
    hue_deg   = (90.0 - theta_deg) % 360.0
    ring_mask = (r >= ring_inner) & (r <= ring_outer)

    if h_start <= h_end:
        in_arc = ring_mask & (hue_deg >= h_start) & (hue_deg <= h_end)
    else:  # wrap-around
        in_arc = ring_mask & ((hue_deg >= h_start) | (hue_deg <= h_end))

    # Darken out-of-arc ring pixels; the selected arc keeps its full HSV colours.
    out_arc = ring_mask & ~in_arc
    img[out_arc] = (img[out_arc].astype(np.float32) * 0.25).astype(np.uint8)

    # Draw handles (h_start = white, h_end = light grey), both with black border.
    for h_angle, fill_col in ((h_start, (255, 255, 255)), (h_end, (200, 200, 200))):
        theta_rad = np.radians(90.0 - h_angle)
        hx = cx + handle_track_r * np.cos(theta_rad)
        hy = cy - handle_track_r * np.sin(theta_rad)
        dist_sq     = (xs + 0.5 - hx) ** 2 + (ys + 0.5 - hy) ** 2
        fill_mask   = dist_sq <= handle_radius ** 2
        border_mask = (dist_sq > handle_radius ** 2) & \
                      (dist_sq <= (handle_radius + 1.5) ** 2)
        img[fill_mask]   = fill_col
        img[border_mask] = (0, 0, 0)

    return img


def _make_hue_range_circle(
        h_start_init: float,
        h_end_init: float,
        em: float,
        renderer,
        size_px: int = 200,
) -> Tuple[Dict, Callable[[], list]]:
    """Creates a circular hue-range selector widget.

    Displays a hue wheel with two drag handles — one for *h_start* (white)
    and one for *h_end* (grey) — defining a clockwise arc.  Wrap-around
    ranges (e.g. 330°–30°) are fully supported.  Two companion
    ``NumberEdit`` boxes provide a precise text-entry alternative.

    Implementation note: the wheel is rendered as a 2-D background image on
    a ``SceneWidget`` (empty 3-D scene).  ``ImageWidget`` was not used
    because it never delivers ``DRAG`` events in Open3D 0.19.
    ``SceneWidget`` does deliver them; the 3-D camera is locked out by
    returning ``HANDLED`` for every mouse event so it never receives input.

    Returns ``(fragments_dict, get_hue_range_fn)`` where fragments_dict contains
    the independent UI elements ("top", "scene", "bottom") so the caller can place
    them in a split-panel layout, avoiding Open3D's nested auto-layout bugs.
    ``fragments_dict["refresh"]`` is the ``_refresh`` closure; call it after any
    layout change that resizes ``scene`` to re-apply the correct hue overlay.
    """
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    state = {
        'h_start':  float(h_start_init),
        'h_end':    float(h_end_init),
        'dragging': None,   # None | 'start' | 'end'
    }

    base_wheel   = _render_hue_wheel(size_px)
    ring_outer_r = size_px * 0.48
    ring_inner_r = size_px * 0.32
    handle_r     = max(4, int(size_px * 0.040))
    handle_tr    = (ring_inner_r + ring_outer_r) / 2.0
    cx = cy      = size_px / 2.0

    def _to_bg_image(rgb_arr: np.ndarray):
        rgba = np.dstack([rgb_arr, np.full(rgb_arr.shape[:2], 255, dtype=np.uint8)])
        return o3d.geometry.Image(np.ascontiguousarray(rgba))

    def _fresh_overlay() -> np.ndarray:
        return _render_hue_arc_overlay(
            base_wheel, state['h_start'], state['h_end'],
            handle_radius=handle_r,
        )

    # SceneWidget with an empty 3-D scene — display is via set_background,
    # which renders a flat 2-D quad unaffected by camera position.
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(renderer)
    scene_widget.enable_scene_caching(False)
    scene_widget.scene.set_background([0, 0, 0, 1], _to_bg_image(_fresh_overlay()))

    # --- Companion NumberEdit boxes ---
    start_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
    start_edit.double_value = float(h_start_init)
    end_edit   = gui.NumberEdit(gui.NumberEdit.DOUBLE)
    end_edit.double_value   = float(h_end_init)

    syncing = [False]

    def _refresh() -> None:
        scene_widget.scene.set_background([0, 0, 0, 1], _to_bg_image(_fresh_overlay()))

    def _on_start_edit(v: float):
        if syncing[0]:
            return
        syncing[0] = True
        state['h_start'] = float(max(0.0, min(360.0, v)))
        start_edit.double_value = state['h_start']
        _refresh()
        syncing[0] = False

    def _on_end_edit(v: float):
        if syncing[0]:
            return
        syncing[0] = True
        state['h_end'] = float(max(0.0, min(360.0, v)))
        end_edit.double_value = state['h_end']
        _refresh()
        syncing[0] = False

    start_edit.set_on_value_changed(_on_start_edit)
    end_edit.set_on_value_changed(_on_end_edit)

    # --- Mouse handler ---
    def _handle_image_pos(h_angle: float):
        """Return (hx, hy) in image-space pixels for hue angle h_angle."""
        theta_rad = np.radians(90.0 - h_angle)
        return (cx + handle_tr * np.cos(theta_rad),
                cy - handle_tr * np.sin(theta_rad))

    def _on_mouse(event):
        fr = scene_widget.frame
        px = event.x - fr.x
        py = event.y - fr.y
        fw, fh = fr.width, fr.height

        if event.type == gui.MouseEvent.BUTTON_DOWN:
            if fw > 0 and fh > 0:
                ix = px / fw * size_px
                iy = py / fh * size_px
                if 0 <= ix <= size_px and 0 <= iy <= size_px:
                    hx_s, hy_s = _handle_image_pos(state['h_start'])
                    hx_e, hy_e = _handle_image_pos(state['h_end'])
                    d_start = (ix - hx_s) ** 2 + (iy - hy_s) ** 2
                    d_end   = (ix - hx_e) ** 2 + (iy - hy_e) ** 2
                    grab_r2 = (handle_r * 2.5) ** 2
                    if d_start <= grab_r2 or d_end <= grab_r2:
                        state['dragging'] = 'start' if d_start <= d_end else 'end'
            # Re-apply background: clicking on a SceneWidget triggers an
            # internal scene reset that clears the set_background image.
            _refresh()
            # Always HANDLED — the 3-D camera must never receive mouse input.
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type in (gui.MouseEvent.DRAG, gui.MouseEvent.MOVE):
            if state['dragging'] is not None and fw > 0 and fh > 0:
                ix = px / fw * size_px
                iy = py / fh * size_px
                ddx = ix - cx
                ddy = iy - cy
                if ddx * ddx + ddy * ddy >= 1.0:
                    hue = (90.0 - np.degrees(np.arctan2(-ddy, ddx))) % 360.0
                    if state['dragging'] == 'start':
                        state['h_start'] = hue
                        syncing[0] = True
                        start_edit.double_value = round(hue, 1)
                        syncing[0] = False
                    else:
                        state['h_end'] = hue
                        syncing[0] = True
                        end_edit.double_value = round(hue, 1)
                        syncing[0] = False
                    _refresh()
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type == gui.MouseEvent.BUTTON_UP:
            state['dragging'] = None
            _refresh()
            return gui.Widget.EventCallbackResult.HANDLED

        # Catch scroll, right-drag, etc. — camera must not respond.
        return gui.Widget.EventCallbackResult.HANDLED

    scene_widget.set_on_mouse(_on_mouse)

    # --- Layout ---
    edit_row = gui.Horiz(0.25 * em)
    edit_row.add_child(gui.Label(" H start:"))
    edit_row.add_child(start_edit)
    edit_row.add_child(gui.Label(" end:"))
    edit_row.add_child(end_edit)

    title_lbl = gui.Label("H range (drag handles, wrap-around supported):")

    def get_hue_range() -> list:
        return [state['h_start'], state['h_end']]

    return {
        "top": title_lbl,
        "scene": scene_widget,
        "bottom": edit_row,
        "refresh": _refresh,
    }, get_hue_range


# ---------------------------------------------------------------------------
# Filament interactive display path
# ---------------------------------------------------------------------------

def _display_filament(
        segmenter,
        pcd_input,
        window_name: str,
        params_key: str,
        processing_func: Callable,
        is_cluster_step: bool,
):
    """Run the Filament-based interactive GUI for one segmentation step.

    ``segmenter`` is an ``ArmSegmentation`` instance.  The function reads and
    writes ``segmenter.params[params_key]``, ``segmenter._SLIDER_CONFIGS``,
    ``segmenter.was_modified``, and calls ``segmenter._hue_in_range()``.

    All ``open3d.visualization.gui`` and ``open3d.visualization.rendering``
    imports happen inside this function so batch-mode callers never load the
    GUI/OpenGL stack.
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    # Obtain screen dimensions before initialising the GUI toolkit so that
    # the window is created at full-screen size (maximised by default).
    screen_w, screen_h = _get_screen_size()
    gui.Application.instance.initialize()

    w = gui.Application.instance.create_window(window_name, screen_w, screen_h)

    # Use a dictionary to hold state that needs to be modified by callbacks
    state = {
        'pcd_processed': None,
        'camera_set':    False,   # True after the first setup_camera call
        'pcd_colors':    None,    # float RGB array matching the current cloud
        'pcd_points':    None,    # float XYZ array matching the current cloud
    }

    import open3d as o3d
    state['pcd_processed'] = o3d.geometry.PointCloud()

    # --- 3D Scene Widget ---
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    scene.scene.set_background([0.1, 0.2, 0.3, 1.0])  # Dark background
    w.add_child(scene)

    # --- GUI Controls Layout (Strategy C: Split Panel) ---
    em = w.theme.font_size
    panel_top = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.0))
    panel_bottom = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.0, 0.5 * em, 0.5 * em))
    hue_scene_widget = None

    current_panel = panel_top

    # Add the split panels directly to the window (hue_scene_widget added conditionally later)
    w.add_child(panel_top)
    w.add_child(panel_bottom)

    # --- Retrieve slider config for this step (may be empty) ---
    step_params = segmenter.params[params_key]
    slider_cfg = segmenter._SLIDER_CONFIGS.get(params_key, {})

    # widgets maps: key -> (widget, get_value_fn)
    #                   or list of (widget, get_value_fn)   for list parameters
    widgets = {}

    def _make_paired_row(label_text: str, cfg: dict, initial_value):
        """
        Builds a horizontal row: [label] [slider] [textbox].
        The slider and textbox are kept in bidirectional sync so the user
        can either drag the handle or type a value directly.
        Returns (row, get_value_fn).
        """
        type_str = cfg.get('type', 'float')
        slider = _make_slider(cfg, initial_value)

        textbox = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        textbox.double_value = float(int(round(initial_value)) if type_str == 'int' else initial_value)
        textbox.set_preferred_width(4 * em)

        # Guard prevents the two callbacks from triggering each other.
        syncing = [False]

        def _on_slider(new_v):
            if syncing[0]:
                return
            syncing[0] = True
            textbox.double_value = float(int(round(new_v)) if type_str == 'int' else new_v)
            syncing[0] = False

        def _on_textbox(new_v):
            if syncing[0]:
                return
            syncing[0] = True
            clamped = max(cfg['min'], min(cfg['max'], new_v))
            if type_str == 'int':
                slider.int_value = int(round(clamped))
            else:
                slider.double_value = clamped
            syncing[0] = False

        slider.set_on_value_changed(_on_slider)
        textbox.set_on_value_changed(_on_textbox)

        row = gui.Horiz(0.25 * em)
        row.add_child(gui.Label(label_text))
        row.add_child(slider)
        row.add_child(textbox)

        if type_str == 'int':
            get_value = lambda s=slider: s.int_value
        else:
            get_value = lambda s=slider: s.double_value

        return row, get_value

    for key, value in step_params.items():
        # Skip boolean flags — they are not tunable via numeric widgets.
        if isinstance(value, bool):
            continue

        key_cfg = slider_cfg.get(key)

        if isinstance(value, (float, int)):
            if key_cfg is not None:
                # --- Slider ---
                row, get_value = _make_paired_row(f"{key}:", key_cfg, value)
                current_panel.add_child(row)
                widgets[key] = (None, get_value)
            else:
                # --- Fallback: NumberEdit ---
                row = gui.Horiz(0.25 * em)
                row.add_child(gui.Label(f"{key}:"))
                row.add_stretch()
                widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                widget.double_value = value
                row.add_child(widget)
                current_panel.add_child(row)
                widgets[key] = (widget, lambda ww=widget: ww.double_value)

        elif isinstance(value, list) and all(isinstance(i, (float, int)) for i in value):
            # --- Circular hue range selector ---
            if (key_cfg is not None and isinstance(key_cfg, dict)
                    and key_cfg.get('is_hue_circle') and len(value) == 2):
                hue_elements, get_range = _make_hue_range_circle(
                    value[0], value[1], em=em, renderer=w.renderer
                )

                panel_top.add_child(hue_elements["top"])

                hue_scene_widget = hue_elements["scene"]
                hue_refresh_fn = hue_elements["refresh"]
                w.add_child(hue_scene_widget)  # Added directly to window

                # Redirect any future components into the bottom panel
                current_panel = panel_bottom
                current_panel.add_child(hue_elements["bottom"])

                widgets[key] = (None, get_range)

            # --- Coupled range slider (lo ≤ hi enforced) ---
            elif (key_cfg is not None and isinstance(key_cfg, dict)
                    and key_cfg.get('is_range') and len(value) == 2):
                cfgs = key_cfg['cfgs']
                container, get_range = _make_range_slider_row(
                    f"{key}:", cfgs[0], cfgs[1], value[0], value[1], em
                )
                current_panel.add_child(container)
                widgets[key] = (None, get_range)
            else:
                # --- Individual sliders (or fallback NumberEdits) per element ---
                current_panel.add_child(gui.Label(f"{key}:"))
                widgets[key] = []

                for i, v in enumerate(value):
                    if key_cfg is not None and isinstance(key_cfg, list) and i < len(key_cfg):
                        item_cfg = key_cfg[i]
                        component_label = item_cfg.get('label', str(i))
                        row, get_value = _make_paired_row(f"  {component_label}:", item_cfg, v)
                        current_panel.add_child(row)
                        widgets[key].append((None, get_value))
                    else:
                        # --- Fallback: NumberEdit ---
                        row = gui.Horiz(0.25 * em)
                        row.add_child(gui.Label(f"  [{i}]"))
                        row.add_stretch()
                        widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                        widget.double_value = v
                        row.add_child(widget)
                        current_panel.add_child(row)
                        widgets[key].append((widget, lambda ww=widget: ww.double_value))

    # --- HSV hover readout label (updated by on_hover) ---
    hsv_label = gui.Label("HSV: —")
    current_panel.add_child(hsv_label)

    # Tracks whether the initial programmatic on_process() call has completed.
    # Only calls triggered after that (button click, space bar) count as
    # user edits and should set segmenter.was_modified.
    _gui_initialized = [False]

    def on_process():
        """Callback to update parameters and re-run processing."""
        # Mark that the operator applied at least one change in this session,
        # but only for user-triggered calls (not the initial programmatic run).
        if _gui_initialized[0]:
            segmenter.was_modified = True

        # 1. Update params from widgets
        for key, widget_or_list in widgets.items():
            if isinstance(widget_or_list, list):
                segmenter.params[params_key][key] = [get_v() for _, get_v in widget_or_list]
            else:
                _, get_v = widget_or_list
                segmenter.params[params_key][key] = get_v()

        # 2. Rerun the processing function
        result = processing_func(pcd_input, segmenter.params[params_key])

        # 3. Update the scene
        scene.scene.clear_geometry()
        material = rendering.MaterialRecord()

        if is_cluster_step:
            pcd_to_show, state['pcd_processed'] = result
        else:
            pcd_to_show = state['pcd_processed'] = result

        if len(pcd_to_show.points) > 0:
            scene.scene.add_geometry("processed_pcd", pcd_to_show, material)

            # Cache point data for the hover callback.
            state['pcd_colors'] = np.asarray(pcd_to_show.colors).copy()
            state['pcd_points'] = np.asarray(pcd_to_show.points).copy()

            # Only initialise the camera on the very first run; subsequent
            # runs preserve whatever orientation the user has navigated to.
            if not state['camera_set']:
                scene.setup_camera(
                    60,
                    pcd_to_show.get_axis_aligned_bounding_box(),
                    pcd_to_show.get_center(),
                )
                state['camera_set'] = True
        else:
            state['pcd_colors'] = None
            state['pcd_points'] = None

    # --- HSV hover callback ---
    # Strategy: project every cloud point forward into screen space, then
    # find the 2D-nearest to the cursor.  This is simpler and correct:
    # the old "unproject cursor → 3D ray → KD-tree" approach always sampled
    # at centroid depth, so hovering over background gave the same result as
    # hovering over a point.  The screen-space approach naturally returns
    # "no match" when the cursor is over empty background.
    _HOVER_MAX_PX = 15   # screen-pixel radius that counts as "over a point"

    def on_hover(event):
        if event.type != gui.MouseEvent.MOVE:
            return gui.Widget.EventCallbackResult.IGNORED

        pcd_colors = state['pcd_colors']
        pcd_points = state['pcd_points']

        if pcd_colors is None or pcd_points is None or len(pcd_colors) == 0:
            return gui.Widget.EventCallbackResult.IGNORED

        # Pixel coordinates relative to the scene widget.
        mx = event.x - scene.frame.x
        my = event.y - scene.frame.y
        W  = scene.frame.width
        H  = scene.frame.height
        if W <= 0 or H <= 0 or mx < 0 or my < 0 or mx >= W or my >= H:
            return gui.Widget.EventCallbackResult.IGNORED

        # Forward-project all cloud points into screen space (vectorised).
        camera    = scene.scene.camera
        proj_view = np.array(camera.get_projection_matrix()) @ \
                    np.array(camera.get_view_matrix())          # 4×4

        pts_h = np.column_stack([pcd_points,
                                 np.ones(len(pcd_points))])     # N×4
        clip  = pts_h @ proj_view.T                             # N×4

        # Keep only points in front of the camera (positive w and z).
        visible = clip[:, 3] > 0
        if not np.any(visible):
            return gui.Widget.EventCallbackResult.IGNORED

        clip_v = clip[visible]
        ndc_x  =  clip_v[:, 0] / clip_v[:, 3]
        ndc_y  =  clip_v[:, 1] / clip_v[:, 3]

        sx = (ndc_x + 1.0) * 0.5 * W
        sy = (1.0 - ndc_y) * 0.5 * H

        # Nearest visible point to the cursor in screen space.
        dx, dy   = sx - mx, sy - my
        dists_sq = dx * dx + dy * dy
        best_local = int(np.argmin(dists_sq))

        if dists_sq[best_local] > _HOVER_MAX_PX ** 2:
            # Cursor is over background — clear the readout.
            hsv_label.text = "HSV: —"
            return gui.Widget.EventCallbackResult.IGNORED

        actual_idx = int(np.where(visible)[0][best_local])
        rgb = pcd_colors[actual_idx]
        h, s, v = colorsys.rgb_to_hsv(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        h_deg = h * 360.0
        if params_key == 'color_skin_filter':
            h_start, h_end = segmenter.params['color_skin_filter']['hsv_h_range']
            in_h = bool(segmenter._hue_in_range(np.array([h_deg]), h_start, h_end)[0])
            indicator = "✓" if in_h else "✗"
            hsv_label.text = f"H: {h_deg:.1f}° ({indicator})  S: {s:.3f}  V: {v:.3f}"
        else:
            hsv_label.text = f"H: {h_deg:.1f}°  S: {s:.3f}  V: {v:.3f}"

        return gui.Widget.EventCallbackResult.IGNORED

    scene.set_on_mouse(on_hover)

    # --- Add Buttons ---
    process_button = gui.Button("Process")
    process_button.set_on_clicked(on_process)
    current_panel.add_child(process_button)

    continue_button = gui.Button("Continue")
    continue_button.set_on_clicked(gui.Application.instance.quit)
    current_panel.add_child(continue_button)

    # --- Space-bar shortcut → Process ---
    def on_key(key_event):
        if (key_event.key == gui.KeyName.SPACE
                and key_event.type == gui.KeyEvent.DOWN):
            on_process()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    scene.set_on_key(on_key)

    # --- Set window layout and run ---
    def on_layout(layout_context):
        r = w.content_rect
        panel_w = max(1, r.width // 5)
        scene_w = r.width - panel_w
        px = r.get_right() - panel_w

        scene.frame = gui.Rect(r.x, r.y, scene_w, r.height)

        if hue_scene_widget is not None:
            # 1. Top Panel
            try:
                top_pref = panel_top.calc_preferred_size(layout_context, gui.Widget.Constraints())
                top_h = top_pref.height
            except Exception:
                top_h = int(3.5 * em)

            panel_top.frame = gui.Rect(px, r.y, panel_w, top_h)

            # 2. Hue Circle (Perfect Square)
            avail_h = max(10, r.height - top_h)
            bottom_min_h = int(10 * em)  # Reserve space for bottom sliders/buttons
            max_sq = max(10, avail_h - bottom_min_h)

            sq = min(panel_w, max_sq)
            x_offset = px + (panel_w - sq) // 2

            hue_scene_widget.frame = gui.Rect(x_offset, r.y + top_h, sq, sq)

            # 3. Bottom Panel
            bottom_y = r.y + top_h + sq
            panel_bottom.frame = gui.Rect(px, bottom_y, panel_w, max(1, r.height - bottom_y))

            hue_refresh_fn()
        else:
            # Fallback to standard layout if no hue circle is rendered for this step
            try:
                top_pref = panel_top.calc_preferred_size(layout_context, gui.Widget.Constraints())
                top_h = top_pref.height
            except Exception:
                top_h = r.height // 2

            panel_top.frame = gui.Rect(px, r.y, panel_w, r.height)
            # Bottom panel stays at 0-size so it doesn't conflict
            panel_bottom.frame = gui.Rect(px, r.y, 0, 0)

    w.set_on_layout(on_layout)

    on_process()  # Initial run — does NOT set was_modified
    _gui_initialized[0] = True  # Subsequent on_process calls are user-triggered

    gui.Application.instance.run()

    return state['pcd_processed']


# ---------------------------------------------------------------------------
# Public entry point — routes to Filament or legacy based on probe result
# ---------------------------------------------------------------------------

def display_pointcloud_interactive(
        segmenter,
        pcd_input,
        window_name: str,
        params_key: str,
        processing_func: Callable,
        is_cluster_step: bool,
):
    """Launch the interactive GUI for one segmentation step.

    On machines where the Filament OpenGL backend is available the full
    slider GUI (``_display_filament``) is used.  On machines where Filament
    fails to create an OpenGL context (e.g. NVIDIA driver 610.47) the legacy
    ``VisualizerWithKeyCallback`` fallback (``_display_legacy``) is used
    instead and a ``RuntimeWarning`` is emitted.

    Parameters
    ----------
    segmenter:
        An ``ArmSegmentation`` instance whose params and callbacks are used.
    pcd_input:
        The input point cloud for this step.
    window_name:
        Title shown in the GUI window.
    params_key:
        Key into ``segmenter.params`` that this step controls.
    processing_func:
        Callable ``(pcd, params) -> result`` that applies the step.
    is_cluster_step:
        If True, ``processing_func`` returns ``(all_clusters_pcd, arm_pcd)``
        and the second element is taken as the step output.

    Returns
    -------
    open3d.geometry.PointCloud
        The processed point cloud after the user confirms.
    """
    if _is_filament_available():
        return _display_filament(
            segmenter, pcd_input, window_name, params_key,
            processing_func, is_cluster_step,
        )
    else:
        return _display_legacy(
            segmenter, pcd_input, window_name, params_key,
            processing_func, is_cluster_step,
        )


# ---------------------------------------------------------------------------
# Legacy VisualizerWithKeyCallback path (Filament unavailable)
# ---------------------------------------------------------------------------

def _build_param_entries(params: dict, slider_cfg: dict) -> List[dict]:
    """Flatten ``_SLIDER_CONFIGS`` for one pipeline step into a list of tunable
    parameter entries suitable for keyboard navigation.

    Each entry is a dict with:
    - ``params_key``  : the key in ``params`` whose value is written back
    - ``label``       : human-readable display label
    - ``is_hue``      : True for hue-angle parameters (wrap-around arithmetic)
    - ``is_range_lo`` : True if this entry controls the *low* element of a list
    - ``is_range_hi`` : True if this entry controls the *high* element of a list
    - ``list_index``  : index into the list for range/vector parameters; None
                        for scalar parameters
    - ``step``        : small adjustment step
    - ``big_step``    : large adjustment step (≈10× step)
    - ``min_val``     : lower clamp (or None)
    - ``max_val``     : upper clamp (or None)

    Parameters that have no numeric representation (booleans, strings, or
    lists without slider config) are skipped.
    """
    entries: List[dict] = []

    for param_key, value in params.items():
        if isinstance(value, bool):
            continue

        cfg = slider_cfg.get(param_key)

        if isinstance(value, (int, float)):
            if cfg is not None:
                min_val  = cfg.get('min')
                max_val  = cfg.get('max')
                step = cfg.get('step', 1.0 if cfg.get('type') == 'int' else 0.1)
                big_step = (
                    (max_val - min_val) / 10.0
                    if (min_val is not None and max_val is not None)
                    else step * 10.0
                )
                label = cfg.get('label', param_key.replace('_', ' ').title())
            else:
                min_val  = None
                max_val  = None
                step     = 1.0 if isinstance(value, int) else 0.1
                big_step = step * 10.0
                label    = param_key.replace('_', ' ').title()

            entries.append({
                'params_key':  param_key,
                'label':       label,
                'is_hue':      False,
                'is_range_lo': False,
                'is_range_hi': False,
                'list_index':  None,
                'step':        step,
                'big_step':    big_step,
                'min_val':     min_val,
                'max_val':     max_val,
            })

        elif isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            if cfg is None:
                continue

            if cfg.get('is_hue_circle'):
                # Two hue-angle values (h_start, h_end) — both wrap 0–360
                for idx, sub_label in ((0, 'H start'), (1, 'H end')):
                    entries.append({
                        'params_key':  param_key,
                        'label':       sub_label,
                        'is_hue':      True,
                        'is_range_lo': False,
                        'is_range_hi': False,
                        'list_index':  idx,
                        'step':        1.0,
                        'big_step':    10.0,
                        'min_val':     0.0,
                        'max_val':     360.0,
                    })

            elif cfg.get('is_range') and len(value) == 2:
                cfgs = cfg['cfgs']
                lo_cfg, hi_cfg = cfgs[0], cfgs[1]

                def _step_from_cfg(c):
                    return c.get('step', 1.0 if c.get('type') == 'int' else 0.01)

                def _big_from_cfg(c):
                    mn, mx = c.get('min'), c.get('max')
                    s = _step_from_cfg(c)
                    return (mx - mn) / 10.0 if mn is not None and mx is not None else s * 10.0

                entries.append({
                    'params_key':  param_key,
                    'label':       lo_cfg.get('label', 'lo'),
                    'is_hue':      False,
                    'is_range_lo': True,
                    'is_range_hi': False,
                    'list_index':  0,
                    'step':        _step_from_cfg(lo_cfg),
                    'big_step':    _big_from_cfg(lo_cfg),
                    'min_val':     lo_cfg.get('min'),
                    'max_val':     lo_cfg.get('max'),
                })
                entries.append({
                    'params_key':  param_key,
                    'label':       hi_cfg.get('label', 'hi'),
                    'is_hue':      False,
                    'is_range_lo': False,
                    'is_range_hi': True,
                    'list_index':  1,
                    'step':        _step_from_cfg(hi_cfg),
                    'big_step':    _big_from_cfg(hi_cfg),
                    'min_val':     hi_cfg.get('min'),
                    'max_val':     hi_cfg.get('max'),
                })

            elif isinstance(cfg, list):
                for i, v in enumerate(value):
                    if i < len(cfg):
                        item_cfg = cfg[i]
                        min_val  = item_cfg.get('min')
                        max_val  = item_cfg.get('max')
                        step = item_cfg.get('step', 1.0 if item_cfg.get('type') == 'int' else 0.1)
                        big_step = (
                            (max_val - min_val) / 10.0
                            if (min_val is not None and max_val is not None)
                            else step * 10.0
                        )
                        label = item_cfg.get('label', f"{param_key}[{i}]")
                    else:
                        min_val  = None
                        max_val  = None
                        step     = 1.0 if isinstance(v, int) else 0.1
                        big_step = step * 10.0
                        label    = f"{param_key}[{i}]"

                    entries.append({
                        'params_key':  param_key,
                        'label':       label,
                        'is_hue':      False,
                        'is_range_lo': False,
                        'is_range_hi': False,
                        'list_index':  i,
                        'step':        step,
                        'big_step':    big_step,
                        'min_val':     min_val,
                        'max_val':     max_val,
                    })

    return entries


def _get_entry_value(params: dict, entry: dict):
    """Read the current numeric value for *entry* from *params*."""
    v = params[entry['params_key']]
    if entry['list_index'] is not None:
        return v[entry['list_index']]
    return v


def _set_entry_value(params: dict, entry: dict, new_value) -> None:
    """Write *new_value* for *entry* back into *params*."""
    if entry['list_index'] is not None:
        params[entry['params_key']][entry['list_index']] = new_value
    else:
        params[entry['params_key']] = new_value


def _print_legacy_status(entries: List[dict], params: dict, selected_idx: int) -> None:
    """Print a formatted table of all tunable parameters to the terminal.

    The currently selected entry is highlighted with `` >> `` and a trailing
    ``  <<`` marker so it stands out in a plain-text terminal.
    """
    if not entries:
        print("  [no tunable parameters for this step]")
        return

    print("\n  ── Parameters ─────────────────────────────────────────")
    for i, entry in enumerate(entries):
        value = _get_entry_value(params, entry)
        if isinstance(value, float):
            val_str = f"{value:.4g}"
        else:
            val_str = str(value)

        if i == selected_idx:
            marker = ">>"
            tail   = "  <<"
        else:
            marker = "  "
            tail   = ""

        print(f"  {marker} [{i:2d}]  {entry['label']:<22s} = {val_str}{tail}")
    print("  ────────────────────────────────────────────────────────")


def _print_legacy_help() -> None:
    """Print the keyboard shortcut reference for the legacy visualizer."""
    print(
        "\n"
        "  ── Legacy Visualizer — keyboard controls ────────────────\n"
        "  Up / Down          Select previous / next parameter\n"
        "  Left / Right       Adjust selected parameter by small step\n"
        "  -  /  =            Adjust selected parameter by large step\n"
        "  Space / Enter      Run processing with current parameters\n"
        "  H                  Show this help message\n"
        "  Q / Esc            Accept current result and continue\n"
        "  ─────────────────────────────────────────────────────────\n"
    )


def _display_legacy(
        segmenter,
        pcd_input,
        window_name: str,
        params_key: str,
        processing_func: Callable,
        is_cluster_step: bool,
):
    """Run the legacy ``VisualizerWithKeyCallback`` UI for one segmentation step.

    Used as a fallback when the Filament OpenGL backend is unavailable.
    Parameter tuning is entirely keyboard-driven: Up/Down to select a
    parameter, Left/Right (or -/=) to adjust it, Space/Enter to apply,
    Q/Esc to accept and continue.

    All ``open3d`` imports happen inside this function so batch-mode callers
    never load any GUI/OpenGL stack.
    """
    import open3d as o3d

    step_params = segmenter.params[params_key]
    slider_cfg  = segmenter._SLIDER_CONFIGS.get(params_key, {})
    entries     = _build_param_entries(step_params, slider_cfg)

    # Mutable state shared by all key callbacks (captured by reference via list).
    state = {
        'selected':     0,
        'pcd_current':  None,
        'done':         False,
        'initialized':  False,
    }

    def _run_processing():
        result = processing_func(pcd_input, segmenter.params[params_key])
        if is_cluster_step:
            _all_clusters, arm_pcd = result
            return arm_pcd
        return result

    def _update_vis(vis, new_pcd):
        vis.clear_geometries()
        state['pcd_current'] = new_pcd
        if new_pcd is not None and len(new_pcd.points) > 0:
            vis.add_geometry(new_pcd, reset_bounding_box=False)
            vis.update_geometry(new_pcd)
        vis.update_renderer()

    def _adjust(delta: float) -> None:
        if not entries:
            return
        idx   = state['selected']
        entry = entries[idx]
        cur   = _get_entry_value(segmenter.params[params_key], entry)

        if entry['is_hue']:
            # Hue wraps 0–360 modularly.
            new_val = (cur + delta) % 360.0
        else:
            new_val = cur + delta
            if entry['min_val'] is not None:
                new_val = max(entry['min_val'], new_val)
            if entry['max_val'] is not None:
                new_val = min(entry['max_val'], new_val)

            # Enforce lo ≤ hi for range pairs.
            if entry['is_range_lo']:
                hi_val = segmenter.params[params_key][entry['params_key']][1]
                new_val = min(new_val, hi_val)
            elif entry['is_range_hi']:
                lo_val = segmenter.params[params_key][entry['params_key']][0]
                new_val = max(new_val, lo_val)

        _set_entry_value(segmenter.params[params_key], entry, new_val)
        _print_legacy_status(entries, segmenter.params[params_key], state['selected'])

    # --- Key callbacks ---

    def _on_up(vis):
        if entries:
            state['selected'] = (state['selected'] - 1) % len(entries)
        _print_legacy_status(entries, segmenter.params[params_key], state['selected'])

    def _on_down(vis):
        if entries:
            state['selected'] = (state['selected'] + 1) % len(entries)
        _print_legacy_status(entries, segmenter.params[params_key], state['selected'])

    def _on_left(vis):
        if entries:
            _adjust(-entries[state['selected']]['step'])

    def _on_right(vis):
        if entries:
            _adjust(+entries[state['selected']]['step'])

    def _on_minus(vis):
        if entries:
            _adjust(-entries[state['selected']]['big_step'])

    def _on_equals(vis):
        if entries:
            _adjust(+entries[state['selected']]['big_step'])

    def _on_process(vis):
        if state['initialized']:
            segmenter.was_modified = True
        new_pcd = _run_processing()
        _update_vis(vis, new_pcd)
        _print_legacy_status(entries, segmenter.params[params_key], state['selected'])

    def _on_quit(vis):
        state['done'] = True
        vis.close()

    def _on_help(vis):
        _print_legacy_help()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)

    vis.register_key_callback(_GLFW_KEY_UP,    _on_up)
    vis.register_key_callback(_GLFW_KEY_DOWN,  _on_down)
    vis.register_key_callback(_GLFW_KEY_LEFT,  _on_left)
    vis.register_key_callback(_GLFW_KEY_RIGHT, _on_right)
    vis.register_key_callback(45,              _on_minus)   # ASCII '-'
    vis.register_key_callback(61,              _on_equals)  # ASCII '='
    vis.register_key_callback(32,              _on_process) # ASCII Space
    vis.register_key_callback(_GLFW_KEY_ENTER, _on_process)
    vis.register_key_callback(_GLFW_KEY_ESC,   _on_quit)
    vis.register_key_callback(ord('Q'),        _on_quit)
    vis.register_key_callback(ord('H'),        _on_help)

    # Initial processing run
    initial_pcd = _run_processing()
    state['pcd_current'] = initial_pcd
    if initial_pcd is not None and len(initial_pcd.points) > 0:
        vis.add_geometry(initial_pcd)
    else:
        print(f"[{window_name}] Empty point cloud — no points to display.")

    state['initialized'] = True
    _print_legacy_help()
    _print_legacy_status(entries, segmenter.params[params_key], state['selected'])

    vis.run()
    vis.destroy_window()

    return state['pcd_current']

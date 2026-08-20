"""Interactive viewer for a precomputed per-vertex contact depth field.

Filter ``[3]`` of the contact-depth pipe: a **pure sink**.  It receives a
finished field series, context geometry and a *precomputed* colour range, and
draws them.  It computes nothing.

Purity contract
---------------
This module must not know about Open3D, file paths, session identity, how the
depth field was computed, or **any statistic it would have to derive itself**.
In particular ``clim_penetration_mm`` arrives from the producer: per-frame
autoscaling would make the animation lie about relative depth (guide 02 §9),
and the per-frame maximum shown in the readout is carried on
:class:`ContactDepthFrameView`, not recomputed here.  Context geometry arrives
as :class:`pyvista.PolyData`, so the Open3D boundary is crossed by the caller,
never by this module.

If a number is needed that the input does not carry, the correct fix is to
extend the producer's contract — not to compute it here.

Design invariants (inherited; each one is a paid-for bug)
--------------------------------------------------------
* **The** ``QtInteractor`` **is the sole central widget.**  All controls live in
  a bottom :class:`QDockWidget`.  VTK's OpenGL window captures the mouse at OS
  level, so any widget sharing its layout cell becomes unclickable and no
  ``raise_()`` / translucency / focus trick fixes it.
  (``note-rf-explorer-post-layout-bugfix-status.md``)
* ``_update_frame()`` **never calls** ``plotter.clear()``, ``add_mesh()`` or
  ``remove_actor()``.  Every actor is registered once in ``_init_actors()`` and
  the datasets are mutated in place.  Re-adding a mesh per frame re-enters
  PyVista's scalar-bar range logic, which does not preserve a global ``clim``
  (see the version note below).
* ``renderer.ResetCameraClippingRange()`` runs before every ``render()``.  VTK
  derives its near/far planes from the aggregate scene bounds, and an in-place
  dataset swap that changes those bounds otherwise leaves geometry silently
  clipped away.  (``bug-neural-kinect-viewer-initial-render.md``)
* The first render is **deferred** to ``showEvent`` →
  ``QTimer.singleShot(0, ...)``, because before the event loop turns the VTK
  render window has zero pixel size.
* Colourmap is ``inferno``.  ``jet`` has non-monotonic lightness, invents
  boundaries the data does not contain, and is hostile to colour-vision
  deficiency.  (``investigation-jet-colormap-perceptual-problems.md``)
* No dependency on ``rf_camera_settings.json``.  The camera comes from the
  scene bounds.  (``note-rf-camera-settings-connections.md``)

PyVista version note (measured on 0.47.1, this environment)
-----------------------------------------------------------
The knowledge base records, against PyVista 0.46.1, that a ``remove_actor`` +
``add_mesh`` cycle makes the cached ``clim`` *only ever expand*, so one
degenerate frame permanently corrupts the lookup table.  Re-measured on the
installed 0.47.1 the failure mode has **changed, not disappeared**: with no
explicit ``clim`` the mapper now tracks each frame's own data range (plain
per-frame autoscale), which lies about relative depth just as badly.  With an
explicit ``clim`` the range is held across in-place updates, ``DeepCopy``
resizes and empty frames.  The mitigation is therefore unchanged and is what
this module does: pass a global ``clim`` once, mutate in place, and re-assert
``actor.mapper.scalar_range`` after every dataset swap.

Also measured on 0.47.1: ``add_mesh`` raises ``ValueError`` on a zero-point
mesh unless ``allow_empty_mesh`` is set, and ``pv.Sphere()`` hard-crashes this
environment's interpreter (delay-load DLL failure ``0xC06D007F``) — hence the
``pv.Box`` bounds proxy and no sphere markers.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# PyVista resolves its Qt binding from QT_API.  Both halves of the process must
# agree on PyQt5 or widget types cross and construction fails with an
# "unexpected type" error.  Set before `pyvistaqt` is imported.
os.environ.setdefault("QT_API", "pyqt5")

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

__all__ = [
    "CONTACT_SCALAR_NAME",
    "ContactDepthFieldViewer",
    "ContactDepthFrameView",
    "polydata_from_triangle_arrays",
]


#: Name of the point-data array the contact actor is mapped to.  Defined once;
#: the mapper, the in-place updates and the scalar bar all refer to this.
CONTACT_SCALAR_NAME: str = "penetration_depth_mm"

#: Colourbar title.  Penetration depth is ``-signed_depth_mm``: positive, and
#: larger means deeper into the forearm.
SCALAR_BAR_TITLE: str = "Penetration depth (mm)"

#: Perceptually uniform, monotonic in lightness, CVD-safe, greyscale-safe.
COLORMAP: str = "inferno"

# Slider-drag throttle.  Measured on ST14-01/block-order-01 at 1600x950, one
# frame costs ~2 ms of dataset updates and ~62 ms inside VTK's render (the
# translucent hand forces the depth-sorted transparency pass).  80 ms is the
# top of the plan's 30-80 ms band and the value that keeps at most one render
# outstanding instead of building a backlog behind the drag.
_DRAG_DEBOUNCE_MS: int = 80
_PLAY_INTERVAL_MS: int = 33  # Requests the Kinect's 30 fps; VTK is the limiter.
_DEFAULT_CONTACT_POINT_SIZE: float = 12.0
_HAND_OPACITY: float = 0.30
_FOREARM_COLOR: str = "lightgray"
_HAND_COLOR: str = "lightsteelblue"


# =============================================================================
# Input contract
# =============================================================================


@dataclass(frozen=True)
class ContactDepthFrameView:
    """One frame as the viewer consumes it.

    The producer owns every semantic decision recorded here, including how the
    two field-absent outcomes are *named* and *coloured*.  The viewer renders
    what it is given, which is what makes it structurally incapable of
    collapsing "no contact" into "pose absent".

    Attributes:
        frame_index: Kinect frame index.
        time_s: Frame timestamp in seconds.
        status_label: Display text for this frame's outcome, e.g. ``"CONTACT"``,
            ``"NO CONTACT"``, ``"POSE ABSENT"``.  Must be non-empty.
        status_color: Colour the readout uses for ``status_label``.  Supplied by
            the producer so distinct outcomes stay visually distinct.
        points: ``(N, 3)`` contact-patch vertex positions in millimetres, or
            ``None`` when this frame carries no field.
        penetration_depth_mm: ``(N,)`` positive-is-deeper penetration depths,
            index-aligned with :attr:`points`, or ``None``.  Already sign-flipped
            by the producer; the viewer performs no arithmetic on it.
        max_penetration_depth_mm: The frame's own maximum penetration, supplied
            by the producer, or ``None`` when there is no field.

    Raises:
        ValueError: If the field arrays are inconsistent with each other or
            with the presence of a field.
    """

    frame_index: int
    time_s: float
    status_label: str
    status_color: str
    points: Optional[np.ndarray] = None
    penetration_depth_mm: Optional[np.ndarray] = None
    max_penetration_depth_mm: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.status_label:
            raise ValueError(
                f"Frame {self.frame_index}: status_label is empty. Every frame must "
                "state its outcome on screen; an unlabelled frame would make "
                "'no contact' and 'pose absent' indistinguishable."
            )
        if not self.status_color:
            raise ValueError(f"Frame {self.frame_index}: status_color is empty.")

        has_points = self.points is not None
        has_depths = self.penetration_depth_mm is not None
        if has_points != has_depths:
            raise ValueError(
                f"Frame {self.frame_index}: points and penetration_depth_mm must be "
                f"present or absent together (got {has_points} / {has_depths})."
            )
        if has_points != (self.max_penetration_depth_mm is not None):
            raise ValueError(
                f"Frame {self.frame_index}: max_penetration_depth_mm must accompany "
                "a field and must be absent without one."
            )
        if not has_points:
            return

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"Frame {self.frame_index}: points must be (N, 3), got "
                f"{self.points.shape}."
            )
        if len(self.points) != len(self.penetration_depth_mm):
            raise ValueError(
                f"Frame {self.frame_index}: {len(self.points)} points vs "
                f"{len(self.penetration_depth_mm)} depths; the field must be "
                "index-aligned with the contact points."
            )
        if len(self.points) == 0:
            raise ValueError(
                f"Frame {self.frame_index}: an empty field is not a field. A frame "
                "with no contact must carry points=None and say so in status_label."
            )

    @property
    def has_field(self) -> bool:
        """Whether this frame carries a contact depth field."""
        return self.points is not None

    @property
    def vertex_count(self) -> int:
        """Number of contact-patch vertices; ``0`` when there is no field."""
        return 0 if self.points is None else len(self.points)


def polydata_from_triangle_arrays(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> pv.PolyData:
    """Build a :class:`pyvista.PolyData` surface from vertex and triangle arrays.

    Provided here so callers can convert their own mesh representation (Open3D,
    trimesh, raw arrays) into the viewer's input type without this module ever
    importing those libraries.

    Args:
        vertices: ``(V, 3)`` vertex positions.
        triangles: ``(T, 3)`` vertex indices.

    Returns:
        The equivalent triangular surface.

    Raises:
        ValueError: If either array has the wrong shape, or the mesh is empty.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(triangles, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {verts.shape}.")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"triangles must be (T, 3), got {tris.shape}.")
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError(
            f"Cannot build a surface from {len(verts)} vertices and {len(tris)} "
            "triangles; context geometry must be a real mesh."
        )
    faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    return pv.PolyData(verts, faces)


# =============================================================================
# The viewer
# =============================================================================


class ContactDepthFieldViewer(QMainWindow):
    """Scrub a recording's per-vertex contact depth field in 3D.

    The window shows the forearm terrain (opaque grey) and the hand
    (translucent) as anatomical context, with the contact patch drawn on top as
    points coloured by penetration depth against a **fixed, global** colour
    scale.  A frame slider, a play/pause button and an on-screen readout of
    frame index, time, outcome and contact-vertex count complete it.

    Args:
        frames: One :class:`ContactDepthFrameView` per frame, in frame order.
        forearm_meshes_by_frame: Static forearm terrain keyed by the frame index
            from which it takes effect.  Key ``0`` is required, mirroring the
            producer's own guarantee.
        hand_mesh_provider: ``frame_index -> hand surface or None``.  Returning
            ``None`` hides the hand actor, which is how an absent pose reads on
            screen.  Called once per displayed frame, so it must be cheap; it
            must not compute depth.
        clim_penetration_mm: ``(low, high)`` colour limits in penetration
            millimetres, computed **once** over the whole recording by the
            producer, or ``None`` when the recording contains no contact at all.
        recording_label: Text identifying the recording, shown in the readout.
        contact_point_size: Initial rendered size of a contact vertex.
        window_title: Window title.

    Raises:
        ValueError: If ``frames`` is empty, if the forearm dictionary has no
            key ``0``, or if ``clim_penetration_mm`` disagrees with whether the
            series actually contains any field.
    """

    def __init__(
        self,
        *,
        frames: Sequence[ContactDepthFrameView],
        forearm_meshes_by_frame: Dict[int, pv.PolyData],
        hand_mesh_provider: Callable[[int], Optional[pv.PolyData]],
        clim_penetration_mm: Optional[Tuple[float, float]],
        recording_label: str,
        contact_point_size: float = _DEFAULT_CONTACT_POINT_SIZE,
        window_title: str = "Contact depth field",
    ) -> None:
        super().__init__()

        self._frames: Tuple[ContactDepthFrameView, ...] = tuple(frames)
        self._forearm_meshes = dict(forearm_meshes_by_frame)
        self._hand_mesh_provider = hand_mesh_provider
        self._clim = clim_penetration_mm
        self._recording_label = recording_label
        self._contact_point_size = float(contact_point_size)

        self._validate_inputs()

        self._total_frames = len(self._frames)
        self._forearm_keys: List[int] = sorted(self._forearm_meshes)
        self.current_index: int = 0

        # --- Playback and drag throttling -----------------------------------
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_advance)

        self._drag_timer = QTimer(self)
        self._drag_timer.timeout.connect(self._on_drag_timer_fired)
        self._drag_timer.setInterval(_DRAG_DEBOUNCE_MS)
        self._slider_dragging: bool = False
        self._pending_drag_frame: Optional[int] = None

        # Before the event loop turns, the VTK render window has zero pixels;
        # the first draw is deferred to showEvent.
        self._initial_render_done: bool = False

        # --- Actor state ----------------------------------------------------
        self._active_forearm_key: Optional[int] = None
        self._hand_triangle_count: int = 0
        self._bounds_proxy_active: bool = False
        self._visible = {"forearm": True, "hand": True, "contact": True}

        self.setWindowTitle(window_title)
        self.resize(1600, 950)
        self._build_ui()
        self._init_actors()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Reject an input set the viewer could only render misleadingly."""
        if not self._frames:
            raise ValueError("frames is empty; there is nothing to display.")
        if 0 not in self._forearm_meshes:
            raise ValueError(
                "forearm_meshes_by_frame has no key 0, so frame 0 has no terrain "
                f"in effect; keys present: {sorted(self._forearm_meshes)}."
            )

        any_field = any(frame.has_field for frame in self._frames)
        if any_field and self._clim is None:
            raise ValueError(
                "clim_penetration_mm is None but the series contains contact "
                "frames. The colour scale must be supplied by the producer; this "
                "viewer will not invent one from the frames it happens to hold."
            )
        if not any_field and self._clim is not None:
            raise ValueError(
                f"clim_penetration_mm is {self._clim} but no frame carries a field. "
                "A colour scale over data that does not exist would be fiction."
            )
        if self._clim is not None:
            low, high = self._clim
            if not (np.isfinite(low) and np.isfinite(high)):
                raise ValueError(f"clim_penetration_mm is not finite: {self._clim}.")
            if high < low:
                raise ValueError(
                    f"clim_penetration_mm is inverted: {self._clim} (high < low)."
                )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Lay out the window: interactor central, every control in a bottom dock.

        The ``QtInteractor`` is the sole central widget.  VTK's OpenGL surface
        grabs mouse events at OS level, so a control sharing its layout cell
        becomes permanently unclickable — a docked bar is the only arrangement
        that reliably works here.
        """
        plotter_host = QWidget()
        plotter_layout = QVBoxLayout(plotter_host)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(plotter_host)
        self.plotter.set_background("black")
        # add_mesh refuses a zero-point dataset on PyVista >= 0.47 unless this is
        # set.  Scoped to this plotter's theme, never the global theme.
        self.plotter.theme.allow_empty_mesh = True
        plotter_layout.addWidget(self.plotter.interactor)
        self.setCentralWidget(plotter_host)

        dock = QDockWidget("Playback", self)
        dock.setObjectName("playback_dock")
        dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(self._build_control_bar())
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _build_control_bar(self) -> QWidget:
        """Return the bottom bar: play/pause, slider, counters, visibility toggles."""
        bar = QWidget()
        layout = QHBoxLayout(bar)

        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(80)
        self.play_button.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_button)

        layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(len(self._frames) - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.frame_slider, stretch=1)

        self.frame_label = QLabel(f"1 / {len(self._frames)}")
        self.frame_label.setMinimumWidth(110)
        layout.addWidget(self.frame_label)

        # The denominator behind the colour: a 3-vertex patch and a 300-vertex
        # patch must not read as equally authoritative (guide 03 §7).
        self.vertex_label = QLabel("vertices: -")
        self.vertex_label.setMinimumWidth(120)
        layout.addWidget(self.vertex_label)

        for key, text in (
            ("forearm", "Forearm"),
            ("hand", "Hand"),
            ("contact", "Contact"),
        ):
            box = QCheckBox(text)
            box.setChecked(True)
            box.stateChanged.connect(
                lambda state, k=key: self._on_visibility_changed(k, state)
            )
            layout.addWidget(box)

        reset_button = QPushButton("Reset view")
        reset_button.clicked.connect(self._reset_view)
        layout.addWidget(reset_button)

        return bar

    # ------------------------------------------------------------------
    # Actor initialisation — called once; datasets mutated in place after
    # ------------------------------------------------------------------

    def _init_actors(self) -> None:
        """Register every actor exactly once.

        After this call ``_update_frame()`` only mutates datasets and toggles
        actor visibility.  Nothing is added or removed per frame (except the
        one-shot bounds proxy), which is what keeps the global colour scale
        intact.
        """
        # --- Context: forearm terrain (opaque grey) -------------------------
        self._mesh_forearm = pv.PolyData()
        self._mesh_forearm.DeepCopy(self._forearm_meshes[0])
        self._active_forearm_key = 0
        self._actor_forearm = self.plotter.add_mesh(
            self._mesh_forearm,
            color=_FOREARM_COLOR,
            name="forearm",
            opacity=1.0,
            copy_mesh=False,
        )

        # --- Context: hand (translucent, so the patch beneath stays visible) -
        self._mesh_hand = pv.PolyData(np.empty((0, 3), dtype=np.float32))
        self._actor_hand = self.plotter.add_mesh(
            self._mesh_hand,
            color=_HAND_COLOR,
            name="hand",
            opacity=_HAND_OPACITY,
            copy_mesh=False,
        )
        self._hand_triangle_count = -1  # force a rebuild on the first update
        self._actor_hand.VisibilityOff()

        # --- The field itself ------------------------------------------------
        self._mesh_contact = _empty_contact_polydata()
        if self._clim is None:
            # No contact anywhere in the recording.  Registering a scalar bar
            # over a range that does not exist would be fiction, so the actor
            # and its colourbar are simply not created.
            self._actor_contact = None
        else:
            self._actor_contact = self.plotter.add_mesh(
                self._mesh_contact,
                scalars=CONTACT_SCALAR_NAME,
                cmap=COLORMAP,
                clim=self._clim,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": SCALAR_BAR_TITLE,
                    "vertical": True,
                    "n_labels": 6,
                    "fmt": "%.2f",
                    "title_font_size": 16,
                    "label_font_size": 13,
                    # Without an explicit colour the bar's text inherits the
                    # theme's default black and vanishes against the black
                    # background — the bar renders, its title and tick labels
                    # do not.
                    "color": "white",
                    "position_x": 0.85,
                    "position_y": 0.12,
                    "width": 0.05,
                    "height": 0.72,
                },
                name="contact_field",
                render_points_as_spheres=True,
                point_size=self._contact_point_size,
                copy_mesh=False,
            )

        # --- Decorations ------------------------------------------------------
        self.plotter.add_axes(interactive=False)
        self._readout = self.plotter.add_text(
            "",
            position="upper_left",
            font_size=11,
            color="white",
            name="readout",
        )
        # The outcome gets its own actor so it can be recoloured per frame.  The
        # colour is the producer's, not this module's: it is the second channel
        # (after the label text, and the hand actor's visibility) that keeps "no
        # contact" and "pose absent" from reading as the same fact.
        self._status_readout = self.plotter.add_text(
            "",
            position="upper_right",
            font_size=13,
            color="white",
            name="status_readout",
        )

        # --- Bounds proxy -----------------------------------------------------
        # Defence in depth: the forearm alone already guarantees non-degenerate
        # scene bounds, but a frame-0 scene whose only *dynamic* geometry is
        # empty is exactly the configuration that produced a blank viewport
        # before.  pv.Box, not pv.Sphere: the sphere source hard-crashes this
        # environment's interpreter.
        x0, x1, y0, y1, z0, z1 = self._mesh_forearm.bounds
        pad = 0.5 * max(x1 - x0, y1 - y0, z1 - z0, 1.0)
        self.plotter.add_mesh(
            pv.Box(
                bounds=(x0 - pad, x1 + pad, y0 - pad, y1 + pad, z0 - pad, z1 + pad)
            ),
            opacity=0.001,
            name="_bounds_proxy",
            pickable=False,
        )
        self._bounds_proxy_active = True

        self._reset_view()

    # ------------------------------------------------------------------
    # Frame update — the hot path
    # ------------------------------------------------------------------

    def _update_frame(self, frame_index: int) -> None:
        """Draw ``frame_index``: mutate datasets in place, then render once."""
        frame = self._frames[frame_index]
        self.current_index = frame_index

        self._update_forearm(frame_index)
        self._update_hand(frame_index)
        self._update_contact(frame)
        self._retire_bounds_proxy(frame)
        # Before the render, not after: the on-screen annotation is a VTK actor
        # like any other, so text written after the render would only appear on
        # the *next* one and the readout would trail the geometry by a frame.
        self._update_readout(frame)

        # Unconditional: VTK's near/far planes come from the aggregate scene
        # bounds, which every in-place dataset swap above can change.  This is a
        # strict superset of the required empty <-> non-empty transitions and
        # costs O(n_actors) on a four-actor scene.
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def _update_forearm(self, frame_index: int) -> None:
        """Swap the terrain when a new forearm reference takes effect."""
        position = bisect.bisect_right(self._forearm_keys, frame_index) - 1
        if position < 0:
            raise ValueError(
                f"Frame {frame_index} precedes every forearm reference key "
                f"{self._forearm_keys}; key 0 was validated at construction, so "
                "the key list has been mutated."
            )
        key = self._forearm_keys[position]
        if key != self._active_forearm_key:
            self._mesh_forearm.DeepCopy(self._forearm_meshes[key])
            self._mesh_forearm.Modified()
            self._active_forearm_key = key

        self._actor_forearm.SetVisibility(self._visible["forearm"])

    def _update_hand(self, frame_index: int) -> None:
        """Update the translucent hand, or hide it when the pose is absent."""
        mesh = self._hand_mesh_provider(frame_index)
        if mesh is None or not self._visible["hand"]:
            self._actor_hand.VisibilityOff()
            return

        triangle_count = mesh.n_faces_strict
        if triangle_count == self._hand_triangle_count:
            # Topology unchanged: a points-only update skips the dataset rebuild.
            self._mesh_hand.points = mesh.points
            self._mesh_hand.Modified()
        else:
            self._mesh_hand.DeepCopy(mesh)
            self._mesh_hand.Modified()
            self._hand_triangle_count = triangle_count
        self._actor_hand.VisibilityOn()

    def _update_contact(self, frame: ContactDepthFrameView) -> None:
        """Swap the contact patch, holding the global colour scale fixed."""
        if self._actor_contact is None:
            return

        self._actor_contact.SetVisibility(self._visible["contact"])

        if frame.has_field:
            replacement = pv.PolyData(frame.points.astype(np.float32))
            replacement[CONTACT_SCALAR_NAME] = np.asarray(
                frame.penetration_depth_mm, dtype=np.float64
            )
            replacement.set_active_scalars(CONTACT_SCALAR_NAME)
        else:
            replacement = _empty_contact_polydata()

        self._mesh_contact.DeepCopy(replacement)
        self._mesh_contact.Modified()
        # Re-asserted after every dataset swap: this single line is what keeps
        # the colour of a given depth identical at every frame.  Measured to
        # survive DeepCopy on 0.47.1 already, but it is the invariant the whole
        # visualisation rests on, so it is stated rather than assumed.
        self._actor_contact.mapper.scalar_range = self._clim

    def _retire_bounds_proxy(self, frame: ContactDepthFrameView) -> None:
        """Drop the invisible bounds box once real contact geometry has appeared."""
        if not self._bounds_proxy_active or not frame.has_field:
            return
        self._bounds_proxy_active = False
        self.plotter.remove_actor("_bounds_proxy")

    def _update_readout(self, frame: ContactDepthFrameView) -> None:
        """Refresh the on-screen and bottom-bar readouts.

        Reports the contact-vertex count alongside the depth so a three-vertex
        patch cannot be mistaken for as solid a measurement as a three-hundred
        vertex one, and restates the colour scale as *fixed* so no one reads the
        colours as per-frame relative.
        """
        self._status_readout.SetText(3, frame.status_label)  # 3 = upper-right
        self._status_readout.GetTextProperty().SetColor(
            *pv.Color(frame.status_color).float_rgb
        )

        lines = [
            self._recording_label,
            f"frame {frame.frame_index + 1} / {self._total_frames}"
            f"    t = {frame.time_s:.3f} s",
        ]
        if frame.has_field:
            lines.append(
                f"contact vertices: {frame.vertex_count}"
                f"    max depth this frame: {frame.max_penetration_depth_mm:.3f} mm"
            )
        else:
            lines.append("contact vertices: 0    (no depth field this frame)")

        if self._clim is None:
            lines.append("colour scale: undefined — no contact in this recording")
        else:
            lines.append(
                f"colour scale (fixed, whole recording): "
                f"{self._clim[0]:.3f} to {self._clim[1]:.3f} mm"
            )

        self._readout.SetText(2, "\n".join(lines))  # 2 = upper-left corner
        self.frame_label.setText(f"{frame.frame_index + 1} / {self._total_frames}")
        self.vertex_label.setText(f"vertices: {frame.vertex_count}")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _reset_view(self) -> None:
        """Frame the whole scene isometrically.

        Deliberately derived from the scene bounds rather than from
        ``rf_camera_settings.json``: this viewer works in Kinect Space 1 and has
        no business failing on a downstream calibration artefact the user may
        never have produced.
        """
        self.plotter.view_isometric()
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    # ------------------------------------------------------------------
    # Deferred first render
    # ------------------------------------------------------------------

    def _deferred_start(self) -> None:
        """Initialise the VTK interactor, size it, then draw frame 0."""
        self.plotter.interactor.Initialize()
        size = self.plotter.interactor.size()
        if size.width() > 0 and size.height() > 0:
            self.plotter.render_window.SetSize(size.width(), size.height())
        self._reset_view()
        self._update_frame(0)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming.
        """Draw the first frame only once the window has real pixel dimensions."""
        super().showEvent(event)
        if self._initial_render_done:
            return
        self._initial_render_done = True
        primary = QApplication.primaryScreen()
        if primary is not None:
            self.move(primary.geometry().topLeft())
        QTimer.singleShot(0, self._deferred_start)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        self._pending_drag_frame = None
        self._drag_timer.start()

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self._drag_timer.stop()
        self._pending_drag_frame = None
        self._update_frame(self.frame_slider.value())

    def _on_drag_timer_fired(self) -> None:
        """Render at most one frame per debounce tick while the slider is held."""
        if self._pending_drag_frame is None:
            return
        frame = self._pending_drag_frame
        self._pending_drag_frame = None
        self._update_frame(frame)

    def _on_slider_change(self, value: int) -> None:
        if self._slider_dragging:
            self.frame_label.setText(f"{value + 1} / {self._total_frames}")
            self._pending_drag_frame = value
        else:
            self._update_frame(value)

    def _on_visibility_changed(self, key: str, state: int) -> None:
        self._visible[key] = state == Qt.Checked
        self._update_frame(self.current_index)

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("Play")
        else:
            self._play_timer.start(_PLAY_INTERVAL_MS)
            self.play_button.setText("Pause")

    def _play_advance(self) -> None:
        """Advance via the slider so there is exactly one frame-update path."""
        self.frame_slider.setValue((self.current_index + 1) % self._total_frames)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming.
        """Stop the timers and release the VTK render window."""
        self._play_timer.stop()
        self._drag_timer.stop()
        self.plotter.close()
        super().closeEvent(event)


def _empty_contact_polydata() -> pv.PolyData:
    """A zero-point dataset that still carries the mapped scalar array.

    The array must exist even when empty, or the mapper loses its binding to
    :data:`CONTACT_SCALAR_NAME` the first time a no-contact frame is shown.
    """
    mesh = pv.PolyData(np.empty((0, 3), dtype=np.float32))
    mesh[CONTACT_SCALAR_NAME] = np.empty((0,), dtype=np.float64)
    return mesh

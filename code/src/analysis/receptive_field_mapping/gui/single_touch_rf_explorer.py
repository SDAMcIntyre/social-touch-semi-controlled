"""Single-Touch RF Explorer — data model, loader, and viewer.

Provides ``SingleTouchRFViewerData``, ``load_single_touch_rf_data()``, and
``SingleTouchRFExplorer``.  The dataclass and loader read the sparse per-touch
RF ``.npz`` files produced by ``rf_single_touch_pipeline.py::run_single_touch_rf_mapping()``.
The viewer renders them as heatmaps on a 3D forearm point cloud with cascading
Session / Block / Trial / Touch toolbar dropdowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMainWindow,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.rf_data_loader import (
    load_forearm_vertex_colors,
    load_forearm_vertices,
)

logger = logging.getLogger(__name__)

_VALID_NEURON_MODES = ("iff", "spike")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SingleTouchRFViewerData:
    """All data needed by the Single-Touch RF Explorer viewer.

    Attributes
    ----------
    session_id:
        Unique session identifier string.
    forearm_vertices:
        ``(V, 3)`` float64 array of forearm PLY vertex coordinates.
    forearm_vertex_colors:
        ``(V, 3)`` uint8 RGB array, or ``None`` if the PLY has no colours.
    touch_id_map:
        Mapping from ``(block_order_id, trial_id, single_touch_id)`` 3-tuples
        of Python types ``(str, int, int)`` to integer touch IDs used as keys
        into ``rf_data``.
    rf_data:
        Mapping from integer touch ID to a (possibly empty) list of
        ``(vertex_idx: int, mean_value: float)`` pairs — the sparse RF map
        for that touch.
    neuron_mode:
        Either ``"iff"`` or ``"spike"``, as stored in the ``.npz`` file.
    block_order_ids:
        Sorted list of unique ``block_order_id`` strings appearing in
        ``touch_id_map``.
    trial_ids_by_block:
        ``block_order_id`` → sorted list of ``trial_id`` integers.
    touches_by_block_trial:
        ``(block_order_id, trial_id)`` → sorted list of ``single_touch_id``
        integers (i.e. the third element of the ``touch_id_map`` key).
    session_max_value:
        Maximum ``mean_value`` across all touches and all contacted vertices.
        Used as the upper colour limit for stable cross-touch comparison.
        Set to ``1.0`` when no contacted vertices exist (all-empty session).
    """

    session_id: str
    forearm_vertices: np.ndarray               # (V, 3) float64
    forearm_vertex_colors: Optional[np.ndarray]  # (V, 3) uint8, or None
    touch_id_map: dict                          # (str, int, int) -> int
    rf_data: dict                               # int -> list[(int, float)]
    neuron_mode: str                            # "iff" or "spike"
    block_order_ids: list                       # sorted unique str
    trial_ids_by_block: dict                    # str -> sorted list[int]
    touches_by_block_trial: dict                # (str, int) -> sorted list[int]
    session_max_value: float                    # max mean_value; >=1.0 if no contacts


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_single_touch_rf_data(
    npz_path: Path,
    forearm_ply_path: Path,
    session_id: str,
) -> SingleTouchRFViewerData:
    """Load per-touch RF maps from an ``.npz`` file into ``SingleTouchRFViewerData``.

    Parameters
    ----------
    npz_path:
        Path to ``single_touch_rf_maps.npz`` produced by
        ``run_single_touch_rf_mapping()``.
    forearm_ply_path:
        Path to the forearm PLY file for this session.  Used to load vertex
        coordinates and (optionally) vertex colours.
    session_id:
        Session identifier string, stored verbatim on the returned dataclass.

    Returns
    -------
    ``SingleTouchRFViewerData`` with fully reconstructed hierarchy and
    a ``session_max_value`` that is always ``>= 1.0`` when all RF data is
    empty (fail-fast guarantee: the field is never ``0``).

    Raises
    ------
    ValueError
        If ``npz_path`` does not exist, required keys are missing, the
        ``neuron_mode`` field holds an unrecognised value, or the forearm PLY
        cannot be loaded.
    """
    npz_path = Path(npz_path)
    forearm_ply_path = Path(forearm_ply_path)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not npz_path.exists():
        raise ValueError(
            f"load_single_touch_rf_data: npz_path does not exist: {npz_path}"
        )
    if not forearm_ply_path.exists():
        raise ValueError(
            f"load_single_touch_rf_data: forearm_ply_path does not exist: "
            f"{forearm_ply_path}"
        )

    # ------------------------------------------------------------------
    # Load .npz
    # ------------------------------------------------------------------
    raw = np.load(npz_path, allow_pickle=True)

    required_keys = ("touch_id_map", "rf_data", "neuron_mode")
    missing = [k for k in required_keys if k not in raw]
    if missing:
        raise ValueError(
            f"load_single_touch_rf_data: npz_path is missing required keys "
            f"{missing}: {npz_path}"
        )

    # Dicts are stored as 0-d object arrays; unwrap with .item()
    touch_id_map_raw: dict = raw["touch_id_map"].item()
    rf_data_raw: dict = raw["rf_data"].item()
    neuron_mode: str = str(raw["neuron_mode"])

    if neuron_mode not in _VALID_NEURON_MODES:
        raise ValueError(
            f"load_single_touch_rf_data: unrecognised neuron_mode {neuron_mode!r} "
            f"in {npz_path}. Expected one of {_VALID_NEURON_MODES}."
        )

    # ------------------------------------------------------------------
    # Normalize touch_id_map keys to (str, int, int)
    #
    # numpy pickle may preserve numpy scalar types (numpy.str_, numpy.int64,
    # etc.) rather than plain Python types.  Normalise explicitly so that
    # downstream dict lookups with plain-Python keys always succeed.
    # ------------------------------------------------------------------
    touch_id_map: dict = {}
    for raw_key, touch_int_id in touch_id_map_raw.items():
        if len(raw_key) != 3:
            raise ValueError(
                f"load_single_touch_rf_data: touch_id_map key {raw_key!r} "
                f"has unexpected length {len(raw_key)} (expected 3): {npz_path}"
            )
        block_order_id = str(raw_key[0])
        trial_id = int(raw_key[1])
        single_touch_id = int(raw_key[2])
        touch_id_map[(block_order_id, trial_id, single_touch_id)] = int(touch_int_id)

    # Normalize rf_data keys to plain int
    rf_data: dict = {}
    for raw_touch_id, pairs in rf_data_raw.items():
        norm_id = int(raw_touch_id)
        # pairs is a list of (vertex_idx, mean_value); normalize element types.
        # NaN values are kept here so the viewer can distinguish "touched but no
        # neural data" from "no contact" via the info label; rendering filters
        # them out so they don't tint the cloud uniformly via nan_color.
        normalized_pairs = [(int(vi), float(mv)) for vi, mv in pairs]
        rf_data[norm_id] = normalized_pairs

    # ------------------------------------------------------------------
    # Reconstruct block / trial / touch hierarchy
    # ------------------------------------------------------------------
    block_set: set[str] = set()
    trial_map: dict[str, set[int]] = {}
    touch_map: dict[tuple, set[int]] = {}

    for (block_order_id, trial_id, single_touch_id) in touch_id_map:
        block_set.add(block_order_id)

        if block_order_id not in trial_map:
            trial_map[block_order_id] = set()
        trial_map[block_order_id].add(trial_id)

        bt_key = (block_order_id, trial_id)
        if bt_key not in touch_map:
            touch_map[bt_key] = set()
        touch_map[bt_key].add(single_touch_id)

    # Sort block_order_ids numerically when possible, lexicographically otherwise.
    def _block_sort_key(bid: str) -> tuple:
        try:
            return (0, int(bid))
        except ValueError:
            return (1, bid)

    block_order_ids: list[str] = sorted(block_set, key=_block_sort_key)
    trial_ids_by_block: dict[str, list[int]] = {
        bid: sorted(trial_map[bid]) for bid in block_order_ids
    }
    touches_by_block_trial: dict[tuple, list[int]] = {
        bt_key: sorted(touch_ids) for bt_key, touch_ids in touch_map.items()
    }

    # ------------------------------------------------------------------
    # Compute session_max_value
    # ------------------------------------------------------------------
    session_max_value: float = 0.0
    for pairs in rf_data.values():
        for _vertex_idx, mean_value in pairs:
            if mean_value > session_max_value:
                session_max_value = mean_value

    # Edge case: all touches have no contacted vertices (empty RF data).
    # Use fallback clim (0, 1) so the viewer colour bar remains valid.
    if session_max_value == 0.0:
        logger.warning(
            "load_single_touch_rf_data: session %r has no contacted vertices "
            "in any touch — using fallback session_max_value=1.0.",
            session_id,
        )
        session_max_value = 1.0

    # ------------------------------------------------------------------
    # Load forearm geometry
    # ------------------------------------------------------------------
    forearm_vertices = load_forearm_vertices(forearm_ply_path)
    if forearm_vertices is None:
        raise ValueError(
            f"load_single_touch_rf_data: could not load forearm vertices from "
            f"{forearm_ply_path} for session {session_id!r}."
        )

    forearm_vertex_colors = load_forearm_vertex_colors(forearm_ply_path)

    # ------------------------------------------------------------------
    # Assemble and return
    # ------------------------------------------------------------------
    logger.info(
        "load_single_touch_rf_data: session=%r | %d blocks | %d touches | "
        "session_max_value=%.4f | neuron_mode=%r",
        session_id,
        len(block_order_ids),
        len(touch_id_map),
        session_max_value,
        neuron_mode,
    )

    return SingleTouchRFViewerData(
        session_id=session_id,
        forearm_vertices=forearm_vertices,
        forearm_vertex_colors=forearm_vertex_colors,
        touch_id_map=touch_id_map,
        rf_data=rf_data,
        neuron_mode=neuron_mode,
        block_order_ids=block_order_ids,
        trial_ids_by_block=trial_ids_by_block,
        touches_by_block_trial=touches_by_block_trial,
        session_max_value=session_max_value,
    )


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class SingleTouchRFExplorer(QMainWindow):
    """QMainWindow with a single 3D forearm view for per-touch RF heatmaps.

    Toolbar provides cascading Session / Block / Trial / Touch dropdowns.
    Selecting a touch renders its pre-computed RF heatmap (sparse vertex-value
    pairs from the ``.npz`` file) onto the forearm point cloud with a stable
    per-session colour scale.

    Parameters
    ----------
    data:
        Pre-loaded data for the initial session.
    sessions:
        Optional list of ``(label, SingleTouchRFViewerData)`` tuples for the
        session selector.  If ``None``, defaults to
        ``[("{session_id}", data)]``.
    title:
        Window title override.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        data: SingleTouchRFViewerData,
        sessions: Optional[List[Tuple[str, SingleTouchRFViewerData]]] = None,
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if title is None:
            title = f"Single-Touch RF Explorer — {data.session_id}"
        self.setWindowTitle(title)

        self._data = data
        self._sessions: List[Tuple[str, SingleTouchRFViewerData]] = (
            sessions if sessions is not None else [(data.session_id, data)]
        )
        self._initialized = False
        self._cloud: Optional[pv.PolyData] = None
        # Camera state keyed by session_id — preserves view when switching touches.
        self._camera_states: dict[str, object] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        self._plotter = QtInteractor(self)
        self._plotter.interactor.installEventFilter(self)
        root.addWidget(self._plotter.interactor, 1)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Session combo
        toolbar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(160)
        for label, _ in self._sessions:
            self._session_combo.addItem(label)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self._session_combo)

        toolbar.addSeparator()

        # Block combo
        toolbar.addWidget(QLabel("Block:"))
        self._block_combo = QComboBox()
        self._block_combo.setMinimumWidth(80)
        self._block_combo.currentIndexChanged.connect(self._on_block_changed)
        toolbar.addWidget(self._block_combo)

        toolbar.addSeparator()

        # Trial combo
        toolbar.addWidget(QLabel("Trial:"))
        self._trial_combo = QComboBox()
        self._trial_combo.setMinimumWidth(100)
        self._trial_combo.currentIndexChanged.connect(self._on_trial_changed)
        toolbar.addWidget(self._trial_combo)

        toolbar.addSeparator()

        # Touch combo
        toolbar.addWidget(QLabel("Touch:"))
        self._touch_combo = QComboBox()
        self._touch_combo.setMinimumWidth(120)
        self._touch_combo.currentIndexChanged.connect(self._on_touch_changed)
        toolbar.addWidget(self._touch_combo)

        toolbar.addSeparator()

        # Info label — updated whenever a touch is selected.
        self._info_label = QLabel("")
        toolbar.addWidget(self._info_label)

    # ------------------------------------------------------------------
    # Combo cascade helpers
    # ------------------------------------------------------------------

    def _populate_block_combo(self, data: SingleTouchRFViewerData) -> None:
        self._block_combo.blockSignals(True)
        self._block_combo.clear()
        for bid in data.block_order_ids:
            self._block_combo.addItem(f"Block {bid}")
        self._block_combo.blockSignals(False)

        if data.block_order_ids:
            self._populate_trial_combo(data, data.block_order_ids[0])

    def _populate_trial_combo(
        self, data: SingleTouchRFViewerData, block_id: str
    ) -> None:
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        for tid in data.trial_ids_by_block.get(block_id, []):
            self._trial_combo.addItem(f"Trial {tid}")
        self._trial_combo.blockSignals(False)

        trial_ids = data.trial_ids_by_block.get(block_id, [])
        if trial_ids:
            self._populate_touch_combo(data, block_id, trial_ids[0])

    def _populate_touch_combo(
        self,
        data: SingleTouchRFViewerData,
        block_id: str,
        trial_id: int,
    ) -> None:
        self._touch_combo.blockSignals(True)
        self._touch_combo.clear()
        for stid in data.touches_by_block_trial.get((block_id, trial_id), []):
            self._touch_combo.addItem(f"Touch {stid}")
        self._touch_combo.blockSignals(False)

        if self._touch_combo.count() > 0:
            self._touch_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Current-selection helpers
    # ------------------------------------------------------------------

    def _current_block_id(self) -> Optional[str]:
        idx = self._block_combo.currentIndex()
        if idx < 0 or idx >= len(self._data.block_order_ids):
            return None
        return self._data.block_order_ids[idx]

    def _current_trial_id(self) -> Optional[int]:
        bid = self._current_block_id()
        if bid is None:
            return None
        trial_ids = self._data.trial_ids_by_block.get(bid, [])
        idx = self._trial_combo.currentIndex()
        if idx < 0 or idx >= len(trial_ids):
            return None
        return trial_ids[idx]

    def _current_single_touch_id(self) -> Optional[int]:
        bid = self._current_block_id()
        tid = self._current_trial_id()
        if bid is None or tid is None:
            return None
        touch_ids = self._data.touches_by_block_trial.get((bid, tid), [])
        idx = self._touch_combo.currentIndex()
        if idx < 0 or idx >= len(touch_ids):
            return None
        return touch_ids[idx]

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._sessions):
            raise ValueError(
                f"SingleTouchRFExplorer: session index {index} out of range "
                f"(have {len(self._sessions)} sessions)"
            )
        _, new_data = self._sessions[index]
        self._data = new_data
        self._populate_block_combo(new_data)

        if self._initialized:
            self._render_forearm()
            self._select_first_touch()

    def _on_block_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        bid = self._current_block_id()
        if bid is None:
            return
        self._populate_trial_combo(self._data, bid)
        self._select_first_touch()

    def _on_trial_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        bid = self._current_block_id()
        if bid is None:
            return
        trial_ids = self._data.trial_ids_by_block.get(bid, [])
        if index >= len(trial_ids):
            return
        trial_id = trial_ids[index]
        self._populate_touch_combo(self._data, bid, trial_id)
        stid = self._current_single_touch_id()
        if stid is not None:
            self._update_heatmap(stid)

    def _on_touch_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        stid = self._current_single_touch_id()
        if stid is not None:
            self._save_camera_state()
            self._update_heatmap(stid)

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def _render_forearm(self) -> None:
        """(Re-)build the forearm point cloud actor on the plotter."""
        self._plotter.clear()
        self._plotter.set_background("black")

        vertices = self._data.forearm_vertices
        n_verts = len(vertices)

        cloud = pv.PolyData(vertices)
        cloud["heatmap"] = np.full(n_verts, np.nan, dtype=np.float64)
        self._cloud = cloud

        self._plotter.add_mesh(
            cloud,
            scalars="heatmap",
            cmap="jet",
            clim=(0.0, self._data.session_max_value),
            nan_color=[0.3, 0.3, 0.3],
            show_scalar_bar=True,
            render_points_as_spheres=False,
            point_size=3,
            name="forearm",
            copy_mesh=False,
        )
        self._plotter.view_xy()
        self._plotter.render()

    def _update_heatmap(self, single_touch_id: int) -> None:
        """Display the RF heatmap for *single_touch_id* on the forearm cloud.

        Looks up the integer touch key via ``touch_id_map``, builds a full-length
        scalar array (NaN for uncontacted vertices), assigns it, and renders.
        """
        if self._cloud is None:
            return

        bid = self._current_block_id()
        tid = self._current_trial_id()
        if bid is None or tid is None:
            return

        touch_key = (bid, tid, single_touch_id)
        if touch_key not in self._data.touch_id_map:
            raise ValueError(
                f"SingleTouchRFExplorer._update_heatmap: touch key {touch_key!r} "
                f"not found in touch_id_map for session {self._data.session_id!r}."
            )
        touch_int_id = self._data.touch_id_map[touch_key]

        pairs = self._data.rf_data.get(touch_int_id, [])

        n_verts = len(self._data.forearm_vertices)
        scalars = np.full(n_verts, np.nan, dtype=np.float64)
        for vertex_idx, mean_value in pairs:
            scalars[vertex_idx] = mean_value

        self._cloud["heatmap"] = scalars
        self._cloud.Modified()

        n_contacted = len(pairs)
        n_valid = sum(1 for _, v in pairs if not np.isnan(v))
        if n_contacted == 0:
            label = f"Touch {single_touch_id} | no contact / no neural data"
        elif n_valid == 0:
            # Legacy .npz files (pre-NaN-filter fix) can still land here.
            label = (
                f"Touch {single_touch_id} | {n_contacted} contacted vertices, "
                f"no neural data"
            )
        elif n_valid < n_contacted:
            label = (
                f"Touch {single_touch_id} | {n_valid}/{n_contacted} vertices | "
                f"{self._data.neuron_mode}"
            )
        else:
            label = (
                f"Touch {single_touch_id} | {n_contacted} vertices | "
                f"{self._data.neuron_mode}"
            )
        self._info_label.setText(label)

        self._restore_camera_state()
        self._plotter.render()

    # ------------------------------------------------------------------
    # Camera state persistence
    # ------------------------------------------------------------------

    def _save_camera_state(self) -> None:
        """Snapshot the current camera position for the active session."""
        try:
            cam = self._plotter.camera
            self._camera_states[self._data.session_id] = {
                "position": cam.GetPosition(),
                "focal_point": cam.GetFocalPoint(),
                "view_up": cam.GetViewUp(),
            }
        except Exception:
            pass

    def _restore_camera_state(self) -> None:
        """Restore a previously saved camera for the active session, if any."""
        state = self._camera_states.get(self._data.session_id)
        if state is None:
            return
        try:
            cam = self._plotter.camera
            cam.SetPosition(state["position"])
            cam.SetFocalPoint(state["focal_point"])
            cam.SetViewUp(state["view_up"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # First-touch helper
    # ------------------------------------------------------------------

    def _select_first_touch(self) -> None:
        """Render the heatmap for the first available touch in the current selection."""
        stid = self._current_single_touch_id()
        if stid is not None:
            self._update_heatmap(stid)

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Right:
                self._navigate(+1)
                return True
            if key == Qt.Key_Left:
                self._navigate(-1)
                return True
        return super().eventFilter(obj, event)

    def _navigate(self, direction: int) -> None:
        """Navigate touches (±1), cascading to trial then block at boundaries.

        direction=+1 advances (right), direction=-1 retreats (left).
        Stops silently at the first/last touch of the session.
        """
        if not self._initialized:
            return

        block_idx = self._block_combo.currentIndex()
        if block_idx < 0 or block_idx >= len(self._data.block_order_ids):
            return
        bid = self._data.block_order_ids[block_idx]

        trial_ids = self._data.trial_ids_by_block.get(bid, [])
        trial_idx = self._trial_combo.currentIndex()
        if trial_idx < 0 or trial_idx >= len(trial_ids):
            return
        tid = trial_ids[trial_idx]

        touch_ids = self._data.touches_by_block_trial.get((bid, tid), [])
        touch_idx = self._touch_combo.currentIndex()
        if touch_idx < 0 or touch_idx >= len(touch_ids):
            return

        new_bid = bid
        new_tid = tid
        new_touch_ids = touch_ids
        new_touch_idx = touch_idx + direction

        if 0 <= new_touch_idx < len(touch_ids):
            # Stays in same trial/block — only touch index changes.
            pass
        else:
            # Try adjacent trial within the same block.
            new_trial_idx = trial_idx + direction
            if 0 <= new_trial_idx < len(trial_ids):
                new_tid = trial_ids[new_trial_idx]
                new_touch_ids = self._data.touches_by_block_trial.get(
                    (new_bid, new_tid), []
                )
                if not new_touch_ids:
                    return
                new_touch_idx = 0 if direction > 0 else len(new_touch_ids) - 1

                self._trial_combo.blockSignals(True)
                self._trial_combo.setCurrentIndex(new_trial_idx)
                self._trial_combo.blockSignals(False)
            else:
                # Try adjacent block.
                new_block_idx = block_idx + direction
                if new_block_idx < 0 or new_block_idx >= len(self._data.block_order_ids):
                    return

                new_bid = self._data.block_order_ids[new_block_idx]
                new_trial_ids = self._data.trial_ids_by_block.get(new_bid, [])
                if not new_trial_ids:
                    return
                new_trial_idx = 0 if direction > 0 else len(new_trial_ids) - 1
                new_tid = new_trial_ids[new_trial_idx]
                new_touch_ids = self._data.touches_by_block_trial.get(
                    (new_bid, new_tid), []
                )
                if not new_touch_ids:
                    return
                new_touch_idx = 0 if direction > 0 else len(new_touch_ids) - 1

                self._block_combo.blockSignals(True)
                self._block_combo.setCurrentIndex(new_block_idx)
                self._block_combo.blockSignals(False)

                self._trial_combo.blockSignals(True)
                self._trial_combo.clear()
                for t in new_trial_ids:
                    self._trial_combo.addItem(f"Trial {t}")
                self._trial_combo.setCurrentIndex(new_trial_idx)
                self._trial_combo.blockSignals(False)

            # Repopulate touch combo for the new trial.
            self._touch_combo.blockSignals(True)
            self._touch_combo.clear()
            for stid in new_touch_ids:
                self._touch_combo.addItem(f"Touch {stid}")
            self._touch_combo.blockSignals(False)

        self._save_camera_state()
        self._touch_combo.blockSignals(True)
        self._touch_combo.setCurrentIndex(new_touch_idx)
        self._touch_combo.blockSignals(False)
        self._update_heatmap(new_touch_ids[new_touch_idx])

    # ------------------------------------------------------------------
    # Deferred VTK initialisation
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self) -> None:
        """Finish VTK init after the window is visible and sized."""
        try:
            self._plotter.interactor.Initialize()
        except Exception:
            pass
        sz = self._plotter.interactor.size()
        if sz.width() > 0 and sz.height() > 0:
            self._plotter.render_window.SetSize(sz.width(), sz.height())

        self._populate_block_combo(self._data)
        self._render_forearm()
        self._select_first_touch()

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._plotter.close()
        super().closeEvent(event)

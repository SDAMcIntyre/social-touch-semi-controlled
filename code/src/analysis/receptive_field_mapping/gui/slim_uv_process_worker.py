"""QThread worker that runs the SLIM UV pipeline in the background.

Used by the per-session SLIM UV config GUI to keep the main thread responsive
during mesh build, cleaning, and SLIM flattening. Only numpy arrays and
:class:`SlimStep` dataclass instances cross the thread boundary — no VTK or
PyVista objects are ever instantiated inside :meth:`run`.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
    build_slim_steps,
    run_slim_pipeline_core,
)
from analysis.receptive_field_mapping.surface.slim_uv_config_io import (
    SlimUvConfig,
)

logger = logging.getLogger(__name__)


class SlimUvProcessWorker(QThread):
    """Run ``run_slim_pipeline_core`` + ``build_slim_steps`` off the GUI thread.

    Signals
    -------
    result_ready(object)
        Emitted exactly once when ``run()`` finishes (success or failure). The
        payload is a dict with keys ``ok`` (bool), ``result`` (the core dict
        from :func:`run_slim_pipeline_core` or ``None``), ``steps`` (the list
        from :func:`build_slim_steps` or ``None``), and ``error`` (formatted
        exception string or ``None``).
    progress(str)
        Emitted periodically with human-readable status messages for the
        GUI status label.

    Note
    ----
    Named ``result_ready`` rather than ``finished`` to avoid colliding with
    ``QThread.finished``, which is a built-in parameterless signal.
    """

    result_ready = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(
        self,
        forearm_ply_path: Path,
        rf_maps_npz: Path,
        config: SlimUvConfig,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._forearm_ply_path = forearm_ply_path
        self._rf_maps_npz = rf_maps_npz
        self._config = config

    def run(self) -> None:
        try:
            self.progress.emit("Starting SLIM UV pipeline...")

            # YAML sentinel 0.0 means "auto" — translate to None for the core.
            max_edge_mm = (
                None if self._config.max_edge_mm == 0.0
                else float(self._config.max_edge_mm)
            )

            self.progress.emit(
                f"Building and cleaning mesh ({self._config.mesh_method})..."
            )
            core_result = run_slim_pipeline_core(
                self._forearm_ply_path,
                self._rf_maps_npz,
                mesh_method=self._config.mesh_method,
                max_edge_mm=max_edge_mm,
                clean_steps=self._config.clean_steps.to_dict(),
                n_iter=self._config.n_iter,
                collect_diagnostics=True,
            )
            self.progress.emit("SLIM flatten complete; building step list...")

            steps = build_slim_steps(
                V_raw=core_result["V_raw"],
                F_raw=core_result["F_raw"],
                clean_diag=core_result["clean_diag"],
                V_clean=core_result["V_clean"],
                F_clean=core_result["F_clean"],
                centroid_3d=core_result["centroid_3d"],
                bloop_pre=core_result["bloop_pre"],
                slim_diag=core_result["slim_diag"],
                V_final=core_result["V_final"],
                F_final=core_result["F_final"],
                uv_final=core_result["uv_final"],
                raw_mesh_colors=core_result["raw_mesh_colors"],
                clean_mesh_colors=core_result["clean_mesh_colors"],
                mesh_method=self._config.mesh_method,
            )

            self.progress.emit(f"Ready — {len(steps)} step(s) available.")
            self.result_ready.emit({
                "ok": True,
                "result": core_result,
                "steps": steps,
                "error": None,
            })
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            logger.exception("SlimUvProcessWorker failed")
            self.progress.emit(f"Error: {exc}")
            self.result_ready.emit({
                "ok": False,
                "result": None,
                "steps": None,
                "error": msg,
            })

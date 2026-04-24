"""Dialogs for editing touch_clustering ``reduction`` and ``evaluation`` config blocks.

:class:`ReductionConfigDialog` — Stage 3 (dimensionality reduction): scaler,
variance filter, and decomposition options.

:class:`EvaluationConfigDialog` — Stage 5 (cluster evaluation): internal
metrics and bootstrap stability scoring.

The ``stability`` sub-map serialises as ``null`` when the Enable checkbox is
unchecked, and as a full nested dict when checked.  Write-back uses
``DagConfigModel.set_task_option`` — no changes to the model layer are needed.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ruamel.yaml.comments import CommentedSeq

__all__ = ["ReductionConfigDialog", "EvaluationConfigDialog"]

_SCALERS: list[str] = ["standard", "robust", "none"]
_DECOMP_METHODS: list[str] = ["pca"]
_SCORES: list[str] = ["adjusted_rand_index"]
_INTERNAL_METRICS: list[str] = ["silhouette", "davies_bouldin", "calinski_harabasz"]


class ReductionConfigDialog(QDialog):
    """Dialog for editing the touch_clustering ``reduction`` YAML block (Stage 3).

    Pass the current YAML values for the reduction block via ``reduction_cfg``.
    After ``exec_()`` returns ``Accepted``, read the new dict back via
    :meth:`get_reduction`.

    Validation: the variance filter threshold must be parseable as a float
    before the dialog accepts.
    """

    def __init__(
        self,
        task_name: str,
        *,
        reduction_cfg: dict | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        cfg = dict(reduction_cfg or {})

        self.setWindowTitle(f"Reduction — {task_name}")
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Scaler ---------------------------------------------------------
        scaler_box = QGroupBox("Scaler")
        scaler_row = QHBoxLayout(scaler_box)
        self._scaler_combo = QComboBox()
        for s in _SCALERS:
            self._scaler_combo.addItem(s)
        current = cfg.get("scaler") or "standard"
        idx = _SCALERS.index(current) if current in _SCALERS else 0
        self._scaler_combo.setCurrentIndex(idx)
        scaler_row.addWidget(self._scaler_combo)
        scaler_row.addStretch()
        layout.addWidget(scaler_box)

        # Variance filter ------------------------------------------------
        vf_box = QGroupBox("Variance Filter")
        vf_layout = QVBoxLayout(vf_box)
        vf_cfg = cfg.get("variance_filter")
        self._vf_check = QCheckBox("Enable")
        self._vf_check.setChecked(vf_cfg is not None)
        vf_layout.addWidget(self._vf_check)

        self._vf_nested = QWidget()
        vf_nested_layout = QHBoxLayout(self._vf_nested)
        vf_nested_layout.setContentsMargins(20, 0, 0, 0)
        vf_nested_layout.addWidget(QLabel("Threshold:"))
        self._vf_threshold = QLineEdit()
        self._vf_threshold.setFixedWidth(120)
        threshold_val = (vf_cfg or {}).get("threshold", 0.01)
        self._vf_threshold.setText(str(threshold_val))
        vf_nested_layout.addWidget(self._vf_threshold)
        vf_nested_layout.addStretch()
        vf_layout.addWidget(self._vf_nested)
        layout.addWidget(vf_box)

        # Decomposition --------------------------------------------------
        decomp_box = QGroupBox("Decomposition")
        decomp_layout = QVBoxLayout(decomp_box)
        decomp_cfg = cfg.get("decomposition")
        self._decomp_check = QCheckBox("Enable")
        self._decomp_check.setChecked(decomp_cfg is not None)
        decomp_layout.addWidget(self._decomp_check)

        self._decomp_nested = QWidget()
        decomp_nested_form = QFormLayout(self._decomp_nested)
        decomp_nested_form.setContentsMargins(20, 0, 0, 0)
        self._decomp_method_combo = QComboBox()
        for m in _DECOMP_METHODS:
            self._decomp_method_combo.addItem(m)
        decomp_method = (decomp_cfg or {}).get("method", "pca")
        if decomp_method in _DECOMP_METHODS:
            self._decomp_method_combo.setCurrentIndex(_DECOMP_METHODS.index(decomp_method))
        decomp_nested_form.addRow("Method:", self._decomp_method_combo)
        self._decomp_n = QSpinBox()
        self._decomp_n.setMinimum(1)
        self._decomp_n.setMaximum(9999)
        self._decomp_n.setValue(int((decomp_cfg or {}).get("n_components", 10)))
        decomp_nested_form.addRow("N components:", self._decomp_n)
        decomp_layout.addWidget(self._decomp_nested)
        layout.addWidget(decomp_box)

        layout.addStretch()

        # Wire show/hide after widgets are built
        self._vf_nested.setVisible(vf_cfg is not None)
        self._decomp_nested.setVisible(decomp_cfg is not None)
        self._vf_check.toggled.connect(self._vf_nested.setVisible)
        self._decomp_check.toggled.connect(self._decomp_nested.setVisible)

        # Error label and button box -------------------------------------
        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setVisible(False)
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Validation & acceptance
    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        if self._vf_check.isChecked():
            try:
                float(self._vf_threshold.text())
            except ValueError:
                self._show_error("Variance filter threshold must be a number.")
                return

        self._error_label.setVisible(False)
        self.accept()

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

    # ------------------------------------------------------------------
    # Getter
    # ------------------------------------------------------------------

    def get_reduction(self) -> dict:
        """Return the reduction config dict reflecting current widget state."""
        scaler = self._scaler_combo.currentText()
        variance_filter = None
        if self._vf_check.isChecked():
            variance_filter = {"threshold": float(self._vf_threshold.text())}
        decomposition = None
        if self._decomp_check.isChecked():
            decomposition = {
                "method": self._decomp_method_combo.currentText(),
                "n_components": self._decomp_n.value(),
            }
        return {
            "scaler": scaler,
            "variance_filter": variance_filter,
            "decomposition": decomposition,
        }


class EvaluationConfigDialog(QDialog):
    """Dialog for editing the touch_clustering ``evaluation`` YAML block (Stage 5).

    Pass the current YAML values for the evaluation block via ``evaluation_cfg``.
    After ``exec_()`` returns ``Accepted``, read the new dict back via
    :meth:`get_evaluation`.

    ``internal_metrics`` is a flow-style ``CommentedSeq`` for round-trip YAML
    fidelity.  ``stability`` is ``None`` when disabled (serialises as YAML
    ``null``).  Unknown ``internal_metrics`` values present in the original
    config are silently dropped on write-back.
    """

    def __init__(
        self,
        task_name: str,
        *,
        evaluation_cfg: dict | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        cfg = dict(evaluation_cfg or {})

        self.setWindowTitle(f"Evaluation — {task_name}")
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Internal metrics -----------------------------------------------
        metrics_box = QGroupBox("Internal Metrics")
        metrics_layout = QVBoxLayout(metrics_box)
        current_metrics = set(cfg.get("internal_metrics") or [])
        self._metric_checks: dict[str, QCheckBox] = {}
        for m in _INTERNAL_METRICS:
            cb = QCheckBox(m.replace("_", " ").title())
            cb.setChecked(m in current_metrics)
            metrics_layout.addWidget(cb)
            self._metric_checks[m] = cb
        layout.addWidget(metrics_box)

        # Stability ------------------------------------------------------
        stab_box = QGroupBox("Stability")
        stab_layout = QVBoxLayout(stab_box)
        stab_cfg = cfg.get("stability")
        # A dict with method=None is treated as disabled
        if isinstance(stab_cfg, dict) and stab_cfg.get("method") is None:
            stab_cfg = None
        stab_enabled = stab_cfg is not None

        self._stab_check = QCheckBox("Enable bootstrap stability scoring")
        self._stab_check.setChecked(stab_enabled)
        stab_layout.addWidget(self._stab_check)

        self._stab_nested = QWidget()
        stab_nested_form = QFormLayout(self._stab_nested)
        stab_nested_form.setContentsMargins(20, 0, 0, 0)

        self._stab_n_rounds = QSpinBox()
        self._stab_n_rounds.setMinimum(1)
        self._stab_n_rounds.setMaximum(9999)
        self._stab_n_rounds.setValue(int((stab_cfg or {}).get("n_rounds", 20)))
        stab_nested_form.addRow("N rounds:", self._stab_n_rounds)

        self._stab_subsample = QDoubleSpinBox()
        self._stab_subsample.setMinimum(0.01)
        self._stab_subsample.setMaximum(1.0)
        self._stab_subsample.setSingleStep(0.05)
        self._stab_subsample.setDecimals(2)
        self._stab_subsample.setValue(float((stab_cfg or {}).get("subsample_fraction", 0.8)))
        stab_nested_form.addRow("Subsample fraction:", self._stab_subsample)

        self._stab_score_combo = QComboBox()
        for s in _SCORES:
            self._stab_score_combo.addItem(s.replace("_", " ").title())
        stab_score = (stab_cfg or {}).get("score", "adjusted_rand_index")
        if stab_score in _SCORES:
            self._stab_score_combo.setCurrentIndex(_SCORES.index(stab_score))
        stab_nested_form.addRow("Score:", self._stab_score_combo)

        stab_layout.addWidget(self._stab_nested)
        layout.addWidget(stab_box)

        layout.addStretch()

        # Wire show/hide after widgets are built
        self._stab_nested.setVisible(stab_enabled)
        self._stab_check.toggled.connect(self._stab_nested.setVisible)

        # Button box -----------------------------------------------------
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        self.accept()

    # ------------------------------------------------------------------
    # Getter
    # ------------------------------------------------------------------

    def get_evaluation(self) -> dict:
        """Return the evaluation config dict reflecting current widget state.

        ``internal_metrics`` is a flow-style ``CommentedSeq`` for round-trip
        YAML fidelity.  ``stability`` is ``None`` when disabled.
        """
        metrics = [m for m in _INTERNAL_METRICS if self._metric_checks[m].isChecked()]
        seq = CommentedSeq(metrics)
        seq.fa.set_flow_style()

        stability = None
        if self._stab_check.isChecked():
            stability = {
                "method": "bootstrap",
                "n_rounds": self._stab_n_rounds.value(),
                "subsample_fraction": self._stab_subsample.value(),
                "score": _SCORES[self._stab_score_combo.currentIndex()],
            }
        return {
            "internal_metrics": seq,
            "stability": stability,
        }

"""Small dialog for the pipeline parameters the stages actually read."""

from __future__ import annotations

from ai_insar.gui.qt import QtWidgets
from ai_insar.pipeline.engine import PipelineConfig
from ai_insar.pipeline.timeseries import TIMESERIES_LABELS, TIMESERIES_METHODS


class SettingsDialog(QtWidgets.QDialog):
    """Edits a :class:`PipelineConfig` in place."""

    def __init__(self, config: PipelineConfig, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pipeline settings")
        self._config = config
        form = QtWidgets.QFormLayout()

        self._method = QtWidgets.QComboBox()
        for key in TIMESERIES_METHODS:
            self._method.addItem(TIMESERIES_LABELS[key], key)
        current = self._method.findData(config.timeseries_method)
        if current >= 0:
            self._method.setCurrentIndex(current)
        form.addRow("Time series", self._method)

        self._span = QtWidgets.QSpinBox()
        self._span.setRange(1, 30)
        self._span.setValue(config.max_date_span)
        form.addRow("Max date span (pairs)", self._span)

        self._alpha = QtWidgets.QDoubleSpinBox()
        self._alpha.setRange(0.0, 1.0)
        self._alpha.setSingleStep(0.05)
        self._alpha.setDecimals(2)
        self._alpha.setValue(config.goldstein_alpha)
        form.addRow("Goldstein alpha", self._alpha)

        self._window = QtWidgets.QSpinBox()
        self._window.setRange(8, 256)
        self._window.setSingleStep(2)
        self._window.setValue(config.goldstein_window)
        form.addRow("Goldstein window (px)", self._window)

        self._coh = QtWidgets.QSpinBox()
        self._coh.setRange(1, 31)
        self._coh.setValue(config.coherence_window)
        form.addRow("Coherence window (px)", self._coh)

        self._min_coh = QtWidgets.QDoubleSpinBox()
        self._min_coh.setRange(0.0, 1.0)
        self._min_coh.setSingleStep(0.05)
        self._min_coh.setDecimals(2)
        self._min_coh.setValue(config.unwrap_min_coherence)
        form.addRow("Unwrap min coherence", self._min_coh)

        self._sigma = QtWidgets.QDoubleSpinBox()
        self._sigma.setRange(1.0, 10.0)
        self._sigma.setSingleStep(0.5)
        self._sigma.setDecimals(1)
        self._sigma.setValue(config.detection_sigma)
        form.addRow("Detection sigma", self._sigma)

        self._fraction = QtWidgets.QDoubleSpinBox()
        self._fraction.setRange(0.05, 0.9)
        self._fraction.setSingleStep(0.02)
        self._fraction.setDecimals(2)
        self._fraction.setValue(config.detection_peak_fraction)
        form.addRow("Detection peak fraction", self._fraction)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        note = QtWidgets.QLabel(
            "Interferograms stay later × conjugate(earlier). "
            "Changing these values does not rerun stages; run them again to apply."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _apply(self) -> None:
        self._config.timeseries_method = str(self._method.currentData())
        self._config.max_date_span = int(self._span.value())
        self._config.goldstein_alpha = float(self._alpha.value())
        self._config.goldstein_window = int(self._window.value())
        self._config.coherence_window = int(self._coh.value())
        self._config.unwrap_min_coherence = float(self._min_coh.value())
        self._config.detection_sigma = float(self._sigma.value())
        self._config.detection_peak_fraction = float(self._fraction.value())
        try:
            # Re-run the dataclass checks without discarding the other fields.
            PipelineConfig(
                max_date_span=self._config.max_date_span,
                goldstein_alpha=self._config.goldstein_alpha,
                goldstein_window=self._config.goldstein_window,
                goldstein_overlap=self._config.goldstein_overlap,
                goldstein_spectrum_smooth=self._config.goldstein_spectrum_smooth,
                coherence_window=self._config.coherence_window,
                coherence_azimuth_window=self._config.coherence_azimuth_window,
                unwrap_min_coherence=self._config.unwrap_min_coherence,
                ramp_border_px=self._config.ramp_border_px,
                detection_sigma=self._config.detection_sigma,
                detection_peak_fraction=self._config.detection_peak_fraction,
                detection_min_coherence=self._config.detection_min_coherence,
                detection_min_pixels=self._config.detection_min_pixels,
                timeseries_method=self._config.timeseries_method,
                ps_da_max=self._config.ps_da_max,
                ps_min_coherence=self._config.ps_min_coherence,
                nsbas_weight=self._config.nsbas_weight,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Pipeline settings", str(exc))
            return
        self.accept()

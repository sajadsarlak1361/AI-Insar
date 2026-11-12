"""Colour limits and units for the PS-linear Google Earth export."""

from __future__ import annotations

from ai_insar.gui.qt import QtWidgets


class PsExportDialog(QtWidgets.QDialog):
    """Choose LOS velocity or displacement, the unit, and optional colour limits."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export PS linear")
        self.setObjectName("psExportDialog")
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Only selected persistent-scatterer pixels are written. "
            "Empty pixels are transparent. Red is subsidence, blue is uplift. "
            "Leave the limits blank to use a symmetric 2–98% scale about zero. "
            "The legend is a separate colour bar."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.quantity = QtWidgets.QComboBox()
        self.quantity.setObjectName("psExportQuantity")
        self.quantity.addItem("LOS velocity", "velocity")
        self.quantity.addItem("LOS displacement", "displacement")
        self.quantity.currentIndexChanged.connect(self._sync_unit)
        self.unit = QtWidgets.QComboBox()
        self.unit.setObjectName("psExportUnit")
        self.date_index = QtWidgets.QSpinBox()
        self.date_index.setObjectName("psExportDate")
        self.date_index.setRange(0, 10_000)
        self.vmin = QtWidgets.QLineEdit()
        self.vmin.setObjectName("psExportVmin")
        self.vmin.setPlaceholderText("auto")
        self.vmax = QtWidgets.QLineEdit()
        self.vmax.setObjectName("psExportVmax")
        self.vmax.setPlaceholderText("auto")
        form.addRow("Value", self.quantity)
        form.addRow("Units", self.unit)
        form.addRow("Displacement date index", self.date_index)
        form.addRow("Minimum", self.vmin)
        form.addRow("Maximum", self.vmax)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._sync_unit()

    def _sync_unit(self) -> None:
        self.unit.clear()
        if self.quantity.currentData() == "displacement":
            self.unit.addItem("mm", ("mm", 1000.0))
            self.unit.addItem("m", ("m", 1.0))
            self.date_index.setEnabled(True)
            return
        self.unit.addItem("mm/yr", ("mm/yr", 1000.0))
        self.unit.addItem("m/yr", ("m/yr", 1.0))
        self.date_index.setEnabled(False)

    def settings(self) -> dict:
        unit, scale = self.unit.currentData()
        return {
            "quantity": self.quantity.currentData(),
            "unit": unit,
            "scale": scale,
            "date_index": int(self.date_index.value()),
            "vmin": _optional_float(self.vmin.text()),
            "vmax": _optional_float(self.vmax.text()),
        }


def _optional_float(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    return float(stripped)

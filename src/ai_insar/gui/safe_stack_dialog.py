"""Crop, swath, and polarisation for a multi-zip SAFE stack."""

from __future__ import annotations

from ai_insar.gui.qt import QtWidgets
from ai_insar.safe import DEFAULT_POLARISATION, DEFAULT_SWATH

_SWATHS = ("IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5", "WV1", "WV2")
_POLARISATIONS = ("VV", "VH", "HH", "HV")


class SafeStackDialog(QtWidgets.QDialog):
    """One crop window applied to every selected SAFE zip."""

    def __init__(self, parent: QtWidgets.QWidget | None, *, n_files: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open SAFE stack")
        self.setObjectName("safeStackDialog")
        self.resize(520, 360)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            f"Read {n_files} Sentinel-1 SAFE zip(s) into one complex64 stack. "
            "The zips stay zipped. Every date uses this swath, polarisation, "
            "and crop, then the dates are sorted by the manifest start time.\n\n"
            "Teaching crop-window alignment: the same row and column window "
            "is taken from each measurement. There is no coregistration, "
            "deburst, or calibration. With two or more dates, the interferogram "
            "stage can run as later × conjugate(earlier) on this stack."
        )
        note.setWordWrap(True)
        note.setObjectName("safeStackNote")
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        self.swath_box = QtWidgets.QComboBox()
        self.swath_box.setObjectName("safeStackSwath")
        self.swath_box.setEditable(True)
        self.swath_box.addItems(_SWATHS)
        self.swath_box.setCurrentText(DEFAULT_SWATH)
        self.pol_box = QtWidgets.QComboBox()
        self.pol_box.setObjectName("safeStackPol")
        self.pol_box.setEditable(True)
        self.pol_box.addItems(_POLARISATIONS)
        self.pol_box.setCurrentText(DEFAULT_POLARISATION)
        self.row = self._pixel(0)
        self.col = self._pixel(0)
        self.height = self._pixel(256)
        self.width = self._pixel(256)
        self.row.setObjectName("safeStackRow")
        self.col.setObjectName("safeStackCol")
        self.height.setObjectName("safeStackHeight")
        self.width.setObjectName("safeStackWidth")
        form.addRow("Swath", self.swath_box)
        form.addRow("Polarisation", self.pol_box)
        form.addRow("Row", self.row)
        form.addRow("Column", self.col)
        form.addRow("Height", self.height)
        form.addRow("Width", self.width)
        layout.addLayout(form)

        self.form_ifg = QtWidgets.QCheckBox("Form interferograms after the stack is loaded")
        self.form_ifg.setObjectName("safeStackFormIfg")
        self.form_ifg.setChecked(True)
        self.form_ifg.setToolTip(
            "Runs stage 2 on this stack: each pair is later × conjugate(earlier)."
        )
        layout.addWidget(self.form_ifg)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def swath(self) -> str:
        return self.swath_box.currentText().strip().upper()

    def polarisation(self) -> str:
        return self.pol_box.currentText().strip().upper()

    def _pixel(self, value: int) -> QtWidgets.QSpinBox:
        box = QtWidgets.QSpinBox()
        box.setRange(0, 200_000)
        box.setValue(value)
        return box

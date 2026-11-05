"""Open several SAFE folders or zips, then load one crop-window stack.

The product table matches SNAP's ProductSet-Reader. Create Stack parameters
are on the second tab. Resampling stays NONE: the same crop is read from
each product, with no complex resampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ai_insar.gui.product_set import OpenedPaths, ProductSetWidget
from ai_insar.gui.qt import QtWidgets
from ai_insar.safe import DEFAULT_POLARISATION, DEFAULT_SWATH
from ai_insar.snap_products import OPERATOR_HELP

_SWATHS = ("IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5", "WV1", "WV2")
_POLARISATIONS = ("VV", "VH", "HH", "HV")
_RESAMPLING = (
    "NONE",
    "NEAREST_NEIGHBOUR",
    "BILINEAR_INTERPOLATION",
    "CUBIC_CONVOLUTION",
)
_OFFSETS = ("Orbit", "Product Geolocation", "GCP")
_EXTENTS = ("Master", "Maximum", "Minimum")


def _disable_items(box: QtWidgets.QComboBox, allowed: str, note: str) -> None:
    model = box.model()
    for index in range(box.count()):
        if box.itemText(index) == allowed:
            continue
        item = model.item(index) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(False)
            item.setToolTip(note)


class SafeStackDialog(QtWidgets.QDialog):
    """Product list plus the crop applied to every selected SAFE product."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        n_files: int,
        initial_paths: Sequence[Path | str] | None = None,
        opened_paths: OpenedPaths | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open SAFE stack")
        self.setObjectName("safeStackDialog")
        self.resize(820, 640)
        layout = QtWidgets.QVBoxLayout(self)

        help_line = QtWidgets.QLabel(OPERATOR_HELP["stack"])
        help_line.setObjectName("operatorHelp")
        help_line.setWordWrap(True)
        layout.addWidget(help_line)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("safeStackTabs")
        self.product_set = ProductSetWidget(self, opened_paths=opened_paths)
        self.product_set.changed.connect(self._sync_reference)
        reader = QtWidgets.QWidget()
        reader_layout = QtWidgets.QVBoxLayout(reader)
        reader_note = QtWidgets.QLabel(
            "Add SAFE folders or .zip files. The first row is the reference in this list. "
            "The SLC cube is still sorted by the manifest start time. "
            "Zip files stay zipped."
        )
        reader_note.setWordWrap(True)
        reader_layout.addWidget(reader_note)
        reader_layout.addWidget(self.product_set)
        self.tabs.addTab(reader, "ProductSet-Reader")
        self.tabs.addTab(self._create_tab(n_files), "Create Stack")
        layout.addWidget(self.tabs, stretch=1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        run = QtWidgets.QPushButton("Run")
        run.setObjectName("safeStackRun")
        run.clicked.connect(self._on_run)
        close = QtWidgets.QPushButton("Close")
        close.setObjectName("safeStackClose")
        close.clicked.connect(self.reject)
        buttons.addWidget(run)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        if initial_paths:
            self.product_set.add_paths(initial_paths)
        self._sync_reference()

    def _create_tab(self, n_files: int) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        note = QtWidgets.QLabel(
            f"Read {n_files} Sentinel-1 SAFE zip(s) into one complex64 stack. "
            "The zips stay zipped. Every date uses this swath, polarisation, "
            "and crop, then the dates are sorted by the manifest start time.\n\n"
            "Teaching crop-window alignment: the same row and column window "
            "is taken from each measurement. There is no coregistration, "
            "deburst, or calibration. With two or more dates, the interferogram "
            "stage can run as later × conjugate(earlier) on this stack. "
            "SAFE folders are accepted as well as .zip files."
        )
        note.setWordWrap(True)
        note.setObjectName("safeStackNote")
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        self.reference = QtWidgets.QComboBox()
        self.reference.setObjectName("createStackReference")
        self.reference.setToolTip(
            "User-chosen reference. Every other selected product stays in the stack."
        )
        self.resampling = QtWidgets.QComboBox()
        self.resampling.setObjectName("createStackResampling")
        self.resampling.addItems(_RESAMPLING)
        self.resampling.setCurrentText("NONE")
        _disable_items(
            self.resampling,
            "NONE",
            "Not implemented. InSAR keeps Resampling NONE (the same crop window).",
        )
        self.initial_offset = QtWidgets.QComboBox()
        self.initial_offset.setObjectName("createStackOffset")
        self.initial_offset.addItems(_OFFSETS)
        self.initial_offset.setCurrentText("Orbit")
        _disable_items(
            self.initial_offset,
            "Orbit",
            "Only Orbit is listed. The orbit offset is not applied; the crop window is shared.",
        )
        self.initial_offset.setEnabled(False)
        self.initial_offset.setToolTip(
            "Initial Offset Method Orbit is not applied. Resampling NONE reads the same crop from each product."
        )
        self.output_extents = QtWidgets.QComboBox()
        self.output_extents.setObjectName("createStackExtents")
        self.output_extents.addItems(_EXTENTS)
        self.output_extents.setCurrentText("Master")
        _disable_items(
            self.output_extents,
            "Master",
            "Not implemented. Output Extents stays Master: the crop window below.",
        )
        self.find_reference = QtWidgets.QPushButton("Find Optimal Reference")
        self.find_reference.setObjectName("createStackFindReference")
        self.find_reference.setToolTip(
            "Not implemented. The SLC cube is time-sorted by the manifest start time."
        )
        self.find_reference.setEnabled(False)
        form.addRow("Reference:", self.reference)
        form.addRow("Resampling Type", self.resampling)
        form.addRow("Initial Offset Method", self.initial_offset)
        form.addRow("Output Extents", self.output_extents)
        form.addRow(self.find_reference)

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
        self.burst_first = self._pixel(0)
        self.burst_last = self._pixel(0)
        self.burst_first.setObjectName("safeStackBurstFirst")
        self.burst_last.setObjectName("safeStackBurstLast")
        self.burst_first.setToolTip("First burst, 1-based. 0 with last 0 means the burst range is not limited.")
        self.burst_last.setToolTip("Last burst, 1-based. Checked against each product. The pixel crop is separate.")
        form.addRow("First burst (0 = all)", self.burst_first)
        form.addRow("Last burst (0 = all)", self.burst_last)
        layout.addLayout(form)

        self.form_ifg = QtWidgets.QCheckBox("Form interferograms after the stack is loaded")
        self.form_ifg.setObjectName("safeStackFormIfg")
        self.form_ifg.setChecked(True)
        self.form_ifg.setToolTip(
            "Runs stage 2 on this stack: each pair is later × conjugate(earlier)."
        )
        layout.addWidget(self.form_ifg)
        layout.addStretch(1)
        return page

    def _pixel(self, value: int) -> QtWidgets.QSpinBox:
        box = QtWidgets.QSpinBox()
        box.setRange(0, 200_000)
        box.setValue(value)
        return box

    def swath(self) -> str:
        return self.swath_box.currentText().strip().upper()

    def polarisation(self) -> str:
        return self.pol_box.currentText().strip().upper()

    def paths(self) -> list[Path]:
        return self.product_set.paths()

    def reference_path(self) -> Path | None:
        data = self.reference.currentData()
        if data:
            return Path(str(data))
        paths = self.paths()
        return paths[0] if paths else None

    def burst_first_value(self) -> int:
        return int(self.burst_first.value())

    def burst_last_value(self) -> int:
        return int(self.burst_last.value())

    def _sync_reference(self) -> None:
        current = self.reference.currentData()
        self.reference.blockSignals(True)
        self.reference.clear()
        for row in self.product_set.rows():
            self.reference.addItem(row.file_name, str(row.path))
        if current is not None:
            index = self.reference.findData(current)
            if index >= 0:
                self.reference.setCurrentIndex(index)
        self.reference.blockSignals(False)

    def _on_run(self) -> None:
        from ai_insar.safe_stack import inspect_safe_products

        paths = self.paths()
        if not paths:
            self.product_set.status.setText(
                "Error: [NodeId: ProductSet-Reader] Please add a source product"
            )
            return
        if self.resampling.currentText() != "NONE":
            self.product_set.status.setText("InSAR requires Resampling Type NONE.")
            return
        if int(self.height.value()) < 1 or int(self.width.value()) < 1:
            self.product_set.status.setText("Crop height and width must be at least 1.")
            return
        try:
            inspect_safe_products(
                paths,
                swath=self.swath(),
                pol=self.polarisation(),
                reference=self.reference_path(),
                burst_first=self.burst_first_value(),
                burst_last=self.burst_last_value(),
            )
        except ValueError as exc:
            self.product_set.status.setText(str(exc))
            return
        self.accept()

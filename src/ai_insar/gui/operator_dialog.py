"""SNAP-style operator dialogs: I/O tab, processing tab, Run, and Close.

Controls that this build cannot apply stay visible and disabled, with a
short note. Run writes the request into the existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_insar.gui.qt import QtWidgets
from ai_insar.pipeline.engine import PipelineConfig
from ai_insar.snap_products import OPERATOR_HELP, PRODUCT_SUFFIX

_FLAT_NOTE = "Flat-earth subtraction is not implemented. It is not applied."
_COPERNICUS_NOTE = "Copernicus DEM auto-download is not in this build."
_DIMAP_NOTE = "BEAM-DIMAP is not written. NumPy .npy is the on-disk product."
_ADAPTIVE_NOTE = "Adaptive alpha is not implemented. The exponent is constant."
_MASK_NOTE = "The coherence mask is not implemented."
_ORBIT_POLY_NOTE = "Orbit interpolation is not applied. This build uses one perpendicular baseline."
_TILE_NOTE = "DEM tile extension is not implemented."
_EXTRA_BANDS = "This optional band is not written."


@dataclass
class OperatorRequest:
    """Values the main window applies after Run."""

    target: str
    save: bool
    directory: str
    open_in_app: bool
    skipped: tuple[str, ...] = ()


def _disable(widget: QtWidgets.QWidget, note: str) -> None:
    widget.setEnabled(False)
    widget.setToolTip(note)


def _disable_item(box: QtWidgets.QComboBox, text: str, note: str) -> None:
    index = box.findText(text)
    if index < 0:
        return
    model = box.model()
    item = model.item(index) if hasattr(model, "item") else None
    if item is not None:
        item.setEnabled(False)
        item.setToolTip(note)


def _combo(items: tuple[str, ...], current: str, name: str, disabled: dict[str, str] | None = None) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox()
    box.setObjectName(name)
    box.addItems(list(items))
    box.setCurrentText(current)
    for text, note in (disabled or {}).items():
        _disable_item(box, text, note)
    return box


def _editable(items: tuple[str, ...], current: str, name: str) -> QtWidgets.QComboBox:
    box = _combo(items, current, name)
    box.setEditable(True)
    box.setCurrentText(current)
    return box


def _spin(value: int, low: int, high: int, name: str) -> QtWidgets.QSpinBox:
    box = QtWidgets.QSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    box.setObjectName(name)
    return box


def _double(value: float, low: float, high: float, name: str, *, decimals: int = 2, step: float = 0.1) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setObjectName(name)
    return box


class OperatorDialog(QtWidgets.QDialog):
    """Shared shell: help strip, I/O Parameters, Processing Parameters, Run, Close."""

    def __init__(
        self,
        title: str,
        help_line: str,
        *,
        suffix: str,
        object_name: str,
        notes: str,
        parent: QtWidgets.QWidget | None = None,
        sources: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName(object_name)
        self.resize(760, 560)
        self._suffix = suffix
        root = QtWidgets.QVBoxLayout(self)

        help_label = QtWidgets.QLabel(help_line)
        help_label.setObjectName("operatorHelp")
        help_label.setWordWrap(True)
        root.addWidget(help_label)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("operatorTabs")
        self.tabs.addTab(self._build_io(sources or [], suffix), "I/O Parameters")
        self._processing = QtWidgets.QWidget()
        self._processing_layout = QtWidgets.QVBoxLayout(self._processing)
        self.tabs.addTab(self._processing, "Processing Parameters")
        root.addWidget(self.tabs, stretch=1)

        note = QtWidgets.QLabel(notes)
        note.setObjectName("operatorNotes")
        note.setWordWrap(True)
        root.addWidget(note)

        self.error = QtWidgets.QLabel("")
        self.error.setObjectName("operatorError")
        self.error.setWordWrap(True)
        root.addWidget(self.error)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        run = QtWidgets.QPushButton("Run")
        run.setObjectName("operatorRun")
        run.clicked.connect(self._on_run)
        close = QtWidgets.QPushButton("Close")
        close.setObjectName("operatorClose")
        close.clicked.connect(self.reject)
        buttons.addWidget(run)
        buttons.addWidget(close)
        root.addLayout(buttons)

    def _build_io(self, sources: list[str], suffix: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        source_row = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.setObjectName("ioSource")
        self.source.setEditable(True)
        for name in sources:
            self.source.addItem(name)
        if self.source.count() == 0:
            self.source.addItem("Workspace SLC stack")
        browse = QtWidgets.QPushButton("…")
        browse.setObjectName("ioSourceBrowse")
        browse.setToolTip("Choose a source product file")
        browse.clicked.connect(self._browse_source)
        source_row.addWidget(self.source, stretch=1)
        source_row.addWidget(browse)
        form.addRow("Source product:", source_row)

        self.target = QtWidgets.QLineEdit(f"target{suffix}")
        self.target.setObjectName("ioTarget")
        form.addRow("Target product name:", self.target)

        save_row = QtWidgets.QHBoxLayout()
        self.save_box = QtWidgets.QCheckBox("Save as")
        self.save_box.setObjectName("ioSave")
        self.save_box.setChecked(True)
        self.format_box = _combo(
            ("NumPy (.npy)", "BEAM-DIMAP"),
            "NumPy (.npy)",
            "ioFormat",
            {"BEAM-DIMAP": _DIMAP_NOTE},
        )
        self.format_box.setToolTip("SNAP's default format is BEAM-DIMAP. This build writes NumPy.")
        save_row.addWidget(self.save_box)
        save_row.addWidget(self.format_box, stretch=1)
        form.addRow(save_row)

        directory_row = QtWidgets.QHBoxLayout()
        self.directory = QtWidgets.QLineEdit("")
        self.directory.setObjectName("ioDirectory")
        directory_button = QtWidgets.QPushButton("…")
        directory_button.setObjectName("ioDirectoryBrowse")
        directory_button.clicked.connect(self._browse_directory)
        directory_row.addWidget(self.directory, stretch=1)
        directory_row.addWidget(directory_button)
        form.addRow("Directory:", directory_row)

        self.open_box = QtWidgets.QCheckBox("Open in AI Insar")
        self.open_box.setObjectName("ioOpen")
        self.open_box.setChecked(True)
        form.addRow(self.open_box)
        return page

    def add_processing(self, layout: QtWidgets.QLayout) -> None:
        self._processing_layout.addLayout(layout)
        self._processing_layout.addStretch(1)

    def _browse_source(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Source product",
            "",
            "Sentinel-1 SAFE zip (*.zip);;NumPy (*.npy);;All files (*)",
        )
        if path:
            self.source.setCurrentText(path)

    def _browse_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory")
        if directory:
            self.directory.setText(directory)

    def validate_io(self) -> str:
        if not self.source.currentText().strip():
            return "Select a source product."
        if not self.target.text().strip():
            return "Enter a target product name."
        if self.save_box.isChecked():
            if self.format_box.currentText() != "NumPy (.npy)":
                return _DIMAP_NOTE
            if not self.directory.text().strip():
                return "Choose a directory, or uncheck Save as."
        return ""

    def validate(self) -> str:
        return self.validate_io()

    def io_request(self, skipped: tuple[str, ...] = ()) -> OperatorRequest:
        return OperatorRequest(
            target=self.target.text().strip(),
            save=self.save_box.isChecked(),
            directory=self.directory.text().strip(),
            open_in_app=self.open_box.isChecked(),
            skipped=skipped,
        )

    def _on_run(self) -> None:
        message = self.validate()
        if message:
            self.error.setText(message)
            return
        self.error.setText("")
        self.accept()


class InterferogramDialog(OperatorDialog):
    """Interferogram Formation. Flat-earth controls are shown and disabled."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, sources: list[str] | None = None) -> None:
        super().__init__(
            "Interferogram Formation",
            OPERATOR_HELP["interferogram"],
            parent=parent,
            sources=sources,
            suffix=PRODUCT_SUFFIX["interferogram"],
            object_name="interferogramDialog",
            notes=(
                "Flat-earth subtraction, the residual ramp, Copernicus DEM download, "
                "tile extension, and the optional flat-earth, elevation, and lat/lon bands "
                "are disabled. Interferograms are later × conjugate(earlier)."
            ),
        )
        form = QtWidgets.QFormLayout()
        self.subtract_flat = QtWidgets.QCheckBox('Subtract Flat-Earth Phase')
        self.subtract_flat.setObjectName("ifgSubtractFlatEarth")
        self.subtract_flat.setChecked(True)
        _disable(self.subtract_flat, _FLAT_NOTE)
        self.flat_degree = _editable(("3", "4", "5", "6", "7"), "5", "ifgFlatDegree")
        _disable(self.flat_degree, _FLAT_NOTE)
        self.flat_points = _editable(("301", "401", "501", "601", "701"), "501", "ifgFlatPoints")
        _disable(self.flat_points, _FLAT_NOTE)
        self.orbit_degree = _editable(("2", "3", "4", "5"), "3", "ifgOrbitDegree")
        _disable(self.orbit_degree, _ORBIT_POLY_NOTE)
        self.output_flat = QtWidgets.QCheckBox("Output Flat Earth Phase")
        self.output_flat.setObjectName("ifgOutputFlatEarth")
        _disable(self.output_flat, _EXTRA_BANDS)
        self.subtract_topo = QtWidgets.QCheckBox("Subtract Topographic Phase")
        self.subtract_topo.setObjectName("ifgSubtractTopo")
        self.subtract_topo.setToolTip("Uses the DEM and the workspace perpendicular baseline.")
        self.subtract_ramp = QtWidgets.QCheckBox("Subtract Residual Ramp (GSLC)")
        self.subtract_ramp.setObjectName("ifgSubtractRamp")
        _disable(self.subtract_ramp, "Residual GSLC ramp removal is not implemented.")
        self.dem = _combo(
            (
                "Workspace DEM (.npy)",
                "External DEM (.npy)",
                "Copernicus 30m Global DEM (Auto Download)",
            ),
            "Workspace DEM (.npy)",
            "ifgDem",
            {"Copernicus 30m Global DEM (Auto Download)": _COPERNICUS_NOTE},
        )
        self.dem_path = QtWidgets.QLineEdit("")
        self.dem_path.setObjectName("ifgDemPath")
        self.dem_path.setPlaceholderText("External DEM .npy, same shape as the SLC grid")
        dem_row = QtWidgets.QHBoxLayout()
        dem_row.addWidget(self.dem, stretch=1)
        browse = QtWidgets.QPushButton("…")
        browse.setObjectName("ifgDemBrowse")
        browse.clicked.connect(self._browse_dem)
        dem_row.addWidget(self.dem_path, stretch=1)
        dem_row.addWidget(browse)
        self.tile = _editable(("50", "100", "150"), "100", "ifgTile")
        _disable(self.tile, _TILE_NOTE)
        self.output_topo = QtWidgets.QCheckBox("Output Topographic Phase")
        self.output_topo.setObjectName("ifgOutputTopo")
        _disable(self.output_topo, _EXTRA_BANDS)
        self.output_elevation = QtWidgets.QCheckBox("Output Elevation")
        self.output_elevation.setObjectName("ifgOutputElevation")
        _disable(self.output_elevation, _EXTRA_BANDS)
        self.output_latlon = QtWidgets.QCheckBox("Output Orthorectified Lat/Lon")
        self.output_latlon.setObjectName("ifgOutputLatLon")
        _disable(self.output_latlon, _EXTRA_BANDS)
        self.output_coherence = QtWidgets.QCheckBox("Output Coherence")
        self.output_coherence.setObjectName("ifgOutputCoherence")
        self.output_coherence.setChecked(True)
        self.square_pixel = QtWidgets.QCheckBox("Square Pixel")
        self.square_pixel.setObjectName("ifgSquarePixel")
        self.square_pixel.setChecked(True)
        self.square_pixel.setToolTip(
            "Square Pixel uses the range window in both directions. Ground spacing is not applied."
        )
        self.independent = QtWidgets.QCheckBox("Independent Window Sizes")
        self.independent.setObjectName("ifgIndependentWindows")
        self.coherence_range = _spin(10, 1, 99, "ifgCohRange")
        self.coherence_azimuth = _spin(2, 1, 99, "ifgCohAzimuth")
        self.square_pixel.toggled.connect(self._on_square)
        self.independent.toggled.connect(self._on_independent)
        self.subtract_topo.toggled.connect(self._sync_dem)
        self.dem.currentTextChanged.connect(lambda _text: self._sync_dem(self.subtract_topo.isChecked()))
        self._on_square(True)
        self._sync_dem(False)

        form.addRow(self.subtract_flat)
        form.addRow('Degree of "Flat Earth" polynomial', self.flat_degree)
        form.addRow('Number of "Flat Earth" estimation points', self.flat_points)
        form.addRow("Orbit interpolation degree", self.orbit_degree)
        form.addRow(self.output_flat)
        form.addRow(self.subtract_topo)
        form.addRow(self.subtract_ramp)
        form.addRow("Digital Elevation Model", dem_row)
        form.addRow("Tile Extension [%]", self.tile)
        form.addRow(self.output_topo)
        form.addRow(self.output_elevation)
        form.addRow(self.output_latlon)
        form.addRow(self.output_coherence)
        form.addRow(self.square_pixel)
        form.addRow(self.independent)
        form.addRow("Coherence Range Window Size", self.coherence_range)
        form.addRow("Coherence Azimuth Window Size", self.coherence_azimuth)
        self.add_processing(form)

    def _browse_dem(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(self, "DEM", "", "NumPy DEM (*.npy)")
        if path:
            self.dem.setCurrentText("External DEM (.npy)")
            self.dem_path.setText(path)
            self.subtract_topo.setChecked(True)

    def _sync_dem(self, checked: bool) -> None:
        self.dem.setEnabled(bool(checked))
        self.dem_path.setEnabled(bool(checked) and self.dem.currentText().startswith("External"))

    def _on_square(self, checked: bool) -> None:
        if checked:
            self.independent.blockSignals(True)
            self.independent.setChecked(False)
            self.independent.blockSignals(False)
        self.coherence_azimuth.setEnabled(not self.square_pixel.isChecked())

    def _on_independent(self, checked: bool) -> None:
        if checked:
            self.square_pixel.blockSignals(True)
            self.square_pixel.setChecked(False)
            self.square_pixel.blockSignals(False)
        self.coherence_azimuth.setEnabled(not self.square_pixel.isChecked())

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        if self.coherence_range.value() < 1:
            return "Coherence range window must be at least 1."
        if self.subtract_topo.isChecked() and self.dem.currentText().startswith("Copernicus"):
            return _COPERNICUS_NOTE
        if self.subtract_topo.isChecked() and self.dem.currentText().startswith("External") and not self.dem_path.text().strip():
            return "Choose an external DEM .npy, or pick Workspace DEM."
        return ""

    def parameters(self) -> dict:
        request = self.io_request(
            skipped=(
                "Subtract flat-earth phase is not applied.",
                "Residual ramp removal is not applied.",
            )
        )
        return {
            "request": request,
            "subtract_topo": self.subtract_topo.isChecked(),
            "dem_choice": self.dem.currentText(),
            "dem_path": self.dem_path.text().strip(),
            "output_coherence": self.output_coherence.isChecked(),
            "square_pixel": self.square_pixel.isChecked(),
            "coherence_range": int(self.coherence_range.value()),
            "coherence_azimuth": int(self.coherence_azimuth.value()),
        }


class CoherenceDialog(OperatorDialog):
    """Coherence Estimation. Window sizes are applied; phase screens are not."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, sources: list[str] | None = None) -> None:
        super().__init__(
            "Coherence Estimation",
            OPERATOR_HELP["coherence"],
            parent=parent,
            sources=sources,
            suffix=PRODUCT_SUFFIX["coherence"],
            object_name="coherenceDialog",
            notes=(
                "Flat-earth and topographic subtraction, coherence-bias correction, "
                "local ramp removal, and Copernicus DEM download are disabled. "
                "The estimate is the SLC-pair window. Square Pixel uses equal pixel counts."
            ),
        )
        form = QtWidgets.QFormLayout()
        self.single_reference = QtWidgets.QCheckBox("Single Reference")
        self.single_reference.setObjectName("cohSingleReference")
        self.single_reference.setChecked(True)
        self.single_reference.setToolTip("Pairs are date 0 with each later date. Uncheck to keep the current network.")
        self.subtract_flat = QtWidgets.QCheckBox("Subtract flat-earth phase")
        self.subtract_flat.setObjectName("cohSubtractFlat")
        self.subtract_flat.setChecked(True)
        _disable(self.subtract_flat, _FLAT_NOTE)
        self.flat_degree = _editable(("3", "4", "5", "6"), "5", "cohFlatDegree")
        _disable(self.flat_degree, _FLAT_NOTE)
        self.flat_points = _editable(("301", "501", "701"), "501", "cohFlatPoints")
        _disable(self.flat_points, _FLAT_NOTE)
        self.orbit_degree = _editable(("2", "3", "4"), "3", "cohOrbitDegree")
        _disable(self.orbit_degree, _ORBIT_POLY_NOTE)
        self.subtract_topo = QtWidgets.QCheckBox("Subtract topographic phase")
        self.subtract_topo.setObjectName("cohSubtractTopo")
        self.subtract_topo.setChecked(True)
        _disable(self.subtract_topo, "Coherence is estimated from the SLC pair, not from a topo-removed interferogram.")
        self.dem = _combo(
            ("Copernicus 30m Global DEM (Auto Download)", "Workspace DEM (.npy)"),
            "Copernicus 30m Global DEM (Auto Download)",
            "cohDem",
            {
                "Copernicus 30m Global DEM (Auto Download)": _COPERNICUS_NOTE,
                "Workspace DEM (.npy)": "Topographic subtraction is not applied inside coherence.",
            },
        )
        _disable(self.dem, "Topographic subtraction is not applied inside coherence.")
        self.tile = _editable(("50", "100", "150"), "100", "cohTile")
        _disable(self.tile, _TILE_NOTE)
        self.square_pixel = QtWidgets.QCheckBox("Square Pixel")
        self.square_pixel.setObjectName("cohSquarePixel")
        self.square_pixel.setChecked(True)
        self.square_pixel.setToolTip(
            "SNAP would size the azimuth window from pixel spacing. This build uses the range size for both axes."
        )
        self.independent = QtWidgets.QCheckBox("Independent Window Sizes")
        self.independent.setObjectName("cohIndependentWindows")
        self.coherence_range = _spin(10, 1, 99, "cohRange")
        self.coherence_azimuth = _spin(2, 1, 99, "cohAzimuth")
        self.correct_bias = QtWidgets.QCheckBox("Correct Coherence Bias")
        self.correct_bias.setObjectName("cohCorrectBias")
        self.correct_bias.setChecked(True)
        _disable(self.correct_bias, "Coherence bias correction is not implemented.")
        self.remove_ramp = QtWidgets.QCheckBox("Remove Local Phase Ramp")
        self.remove_ramp.setObjectName("cohRemoveRamp")
        _disable(self.remove_ramp, "Local phase-ramp removal is not implemented.")
        self.square_pixel.toggled.connect(self._on_square)
        self.independent.toggled.connect(self._on_independent)
        self._on_square(True)

        form.addRow(self.single_reference)
        form.addRow(self.subtract_flat)
        form.addRow('Degree of "Flat Earth" polynomial', self.flat_degree)
        form.addRow('Number of "Flat Earth" estimation points', self.flat_points)
        form.addRow("Orbit interpolation degree", self.orbit_degree)
        form.addRow(self.subtract_topo)
        form.addRow("Digital Elevation Model", self.dem)
        form.addRow("Tile Extension [%]", self.tile)
        form.addRow(self.square_pixel)
        form.addRow(self.independent)
        form.addRow("Coherence Range Window Size", self.coherence_range)
        form.addRow("Coherence Azimuth Window Size", self.coherence_azimuth)
        form.addRow(self.correct_bias)
        form.addRow(self.remove_ramp)
        self.add_processing(form)

    def _on_square(self, checked: bool) -> None:
        if checked:
            self.independent.blockSignals(True)
            self.independent.setChecked(False)
            self.independent.blockSignals(False)
        self.coherence_azimuth.setEnabled(not self.square_pixel.isChecked())

    def _on_independent(self, checked: bool) -> None:
        if checked:
            self.square_pixel.blockSignals(True)
            self.square_pixel.setChecked(False)
            self.square_pixel.blockSignals(False)
        self.coherence_azimuth.setEnabled(not self.square_pixel.isChecked())

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        if self.coherence_range.value() < 1:
            return "Coherence range window must be at least 1."
        if not self.square_pixel.isChecked() and self.coherence_azimuth.value() < 1:
            return "Coherence azimuth window must be at least 1."
        return ""

    def parameters(self) -> dict:
        request = self.io_request(
            skipped=(
                "Subtract flat-earth phase is not applied.",
                "Subtract topographic phase is not applied inside coherence.",
                "Coherence bias correction is not applied.",
            )
        )
        return {
            "request": request,
            "single_reference": self.single_reference.isChecked(),
            "square_pixel": self.square_pixel.isChecked(),
            "coherence_range": int(self.coherence_range.value()),
            "coherence_azimuth": int(self.coherence_azimuth.value()),
        }


class TopoPhaseDialog(OperatorDialog):
    """Topographic Phase Removal using the in-tree baseline formula."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        sources: list[str] | None = None,
        baseline_m: float = 0.0,
        slant_range_m: float = 800_000.0,
        incidence_deg: float = 35.0,
    ) -> None:
        super().__init__(
            "Topographic Phase Removal",
            OPERATOR_HELP["topo"],
            parent=parent,
            sources=sources,
            suffix=PRODUCT_SUFFIX["topo"],
            object_name="topoPhaseDialog",
            notes=(
                "Orbit polynomial degree, Copernicus auto-download, tile extension, "
                "and the optional output bands are disabled. "
                "The phase uses the perpendicular baseline and a DEM on the SLC grid."
            ),
        )
        form = QtWidgets.QFormLayout()
        self.orbit_degree = QtWidgets.QLineEdit("3")
        self.orbit_degree.setObjectName("topoOrbitDegree")
        _disable(self.orbit_degree, _ORBIT_POLY_NOTE)
        self.dem = _combo(
            (
                "Workspace DEM (.npy)",
                "External DEM (.npy)",
                "Copernicus 30m Global DEM (Auto Download)",
            ),
            "Workspace DEM (.npy)",
            "topoDem",
            {"Copernicus 30m Global DEM (Auto Download)": _COPERNICUS_NOTE},
        )
        self.dem_path = QtWidgets.QLineEdit("")
        self.dem_path.setObjectName("topoDemPath")
        self.dem_path.setPlaceholderText("External DEM .npy")
        dem_row = QtWidgets.QHBoxLayout()
        dem_row.addWidget(self.dem, stretch=1)
        browse = QtWidgets.QPushButton("…")
        browse.setObjectName("topoDemBrowse")
        browse.clicked.connect(self._browse_dem)
        dem_row.addWidget(self.dem_path, stretch=1)
        dem_row.addWidget(browse)
        self.tile = _editable(("50", "100", "150"), "100", "topoTile")
        _disable(self.tile, _TILE_NOTE)
        self.output_phase = QtWidgets.QCheckBox("Output topographic phase band")
        self.output_phase.setObjectName("topoOutputPhase")
        _disable(self.output_phase, _EXTRA_BANDS)
        self.output_elevation = QtWidgets.QCheckBox("Output elevation band")
        self.output_elevation.setObjectName("topoOutputElevation")
        _disable(self.output_elevation, _EXTRA_BANDS)
        self.output_latlon = QtWidgets.QCheckBox("Output orthorectified Lat/Lon bands")
        self.output_latlon.setObjectName("topoOutputLatLon")
        _disable(self.output_latlon, _EXTRA_BANDS)
        self.baseline = _double(baseline_m, -5000.0, 5000.0, "topoBaseline", decimals=1, step=10.0)
        self.baseline.setToolTip("Perpendicular baseline in metres. 0 leaves the interferogram unchanged.")
        self.slant = _double(slant_range_m, 1.0, 2_000_000.0, "topoSlant", decimals=0, step=1000.0)
        self.incidence = _double(incidence_deg, 1.0, 89.0, "topoIncidence", decimals=1, step=1.0)
        form.addRow("Orbit Interpolation Degree", self.orbit_degree)
        form.addRow("Digital Elevation Model", dem_row)
        form.addRow("Tile Extension [%]", self.tile)
        form.addRow(self.output_phase)
        form.addRow(self.output_elevation)
        form.addRow(self.output_latlon)
        form.addRow("Perpendicular baseline (m)", self.baseline)
        form.addRow("Slant range (m)", self.slant)
        form.addRow("Incidence (deg)", self.incidence)
        self.add_processing(form)

    def _browse_dem(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(self, "DEM", "", "NumPy DEM (*.npy)")
        if path:
            self.dem.setCurrentText("External DEM (.npy)")
            self.dem_path.setText(path)

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        if self.dem.currentText().startswith("Copernicus"):
            return _COPERNICUS_NOTE
        if self.dem.currentText().startswith("External") and not self.dem_path.text().strip():
            return "Choose an external DEM .npy, or pick Workspace DEM."
        return ""

    def parameters(self) -> dict:
        return {
            "request": self.io_request(skipped=("Optional topographic, elevation, and lat/lon bands are not written.",)),
            "dem_choice": self.dem.currentText(),
            "dem_path": self.dem_path.text().strip(),
            "baseline_m": float(self.baseline.value()),
            "slant_range_m": float(self.slant.value()),
            "incidence_deg": float(self.incidence.value()),
        }


class GoldsteinDialog(OperatorDialog):
    """Goldstein Phase Filtering. Alpha, FFT size, and spectral window are applied."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        sources: list[str] | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        del config
        super().__init__(
            "Goldstein Phase Filtering",
            OPERATOR_HELP["goldstein"],
            parent=parent,
            sources=sources,
            suffix=PRODUCT_SUFFIX["goldstein"],
            object_name="goldsteinDialog",
            notes=(
                "Adaptive alpha, the FFT block extent in metres, and the coherence mask "
                "are disabled. Alpha, FFT size, and the spectral window width are applied."
            ),
        )
        form = QtWidgets.QFormLayout()
        self.alpha = _double(1.0, 0.0, 1.0, "goldAlpha", decimals=2, step=0.05)
        self.alpha.setToolTip("Adaptive filter exponent in (0, 1]. 0 reconstructs the input in this build.")
        self.adaptive = QtWidgets.QCheckBox("Use adaptive alpha")
        self.adaptive.setObjectName("goldAdaptive")
        _disable(self.adaptive, _ADAPTIVE_NOTE)
        self.alpha_min = _double(0.2, 0.0, 1.0, "goldAlphaMin", decimals=2, step=0.05)
        _disable(self.alpha_min, _ADAPTIVE_NOTE)
        self.alpha_max = _double(1.0, 0.0, 1.0, "goldAlphaMax", decimals=2, step=0.05)
        _disable(self.alpha_max, _ADAPTIVE_NOTE)
        self.fft_size = _combo(("32", "64", "128", "256"), "64", "goldFftSize")
        self.block_extent = _double(0.0, 0.0, 10000.0, "goldBlockExtent", decimals=1, step=1.0)
        _disable(self.block_extent, "FFT block extent in metres is not implemented. FFT size is in pixels.")
        self.window = _combo(("3", "5", "7"), "3", "goldWindow")
        self.window.setToolTip("Spectral smoothing width passed to the in-tree Goldstein filter.")
        self.coherence_mask = QtWidgets.QCheckBox("Use coherence mask")
        self.coherence_mask.setObjectName("goldCoherenceMask")
        _disable(self.coherence_mask, _MASK_NOTE)
        self.coherence_threshold = _double(0.2, 0.0, 1.0, "goldCoherenceThreshold", decimals=2, step=0.05)
        _disable(self.coherence_threshold, _MASK_NOTE)
        form.addRow("Adaptive Filter Exponent in (0, 1]", self.alpha)
        form.addRow(self.adaptive)
        form.addRow("Adaptive alpha minimum", self.alpha_min)
        form.addRow("Adaptive alpha maximum", self.alpha_max)
        form.addRow("FFT Size", self.fft_size)
        form.addRow("FFT Block Extent (m)", self.block_extent)
        form.addRow("Window Size", self.window)
        form.addRow(self.coherence_mask)
        form.addRow("Coherence Threshold in [0, 1]", self.coherence_threshold)
        self.add_processing(form)

    def parameters(self) -> dict:
        return {
            "request": self.io_request(skipped=("Adaptive alpha is not applied.", "Coherence mask is not applied.")),
            "alpha": float(self.alpha.value()),
            "fft_size": int(self.fft_size.currentText()),
            "window_size": int(self.window.currentText()),
        }


class ApplyOrbitDialog(OperatorDialog):
    """Apply Orbit File. The dialog validates; orbit files are not downloaded."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, sources: list[str] | None = None) -> None:
        super().__init__(
            "Apply Orbit File",
            OPERATOR_HELP["apply_orbit"],
            parent=parent,
            sources=sources,
            suffix=PRODUCT_SUFFIX["apply_orbit"],
            object_name="applyOrbitDialog",
            notes=(
                "Orbit files are not downloaded in this build. "
                "Run checks the orbit type and polynomial degree, then leaves state vectors unchanged."
            ),
        )
        form = QtWidgets.QFormLayout()
        self.orbit_type = _combo(
            (
                "Sentinel Precise (Auto Download)",
                "Sentinel Restituted (Auto Download)",
                "DORIS_VOR",
                "DORIS_POR",
                "DELFT_PRECISE_ENVISAT",
                "DELFT_PRECISE_ERS_1",
                "DELFT_PRECISE_ERS_2",
                "PRARE_PRECISE_ERS_1",
                "PRARE_PRECISE_ERS_2",
            ),
            "Sentinel Precise (Auto Download)",
            "orbitType",
        )
        self.degree = QtWidgets.QLineEdit("3")
        self.degree.setObjectName("orbitDegree")
        self.do_not_fail = QtWidgets.QCheckBox("Do not fail if new orbit file is not found")
        self.do_not_fail.setObjectName("orbitDoNotFail")
        form.addRow("Orbit State Vectors", self.orbit_type)
        form.addRow("Polynomial Degree", self.degree)
        form.addRow(self.do_not_fail)
        self.add_processing(form)

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        text = self.degree.text().strip()
        try:
            degree = int(text)
        except ValueError:
            return "Polynomial degree must be an integer."
        if degree < 1:
            return "Polynomial degree must be at least 1."
        if not self.orbit_type.currentText().strip():
            return "Select an orbit type."
        return ""

    def parameters(self) -> dict:
        return {
            "request": self.io_request(),
            "orbit_type": self.orbit_type.currentText(),
            "polynomial_degree": int(self.degree.text().strip()),
            "do_not_fail": self.do_not_fail.isChecked(),
            "source": self.source.currentText(),
        }

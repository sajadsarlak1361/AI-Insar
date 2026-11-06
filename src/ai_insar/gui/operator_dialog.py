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

# Rows actually read from the Interferogram Formation Save-as list in the
# SNAP 14 batch-3 audit. Uncertain spellings and the unscrolled gap are omitted.
# "NumPy (.npy)" is this application's writer; it was not in that SNAP list.
OBSERVED_SAVE_FORMATS = (
    "BEAM-DIMAP",
    "BMP",
    "CSV",
    "ENVI",
    "GDAL-BMP-WRITER",
    "GDAL-BT-WRITER",
    "GDAL-COG-WRITER",
    "GDAL-GTIFF-WRITER",
    "GDAL-HFA-WRITER",
    "GDAL-ILWIS-WRITER",
    "GDAL-KRO-WRITER",
    "GDAL-MFF-WRITER",
    "GDAL-NITF-WRITER",
    "GDAL-RMF-WRITER",
    "GDAL-RST-WRITER",
    "GDAL-SAGA-WRITER",
    "GDAL-SGI-WRITER",
    "GIF",
    "Gamma",
    "Generic Binary BSQ",
    "GeoTIFF",
    "GeoTIFF-BigTIFF",
    "HDF5",
    "JP2",
    "JPEG2000",
    "JPG",
    "NetCDF4-BEAM",
    "NetCDF4-CF",
    "PNG",
    "NumPy (.npy)",
)


@dataclass
class OperatorRequest:
    """Values the main window applies after Run."""

    target: str
    save: bool
    directory: str
    open_in_app: bool
    skipped: tuple[str, ...] = ()
    format_name: str = "NumPy (.npy)"


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
        source_label: str = "Source product:",
        target_text: str | None = None,
        directory_text: str = "",
        formats: tuple[str, ...] | None = None,
        format_default: str | None = None,
        seed_workspace_source: bool = True,
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
        self.tabs.addTab(
            self._build_io(
                sources or [],
                suffix,
                source_label=source_label,
                target_text=target_text,
                directory_text=directory_text,
                formats=formats,
                format_default=format_default,
                seed_workspace_source=seed_workspace_source,
            ),
            "I/O Parameters",
        )
        self._processing = QtWidgets.QWidget()
        self._processing_layout = QtWidgets.QVBoxLayout(self._processing)
        self.tabs.addTab(self._processing, "Processing Parameters")
        root.addWidget(self.tabs, stretch=1)

        self.notes = QtWidgets.QLabel(notes)
        self.notes.setObjectName("operatorNotes")
        self.notes.setWordWrap(True)
        root.addWidget(self.notes)

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

    def _build_io(
        self,
        sources: list[str],
        suffix: str,
        *,
        source_label: str,
        target_text: str | None,
        directory_text: str,
        formats: tuple[str, ...] | None,
        format_default: str | None,
        seed_workspace_source: bool,
    ) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        source_row = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.setObjectName("ioSource")
        self.source.setEditable(True)
        for name in sources:
            self.source.addItem(name)
        if self.source.count() == 0 and seed_workspace_source:
            self.source.addItem("Workspace SLC stack")
        browse = QtWidgets.QPushButton("…")
        browse.setObjectName("ioSourceBrowse")
        browse.setToolTip("Choose a source product file")
        browse.clicked.connect(self._browse_source)
        source_row.addWidget(self.source, stretch=1)
        source_row.addWidget(browse)
        self._source_label = source_label
        form.addRow(source_label, source_row)

        self.target = QtWidgets.QLineEdit(f"target{suffix}" if target_text is None else target_text)
        self.target.setObjectName("ioTarget")
        form.addRow("Name:", self.target)

        save_row = QtWidgets.QHBoxLayout()
        self.save_box = QtWidgets.QCheckBox("Save as")
        self.save_box.setObjectName("ioSave")
        self.save_box.setChecked(True)
        names = formats if formats is not None else ("NumPy (.npy)", "BEAM-DIMAP")
        current = format_default if format_default is not None else names[0]
        disabled = {} if formats is not None else {"BEAM-DIMAP": _DIMAP_NOTE}
        self.format_box = _combo(names, current, "ioFormat", disabled)
        if formats is not None:
            self.format_box.setToolTip(
                "Partial Save-as list read from SNAP 14 Interferogram Formation. "
                "Uncertain rows were omitted. NumPy (.npy) is this build's writer and was not in that list."
            )
        else:
            self.format_box.setToolTip("SNAP's default format is BEAM-DIMAP. This build writes NumPy.")
        save_row.addWidget(self.save_box)
        save_row.addWidget(self.format_box, stretch=1)
        form.addRow(save_row)

        directory_row = QtWidgets.QHBoxLayout()
        self.directory = QtWidgets.QLineEdit(directory_text)
        self.directory.setObjectName("ioDirectory")
        directory_button = QtWidgets.QPushButton("…")
        directory_button.setObjectName("ioDirectoryBrowse")
        directory_button.clicked.connect(self._browse_directory)
        directory_row.addWidget(self.directory, stretch=1)
        directory_row.addWidget(directory_button)
        form.addRow("Directory:", directory_row)

        self.open_box = QtWidgets.QCheckBox("Open in SNAP 14")
        self.open_box.setObjectName("ioOpen")
        self.open_box.setChecked(True)
        self.open_box.setToolTip("Keeps the result in this workspace. The label matches the SNAP 14 dialog.")
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
            "Sentinel-1 SAFE zip (*.zip);;BEAM-DIMAP (*.dim);;NumPy (*.npy);;All files (*)",
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
        if self.save_box.isChecked():
            if not self.target.text().strip():
                return "Enter a target product name."
            written = {"NumPy (.npy)", "BEAM-DIMAP", "GeoTIFF", "Google Earth KMZ"}
            if self.format_box.currentText() not in written:
                return (
                    f"{self.format_box.currentText()} is not written by this build. "
                    "Choose BEAM-DIMAP, GeoTIFF, KMZ, or NumPy (.npy), or uncheck Save as."
                )
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
            format_name=self.format_box.currentText(),
        )

    def _on_run(self) -> None:
        message = self.validate()
        if message:
            self.error.setText(message)
            return
        self.error.setText("")
        self.accept()


class InterferogramDialog(OperatorDialog):
    """Interferogram Formation, using the controls recorded in the SNAP 14 batch-3 audit."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, sources: list[str] | None = None) -> None:
        super().__init__(
            "Interferogram Formation",
            OPERATOR_HELP["interferogram"],
            parent=parent,
            sources=sources,
            suffix=PRODUCT_SUFFIX["interferogram"],
            object_name="interferogramDialog",
            notes=(
                "SNAP 14 batch 3: Subtract Flat-Earth Phase is checked. "
                "Run subtracts the ellipsoid range difference when both radar windows and both precise orbits are loaded. "
                "The DEM list was not opened while Subtract Topographic Phase was unchecked."
            ),
            source_label="Source product:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
        )
        form = QtWidgets.QFormLayout()
        self.subtract_flat = QtWidgets.QCheckBox("Subtract Flat-Earth Phase")
        self.subtract_flat.setObjectName("ifgSubtractFlatEarth")
        self.subtract_flat.setChecked(True)
        self.subtract_flat.setToolTip(
            "Subtracts -4π/λ (R_later − R_early) on the ellipsoid when both radar windows "
            "and both precise orbits are loaded."
        )
        self.flat_degree = _combo(tuple(str(value) for value in range(1, 9)), "5", "ifgFlatDegree")
        self.flat_points = _combo(
            ("301", "401", "501", "601", "701", "801", "901", "1001"),
            "501",
            "ifgFlatPoints",
        )
        self.orbit_degree = _combo(tuple(str(value) for value in range(1, 6)), "3", "ifgOrbitDegree")
        self.output_flat = QtWidgets.QCheckBox("Output Flat Earth Phase")
        self.output_flat.setObjectName("ifgOutputFlatEarth")
        self.subtract_topo = QtWidgets.QCheckBox("Subtract Topographic Phase")
        self.subtract_topo.setObjectName("ifgSubtractTopo")
        self.subtract_ramp = QtWidgets.QCheckBox("Subtract Residual Ramp (GSLC)")
        self.subtract_ramp.setObjectName("ifgSubtractRamp")
        self.dem = QtWidgets.QLineEdit("Copernicus 30m Global DEM (Auto Download)")
        self.dem.setObjectName("ifgDem")
        self.dem.setReadOnly(True)
        _disable(
            self.dem,
            "Shown while Subtract Topographic Phase is unchecked. "
            "The SNAP 14 audit did not open this list, so options are not recorded.",
        )
        self.tile = QtWidgets.QLineEdit("100")
        self.tile.setObjectName("ifgTile")
        self.tile.setReadOnly(True)
        _disable(self.tile, "Shown as 100 and not interactive while Subtract Topographic Phase is unchecked.")
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
        self.coherence_range = QtWidgets.QLineEdit("10")
        self.coherence_range.setObjectName("ifgCohRange")
        self.coherence_azimuth = QtWidgets.QLineEdit("2")
        self.coherence_azimuth.setObjectName("ifgCohAzimuth")
        self.square_pixel.toggled.connect(self._on_square)
        self.independent.toggled.connect(self._on_independent)
        self.subtract_topo.toggled.connect(self._sync_dem)
        self._on_square(True)
        self._sync_dem(False)

        form.addRow(self.subtract_flat)
        form.addRow('Degree of "Flat Earth" polynomial', self.flat_degree)
        form.addRow('Number of "Flat Earth" estimation points', self.flat_points)
        form.addRow("Orbit interpolation degree", self.orbit_degree)
        form.addRow(self.output_flat)
        form.addRow(self.subtract_topo)
        form.addRow(self.subtract_ramp)
        form.addRow("Digital Elevation Model", self.dem)
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

    def _sync_dem(self, checked: bool) -> None:
        # The audit did not toggle Subtract Topographic Phase, so the DEM list stays closed.
        del checked

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
        if self.subtract_ramp.isChecked():
            return "Subtract Residual Ramp (GSLC) is not implemented."
        try:
            if int(self.coherence_range.text().strip()) < 1:
                return "Coherence range window must be at least 1."
        except ValueError:
            return "Coherence range window must be an integer."
        if not self.square_pixel.isChecked():
            try:
                if int(self.coherence_azimuth.text().strip()) < 1:
                    return "Coherence azimuth window must be at least 1."
            except ValueError:
                return "Coherence azimuth window must be an integer."
        return ""

    def parameters(self) -> dict:
        request = self.io_request()
        azimuth = self.coherence_azimuth.text().strip()
        return {
            "request": request,
            "subtract_flat": self.subtract_flat.isChecked(),
            "subtract_topo": self.subtract_topo.isChecked(),
            "subtract_ramp": self.subtract_ramp.isChecked(),
            "output_flat": self.output_flat.isChecked(),
            "dem_choice": self.dem.text(),
            "dem_path": "",
            "output_coherence": self.output_coherence.isChecked(),
            "square_pixel": self.square_pixel.isChecked(),
            "coherence_range": int(self.coherence_range.text().strip() or "0"),
            "coherence_azimuth": int(azimuth or "0"),
            "flat_degree": int(self.flat_degree.currentText()),
            "flat_points": int(self.flat_points.currentText()),
            "orbit_degree": int(self.orbit_degree.currentText()),
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
                "SNAP 14 batch 4: Use adaptive alpha is checked and the numeric fields stay white. "
                "Checked adaptive alpha uses α = clip(1 − block-mean coherence, minimum, maximum). "
                "It is refused when no coherence image exists. A non-zero FFT block extent in metres "
                "is refused because this stack has no pixel spacing."
            ),
            source_label="Source product:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
        )
        form = QtWidgets.QFormLayout()
        self.alpha = _double(1.0, 0.0, 1.0, "goldAlpha", decimals=2, step=0.05)
        self.adaptive = QtWidgets.QCheckBox("Use adaptive alpha")
        self.adaptive.setObjectName("goldAdaptive")
        self.adaptive.setChecked(True)
        self.alpha_min = _double(0.2, 0.0, 1.0, "goldAlphaMin", decimals=2, step=0.05)
        self.alpha_max = _double(1.0, 0.0, 1.0, "goldAlphaMax", decimals=2, step=0.05)
        self.fft_size = _combo(("32", "64", "128", "256"), "64", "goldFftSize")
        self.block_extent = _double(0.0, 0.0, 10000.0, "goldBlockExtent", decimals=1, step=1.0)
        self.window = _combo(("3", "5", "7"), "3", "goldWindow")
        self.coherence_mask = QtWidgets.QCheckBox("Use coherence mask")
        self.coherence_mask.setObjectName("goldCoherenceMask")
        self.coherence_mask.setChecked(False)
        self.coherence_threshold = _double(0.2, 0.0, 1.0, "goldCoherenceThreshold", decimals=2, step=0.05)
        form.addRow("Adaptive Filter Exponent in (0,1]:", self.alpha)
        form.addRow(self.adaptive)
        form.addRow("Adaptive alpha minimum:", self.alpha_min)
        form.addRow("Adaptive alpha maximum:", self.alpha_max)
        form.addRow("FFT Size:", self.fft_size)
        form.addRow("FFT Block Extent (m):", self.block_extent)
        form.addRow("Window Size:", self.window)
        form.addRow(self.coherence_mask)
        form.addRow("Coherence Threshold in [0,1]:", self.coherence_threshold)
        self.add_processing(form)

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        if float(self.block_extent.value()) != 0.0:
            return (
                "FFT Block Extent (m) is not zero. This stack has no pixel spacing, "
                "so a metre block size is not converted."
            )
        if float(self.alpha_min.value()) > float(self.alpha_max.value()):
            return "Adaptive alpha minimum is above the maximum."
        return ""

    def parameters(self) -> dict:
        return {
            "request": self.io_request(),
            "alpha": float(self.alpha.value()),
            "adaptive": self.adaptive.isChecked(),
            "alpha_min": float(self.alpha_min.value()),
            "alpha_max": float(self.alpha_max.value()),
            "fft_size": int(self.fft_size.currentText()),
            "block_extent_m": float(self.block_extent.value()),
            "window_size": int(self.window.currentText()),
            "coherence_mask": self.coherence_mask.isChecked(),
            "coherence_threshold": float(self.coherence_threshold.value()),
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
                "SNAP 14 batch 3 recorded these eight orbit types. "
                "A local EOF in 'Local orbit file' is applied to the Source product. "
                "The other date stays as it is. With that path empty, Run fails unless "
                "'Do not fail if new orbit file is not found' is checked."
            ),
            source_label="source:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
        )
        form = QtWidgets.QFormLayout()
        self.orbit_type = _combo(
            (
                "Sentinel Precise (Auto Download)",
                "Sentinel Restituted (Auto Download)",
                "DORIS Preliminary POR (ENVISAT)",
                "DORIS Precise VOR (ENVISAT) (Auto Download)",
                "DELFT Precise (ENVISAT, ERS1&2) (Auto Download)",
                "PRARE Precise (ERS1&2) (Auto Download)",
                "Kompsat5 Precise",
                "Biomass Precise (MAAP) (Auto Download)",
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
        file_row = QtWidgets.QHBoxLayout()
        self.orbit_file = QtWidgets.QLineEdit("")
        self.orbit_file.setObjectName("orbitLocalFile")
        self.orbit_file.setPlaceholderText("Local Sentinel Precise .EOF or .EOF.zip")
        browse = QtWidgets.QPushButton("…")
        browse.setObjectName("orbitLocalBrowse")
        browse.setToolTip("Choose a precise or restituted orbit file. Nothing is downloaded.")
        browse.clicked.connect(self._browse_orbit_file)
        file_row.addWidget(self.orbit_file, stretch=1)
        file_row.addWidget(browse)
        form.addRow("Local orbit file:", file_row)
        self.add_processing(form)

    def _browse_orbit_file(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Orbit file",
            "",
            "Sentinel orbit (*.EOF *.eof *.zip);;All files (*)",
        )
        if path:
            self.orbit_file.setText(path)

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
            "orbit_file": self.orbit_file.text().strip(),
        }

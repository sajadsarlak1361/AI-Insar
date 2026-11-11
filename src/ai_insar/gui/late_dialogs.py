"""Phase to Displacement, terrain correction, subset, and save.

Phase to Displacement has a blank processing panel when no source bands are
bound; the conversion is still ``d = -(λ/4π) Δφ`` on unwrapped radians.
Terrain-correction lists are the ones opened in the SNAP 14 batch-5 audit.
Subset does not copy the native spinner exception. Save writes NumPy,
GeoTIFF, or KMZ and does not report a BEAM-DIMAP product.
"""

from __future__ import annotations

from ai_insar.gui.operator_dialog import OBSERVED_SAVE_FORMATS, OperatorDialog
from ai_insar.gui.qt import QtWidgets
from ai_insar.snap_products import OPERATOR_HELP, PRODUCT_SUFFIX

_DEM_OPTIONS = (
    "ACE2_5Min (Auto Download)",
    "ACE30 (Auto Download)",
    "ASTER 1sec GDEM",
    "CDEM (Auto Download)",
    "Copernicus 30m Global DEM (Auto Download)",
    "Copernicus 90m Global DEM (Auto Download)",
    "GETASSE30 (Auto Download)",
    "SRTM 1Sec Grid",
    "SRTM 1Sec HGT (Auto Download)",
    "SRTM 3Sec (Auto Download)",
    "External DEM",
    "WGS84 ellipsoid (not terrain correction)",
)
_DEM_RESAMPLING = (
    "NEAREST_NEIGHBOUR",
    "BILINEAR_INTERPOLATION",
    "CUBIC_CONVOLUTION",
    "BISINC_5_POINT_INTERPOLATION",
    "BISINC_11_POINT_INTERPOLATION",
    "BISINC_21_POINT_INTERPOLATION",
    "BICUBIC_INTERPOLATION",
    "DELAUNAY_INTERPOLATION",
)
_IMAGE_RESAMPLING = _DEM_RESAMPLING[:-1]


def _combo(items: tuple[str, ...], current: str, name: str) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox()
    box.setObjectName(name)
    box.addItems(list(items))
    box.setCurrentText(current)
    return box


class PhaseToDisplacementDialog(OperatorDialog):
    """Blank processing panel. The equation is applied to unwrapped phase."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, sources: list[str] | None = None) -> None:
        super().__init__(
            "Phase to Displacement",
            OPERATOR_HELP["displacement"],
            parent=parent,
            sources=sources,
            suffix=PRODUCT_SUFFIX["displacement"],
            object_name="phaseToDisplacementDialog",
            notes=(
                "The processing panel has no extra controls. "
                "Run converts unwrapped radians with d = -(λ/4π) Δφ. "
                "λ comes from the SLC metadata. A raw SAFE is not unwrapped phase."
            ),
            source_label="Source:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
        )
        form = QtWidgets.QFormLayout()
        blank = QtWidgets.QLabel("")
        blank.setObjectName("phaseProcessingBlank")
        form.addRow(blank)
        self.add_processing(form)

    def parameters(self) -> dict:
        return {"request": self.io_request()}

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        return ""


class TerrainCorrectionDialog(OperatorDialog):
    """Range-Doppler terrain correction. Only bilinear WGS84 output is produced."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, sources: list[str] | None = None) -> None:
        super().__init__(
            "Range Doppler Terrain Correction",
            "Each output latitude and longitude is located by zero-Doppler range and bilinearly resampled. "
            "Auto-download DEMs are not fetched. External DEM or ellipsoid height 0 is used.",
            parent=parent,
            sources=sources,
            suffix="_TC",
            object_name="terrainCorrectionDialog",
            notes=(
                "Auto-download DEMs are not fetched. "
                "Bilinear resampling onto WGS84 geographic coordinates is implemented."
            ),
            source_label="Source:",
            target_text="",
            directory_text="/home/box",
            formats=("BEAM-DIMAP", "GeoTIFF", "NumPy (.npy)", "Google Earth KMZ"),
            format_default="BEAM-DIMAP",
        )
        form = QtWidgets.QFormLayout()
        self.bands = QtWidgets.QListWidget()
        self.bands.setObjectName("tcSourceBands")
        self.dem = _combo(_DEM_OPTIONS, "Copernicus 30m Global DEM (Auto Download)", "tcDem")
        self.dem_resampling = _combo(_DEM_RESAMPLING, "BILINEAR_INTERPOLATION", "tcDemResampling")
        self.image_resampling = _combo(_IMAGE_RESAMPLING, "BILINEAR_INTERPOLATION", "tcImageResampling")
        self.spacing_m = QtWidgets.QLineEdit("")
        self.spacing_m.setObjectName("tcSpacingM")
        self.spacing_deg = QtWidgets.QLineEdit("")
        self.spacing_deg.setObjectName("tcSpacingDeg")
        self.dem_path = QtWidgets.QLineEdit("")
        self.dem_path.setObjectName("tcDemPath")
        self.projection = QtWidgets.QLabel("WGS84(DD)")
        self.projection.setObjectName("tcProjection")
        self.mask_elevation = QtWidgets.QCheckBox("Mask out areas without elevation")
        self.mask_elevation.setObjectName("tcMaskElevation")
        self.mask_elevation.setChecked(True)
        self.output_complex = QtWidgets.QCheckBox("Output complex data")
        self.output_complex.setObjectName("tcOutputComplex")
        form.addRow("Source Bands:", self.bands)
        form.addRow("Digital Elevation Model:", self.dem)
        form.addRow("External DEM:", self.dem_path)
        form.addRow("DEM Resampling Method:", self.dem_resampling)
        form.addRow("Image Resampling Method:", self.image_resampling)
        form.addRow("Pixel Spacing (m):", self.spacing_m)
        form.addRow("Pixel Spacing (deg):", self.spacing_deg)
        form.addRow("Map Projection:", self.projection)
        form.addRow(self.mask_elevation)
        form.addRow(self.output_complex)
        self.geo_region = QtWidgets.QPlainTextEdit()
        self.geo_region.setObjectName("tcGeoRegion")
        self.geo_region.setFixedHeight(64)
        self.geo_region.setPlaceholderText("POLYGON ((lon lat, ...))  optional output geoRegion")
        form.addRow("Output geoRegion:", self.geo_region)
        self.add_processing(form)

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        dem_method = self.dem_resampling.currentText()
        if dem_method not in {"NEAREST_NEIGHBOUR", "BILINEAR_INTERPOLATION"}:
            return f"DEM resampling {dem_method} is not applied."
        text = self.geo_region.toPlainText().strip()
        if text:
            from ai_insar.pipeline.aoi import AoiError, parse_wkt

            try:
                parse_wkt(text)
            except AoiError as exc:
                return str(exc)
        return ""

    def parameters(self) -> dict:
        def _optional(text: str) -> float | None:
            stripped = text.strip()
            if not stripped:
                return None
            return float(stripped)

        return {
            "request": self.io_request(),
            "dem": self.dem.currentText(),
            "dem_path": self.dem_path.text().strip(),
            "dem_resampling": self.dem_resampling.currentText(),
            "image_resampling": self.image_resampling.currentText(),
            "spacing_m": _optional(self.spacing_m.text()),
            "spacing_deg": _optional(self.spacing_deg.text()),
            "mask_elevation": self.mask_elevation.isChecked(),
            "output_complex": self.output_complex.isChecked(),
            "projection": self.projection.text(),
            "geoRegion": self.geo_region.toPlainText().strip(),
        }


class SubsetDialog(QtWidgets.QDialog):
    """SNAP Subset: pixel region, geographic region, bands, and copy metadata.

    Scene end coordinates are inclusive, matching the Subset dialog. The
    returned ``region`` is ``x, y, width, height``. ``row_end`` and
    ``col_end`` stay exclusive so the existing crop still uses slices.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        height: int = 1,
        width: int = 1,
        bands: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Subset")
        self.setObjectName("subsetDialog")
        self.resize(520, 560)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Crops the in-memory product. The source SAFE or TIFF is not modified. "
            "Pixel Coordinates use inclusive scene end indexes. "
            "Geo Coordinates take a WKT polygon or a latitude/longitude box."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        rows = max(int(height), 1)
        cols = max(int(width), 1)
        pixel = QtWidgets.QFormLayout()
        self.scene_start_x = self._spin("subsetSceneStartX", cols - 1, 0)
        self.scene_start_y = self._spin("subsetSceneStartY", rows - 1, 0)
        self.scene_end_x = self._spin("subsetSceneEndX", cols - 1, cols - 1)
        self.scene_end_y = self._spin("subsetSceneEndY", rows - 1, rows - 1)
        self.sub_sampling_x = self._spin("subsetSubSamplingX", 100, 1, minimum=1)
        self.sub_sampling_y = self._spin("subsetSubSamplingY", 100, 1, minimum=1)
        pixel.addRow("Scene start X:", self.scene_start_x)
        pixel.addRow("Scene start Y:", self.scene_start_y)
        pixel.addRow("Scene end X:", self.scene_end_x)
        pixel.addRow("Scene end Y:", self.scene_end_y)
        pixel.addRow("Subsampling X:", self.sub_sampling_x)
        pixel.addRow("Subsampling Y:", self.sub_sampling_y)
        geo = QtWidgets.QFormLayout()
        self.north = self._line("subsetNorth")
        self.west = self._line("subsetWest")
        self.south = self._line("subsetSouth")
        self.east = self._line("subsetEast")
        self.geo_region = QtWidgets.QPlainTextEdit()
        self.geo_region.setObjectName("subsetGeoRegion")
        self.geo_region.setFixedHeight(72)
        self.geo_region.setPlaceholderText("POLYGON ((lon lat, lon lat, ...))")
        geo.addRow("North latitude bound:", self.north)
        geo.addRow("West longitude bound:", self.west)
        geo.addRow("South latitude bound:", self.south)
        geo.addRow("East longitude bound:", self.east)
        geo.addRow("geoRegion:", self.geo_region)
        self.bands = QtWidgets.QListWidget()
        self.bands.setObjectName("subsetSourceBands")
        self.bands.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        for name in bands or []:
            item = QtWidgets.QListWidgetItem(name)
            self.bands.addItem(item)
            item.setSelected(True)
        self.copy_metadata = QtWidgets.QCheckBox("Copy metadata")
        self.copy_metadata.setObjectName("subsetCopyMetadata")
        self.copy_metadata.setChecked(True)
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("subsetTabs")
        pixel_page = QtWidgets.QWidget()
        pixel_page.setLayout(pixel)
        geo_page = QtWidgets.QWidget()
        geo_page.setLayout(geo)
        band_page = QtWidgets.QWidget()
        band_layout = QtWidgets.QVBoxLayout(band_page)
        band_layout.addWidget(self.bands)
        tabs.addTab(pixel_page, "Pixel Coordinates")
        tabs.addTab(geo_page, "Geo Coordinates")
        tabs.addTab(band_page, "Bands")
        layout.addWidget(tabs)
        layout.addWidget(self.copy_metadata)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._error = QtWidgets.QLabel("")
        self._error.setObjectName("subsetError")
        self._error.setWordWrap(True)
        layout.addWidget(self._error)
        for box in (self.north, self.south, self.west, self.east):
            box.editingFinished.connect(self._bounds_to_wkt)

    def _spin(self, name: str, maximum: int, value: int, *, minimum: int = 0) -> QtWidgets.QSpinBox:
        box = QtWidgets.QSpinBox()
        box.setObjectName(name)
        box.setRange(minimum, max(maximum, minimum))
        box.setValue(value)
        return box

    def _line(self, name: str) -> QtWidgets.QLineEdit:
        box = QtWidgets.QLineEdit()
        box.setObjectName(name)
        return box

    def _bounds_to_wkt(self) -> None:
        from ai_insar.pipeline.aoi import AoiError, format_wkt, polygon_from_bounds

        try:
            polygon = polygon_from_bounds(
                float(self.north.text()),
                float(self.south.text()),
                float(self.west.text()),
                float(self.east.text()),
            )
        except (AoiError, ValueError):
            return
        self.geo_region.setPlainText(format_wkt(polygon))

    def _accept(self) -> None:
        if self.scene_start_x.value() > self.scene_end_x.value() or self.scene_start_y.value() > self.scene_end_y.value():
            self._error.setText("The subset is empty. Scene end must be greater than or equal to scene start.")
            return
        text = self.geo_region.toPlainText().strip()
        if text:
            from ai_insar.pipeline.aoi import AoiError, parse_wkt

            try:
                parse_wkt(text)
            except AoiError as exc:
                self._error.setText(str(exc))
                return
        self.accept()

    def parameters(self) -> dict:
        start_x = int(self.scene_start_x.value())
        start_y = int(self.scene_start_y.value())
        end_x = int(self.scene_end_x.value())
        end_y = int(self.scene_end_y.value())
        width = end_x - start_x + 1
        height = end_y - start_y + 1
        selected = [item.text() for item in self.bands.selectedItems()]
        return {
            "region": {"x": start_x, "y": start_y, "width": width, "height": height},
            "geoRegion": self.geo_region.toPlainText().strip(),
            "subSamplingX": int(self.sub_sampling_x.value()),
            "subSamplingY": int(self.sub_sampling_y.value()),
            "sourceBands": selected,
            "copyMetadata": self.copy_metadata.isChecked(),
            "scene_start_x": start_x,
            "scene_start_y": start_y,
            "scene_end_x": end_x,
            "scene_end_y": end_y,
            "row_start": start_y,
            "row_end": start_y + height,
            "col_start": start_x,
            "col_end": start_x + width,
        }


class SaveProductDialog(QtWidgets.QDialog):
    """Write a product this build can actually read back. BEAM-DIMAP is refused."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Product As")
        self.setObjectName("saveProductDialog")
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "SNAP asks to convert a SAFE to BEAM-DIMAP before saving. "
            "This build does not write BEAM-DIMAP. "
            "NumPy, GeoTIFF, and KMZ are written as selected."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.target = QtWidgets.QLineEdit("product")
        self.target.setObjectName("saveTarget")
        self.format_box = _combo(("NumPy (.npy)", "GeoTIFF", "Google Earth KMZ"), "NumPy (.npy)", "saveFormat")
        self.directory = QtWidgets.QLineEdit("")
        self.directory.setObjectName("saveDirectory")
        self.band = _combo(("SLC", "Interferogram", "Unwrapped phase", "Displacement", "Geocoded"), "SLC", "saveBand")
        form.addRow("Name:", self.target)
        form.addRow("Save as:", self.format_box)
        form.addRow("Directory:", self.directory)
        form.addRow("Band:", self.band)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def parameters(self) -> dict:
        return {
            "target": self.target.text().strip() or "product",
            "format_name": self.format_box.currentText(),
            "directory": self.directory.text().strip(),
            "band": self.band.currentText(),
        }

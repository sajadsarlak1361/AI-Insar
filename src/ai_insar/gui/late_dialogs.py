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
        self.add_processing(form)

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        dem_method = self.dem_resampling.currentText()
        if dem_method not in {"NEAREST_NEIGHBOUR", "BILINEAR_INTERPOLATION"}:
            return f"DEM resampling {dem_method} is not applied."
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
        }


class SubsetDialog(QtWidgets.QDialog):
    """Line and sample subset. Bounds always satisfy minimum <= value <= maximum."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, height: int = 1, width: int = 1) -> None:
        super().__init__(parent)
        self.setWindowTitle("Subset")
        self.setObjectName("subsetDialog")
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Crops the in-memory product. The source SAFE or TIFF is not modified. "
            "Line and sample numbers are 0-based and the end is exclusive."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        rows = max(int(height), 1)
        cols = max(int(width), 1)
        self.row_start = QtWidgets.QSpinBox()
        self.row_end = QtWidgets.QSpinBox()
        self.col_start = QtWidgets.QSpinBox()
        self.col_end = QtWidgets.QSpinBox()
        for box, limit, name in (
            (self.row_start, rows, "subsetRowStart"),
            (self.row_end, rows, "subsetRowEnd"),
            (self.col_start, cols, "subsetColStart"),
            (self.col_end, cols, "subsetColEnd"),
        ):
            box.setObjectName(name)
            box.setRange(0, limit)
        self.row_start.setValue(0)
        self.row_end.setValue(rows)
        self.col_start.setValue(0)
        self.col_end.setValue(cols)
        form.addRow("First line", self.row_start)
        form.addRow("Last line", self.row_end)
        form.addRow("First sample", self.col_start)
        form.addRow("Last sample", self.col_end)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._error = QtWidgets.QLabel("")
        self._error.setObjectName("subsetError")
        layout.addWidget(self._error)

    def _accept(self) -> None:
        if self.row_start.value() >= self.row_end.value() or self.col_start.value() >= self.col_end.value():
            self._error.setText("The subset is empty. Last line and last sample must be greater than the first.")
            return
        self.accept()

    def parameters(self) -> dict:
        return {
            "row_start": int(self.row_start.value()),
            "row_end": int(self.row_end.value()),
            "col_start": int(self.col_start.value()),
            "col_end": int(self.col_end.value()),
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

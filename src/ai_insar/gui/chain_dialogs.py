"""SNAP 14 batch-3 dialogs for TOPS Split, Deburst, ESD, and Back Geocoding.

Control defaults and option lists are the ones read on the installed SNAP 14
desktop with an empty Product Explorer. Data-dependent fields stay empty
until a source product is bound. Run does not substitute another algorithm.
"""

from __future__ import annotations

from ai_insar.gui.operator_dialog import OBSERVED_SAVE_FORMATS, OperatorDialog
from ai_insar.gui.product_set import ProductSetWidget
from ai_insar.gui.qt import QtWidgets
from ai_insar.safe import SafeProduct
from ai_insar.snap_products import OPERATOR_HELP, PRODUCT_SUFFIX

_BACKGEO_BLOCK = (
    "S-1 Back Geocoding was not run. Range-Doppler coregistration needs the reference "
    "and the secondary as split radar windows, with an orbit for each date. "
    "A SAFE crop is not this operator, and DEM auto-download is not fetched."
)
_ESD_BLOCK = (
    "S-1 Enhanced Spectral Diversity was not run. Burst-overlap spectral diversity "
    "needs the two burst looks from a coregistered pair. No overlap is loaded."
)
_DEBURST_BLOCK = (
    "S-1 TOPS Deburst was not run. It needs a split swath so the overlap can be cut "
    "at the mid azimuth time. No split is loaded, so this is not a deburst."
)

_DEM_OPTIONS = (
    "ACE2_5Min (Auto Download)",
    "ACE30 (Auto Download)",
    "ASTER 1sec GDEM",
    "CDEM (Auto Download)",
    "Copernicus 30m Global DEM (Auto Download)",
    "Copernicus 90m Global DEM (Auto Download)",
    "GETASSE30 (Auto Download)",
    "SRTM 1Sec Grid",
    "SRTM 15Sec HGT (Auto Download)",
    "SRTM 3Sec (Auto Download)",
    "External DEM",
)
_RESAMPLING = (
    "NEAREST_NEIGHBOUR",
    "BILINEAR_INTERPOLATION",
    "CUBIC_CONVOLUTION",
    "BISINC_5_POINT_INTERPOLATION",
    "BISINC_11_POINT_INTERPOLATION",
    "BISINC_21_POINT_INTERPOLATION",
    "BICUBIC_INTERPOLATION",
)

_ESD_COMBOS = (
    ("Registration Window Width", "512", ("32", "64", "128", "256", "512", "1024", "2048"), "esdWinWidth"),
    ("Registration Window Height", "512", ("32", "64", "128", "256", "512", "1024", "2048"), "esdWinHeight"),
    ("Search Window Accuracy in Azimuth Direction", "16", ("2", "4", "8", "16", "32", "64"), "esdAzAccuracy"),
    ("Search Window Accuracy in Range Direction", "16", ("2", "4", "8", "16", "32", "64"), "esdRgAccuracy"),
    ("Window oversampling factor", "128", ("32", "64", "128", "256"), "esdOversample"),
    ("ESD Estimator", "Periodogram", ("Average", "Periodogram"), "esdEstimator"),
    ("Weight function", "Inv Quadratic", ("None", "Linear", "Quadratic", "Inv Quadratic"), "esdWeight"),
    ("Temporal baseline type", "Number of images", ("Number of images", "Number of days"), "esdBaselineType"),
    ("Integration method", "L1 and L2", ("L1", "L2", "L1 and L2"), "esdIntegration"),
)
_ESD_TEXTS = (
    ("Cross-Correlation Threshold", "0.1", "esdXcorr"),
    ("Coherence Threshold for Outlier Removal", "0.3", "esdCohThreshold"),
    ("Number of Windows Per Overlap for ESD", "10", "esdWindows"),
    ("Maximum temporal baseline (inclusive)", "4", "esdMaxBaseline"),
    ("The overall range shift in pixels", "0.0", "esdRangeShift"),
    ("The overall azimuth shift in pixels", "0.0", "esdAzimuthShift"),
)
_ESD_CHECKS = (
    ("Do not write target bands (store range and azimuth offsets in json files).", "esdNoWrite"),
    ("Use user supplied range shift (please enter it below)", "esdUserRange"),
    ("Use user supplied azimuth shift (please enter it below)", "esdUserAzimuth"),
)


def _combo(items: tuple[str, ...], current: str, name: str) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox()
    box.setObjectName(name)
    box.addItems(list(items))
    box.setCurrentText(current)
    return box


_SPLIT_EMPTY_NOTE = (
    "With no source, Subswath is an empty combo and Bursts reads "
    "'1 to 9999 (max number of bursts: 9999)'. IW names are not shown until a product is bound."
)


class TopsSplitDialog(OperatorDialog):
    """S-1 TOPS Split. Subswath and polarisations stay empty until a product is bound."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(
            "S-1 TOPS Split",
            "TOPS Split keeps one subswath, polarisation, and burst range. It does not deburst.",
            suffix=PRODUCT_SUFFIX.get("split", "_split"),
            object_name="topsSplitDialog",
            notes=_SPLIT_EMPTY_NOTE,
            parent=parent,
            source_label="source:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
            seed_workspace_source=False,
        )
        self._product: SafeProduct | None = None
        form = QtWidgets.QFormLayout()
        self.subswath = QtWidgets.QComboBox()
        self.subswath.setObjectName("splitSubswath")
        self.pol_label = QtWidgets.QLabel("")
        self.pol_label.setObjectName("splitPolarisations")
        self.pol_list = QtWidgets.QListWidget()
        self.pol_list.setObjectName("splitPolarisationList")
        self.pol_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.burst_text = QtWidgets.QLabel("1 to 9999 (max number of bursts: 9999)")
        self.burst_text.setObjectName("splitBurstText")
        self.burst_first = QtWidgets.QSpinBox()
        self.burst_first.setObjectName("splitBurstFirst")
        self.burst_last = QtWidgets.QSpinBox()
        self.burst_last.setObjectName("splitBurstLast")
        for box in (self.burst_first, self.burst_last):
            box.setRange(1, 9999)
            # Apply the typed burst when the box loses focus. Updating on every
            # digit keeps a 1..9999 box resizing and stalls entry.
            box.setKeyboardTracking(False)
        self.burst_first.setValue(1)
        self.burst_last.setValue(9999)
        self.subswath.currentTextChanged.connect(lambda _text: self._refresh_bursts())
        self.source.currentIndexChanged.connect(self._on_source_index)
        line = self.source.lineEdit()
        if line is not None:
            line.editingFinished.connect(self._on_source_finished)
        form.addRow("Subswath:", self.subswath)
        form.addRow("Polarisations:", self.pol_label)
        form.addRow(self.pol_list)
        form.addRow("Bursts:", self.burst_text)
        burst_row = QtWidgets.QHBoxLayout()
        burst_row.addWidget(self.burst_first)
        burst_row.addWidget(QtWidgets.QLabel("to"))
        burst_row.addWidget(self.burst_last)
        form.addRow(burst_row)
        self.add_processing(form)

    def _on_source_index(self, index: int) -> None:
        """A listed SAFE is already parsed. Bind it without reading the product again."""
        if getattr(self, "_binding", False) or index < 0:
            return
        cached = self.source.itemData(index)
        text = self.source.itemText(index).strip()
        if text and self.source.currentText().strip() != text:
            return
        from ai_insar.safe import SafeProduct

        if isinstance(cached, SafeProduct):
            self.error.setText("")
            self.bind_product(cached)
            return
        if text:
            self._resolve_source_text(text)

    def _on_source_finished(self) -> None:
        """Resolve a typed SAFE folder when the field is left, not on each key."""
        if getattr(self, "_binding", False):
            return
        text = self.source.currentText().strip()
        if self._product is not None and str(self._product.root) == text:
            self.error.setText("")
            self._set_bound_note()
            return
        self._resolve_source_text(text)

    def _resolve_source_text(self, text: str) -> None:
        """Bind a SAFE folder or zip. A missing path does not change the current product."""
        from pathlib import Path

        from ai_insar.safe import SafeProduct, resolve_safe

        raw = text.strip()
        if not raw:
            return
        index = self.source.findText(raw)
        if index >= 0:
            cached = self.source.itemData(index)
            if isinstance(cached, SafeProduct):
                self.error.setText("")
                self.bind_product(cached)
                return
        path = Path(raw)
        if not path.exists():
            return
        try:
            product = resolve_safe(path)
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))
            if self._product is None:
                self.notes.setText(_SPLIT_EMPTY_NOTE)
            return
        self.error.setText("")
        self.bind_product(product)

    def bind_product(self, product: SafeProduct) -> None:
        """Fill subswath, polarisation, and burst count from this product only."""
        self._binding = True
        try:
            self._bind_product(product)
        finally:
            self._binding = False

    def _bind_product(self, product: SafeProduct) -> None:
        self._product = product
        self.error.setText("")
        if self.source.currentText().strip() != str(product.root):
            self.source.setCurrentText(str(product.root))
        listed = self.source.findText(str(product.root))
        if listed >= 0 and not isinstance(self.source.itemData(listed), SafeProduct):
            self.source.setItemData(listed, product)
        elif listed < 0:
            self.source.addItem(str(product.root), product)
        self.subswath.blockSignals(True)
        self.subswath.clear()
        # The loaded-SAFE audit shows a blank initial subswath, not IW1.
        self.subswath.addItem("")
        self.subswath.addItems(list(product.swaths))
        self.subswath.setCurrentIndex(0)
        self.subswath.blockSignals(False)
        self.pol_list.clear()
        self.pol_list.addItems(list(product.polarisations))
        self.pol_label.setText(", ".join(product.polarisations))
        stem = product.product_id
        if stem.upper().endswith(".SAFE"):
            stem = stem[:-5]
        self.target.setText(f"{stem}_split")
        self._refresh_bursts()
        self._set_bound_note()

    def _set_bound_note(self) -> None:
        product = self._product
        if product is None:
            self.notes.setText(_SPLIT_EMPTY_NOTE)
            return
        swath = self.subswath.currentText().strip()
        swaths = ", ".join(product.swaths) or "none"
        if swath:
            self.notes.setText(
                f"Source {product.product_id} is bound. {swath} bursts are "
                f"{self.burst_text.text()}. This is not a deburst."
            )
            return
        self.notes.setText(
            f"Source {product.product_id} is bound. Swaths: {swaths}. "
            "Choose a subswath to set its burst count. This is not a deburst."
        )

    def _max_bursts(self) -> int:
        product = self._product
        if product is None:
            return 0
        counts = [len(measurement.bursts) for measurement in product.measurements if measurement.bursts]
        return max(counts) if counts else 0

    def _refresh_bursts(self) -> None:
        product = self._product
        swath = self.subswath.currentText().strip()
        count = 0
        if product is not None and swath:
            for measurement in product.measurements:
                if measurement.swath.upper() == swath.upper() and measurement.bursts:
                    count = len(measurement.bursts)
                    break
        if count < 1:
            count = self._max_bursts()
        if count < 1:
            self.burst_text.setText("1 to 9999 (max number of bursts: 9999)")
            self.burst_first.setRange(1, 9999)
            self.burst_last.setRange(1, 9999)
            self.burst_first.setValue(1)
            self.burst_last.setValue(9999)
            self._set_bound_note()
            return
        previous_first = int(self.burst_first.value())
        previous_last = int(self.burst_last.value())
        self.burst_text.setText(f"1 to {count} (max number of bursts: {count})")
        self.burst_first.setRange(1, count)
        self.burst_last.setRange(1, count)
        if 1 <= previous_first <= previous_last <= count:
            self.burst_first.setValue(previous_first)
            self.burst_last.setValue(previous_last)
        else:
            self.burst_first.setValue(1)
            self.burst_last.setValue(count)
        self._set_bound_note()

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        if self.subswath.count() == 0 or not self.subswath.currentText().strip():
            return "Subswath is empty. Bind a Sentinel-1 source before Run."
        if self.pol_list.currentItem() is None:
            return "Select one polarisation."
        if self.burst_first.value() > self.burst_last.value():
            return "Burst range is reversed."
        if "9999" in self.burst_text.text() and self._product is None:
            return "Burst count is the empty-product placeholder. Bind a source before Run."
        return ""

    def parameters(self) -> dict:
        item = self.pol_list.currentItem()
        return {
            "request": self.io_request(),
            "source": self.source.currentText().strip(),
            "swath": self.subswath.currentText().strip(),
            "polarisation": "" if item is None else item.text(),
            "burst_first": int(self.burst_first.value()),
            "burst_last": int(self.burst_last.value()),
        }


class TopsDeburstDialog(OperatorDialog):
    """S-1 TOPS Deburst. The only processing control is an empty polarisation list."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, can_run: bool = False) -> None:
        self._can_run = can_run
        super().__init__(
            "S-1 TOPS Deburst",
            "Deburst keeps lines up to the mid azimuth time of each burst overlap and drops the rest.",
            suffix=PRODUCT_SUFFIX["deburst"],
            object_name="topsDeburstDialog",
            notes=_DEBURST_BLOCK,
            parent=parent,
            source_label="source:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
            seed_workspace_source=False,
        )
        form = QtWidgets.QFormLayout()
        self.pol_label = QtWidgets.QLabel("Polarisations:")
        self.pol_label.setObjectName("deburstPolarisationLabel")
        self.pol_list = QtWidgets.QListWidget()
        self.pol_list.setObjectName("deburstPolarisations")
        form.addRow(self.pol_label)
        form.addRow(self.pol_list)
        self.add_processing(form)

    def bind_product(self, product: SafeProduct) -> None:
        self.source.setCurrentText(str(product.root))
        self.pol_list.clear()
        self.pol_list.addItems(list(product.polarisations))

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        if not self._can_run:
            return _DEBURST_BLOCK
        return ""

    def block_reason(self) -> str:
        return _DEBURST_BLOCK

    def parameters(self) -> dict:
        item = self.pol_list.currentItem()
        return {
            "request": self.io_request(),
            "polarisation": "" if item is None else item.text(),
        }


class EsdDialog(OperatorDialog):
    """S-1 Enhanced Spectral Diversity. Eighteen processing controls, SNAP 14 defaults."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        sources: list[str] | None = None,
        can_run: bool = False,
    ) -> None:
        self._can_run = can_run
        super().__init__(
            "S-1 Enhanced Spectral Diversity",
            "ESD estimates one azimuth shift from the burst-overlap double difference.",
            suffix="_ESD",
            object_name="esdDialog",
            notes=_ESD_BLOCK,
            parent=parent,
            sources=sources,
            source_label="source:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
            seed_workspace_source=False,
        )
        form = QtWidgets.QFormLayout()
        self.controls: list[QtWidgets.QWidget] = []
        for label, default, options, name in _ESD_COMBOS:
            box = _combo(options, default, name)
            form.addRow(label, box)
            self.controls.append(box)
        for label, default, name in _ESD_TEXTS:
            field = QtWidgets.QLineEdit(default)
            field.setObjectName(name)
            form.addRow(label, field)
            self.controls.append(field)
        for label, name in _ESD_CHECKS:
            box = QtWidgets.QCheckBox(label)
            box.setObjectName(name)
            box.setChecked(False)
            form.addRow(box)
            self.controls.append(box)
        self.add_processing(form)

    def processing_control_count(self) -> int:
        return len(self.controls)

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        for name in ("esdXcorr", "esdCohThreshold", "esdRangeShift", "esdAzimuthShift"):
            field = self.findChild(QtWidgets.QLineEdit, name)
            if field is None:
                return f"Missing {name}."
            try:
                float(field.text().strip())
            except ValueError:
                return f"{name} must be a number."
        if not self._can_run:
            return _ESD_BLOCK
        return ""

    def block_reason(self) -> str:
        return _ESD_BLOCK

    def parameters(self) -> dict:
        estimator = self.findChild(QtWidgets.QComboBox, "esdEstimator")
        user_azimuth = self.findChild(QtWidgets.QCheckBox, "esdUserAzimuth")
        azimuth = self.findChild(QtWidgets.QLineEdit, "esdAzimuthShift")
        return {
            "request": self.io_request(),
            "estimator": "" if estimator is None else estimator.currentText(),
            "user_azimuth": False if user_azimuth is None else user_azimuth.isChecked(),
            "azimuth_shift": 0.0 if azimuth is None else float(azimuth.text().strip() or "0"),
        }


class BackGeocodingDialog(QtWidgets.QDialog):
    """Graph dialog: ProductSet-Reader, Back-Geocoding, Write. Run does not coregister."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, runner=None) -> None:
        self._runner = runner
        super().__init__(parent)
        self.setWindowTitle("S-1 Back Geocoding")
        self.setObjectName("backGeocodingDialog")
        self.resize(840, 640)
        layout = QtWidgets.QVBoxLayout(self)
        help_line = QtWidgets.QLabel(
            "Back Geocoding coregisters split SLC bursts with orbits and a DEM "
            "(deramp, resample, reramp). " + _BACKGEO_BLOCK
        )
        help_line.setObjectName("operatorHelp")
        help_line.setWordWrap(True)
        layout.addWidget(help_line)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("backGeocodingTabs")
        self.product_set = ProductSetWidget(self)
        reader = QtWidgets.QWidget()
        reader_layout = QtWidgets.QVBoxLayout(reader)
        reader_layout.addWidget(self.product_set)
        self.tabs.addTab(reader, "ProductSet-Reader")
        self.tabs.addTab(self._processing_tab(), "Back-Geocoding")
        self.tabs.addTab(self._write_tab(), "Write")
        layout.addWidget(self.tabs, stretch=1)

        self.status = QtWidgets.QLabel("Error: [NodeId: ProductSet-Reader] Please add a source product")
        self.status.setObjectName("backGeocodingStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        for label, name, slot in (
            ("Save", "backGeoSave", self._save),
            ("Help", "backGeoHelp", self._help),
            ("Validate", "backGeoValidate", self._validate_clicked),
            ("Run", "backGeoRun", self._run),
        ):
            button = QtWidgets.QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.product_set.changed.connect(self._sync_status)

    def _processing_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.dem = _combo(_DEM_OPTIONS, "Copernicus 30m Global DEM (Auto Download)", "backGeoDem")
        self.dem.setToolTip(
            "Eleven DEM names were visible. SNAP 14 showed a scrollbar, so the list may be longer."
        )
        self.dem_resampling = _combo(_RESAMPLING, "BILINEAR_INTERPOLATION", "backGeoDemResampling")
        self.resampling = _combo(_RESAMPLING, "BILINEAR_INTERPOLATION", "backGeoResampling")
        self.mask_elevation = QtWidgets.QCheckBox("Mask out areas with no elevation")
        self.mask_elevation.setObjectName("backGeoMaskElevation")
        self.mask_elevation.setChecked(True)
        self.output_deramp = QtWidgets.QCheckBox("Output Deramp and Demod Phase")
        self.output_deramp.setObjectName("backGeoOutputDeramp")
        self.output_deramp.setChecked(False)
        self.disable_reramp = QtWidgets.QCheckBox("Disable Reramp")
        self.disable_reramp.setObjectName("backGeoDisableReramp")
        self.disable_reramp.setChecked(False)
        form.addRow("Digital Elevation Model", self.dem)
        form.addRow("DEM Resampling Method", self.dem_resampling)
        form.addRow("Resampling Type", self.resampling)
        form.addRow(self.mask_elevation)
        form.addRow(self.output_deramp)
        form.addRow(self.disable_reramp)
        return page

    def _write_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.target = QtWidgets.QLineEdit("target")
        self.target.setObjectName("backGeoTarget")
        self.format_box = _combo(("BEAM-DIMAP", "NumPy (.npy)"), "BEAM-DIMAP", "backGeoFormat")
        self.directory = QtWidgets.QLineEdit("/home/box")
        self.directory.setObjectName("backGeoDirectory")
        form.addRow("Name:", self.target)
        form.addRow("Save as", self.format_box)
        form.addRow("Directory:", self.directory)
        return page

    def _sync_status(self) -> None:
        if self.product_set.table.rowCount() == 0:
            self.status.setText("Error: [NodeId: ProductSet-Reader] Please add a source product")
        else:
            self.status.setText("")

    def validate(self) -> str:
        if not self.product_set.paths():
            return "Error: [NodeId: ProductSet-Reader] Please add a source product"
        if len(self.product_set.paths()) < 2:
            return _BACKGEO_BLOCK
        if self.resampling.currentText() != "BILINEAR_INTERPOLATION":
            return (
                f"Resampling Type {self.resampling.currentText()} is not applied. "
                "Only BILINEAR_INTERPOLATION is implemented."
            )
        return ""

    def parameters(self) -> dict:
        return {
            "paths": [str(path) for path in self.product_set.paths()],
            "dem": self.dem.currentText(),
            "resampling": self.resampling.currentText(),
            "mask_elevation": self.mask_elevation.isChecked(),
            "output_deramp": self.output_deramp.isChecked(),
            "disable_reramp": self.disable_reramp.isChecked(),
            "target": self.target.text().strip(),
            "format_name": self.format_box.currentText(),
            "directory": self.directory.text().strip(),
        }

    def block_reason(self) -> str:
        return _BACKGEO_BLOCK

    def _validate_clicked(self) -> None:
        self.status.setText(self.validate())

    def _help(self) -> None:
        self.status.setText(OPERATOR_HELP["stack"] + " " + _BACKGEO_BLOCK)

    def _save(self) -> None:
        self.status.setText("BEAM-DIMAP graph save is not written by this build.")

    def _run(self) -> None:
        message = self.validate()
        if message:
            self.status.setText(message)
            return
        if self._runner is None:
            self.status.setText(_BACKGEO_BLOCK)
            return
        try:
            self._runner(self.parameters())
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.accept()

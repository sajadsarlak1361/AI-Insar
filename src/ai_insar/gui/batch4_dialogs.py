"""SNAP 14 batch-4 dialogs: Multilook, Snaphu Export, Batch Unwrap, Import.

Step 9 (Subset and Save Product As) is not here. Those menu items were
disabled with an empty Product Explorer, and the dialogs were not opened.
"""

from __future__ import annotations

from ai_insar.gui.operator_dialog import OBSERVED_SAVE_FORMATS, OperatorDialog
from ai_insar.gui.qt import QtWidgets

_MULTILOOK_NOTE = "Note: Detection for complex data is done without resampling."
_SQUARE_BLOCK = (
    "GR Square Pixel is checked. Azimuth looks are blank because ground-range "
    "and azimuth spacing were not on the empty-product dialog, and this stack "
    "does not supply them. Check Independent Looks and enter both look counts."
)
_SOURCE_MISSING = "Source product not selected"


class MultilookDialog(OperatorDialog):
    """Multilooking. Azimuth looks stay blank while GR Square Pixel is checked."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, bands: list[str] | None = None) -> None:
        super().__init__(
            "Multilooking",
            "Multilook averages whole look cells. GR Square Pixel cannot invent the azimuth count.",
            suffix="_ML",
            object_name="multilookDialog",
            notes=_MULTILOOK_NOTE,
            parent=parent,
            source_label="Source:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
            seed_workspace_source=False,
        )
        form = QtWidgets.QFormLayout()
        self.bands = QtWidgets.QListWidget()
        self.bands.setObjectName("multilookBands")
        self.bands.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        for name in bands or []:
            self.bands.addItem(name)
        self.gr_square = QtWidgets.QCheckBox("GR Square Pixel")
        self.gr_square.setObjectName("multilookSquare")
        self.gr_square.setChecked(True)
        self.independent = QtWidgets.QCheckBox("Independent Looks")
        self.independent.setObjectName("multilookIndependent")
        self.independent.setChecked(False)
        self.range_looks = QtWidgets.QLineEdit("1")
        self.range_looks.setObjectName("multilookRangeLooks")
        self.azimuth_looks = QtWidgets.QLineEdit("")
        self.azimuth_looks.setObjectName("multilookAzimuthLooks")
        self.azimuth_looks.setEnabled(False)
        self.mean_square = QtWidgets.QLineEdit("")
        self.mean_square.setObjectName("multilookMeanSquare")
        self.mean_square.setEnabled(False)
        self.output_intensity = QtWidgets.QCheckBox("Output Intensity")
        self.output_intensity.setObjectName("multilookOutputIntensity")
        self.output_intensity.setChecked(False)
        self.gr_square.toggled.connect(self._on_square)
        self.independent.toggled.connect(self._on_independent)
        form.addRow("Source Bands:", self.bands)
        form.addRow(self.gr_square)
        form.addRow(self.independent)
        form.addRow("Number of Range Looks:", self.range_looks)
        form.addRow("Number of Azimuth Looks:", self.azimuth_looks)
        form.addRow("Mean GR Square Pixel:", self.mean_square)
        form.addRow(self.output_intensity)
        note = QtWidgets.QLabel(_MULTILOOK_NOTE)
        note.setObjectName("multilookNote")
        note.setWordWrap(True)
        form.addRow(note)
        self.add_processing(form)

    def _on_square(self, checked: bool) -> None:
        if not checked:
            return
        self.independent.blockSignals(True)
        self.independent.setChecked(False)
        self.independent.blockSignals(False)
        self.azimuth_looks.clear()
        self.azimuth_looks.setEnabled(False)

    def _on_independent(self, checked: bool) -> None:
        if checked:
            self.gr_square.blockSignals(True)
            self.gr_square.setChecked(False)
            self.gr_square.blockSignals(False)
            self.azimuth_looks.setEnabled(True)
            return
        self.gr_square.blockSignals(True)
        self.gr_square.setChecked(True)
        self.gr_square.blockSignals(False)
        self.azimuth_looks.clear()
        self.azimuth_looks.setEnabled(False)

    def selected_bands(self) -> list[str]:
        return [item.text() for item in self.bands.selectedItems()]

    def validate(self) -> str:
        message = self.validate_io()
        if message:
            return message
        if not self.selected_bands():
            return "Source Bands is empty."
        if self.gr_square.isChecked():
            return _SQUARE_BLOCK
        try:
            range_looks = int(self.range_looks.text().strip())
            azimuth_looks = int(self.azimuth_looks.text().strip())
        except ValueError:
            return "Independent Looks needs integer range and azimuth look counts."
        if range_looks < 1 or azimuth_looks < 1:
            return "Look counts must be at least 1."
        return ""

    def parameters(self) -> dict:
        return {
            "request": self.io_request(),
            "bands": self.selected_bands(),
            "gr_square": self.gr_square.isChecked(),
            "independent": self.independent.isChecked(),
            "range_looks": int(self.range_looks.text().strip() or "0"),
            "azimuth_looks": int(self.azimuth_looks.text().strip() or "0"),
            "output_intensity": self.output_intensity.isChecked(),
        }


class SnaphuExportDialog(QtWidgets.QDialog):
    """Read + SnaphuExport graph. The source combo starts blank."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, has_interferogram: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Snaphu Export")
        self.setObjectName("snaphuExportDialog")
        self.resize(760, 560)
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._read_tab(has_interferogram), "Read")
        self.tabs.addTab(self._export_tab(), "SnaphuExport")
        layout.addWidget(self.tabs, stretch=1)
        self.status = QtWidgets.QLabel(_SOURCE_MISSING)
        self.status.setObjectName("snaphuExportStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        for label, name, slot in (
            ("Save", "snaphuExportSave", self._not_dimap),
            ("Help", "snaphuExportHelp", self._help),
            ("Validate", "snaphuExportValidate", self._show_validate),
            ("Run", "snaphuExportRun", self._run),
        ):
            button = QtWidgets.QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._accepted = False

    def _read_tab(self, has_interferogram: bool) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        row = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.setObjectName("snaphuExportSource")
        self.source.addItem("")
        if has_interferogram:
            self.source.addItem("Workspace interferogram")
        self.source.setCurrentIndex(0)
        browse = QtWidgets.QPushButton("…")
        browse.setObjectName("snaphuExportBrowse")
        browse.clicked.connect(self._browse_source)
        row.addWidget(self.source, stretch=1)
        row.addWidget(browse)
        self.data_format = QtWidgets.QComboBox()
        self.data_format.setObjectName("snaphuExportFormat")
        advanced = QtWidgets.QPushButton("Advanced options")
        advanced.setObjectName("snaphuExportAdvanced")
        advanced.setEnabled(False)
        form.addRow("Name:", row)
        form.addRow("Data Format:", self.data_format)
        form.addRow(advanced)
        return page

    def _export_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        folder_row = QtWidgets.QHBoxLayout()
        self.target_folder = QtWidgets.QLineEdit("")
        self.target_folder.setObjectName("snaphuExportFolder")
        folder_button = QtWidgets.QPushButton("…")
        folder_button.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.target_folder, stretch=1)
        folder_row.addWidget(folder_button)
        self.stat_mode = _combo(("TOPO", "DEFO", "SMOOTH", "NOSTATCOSTS"), "DEFO", "snaphuStatMode")
        self.init_method = _combo(("MST", "MCF"), "MCF", "snaphuInitMethod")
        self.tile_rows = QtWidgets.QLineEdit("10")
        self.tile_rows.setObjectName("snaphuTileRows")
        self.tile_columns = QtWidgets.QLineEdit("10")
        self.tile_columns.setObjectName("snaphuTileColumns")
        self.processors = QtWidgets.QLineEdit("4")
        self.processors.setObjectName("snaphuProcessors")
        self.row_overlap = QtWidgets.QLineEdit("0")
        self.row_overlap.setObjectName("snaphuRowOverlap")
        self.column_overlap = QtWidgets.QLineEdit("0")
        self.column_overlap.setObjectName("snaphuColumnOverlap")
        self.tile_cost = QtWidgets.QLineEdit("500")
        self.tile_cost.setObjectName("snaphuTileCost")
        form.addRow("Target folder:", folder_row)
        form.addRow("Statistical-cost mode:", self.stat_mode)
        form.addRow("Initial method:", self.init_method)
        form.addRow("Number of Tile Rows:", self.tile_rows)
        form.addRow("Number of Tile Columns:", self.tile_columns)
        form.addRow("Number of Processors:", self.processors)
        form.addRow("Row Overlap:", self.row_overlap)
        form.addRow("Column Overlap:", self.column_overlap)
        form.addRow("Tile Cost Threshold:", self.tile_cost)
        return page

    def _browse_source(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(self, "Source product")
        if path:
            self.source.setCurrentText(path)

    def _browse_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Target folder")
        if folder:
            self.target_folder.setText(folder)

    def validate(self) -> str:
        if not self.source.currentText().strip():
            return _SOURCE_MISSING
        if self.source.currentText().strip() != "Workspace interferogram":
            return "Only the workspace interferogram can be exported. A foreign file was not read."
        if not self.target_folder.text().strip():
            return "Choose a target folder."
        for name, field in (
            ("Number of Tile Rows", self.tile_rows),
            ("Number of Tile Columns", self.tile_columns),
            ("Number of Processors", self.processors),
            ("Row Overlap", self.row_overlap),
            ("Column Overlap", self.column_overlap),
            ("Tile Cost Threshold", self.tile_cost),
        ):
            try:
                int(field.text().strip())
            except ValueError:
                return f"{name} must be an integer."
        return ""

    def parameters(self) -> dict:
        return {
            "source": self.source.currentText().strip(),
            "folder": self.target_folder.text().strip(),
            "stat_mode": self.stat_mode.currentText(),
            "init_method": self.init_method.currentText(),
            "tile_rows": int(self.tile_rows.text().strip()),
            "tile_columns": int(self.tile_columns.text().strip()),
            "processors": int(self.processors.text().strip()),
            "row_overlap": int(self.row_overlap.text().strip()),
            "column_overlap": int(self.column_overlap.text().strip()),
            "tile_cost_threshold": int(self.tile_cost.text().strip()),
        }

    def _show_validate(self) -> None:
        self.status.setText(self.validate() or "OK")

    def _help(self) -> None:
        self.status.setText(
            "Export writes wrapped phase and coherence as float32 ENVI plus snaphu.conf. "
            "It does not unwrap."
        )

    def _not_dimap(self) -> None:
        self.status.setText("This graph has no BEAM-DIMAP Write tab. Run writes the SNAPHU files.")

    def _run(self) -> None:
        message = self.validate()
        if message:
            self.status.setText(message)
            return
        self._accepted = True
        self.accept()


class BatchSnaphuUnwrapDialog(OperatorDialog):
    """Installed unwrapping entry: two folder fields, no in-tree unwrap."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(
            "Batch Snaphu Unwrap",
            "Runs the external snaphu binary on an export folder. Fast unwrap is not a substitute.",
            suffix="",
            object_name="batchSnaphuDialog",
            notes="Snaphu Install Location must contain an executable named snaphu. Nothing is unwrapped in-process.",
            parent=parent,
            source_label="Source:",
            target_text="",
            directory_text="/home/box",
            formats=OBSERVED_SAVE_FORMATS,
            format_default="BEAM-DIMAP",
            seed_workspace_source=False,
        )
        form = QtWidgets.QFormLayout()
        self.export_folder = self._folder("snaphuUnwrapExport")
        self.install_location = self._folder("snaphuUnwrapInstall")
        form.addRow("Snaphu Export folder:", self.export_folder)
        form.addRow("Snaphu Install Location:", self.install_location)
        self.add_processing(form)

    def _folder(self, name: str) -> QtWidgets.QLineEdit:
        field = QtWidgets.QLineEdit("")
        field.setObjectName(name)
        return field

    def validate(self) -> str:
        if not self.export_folder.text().strip() or not self.install_location.text().strip():
            return "Snaphu Export folder and Snaphu Install Location are both required."
        return ""

    def parameters(self) -> dict:
        return {
            "export_folder": self.export_folder.text().strip(),
            "install_location": self.install_location.text().strip(),
        }


class SnaphuImportDialog(QtWidgets.QDialog):
    """Four tabs. Both Read inputs start empty. The checkbox is unchecked."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Snaphu Import")
        self.setObjectName("snaphuImportDialog")
        self.resize(760, 520)
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("snaphuImportTabs")
        self.phase_source = QtWidgets.QLineEdit("")
        self.phase_source.setObjectName("snaphuImportPhase")
        self.unwrapped_source = QtWidgets.QLineEdit("")
        self.unwrapped_source.setObjectName("snaphuImportUnwrapped")
        self.tabs.addTab(self._read_page(self.phase_source), "1-Read-Phase")
        self.tabs.addTab(self._read_page(self.unwrapped_source), "2-Read-Unwrapped-Phase")
        import_page = QtWidgets.QWidget()
        import_layout = QtWidgets.QVBoxLayout(import_page)
        self.skip_wrapped = QtWidgets.QCheckBox("Do NOT save Wrapped interferogram in the target product")
        self.skip_wrapped.setObjectName("snaphuImportSkipWrapped")
        self.skip_wrapped.setChecked(False)
        import_layout.addWidget(self.skip_wrapped)
        import_layout.addStretch(1)
        self.tabs.addTab(import_page, "3-SnaphuImport")
        self.tabs.addTab(self._write_page(), "4-Write")
        layout.addWidget(self.tabs, stretch=1)
        self.status = QtWidgets.QLabel(_SOURCE_MISSING)
        self.status.setObjectName("snaphuImportStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        for label, name, slot in (
            ("Save", "snaphuImportSave", self._not_dimap),
            ("Help", "snaphuImportHelp", self._help),
            ("Validate", "snaphuImportValidate", self._show_validate),
            ("Run", "snaphuImportRun", self._run),
        ):
            button = QtWidgets.QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

    def _read_page(self, field: QtWidgets.QLineEdit) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(field, stretch=1)
        browse = QtWidgets.QPushButton("…")
        browse.clicked.connect(lambda _checked=False, target=field: self._browse(target))
        row.addWidget(browse)
        data_format = QtWidgets.QComboBox()
        advanced = QtWidgets.QPushButton("Advanced options")
        advanced.setEnabled(False)
        form.addRow("Name:", row)
        form.addRow("Data Format:", data_format)
        form.addRow(advanced)
        return page

    def _write_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.target = QtWidgets.QLineEdit("target")
        self.target.setObjectName("snaphuImportTarget")
        self.format_box = _combo(("BEAM-DIMAP", "NumPy (.npy)"), "BEAM-DIMAP", "snaphuImportFormat")
        directory_row = QtWidgets.QHBoxLayout()
        self.directory = QtWidgets.QLineEdit("/home/box")
        self.directory.setObjectName("snaphuImportDirectory")
        directory_row.addWidget(self.directory, stretch=1)
        directory_row.addWidget(QtWidgets.QPushButton("…"))
        form.addRow("Name:", self.target)
        form.addRow("Save as:", self.format_box)
        form.addRow("Directory:", directory_row)
        return page

    def _browse(self, field: QtWidgets.QLineEdit) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Snaphu folder")
        if folder:
            field.setText(folder)

    def validate(self) -> str:
        if not self.phase_source.text().strip() or not self.unwrapped_source.text().strip():
            return _SOURCE_MISSING
        return ""

    def parameters(self) -> dict:
        return {
            "phase_folder": self.phase_source.text().strip(),
            "unwrapped_folder": self.unwrapped_source.text().strip(),
            "skip_wrapped": self.skip_wrapped.isChecked(),
            "target": self.target.text().strip(),
            "format": self.format_box.currentText(),
            "directory": self.directory.text().strip(),
        }

    def _show_validate(self) -> None:
        self.status.setText(self.validate() or "OK")

    def _help(self) -> None:
        self.status.setText("Import reads unwrap.img with the ENVI header and snaphu_meta.json. It does not unwrap.")

    def _not_dimap(self) -> None:
        self.status.setText("BEAM-DIMAP is not written. Run loads the unwrapped phase into the workspace.")

    def _run(self) -> None:
        message = self.validate()
        if message:
            self.status.setText(message)
            return
        self.accept()


def _combo(items: tuple[str, ...], current: str, name: str) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox()
    box.setObjectName(name)
    box.addItems(list(items))
    box.setCurrentText(current)
    return box

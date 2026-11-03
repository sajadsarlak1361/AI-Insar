"""Dialog for ASF Sentinel-1 SLC search, download, and local crop import."""

from __future__ import annotations

from pathlib import Path

from ai_insar.gui.qt import Qt, QtWidgets
from ai_insar.pipeline.synthetic import SLCStack
from ai_insar.sentinel import (
    SentinelGranule,
    credentials_summary,
    download_sentinel1_slc,
    earthdata_credentials,
    import_slc_paths,
    search_sentinel1_slc,
)

_PRESETS = {
    "Custom": None,
    "San Francisco Bay": (-122.6, 37.6, -122.2, 37.9),
}


class FetchSentinelDialog(QtWidgets.QDialog):
    """Search ASF and import a complex64 crop into the workbench."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, search_fn=search_sentinel1_slc) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fetch Sentinel-1 SLC")
        self.resize(860, 640)
        self._search_fn = search_fn
        self._granules: list[SentinelGranule] = []
        self.stack: SLCStack | None = None
        self.downloaded_zips: list[Path] = []
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Search the public ASF catalog (no login). Import a crop from a SAFE "
            "folder, manifest.safe, a product .zip, a measurement TIFF, or a complex "
            ".npy you already have. A downloaded zip is read directly and does not "
            "need to be extracted first. Downloading the zip needs an Earthdata login "
            "and is often several GB per product. The result is an in-memory complex64 "
            "stack, not a geocoded scene."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(_PRESETS))
        form.addRow("Place", self.preset)
        self.west = self._degree(-180.0, 180.0, -122.6)
        self.south = self._degree(-90.0, 90.0, 37.6)
        self.east = self._degree(-180.0, 180.0, -122.2)
        self.north = self._degree(-90.0, 90.0, 37.9)
        self.preset.currentTextChanged.connect(self._apply_preset)
        self.preset.setCurrentText("San Francisco Bay")
        box = QtWidgets.QHBoxLayout()
        for widget in (self.west, self.south, self.east, self.north):
            box.addWidget(widget)
        form.addRow("West, south, east, north", box)
        self.start = QtWidgets.QLineEdit("2024-01-01")
        self.end = QtWidgets.QLineEdit("2024-01-31")
        form.addRow("Start (YYYY-MM-DD)", self.start)
        form.addRow("End (YYYY-MM-DD)", self.end)
        self.max_results = QtWidgets.QSpinBox()
        self.max_results.setRange(1, 50)
        self.max_results.setValue(10)
        form.addRow("Max results", self.max_results)
        layout.addLayout(form)

        search_row = QtWidgets.QHBoxLayout()
        search_button = QtWidgets.QPushButton("Search ASF")
        search_button.clicked.connect(lambda _checked=False: self.run_search())
        search_row.addWidget(search_button)
        search_row.addStretch(1)
        layout.addLayout(search_row)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Scene", "Start", "Platform", "Direction", "MB", "Polarization"]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        crop = QtWidgets.QHBoxLayout()
        self.row = self._pixel(0)
        self.col = self._pixel(0)
        self.height = self._pixel(256)
        self.width = self._pixel(256)
        self.swath = QtWidgets.QComboBox()
        self.swath.setObjectName("slcSwath")
        self.swath.addItems(["IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"])
        self.swath.setEditable(True)
        self.pol = QtWidgets.QComboBox()
        self.pol.setObjectName("slcPol")
        self.pol.addItems(["VV", "VH", "HH", "HV"])
        for label, widget in (
            ("Swath", self.swath),
            ("Pol", self.pol),
            ("Row", self.row),
            ("Col", self.col),
            ("Height", self.height),
            ("Width", self.width),
        ):
            crop.addWidget(QtWidgets.QLabel(label))
            crop.addWidget(widget)
        crop.addStretch(1)
        layout.addLayout(crop)
        crop_note = QtWidgets.QLabel(
            "The same crop is read from every date. A window that starts at 0,0 and "
            "is larger than a small raster uses the whole raster."
        )
        crop_note.setWordWrap(True)
        layout.addWidget(crop_note)

        self.status = QtWidgets.QLabel(credentials_summary())
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        download = QtWidgets.QPushButton("Download selected…")
        download.clicked.connect(lambda _checked=False: self.download_selected())
        import_files = QtWidgets.QPushButton("Import files…")
        import_files.clicked.connect(lambda _checked=False: self.import_files())
        import_folder = QtWidgets.QPushButton("Import SAFE folder…")
        import_folder.clicked.connect(lambda _checked=False: self.import_folder())
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        buttons.addWidget(download)
        buttons.addWidget(import_files)
        buttons.addWidget(import_folder)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _degree(self, low: float, high: float, value: float) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(4)
        box.setSingleStep(0.1)
        box.setValue(value)
        return box

    def _pixel(self, value: int) -> QtWidgets.QSpinBox:
        box = QtWidgets.QSpinBox()
        box.setRange(0, 200_000)
        box.setValue(value)
        return box

    def _apply_preset(self, name: str) -> None:
        bbox = _PRESETS.get(name)
        if bbox is None:
            return
        west, south, east, north = bbox
        self.west.setValue(west)
        self.south.setValue(south)
        self.east.setValue(east)
        self.north.setValue(north)

    def run_search(self) -> None:
        try:
            granules = self._search_fn(
                west=float(self.west.value()),
                south=float(self.south.value()),
                east=float(self.east.value()),
                north=float(self.north.value()),
                start=self.start.text().strip(),
                end=self.end.text().strip(),
                max_results=int(self.max_results.value()),
            )
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self._granules = list(granules)
        self.table.setRowCount(len(self._granules))
        for index, granule in enumerate(self._granules):
            size = "" if granule.size_mb is None else f"{granule.size_mb:.0f}"
            values = [
                granule.granule_name,
                granule.start_time,
                granule.platform,
                granule.flight_direction,
                size,
                granule.polarization,
            ]
            for column, value in enumerate(values):
                self.table.setItem(index, column, QtWidgets.QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.status.setText(
            f"{len(self._granules)} SLC granule(s). "
            + credentials_summary()
        )

    def selected_granules(self) -> list[SentinelGranule]:
        rows = sorted({item.row() for item in self.table.selectedIndexes()})
        return [self._granules[row] for row in rows if 0 <= row < len(self._granules)]

    def download_selected(self) -> None:
        granules = self.selected_granules()
        if not granules:
            self.status.setText("Select one or more rows to download.")
            return
        creds = earthdata_credentials()
        if creds is None:
            self.status.setText(credentials_summary())
            return
        username, password, _source = creds
        sizes = [item.size_mb or 0.0 for item in granules]
        total = sum(sizes)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Download Sentinel-1 SLC",
            f"Download {len(granules)} SLC zip file(s)"
            + (f" (about {total:.0f} MB in the catalog)" if total else "")
            + "? Full SLC products are often several GB each. "
            "The password is sent only to Earthdata and is not saved here.",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Download folder")
        if not directory:
            return
        try:
            saved = download_sentinel1_slc(
                granules,
                Path(directory),
                username=username,
                password=password,
            )
        except Exception as exc:
            self.status.setText(str(exc))
            return
        names = ", ".join(path.name for path in saved[:4]) or "the download folder"
        self.downloaded_zips = list(saved)
        self.status.setText(
            f"Downloaded {len(saved)} zip file(s) to {directory}: {names}. "
            "The SAFE zip is opened directly. It is not extracted."
        )
        if not saved:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Open downloaded SAFE zip",
            f"Open {len(saved)} downloaded zip file(s) now and read the "
            f"{self.swath.currentText().strip() or 'IW1'} "
            f"{self.pol.currentText().strip() or 'VV'} crop into the SLC stack? "
            "The zip is not extracted first.",
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.load_paths(saved)

    def import_files(self) -> None:
        paths, _selected = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Import SLC rasters",
            "",
            "Sentinel-1 SAFE zip (*.zip);;SLC rasters (*.tif *.tiff *.npy manifest.safe)",
        )
        if not paths:
            return
        self.load_paths([Path(path) for path in paths])

    def import_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Import SAFE folder")
        if not directory:
            return
        self.load_paths([Path(directory)])

    def load_paths(self, paths: list[Path]) -> None:
        try:
            stack = import_slc_paths(
                paths,
                row=int(self.row.value()),
                col=int(self.col.value()),
                height=int(self.height.value()),
                width=int(self.width.value()),
                swath=self.swath.currentText().strip() or "IW1",
                pol=self.pol.currentText().strip() or "VV",
            )
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.stack = stack
        zips = [path for path in paths if path.suffix.lower() == ".zip"]
        if zips:
            self.downloaded_zips = zips
        self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

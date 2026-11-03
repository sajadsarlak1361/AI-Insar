"""Open many Sentinel-1 SAFE zips into one SLC stack."""

from __future__ import annotations

from pathlib import Path

from ai_insar.gui.qt import QtWidgets
from ai_insar.safe import DEFAULT_POLARISATION, DEFAULT_SWATH
from ai_insar.sentinel import SafeStackLoad, expand_safe_inputs, load_safe_slc_stack


class SafeStackDialog(QtWidgets.QDialog):
    """Pick many SAFE zips, or a folder of them, and crop one band into a stack."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add many SAFE zips to stack")
        self.setObjectName("safeStackDialog")
        self.resize(720, 480)
        self.paths: list[Path] = []
        self.load_result: SafeStackLoad | None = None
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Choose several Sentinel-1 SAFE .zip files, or a folder that contains them. "
            ".SAFE folders are optional. The zip is read in place and is not extracted. "
            "Each product contributes one date: the same swath and polarisation "
            "(default IW1 VV) and the same pixel window. Dates come from the product "
            "start time and are sorted. A product that does not contain that band is skipped."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        note = QtWidgets.QLabel(
            "The products are not coregistered. There is no DEM, no orbit download, "
            "and no TOPS deburst. The shared window is a teaching crop. "
            "Interferograms on the stack are later × conjugate(earlier), "
            "the same as on a synthetic or .npy stack."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QtWidgets.QHBoxLayout()
        choose_zips = QtWidgets.QPushButton("Choose SAFE zips…")
        choose_zips.setObjectName("chooseSafeZipsButton")
        choose_zips.clicked.connect(lambda _checked=False: self.choose_zips())
        choose_folder = QtWidgets.QPushButton("Choose folder of zips…")
        choose_folder.setObjectName("chooseSafeFolderButton")
        choose_folder.clicked.connect(lambda _checked=False: self.choose_folder())
        clear = QtWidgets.QPushButton("Clear list")
        clear.clicked.connect(lambda _checked=False: self.clear_paths())
        buttons.addWidget(choose_zips)
        buttons.addWidget(choose_folder)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.files = QtWidgets.QListWidget()
        self.files.setObjectName("safeStackFileList")
        layout.addWidget(self.files, stretch=1)

        crop = QtWidgets.QHBoxLayout()
        self.swath = QtWidgets.QComboBox()
        self.swath.setObjectName("safeStackSwath")
        self.swath.addItems(["IW1", "IW2", "IW3"])
        self.swath.setEditable(True)
        self.swath.setCurrentText(DEFAULT_SWATH)
        self.pol = QtWidgets.QComboBox()
        self.pol.setObjectName("safeStackPol")
        self.pol.addItems(["VV", "VH", "HH", "HV"])
        self.pol.setEditable(True)
        self.pol.setCurrentText(DEFAULT_POLARISATION)
        self.row = self._pixel(0)
        self.col = self._pixel(0)
        self.height = self._pixel(256)
        self.width = self._pixel(256)
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

        self.status = QtWidgets.QLabel("No SAFE products selected.")
        self.status.setObjectName("safeStackStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        actions = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        actions.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Build SLC stack")
        actions.accepted.connect(self.build_stack)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)

    def _pixel(self, value: int) -> QtWidgets.QSpinBox:
        box = QtWidgets.QSpinBox()
        box.setRange(0, 200_000)
        box.setValue(value)
        return box

    def choose_zips(self) -> None:
        paths, _selected = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add many SAFE zips to stack",
            "",
            "Sentinel-1 SAFE zip (*.zip);;Sentinel-1 manifest (manifest.safe)",
        )
        if not paths:
            return
        self._add(paths)

    def choose_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of SAFE zips")
        if not directory:
            return
        self._add([directory])

    def clear_paths(self) -> None:
        self.paths = []
        self.files.clear()
        self.status.setText("No SAFE products selected.")

    def add_paths(self, paths: list[Path]) -> None:
        """Add zips, SAFE folders, or a folder that contains them."""
        expanded = expand_safe_inputs([Path(path) for path in paths])
        known = {path.resolve() for path in self.paths}
        for path in expanded:
            resolved = path.resolve()
            if resolved in known:
                continue
            known.add(resolved)
            self.paths.append(path)
            self.files.addItem(path.name)
        count = len(self.paths)
        noun = "product" if count == 1 else "products"
        self.status.setText(f"{count} SAFE {noun} selected. Zips are not extracted.")

    def _add(self, paths: list[str | Path]) -> None:
        try:
            self.add_paths([Path(path) for path in paths])
        except ValueError as exc:
            self.status.setText(str(exc))

    def build_stack(self) -> None:
        if not self.paths:
            self.status.setText("Choose one or more Sentinel-1 SAFE zips first.")
            return
        swath = self.swath.currentText().strip() or DEFAULT_SWATH
        pol = self.pol.currentText().strip() or DEFAULT_POLARISATION

        def progress(index: int, total: int, message: str) -> None:
            text = f"SAFE stack {index}/{total}: {message}"
            self.status.setText(text)
            parent = self.parent()
            if isinstance(parent, QtWidgets.QMainWindow):
                parent.statusBar().showMessage(text)
            QtWidgets.QApplication.processEvents()

        try:
            self.load_result = load_safe_slc_stack(
                self.paths,
                row=int(self.row.value()),
                col=int(self.col.value()),
                height=int(self.height.value()),
                width=int(self.width.value()),
                swath=swath,
                pol=pol,
                on_progress=progress,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.load_result = None
            self.status.setText(str(exc))
            return
        stack = self.load_result.stack
        skipped = len(self.load_result.skipped)
        extra = f" Skipped {skipped}." if skipped else ""
        self.status.setText(
            f"Loaded {stack.n_dates} dates, {stack.height}×{stack.width}.{extra}"
        )
        self.accept()

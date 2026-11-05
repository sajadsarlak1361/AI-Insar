"""Main window: numbered stages, a raster viewer, and a log."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ai_insar import __version__
from ai_insar.conventions import C_BAND_WAVELENGTH_M, CONVENTIONS_TEXT, LIMITATIONS_TEXT, los_from_phase
from ai_insar.gmtsar_flow import FLOWCHART_TEXT
from ai_insar.gui.qt import Qt, QtGui, QtWidgets
from ai_insar.gui.render import amplitude_rgb, colorize, overlay_mask, symmetric_limits
from ai_insar.gui.batch4_dialogs import (
    BatchSnaphuUnwrapDialog,
    MultilookDialog,
    SnaphuExportDialog,
    SnaphuImportDialog,
)
from ai_insar.gui.chain_dialogs import (
    BackGeocodingDialog,
    EsdDialog,
    TopsDeburstDialog,
    TopsSplitDialog,
    _BACKGEO_BLOCK,
    _DEBURST_BLOCK,
    _ESD_BLOCK,
)
from ai_insar.gui.late_dialogs import (
    PhaseToDisplacementDialog,
    SaveProductDialog,
    SubsetDialog,
    TerrainCorrectionDialog,
)
from ai_insar.gui.fetch_dialog import FetchSentinelDialog
from ai_insar.gui.operator_dialog import (
    ApplyOrbitDialog,
    CoherenceDialog,
    GoldsteinDialog,
    InterferogramDialog,
    TopoPhaseDialog,
)
from ai_insar.gui.product_explorer import ProductExplorer, SlcCropDialog
from ai_insar.gui.safe_stack_dialog import SafeStackDialog
from ai_insar.gui.settings import SettingsDialog
from ai_insar.gui.viewer import ColorBar, ImageView
from ai_insar.export_maps import export_displacement_map, export_velocity_map
from ai_insar.io_stack import load_slc, save_products
from ai_insar.pipeline.align import align_slc_stack
from ai_insar.pipeline.coherence import COHERENCE_LABEL, COHERENCE_PRESETS, COHERENCE_SUMMARY
from ai_insar.pipeline.topo import apply_topographic_phase, topographic_phase_rad
from ai_insar.pipeline.interferogram import form_interferograms
from ai_insar.pipeline.engine import (
    Workspace,
    compute_coherence,
    compute_detection,
    compute_filter,
    compute_interferograms,
    compute_timeseries,
    compute_unwrap,
    run_all,
)
from ai_insar.pipeline.filtering import GOLDSTEIN_LABEL, GOLDSTEIN_PRESETS, GOLDSTEIN_SUMMARY, goldstein_filter
from ai_insar.pipeline.synthetic import SLCStack, generate_demo_stack
from ai_insar.pipeline.tops import split_swath
from ai_insar.safe import format_product_status, preview_amplitude, resolve_safe, select_measurement
from ai_insar.snap_products import PRODUCT_SUFFIX, write_npy_product, write_orbit_stub

# Zip is the primary File → Open choice. A manifest is an optional extra.
SAFE_OPEN_FILTER = "Sentinel-1 SAFE zip (*.zip);;Sentinel-1 manifest (manifest.safe)"
from ai_insar.sentinel import import_slc_paths
from ai_insar.pipeline.timeseries import TIMESERIES_LABELS, TIMESERIES_METHODS, TIMESERIES_SUMMARIES
from ai_insar.pipeline.unwrap import UNWRAP_LABELS, UNWRAP_METHODS, UNWRAP_SUMMARIES

_LAYERS = (
    "SLC amplitude",
    "Interferogram phase",
    "Filtered phase",
    "Coherence",
    "Unwrapped phase",
    "LOS displacement",
    "LOS velocity",
    "Detection mask",
    "Velocity + mask",
)

_PAIR_LAYERS = {
    "Interferogram phase",
    "Filtered phase",
    "Coherence",
    "Unwrapped phase",
}

_STAGE_LABELS = (
    "PREPROCESS · Generate / load sample SLC",
    "MAKE INTERFEROGRAMS · Interferogram",
    "MAKE INTERFEROGRAMS · Goldstein",
    "MAKE INTERFEROGRAMS · Coherence",
    "UNWRAP PHASE · Unwrap",
    "SBAS · SBAS / time series",
    "Displacement detection mask",
)

_STAGE_HELP = (
    "PREPROCESS. Synthetic C-band SLC stack (dates × height × width, complex64). File → Load SLC stack or Open SAFE stack replaces it. Pipeline → ALIGN SLC and MAKE TOPO_RA are optional and are not part of Run all stages.",
    "MAKE INTERFEROGRAMS. Form each pair as later × conjugate(earlier) on a short-baseline network.",
    "MAKE INTERFEROGRAMS. Goldstein spectral filter. The Goldstein menu sets alpha and runs this stage. Unwrapping uses this phase.",
    "MAKE INTERFEROGRAMS. Windowed coherence from the SLC pair. The Coherence menu sets the window. Filtering does not change this estimate.",
    "UNWRAP PHASE. Fast unwrap by default (Gaussian smooth, then separable numpy.unwrap). Quality-guided stays in its own menu. A plane is then removed on the image border.",
    "SBAS time series after the pair chain. Each other method has its own menu. Negative velocity is away from the radar.",
    "Coherence-gated velocity anomaly. Not a trained model. GEOCODE is Pipeline → GEOCODE: products stay on the radar grid.",
)

_LAYER_UNITS = {
    "SLC amplitude": "amplitude",
    "Interferogram phase": "rad",
    "Filtered phase": "rad",
    "Coherence": "coherence",
    "Unwrapped phase": "rad",
    "LOS displacement": "m",
    "LOS velocity": "m/yr",
    "Detection mask": "mask",
    "Velocity + mask": "m/yr",
}

_STYLESHEET = """
QMainWindow, QWidget { background: #1b212c; color: #e7edf5; }
QLabel#titleLabel { font-size: 18px; font-weight: 650; color: #f4f7fb; }
QLabel#subtle { color: #9aa7bd; }
QPushButton {
    background: #2a3344;
    border: 1px solid #3c4a63;
    border-radius: 8px;
    padding: 8px 12px;
    text-align: left;
}
QPushButton:hover { background: #34445c; }
QPushButton:disabled { color: #8894a8; }
QPushButton[stageState="done"] { border-left: 4px solid #3ddc97; }
QPushButton[stageState="error"] { border-left: 4px solid #ff6b6b; }
QPushButton#runAll {
    background: #2d6cdf;
    border: 1px solid #2d6cdf;
    font-weight: 600;
    text-align: center;
}
QPushButton#runAll:hover { background: #3d7cf0; }
QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget {
    background: #12171f;
    border: 1px solid #3c4a63;
    border-radius: 4px;
    padding: 4px;
    color: #e7edf5;
}
QHeaderView::section { background: #1b212c; color: #e7edf5; padding: 4px; }
QTabBar::tab { background: #2a3344; color: #e7edf5; padding: 6px 10px; }
QTabBar::tab:selected { background: #2d6cdf; }
QPlainTextEdit { font-family: monospace; color: #d5deea; }
QMenuBar { background: #151a23; }
QMenuBar::item:selected { background: #2d6cdf; }
QMenu { background: #1b212c; border: 1px solid #3c4a63; }
QMenu::item:selected { background: #2d6cdf; }
QStatusBar { background: #151a23; color: #c5d0e0; }
QSplitter::handle { background: #2c3648; }
QDockWidget { color: #e7edf5; }
QDockWidget::title { background: #151a23; padding: 6px; }
QTreeWidget { background: #12171f; color: #e7edf5; border: 1px solid #3c4a63; }
QTreeWidget::item:selected { background: #2d6cdf; }
"""


def _crop(array, row0: int, row1: int, col0: int, col1: int):
    if array is None:
        return None
    if array.shape[-2] < row1 or array.shape[-1] < col1:
        return array
    return np.array(array[..., row0:row1, col0:col1], copy=True)


def _covers(grid, start: int, stop: int) -> bool:
    return grid.row <= start and grid.row + grid.shape[0] >= stop


def _slice(grid, start: int, stop: int):
    local = start - grid.row
    return grid.data[local : local + (stop - start)]


class MainWindow(QtWidgets.QMainWindow):
    """Desktop shell around :class:`~ai_insar.pipeline.engine.Workspace`."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Insar")
        self.resize(1180, 780)
        self.workspace = Workspace()
        self._busy = False
        self._preferred_layer: str | None = None
        self.safe_product = None
        self._product_status: str | None = None
        self.stage_buttons: list[QtWidgets.QPushButton] = []
        self._build()
        self.setStyleSheet(_STYLESHEET)
        self._log(
            f"AI Insar {__version__}. Primary input is an SLC stack "
            "(complex64, shape dates × height × width). "
            "Interferograms are later × conjugate(earlier)."
        )
        self._sync_stage_states()
        self._refresh_view()

    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        body = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self._stage_panel())
        body.addWidget(self._viewer_panel())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        log_wrap = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_wrap)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_title = QtWidgets.QLabel("Log")
        log_title.setObjectName("subtle")
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(body)
        splitter.addWidget(log_wrap)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)
        self._build_menus()
        self.product_explorer = ProductExplorer(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.product_explorer)
        self.product_explorer.bandActivated.connect(
            lambda swath, pol: self._guard(lambda: self.preview_safe_band(swath, pol), None)
        )
        self.product_explorer.quicklookActivated.connect(
            lambda: self._guard(self.preview_safe_quicklook, None)
        )
        self.product_explorer.addToStackRequested.connect(self._click(self.add_safe_to_stack, 0))
        self.statusBar().showMessage("Ready")

    def _stage_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(380)
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("AI Insar")
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel("SLC stack → displacement mask")
        subtitle.setObjectName("subtle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        actions = (
            self.generate_sample,
            self.run_interferograms,
            self.run_filter,
            self.run_coherence,
            self.run_unwrap,
            self.run_timeseries,
            self.run_detection,
        )
        for number, (label, action, hint) in enumerate(zip(_STAGE_LABELS, actions, _STAGE_HELP), start=1):
            if number == 6:
                layout.addWidget(self._method_picker())
            button = QtWidgets.QPushButton(f"{number}    {label}")
            button.setObjectName(f"stageButton{number}")
            button.setToolTip(hint)
            button.setMinimumHeight(40)
            button.clicked.connect(self._click(action, number - 1))
            self.stage_buttons.append(button)
            layout.addWidget(button)

        self.run_all_button = QtWidgets.QPushButton("Run all stages")
        self.run_all_button.setObjectName("runAll")
        self.run_all_button.setMinimumHeight(40)
        self.run_all_button.clicked.connect(self._click(self.run_all_stages, None))
        layout.addWidget(self.run_all_button)
        layout.addStretch(1)
        return panel

    def _viewer_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("View"))
        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.addItems(_LAYERS)
        self.layer_combo.currentIndexChanged.connect(lambda _index: self._refresh_view())
        controls.addWidget(self.layer_combo, stretch=1)
        self.slider_label = QtWidgets.QLabel("Index")
        self.slider_label.setObjectName("subtle")
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(lambda _value: self._refresh_view())
        controls.addWidget(self.slider_label)
        controls.addWidget(self.slider, stretch=1)
        self.export_button = QtWidgets.QPushButton("Export map")
        self.export_button.setObjectName("exportMapButton")
        self.export_button.setToolTip(
            "Write the LOS velocity map, or the displacement date on screen, "
            "as .npy, a PNG preview, and a float TIFF when tifffile is installed."
        )
        self.export_button.clicked.connect(self._click(self.export_current_map, None))
        controls.addWidget(self.export_button)
        layout.addLayout(controls)

        self.viewer = ImageView()
        self.viewer.pixelActivated.connect(self._on_pixel)
        layout.addWidget(self.viewer, stretch=1)

        self.caption = QtWidgets.QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setObjectName("subtle")
        layout.addWidget(self.caption)
        self.colorbar = ColorBar()
        layout.addWidget(self.colorbar)
        self.readout = QtWidgets.QLabel("Click the image to read a pixel.")
        self.readout.setObjectName("subtle")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        return panel

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        generate = file_menu.addAction("Generate sample SLC")
        generate.setShortcut(QtGui.QKeySequence("Ctrl+N"))
        generate.triggered.connect(self._click(self.generate_sample, 0))
        load = file_menu.addAction("Load SLC stack…")
        load.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        load.triggered.connect(self._click(self.load_stack, 0))
        open_product = QtGui.QAction("Open Sentinel-1 SAFE Product…", self)
        open_product.setObjectName("openSafeProductAction")
        open_product.triggered.connect(self._click(self.open_safe_product, None))
        file_menu.addAction(open_product)
        open_folder = QtGui.QAction("Open SAFE folder…", self)
        open_folder.setObjectName("openSafeFolderAction")
        open_folder.triggered.connect(self._click(self.open_safe_folder, None))
        file_menu.addAction(open_folder)
        open_stack = QtGui.QAction("Open SAFE stack…", self)
        open_stack.setObjectName("openSafeStackAction")
        open_stack.setToolTip(
            "Add several Sentinel-1 SAFE folders or .zip files. "
            "The list shows file name, type, acquisition, track, and orbit. "
            "One crop, swath, and polarisation become a time-sorted complex64 stack."
        )
        open_stack.triggered.connect(self._click(self.open_safe_stack, None))
        file_menu.addAction(open_stack)
        fetch = QtGui.QAction("Fetch Sentinel-1 SLC…", self)
        fetch.setObjectName("fetchSentinelAction")
        fetch.triggered.connect(self._click(self.fetch_sentinel, 0))
        file_menu.addAction(fetch)
        save = file_menu.addAction("Save products…")
        save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        save.triggered.connect(self._click(self.save_outputs, None))
        save_as = file_menu.addAction("Save Product As…")
        save_as.setObjectName("fileSaveProductAs")
        save_as.triggered.connect(self._click(self.open_save_product_dialog, None))
        view_kmz = file_menu.addAction("View as Google Earth KMZ…")
        view_kmz.setObjectName("fileViewKmz")
        view_kmz.triggered.connect(self._click(self.export_kmz, None))
        self._export_velocity_action = QtGui.QAction("Export velocity map…", self)
        self._export_velocity_action.setObjectName("exportVelocityAction")
        self._export_velocity_action.triggered.connect(self._click(self.export_velocity_map, None))
        self._export_displacement_action = QtGui.QAction("Export displacement map…", self)
        self._export_displacement_action.setObjectName("exportDisplacementAction")
        self._export_displacement_action.triggered.connect(self._click(self.export_displacement_map, None))
        file_menu.addAction(self._export_velocity_action)
        file_menu.addAction(self._export_displacement_action)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Exit")
        quit_action.triggered.connect(self.close)

        data_menu = self.menuBar().addMenu("Data")
        data_menu.addAction(fetch)
        data_menu.addAction(open_product)
        data_menu.addAction(open_folder)
        data_menu.addAction(open_stack)
        data_menu.addAction("Import local SAFE / TIFF…").triggered.connect(
            self._click(self.fetch_sentinel, 0)
        )

        pipeline = self.menuBar().addMenu("Pipeline")
        pipeline.setObjectName("pipelineMenu")
        gmtsar_steps = (
            ("PREPROCESS", self.run_preprocess, 0, "pipelinePreprocess"),
            ("ALIGN SLC", self.run_align, 0, "pipelineAlign"),
            ("MAKE TOPO_RA", self.open_topo_dialog, 1, "pipelineTopo"),
            ("MAKE INTERFEROGRAMS", self.run_make_interferograms, 1, "pipelineInterferograms"),
            ("UNWRAP PHASE", self.run_unwrap, 4, "pipelineUnwrap"),
            ("GEOCODE", self.run_geocode, None, "pipelineGeocode"),
        )
        for label, action, index, object_name in gmtsar_steps:
            item = pipeline.addAction(label)
            item.setObjectName(object_name)
            item.triggered.connect(self._click(action, index))
        apply_orbit = pipeline.addAction("Apply Orbit File…")
        apply_orbit.setObjectName("applyOrbitAction")
        apply_orbit.triggered.connect(self._click(self.open_orbit_dialog, None))
        pipeline.addSeparator()
        steps = (
            ("Interferogram", self.open_interferogram_dialog, 1),
            ("Goldstein filter", self.run_filter, 2),
            ("Coherence", self.run_coherence, 3),
            ("Unwrap", self.run_unwrap, 4),
            ("Run time series", self.run_timeseries, 5),
            ("Displacement detection mask", self.run_detection, 6),
        )
        for label, action, index in steps:
            item = pipeline.addAction(label)
            if label == "Interferogram":
                item.setObjectName("interferogramFormationAction")
            item.triggered.connect(self._click(action, index))
        pipeline.addSeparator()
        run_all = pipeline.addAction("Run all stages")
        run_all.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        run_all.triggered.connect(self._click(self.run_all_stages, None))
        settings = pipeline.addAction("Settings…")
        settings.triggered.connect(self.open_settings)

        self._build_filter_menus()
        self._build_unwrap_menus()
        self._build_method_menus()
        self._build_radar_menu()

        view = self.menuBar().addMenu("View")
        for name in _LAYERS:
            item = view.addAction(name)
            item.triggered.connect(lambda _checked=False, layer=name: self._select_layer(layer, refresh=True))
        view.addSeparator()
        view.addAction(self._export_velocity_action)
        view.addAction(self._export_displacement_action)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("GMTSAR flowchart").triggered.connect(
            lambda: self._show_text("GMTSAR flowchart", FLOWCHART_TEXT)
        )
        help_menu.addAction("Sign conventions").triggered.connect(
            lambda: self._show_text("Sign conventions", CONVENTIONS_TEXT)
        )
        help_menu.addAction("Limitations").triggered.connect(
            lambda: self._show_text("Limitations", LIMITATIONS_TEXT)
        )
        help_menu.addAction("About").triggered.connect(self._about)

    def _click(self, action, stage: int | None):
        def handler(_checked: bool = False) -> None:
            self._guard(action, stage)

        return handler

    def _guard(self, action, stage: int | None) -> None:
        if self._busy:
            return
        self._busy = True
        QtWidgets.QApplication.processEvents()
        try:
            action()
            self._sync_stage_states()
            if self._preferred_layer:
                self._select_layer(self._preferred_layer, refresh=False)
                self._preferred_layer = None
            self._refresh_view()
            if self._product_status:
                self.statusBar().showMessage(self._product_status)
            else:
                self.statusBar().showMessage("Ready", 4000)
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self.statusBar().showMessage(str(exc), 12000)
            self._sync_stage_states()
            if stage is not None:
                self._paint_stage(stage, "error")
        finally:
            self._busy = False

    def generate_sample(self) -> None:
        self.workspace.stack = generate_demo_stack()
        self.workspace.dem_m = None
        self.workspace.safe_inputs = []
        self.workspace.clear_products()
        stack = self.workspace.stack
        self._log(f"Generated sample SLC {stack.data.shape} {stack.data.dtype}. {stack.description}")
        self._preferred_layer = "SLC amplitude"

    def load_stack(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load SLC stack",
            "",
            "NumPy SLC (*.npy)",
        )
        if not path:
            return
        stack = load_slc(path)
        self.workspace.stack = stack
        self.workspace.dem_m = None
        self.workspace.safe_inputs = []
        self.workspace.clear_products()
        self._log(f"Loaded SLC {stack.data.shape} {stack.data.dtype} from {path}. {stack.description}")
        self._preferred_layer = "SLC amplitude"

    def _opened_safe_paths(self) -> list:
        if self.safe_product is None:
            return []
        return [self.safe_product.root]

    def _source_names(self) -> list[str]:
        names: list[str] = []
        if self.workspace.stack is not None:
            names.append(f"Workspace SLC stack {self.workspace.stack.data.shape}")
        if self.safe_product is not None:
            names.append(self.safe_product.product_id)
        return names

    def open_safe_product(self) -> None:
        """Choose one or more SAFE zips or folders, then list every one before loading."""
        self._open_product_set_from_chooser()

    def open_safe_folder(self) -> None:
        """Choose several SAFE folders (and zips) in one dialog. None are dropped."""
        self._open_product_set_from_chooser()

    def open_safe_stack(self) -> None:
        """List SAFE folders and zips, then build one time-sorted complex64 stack."""
        self._open_product_set_from_chooser()

    def _open_product_set_from_chooser(self) -> None:
        from ai_insar.gui.product_set import choose_safe_products

        paths = choose_safe_products(self)
        if not paths:
            return
        dialog = SafeStackDialog(
            self,
            n_files=len(paths),
            initial_paths=paths,
            opened_paths=self._opened_safe_paths,
        )
        self._log(dialog.product_set.summary())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            self._log(f"SAFE list closed. {len(paths)} product(s) were not loaded.")
            return
        self.load_from_product_set(dialog)

    def load_from_product_set(self, dialog: SafeStackDialog) -> None:
        """Load every listed product and write the count, names, and metadata to the log."""
        paths = dialog.paths()
        if not paths:
            raise RuntimeError("ProductSet-Reader: please add a source product.")
        if dialog.resampling.currentText() != "NONE":
            raise RuntimeError("InSAR requires Resampling Type NONE.")
        self._log(dialog.product_set.summary())
        self._product_status = f"{len(paths)} Products"
        self.load_safe_stack(
            paths,
            swath=dialog.swath(),
            pol=dialog.polarisation(),
            row=int(dialog.row.value()),
            col=int(dialog.col.value()),
            height=int(dialog.height.value()),
            width=int(dialog.width.value()),
            form_interferograms=dialog.form_ifg.isChecked(),
            reference=dialog.reference_path(),
            burst_first=dialog.burst_first_value(),
            burst_last=dialog.burst_last_value(),
        )

    def load_safe_stack(
        self,
        paths,
        *,
        swath: str,
        pol: str,
        row: int,
        col: int,
        height: int,
        width: int,
        form_interferograms: bool = True,
        reference=None,
        burst_first: int = 0,
        burst_last: int = 0,
    ) -> None:
        """Read every path, then optionally form interferograms. All products stay in the cube."""
        from pathlib import Path

        from ai_insar.safe_stack import SafeStackCancelled, inspect_safe_products, stack_safe_zips

        files = [Path(path) for path in paths]
        records = inspect_safe_products(
            files,
            swath=swath,
            pol=pol,
            reference=reference,
            burst_first=burst_first,
            burst_last=burst_last,
        )
        dialog = QtWidgets.QProgressDialog(
            "Reading SAFE products…",
            "Cancel",
            0,
            max(len(files) * 2, 1),
            self,
        )
        dialog.setWindowTitle("Open SAFE stack")
        dialog.setObjectName("safeStackProgress")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        def progress(step: int, steps: int, label: str):
            if dialog.wasCanceled():
                return False
            dialog.setMaximum(steps)
            dialog.setValue(max(0, step - 1))
            dialog.setLabelText(label)
            QtWidgets.QApplication.processEvents()
            if dialog.wasCanceled():
                return False
            return None

        try:
            stack = stack_safe_zips(
                files,
                swath=swath,
                pol=pol,
                row=row,
                col=col,
                height=height,
                width=width,
                progress=progress,
            )
        except SafeStackCancelled:
            self._log("SAFE stack read cancelled.")
            return
        finally:
            dialog.setValue(dialog.maximum())
            dialog.close()

        self.workspace.stack = stack
        self.workspace.dem_m = None
        self.workspace.safe_inputs = list(records)
        reference = next(item for item in records if item.is_reference)
        day = (reference.acquisition or "")[:10]
        matched = np.where(stack.dates.astype(str) == day)[0] if day else np.array([])
        self.workspace.reference_index = int(matched[0]) if matched.size == 1 else None
        self.workspace.clear_products()
        reference_name = next(item.file_name for item in records if item.is_reference)
        names = ", ".join(item.file_name for item in records)
        tracks = {item.track for item in records if item.track}
        self._log(
            f"SAFE stack {stack.data.shape} {stack.data.dtype} from {len(records)} product(s): {names}. "
            f"Reference: {reference_name}. "
            f"{stack.description}"
        )
        if len(tracks) > 1:
            self._log(
                "These products do not share one relative orbit (track): "
                + ", ".join(f"{item.file_name} track {item.track}" for item in records)
                + ". All of them stay in the stack."
            )
        if records[0].burst_first is not None:
            self._log(
                f"Burst range {records[0].burst_first}–{records[0].burst_last} is stored for each product. "
                "The array is the shared row/column crop, not a burst stitch."
            )
        self._preferred_layer = "SLC amplitude"
        if form_interferograms and stack.n_dates >= 2:
            self.run_interferograms()
        elif form_interferograms:
            self._log("One SAFE zip is in the stack. A second date is required before an interferogram.")

    def open_safe_path(self, path) -> None:
        """Inspect a SAFE zip (or a folder / manifest). Pixels are not read."""
        from pathlib import Path

        product = resolve_safe(Path(path))
        self.safe_product = product
        self.product_explorer.set_product(product)
        self.product_explorer.show()
        self._product_status = format_product_status(product)
        self.statusBar().showMessage(self._product_status)
        self._log(f"Opened Sentinel-1 product {product.product_id}. {self._product_status}")

    def preview_safe_band(self, swath: str, pol: str) -> None:
        product = self._require_safe_product()
        measurement = select_measurement(product, swath, pol)
        amplitude = preview_amplitude(product, measurement)
        self.product_explorer.show_preview(
            amplitude,
            f"Amplitude {measurement.swath} {measurement.polarisation}, "
            f"{amplitude.shape[0]}×{amplitude.shape[1]} (decimated). Not geocoded.",
        )
        self._log(
            f"Preview {measurement.swath} {measurement.polarisation} amplitude "
            f"{amplitude.shape[0]}×{amplitude.shape[1]}."
        )

    def preview_safe_quicklook(self) -> None:
        product = self._require_safe_product()
        amplitude = preview_amplitude(product)
        self.product_explorer.show_preview(
            amplitude,
            f"Quick-look {amplitude.shape[0]}×{amplitude.shape[1]}.",
        )
        self._log(f"Preview quick-look {amplitude.shape[0]}×{amplitude.shape[1]}.")

    def add_safe_to_stack(self) -> None:
        product = self._require_safe_product()
        swath, pol = self.product_explorer.selected_swath_pol()
        dialog = SlcCropDialog(self, product_id=product.product_id, swath=swath, pol=pol)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.import_safe_crop(
            product,
            swath=swath,
            pol=pol,
            row=int(dialog.row.value()),
            col=int(dialog.col.value()),
            height=int(dialog.height.value()),
            width=int(dialog.width.value()),
        )

    def import_safe_crop(self, product, *, swath: str, pol: str, row: int, col: int, height: int, width: int) -> None:
        stack = import_slc_paths(
            [product.root],
            swath=swath,
            pol=pol,
            row=row,
            col=col,
            height=height,
            width=width,
        )
        self.workspace.stack = stack
        self.workspace.dem_m = None
        self.workspace.clear_products()
        self._log(
            f"Added {swath} {pol} from {product.product_id} to the SLC stack "
            f"{stack.data.shape} {stack.data.dtype}. {stack.description}"
        )
        self._preferred_layer = "SLC amplitude"

    def _require_safe_product(self):
        if self.safe_product is None:
            raise RuntimeError("Open a Sentinel-1 SAFE product first.")
        return self.safe_product

    def fetch_sentinel(self) -> None:
        dialog = FetchSentinelDialog(self)
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        if dialog.downloaded_zips:
            self.open_safe_path(dialog.downloaded_zips[0])
        if not accepted or dialog.stack is None:
            return
        self.workspace.stack = dialog.stack
        self.workspace.dem_m = None
        self.workspace.clear_products()
        stack = dialog.stack
        self._log(
            f"Imported SLC {stack.data.shape} {stack.data.dtype}. {stack.description}"
        )
        self._preferred_layer = "SLC amplitude"

    def save_outputs(self) -> None:
        if self.workspace.stack is None:
            raise RuntimeError("Nothing to save yet. Generate or load an SLC stack first.")
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Save products")
        if not directory:
            return
        save_products(self.workspace, directory)
        self._log(f"Saved products to {directory}")

    def run_preprocess(self) -> None:
        """GMTSAR PREPROCESS: keep a loaded stack, or build the sample stack."""
        if self.workspace.stack is None:
            self.generate_sample()
            self._log("PREPROCESS: built the sample SLC because no stack was loaded.")
            return
        stack = self.workspace.stack
        self._log(
            f"PREPROCESS: SLC stack {stack.data.shape} {stack.data.dtype} is loaded "
            f"({stack.sensor}). File → Load SLC stack or Open SAFE stack replaces it. "
            "A SAFE manifest can show the orbit number. That orbit is not used to resample the crop."
        )

    def run_align(self) -> None:
        """GMTSAR ALIGN SLC: constant amplitude shift onto the first date."""
        if self.workspace.stack is None:
            raise RuntimeError("Load or generate an SLC stack before ALIGN SLC.")
        aligned, shifts = align_slc_stack(self.workspace.stack.data, reference=0)
        self.workspace.stack.data = aligned
        self.workspace.clear_products()
        pieces = [f"date {index}: azi {shift_y:+.2f} px, rng {shift_x:+.2f} px" for index, (shift_y, shift_x) in enumerate(shifts)]
        self._log(
            "ALIGN SLC: amplitude cross-correlation onto date 0, FFT phase ramp. "
            + "; ".join(pieces)
            + ". Interferograms were cleared. Run all stages does not run this step."
        )
        self._preferred_layer = "SLC amplitude"

    def run_make_interferograms(self) -> None:
        """GMTSAR MAKE INTERFEROGRAMS: phase, Goldstein, and coherence."""
        self.run_interferograms()
        self.run_filter()
        self.run_coherence()

    def run_topo(self) -> None:
        """Open Topographic Phase Removal and wait for the DEM and baseline."""
        self.open_topo_dialog()

    def _build_radar_menu(self) -> None:
        """The six SNAP 14 dialogs observed for handwritten steps 1–6."""
        menu = self.menuBar().addMenu("Radar")
        menu.setObjectName("radarMenu")
        actions = (
            ("S-1 TOPS Split", self.open_split_dialog, "radarTopsSplit"),
            ("Apply Orbit File", self.open_orbit_dialog, "radarApplyOrbit"),
            ("S-1 Back Geocoding", self.open_backgeocoding_dialog, "radarBackGeocoding"),
            ("S-1 Enhanced Spectral Diversity", self.open_esd_dialog, "radarEsd"),
            ("Interferogram Formation", self.open_interferogram_dialog, "radarInterferogram"),
            ("S-1 TOPS Deburst", self.open_deburst_dialog, "radarTopsDeburst"),
            ("Multilooking", self.open_multilook_dialog, "radarMultilook"),
            ("Goldstein Phase Filtering", self.open_goldstein_dialog, "radarGoldstein"),
            ("Snaphu Export", self.open_snaphu_export_dialog, "radarSnaphuExport"),
            ("Batch Snaphu Unwrap", self.open_batch_snaphu_dialog, "radarBatchSnaphu"),
            ("Snaphu Import", self.open_snaphu_import_dialog, "radarSnaphuImport"),
            ("Phase to Displacement", self.open_phase_dialog, "radarPhaseToDisplacement"),
            ("Range-Doppler Terrain Correction", self.open_terrain_dialog, "radarTerrainCorrection"),
        )
        for label, callback, object_name in actions:
            item = menu.addAction(label)
            item.setObjectName(object_name)
            item.triggered.connect(self._click(callback, None))
        subset = menu.addAction("Subset…")
        subset.setObjectName("radarSubset")
        subset.triggered.connect(self._click(self.open_subset_dialog, None))
        save_as = menu.addAction("Save Product As…")
        save_as.setObjectName("radarSaveProductAs")
        save_as.triggered.connect(self._click(self.open_save_product_dialog, None))

    def open_split_dialog(self) -> None:
        dialog = TopsSplitDialog(self)
        if self.safe_product is not None:
            dialog.bind_product(self.safe_product)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_split(dialog.parameters())

    def open_deburst_dialog(self) -> None:
        dialog = TopsDeburstDialog(self, can_run=self.workspace.reference_grid is not None)
        if self.safe_product is not None:
            dialog.bind_product(self.safe_product)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_deburst(dialog.parameters())

    def open_esd_dialog(self) -> None:
        ready = self.workspace.reference_grid is not None and self.workspace.secondary_grid is not None
        dialog = EsdDialog(self, sources=self._source_names(), can_run=ready)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_esd(dialog.parameters())

    def open_backgeocoding_dialog(self) -> None:
        dialog = BackGeocodingDialog(self, runner=self.apply_backgeocoding)
        dialog.exec()

    def open_phase_dialog(self) -> None:
        dialog = PhaseToDisplacementDialog(self, sources=self._source_names())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_phase_to_displacement(dialog.parameters())

    def open_terrain_dialog(self) -> None:
        dialog = TerrainCorrectionDialog(self, sources=self._source_names())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_terrain_correction(dialog.parameters())

    def open_subset_dialog(self) -> None:
        height, width = self._spatial_shape()
        if height < 1 or width < 1:
            raise RuntimeError("Subset needs a loaded raster. Nothing is loaded.")
        dialog = SubsetDialog(self, height=height, width=width)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_subset(dialog.parameters())

    def open_save_product_dialog(self) -> None:
        dialog = SaveProductDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_save_product(dialog.parameters())

    def open_multilook_dialog(self) -> None:
        bands: list[str] = []
        if self.workspace.interferograms is not None:
            bands.append("Interferogram")
        if self.workspace.coherence is not None:
            bands.append("Coherence")
        dialog = MultilookDialog(self, bands=bands)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_multilook(dialog.parameters())

    def open_snaphu_export_dialog(self) -> None:
        dialog = SnaphuExportDialog(self, has_interferogram=self.workspace.interferograms is not None)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_snaphu_export(dialog.parameters())

    def open_batch_snaphu_dialog(self) -> None:
        dialog = BatchSnaphuUnwrapDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_batch_snaphu(dialog.parameters())

    def open_snaphu_import_dialog(self) -> None:
        dialog = SnaphuImportDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_snaphu_import(dialog.parameters())

    def apply_multilook(self, params: dict) -> None:
        from ai_insar.pipeline.multilook import multilook_array

        if params["gr_square"]:
            model = getattr(self.workspace, "swath_model", None)
            if model is None:
                raise RuntimeError(
                    "GR Square Pixel needs the Sentinel-1 swath geometry "
                    "(azimuth time interval, incidence, and orbit). It is not on this stack."
                )
            from ai_insar.pipeline.tops_ops import square_pixel_azimuth_looks

            params = dict(params)
            params["azimuth_looks"] = square_pixel_azimuth_looks(
                model, int(params["range_looks"]), getattr(self.workspace, "precise_orbit", None)
            )
            params["gr_square"] = False
            self._log(
                f"GR Square Pixel: {params['range_looks']} range looks -> "
                f"{params['azimuth_looks']} azimuth looks from the ground spacings."
            )
        bands = list(params["bands"])
        if not bands:
            raise RuntimeError("Source Bands is empty.")
        range_looks = int(params["range_looks"])
        azimuth_looks = int(params["azimuth_looks"])
        intensity = bool(params["output_intensity"])
        if "Interferogram" in bands:
            if self.workspace.interferograms is None:
                raise RuntimeError("No interferogram is loaded.")
            self.workspace.interferograms = multilook_array(
                self.workspace.interferograms,
                range_looks,
                azimuth_looks,
                output_intensity=intensity,
            )
            self.workspace.filtered = None
            self.workspace.unwrapped = None
            self.workspace.los_m = None
            self.workspace.velocity_m_yr = None
            self.workspace.mask = None
            if "Coherence" not in bands:
                self.workspace.coherence = None
                self.workspace.mean_coherence = None
        if "Coherence" in bands:
            if self.workspace.coherence is None:
                raise RuntimeError("No coherence is loaded.")
            self.workspace.coherence = multilook_array(
                self.workspace.coherence,
                range_looks,
                azimuth_looks,
                output_intensity=False,
            )
            self.workspace.mean_coherence = np.mean(self.workspace.coherence, axis=0).astype(np.float32)
        shape = self.workspace.interferograms.shape if self.workspace.interferograms is not None else self.workspace.coherence.shape
        self._log(
            f"Multilook independent looks: {azimuth_looks} azimuth × {range_looks} range. "
            f"Output shape {shape}. "
            + (
                "Output is intensity."
                if intensity
                else "Complex samples were averaged. This is not a resample."
            )
        )

    def apply_snaphu_export(self, params: dict) -> None:
        from pathlib import Path

        from ai_insar.pipeline.snaphu_io import export_snaphu

        if self.workspace.interferograms is None or self.workspace.coherence is None:
            raise RuntimeError("Snaphu export needs an interferogram and a coherence image. Zeros were not written.")
        if self.workspace.interferograms.shape != self.workspace.coherence.shape:
            raise RuntimeError("Interferogram and coherence shapes differ.")
        pair_count = int(self.workspace.interferograms.shape[0])
        phase = np.angle(self.workspace.interferograms[0]).astype(np.float64)
        coherence = np.asarray(self.workspace.coherence[0], dtype=np.float64)
        wavelength = None if self.workspace.stack is None else float(self.workspace.stack.wavelength_m)
        conf = export_snaphu(
            params["folder"],
            phase,
            coherence,
            stat_mode=params["stat_mode"],
            init_method=params["init_method"],
            tile_rows=int(params["tile_rows"]),
            tile_columns=int(params["tile_columns"]),
            processors=int(params["processors"]),
            row_overlap=int(params["row_overlap"]),
            column_overlap=int(params["column_overlap"]),
            tile_cost_threshold=int(params["tile_cost_threshold"]),
            wavelength_m=wavelength,
            pair_index=0,
            pair_count=pair_count,
        )
        extra = ""
        if pair_count > 1:
            extra = f" Pairs 1..{pair_count - 1} were not exported."
        self._log(
            f"Snaphu export wrote {conf.name} in {Path(params['folder'])} "
            f"for pair 0 of {pair_count}, shape {phase.shape}. Not unwrapped.{extra}"
        )

    def apply_batch_snaphu(self, params: dict) -> None:
        from ai_insar.pipeline.snaphu_io import run_snaphu

        completed = run_snaphu(params["export_folder"], params["install_location"])
        self._log(
            "External snaphu finished with exit 0. "
            f"stdout: {(completed.stdout or '').strip() or '(empty)'}. "
            "Fast unwrap was not used."
        )

    def apply_snaphu_import(self, params: dict) -> None:
        from pathlib import Path

        from ai_insar.pipeline.snaphu_io import import_unwrapped

        phase_folder = Path(params["phase_folder"])
        unwrap_folder = Path(params["unwrapped_folder"])
        if phase_folder.resolve() != unwrap_folder.resolve():
            raise RuntimeError("The phase folder and the unwrapped folder must be the same export.")
        unwrapped = import_unwrapped(unwrap_folder)
        self.workspace.unwrapped = unwrapped[None, ...]
        self.workspace.los_m = None
        self.workspace.velocity_m_yr = None
        self.workspace.mask = None
        if params["skip_wrapped"]:
            self._log(
                "Do NOT save Wrapped interferogram is checked. "
                "BEAM-DIMAP was not written. The wrapped interferogram remains in the workspace."
            )
        self._log(
            f"Imported unwrapped phase {unwrapped.shape} from {unwrap_folder / 'unwrap.img'}. "
            "The values were read from the file. They were not unwrapped again."
        )

    def apply_split(self, params: dict) -> None:
        """Read one subswath's burst lines. This does not deburst or coregister."""
        from pathlib import Path

        from ai_insar.conventions import C_BAND_WAVELENGTH_M, SPEED_OF_LIGHT_M_S
        from ai_insar.safe import resolve_safe
        from ai_insar.sentinel import date_from_text

        product = resolve_safe(params["source"])
        data, measurement, (row, height, width) = split_swath(
            product,
            swath=params["swath"],
            pol=params["polarisation"],
            burst_first=int(params["burst_first"]),
            burst_last=int(params["burst_last"]),
        )
        if data.ndim != 2:
            raise RuntimeError(f"TOPS Split returned {data.shape}, expected a 2-D window.")
        day = date_from_text(product.start_utc) or date_from_text(product.product_id)
        if day is None:
            raise RuntimeError(f"{product.product_id} has no acquisition date.")
        wavelength = C_BAND_WAVELENGTH_M
        if measurement.radar_frequency:
            wavelength = SPEED_OF_LIGHT_M_S / float(measurement.radar_frequency)
        description = (
            f"TOPS Split {measurement.swath} {measurement.polarisation} "
            f"bursts {int(params['burst_first'])}–{int(params['burst_last'])}, "
            f"rows {row}:{row + height}, samples {width}, from {Path(params['source']).name}. "
            "Burst overlap is still in the lines. This is not a deburst."
        )
        self.workspace.stack = SLCStack(
            data=np.asarray(data, dtype=np.complex64)[None, ...],
            dates=np.array([day], dtype="datetime64[D]"),
            wavelength_m=float(wavelength),
            sensor="Sentinel-1 SLC",
            description=description,
        )
        self.workspace.clear_products()
        self._store_split_grid(product, measurement, data, row)
        self._log(description)
        self._preferred_layer = "SLC amplitude"

    def _store_split_grid(self, product, measurement, data: np.ndarray, row: int) -> None:
        """Keep the split window as a radar grid. Zip annotations are read in place."""
        from ai_insar.pipeline.s1_annotation import read_swath_from_product
        from ai_insar.pipeline.tops_ops import RadarGrid

        if not measurement.annotation_href:
            self._log("TOPS Split kept the burst lines. The measurement has no annotation.")
            return
        model = read_swath_from_product(product, measurement)
        grid = RadarGrid(int(row), 0, np.asarray(data, dtype=np.complex64), model)
        selected = any(
            item.is_reference and item.product_id == product.product_id for item in self.workspace.safe_inputs
        )
        if selected or self.workspace.reference_grid is None:
            if selected and self.workspace.reference_grid is not None:
                self.workspace.secondary_grid = self.workspace.reference_grid
                self.workspace.secondary_orbit = self.workspace.precise_orbit
            self.workspace.reference_grid = grid
            self.workspace.swath_model = model
            if self.workspace.precise_orbit is None:
                self.workspace.precise_orbit = model.orbit
            self._log(f"Stored the reference radar window {data.shape} from the SAFE annotation.")
            return
        self.workspace.secondary_grid = grid
        self.workspace.secondary_orbit = model.orbit
        self._log(f"Stored the secondary radar window {data.shape} from the SAFE annotation.")

    def open_interferogram_dialog(self) -> None:
        dialog = InterferogramDialog(self, sources=self._source_names())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_interferogram_params(dialog.parameters())

    def open_coherence_dialog(self) -> None:
        dialog = CoherenceDialog(self, sources=self._source_names())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_coherence_params(dialog.parameters())

    def open_topo_dialog(self) -> None:
        dialog = TopoPhaseDialog(
            self,
            sources=self._source_names(),
            baseline_m=float(self.workspace.topo_baseline_perp_m),
            slant_range_m=float(self.workspace.topo_slant_range_m),
            incidence_deg=float(self.workspace.topo_incidence_deg),
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_topo_params(dialog.parameters())

    def open_goldstein_dialog(self) -> None:
        dialog = GoldsteinDialog(self, sources=self._source_names(), config=self.workspace.config)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_goldstein_params(dialog.parameters())

    def open_orbit_dialog(self) -> None:
        dialog = ApplyOrbitDialog(self, sources=self._source_names())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_orbit_params(dialog.parameters())

    def apply_interferogram_params(self, params: dict) -> None:
        request = params["request"]
        for line in request.skipped:
            self._log(line)
        if params.get("subtract_ramp"):
            raise RuntimeError("Subtract Residual Ramp (GSLC) is not implemented.")
        self.run_interferograms()
        if params.get("subtract_flat") or params.get("subtract_topo"):
            self._remove_geometric_phase(params)
        if params.get("output_coherence"):
            self._set_coherence_windows(
                int(params["coherence_range"]),
                None if params["square_pixel"] else int(params["coherence_azimuth"]),
                square=bool(params["square_pixel"]),
            )
            self.run_coherence()
        if request.open_in_app:
            self._preferred_layer = "Coherence" if params.get("output_coherence") else "Interferogram phase"
        self._save_request(request, self.workspace.interferograms, PRODUCT_SUFFIX["interferogram"])
        if params.get("output_coherence") and request.save and self.workspace.coherence is not None:
            self._save_request(request, self.workspace.coherence, PRODUCT_SUFFIX["coherence"])

    def _remove_geometric_phase(self, params: dict) -> None:
        """Subtract ellipsoid flat-earth and, when asked, a constant-height topographic term."""
        from ai_insar.pipeline.tops_ops import RadarGrid, flat_earth_phase, subtract_phase

        early = getattr(self.workspace, "reference_grid", None)
        later = getattr(self.workspace, "secondary_grid", None)
        early_orbit = getattr(self.workspace, "precise_orbit", None)
        later_orbit = getattr(self.workspace, "secondary_orbit", None)
        if early is None or later is None or early_orbit is None or later_orbit is None:
            raise RuntimeError(
                "Flat-earth or topographic subtraction needs both radar windows and both precise orbits. "
                "They are not on this stack."
            )
        dem = None
        if params.get("subtract_topo"):
            dem = getattr(self.workspace, "elevation", None)
            if dem is None:
                raise RuntimeError(
                    "Topographic subtraction needs an elevation model sampled at each pixel. "
                    "A single DEM mean is not used."
                )
        phase = flat_earth_phase(
            early,
            later,
            early_orbit=early_orbit,
            later_orbit=later_orbit,
            degree=int(self._combo_int(params, "orbit_degree", 3)),
            height_m=0.0,
            dem=dem,
        )
        ifg = self.workspace.interferograms
        if ifg is None or ifg.shape[0] < 1 or ifg.shape[1:] != phase.shape:
            raise RuntimeError("The interferogram grid does not match the radar windows.")
        ifg = np.array(ifg, copy=True)
        ifg[0] = subtract_phase(ifg[0], phase)
        self.workspace.interferograms = ifg
        self._log(
            f"Removed geometric phase (height {height:.1f} m) from pair 0. "
            f"RMS {float(np.sqrt(np.mean(phase**2))):.3f} rad."
        )

    def _combo_int(self, params: dict, key: str, default: int) -> int:
        try:
            return int(params.get(key, default))
        except (TypeError, ValueError):
            return default

    def apply_coherence_params(self, params: dict) -> None:
        if self.workspace.stack is None or self.workspace.pairs is None:
            raise RuntimeError("Form interferograms before estimating coherence.")
        request = params["request"]
        for line in request.skipped:
            self._log(line)
        self._set_coherence_windows(
            int(params["coherence_range"]),
            None if params["square_pixel"] else int(params["coherence_azimuth"]),
            square=bool(params["square_pixel"]),
        )
        if params["single_reference"]:
            pairs = [(0, later) for later in range(1, self.workspace.stack.n_dates)]
            if pairs != list(self.workspace.pairs):
                self.workspace.pairs = pairs
                self.workspace.interferograms = form_interferograms(self.workspace.stack.data, pairs)
                self.workspace.filtered = None
                self.workspace.unwrapped = None
                self.workspace.los_m = None
                self.workspace.velocity_m_yr = None
                self.workspace.mask = None
                self._log(
                    "Single reference: pairs are date 0 with each later date. "
                    "Interferograms were recomputed."
                )
        self.run_coherence()
        if request.open_in_app:
            self._preferred_layer = "Coherence"
        self._save_request(request, self.workspace.coherence, PRODUCT_SUFFIX["coherence"])

    def apply_goldstein_params(self, params: dict) -> None:
        request = params["request"]
        for line in request.skipped:
            self._log(line)
        written = {"NumPy (.npy)", "BEAM-DIMAP", "GeoTIFF", "Google Earth KMZ"}
        if request.save and request.format_name not in written:
            raise RuntimeError(f"{request.format_name} is not written by this build.")
        alpha = float(params["alpha"])
        fft_size = int(params["fft_size"])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if fft_size < 8:
            raise ValueError("FFT size must be at least 8")
        if float(params.get("block_extent_m", 0.0)) != 0.0:
            raise RuntimeError(
                "FFT Block Extent (m) is not zero. Pixel spacing is not available, "
                "so the FFT size in pixels was not replaced."
            )
        self.workspace.config.goldstein_alpha = alpha
        self.workspace.config.goldstein_window = fft_size
        self.workspace.config.goldstein_spectrum_smooth = int(params["window_size"])
        if params.get("adaptive") or params.get("coherence_mask"):
            self._goldstein_with_coherence(params)
        else:
            self._log(
                f"Goldstein constant alpha={alpha:.2f}, FFT size={fft_size}, "
                f"window={int(params['window_size'])}. Suffix {PRODUCT_SUFFIX['goldstein']}."
            )
            self.run_filter()
        if request.open_in_app:
            self._preferred_layer = "Filtered phase"
        if request.save:
            self._save_request(request, self.workspace.filtered, PRODUCT_SUFFIX["goldstein"])

    def _goldstein_with_coherence(self, params: dict) -> None:
        if self.workspace.interferograms is None or self.workspace.coherence is None:
            raise RuntimeError(
                "Use adaptive alpha or the coherence mask needs a coherence image "
                "of the same shape as the interferogram. Constant alpha was not substituted."
            )
        if self.workspace.coherence.shape != self.workspace.interferograms.shape:
            raise RuntimeError("Coherence and interferogram shapes differ.")
        filtered = np.empty_like(self.workspace.interferograms, dtype=np.complex64)
        threshold = float(params.get("coherence_threshold", 0.2))
        use_mask = bool(params.get("coherence_mask"))
        adaptive = bool(params.get("adaptive"))
        for index in range(self.workspace.interferograms.shape[0]):
            plane = np.array(self.workspace.interferograms[index], dtype=np.complex64, copy=True)
            coherence = self.workspace.coherence[index]
            if use_mask:
                plane[coherence < threshold] = 0
            filtered[index] = goldstein_filter(
                plane,
                alpha=float(params["alpha"]),
                window=int(params["fft_size"]),
                spectrum_smooth=int(params["window_size"]),
                coherence=coherence if adaptive else None,
                alpha_min=float(params.get("alpha_min", 0.2)),
                alpha_max=float(params.get("alpha_max", 1.0)),
            )
        self.workspace.filtered = filtered
        self.workspace.unwrapped = None
        self.workspace.los_m = None
        self.workspace.velocity_m_yr = None
        self.workspace.mask = None
        mode = "adaptive block coherence" if adaptive else "constant alpha"
        if use_mask:
            mode += f", coherence mask < {threshold:.2f}"
        self._log(f"Goldstein {mode}. FFT size={int(params['fft_size'])}. Fast unwrap was not used.")

    def apply_topo_params(self, params: dict) -> None:
        request = params["request"]
        for line in request.skipped:
            self._log(line)
        self._apply_topo_choice(
            params["dem_choice"],
            params["dem_path"],
            baseline_m=float(params["baseline_m"]),
            slant_range_m=float(params["slant_range_m"]),
            incidence_deg=float(params["incidence_deg"]),
        )
        if request.open_in_app:
            self._preferred_layer = "Interferogram phase"
        self._save_request(request, self.workspace.interferograms, PRODUCT_SUFFIX["topo"])

    def apply_orbit_params(self, params: dict) -> None:
        request = params["request"]
        degree = int(params["polynomial_degree"])
        if degree < 1:
            raise ValueError("Polynomial degree must be at least 1.")
        orbit_file = str(params.get("orbit_file") or "").strip()
        if orbit_file:
            self._install_precise_orbit(orbit_file, degree, params)
            return
        orbit_type = str(params.get("orbit_type") or "")
        if orbit_type.startswith("Sentinel Precise") or orbit_type.startswith("Sentinel Restituted"):
            fetched = self._download_sentinel_orbit(params, degree)
            if fetched:
                return
        if not params["do_not_fail"]:
            raise RuntimeError(
                "Apply Orbit File: no orbit-file download is installed. "
                f"Requested {params['orbit_type']}, polynomial degree {degree}. "
                "State vectors were not changed."
            )
        self._log(
            "Apply Orbit File: 'Do not fail if new orbit file is not found' is checked. "
            "No orbit file was fetched. State vectors were not changed. "
            f"Requested {params['orbit_type']}, polynomial degree {degree}."
        )
        if not request.save:
            return
        path = write_orbit_stub(
            request.directory,
            request.target,
            {
                "applied": False,
                "reason": "Orbit files are not downloaded in this build.",
                "orbit_type": params["orbit_type"],
                "polynomial_degree": degree,
                "do_not_fail": bool(params["do_not_fail"]),
                "source": params.get("source", ""),
            },
        )
        self._log(f"Saved {path.name} ({PRODUCT_SUFFIX['apply_orbit']}).")

    def _install_precise_orbit(self, orbit_file: str, degree: int, params: dict) -> None:
        from pathlib import Path

        from ai_insar.pipeline.s1_annotation import read_eof_orbit
        from ai_insar.pipeline.tops_ops import RadarGrid, _with_orbit

        path = Path(orbit_file)
        if not path.is_file():
            raise RuntimeError(f"Orbit file {path} was not found. State vectors were not changed.")
        precise = read_eof_orbit(path)
        request = params["request"]
        model = getattr(self.workspace, "swath_model", None)
        shift = None
        when = None
        if model is not None and model.bursts:
            when = model.bursts[len(model.bursts) // 2].azimuth_time
            if model.orbit.time[0] <= when <= model.orbit.time[-1] and precise.time[0] <= when <= precise.time[-1]:
                before, _velocity = model.orbit.interpolate(when, degree)
                after, velocity = precise.interpolate(when, degree)
                shift = float(np.linalg.norm(after - before))
                updated = _with_orbit(model, precise)
                self.workspace.swath_model = updated
                grid = self.workspace.reference_grid
                if grid is not None:
                    self.workspace.reference_grid = RadarGrid(grid.row, grid.col, grid.data, updated)
                self.workspace.precise_orbit = precise
                self._log(
                    f"Apply Orbit File replaced state vectors from {path.name}. "
                    f"Shift {shift:.3f} m at {when}. Speed {float(np.linalg.norm(velocity)):.1f} m/s."
                )
            elif precise.time[0] <= when <= precise.time[-1]:
                self.workspace.secondary_orbit = precise
                self._log(f"Apply Orbit File stored {path.name} as the secondary orbit. It covers {when}.")
            else:
                raise RuntimeError(
                    f"{path.name} does not cover {when}. State vectors were not changed."
                )
        else:
            self.workspace.precise_orbit = precise
            self._log(
                f"Apply Orbit File stored {path.name}. "
                "No swath annotation was loaded, so the metre shift was not measured."
            )
        if not request.save:
            return
        payload = {
            "applied": True,
            "orbit_file": str(path),
            "polynomial_degree": degree,
            "shift_m": shift,
            "time": None if when is None else str(when),
        }
        written = write_orbit_stub(request.directory, request.target, payload)
        self._log(f"Saved {written.name} ({PRODUCT_SUFFIX['apply_orbit']}).")

    def apply_deburst(self, params: dict) -> None:
        from ai_insar.pipeline.tops_ops import deburst_window

        grid = self.workspace.reference_grid
        if grid is None:
            raise RuntimeError(_DEBURST_BLOCK)
        result = deburst_window(grid.data, grid.model, grid.row, grid.col)
        self.workspace.reference_grid = result
        if self.workspace.stack is not None and self.workspace.stack.data.shape[0] == 1:
            self.workspace.stack.data = result.data[None, ...]
        self.workspace.clear_products()
        self._log(
            f"TOPS Deburst kept {result.data.shape[0]} lines starting at row {result.row}. "
            "The overlap was cut at the mid azimuth time."
        )
        del params

    def apply_backgeocoding(self, params: dict) -> None:
        from ai_insar.pipeline.tops_ops import RadarGrid, backgeocode

        early = self.workspace.reference_grid
        late = self.workspace.secondary_grid
        early_orbit = self.workspace.precise_orbit
        late_orbit = self.workspace.secondary_orbit
        if early is None or late is None or early_orbit is None or late_orbit is None:
            raise RuntimeError(_BACKGEO_BLOCK)
        dem = self._elevation_model(str(params.get("dem") or ""), str(params.get("dem_path") or ""), early, early_orbit)
        resampled = backgeocode(
            early,
            late,
            reference_orbit=early_orbit,
            secondary_orbit=late_orbit,
            dem=dem,
            reramp=not bool(params.get("disable_reramp")),
            resampling=str(params.get("resampling") or "BILINEAR_INTERPOLATION"),
            max_solves=4096,
            max_residual_px=0.05,
        )
        self._log(
            f"Back-geocoding map stride {backgeocode.last_stride}, "
            f"check residual {backgeocode.last_residual_px:.4f} pixels."
        )
        self.workspace.secondary_grid = RadarGrid(early.row, early.col, resampled, early.model)
        self._log(
            f"Back Geocoding resampled the secondary onto the reference grid {resampled.shape}. "
            "Deramp, range-Doppler, bilinear"
            + (" , reramp." if not params.get("disable_reramp") else ". Reramp disabled.")
        )

    def apply_esd(self, params: dict) -> None:
        from ai_insar.pipeline.tops_ops import (
            burst_overlap_rows,
            enhanced_spectral_diversity,
            overlap_doppler_separation_hz,
        )

        early = self.workspace.reference_grid
        late = self.workspace.secondary_grid
        if early is None or late is None or early.shape != late.shape or early.row != late.row:
            raise RuntimeError(_ESD_BLOCK)
        estimator = str(params.get("estimator") or "Periodogram")
        if params.get("user_azimuth"):
            from ai_insar.pipeline.tops_ops import RadarGrid, apply_azimuth_pixel_shift

            pixels = float(params["azimuth_shift"])
            shifted = apply_azimuth_pixel_shift(late.data, pixels, early.model.azimuth_time_interval_s)
            self.workspace.secondary_grid = RadarGrid(late.row, late.col, shifted, late.model)
            self.workspace.esd_azimuth_shift_s = pixels * early.model.azimuth_time_interval_s
            self._log(
                f"ESD applied the supplied azimuth shift {pixels:.4f} lines "
                f"({self.workspace.esd_azimuth_shift_s:.6e} s) to the secondary."
            )
            return
        model = early.model
        for index in range(len(model.bursts) - 1):
            (start_a, stop_a), (start_b, stop_b) = burst_overlap_rows(model, index)
            if not _covers(early, start_a, stop_a) or not _covers(early, start_b, stop_b):
                continue
            separation = overlap_doppler_separation_hz(model, index)
            shift, coherence = enhanced_spectral_diversity(
                _slice(early, start_a, stop_a),
                _slice(early, start_b, stop_b),
                _slice(late, start_a, stop_a),
                _slice(late, start_b, stop_b),
                separation_hz=separation,
                estimator=estimator,
            )
            self.workspace.esd_azimuth_shift_s = shift
            lines = shift / model.azimuth_time_interval_s
            self._log(
                f"ESD {estimator} burst {index + 1}/{index + 2}: azimuth shift {shift:.6e} s "
                f"({lines:.4f} lines), coherence {coherence:.3f}, spectral separation {separation:.1f} Hz."
            )
            return
        raise RuntimeError(
            "The loaded windows do not cover a burst overlap, so ESD was not run."
        )

    def apply_phase_to_displacement(self, params: dict) -> None:
        if self.workspace.unwrapped is None:
            raise RuntimeError(
                "Phase to Displacement needs unwrapped phase in radians. "
                "A raw SAFE or a wrapped interferogram is not that input."
            )
        wavelength = C_BAND_WAVELENGTH_M
        if self.workspace.stack is not None:
            wavelength = float(self.workspace.stack.wavelength_m)
        displacement = los_from_phase(self.workspace.unwrapped, wavelength)
        self.workspace.los_m = displacement
        self._preferred_layer = "LOS displacement"
        self._log(
            f"Phase to Displacement: d = -(λ/4π) Δφ, λ={wavelength:.6f} m, "
            f"shape {displacement.shape}, metres. Positive is toward the radar."
        )
        request = params["request"]
        self._save_request(request, displacement, PRODUCT_SUFFIX["displacement"])

    def apply_terrain_correction(self, params: dict) -> None:
        from ai_insar.pipeline.tops_ops import ellipsoid_geocode, terrain_correct

        grid = self.workspace.reference_grid
        orbit = self.workspace.precise_orbit
        if grid is None or orbit is None:
            raise RuntimeError("Range-Doppler geocoding needs a split swath and an orbit.")
        projection = str(params.get("projection") or "WGS84(DD)")
        mode = "utm" if "utm" in projection.lower() else "geographic"
        if mode == "geographic" and projection not in {"", "WGS84(DD)", "Geographic Lat/Lon (WGS 84)"}:
            if "geographic" not in projection.lower() and "wgs84" not in projection.lower():
                raise RuntimeError(
                    f"Map projection {projection} is not implemented. "
                    "Geographic WGS84 and automatic UTM are written."
                )
        resampling = str(params.get("image_resampling") or "BILINEAR_INTERPOLATION")
        dem_name = str(params.get("dem") or "")
        out_rows = max(8, min(48, max(grid.shape[0] // 4, 8)))
        out_cols = max(8, min(48, max(grid.shape[1] // 4, 8)))
        if "ellipsoid" in dem_name.lower():
            image, bounds = ellipsoid_geocode(
                grid,
                orbit,
                height_m=0.0,
                out_rows=out_rows,
                out_cols=out_cols,
                resampling=resampling,
            )
            self._log(
                f"Ellipsoid geocoding at height 0, not terrain correction. "
                f"Shape {image.shape}. Bounds {bounds}."
            )
            heights = None
        else:
            dem = self._elevation_model(dem_name, str(params.get("dem_path") or ""), grid, orbit)
            self.workspace.elevation = dem
            image, bounds, heights = terrain_correct(
                grid,
                orbit,
                dem,
                out_rows=out_rows,
                out_cols=out_cols,
                resampling=resampling,
                projection=mode,
            )
            finite = heights[np.isfinite(heights)]
            missing = int(heights.size - finite.size)
            spread = float(np.std(finite)) if finite.size else float("nan")
            self._log(
                f"Range-Doppler terrain correction sampled {dem.provenance}. "
                f"Ellipsoidal height std {spread:.2f} m, no-data pixels {missing}. "
                f"Orthometric datum {dem.orthometric_datum}."
            )
        stored = image if params.get("output_complex") else np.abs(image).astype(np.float32)
        self.workspace.geocoded = stored
        self.workspace.geocode_bounds = bounds
        west, south, east, north = bounds
        self._log(f"Geocoded {stored.shape}. West {west:.5f}, south {south:.5f}, east {east:.5f}, north {north:.5f}.")
        request = params.get("request")
        if request is not None and getattr(request, "save", False):
            self._write_product(request.directory, request.target or "terrain", request.format_name, stored, bounds)

    def _elevation_model(self, dem_name: str, dem_path: str, grid, orbit):
        from pathlib import Path

        from ai_insar.pipeline.dem import (
            ensure_copernicus,
            ensure_geoid,
            ensure_srtm,
            read_copernicus_tif,
            read_srtm_hgt,
        )
        from ai_insar.pipeline.geometry import ecef_to_geodetic, ground_point_at_height

        cache = Path("/tmp/safe-real/dem")
        if not cache.is_dir():
            cache = Path.home() / ".cache" / "ai-insar" / "dem"
        if dem_name == "External DEM":
            if not dem_path:
                raise RuntimeError("External DEM is selected and no file was given.")
            path = Path(dem_path)
            if path.suffix.lower() in {".tif", ".tiff"}:
                return read_copernicus_tif(path, ensure_geoid(cache, "EGM2008"))
            if "hgt" in path.name.lower():
                return read_srtm_hgt(path, ensure_geoid(cache, "EGM96"))
            raise RuntimeError("External DEM must be a GeoTIFF or an SRTM .hgt file. A mean height is not sampled.")
        azimuth = grid.model.line_time(grid.row + grid.shape[0] // 2)
        slant = grid.model.slant_range_m(grid.col + grid.shape[1] // 2)
        ground = ground_point_at_height(orbit, azimuth, slant, 0.0)
        lat, lon, _height = ecef_to_geodetic(ground)
        lat_deg = float(np.rad2deg(lat))
        lon_deg = float(np.rad2deg(lon))
        if "SRTM" in dem_name:
            return ensure_srtm(lat_deg, lon_deg, cache, ensure_geoid(cache, "EGM96"))
        if "Copernicus" in dem_name:
            return ensure_copernicus(lat_deg, lon_deg, cache)
        raise RuntimeError(
            f"{dem_name} is not fetched. Choose Copernicus 30 m, SRTM 1Sec HGT, "
            "an external GeoTIFF or HGT, or WGS84 ellipsoid (not terrain correction)."
        )

    def _download_sentinel_orbit(self, params: dict, degree: int) -> bool:
        from pathlib import Path

        from ai_insar.pipeline.orbit_download import download_orbit
        from ai_insar.safe import resolve_safe

        source = str(params.get("source") or "")
        path = Path(source)
        if not path.exists():
            return False
        try:
            product = resolve_safe(path)
        except Exception:
            return False
        if not product.start_utc:
            raise RuntimeError(f"{product.product_id} has no start time, so an orbit file cannot be chosen.")
        when = np.datetime64(product.start_utc.replace("Z", ""))
        platform = product.product_id[:3].upper()
        kind = "RESORB" if "Restituted" in str(params.get("orbit_type") or "") else "POEORB"
        cache = Path("/tmp/safe-real/orbits")
        cache.mkdir(parents=True, exist_ok=True)
        orbit_path_pair = download_orbit(when, platform, cache, kind=kind)
        _orbit, orbit_path = orbit_path_pair
        copied = dict(params)
        copied["orbit_file"] = str(orbit_path)
        self._install_precise_orbit(str(orbit_path), degree, copied)
        return True

    def apply_subset(self, params: dict) -> None:
        from ai_insar.pipeline.tops_ops import RadarGrid

        row0 = int(params["row_start"])
        row1 = int(params["row_end"])
        col0 = int(params["col_start"])
        col1 = int(params["col_end"])
        if row0 >= row1 or col0 >= col1:
            raise RuntimeError("The subset is empty.")
        changed = False
        if self.workspace.stack is not None:
            self.workspace.stack.data = _crop(self.workspace.stack.data, row0, row1, col0, col1)
            changed = True
        for name in ("interferograms", "filtered", "coherence", "unwrapped", "los_m", "geocoded"):
            value = getattr(self.workspace, name)
            cropped = _crop(value, row0, row1, col0, col1)
            if cropped is not value:
                setattr(self.workspace, name, cropped)
                changed = True
        for name in ("reference_grid", "secondary_grid"):
            grid = getattr(self.workspace, name)
            if grid is None or grid.data.shape[0] < row1 or grid.data.shape[1] < col1:
                continue
            setattr(
                self.workspace,
                name,
                RadarGrid(
                    grid.row + row0,
                    grid.col + col0,
                    np.array(grid.data[row0:row1, col0:col1], copy=True),
                    grid.model,
                ),
            )
            changed = True
        if not changed:
            raise RuntimeError("Subset found no raster of that size.")
        self._log(
            f"Subset rows {row0}:{row1}, samples {col0}:{col1}. "
            "The copy is in memory. Source files were not modified."
        )
        self._preferred_layer = "SLC amplitude"

    def apply_save_product(self, params: dict) -> None:
        array, bounds = self._band_for_save(params["band"])
        self._write_product(params["directory"], params["target"], params["format_name"], array, bounds)

    def export_kmz(self) -> None:
        from pathlib import Path

        from ai_insar.pipeline.tops_ops import write_kmz

        image = self.workspace.geocoded
        bounds = self.workspace.geocode_bounds
        if image is None or bounds is None:
            if self.workspace.reference_grid is None or self.workspace.precise_orbit is None:
                raise RuntimeError(
                    "View as Google Earth KMZ needs a georeferenced grid. "
                    "Run Range-Doppler Terrain Correction first. Coordinates were not invented."
                )
            self.apply_terrain_correction(
                {
                    "request": None,
                    "dem": "WGS84 ellipsoid",
                    "dem_path": "",
                    "image_resampling": "BILINEAR_INTERPOLATION",
                    "mask_elevation": False,
                    "output_complex": False,
                    "projection": "WGS84(DD)",
                }
            )
            image = self.workspace.geocoded
            bounds = self.workspace.geocode_bounds
        directory = Path.cwd() / "products"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "view.kmz"
        write_kmz(path, image, bounds)
        self._log(f"Wrote {path} with a GroundOverlay. The source product was not modified.")

    def _band_for_save(self, band: str):
        if band == "SLC":
            if self.workspace.stack is None:
                raise RuntimeError("No SLC is loaded.")
            return self.workspace.stack.data, None
        if band == "Interferogram":
            if self.workspace.interferograms is None:
                raise RuntimeError("No interferogram is loaded.")
            return self.workspace.interferograms, None
        if band == "Unwrapped phase":
            if self.workspace.unwrapped is None:
                raise RuntimeError("No unwrapped phase is loaded.")
            return self.workspace.unwrapped, None
        if band == "Displacement":
            if self.workspace.los_m is None:
                raise RuntimeError("No displacement is loaded.")
            return self.workspace.los_m, self.workspace.geocode_bounds
        if band == "Geocoded":
            if self.workspace.geocoded is None:
                raise RuntimeError("No geocoded raster is loaded.")
            return self.workspace.geocoded, self.workspace.geocode_bounds
        raise RuntimeError(f"Unknown band {band}.")

    def _write_product(self, directory: str, target: str, format_name: str, array, bounds) -> None:
        from pathlib import Path

        from ai_insar.pipeline.tops_ops import write_geotiff, write_kmz

        if not str(directory).strip():
            raise RuntimeError("Choose a directory.")
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        stem = target.strip() or "product"
        if format_name == "NumPy (.npy)":
            path = folder / f"{stem}.npy"
            np.save(path, np.asarray(array))
        elif format_name == "GeoTIFF":
            path = folder / f"{stem}.tif"
            write_geotiff(path, np.asarray(array), bounds)
        elif format_name == "Google Earth KMZ":
            if bounds is None:
                raise RuntimeError("KMZ needs latitude/longitude bounds from terrain correction.")
            path = folder / f"{stem}.kmz"
            write_kmz(path, np.asarray(array), bounds)
        elif format_name == "BEAM-DIMAP":
            from ai_insar.pipeline.dimap import write_beam_dimap

            path = write_beam_dimap(folder, stem, np.asarray(array), bounds)
        else:
            raise RuntimeError(f"{format_name} is not written by this build.")
        self._log(f"Saved {path.name}. Source files were not modified.")

    def _spatial_shape(self) -> tuple[int, int]:
        if self.workspace.stack is not None:
            return int(self.workspace.stack.data.shape[-2]), int(self.workspace.stack.data.shape[-1])
        grid = self.workspace.reference_grid
        if grid is not None:
            return grid.shape
        return 0, 0

    def _set_coherence_windows(self, range_window: int, azimuth_window: int | None, *, square: bool) -> None:
        if range_window < 1:
            raise ValueError("coherence window must be at least 1")
        self.workspace.config.coherence_window = int(range_window)
        self.workspace.config.coherence_azimuth_window = None if square else azimuth_window
        if square:
            self._log(
                f"Square Pixel: coherence window {range_window}×{range_window} pixels "
                "(ground spacing is not applied)."
            )

    def _apply_topo_choice(
        self,
        choice: str,
        dem_path: str,
        *,
        baseline_m: float,
        slant_range_m: float,
        incidence_deg: float,
    ) -> None:
        if self.workspace.stack is None or self.workspace.interferograms is None:
            raise RuntimeError("Form interferograms before topographic phase removal.")
        dem = self._load_dem(choice, dem_path)
        self.workspace.dem_m = dem
        self.workspace.topo_baseline_perp_m = float(baseline_m)
        self.workspace.topo_slant_range_m = float(slant_range_m)
        self.workspace.topo_incidence_deg = float(incidence_deg)
        if abs(float(baseline_m)) < 1e-9:
            self._log(
                "Topographic phase is 0 because the perpendicular baseline is 0. "
                "Interferograms are unchanged. Suffix _DInSAR."
            )
            return
        phase = topographic_phase_rad(
            dem,
            wavelength_m=self.workspace.stack.wavelength_m,
            baseline_perp_m=float(baseline_m),
            slant_range_m=float(slant_range_m),
            incidence_rad=float(np.deg2rad(incidence_deg)),
        )
        self.workspace.interferograms = apply_topographic_phase(self.workspace.interferograms, phase)
        self.workspace.filtered = None
        self.workspace.unwrapped = None
        self.workspace.los_m = None
        self.workspace.velocity_m_yr = None
        self.workspace.mask = None
        self._log(
            "Topographic phase removed "
            f"(B_perp={baseline_m:.1f} m, slant range {slant_range_m:.0f} m, "
            f"incidence {incidence_deg:.1f}°). Suffix _DInSAR. "
            "Run Goldstein again before unwrapping."
        )

    def _load_dem(self, choice: str, dem_path: str) -> np.ndarray:
        if self.workspace.stack is None:
            raise RuntimeError("Load or generate an SLC stack first.")
        expected = self.workspace.stack.data.shape[1:]
        if choice.startswith("Workspace"):
            if self.workspace.dem_m is None:
                raise RuntimeError("No DEM is loaded. Choose an external DEM .npy.")
            dem = np.asarray(self.workspace.dem_m, dtype=np.float64)
        elif choice.startswith("External"):
            if not dem_path:
                raise RuntimeError("Choose an external DEM .npy file.")
            dem = np.load(dem_path)
            self._log(f"Loaded DEM from {dem_path}.")
        else:
            raise RuntimeError(
                f"{choice} is not available. Copernicus DEM auto-download is not in this build."
            )
        dem = np.asarray(dem, dtype=np.float64)
        if dem.shape != expected:
            raise ValueError(f"DEM shape {dem.shape} must match the SLC grid {expected}")
        return dem

    def _save_request(self, request, array, suffix: str) -> None:
        if not request.save or array is None:
            return
        if getattr(request, "format_name", "NumPy (.npy)") != "NumPy (.npy)":
            raise RuntimeError(f"{request.format_name} is not written by this build.")
        path = write_npy_product(request.directory, request.target, suffix, array)
        self._log(f"Saved {path.name} ({suffix}).")

    def run_geocode(self) -> None:
        """Range-Doppler terrain correction when a split window and orbit are loaded."""
        if self.workspace.reference_grid is None or self.workspace.precise_orbit is None:
            self._log(
                "GEOCODE: products stay on the radar grid. "
                "No split window and orbit are loaded, so no latitude/longitude grid was written."
            )
            return
        self.apply_terrain_correction(
            {
                "request": type("R", (), {"save": False, "open_in_app": True, "format_name": "NumPy (.npy)"})(),
                "dem": "WGS84 ellipsoid",
                "dem_path": "",
                "image_resampling": "BILINEAR_INTERPOLATION",
                "spacing_m": None,
                "spacing_deg": None,
                "mask_elevation": False,
                "output_complex": False,
                "projection": "WGS84(DD)",
            }
        )

    def run_interferograms(self) -> None:
        compute_interferograms(self.workspace, self._log)
        self._preferred_layer = "Interferogram phase"

    def run_filter(self) -> None:
        compute_filter(self.workspace, self._log)
        self._preferred_layer = "Filtered phase"

    def run_goldstein_preset(self, alpha: float) -> None:
        """Set the Goldstein exponent and run stage 3."""
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.workspace.config.goldstein_alpha = alpha
        self._log(f"Goldstein alpha set to {alpha:.2f}.")
        self.run_filter()

    def run_coherence(self) -> None:
        compute_coherence(self.workspace, self._log)
        self._preferred_layer = "Coherence"

    def run_coherence_preset(self, window: int) -> None:
        """Set the coherence boxcar and run stage 4."""
        window = int(window)
        if window < 1:
            raise ValueError("coherence window must be at least 1")
        self.workspace.config.coherence_window = window
        self._log(f"Coherence window set to {window}×{window}.")
        self.run_coherence()

    def run_unwrap(self) -> None:
        compute_unwrap(self.workspace, self._log)
        self._preferred_layer = "Unwrapped phase"

    def run_timeseries(self) -> None:
        compute_timeseries(self.workspace, self._log)
        self._preferred_layer = "LOS velocity"

    def set_timeseries_method(self, method: str) -> None:
        if method not in TIMESERIES_METHODS:
            raise ValueError(f"unknown time-series method {method}")
        if getattr(self, "_setting_method", False):
            return
        self._setting_method = True
        try:
            self._apply_timeseries_method(method)
        finally:
            self._setting_method = False

    def _apply_timeseries_method(self, method: str) -> None:
        self.workspace.config.timeseries_method = method
        label = TIMESERIES_LABELS[method]
        self.stage_buttons[5].setText(f"6    {label} / time series")
        self.stage_buttons[5].setToolTip(TIMESERIES_SUMMARIES[method])
        self.method_combo.blockSignals(True)
        index = self.method_combo.findData(method)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
        self.method_combo.blockSignals(False)
        self._log(f"Time-series method set to {label}. Run stage 6 to apply it.")

    def _build_filter_menus(self) -> None:
        """Top-level menus for the Goldstein filter and SLC-pair coherence."""
        goldstein = self.menuBar().addMenu(GOLDSTEIN_LABEL)
        goldstein.setObjectName("goldsteinMenu")
        run = goldstein.addAction(f"Run {GOLDSTEIN_LABEL}")
        run.setObjectName("runGoldstein")
        run.triggered.connect(self._click(self.open_goldstein_dialog, 2))
        goldstein.addSeparator()
        for key, label, alpha in GOLDSTEIN_PRESETS:
            item = goldstein.addAction(label)
            item.setObjectName(f"goldsteinPreset_{key}")
            item.triggered.connect(self._click(lambda value=alpha: self.run_goldstein_preset(value), 2))
        goldstein.addSeparator()
        about = goldstein.addAction("What this method does")
        about.triggered.connect(lambda _checked=False: self._show_text(GOLDSTEIN_LABEL, GOLDSTEIN_SUMMARY))

        coherence = self.menuBar().addMenu(COHERENCE_LABEL)
        coherence.setObjectName("coherenceMenu")
        run_coh = coherence.addAction(f"Run {COHERENCE_LABEL}")
        run_coh.setObjectName("runCoherence")
        run_coh.triggered.connect(self._click(self.open_coherence_dialog, 3))
        coherence.addSeparator()
        for key, label, window in COHERENCE_PRESETS:
            item = coherence.addAction(label)
            item.setObjectName(f"coherencePreset_{key}")
            item.triggered.connect(self._click(lambda value=window: self.run_coherence_preset(value), 3))
        coherence.addSeparator()
        about_coh = coherence.addAction("What this method does")
        about_coh.triggered.connect(lambda _checked=False: self._show_text(COHERENCE_LABEL, COHERENCE_SUMMARY))

    def _build_unwrap_menus(self) -> None:
        """One top-level menu per unwrap technique."""
        for key in UNWRAP_METHODS:
            label = UNWRAP_LABELS[key]
            menu = self.menuBar().addMenu(label)
            menu.setObjectName(f"unwrapMenu_{key}")
            run = menu.addAction(f"Run {label}")
            run.setObjectName(f"runUnwrap_{key}")
            run.triggered.connect(self._click(lambda name=key: self.run_named_unwrap(name), 4))
            about = menu.addAction("What this method does")
            about.triggered.connect(
                lambda _checked=False, name=key: self._show_text(UNWRAP_LABELS[name], UNWRAP_SUMMARIES[name])
            )

    def run_named_unwrap(self, method: str) -> None:
        if method not in UNWRAP_METHODS:
            raise ValueError(f"unknown unwrap method {method}")
        self.workspace.config.unwrap_method = method
        self.stage_buttons[4].setText(f"5    {UNWRAP_LABELS[method]}")
        self.stage_buttons[4].setToolTip(UNWRAP_SUMMARIES[method])
        self._log(f"Unwrap method set to {UNWRAP_LABELS[method]}.")
        self.run_unwrap()

    def _build_method_menus(self) -> None:
        """One top-level menu per time-series technique, including SBAS."""
        for key in TIMESERIES_METHODS:
            label = TIMESERIES_LABELS[key]
            menu = self.menuBar().addMenu(label)
            menu.setObjectName(f"methodMenu_{key}")
            run = menu.addAction(f"Run {label}")
            run.setObjectName(f"runMethod_{key}")
            run.triggered.connect(self._click(lambda name=key: self.run_named_timeseries(name), 5))
            about = menu.addAction("What this method does")
            about.triggered.connect(
                lambda _checked=False, name=key: self._show_text(
                    TIMESERIES_LABELS[name], TIMESERIES_SUMMARIES[name]
                )
            )

    def run_named_timeseries(self, method: str) -> None:
        """Select one technique and run stage 6 with it."""
        self.set_timeseries_method(method)
        self.run_timeseries()

    def _method_picker(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        caption = QtWidgets.QLabel("Stage 6 method")
        caption.setObjectName("subtle")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.setObjectName("timeseriesMethod")
        for key in TIMESERIES_METHODS:
            self.method_combo.addItem(TIMESERIES_LABELS[key], key)
        self.method_combo.currentIndexChanged.connect(self._method_combo_changed)
        layout.addWidget(caption)
        layout.addWidget(self.method_combo)
        return box

    def _method_combo_changed(self, index: int) -> None:
        method = self.method_combo.itemData(index)
        if method and method != self.workspace.config.timeseries_method:
            self.set_timeseries_method(str(method))

    def export_current_map(self) -> None:
        """Export the displacement date on screen, otherwise the velocity map."""
        layer = self.layer_combo.currentText()
        if layer == "LOS displacement" and self.workspace.los_m is not None:
            self.export_displacement_map()
            return
        if self.workspace.velocity_m_yr is not None:
            self.export_velocity_map()
            return
        if self.workspace.los_m is not None:
            self.export_displacement_map()
            return
        raise RuntimeError(
            "Run a time-series method first. Export writes the velocity map, or one displacement date."
        )

    def export_velocity_map(self) -> None:
        if self.workspace.velocity_m_yr is None:
            raise RuntimeError("Run a time-series method before exporting the velocity map.")
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export velocity map")
        if not directory:
            return
        self.export_velocity_to(directory)

    def export_velocity_to(self, directory: str) -> None:
        written = export_velocity_map(self.workspace.velocity_m_yr, directory)
        self._log_export("LOS velocity", written)
        self._preferred_layer = "LOS velocity"

    def export_displacement_map(self) -> None:
        if self.workspace.los_m is None:
            raise RuntimeError("Run a time-series method before exporting a displacement map.")
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export displacement map")
        if not directory:
            return
        self.export_displacement_to(directory)

    def export_displacement_to(self, directory: str, index: int | None = None) -> None:
        los = self.workspace.los_m
        if los is None:
            raise RuntimeError("Run a time-series method before exporting a displacement map.")
        if index is None:
            index = self._displacement_export_index()
        index = _clamp(index, los.shape[0])
        date = self._date_label(index)
        stem = f"los_m_{date}" if date else f"los_m_{index:03d}"
        vmin, vmax = symmetric_limits(los)
        written = export_displacement_map(los[index], directory, stem=stem, vmin=vmin, vmax=vmax)
        label = f"LOS displacement {date}" if date else f"LOS displacement index {index}"
        self._log_export(label, written)
        self._preferred_layer = "LOS displacement"

    def _displacement_export_index(self) -> int:
        los = self.workspace.los_m
        if los is None:
            return 0
        if self.layer_combo.currentText() == "LOS displacement":
            return _clamp(int(self.slider.value()), los.shape[0])
        return int(los.shape[0] - 1)

    def _date_label(self, index: int) -> str:
        stack = self.workspace.stack
        if stack is None or not (0 <= index < stack.n_dates):
            return ""
        return str(stack.dates[index])

    def _log_export(self, title: str, written: dict) -> None:
        names = ", ".join(str(path) for path in written.values())
        extra = ""
        if "tif" not in written:
            extra = " TIFF was skipped because tifffile is not installed."
        self._log(f"Exported {title}: {names}.{extra} Radar grid, not geocoded.")

    def run_detection(self) -> None:
        compute_detection(self.workspace, self._log)
        self._preferred_layer = "Velocity + mask"

    def run_all_stages(self) -> None:
        if self.workspace.stack is None:
            self.generate_sample()
        run_all(self.workspace, self._log)
        self._preferred_layer = "Velocity + mask"

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.workspace.config, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.set_timeseries_method(self.workspace.config.timeseries_method)

    def _select_layer(self, name: str, *, refresh: bool) -> None:
        index = self.layer_combo.findText(name)
        if index < 0:
            return
        self.layer_combo.blockSignals(True)
        self.layer_combo.setCurrentIndex(index)
        self.layer_combo.blockSignals(False)
        if refresh:
            self._refresh_view()

    def _refresh_view(self) -> None:
        layer = self.layer_combo.currentText()
        self._configure_slider(layer)
        index = int(self.slider.value())
        image, values, vmin, vmax, cmap, caption = self._compose(layer, index)
        self.viewer.set_raster(image, values)
        self.caption.setText(caption)
        if image is None or cmap is None:
            self.colorbar.set_bar(None, "", "", "")
        else:
            row = np.linspace(vmin, vmax, 256, dtype=np.float64)[None, :]
            self.colorbar.set_bar(
                colorize(row, vmin=vmin, vmax=vmax, cmap=cmap),
                _fmt(vmin),
                _fmt(vmax),
                _LAYER_UNITS.get(layer, ""),
            )

    def _configure_slider(self, layer: str) -> None:
        stack = self.workspace.stack
        count = 0
        kind = ""
        if layer == "SLC amplitude" and stack is not None:
            count = stack.n_dates
            kind = "date"
        elif layer == "LOS displacement" and self.workspace.los_m is not None:
            count = int(self.workspace.los_m.shape[0])
            kind = "date"
        elif layer in _PAIR_LAYERS and self.workspace.pairs:
            count = len(self.workspace.pairs)
            kind = "pair"
        self.slider.blockSignals(True)
        current = min(self.slider.value(), max(0, count - 1))
        self.slider.setEnabled(count > 1)
        self.slider.setRange(0, max(0, count - 1))
        self.slider.setValue(current)
        self.slider.blockSignals(False)
        if kind == "date" and stack is not None and count > 1:
            chosen = min(int(self.slider.value()), stack.n_dates - 1)
            self.slider_label.setText(f"Date {stack.dates[chosen]}")
        elif kind == "pair" and self.workspace.pairs and count > 1:
            earlier, later = self.workspace.pairs[min(int(self.slider.value()), count - 1)]
            self.slider_label.setText(f"Pair {earlier}→{later}")
        else:
            self.slider_label.setText("Index")

    def _compose(
        self, layer: str, index: int
    ) -> tuple[np.ndarray | None, np.ndarray | None, float, float, str | None, str]:
        stack = self.workspace.stack
        missing = "This layer is empty until the matching stage has been run."
        if layer == "SLC amplitude":
            if stack is None:
                return None, None, 0.0, 1.0, None, missing
            index = _clamp(index, stack.n_dates)
            field = np.abs(stack.data[index]).astype(np.float64)
            rgb, vmin, vmax = amplitude_rgb(field)
            caption = (
                f"SLC amplitude, date {stack.dates[index]}. "
                f"Shape {tuple(stack.data.shape)}, dtype {stack.data.dtype}."
            )
            return rgb, field, vmin, vmax, "gray", caption
        if layer == "Interferogram phase":
            return self._phase_layer(self.workspace.interferograms, index, "Wrapped interferogram phase")
        if layer == "Filtered phase":
            alpha = self.workspace.config.goldstein_alpha
            return self._phase_layer(
                self.workspace.filtered,
                index,
                f"Goldstein filtered phase (α={alpha:.2f})",
            )
        if layer == "Coherence":
            if self.workspace.coherence is None:
                return None, None, 0.0, 1.0, None, missing
            index = _clamp(index, self.workspace.coherence.shape[0])
            field = self.workspace.coherence[index].astype(np.float64)
            rgb = colorize(field, vmin=0.0, vmax=1.0, cmap="coherence")
            window = self.workspace.config.coherence_window
            pair = ""
            if self.workspace.pairs and index < len(self.workspace.pairs):
                earlier, later = self.workspace.pairs[index]
                pair = f" Dates {earlier} → {later}."
            caption = (
                f"Coherence of pair index {index}, {window}×{window} boxcar on the SLC pair "
                f"(not the filtered phase).{pair}"
            )
            return rgb, field, 0.0, 1.0, "coherence", caption
        if layer == "Unwrapped phase":
            if self.workspace.unwrapped is None:
                return None, None, 0.0, 1.0, None, missing
            index = _clamp(index, self.workspace.unwrapped.shape[0])
            field = self.workspace.unwrapped[index]
            vmin, vmax = _finite_limits(field)
            rgb = colorize(field, vmin=vmin, vmax=vmax, cmap="velocity")
            return (
                rgb,
                field,
                vmin,
                vmax,
                "velocity",
                f"Unwrapped phase of pair index {index}, radians, referenced to the image border.",
            )
        if layer == "LOS displacement":
            if self.workspace.los_m is None:
                return None, None, 0.0, 1.0, None, missing
            index = _clamp(index, self.workspace.los_m.shape[0])
            field = self.workspace.los_m[index]
            vmin, vmax = symmetric_limits(self.workspace.los_m)
            rgb = colorize(field, vmin=vmin, vmax=vmax, cmap="velocity")
            return (
                rgb,
                field,
                vmin,
                vmax,
                "velocity",
                "LOS displacement in meters. Blue is away from the radar, red is toward the radar. "
                "The first date is 0. Radar grid, not geocoded.",
            )
        if layer == "LOS velocity":
            if self.workspace.velocity_m_yr is None:
                return None, None, 0.0, 1.0, None, missing
            field = self.workspace.velocity_m_yr
            vmin, vmax = symmetric_limits(field)
            rgb = colorize(field, vmin=vmin, vmax=vmax, cmap="velocity")
            return (
                rgb,
                field,
                vmin,
                vmax,
                "velocity",
                "LOS velocity in m/yr. Blue is motion away from the radar (subsidence); red is motion toward the radar. "
                "Radar grid, not geocoded. File → Export velocity map writes npy, PNG, and TIFF.",
            )
        if layer == "Detection mask":
            if self.workspace.mask is None or stack is None:
                return None, None, 0.0, 1.0, None, missing
            field = self.workspace.mask.astype(np.float64)
            base, _vmin, _vmax = amplitude_rgb(np.abs(stack.data[0]))
            rgb = overlay_mask(base, self.workspace.mask)
            return (
                rgb,
                field,
                0.0,
                1.0,
                "mask",
                f"Displacement mask on the first-date amplitude. {int(self.workspace.mask.sum())} pixels.",
            )
        if layer == "Velocity + mask":
            if self.workspace.velocity_m_yr is None or self.workspace.mask is None:
                return None, None, 0.0, 1.0, None, missing
            field = self.workspace.velocity_m_yr
            vmin, vmax = symmetric_limits(field)
            base = colorize(field, vmin=vmin, vmax=vmax, cmap="velocity")
            rgb = overlay_mask(base, self.workspace.mask)
            return rgb, field, vmin, vmax, "velocity", "LOS velocity with the displacement mask drawn in orange."
        return None, None, 0.0, 1.0, None, missing

    def _phase_layer(
        self,
        cube: np.ndarray | None,
        index: int,
        title: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float, float, str | None, str]:
        if cube is None:
            return None, None, -np.pi, np.pi, None, "This layer is empty until the matching stage has been run."
        index = _clamp(index, cube.shape[0])
        field = np.angle(cube[index]).astype(np.float64)
        rgb = colorize(field, vmin=-np.pi, vmax=np.pi, cmap="phase")
        pair = ""
        if self.workspace.pairs and index < len(self.workspace.pairs):
            earlier, later = self.workspace.pairs[index]
            pair = f" Dates {earlier} → {later}, formed as later × conjugate(earlier)."
        return rgb, field, -np.pi, np.pi, "phase", f"{title}, radians.{pair}"

    def _on_pixel(self, row: int, col: int) -> None:
        values = self.viewer._values
        if values is None or getattr(values, "ndim", 0) != 2:
            return
        if not (0 <= row < values.shape[0] and 0 <= col < values.shape[1]):
            return
        value = values[row, col]
        text = f"row {row}   col {col}   value {float(value):.6g}"
        if self.workspace.los_m is not None:
            series = self.workspace.los_m[:, row, col]
            millimetres = ", ".join(f"{float(item) * 1000.0:.2f}" for item in series)
            text += f"    LOS mm by date: {millimetres}"
        if self.workspace.velocity_m_yr is not None:
            text += f"    velocity {float(self.workspace.velocity_m_yr[row, col]):.5f} m/yr"
        self.readout.setText(text)
        self.statusBar().showMessage(text)

    def _sync_stage_states(self) -> None:
        ready = (
            self.workspace.stack is not None,
            self.workspace.interferograms is not None,
            self.workspace.filtered is not None,
            self.workspace.coherence is not None,
            self.workspace.unwrapped is not None,
            self.workspace.velocity_m_yr is not None,
            self.workspace.mask is not None,
        )
        for index, done in enumerate(ready):
            self._paint_stage(index, "done" if done else "idle")

    def _paint_stage(self, index: int, state: str) -> None:
        button = self.stage_buttons[index]
        button.setProperty("stageState", state)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {message}")

    def _show_text(self, title: str, text: str) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(640, 460)
        layout = QtWidgets.QVBoxLayout(dialog)
        box = QtWidgets.QPlainTextEdit(text)
        box.setReadOnly(True)
        layout.addWidget(box)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About AI Insar",
            f"AI Insar {__version__}\n\n"
            "A small SLC-first InSAR workbench for interferograms, a Goldstein filter, "
            "SLC-pair coherence, phase unwrapping, network time series, and a displacement mask.\n\n"
            "Velocity and displacement maps export as .npy, a PNG preview, and a float TIFF "
            "on the radar grid (no CRS).\n\n"
            "Not a replacement for ISCE, MintPy, SNAP, or SNAPHU.\n"
            "MIT License.",
        )


def _clamp(index: int, count: int) -> int:
    if count <= 0:
        return 0
    return min(max(int(index), 0), count - 1)


def _finite_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _fmt(value: float) -> str:
    return f"{value:.4g}"

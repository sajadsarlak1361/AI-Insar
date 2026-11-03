"""Main window: numbered stages, a raster viewer, and a log."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ai_insar import __version__
from ai_insar.conventions import CONVENTIONS_TEXT, LIMITATIONS_TEXT
from ai_insar.gui.qt import Qt, QtGui, QtWidgets
from ai_insar.gui.render import amplitude_rgb, colorize, overlay_mask, symmetric_limits
from ai_insar.gui.fetch_dialog import FetchSentinelDialog
from ai_insar.gui.product_explorer import ProductExplorer, SlcCropDialog
from ai_insar.gui.settings import SettingsDialog
from ai_insar.gui.viewer import ColorBar, ImageView
from ai_insar.export_maps import export_displacement_map, export_velocity_map
from ai_insar.io_stack import load_slc, save_products
from ai_insar.pipeline.coherence import COHERENCE_LABEL, COHERENCE_PRESETS, COHERENCE_SUMMARY
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
from ai_insar.pipeline.filtering import GOLDSTEIN_LABEL, GOLDSTEIN_PRESETS, GOLDSTEIN_SUMMARY
from ai_insar.pipeline.synthetic import generate_demo_stack
from ai_insar.safe import format_product_status, preview_amplitude, resolve_safe, select_measurement

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
    "Generate / load sample SLC",
    "Interferogram",
    "Goldstein filter",
    "Coherence",
    "Unwrap",
    "SBAS / time series",
    "Displacement detection mask",
)

_STAGE_HELP = (
    "Synthetic C-band SLC stack (dates × height × width, complex64) with a known subsidence bowl. File → Load SLC stack opens a .npy instead.",
    "Form each pair as later × conjugate(earlier) on a short-baseline network.",
    "Goldstein spectral filter. The Goldstein menu sets alpha and runs this stage. Unwrapping uses this phase.",
    "Windowed coherence from the SLC pair. The Coherence menu sets the window. Filtering does not change this estimate.",
    "Quality-guided unwrap by default. Quality-guided and Fast unwrap are separate menus. A plane is then removed on the image border.",
    "SBAS displacement time series and a linear LOS velocity. Each other method has its own menu. Negative is away from the radar.",
    "Coherence-gated velocity anomaly. Not a trained model.",
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
QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #12171f;
    border: 1px solid #3c4a63;
    border-radius: 4px;
    padding: 4px;
}
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
        fetch = QtGui.QAction("Fetch Sentinel-1 SLC…", self)
        fetch.setObjectName("fetchSentinelAction")
        fetch.triggered.connect(self._click(self.fetch_sentinel, 0))
        file_menu.addAction(fetch)
        save = file_menu.addAction("Save products…")
        save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        save.triggered.connect(self._click(self.save_outputs, None))
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
        data_menu.addAction("Import local SAFE / TIFF…").triggered.connect(
            self._click(self.fetch_sentinel, 0)
        )

        pipeline = self.menuBar().addMenu("Pipeline")
        steps = (
            ("Interferogram", self.run_interferograms, 1),
            ("Goldstein filter", self.run_filter, 2),
            ("Coherence", self.run_coherence, 3),
            ("Unwrap", self.run_unwrap, 4),
            ("Run time series", self.run_timeseries, 5),
            ("Displacement detection mask", self.run_detection, 6),
        )
        for label, action, index in steps:
            item = pipeline.addAction(label)
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

        view = self.menuBar().addMenu("View")
        for name in _LAYERS:
            item = view.addAction(name)
            item.triggered.connect(lambda _checked=False, layer=name: self._select_layer(layer, refresh=True))
        view.addSeparator()
        view.addAction(self._export_velocity_action)
        view.addAction(self._export_displacement_action)

        help_menu = self.menuBar().addMenu("Help")
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
        self.workspace.clear_products()
        self._log(f"Loaded SLC {stack.data.shape} {stack.data.dtype} from {path}. {stack.description}")
        self._preferred_layer = "SLC amplitude"

    def open_safe_product(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Sentinel-1 SAFE zip",
            "",
            SAFE_OPEN_FILTER,
        )
        if not path:
            return
        self.open_safe_path(path)

    def open_safe_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Open SAFE folder")
        if not directory:
            return
        self.open_safe_path(directory)

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
        run.triggered.connect(self._click(self.run_filter, 2))
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
        run_coh.triggered.connect(self._click(self.run_coherence, 3))
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

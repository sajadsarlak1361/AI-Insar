"""SNAP-style product explorer for one open Sentinel-1 SAFE product."""

from __future__ import annotations

from ai_insar.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal
from ai_insar.gui.viewer import ImageView
from ai_insar.safe import (
    DEFAULT_POLARISATION,
    DEFAULT_SWATH,
    SafeProduct,
    measurement_is_present,
)

_ROLE = Qt.ItemDataRole.UserRole


class SlcCropDialog(QtWidgets.QDialog):
    """Crop window used when a SAFE band is added to the SLC stack."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        product_id: str,
        swath: str,
        pol: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add to SLC stack")
        self.setObjectName("slcCropDialog")
        self.resize(480, 220)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            f"Read a complex64 window from {product_id}, band {swath} {pol}. "
            "The same crop is one date in the SLC stack. No deburst or calibration is applied."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.row = self._pixel(0)
        self.col = self._pixel(0)
        self.height = self._pixel(256)
        self.width = self._pixel(256)
        form.addRow("Row", self.row)
        form.addRow("Column", self.col)
        form.addRow("Height", self.height)
        form.addRow("Width", self.width)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pixel(self, value: int) -> QtWidgets.QSpinBox:
        box = QtWidgets.QSpinBox()
        box.setRange(0, 200_000)
        box.setValue(value)
        return box


class ProductExplorer(QtWidgets.QDockWidget):
    """Metadata, swath bands, and a quick-look for the open SAFE product."""

    bandActivated = Signal(str, str)
    quicklookActivated = Signal()
    addToStackRequested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Product Explorer", parent)
        self.setObjectName("productExplorerDock")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._product: SafeProduct | None = None
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setObjectName("productExplorerTree")
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.tree, stretch=1)
        self.add_button = QtWidgets.QPushButton("Add to SLC stack (crop…)")
        self.add_button.setObjectName("addSafeToStackButton")
        self.add_button.clicked.connect(lambda _checked=False: self.addToStackRequested.emit())
        layout.addWidget(self.add_button)
        self.preview_caption = QtWidgets.QLabel("Double-click a band to preview amplitude.")
        self.preview_caption.setObjectName("productPreviewCaption")
        self.preview_caption.setWordWrap(True)
        layout.addWidget(self.preview_caption)
        self.preview = ImageView(empty_text="No amplitude preview yet.")
        self.preview.setObjectName("productPreview")
        self.preview.setMinimumSize(220, 140)
        layout.addWidget(self.preview, stretch=1)
        self.setWidget(body)
        self._products: list[SafeProduct] = []
        self.set_product(None)

    def add_dimap(self, title: str, details: list[tuple[str, str]], bands: list[str]) -> None:
        """Show a BEAM-DIMAP opened from disk. Existing SAFE rows stay."""
        if (
            self.tree.topLevelItemCount() == 1
            and self.tree.topLevelItem(0).text(0) == "No Sentinel-1 product open"
        ):
            self.tree.clear()
        for index in range(self.tree.topLevelItemCount()):
            if self.tree.topLevelItem(index).text(0) == title:
                self.tree.scrollToItem(self.tree.topLevelItem(index))
                return
        root = QtWidgets.QTreeWidgetItem([title])
        root.setData(0, _ROLE, ("dimap", title))
        meta = QtWidgets.QTreeWidgetItem(["Metadata"])
        for label, value in details:
            meta.addChild(QtWidgets.QTreeWidgetItem([f"{label}: {value}"]))
        root.addChild(meta)
        band_node = QtWidgets.QTreeWidgetItem(["Bands"])
        for name in bands:
            item = QtWidgets.QTreeWidgetItem([name])
            item.setData(0, _ROLE, ("dimap-band", name))
            band_node.addChild(item)
        root.addChild(band_node)
        self.tree.addTopLevelItem(root)
        self.tree.scrollToItem(root)
        self.preview_caption.setText(title)

    def set_product(self, product: SafeProduct | None) -> None:
        self.set_products([] if product is None else [product], reference_id=None if product is None else product.product_id)

    def set_products(
        self,
        products: list[SafeProduct],
        *,
        reference_id: str | None = None,
        secondary_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._products = list(products)
        self.tree.clear()
        self.preview.set_raster(None, None)
        if not products:
            self._product = None
            self.preview_caption.setText("Open a Sentinel-1 SAFE product to inspect it.")
            placeholder = QtWidgets.QTreeWidgetItem(["No Sentinel-1 product open"])
            self.tree.addTopLevelItem(placeholder)
            return
        secondary = set(secondary_ids or ())
        self._product = next((item for item in products if item.product_id == reference_id), products[0])
        self.preview_caption.setText("Double-click a band to preview amplitude.")
        focus = None
        for product in products:
            title = _product_title(product, reference_id=reference_id, secondary_ids=secondary)
            root = QtWidgets.QTreeWidgetItem([title])
            root.setData(0, _ROLE, ("product", product.product_id))
            self._append_product(root, product)
            self.tree.addTopLevelItem(root)
            if product.product_id == reference_id or focus is None:
                focus = root
        # Keep every product name on screen. Expanding both SAFE trees hides the second date.
        self.tree.collapseAll()
        if focus is not None:
            self.tree.scrollToItem(focus)

    def product(self) -> SafeProduct | None:
        return self._product

    def _append_product(self, root: QtWidgets.QTreeWidgetItem, product: SafeProduct) -> None:
        meta = QtWidgets.QTreeWidgetItem(["Metadata"])
        fields = (
            ("Platform", product.platform),
            ("Mode", product.mode),
            ("Product type", product.product_type),
            ("Polarisation", "+".join(product.polarisations)),
            ("Swaths", ", ".join(product.swaths)),
            ("Start", product.start_utc),
            ("Stop", product.stop_utc),
            ("Absolute orbit", "" if product.absolute_orbit is None else str(product.absolute_orbit)),
            ("Relative orbit", "" if product.relative_orbit is None else str(product.relative_orbit)),
            ("Pass", product.orbit_pass),
            ("Datatake", product.datatake_id or ""),
            ("Footprint", f"{len(product.footprint)} points"),
        )
        for label, value in fields:
            meta.addChild(QtWidgets.QTreeWidgetItem([f"{label}: {value}"]))
        root.addChild(meta)
        bands = QtWidgets.QTreeWidgetItem(["Bands"])
        swaths = list(product.swaths)
        for measurement in product.measurements:
            if measurement.swath and measurement.swath not in swaths:
                swaths.append(measurement.swath)
        pols = list(product.polarisations)
        for measurement in product.measurements:
            if measurement.polarisation and measurement.polarisation not in pols:
                pols.append(measurement.polarisation)
        for swath in swaths:
            group = QtWidgets.QTreeWidgetItem([swath])
            used = False
            for pol in pols:
                measurement = _find_measurement(product, swath, pol)
                if measurement is None:
                    continue
                used = True
                present = measurement_is_present(product, measurement)
                suffix = "" if present else " (not in product)"
                size = ""
                if measurement.lines and measurement.samples:
                    size = f" ({measurement.lines}×{measurement.samples})"
                kind = "band" if present else "missing"
                for prefix in (f"i_{swath}_{pol}", f"q_{swath}_{pol}", f"Intensity_{swath}_{pol}"):
                    item = QtWidgets.QTreeWidgetItem([prefix + size + suffix])
                    item.setData(0, _ROLE, (kind, swath, pol, product.product_id))
                    if not present:
                        item.setForeground(0, QtGui.QBrush(QtGui.QColor("#8894a8")))
                    group.addChild(item)
            if used:
                bands.addChild(group)
        root.addChild(bands)
        if product.quicklook_href:
            quick = QtWidgets.QTreeWidgetItem(["Quicklook"])
            quick.setData(0, _ROLE, ("quicklook", product.product_id))
            root.addChild(quick)

    def show_preview(self, amplitude, caption: str) -> None:
        from ai_insar.gui.render import amplitude_rgb

        rgb, _vmin, _vmax = amplitude_rgb(amplitude)
        self.preview.set_raster(rgb, amplitude)
        self.preview_caption.setText(caption)

    def selected_swath_pol(self) -> tuple[str, str]:
        item = self.tree.currentItem()
        data = item.data(0, _ROLE) if item is not None else None
        if data and data[0] in {"band", "missing"}:
            self._activate(str(data[3]) if len(data) > 3 else "")
            return str(data[1]), str(data[2])
        return DEFAULT_SWATH, DEFAULT_POLARISATION

    def _activate(self, product_id: str) -> None:
        if not product_id:
            return
        for product in self._products:
            if product.product_id == product_id:
                self._product = product
                return

    def _on_double_click(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, _ROLE)
        if not data:
            return
        if len(data) > 3:
            self._activate(str(data[3]))
        elif len(data) > 1 and data[0] in {"quicklook", "product"}:
            self._activate(str(data[1]))
        if data[0] == "band":
            self.bandActivated.emit(str(data[1]), str(data[2]))
        elif data[0] == "quicklook":
            self.quicklookActivated.emit()
        elif data[0] == "missing":
            self.preview_caption.setText(
                f"{data[1]} {data[2]} is in the manifest, but the measurement TIFF is not in this product."
            )

    def _on_context_menu(self, position: QtCore.QPoint) -> None:
        item = self.tree.itemAt(position)
        if item is not None:
            self.tree.setCurrentItem(item)
        menu = QtWidgets.QMenu(self)
        action = menu.addAction("Add to SLC stack (crop…)")
        action.triggered.connect(lambda _checked=False: self.addToStackRequested.emit())
        menu.exec(self.tree.viewport().mapToGlobal(position))


def _product_title(product: SafeProduct, *, reference_id: str | None, secondary_ids: set[str]) -> str:
    """Date plus product id, with the pair role the explorer must keep visible."""
    when = (product.start_utc or "")[:10]
    if reference_id and product.product_id == reference_id:
        role = " (reference)"
    elif product.product_id in secondary_ids:
        role = " (secondary)"
    else:
        role = ""
    return f"{when} {product.product_id}{role}".strip()


def _find_measurement(product: SafeProduct, swath: str, pol: str):
    for measurement in product.measurements:
        if measurement.swath.upper() == swath.upper() and measurement.polarisation.upper() == pol.upper():
            return measurement
    return None

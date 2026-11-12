"""SNAP ProductSet-Reader list: several SAFE folders or zips, with metadata."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ai_insar.gui.qt import Qt, QtWidgets, Signal
from ai_insar.safe import resolve_safe
from ai_insar.safe_stack import is_safe_input

_COLUMNS = ("File Name", "Type", "Acquisition", "Track", "Orbit")
OpenedPaths = Callable[[], Sequence[Path | str]]


@dataclass
class ProductRow:
    """One ProductSet row. Blank strings mean the manifest did not provide a value."""

    path: Path
    file_name: str
    product_type: str
    acquisition: str
    track: str
    orbit: str


def describe_safe_row(path: Path | str) -> ProductRow:
    """Read Type, Acquisition, Track, and Orbit. Leave a field blank if it is missing."""
    source = Path(path)
    blank = ProductRow(source, source.name, "", "", "", "")
    try:
        product = resolve_safe(source)
    except Exception:
        return blank
    return ProductRow(
        path=source,
        file_name=source.name,
        product_type=product.product_type or "",
        acquisition=product.start_utc or "",
        track="" if product.relative_orbit is None else str(product.relative_orbit),
        orbit="" if product.absolute_orbit is None else str(product.absolute_orbit),
    )


def format_product_list(rows: Sequence[ProductRow]) -> str:
    """Multi-line list: count, then File Name | Type | Acquisition | Track | Orbit."""
    header = f"{len(rows)} Products"
    lines = [header, "File Name | Type | Acquisition | Track | Orbit"]
    for row in rows:
        lines.append(
            " | ".join((row.file_name, row.product_type, row.acquisition, row.track, row.orbit))
        )
    return "\n".join(lines)


def choose_safe_products(parent: QtWidgets.QWidget | None, *, start: str | None = None) -> list[str]:
    """Open the multi-folder / zip chooser used by Add and File → Open."""
    from ai_insar.gui.safe_chooser import choose_safe_products as _choose

    return _choose(parent, start=start)


class ProductSetWidget(QtWidgets.QWidget):
    """Table plus the nine SNAP ProductSet buttons and an ``N Products`` counter."""

    changed = Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        opened_paths: OpenedPaths | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("productSetWidget")
        self._opened_paths = opened_paths
        self._rows: list[ProductRow] = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        body = QtWidgets.QHBoxLayout()
        self.table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self.table.setObjectName("productSetTable")
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        header = self.table.horizontalHeaderItem(3)
        if header is not None:
            header.setToolTip("Relative orbit (track)")
        orbit_header = self.table.horizontalHeaderItem(4)
        if orbit_header is not None:
            orbit_header.setToolTip("Absolute orbit")
        body.addWidget(self.table, stretch=1)

        buttons = QtWidgets.QVBoxLayout()
        self.add_button = self._button("Add", "productSetAdd", self._add_clicked)
        self.add_button.setStatusTip(
            "Add SAFE folders or zip files. Several folders can be selected together."
        )
        self.add_opened_button = self._button("Add Opened", "productSetAddOpened", self._add_opened)
        self.remove_button = self._button("Remove", "productSetRemove", self.remove_selected)
        self.move_top_button = self._button("Move Top", "productSetMoveTop", lambda: self._move(extreme="top"))
        self.move_up_button = self._button("Move Up", "productSetMoveUp", lambda: self._move(delta=-1))
        self.move_down_button = self._button("Move Down", "productSetMoveDown", lambda: self._move(delta=1))
        self.move_bottom_button = self._button(
            "Move Bottom", "productSetMoveBottom", lambda: self._move(extreme="bottom")
        )
        self.refresh_button = self._button("Refresh", "productSetRefresh", self.refresh)
        self.clear_button = self._button("Clear", "productSetClear", self.clear)
        for button in (
            self.add_button,
            self.add_opened_button,
            self.remove_button,
            self.move_top_button,
            self.move_up_button,
            self.move_down_button,
            self.move_bottom_button,
            self.refresh_button,
            self.clear_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        body.addLayout(buttons)
        layout.addLayout(body)

        footer = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("productSetStatus")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, stretch=1)
        self.count_label = QtWidgets.QLabel("0 Products")
        self.count_label.setObjectName("productSetCount")
        footer.addWidget(self.count_label)
        layout.addLayout(footer)
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        self._sync_buttons()

    def _button(self, tip: str, object_name: str, slot) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(tip)
        button.setObjectName(object_name)
        button.setToolTip(tip)
        button.clicked.connect(slot)
        return button

    def rows(self) -> list[ProductRow]:
        return list(self._rows)

    def paths(self) -> list[Path]:
        return [row.path for row in self._rows]

    def summary(self) -> str:
        return format_product_list(self._rows)

    def add_paths(self, paths: Sequence[Path | str]) -> None:
        """Append products. The same path is not listed twice. Specs come from the manifest."""
        known = {row.path.resolve(strict=False) for row in self._rows}
        added = False
        for raw in paths:
            path = Path(raw)
            key = path.resolve(strict=False)
            if key in known:
                continue
            known.add(key)
            self._rows.append(describe_safe_row(path))
            added = True
        if added:
            self._rebuild()
            self.changed.emit()
        else:
            self._sync_buttons()

    def remove_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return
        del self._rows[row]
        self._rebuild()
        if self._rows:
            self.table.selectRow(min(row, len(self._rows) - 1))
        self.changed.emit()

    def refresh(self) -> None:
        """Re-read metadata. Fields that cannot be parsed stay blank."""
        self._rows = [describe_safe_row(row.path) for row in self._rows]
        selected = self.table.currentRow()
        self._rebuild()
        if 0 <= selected < len(self._rows):
            self.table.selectRow(selected)
        self.changed.emit()

    def clear(self) -> None:
        self._rows.clear()
        self._rebuild()
        self.changed.emit()

    def _add_clicked(self) -> None:
        self.add_paths(choose_safe_products(self))

    def _add_opened(self) -> None:
        if self._opened_paths is None:
            return
        self.add_paths(list(self._opened_paths()))

    def _move(self, *, delta: int = 0, extreme: str | None = None) -> None:
        row = self.table.currentRow()
        if row < 0 or not self._rows:
            return
        item = self._rows.pop(row)
        if extreme == "top":
            new = 0
        elif extreme == "bottom":
            new = len(self._rows)
        else:
            new = max(0, min(len(self._rows), row + delta))
        self._rows.insert(new, item)
        self._rebuild()
        self.table.selectRow(new)
        self.changed.emit()

    def _rebuild(self) -> None:
        self.table.setRowCount(len(self._rows))
        for index, row in enumerate(self._rows):
            values = (row.file_name, row.product_type, row.acquisition, row.track, row.orbit)
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if column == 0:
                    item.setToolTip(str(row.path))
                    item.setData(Qt.ItemDataRole.UserRole, str(row.path))
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        count = len(self._rows)
        selected = self.table.currentRow()
        has_selection = 0 <= selected < count
        self.remove_button.setEnabled(has_selection)
        self.move_top_button.setEnabled(has_selection and selected > 0)
        self.move_up_button.setEnabled(has_selection and selected > 0)
        self.move_down_button.setEnabled(has_selection and selected < count - 1)
        self.move_bottom_button.setEnabled(has_selection and selected < count - 1)
        self.refresh_button.setEnabled(count > 0)
        self.clear_button.setEnabled(count > 0)
        opened = list(self._opened_paths()) if self._opened_paths is not None else []
        self.add_opened_button.setEnabled(bool(opened))
        self.count_label.setText(f"{count} Products")
        if count == 0:
            self.status.setText("Error: [NodeId: ProductSet-Reader] Please add a source product")
        else:
            self.status.setText("")

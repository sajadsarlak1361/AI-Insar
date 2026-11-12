"""Multi-select chooser for SAFE folders and zip files.

Qt's stock directory dialog returns one folder. This view uses extended
selection so several ``.SAFE`` folders and ``.zip`` files can be chosen
together. A plain directory that holds more than one product contributes
every child product.
"""

from __future__ import annotations

from pathlib import Path

from ai_insar.gui.qt import QtCore, QtWidgets
from ai_insar.safe_stack import expand_safe_selection, is_safe_input


class SafeProductChooser(QtWidgets.QDialog):
    """Folder-and-zip browser with Ctrl/Shift multi-selection."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, start: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Product")
        self.setObjectName("safeProductChooser")
        self.resize(760, 520)
        self.start_directory = start or str(Path.cwd())
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Select several SAFE folders and .zip files. Ctrl or Shift adds to the selection. "
            "A folder that contains more than one SAFE product adds every product, not the first."
        )
        note.setWordWrap(True)
        note.setObjectName("safeProductChooserNote")
        layout.addWidget(note)

        self.model = QtWidgets.QFileSystemModel(self)
        self.model.setFilter(
            QtCore.QDir.Filter.AllDirs | QtCore.QDir.Filter.Files | QtCore.QDir.Filter.NoDotAndDotDot
        )
        self.model.setReadOnly(True)
        self.model.setRootPath(self.start_directory)
        self.view = QtWidgets.QTreeView()
        self.view.setObjectName("safeProductChooserView")
        self.view.setModel(self.model)
        self.view.setRootIndex(self.model.index(self.start_directory))
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        layout.addWidget(self.view, stretch=1)

        bar = QtWidgets.QHBoxLayout()
        up = QtWidgets.QPushButton("Up")
        up.setObjectName("safeProductChooserUp")
        up.clicked.connect(self._up)
        bar.addWidget(up)
        self.count = QtWidgets.QLabel("0 selected")
        self.count.setObjectName("safeProductChooserCount")
        bar.addWidget(self.count, stretch=1)
        layout.addLayout(bar)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("safeProductChooserStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        ok = QtWidgets.QPushButton("OK")
        ok.setObjectName("safeProductChooserOk")
        ok.clicked.connect(self._accept)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.setObjectName("safeProductChooserCancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self.view.selectionModel().selectionChanged.connect(lambda *_args: self._sync_count())
        self._sync_count()

    def selected_products(self) -> list[Path]:
        """SAFE products implied by the current selection, in view order."""
        selection = self.view.selectionModel()
        if selection is None:
            return []
        paths = [self.model.filePath(index) for index in selection.selectedRows(0)]
        paths = [path for path in paths if path]
        return expand_safe_selection(paths)

    def _sync_count(self) -> None:
        products = self.selected_products()
        self.count.setText(f"{len(products)} selected")

    def _up(self) -> None:
        parent = str(Path(self.start_directory).parent)
        if parent == self.start_directory:
            return
        self.start_directory = parent
        self.model.setRootPath(parent)
        self.view.setRootIndex(self.model.index(parent))
        self._sync_count()

    def _accept(self) -> None:
        products = self.selected_products()
        if not products:
            self.status.setText("Select one or more SAFE folders or .zip files.")
            return
        bad = [path.name for path in products if not is_safe_input(path)]
        if bad and len(bad) == len(products):
            self.status.setText("Not a SAFE folder or zip: " + ", ".join(bad))
            return
        self.accept()


def choose_safe_products(parent: QtWidgets.QWidget | None, *, start: str | None = None) -> list[str]:
    """Open the multi-folder chooser. Returns every selected SAFE product."""
    dialog = SafeProductChooser(parent, start=start)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return []
    return [str(path) for path in dialog.selected_products()]

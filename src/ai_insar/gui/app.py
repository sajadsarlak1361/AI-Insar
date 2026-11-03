"""QApplication bootstrap for the desktop workbench."""

from __future__ import annotations

import sys

from ai_insar.gui.main_window import MainWindow
from ai_insar.gui.qt import Qt, QtGui, QtWidgets


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("AI Insar")
    app.setStyle("Fusion")
    app.setWindowIcon(_icon())
    window = MainWindow()
    window.show()
    return int(app.exec())


def _icon() -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(64, 64)
    pixmap.fill(QtGui.QColor("#1b212c"))
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QtGui.QColor("#2d6cdf"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 12, 12)
    painter.setPen(QtGui.QColor("#f4f7fb"))
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(22)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "AI")
    painter.end()
    return QtGui.QIcon(pixmap)

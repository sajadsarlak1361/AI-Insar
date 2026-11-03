"""Qt binding import. PySide6 is preferred; PyQt6 is accepted as a fallback."""

from __future__ import annotations

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, Signal

    BINDING = "PySide6"
except ImportError:  # pragma: no cover - used only when PySide6 is absent
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt, pyqtSignal as Signal

    BINDING = "PyQt6"

__all__ = ["BINDING", "Qt", "QtCore", "QtGui", "QtWidgets", "Signal"]

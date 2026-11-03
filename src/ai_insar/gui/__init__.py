"""PySide6 (or PyQt6) desktop shell for the SLC pipeline."""

from __future__ import annotations

__all__ = ["main"]


def main() -> int:
    from ai_insar.gui.app import main as gui_main

    return gui_main()

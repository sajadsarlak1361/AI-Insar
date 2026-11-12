"""Save a Create Stack dialog with resampling set away from NONE."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ai_insar.gui.safe_stack_dialog import SafeStackDialog

ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / "tests" / "fixtures" / "pair"


def main() -> None:
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "create-stack.png")
    april = next(PAIR.glob("*20210401*.zip"))
    later = next(PAIR.glob("*20210413*.zip"))
    app = QApplication.instance() or QApplication([])
    dialog = SafeStackDialog(None, n_files=2, initial_paths=[april, later])
    dialog.tabs.setCurrentIndex(1)
    dialog.resampling.setCurrentText("BILINEAR_INTERPOLATION")
    dialog.initial_offset.setCurrentText("Orbit")
    dialog.output_extents.setCurrentText("Master")
    dialog.resize(900, 720)
    dialog.show()
    app.processEvents()
    pixmap = dialog.grab()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(destination)):
        raise SystemExit(f"could not write {destination}")
    print(destination)


if __name__ == "__main__":
    main()

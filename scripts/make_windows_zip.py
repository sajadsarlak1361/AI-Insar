#!/usr/bin/env python3
"""Write dist/ai-insar-windows.zip from a source checkout.

The archive is the project tree plus run_windows.bat. It does not vendor
third-party wheels; the batch file installs them on the Windows machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_insar.packaging import main

if __name__ == "__main__":
    main()

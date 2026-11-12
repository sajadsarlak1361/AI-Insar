"""AI Insar: an SLC-first desktop workbench for small InSAR stacks.

The processing chain reads a complex SLC stack, forms interferograms as
``later * conj(earlier)``, and estimates line-of-sight displacement.
"""

from __future__ import annotations

__version__ = "0.2.1rc1"

__all__ = ["__version__"]

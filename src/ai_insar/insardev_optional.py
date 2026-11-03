"""Optional InSARdev use. SAL modules are not copied into this MIT tree.

InSARdev/core ships three licenses:

- ``insardev`` is InSARdev-SAL-1.0 (source-available, not OSI open source).
  Section 4 allows non-commercial redistribution only when the copy stays
  under that same license, with the copyright notice and the license text.
  This repository is MIT, so those modules are not vendored. Install them
  yourself if you accept SAL. Funded academic, institutional, and
  professional use needs a subscription. See NOTICE.
- ``insardev_pygmtsar`` and ``insardev_toolkit`` are BSD-3-Clause. They are
  not vendored either: they scan GMTSAR-style burst folders and expect
  orbits, a DEM, pandas, geopandas, and xarray. This workbench reads SAFE
  zips with its own reader.
- InSARdev/edu is MIT. The separable unwrap and the complex product
  ``later * conj(earlier)`` follow those examples and are reimplemented
  here. The educational files are not copied. See NOTICE.

ESA SNAP is GPL and is not copied.
"""

from __future__ import annotations

INSARDEV_INSTALL_NOTE = (
    "The insardev package (InSARdev/core, InSARdev-SAL-1.0) is not included "
    "in AI Insar. SAL section 4 does not allow redistributing that source "
    "inside this MIT project. To use it yourself: pip install insardev, and "
    "read https://github.com/InSARdev/core/blob/main/insardev/LICENSE . "
    "Funded academic or professional use needs an active subscription. "
    "AI Insar still forms interferograms with its own NumPy/Numba routine."
)

BSD_INSTALL_NOTE = (
    "insardev_pygmtsar and insardev_toolkit are BSD-3-Clause and are not "
    "copied here. They preprocess burst directories with orbits and a DEM. "
    "Install them from https://github.com/InSARdev/core if you want that "
    "preprocessor. The SAFE zip stack in this workbench does not call them."
)


def insardev_status() -> dict[str, object]:
    """Report whether a user-installed ``insardev`` import succeeds.

    A successful import is the caller's own SAL relationship. This function
    does not run the SAL stack processor.
    """
    try:
        import insardev
    except ImportError:
        return {"available": False, "version": None, "note": INSARDEV_INSTALL_NOTE}
    version = getattr(insardev, "__version__", None)
    return {
        "available": True,
        "version": None if version is None else str(version),
        "note": (
            "insardev is installed. AI Insar does not call it for the SAFE "
            "zip stack or for later × conjugate(earlier). " + INSARDEV_INSTALL_NOTE
        ),
    }


def pygmtsar_status() -> dict[str, object]:
    """Report whether the BSD preprocessor imports. It is not required."""
    try:
        import insardev_pygmtsar
    except ImportError:
        return {"available": False, "note": BSD_INSTALL_NOTE}
    return {
        "available": True,
        "note": BSD_INSTALL_NOTE,
        "version": str(getattr(insardev_pygmtsar, "__version__", "installed")),
    }

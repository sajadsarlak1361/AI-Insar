"""Runtime packages the SAFE workflow imports.

``lxml`` is not one of them. Manifests and annotations are read with the
standard-library XML parser. ``asf_search`` is only the download client.
"""

from __future__ import annotations

# (import name, pip name)
REQUIRED: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("tifffile", "tifffile"),
    ("zarr", "zarr"),
    ("numba", "numba"),
    ("pyproj", "pyproj"),
    ("rasterio", "rasterio"),
    ("imagecodecs", "imagecodecs"),
)
OPTIONAL: tuple[tuple[str, str], ...] = (("asf_search", "asf_search"),)


def missing_modules(packages: tuple[tuple[str, str], ...] = REQUIRED) -> list[str]:
    """Pip names that cannot be imported."""
    missing: list[str] = []
    for module, pip_name in packages:
        try:
            __import__(module)
        except Exception:
            missing.append(pip_name)
    return missing


def dependency_report() -> str:
    """One block listing missing required packages and optional download support."""
    required = missing_modules(REQUIRED)
    optional = missing_modules(OPTIONAL)
    lines = ["AI Insar dependency check"]
    if not required:
        lines.append("Required packages import: numpy, scipy, tifffile, zarr, numba, pyproj, rasterio, imagecodecs.")
    else:
        lines.append("Missing packages required to open a SAFE and run the chain:")
        lines.extend(f"  - {name}" for name in required)
        lines.append('Install them with: pip install -e .')
        lines.append("On Windows, run_windows.bat does that install.")
    if optional:
        lines.append(
            "Optional and not required for a local SAFE: "
            + ", ".join(optional)
            + ' (ASF download only; pip install "ai-insar[slc]").'
        )
    lines.append("lxml is not required. Annotation XML uses the standard library.")
    return "\n".join(lines)


def require_runtime() -> None:
    """Raise SystemExit when a SAFE runtime package is missing."""
    missing = missing_modules()
    print(dependency_report())
    if missing:
        raise SystemExit(1)

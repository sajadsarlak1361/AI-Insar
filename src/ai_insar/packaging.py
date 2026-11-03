"""Build the Windows source zip documented in WINDOWS.md."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

_SKIP_DIRS = {".git", ".venv", "dist", "__pycache__", ".pytest_cache", "build"}
_TOP_FILES = (
    "pyproject.toml",
    "README.md",
    "WINDOWS.md",
    "LICENSE",
    "run_windows.bat",
    ".gitignore",
    ".gitattributes",
)
_TOP_DIRS = ("src", "tests", "scripts")


def project_root() -> Path:
    """Directory that contains ``pyproject.toml`` and ``run_windows.bat``."""
    candidates = [Path(__file__).resolve().parents[2], Path.cwd()]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "run_windows.bat").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the AI Insar project root (pyproject.toml and run_windows.bat)."
    )


def make_windows_zip(output: Path | None = None, root: Path | None = None) -> Path:
    """Zip the source tree so a Windows user can unzip and run ``run_windows.bat``.

    The archive is written to ``dist/ai-insar-windows.zip`` unless ``output``
    is set. Dependencies are not bundled; the batch file creates a venv and
    installs them.
    """
    source = Path(root) if root is not None else project_root()
    destination = Path(output) if output is not None else source / "dist" / "ai-insar-windows.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = list(_iter_project_files(source))
    if not files:
        raise FileNotFoundError(f"no project files found under {source}")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(source).as_posix()
            archive.write(path, arcname=f"ai-insar/{relative}")
    return destination


def _iter_project_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for name in _TOP_FILES:
        path = root / name
        if path.is_file():
            found.append(path)
    for folder in _TOP_DIRS:
        base = root / folder
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in _SKIP_DIRS and not name.endswith(".egg-info")
            ]
            for filename in filenames:
                if filename.endswith((".pyc", ".pyo")):
                    continue
                found.append(Path(dirpath) / filename)
    return found


def main() -> None:
    path = make_windows_zip()
    print(path)


if __name__ == "__main__":
    main()

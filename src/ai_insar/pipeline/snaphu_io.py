"""SNAPHU export, external execution, and import.

``snaphu`` is an optional external program (Curtis Chen and Howard Zebker).
This module does not vendor that program and does not replace it with the
in-tree fast unwrap. The files are little-endian float32 ENVI images plus a
``snaphu.conf`` the external binary can read.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

_STAT_MODES = ("TOPO", "DEFO", "SMOOTH", "NOSTATCOSTS")
_INIT_METHODS = ("MST", "MCF")


def export_snaphu(
    folder: Path | str,
    phase_rad: np.ndarray,
    coherence: np.ndarray,
    *,
    stat_mode: str = "DEFO",
    init_method: str = "MCF",
    tile_rows: int = 10,
    tile_columns: int = 10,
    processors: int = 4,
    row_overlap: int = 0,
    column_overlap: int = 0,
    tile_cost_threshold: int = 500,
    wavelength_m: float | None = None,
    pair_index: int = 0,
    pair_count: int = 1,
) -> Path:
    """Write wrapped phase, coherence, an ENVI header pair, and ``snaphu.conf``.

    ``phase_rad`` and ``coherence`` are 2-D and the same shape. Phase is
    wrapped radians. Only pair ``pair_index`` is written; the count is recorded
    so a multi-pair stack is not silently collapsed.
    """
    phase = np.asarray(phase_rad, dtype=np.float64)
    corr = np.asarray(coherence, dtype=np.float64)
    if phase.ndim != 2 or corr.shape != phase.shape:
        raise ValueError("phase and coherence must be 2-D and the same shape")
    if not np.isfinite(phase).all() or not np.isfinite(corr).all():
        raise ValueError("phase and coherence must be finite")
    mode = str(stat_mode).strip().upper()
    method = str(init_method).strip().upper()
    if mode not in _STAT_MODES:
        raise ValueError(f"statistical-cost mode must be one of {', '.join(_STAT_MODES)}")
    if method not in _INIT_METHODS:
        raise ValueError(f"initial method must be one of {', '.join(_INIT_METHODS)}")
    lines, samples = phase.shape
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    _write_envi(destination / "phase.img", destination / "phase.hdr", phase.astype("<f4"))
    _write_envi(destination / "corr.img", destination / "corr.hdr", corr.astype("<f4"))
    _write_envi_header(destination / "unwrap.hdr", samples=samples, lines=lines)
    conf = destination / "snaphu.conf"
    conf.write_text(
        "\n".join(
            (
                "# Configuration for the external snaphu program.",
                "# snaphu is not part of this MIT package and is not replaced here.",
                f"STATCOSTMODE {mode}",
                f"INITMETHOD {method}",
                "INFILEFORMAT FLOAT_DATA",
                "OUTFILEFORMAT FLOAT_DATA",
                "CORRFILEFORMAT FLOAT_DATA",
                "# phase.img, corr.img, and unwrap.img are passed on the command line.",
                "# snaphu 2.0.6 rejects those names in this file and on the command together.",
                f"LINELENGTH {samples}",
                f"NTILEROW {int(tile_rows)}",
                f"NTILECOL {int(tile_columns)}",
                f"NPROC {int(processors)}",
                f"ROWOVRLP {int(row_overlap)}",
                f"COLOVRLP {int(column_overlap)}",
                f"TILECOSTTHRESH {int(tile_cost_threshold)}",
                "",
            )
        ),
        encoding="utf-8",
    )
    meta = {
        "wrapped": True,
        "phase_unit": "radian",
        "dtype": "float32",
        "byte_order": "little",
        "interleave": "bsq",
        "lines": int(lines),
        "samples": int(samples),
        "pair_index": int(pair_index),
        "pair_count": int(pair_count),
        "wavelength_m": None if wavelength_m is None else float(wavelength_m),
        "stat_mode": mode,
        "init_method": method,
        "executable": None,
    }
    (destination / "snaphu_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return conf


def resolve_snaphu(install: Path | str) -> Path:
    """Return the snaphu executable named by a file or a directory. Do not search PATH."""
    location = Path(install)
    if location.is_file():
        if not os.access(location, os.X_OK):
            raise RuntimeError(f"{location} is not an executable snaphu file.")
        return location
    if location.is_dir():
        for name in ("snaphu", "snaphu.exe"):
            candidate = location / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    raise RuntimeError(
        f"No executable snaphu in {location}. "
        "The in-tree fast unwrap was not used."
    )


def run_snaphu(export_folder: Path | str, install: Path | str) -> subprocess.CompletedProcess[str]:
    """Run ``snaphu -f snaphu.conf`` in the export folder.

    A missing binary or a non-zero exit leaves the workspace unchanged.
    This function does not call the fast unwrap.
    """
    folder = Path(export_folder)
    conf = folder / "snaphu.conf"
    if not conf.is_file():
        raise RuntimeError(f"{conf} is missing. Export before unwrapping.")
    binary = resolve_snaphu(install)
    conf_text = conf.read_text(encoding="utf-8")
    samples = _conf_int(conf_text, "LINELENGTH")
    mode = _conf_word(conf_text, "STATCOSTMODE", "DEFO")
    method = _conf_word(conf_text, "INITMETHOD", "MCF")
    # LINELENGTH stays in the configuration. Passing it again makes snaphu 2.0.6
    # treat the width as a second input file.
    command = [str(binary), "-f", conf.name, "phase.img", "-c", "corr.img", "-o", "unwrap.img"]
    command.append({"DEFO": "-d", "SMOOTH": "-s", "TOPO": "-t"}.get(mode, "-d"))
    if mode == "NOSTATCOSTS":
        command.append("-n")
    command.append("--mcf" if method == "MCF" else "--mst")
    completed = subprocess.run(
        command,
        cwd=folder,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"snaphu exited {completed.returncode}. Fast unwrap was not used. {detail}"
        )
    unwrap = folder / "unwrap.img"
    if not unwrap.is_file():
        raise RuntimeError("snaphu exited 0 but did not write unwrap.img. Fast unwrap was not used.")
    return completed


def import_unwrapped(folder: Path | str) -> np.ndarray:
    """Read ``unwrap.img`` using its ENVI header and the export metadata.

    The header dimensions must match ``snaphu_meta.json``. The array is
    float32 radians as written; this function does not re-unwrap it.
    """
    directory = Path(folder)
    meta_path = directory / "snaphu_meta.json"
    header_path = directory / "unwrap.hdr"
    raw_path = directory / "unwrap.img"
    if not meta_path.is_file() or not header_path.is_file() or not raw_path.is_file():
        raise RuntimeError("Snaphu import needs snaphu_meta.json, unwrap.hdr, and unwrap.img.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("wrapped") is not True or meta.get("phase_unit") != "radian":
        raise RuntimeError("snaphu_meta.json is not a wrapped-radian export from this build.")
    samples, lines, data_type = _read_envi_header(header_path)
    if int(data_type) != 4:
        raise RuntimeError(f"unwrap.hdr data type {data_type} is not float32 (ENVI type 4).")
    if int(samples) != int(meta["samples"]) or int(lines) != int(meta["lines"]):
        raise RuntimeError(
            f"unwrap.hdr is {lines}×{samples} but the wrapped source metadata is "
            f"{meta['lines']}×{meta['samples']}."
        )
    expected = int(lines) * int(samples)
    raw = np.fromfile(raw_path, dtype="<f4")
    if raw.size != expected:
        raise RuntimeError(
            f"unwrap.img has {raw.size} samples; the header requires {expected}."
        )
    return raw.reshape(int(lines), int(samples)).astype(np.float32)


def _write_envi(raw_path: Path, header_path: Path, values: np.ndarray) -> None:
    array = np.asarray(values, dtype="<f4")
    array.tofile(raw_path)
    _write_envi_header(header_path, samples=int(array.shape[1]), lines=int(array.shape[0]))


def _write_envi_header(path: Path, *, samples: int, lines: int) -> None:
    path.write_text(
        "\n".join(
            (
                "ENVI",
                f"samples = {samples}",
                f"lines = {lines}",
                "bands = 1",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 4",
                "interleave = bsq",
                "byte order = 0",
                "",
            )
        ),
        encoding="utf-8",
    )


def _read_envi_header(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8")
    samples = _header_int(text, "samples")
    lines = _header_int(text, "lines")
    data_type = _header_int(text, "data type")
    return samples, lines, data_type


def _conf_word(text: str, key: str, default: str) -> str:
    value = _conf_value(text, key)
    return default if value is None else value


def _conf_int(text: str, key: str) -> int:
    value = _conf_value(text, key)
    if value is None:
        raise RuntimeError(f"snaphu.conf has no {key}.")
    return int(value)


def _conf_value(text: str, key: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0].upper() == key.upper() and len(parts) > 1:
            return parts[1]
    return None


def _header_int(text: str, name: str) -> int:
    match = re.search(rf"^{re.escape(name)}\s*=\s*(\d+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        raise RuntimeError(f"ENVI header has no {name}.")
    return int(match.group(1))

"""SNAP operator suffixes and one-line formulas for dialog help.

The strings quote the SNAP 14 Help forms (Interferogram, Coherence,
Topographic Phase, Goldstein, Apply Orbit, Create Stack, Phase to
Displacement). This module does not copy SNAP or GMTSAR source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Bytecode PRODUCT_SUFFIX values from SNAP 14 operators.
PRODUCT_SUFFIX = {
    "apply_orbit": "_Orb",
    "deburst": "_Deb",
    "stack": "_Stack",
    "interferogram": "_Ifg",
    "coherence": "_Coh",
    "topo": "_DInSAR",
    "goldstein": "_Flt",
    "displacement": "_Disp",
}

# One line each, for the dialog help strip.
OPERATOR_HELP = {
    "interferogram": (
        "I = s_ref · conj(s_sec); φ_ifg = arg(I). "
        "This build stores later × conjugate(earlier). Output suffix _Ifg."
    ),
    "coherence": (
        "γ = |Σ s_ref·conj(s_sec)| / sqrt(Σ|s_ref|² · Σ|s_sec|²) over the window. "
        "Output suffix _Coh."
    ),
    "topo": (
        "φ_topo ≈ −(4π/λ)·B_perp·h / (R sin θ); I_out = I · exp(−j φ_topo). "
        "Output suffix _DInSAR."
    ),
    "goldstein": (
        "Z_filt(u,v) = Z(u,v) · (|Z| ∗ K)^α, K = [1 2 3 2 1] (Goldstein FFT). "
        "Output suffix _Flt."
    ),
    "apply_orbit": (
        "Re-interpolate each orbit state vector with a polynomial and replace metadata. "
        "Output suffix _Orb."
    ),
    "stack": (
        "Collocate secondaries onto the reference. Resampling NONE copies the crop. "
        "A kernel applies the initial offset and the output extent. Output suffix _Stack."
    ),
    "displacement": (
        "d = −(λ/4π)·Δφ_d, with d the slant-range displacement in metres. "
        "Output suffix _Disp."
    ),
}


def suffixed_stem(target: str, suffix: str) -> str:
    """Return ``target`` with ``suffix`` once. Empty names become ``target``."""
    stem = (target or "").strip() or "target"
    if stem.endswith(suffix):
        return stem
    return f"{stem}{suffix}"


def write_npy_product(directory: Path | str, target: str, suffix: str, array: np.ndarray) -> Path:
    """Write one NumPy product whose filename ends with the SNAP suffix."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{suffixed_stem(target, suffix)}.npy"
    np.save(path, np.asarray(array))
    return path


def write_orbit_stub(directory: Path | str, target: str, payload: dict) -> Path:
    """Record an Apply Orbit request. State vectors are not changed."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{suffixed_stem(target, PRODUCT_SUFFIX['apply_orbit'])}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path

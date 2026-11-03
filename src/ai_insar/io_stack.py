"""Load and save SLC stacks and the products derived from them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ai_insar.conventions import C_BAND_WAVELENGTH_M

SCHEMA_VERSION = 1


def validate_slc(slc: np.ndarray) -> np.ndarray:
    """Return ``slc`` as a finite ``complex64`` array of shape ``(dates, height, width)``."""
    array = np.asarray(slc)
    if array.ndim != 3:
        raise ValueError(
            "SLC stack must have shape (dates, height, width); "
            f"got ndim={array.ndim} shape={array.shape}"
        )
    if not np.iscomplexobj(array):
        raise ValueError(
            "SLC stack must be complex (complex64). Real arrays are not SLC data; "
            f"got dtype={array.dtype}"
        )
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"SLC stack dimensions must be positive; got {array.shape}")
    out = np.asarray(array, dtype=np.complex64)
    if not np.isfinite(out.real).all() or not np.isfinite(out.imag).all():
        raise ValueError("SLC stack contains non-finite values")
    return out


def default_dates(n_dates: int, spacing_days: int = 12, start: str = "2020-01-08") -> np.ndarray:
    """Strictly increasing daily dates used when a sidecar does not list any."""
    if n_dates < 1:
        raise ValueError("n_dates must be positive")
    if spacing_days <= 0:
        raise ValueError("spacing_days must be positive")
    origin = np.datetime64(start, "D")
    return origin + np.arange(n_dates, dtype=np.int64) * np.timedelta64(int(spacing_days), "D")


def metadata_dict(stack: Any) -> dict[str, Any]:
    """JSON-ready description of an :class:`~ai_insar.pipeline.synthetic.SLCStack`."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "wavelength_m": float(stack.wavelength_m),
        "dates": [str(day) for day in np.asarray(stack.dates, dtype="datetime64[D]")],
        "sensor": stack.sensor,
        "description": stack.description,
        "shape": list(stack.data.shape),
        "dtype": "complex64",
        "interferogram": "later * conj(earlier)",
        "los_sign": "positive toward the radar",
    }
    return payload


def save_slc(stack: Any, path: Path) -> Path:
    """Write ``slc.npy``-style complex data and a JSON sidecar next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(stack.data, dtype=np.complex64))
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata_dict(stack), indent=2) + "\n", encoding="utf-8")
    return path


def load_slc(path: Path, meta: Path | None = None) -> Any:
    """Load a complex64 ``.npy`` SLC stack and an optional JSON sidecar.

    If the sidecar is missing, the stack is treated as C-band with a 12-day
    repeat. Dates are sorted so index increases with time. The returned object
    is an :class:`~ai_insar.pipeline.synthetic.SLCStack`.
    """
    # Local import avoids a cycle: synthetic.SLCStack validates via this module.
    from ai_insar.pipeline.synthetic import SLCStack

    path = Path(path)
    data = np.load(path)
    sidecar = Path(meta) if meta is not None else path.with_suffix(".json")
    notes: list[str] = []
    sensor = "unknown"
    description = ""
    wavelength = C_BAND_WAVELENGTH_M
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        wavelength = float(payload.get("wavelength_m", wavelength))
        sensor = str(payload.get("sensor", sensor))
        description = str(payload.get("description", ""))
        if payload.get("dates"):
            dates = np.array(payload["dates"], dtype="datetime64[D]")
        else:
            dates = default_dates(int(np.asarray(data).shape[0]))
            notes.append("sidecar has no dates; assumed 12-day spacing")
    else:
        if np.asarray(data).ndim != 3:
            # validate_slc raises the shape error with a clearer message
            dates = default_dates(1)
        else:
            dates = default_dates(int(np.asarray(data).shape[0]))
        notes.append(
            f"no sidecar {sidecar.name}; assumed C-band wavelength "
            f"{wavelength:.6f} m and 12-day spacing from 2020-01-08"
        )
        description = " ".join(notes)

    data = validate_slc(data)
    if dates.shape[0] != data.shape[0]:
        raise ValueError(
            f"sidecar lists {dates.shape[0]} dates but the SLC stack has {data.shape[0]}"
        )
    order = np.argsort(dates, kind="mergesort")
    if not np.all(order == np.arange(order.shape[0])):
        data = data[order]
        dates = dates[order]
        notes.append("dates were out of order and have been sorted ascending")
    if notes and not description:
        description = " ".join(notes)
    elif notes:
        description = (description + " " + " ".join(notes)).strip()
    return SLCStack(
        data=data,
        dates=dates,
        wavelength_m=wavelength,
        sensor=sensor,
        description=description,
    )


def save_products(workspace: Any, directory: Path) -> Path:
    """Write the arrays a finished workspace holds into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stack = workspace.stack
    if stack is None:
        raise ValueError("no SLC stack to describe")
    saved: dict[str, str] = {}

    def _store(name: str, array: np.ndarray | None) -> None:
        if array is None:
            return
        filename = f"{name}.npy"
        np.save(directory / filename, array)
        saved[name] = filename

    _store("velocity_m_yr", workspace.velocity_m_yr)
    _store("mask", None if workspace.mask is None else np.asarray(workspace.mask, dtype=np.uint8))
    _store("los_m", workspace.los_m)
    _store("mean_coherence", workspace.mean_coherence)
    _store("coherence", workspace.coherence)
    _store("unwrapped_phase", workspace.unwrapped)
    _store("interferogram", workspace.interferograms)
    _store("filtered_interferogram", workspace.filtered)
    if stack.truth_velocity_m_yr is not None:
        _store("truth_velocity_m_yr", stack.truth_velocity_m_yr)
    if stack.truth_mask is not None:
        _store("truth_mask", np.asarray(stack.truth_mask, dtype=np.uint8))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stack": metadata_dict(stack),
        "files": saved,
        "pairs": workspace.pairs or [],
    }
    if workspace.config is not None:
        payload["config"] = {
            "max_date_span": workspace.config.max_date_span,
            "goldstein_alpha": workspace.config.goldstein_alpha,
            "goldstein_window": workspace.config.goldstein_window,
            "coherence_window": workspace.config.coherence_window,
            "unwrap_min_coherence": workspace.config.unwrap_min_coherence,
            "detection_sigma": workspace.config.detection_sigma,
            "detection_peak_fraction": workspace.config.detection_peak_fraction,
            "detection_min_coherence": workspace.config.detection_min_coherence,
            "detection_min_pixels": workspace.config.detection_min_pixels,
            "timeseries_method": workspace.config.timeseries_method,
            "unwrap_method": workspace.config.unwrap_method,
        }
    (directory / "products.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return directory

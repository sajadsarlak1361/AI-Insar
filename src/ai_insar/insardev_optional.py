"""Optional InSARdev calls. SAL and BSD sources are not copied into this tree.

InSARdev/core ships three licenses:

- ``insardev`` is InSARdev-SAL-1.0 (source-available, not OSI open source).
  Section 4 allows non-commercial redistribution only when the copy stays
  under that same license, with the copyright notice and the license text.
  This repository is MIT, so those modules are not vendored. The extra
  ``pip install "ai-insar[insardev]"`` installs the upstream package for a
  caller who accepts SAL. Funded academic, institutional, and professional
  use needs a subscription. See NOTICE.
- ``insardev_pygmtsar`` and ``insardev_toolkit`` are BSD-3-Clause. They are
  not vendored either. Wrappers call ``PyGMTSAR.safeload`` and
  ``S1.align_ref`` / ``S1.align_rep`` when those packages are installed and
  the caller supplies a burst layout. A SAFE zip crop does not call them.
- InSARdev/edu is MIT. The separable unwrap and the complex product
  ``later * conj(earlier)`` follow those examples and are reimplemented
  here. The educational files are not copied. See NOTICE.

ESA SNAP is GPL and is not copied.

``Stack.pairs().interferogram()`` with no filter wavelength is the
single-look product ``reference * conjugate(repeat)``. This module passes
the later date as the reference so that product is ``later * conj(earlier)``.
The interferogram stage keeps the array only when it matches the
NumPy/Numba product. ``Stack.load`` is the geocoded-scene reader.
``Stack.align`` is burst-phase alignment on a stack that load already built.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_insar.conventions import C_BAND_WAVELENGTH_M

INSARDEV_INSTALL_NOTE = (
    "The insardev package (InSARdev/core, InSARdev-SAL-1.0) is not included "
    "in this MIT tree. Installing or using it means you accept that license, "
    "including a subscription for funded academic, institutional, and "
    "professional use. "
    'Preferred install: pip install "ai-insar[insardev]" '
    "(Python 3.12 or newer). The same upstream package is pip install insardev. "
    "Read https://github.com/InSARdev/core/blob/main/insardev/LICENSE and "
    "https://github.com/InSARdev/core/blob/main/insardev/SUBSCRIBE . "
    "If the package is missing or its single-look interferogram does not match "
    "later × conjugate(earlier), AI Insar keeps the NumPy/Numba product."
)

BSD_INSTALL_NOTE = (
    "insardev_pygmtsar and insardev_toolkit are BSD-3-Clause and are not "
    "copied here. PyGMTSAR.safeload translates SAFE bursts when you pass "
    "burst ids. S1.align_ref and S1.align_rep coregister a burst directory "
    "when insardev_pygmtsar is installed. They need orbits and a DEM. "
    "The SAFE zip crop in this workbench does not call them. "
    "Install the packages from https://github.com/InSARdev/core if you want "
    "that preprocessor."
)

_MODULES: dict[str, object] = {}
_LAST_INTERFEROGRAM_ERROR: str | None = None
_LAST_LOAD_ERROR: str | None = None


class _MemoryScene:
    """SLC holder used when xarray is absent. A real Stack rejects it."""

    def __init__(self, slc: np.ndarray, dates: np.ndarray) -> None:
        self.slc = np.asarray(slc)
        self.dates = dates
        self.data_vars = ("VV", "BPR")

    def __getitem__(self, key: str):
        if key == "VV":
            return self.slc
        if key == "BPR":
            return np.zeros(self.slc.shape[0], dtype=np.float64)
        raise KeyError(key)


@dataclass(frozen=True)
class StackAttempt:
    """One multi-scene read, either a SAFE zip crop or ``Stack.load``."""

    stack: object | None
    backend: str
    note: str


def clear_optional_cache() -> None:
    """Drop cached imports so tests can swap a fake module in."""
    _MODULES.clear()


def use_insardev_backend(explicit: str | None = None) -> bool:
    """Return whether this call should ask ``insardev`` for interferograms.

    ``numpy``, ``numba``, and ``mit`` force the in-tree product. ``auto``
    and ``insardev`` ask the optional package, then fall back if it is
    missing or its values differ. ``AI_INSAR_BACKEND`` is used when
    ``explicit`` is omitted.
    """
    import os

    if explicit is None:
        raw = os.environ.get("AI_INSAR_BACKEND", "auto")
    else:
        raw = explicit
    name = raw.strip().lower()
    if name in {"numpy", "numba", "mit"}:
        return False
    if name in {"auto", "insardev", ""}:
        return True
    raise ValueError(
        "backend must be 'auto', 'insardev', or 'numpy' "
        "(or set AI_INSAR_BACKEND to one of those)"
    )


def insardev_status() -> dict[str, object]:
    """Report whether a user-installed ``insardev`` import succeeds."""
    module = _load_module("insardev")
    apis = _stack_apis(module)
    if module is None:
        return {
            "available": False,
            "version": None,
            "note": INSARDEV_INSTALL_NOTE,
            "license": "InSARdev-SAL-1.0",
            "apis": apis,
            "interferogram_error": None,
        }
    version = getattr(module, "__version__", None)
    error = _LAST_INTERFEROGRAM_ERROR
    note = (
        f"insardev {version} is installed. Interferograms call "
        "Stack.pairs().interferogram() when that single-look product matches "
        "later × conjugate(earlier). A SAFE zip crop stays on the MIT reader. "
        + INSARDEV_INSTALL_NOTE
    )
    if error:
        note = f"{note} Last interferogram attempt: {error}"
    return {
        "available": True,
        "version": None if version is None else str(version),
        "note": note,
        "license": "InSARdev-SAL-1.0",
        "apis": apis,
        "interferogram_error": error,
    }


def pygmtsar_status() -> dict[str, object]:
    """Report whether the BSD preprocessor imports. It is not required."""
    module = _load_module("insardev_pygmtsar")
    if module is None:
        return {"available": False, "note": BSD_INSTALL_NOTE, "version": None}
    return {
        "available": True,
        "note": BSD_INSTALL_NOTE,
        "version": str(getattr(module, "__version__", "installed")),
    }


def toolkit_status() -> dict[str, object]:
    """Report whether ``insardev_toolkit`` imports. ``insardev`` depends on it."""
    module = _load_module("insardev_toolkit")
    if module is None:
        return {"available": False, "note": BSD_INSTALL_NOTE, "version": None}
    return {
        "available": True,
        "note": BSD_INSTALL_NOTE,
        "version": str(getattr(module, "__version__", "installed")),
    }


def try_insardev_interferogram(
    slc: np.ndarray,
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> np.ndarray | None:
    """Return a complex64 cube from ``insardev``, or ``None`` to keep NumPy/Numba.

    Failures stay inside this function. The MIT product is the result the
    pipeline keeps when this returns ``None``.
    """
    global _LAST_INTERFEROGRAM_ERROR
    module = _load_module("insardev")
    if module is None or not hasattr(module, "Stack"):
        return None
    try:
        cube = _call_interferogram(module, np.asarray(slc), list(pairs))
    except Exception as exc:
        _LAST_INTERFEROGRAM_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _LAST_INTERFEROGRAM_ERROR = None
    return cube


def stack_scenes(paths, **kwargs) -> StackAttempt:
    """Read many SAFE zips, or a geocoded scene list through ``Stack.load``.

    Zip paths use the MIT crop reader. Any other path is handed to
    ``insardev.Stack.load``. Keyword arguments are the zip reader's
    ``swath``, ``pol``, crop, and ``progress`` options.
    """
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("no Sentinel-1 SAFE zips were selected")
    if all(path.suffix.lower() == ".zip" for path in files):
        from ai_insar.safe_stack import stack_safe_zips

        stack = stack_safe_zips(files, **kwargs)
        return StackAttempt(
            stack=stack,
            backend="numpy",
            note=(
                "SAFE zip crop from the MIT reader. Shared row and column window. "
                "insardev.Stack.load reads geocoded zarr scenes, so it is not used here."
            ),
        )
    loaded = _try_load_scenes(files)
    if loaded is not None:
        return loaded
    if _load_module("insardev") is None:
        raise ImportError(INSARDEV_INSTALL_NOTE)
    detail = _LAST_LOAD_ERROR or "Stack.load did not return a complex cube"
    raise ValueError(detail)


def align_loaded_stack(stack, ref: int | str = 0) -> dict[str, object]:
    """Call ``Stack.align`` when the object has that method.

    This is burst-phase alignment on a geocoded stack. A NumPy crop has no
    burst overlaps, so callers pass the object ``Stack.load`` returned.
    """
    method = getattr(stack, "align", None)
    if not callable(method):
        return {
            "backend": "numpy",
            "applied": False,
            "result": None,
            "note": "This object has no Stack.align method. The array is unchanged.",
        }
    try:
        result = method(ref=ref)
    except Exception as exc:
        return {
            "backend": "numpy",
            "applied": False,
            "result": None,
            "note": (
                "Stack.align was not applied "
                f"({type(exc).__name__}: {exc}). The array is unchanged."
            ),
        }
    return {
        "backend": "insardev",
        "applied": True,
        "result": result,
        "note": "Called Stack.align for burst-phase alignment on a loaded stack.",
    }


def coregister_bursts(
    datadir,
    reference: str,
    repeats,
    dem=None,
) -> dict[str, object]:
    """Call ``S1.align_ref`` and ``S1.align_rep`` when ``insardev_pygmtsar`` imports.

    The return values are that package's burst products. They are not
    resampled onto this workbench's crop window. A missing package or a
    failed call leaves ``applied`` false.
    """
    module = _load_module("insardev_pygmtsar")
    if module is None or not hasattr(module, "S1"):
        return {
            "backend": "numpy",
            "applied": False,
            "note": BSD_INSTALL_NOTE,
            "reference": None,
            "aligned": None,
        }
    try:
        scene = module.S1(str(datadir), dem)
        reference_out = scene.align_ref(str(reference))
        prm = reference_out[0] if isinstance(reference_out, tuple) else reference_out
        aligned = [scene.align_rep(str(rep), str(reference), prm) for rep in repeats]
    except Exception as exc:
        return {
            "backend": "numpy",
            "applied": False,
            "note": (
                "Geometric coregistration was not applied "
                f"({type(exc).__name__}: {exc}). "
                "The SAFE zip crop stays on the shared window."
            ),
            "reference": None,
            "aligned": None,
        }
    return {
        "backend": "insardev_pygmtsar",
        "applied": True,
        "note": (
            "Called S1.align_ref and S1.align_rep. Those results are burst "
            "SLC products, not a resampled crop window."
        ),
        "reference": reference_out,
        "aligned": aligned,
    }


def translate_safe_bursts(safedir, datadir, bursts, **kwargs) -> dict[str, object]:
    """Call ``insardev_toolkit.PyGMTSAR.safeload`` when that class imports.

    ``bursts`` are the ids the toolkit expects. This does not read a zip
    crop into a ``complex64`` cube.
    """
    module = _load_module("insardev_toolkit")
    worker = getattr(module, "PyGMTSAR", None) if module is not None else None
    if worker is None:
        return {
            "backend": "numpy",
            "applied": False,
            "result": None,
            "note": BSD_INSTALL_NOTE,
        }
    try:
        result = worker().safeload(safedir, datadir, bursts, **kwargs)
    except Exception as exc:
        return {
            "backend": "numpy",
            "applied": False,
            "result": None,
            "note": (
                "PyGMTSAR.safeload was not applied "
                f"({type(exc).__name__}: {exc})."
            ),
        }
    return {
        "backend": "insardev_toolkit",
        "applied": True,
        "result": result,
        "note": "Called PyGMTSAR.safeload to translate SAFE bursts into the InSARdev layout.",
    }


def _load_module(name: str):
    if name in _MODULES:
        loaded = _MODULES[name]
        return None if loaded is False else loaded
    try:
        module = importlib.import_module(name)
    except Exception:
        _MODULES[name] = False
        return None
    _MODULES[name] = module
    return module


def _stack_apis(module) -> dict[str, bool]:
    stack = getattr(module, "Stack", None) if module is not None else None
    return {
        "Stack": stack is not None,
        "load": callable(getattr(stack, "load", None)),
        "pairs": callable(getattr(stack, "pairs", None)),
        "align": callable(getattr(stack, "align", None)),
    }


def _placeholder_dates(n_dates: int) -> np.ndarray:
    """Sequential days. NumPy 2 rejects ``arange`` with a datetime64 dtype."""
    origin = np.datetime64("2000-01-01", "D")
    return origin + np.arange(n_dates, dtype=np.int64).astype("timedelta64[D]")


def _scene_dataset(slc: np.ndarray):
    n_dates, height, width = slc.shape
    dates = _placeholder_dates(n_dates)
    cube = np.ascontiguousarray(slc, dtype=np.complex64)
    try:
        import xarray as xr
    except Exception:
        return _MemoryScene(cube, dates)
    return xr.Dataset(
        data_vars={
            "VV": (("date", "y", "x"), cube),
            "BPR": ("date", np.zeros(n_dates, dtype=np.float64)),
        },
        coords={
            "date": dates,
            "y": np.arange(height, dtype=np.float64),
            "x": np.arange(width, dtype=np.float64),
        },
    )


def _call_interferogram(module, slc: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    scene = _scene_dataset(slc)
    stack = module.Stack({"scene": scene})
    # interferogram() is reference * conjugate(repeat). Later-as-reference
    # matches later * conjugate(earlier). No wavelength: single-look, no filter.
    ordered = [(int(later), int(earlier)) for earlier, later in pairs]
    paired = stack.pairs(ordered)
    method = getattr(paired, "interferogram", None)
    if not callable(method):
        method = getattr(paired, "phasediff", None)
    if not callable(method):
        raise AttributeError("insardev pairs result has no interferogram()")
    result = method()
    cube = _complex_cube(result, (len(ordered), int(slc.shape[1]), int(slc.shape[2])))
    if cube is None:
        raise TypeError("insardev interferogram did not return a complex cube")
    return np.ascontiguousarray(cube, dtype=np.complex64)


def _try_load_scenes(files: list[Path]) -> StackAttempt | None:
    global _LAST_LOAD_ERROR
    module = _load_module("insardev")
    stack_cls = getattr(module, "Stack", None) if module is not None else None
    if not callable(getattr(stack_cls, "load", None)):
        _LAST_LOAD_ERROR = None
        return None
    urls = [str(path) for path in files]
    try:
        loaded = stack_cls().load(urls)
        cube = _first_complex_volume(loaded)
        if cube is None or cube.ndim != 3:
            raise TypeError(
                "Stack.load did not return a complex volume shaped (dates, height, width)"
            )
        cube = np.ascontiguousarray(cube, dtype=np.complex64)
    except Exception as exc:
        _LAST_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _LAST_LOAD_ERROR = None
    from ai_insar.pipeline.synthetic import SLCStack

    dates = _placeholder_dates(int(cube.shape[0]))
    stack = SLCStack(
        data=cube,
        dates=dates,
        wavelength_m=C_BAND_WAVELENGTH_M,
        sensor="insardev Stack.load",
        description=(
            "Complex cube taken from insardev.Stack.load. "
            "Dates are a sequential placeholder. This wrapper does not read "
            "manifest times or a DEM from the zarr store."
        ),
    )
    return StackAttempt(stack=stack, backend="insardev", note="insardev.Stack.load")


def _complex_cube(result, shape: tuple[int, int, int], seen: set[int] | None = None, depth: int = 0):
    if depth > 6:
        return None
    if seen is None:
        seen = set()
    ident = id(result)
    if ident in seen:
        return None
    seen.add(ident)
    if isinstance(result, np.ndarray) and np.iscomplexobj(result) and result.shape == shape:
        return result
    values = getattr(result, "values", None)
    if values is not None and not isinstance(values, (str, bytes)):
        try:
            array = np.asarray(values)
        except Exception:
            array = None
        if isinstance(array, np.ndarray) and np.iscomplexobj(array) and array.shape == shape:
            return array
    data_vars = getattr(result, "data_vars", None)
    if data_vars is not None and not isinstance(data_vars, (str, bytes)):
        try:
            names = list(data_vars)
        except Exception:
            names = []
        for name in names:
            try:
                item = result[name]
            except Exception:
                continue
            found = _complex_cube(item, shape, seen, depth + 1)
            if found is not None:
                return found
    if hasattr(result, "items") and not isinstance(result, (str, bytes, np.ndarray)):
        try:
            pairs = list(result.items())
        except Exception:
            pairs = None
        if pairs:
            for _, item in pairs:
                found = _complex_cube(item, shape, seen, depth + 1)
                if found is not None:
                    return found
    if isinstance(result, (list, tuple)) and not isinstance(result, str):
        for item in result:
            found = _complex_cube(item, shape, seen, depth + 1)
            if found is not None:
                return found
    return None


def _first_complex_volume(result, seen: set[int] | None = None, depth: int = 0):
    if depth > 6:
        return None
    if seen is None:
        seen = set()
    ident = id(result)
    if ident in seen:
        return None
    seen.add(ident)
    if isinstance(result, np.ndarray) and np.iscomplexobj(result) and result.ndim == 3:
        return result
    values = getattr(result, "values", None)
    if values is not None and not isinstance(values, (str, bytes)):
        try:
            array = np.asarray(values)
        except Exception:
            array = None
        if isinstance(array, np.ndarray) and np.iscomplexobj(array) and array.ndim == 3:
            return array
    data_vars = getattr(result, "data_vars", None)
    if data_vars is not None and not isinstance(data_vars, (str, bytes)):
        try:
            names = list(data_vars)
        except Exception:
            names = []
        for name in names:
            try:
                item = result[name]
            except Exception:
                continue
            found = _first_complex_volume(item, seen, depth + 1)
            if found is not None:
                return found
    if hasattr(result, "items") and not isinstance(result, (str, bytes, np.ndarray)):
        try:
            pairs = list(result.items())
        except Exception:
            pairs = None
        if pairs:
            for _, item in pairs:
                found = _first_complex_volume(item, seen, depth + 1)
                if found is not None:
                    return found
    if isinstance(result, (list, tuple)) and not isinstance(result, str):
        for item in result:
            found = _first_complex_volume(item, seen, depth + 1)
            if found is not None:
                return found
    return None

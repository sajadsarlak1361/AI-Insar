"""Optional insardev extra: call it when present, keep NumPy/Numba when absent."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ai_insar.insardev_optional import (
    BSD_INSTALL_NOTE,
    INSARDEV_INSTALL_NOTE,
    _complex_cube,
    align_loaded_stack,
    clear_optional_cache,
    coregister_bursts,
    insardev_status,
    pygmtsar_status,
    stack_scenes,
    translate_safe_bursts,
)
from ai_insar.pipeline.interferogram import form_interferograms

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_optional():
    clear_optional_cache()
    yield
    clear_optional_cache()


def _sample_slc() -> np.ndarray:
    rng = np.random.default_rng(3)
    slc = rng.normal(size=(3, 4, 5)) + 1j * rng.normal(size=(3, 4, 5))
    return slc.astype(np.complex64)


def _expected(slc: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([slc[late] * np.conj(slc[early]) for early, late in pairs]).astype(np.complex64)


def _slc_of(scene) -> np.ndarray:
    if isinstance(scene, np.ndarray):
        return scene
    held = getattr(scene, "slc", None)
    if isinstance(held, np.ndarray):
        return held
    value = scene["VV"]
    data = getattr(value, "values", value)
    return np.asarray(data)


def _block_module(monkeypatch, name: str) -> None:
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    real = builtins.__import__

    def guarded(modname, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (modname == name or modname.startswith(name + ".")):
            raise ImportError(name)
        return real(modname, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _install_insardev(monkeypatch, seen: dict, *, product="match", api="interferogram"):
    module = types.ModuleType("insardev")
    module.__version__ = "2026.8.28"

    def cube_for(slc, pairs):
        if product == "zeros":
            return np.zeros((len(pairs),) + slc.shape[1:], dtype=np.complex64)
        planes = [slc[ref] * np.conj(slc[rep]) for ref, rep in pairs]
        return np.stack(planes).astype(np.complex64)

    class InterferogramPairs:
        def __init__(self, slc, pairs):
            self.slc = slc
            self.pairs = pairs

        def interferogram(self, wavelength=None, **kwargs):
            seen["interferogram"] = wavelength
            seen["interferogram_kwargs"] = kwargs
            return cube_for(self.slc, self.pairs)

    class PhasediffPairs:
        def __init__(self, slc, pairs):
            self.slc = slc
            self.pairs = pairs

        def phasediff(self, wavelength=None, **kwargs):
            seen["phasediff"] = wavelength
            return cube_for(self.slc, self.pairs)

    class Stack:
        def __init__(self, mapping=None):
            seen["mapping"] = mapping

        def pairs(self, pairs):
            if product == "raise":
                raise RuntimeError("pairs rejected the in-memory cube")
            seen["pairs"] = [tuple(int(item) for item in pair) for pair in pairs]
            scene = next(iter(seen["mapping"].values()))
            paired_type = PhasediffPairs if api == "phasediff" else InterferogramPairs
            return paired_type(_slc_of(scene), pairs)

        def load(self, urls, storage_options=None, debug=False):
            seen["load"] = list(urls)
            if product == "raise":
                raise RuntimeError("no dask client")
            return seen.get("cube")

        def align(self, ref=0, **kwargs):
            seen["align"] = ref
            return f"aligned-{ref}"

    module.Stack = Stack
    monkeypatch.setitem(sys.modules, "insardev", module)
    return module


def test_missing_insardev_keeps_the_mit_product(isolated_optional, monkeypatch):
    _block_module(monkeypatch, "insardev")
    slc = _sample_slc()
    pairs = [(0, 1), (1, 2)]
    formed = form_interferograms(slc, pairs)
    assert formed.dtype == np.complex64
    assert np.allclose(formed, _expected(slc, pairs))
    assert form_interferograms.last_backend in {"numpy", "numba"}
    status = insardev_status()
    assert status["available"] is False
    assert status["version"] is None
    assert status["license"] == "InSARdev-SAL-1.0"
    assert status["apis"] == {"Stack": False, "load": False, "pairs": False, "align": False}
    assert "InSARdev-SAL-1.0" in INSARDEV_INSTALL_NOTE
    assert "pip install insardev" in str(status["note"])
    assert 'ai-insar[insardev]' in str(status["note"])
    assert "subscription" in str(status["note"])


def test_mocked_insardev_interferogram_is_used(isolated_optional, monkeypatch):
    seen: dict = {}
    _install_insardev(monkeypatch, seen)
    slc = _sample_slc()
    pairs = [(0, 1), (0, 2), (1, 2)]
    formed = form_interferograms(slc, pairs, backend="insardev")
    assert np.allclose(formed, _expected(slc, pairs))
    assert form_interferograms.last_backend == "insardev"
    assert seen["pairs"] == [(1, 0), (2, 0), (2, 1)]
    assert seen["interferogram"] is None
    scene = next(iter(seen["mapping"].values()))
    assert "VV" in getattr(scene, "data_vars", ())
    status = insardev_status()
    assert status["available"] is True
    assert status["version"] == "2026.8.28"
    assert status["apis"]["pairs"] is True
    assert status["interferogram_error"] is None


def test_phasediff_is_the_fallback_method_name(isolated_optional, monkeypatch):
    seen: dict = {}
    _install_insardev(monkeypatch, seen, api="phasediff")
    slc = _sample_slc()
    formed = form_interferograms(slc, [(0, 1)], backend="auto")
    assert np.allclose(formed[0], slc[1] * np.conj(slc[0]))
    assert form_interferograms.last_backend == "insardev"
    assert seen["phasediff"] is None
    assert "interferogram" not in seen


def test_mismatched_or_failing_insardev_falls_back(isolated_optional, monkeypatch):
    slc = _sample_slc()
    pairs = [(0, 1)]
    seen: dict = {}
    _install_insardev(monkeypatch, seen, product="zeros")
    formed = form_interferograms(slc, pairs)
    assert np.allclose(formed, _expected(slc, pairs))
    assert form_interferograms.last_backend in {"numpy", "numba"}
    assert seen["pairs"] == [(1, 0)]

    clear_optional_cache()
    seen.clear()
    _install_insardev(monkeypatch, seen, product="raise")
    formed = form_interferograms(slc, pairs, backend="insardev")
    assert np.allclose(formed, _expected(slc, pairs))
    assert form_interferograms.last_backend in {"numpy", "numba"}
    status = insardev_status()
    assert "RuntimeError" in str(status["interferogram_error"])


def test_numpy_backend_skips_insardev_even_when_installed(isolated_optional, monkeypatch):
    seen: dict = {}
    _install_insardev(monkeypatch, seen)
    monkeypatch.setenv("AI_INSAR_BACKEND", "numpy")
    slc = _sample_slc()
    formed = form_interferograms(slc, [(0, 1)])
    assert np.allclose(formed[0], slc[1] * np.conj(slc[0]))
    assert form_interferograms.last_backend in {"numpy", "numba"}
    assert "pairs" not in seen
    formed = form_interferograms(slc, [(0, 1)], backend="insardev")
    assert form_interferograms.last_backend == "insardev"
    assert seen["pairs"] == [(1, 0)]


def test_bad_backend_name_is_rejected(isolated_optional):
    slc = _sample_slc()
    with pytest.raises(ValueError, match="backend"):
        form_interferograms(slc, [(0, 1)], backend="snap")


def test_reversed_pair_does_not_call_insardev(isolated_optional, monkeypatch):
    seen: dict = {}
    _install_insardev(monkeypatch, seen)
    with pytest.raises(ValueError, match="earlier"):
        form_interferograms(_sample_slc(), [(1, 0)])
    assert "pairs" not in seen


def test_complex_cube_walks_a_batch_of_datasets():
    array = np.array([[[1 + 1j, 2]], [[3j, 4 + 0j]]], dtype=np.complex64)

    class Column:
        def __init__(self, values):
            self.values = values

    class Scene:
        def __init__(self):
            self.data_vars = ["BPR", "VV"]

        def __getitem__(self, key):
            if key == "VV":
                return Column(array)
            return Column(np.zeros(array.shape[1:], dtype=np.float32))

    class Batch(dict):
        pass

    found = _complex_cube([Batch({"scene": Scene()})], array.shape)
    assert found is not None
    assert np.allclose(found, array)
    assert _complex_cube(np.zeros((2, 2), dtype=np.float32), array.shape) is None


def test_zip_stack_stays_on_the_mit_reader(isolated_optional, monkeypatch, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    assert tifffile is not None
    from safe_factory import dated_safe_zip

    seen: dict = {}
    _install_insardev(monkeypatch, seen, product="raise")
    early = dated_safe_zip(tmp_path, "20210401", 2 + 0j)
    late = dated_safe_zip(tmp_path, "20210507", 0 + 3j)
    attempt = stack_scenes([late, early], height=2, width=2)
    assert attempt.backend == "numpy"
    assert attempt.stack is not None
    assert attempt.stack.data.shape == (2, 2, 2)
    assert attempt.stack.dates[0] == np.datetime64("2021-04-01", "D")
    assert "load" not in seen
    assert "crop" in attempt.note


def test_zarr_scene_calls_stack_load(isolated_optional, monkeypatch, tmp_path):
    seen: dict = {}
    _install_insardev(monkeypatch, seen)
    cube = np.array([[[1 + 0j, 2]], [[0 + 3j, 4]]], dtype=np.complex64)
    seen["cube"] = cube
    scene = tmp_path / "scene.zarr"
    scene.mkdir()
    attempt = stack_scenes([scene])
    assert attempt.backend == "insardev"
    assert attempt.stack is not None
    assert np.allclose(attempt.stack.data, cube)
    assert seen["load"] == [str(scene)]
    assert "placeholder" in attempt.stack.description

    clear_optional_cache()
    seen.clear()
    _install_insardev(monkeypatch, seen, product="raise")
    with pytest.raises(ValueError, match="dask client"):
        stack_scenes([scene])


def test_scene_without_insardev_explains_sal(isolated_optional, monkeypatch, tmp_path):
    _block_module(monkeypatch, "insardev")
    scene = tmp_path / "scene.zarr"
    scene.mkdir()
    with pytest.raises(ImportError, match="InSARdev-SAL-1.0"):
        stack_scenes([scene])


def test_coregister_and_safeload_call_bsd_wrappers(isolated_optional, monkeypatch):
    calls: list = []
    pygmtsar = types.ModuleType("insardev_pygmtsar")
    pygmtsar.__version__ = "2026.3.21"

    class S1:
        def __init__(self, datadir, dem=None):
            calls.append(("init", datadir, dem))

        def align_ref(self, burst, **kwargs):
            calls.append(("ref", burst))
            return ("prm-ref", None, None)

        def align_rep(self, burst_rep, burst_ref, prm_ref, **kwargs):
            calls.append(("rep", burst_rep, burst_ref, prm_ref))
            return (burst_rep, "slc", None)

    pygmtsar.S1 = S1
    monkeypatch.setitem(sys.modules, "insardev_pygmtsar", pygmtsar)

    toolkit = types.ModuleType("insardev_toolkit")

    class PyGMTSAR:
        def safeload(self, safedir, datadir, bursts, **kwargs):
            calls.append(("safeload", safedir, datadir, list(bursts), kwargs))
            return "layout"

    toolkit.PyGMTSAR = PyGMTSAR
    monkeypatch.setitem(sys.modules, "insardev_toolkit", toolkit)

    coreg = coregister_bursts("/data", "burst0", ["burst1", "burst2"], dem="/dem.tif")
    assert coreg["applied"] is True
    assert coreg["backend"] == "insardev_pygmtsar"
    assert ("rep", "burst1", "burst0", "prm-ref") in calls
    assert ("rep", "burst2", "burst0", "prm-ref") in calls
    translated = translate_safe_bursts("safes", "out", ["burst0"], skip_exist=True)
    assert translated["applied"] is True
    assert translated["backend"] == "insardev_toolkit"
    assert translated["result"] == "layout"
    assert calls[-1][0] == "safeload"
    assert calls[-1][4]["skip_exist"] is True

    aligned = align_loaded_stack(_Alignable(), ref=1)
    assert aligned["applied"] is True
    assert aligned["backend"] == "insardev"
    assert aligned["result"] == "aligned-1"


class _Alignable:
    def align(self, ref=0):
        return f"aligned-{ref}"


def test_bsd_wrappers_fall_back_when_missing(isolated_optional, monkeypatch):
    _block_module(monkeypatch, "insardev_pygmtsar")
    _block_module(monkeypatch, "insardev_toolkit")
    coreg = coregister_bursts("/data", "burst0", ["burst1"])
    assert coreg["applied"] is False
    assert coreg["backend"] == "numpy"
    assert "BSD-3-Clause" in coreg["note"]
    assert "BSD-3-Clause" in BSD_INSTALL_NOTE
    translated = translate_safe_bursts("safes", "out", ["burst0"])
    assert translated["applied"] is False
    assert translated["backend"] == "numpy"
    broken = align_loaded_stack(_BrokenAlign())
    assert broken["applied"] is False
    assert broken["backend"] == "numpy"
    assert "RuntimeError" in broken["note"]
    status = pygmtsar_status()
    assert status["available"] is False


class _BrokenAlign:
    def align(self, ref=0):
        raise RuntimeError("no burst overlaps")


def test_installed_insardev_matches_or_skips(isolated_optional):
    insardev = pytest.importorskip("insardev")
    if getattr(insardev, "__file__", None) in (None, ""):
        pytest.skip("insardev import is a test double")
    slc = _sample_slc()
    pairs = [(0, 1), (1, 2)]
    formed = form_interferograms(slc, pairs, backend="insardev")
    assert np.allclose(formed, _expected(slc, pairs))
    if form_interferograms.last_backend != "insardev":
        pytest.skip(
            "installed insardev did not return a matching single-look interferogram; "
            "the NumPy/Numba path was kept"
        )


def test_docs_keep_the_extra_honest():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = (ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "InSARdev-SAL-1.0" in notice
    assert "not vendored" in notice
    assert "subscription" in notice
    assert 'pip install "ai-insar[insardev]"' in notice
    assert "GPL" in notice
    assert "SNAP" in notice
    for page in (readme, install):
        assert 'pip install "ai-insar[insardev]"' in page
        assert "InSARdev-SAL-1.0" in page
        assert "subscription" in page
    assert "is not a project extra" not in install
    assert "Stack.pairs" in readme
    assert "NumPy/Numba" in readme
    main, _, extras = project.partition("[project.optional-dependencies]")
    dependencies = main.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "insardev" not in dependencies
    assert "insardev>=" in extras
    assert "python_version>='3.12'" in extras
    assert not (ROOT / "src" / "ai_insar" / "insardev").exists()
    assert not (ROOT / "src" / "ai_insar" / "snap").exists()

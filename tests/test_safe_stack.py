"""Many SAFE zips become one time-sorted complex64 stack, then interferograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")

from ai_insar.insardev_optional import BSD_INSTALL_NOTE, INSARDEV_INSTALL_NOTE, insardev_status
from ai_insar.pipeline.engine import Workspace, compute_interferograms
from ai_insar.pipeline.interferogram import _form_vectorized, _pair_index, form_interferograms
from ai_insar.safe_stack import SafeStackCancelled, stack_safe_zips
from safe_factory import FIXTURE_ZIP, dated_safe_zip

ROOT = Path(__file__).resolve().parents[1]


def test_many_zips_sort_and_form_later_conj_earlier(tmp_path):
    early = dated_safe_zip(tmp_path, "20210401", 2 + 0j)
    late = dated_safe_zip(tmp_path, "20210507", 0 + 3j)
    seen: list[tuple[int, int, str]] = []

    def progress(step, steps, label):
        seen.append((step, steps, label))
        return None

    stack = stack_safe_zips(
        [late, early],
        swath="IW1",
        pol="VV",
        row=0,
        col=0,
        height=2,
        width=2,
        progress=progress,
    )
    assert stack.data.shape == (2, 2, 2)
    assert stack.data.dtype == np.complex64
    assert stack.dates[0] == np.datetime64("2021-04-01", "D")
    assert stack.dates[1] == np.datetime64("2021-05-07", "D")
    assert np.allclose(stack.data[0], 2 + 0j)
    assert np.allclose(stack.data[1], 0 + 3j)
    assert "crop-window alignment" in stack.description
    assert "no coregistration" in stack.description
    assert [item[0] for item in seen] == [1, 2, 3, 4]
    assert seen[0][1] == 4
    assert seen[0][2].startswith("Opening 1 of 2:")
    assert "Reading 1 of 2:" in seen[2][2]
    assert "20210401" in seen[2][2]
    assert "20210507" in seen[3][2]

    formed = form_interferograms(stack.data, [(0, 1)])
    assert formed.dtype == np.complex64
    assert formed.shape == (1, 2, 2)
    assert np.allclose(formed[0], (0 + 3j) * np.conj(np.complex64(2 + 0j)))

    workspace = Workspace()
    workspace.stack = stack
    compute_interferograms(workspace)
    assert workspace.interferograms is not None
    assert np.allclose(workspace.interferograms[0], formed[0])


def test_progress_cancel_and_rejected_inputs(tmp_path):
    early = dated_safe_zip(tmp_path, "20210401", 1 + 0j)
    with pytest.raises(SafeStackCancelled):
        stack_safe_zips([early], progress=lambda *_args: False)
    with pytest.raises(ValueError, match="SAFE zip"):
        stack_safe_zips([tmp_path / "notes.txt"])
    with pytest.raises(ValueError, match="IW1 HH"):
        stack_safe_zips([FIXTURE_ZIP], pol="HH", height=1, width=1)
    same = dated_safe_zip(tmp_path, "20210601", 1 + 1j)
    copy = tmp_path / "copy"
    copy.mkdir()
    twin = dated_safe_zip(copy, "20210601", 4 + 0j)
    with pytest.raises(ValueError, match="strictly increasing"):
        stack_safe_zips([same, twin], height=2, width=2)


def test_vectorized_chunks_match_the_definition():
    rng = np.random.default_rng(11)
    slc = (rng.normal(size=(6, 8, 5)) + 1j * rng.normal(size=(6, 8, 5))).astype(np.complex64)
    pairs = [(0, 1), (1, 4), (2, 5), (0, 5)]
    formed = form_interferograms(slc, pairs)
    earlier, later = _pair_index(slc.shape[0], pairs)
    chunked = _form_vectorized(np.ascontiguousarray(slc), earlier, later)
    for index, (early, late) in enumerate(pairs):
        expected = slc[late] * np.conj(slc[early])
        assert np.allclose(formed[index], expected)
        assert np.allclose(chunked[index], expected)


def test_numba_product_matches_when_installed():
    numba = pytest.importorskip("numba")
    assert numba is not None
    from ai_insar.pipeline.interferogram import _numba_form

    kernel = _numba_form()
    if kernel is None:
        pytest.skip("Numba did not compile the interferogram kernel")
    slc = np.array([[[1 + 2j, 3 + 0j], [0 + 1j, 4 - 1j]], [[2 - 1j, 0 + 2j], [1 + 1j, 5 + 0j]]], dtype=np.complex64)
    formed = kernel(np.ascontiguousarray(slc), np.array([0], dtype=np.int64), np.array([1], dtype=np.int64))
    assert formed.dtype == np.complex64
    assert np.allclose(formed[0], slc[1] * np.conj(slc[0]))


def test_notice_keeps_sal_out_of_the_mit_tree():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "InSARdev-SAL-1.0" in notice
    assert "not vendored" in notice
    assert "ai-insar[insardev]" in notice
    assert "subscription" in notice
    assert "NISAR/nisar_numpy.py" in notice
    assert "GPL" in notice
    assert not (ROOT / "src" / "ai_insar" / "insardev" / "Stack.py").exists()
    assert not (ROOT / "src" / "ai_insar" / "snap").exists()
    status = insardev_status()
    assert "note" in status
    assert "InSARdev-SAL-1.0" in INSARDEV_INSTALL_NOTE
    assert "pip install insardev" in str(status["note"])
    assert "BSD-3-Clause" in BSD_INSTALL_NOTE
    if status["available"]:
        assert status["version"]
    else:
        assert status["version"] is None

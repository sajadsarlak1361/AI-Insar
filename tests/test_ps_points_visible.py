"""PS linear points stay visible when the results view is smaller than the raster.

QPainter.drawImage keeps one source pixel per screen pixel. A persistent
scatterer is one opaque sample on a transparent raster, so that sampling
dropped the point whenever the view scaled the image down.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from ai_insar.gui.main_window import MainWindow
from ai_insar.pipeline.interferogram import date_pairs, form_interferograms
from ai_insar.pipeline.ps_export import METRES_TO_MILLIMETRES
from ai_insar.pipeline.synthetic import generate_subsidence_stack
from ai_insar.pipeline.timeseries import persistent_scatterer_velocity

PAIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pair"
APRIL = PAIR / "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4.zip"
LATER = PAIR / "S1B_IW_SLC__1SDV_20210413T052622_20210413T052650_026269_032297_EFA4.zip"
_BACKGROUND = (0x12, 0x17, 0x1F)


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def _show_ps(window: MainWindow, velocity_m_yr: np.ndarray, mask: np.ndarray, view: tuple[int, int]) -> None:
    window.workspace.velocity_m_yr = velocity_m_yr
    window.workspace.ps_mask = mask
    window.viewer.setFixedSize(*view)
    window._select_layer("LOS velocity", refresh=True)
    window.show()
    QApplication.processEvents()


def _visible_points(window: MainWindow, mask: np.ndarray) -> int:
    viewer = window.viewer
    grab = viewer.grab().toImage()
    target = viewer._target_rect()
    height, width = mask.shape
    rows, cols = np.nonzero(mask)
    sx = target.width() / width
    sy = target.height() / height
    hits = 0
    for row, col in zip(rows.tolist(), cols.tolist()):
        x = int(round(target.x() + (col + 0.5) * sx))
        y = int(round(target.y() + (row + 0.5) * sy))
        x = min(max(x, 0), grab.width() - 1)
        y = min(max(y, 0), grab.height() - 1)
        colour = grab.pixelColor(x, y)
        if colour.alpha() > 0 and (colour.red(), colour.green(), colour.blue()) != _BACKGROUND:
            hits += 1
    return hits


def _assert_points_visible(window: MainWindow, mask: np.ndarray) -> int:
    count = int(np.count_nonzero(mask))
    assert count > 0
    image = window.viewer._image
    assert image is not None and image.hasAlphaChannel()
    opaque = 0
    for row, col in zip(*np.nonzero(mask)):
        colour = image.pixelColor(int(col), int(row))
        assert colour.alpha() == 255
        opaque += 1
    assert opaque == count
    values = np.asarray(window.viewer._values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    low = float(window.colorbar._vmin)
    high = float(window.colorbar._vmax)
    assert window.colorbar._caption == "mm/yr"
    assert finite.size == count
    assert low < 0.0 < high
    # A point inside the colour bar is not clipped. The 2–98% ends can
    # saturate a few outliers; those samples stay opaque and are still drawn.
    inside = finite[(finite >= low) & (finite <= high)]
    assert inside.size > 0
    hits = _visible_points(window, mask)
    assert hits == count
    return hits


def test_sparse_ps_points_survive_a_smaller_results_view(qapp):
    height, width = 240, 320
    velocity = np.full((height, width), np.nan)
    mask = np.zeros((height, width), dtype=bool)
    rows = np.linspace(8, height - 9, 8, dtype=int)
    cols = np.linspace(8, width - 9, 5, dtype=int)
    for row in rows:
        for col in cols:
            velocity[row, col] = -0.05
            mask[row, col] = True
    window = MainWindow()
    _show_ps(window, velocity, mask, (200, 150))
    hits = _assert_points_visible(window, mask)
    assert hits == 40
    window.close()


def test_synthetic_ps_linear_points_are_drawn(qapp):
    stack = generate_subsidence_stack(
        n_dates=5,
        height=48,
        width=64,
        spacing_days=12,
        rate_m_yr=0.05,
        sigma_px=8,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=5,
    )
    pairs = date_pairs(stack.n_dates, 1)
    phase = np.angle(form_interferograms(stack.data, pairs).astype(np.complex128))
    coherence = np.ones(stack.data.shape[1:])
    _los, velocity, mask = persistent_scatterer_velocity(
        stack.data,
        phase,
        pairs,
        stack.times_year(),
        stack.wavelength_m,
        np.ones(phase.shape),
        coherence,
        da_max=1.0,
        min_coherence=0.0,
    )
    # Real PS sets are sparse. Keep a separated subset of the fitted pixels.
    chosen = np.zeros(mask.shape, dtype=bool)
    rows, cols = np.nonzero(mask & np.isfinite(velocity))
    step = max(1, rows.size // 24)
    for row, col in zip(rows[::step].tolist(), cols[::step].tolist()):
        chosen[row, col] = True
        if int(chosen.sum()) == 24:
            break
    shown = np.where(chosen, velocity, np.nan)
    window = MainWindow()
    _show_ps(window, shown, chosen, (120, 90))
    hits = _assert_points_visible(window, chosen)
    peak_mm = float(np.nanmin(velocity)) * METRES_TO_MILLIMETRES
    assert peak_mm == pytest.approx(-50.0, abs=2.0)
    assert hits == int(chosen.sum())
    window.close()


def test_safe_pair_grid_points_are_drawn(qapp, tmp_path: Path):
    from ai_insar.pipeline.safe_pair import run_safe_pair

    result = run_safe_pair(APRIL, LATER, tmp_path)
    interferogram = np.load(result["interferogram"])
    height, width = interferogram.shape
    assert (height, width) == (32, 48)
    velocity = np.full((height, width), np.nan)
    mask = np.zeros((height, width), dtype=bool)
    for row, col in ((4, 6), (4, 24), (4, 40), (16, 6), (16, 24), (16, 40), (27, 6), (27, 40)):
        velocity[row, col] = -0.05
        mask[row, col] = True
    window = MainWindow()
    _show_ps(window, velocity, mask, (80, 64))
    hits = _assert_points_visible(window, mask)
    assert hits == 8
    window.close()

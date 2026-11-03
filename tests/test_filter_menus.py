"""Goldstein and Coherence are top-level menus, not only pipeline items."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu

from ai_insar.conventions import LIMITATIONS_TEXT
from ai_insar.gui.main_window import MainWindow
from ai_insar.pipeline.synthetic import generate_subsidence_stack


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def test_presets_run_filter_and_coherence(qapp):
    window = MainWindow()
    window.workspace.stack = generate_subsidence_stack(
        n_dates=4,
        height=24,
        width=24,
        spacing_days=12,
        rate_m_yr=0.1,
        sigma_px=4,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=1,
    )
    window.run_interferograms()
    window.run_goldstein_preset(0.8)
    assert window.workspace.config.goldstein_alpha == 0.8
    assert window.workspace.filtered is not None
    assert window.workspace.filtered.shape == window.workspace.interferograms.shape
    window.run_coherence_preset(3)
    assert window.workspace.config.coherence_window == 3
    assert window.workspace.coherence.shape == (window.workspace.interferograms.shape[0], 24, 24)
    assert float(np.nanmax(window.workspace.coherence)) <= 1.0
    log = window.log.toPlainText()
    assert "ERROR" not in log
    assert "alpha set to 0.80" in log
    assert "3×3" in log

    titles = [action.text() for action in window.menuBar().actions()]
    assert "Goldstein" in titles
    assert "Coherence" in titles
    strong = window.findChild(QAction, "goldsteinPreset_strong")
    assert strong is not None and "0.8" in strong.text()
    wide = window.findChild(QAction, "coherencePreset_wide")
    assert wide is not None and "7×7" in wide.text()
    goldstein = window.findChild(QMenu, "goldsteinMenu")
    coherence = window.findChild(QMenu, "coherenceMenu")
    assert goldstein is not None and coherence is not None
    window.close()


def test_help_text_names_the_shipped_methods():
    assert "Goldstein" in LIMITATIONS_TEXT
    assert "Seasonal" in LIMITATIONS_TEXT
    assert "Fast unwrap" in LIMITATIONS_TEXT
    assert "no CRS" in LIMITATIONS_TEXT

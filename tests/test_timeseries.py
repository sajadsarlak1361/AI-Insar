"""The stage-6 methods besides SBAS recover a linear subsidence bowl."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from ai_insar.pipeline.engine import PipelineConfig, Workspace, run_all
from ai_insar.pipeline.synthetic import generate_subsidence_stack
from ai_insar.pipeline.timeseries import TIMESERIES_METHODS


def _scene():
    return generate_subsidence_stack(
        n_dates=6,
        height=36,
        width=36,
        spacing_days=36,
        rate_m_yr=0.2,
        sigma_px=7,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=4,
    )


def _peak_error(stack, velocity):
    truth = stack.truth_velocity_m_yr
    row, col = np.unravel_index(int(np.argmax(np.abs(truth))), truth.shape)
    estimated = float(velocity[row, col])
    expected = float(truth[row, col])
    finite = np.isfinite(velocity)
    correlation = float(np.corrcoef(truth[finite], velocity[finite])[0, 1])
    return estimated, expected, correlation, float(finite.mean())


@pytest.mark.parametrize("method", ["stacking", "nsbas", "ps", "linking", "seasonal"])
def test_each_method_recovers_subsidence(method):
    stack = _scene()
    workspace = Workspace(
        stack=stack,
        config=PipelineConfig(
            timeseries_method=method,
            goldstein_alpha=0.35,
            goldstein_window=16,
            coherence_window=3,
        ),
    )
    run_all(workspace)
    estimated, expected, correlation, kept = _peak_error(stack, workspace.velocity_m_yr)
    assert estimated < 0
    assert expected < 0
    assert abs(estimated - expected) / abs(expected) < 0.25
    assert correlation > 0.9
    assert kept > 0.5
    assert workspace.config.timeseries_method == method


def test_separable_unwrap_tracks_a_smooth_bowl():
    from ai_insar.pipeline.unwrap import unwrap_separable

    yy, xx = np.mgrid[0:60, 0:60]
    true = 6.0 * np.exp(-((yy - 30.0) ** 2 + (xx - 30.0) ** 2) / (2.0 * 12.0**2))
    wrapped = np.angle(np.exp(1j * true))
    unwrapped = unwrap_separable(wrapped, np.ones_like(true), sigma=1.5, min_coherence=0.0)
    finite = np.isfinite(unwrapped)
    correlation = float(np.corrcoef(true[finite], unwrapped[finite])[0, 1])
    assert correlation > 0.95


def test_emi_recovers_a_subfringe_phase_series():
    from ai_insar.pipeline.timeseries import emi_principal_phase

    dates = 4
    true = np.array([0.0, 0.25, 0.55, 0.9])
    pairs = [(i, j) for i in range(dates) for j in range(i + 1, dates)]
    phase = np.zeros((len(pairs), 2, 2))
    coherence = np.full_like(phase, 0.95)
    for index, (earlier, later) in enumerate(pairs):
        phase[index] = true[later] - true[earlier]
    linked = emi_principal_phase(coherence, phase, pairs, dates)
    offset = linked[:, 0, 0] - true
    assert np.max(np.abs(linked[:, 0, 0] - true - np.median(offset))) < 0.05


def test_sbas_remains_the_default():
    assert PipelineConfig().timeseries_method == "sbas"
    assert "sbas" in TIMESERIES_METHODS
    assert len(TIMESERIES_METHODS) >= 5


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="timeseries_method"):
        PipelineConfig(timeseries_method="mintpy")


def test_method_combo_changes_stage_button():
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication, QMenu

    from ai_insar.gui.main_window import MainWindow

    _app = QApplication.instance() or QApplication([])
    window = MainWindow()
    labels = [window.method_combo.itemText(index) for index in range(window.method_combo.count())]
    assert labels == ["SBAS", "Stacking", "NSBAS", "PS linear", "Phase linking", "Seasonal"]
    assert "SBAS" in window.stage_buttons[5].text()
    window.set_timeseries_method("stacking")
    assert window.workspace.config.timeseries_method == "stacking"
    assert "Stacking" in window.stage_buttons[5].text()
    window.set_timeseries_method("ps")
    assert "PS linear" in window.stage_buttons[5].text()
    titles = [action.text() for action in window.menuBar().actions()]
    for name in ("SBAS", "Stacking", "NSBAS", "PS linear", "Phase linking", "Seasonal", "Quality-guided", "Fast unwrap"):
        assert name in titles
    stacking = window.findChild(QMenu, "methodMenu_stacking")
    assert stacking is not None
    run = window.findChild(QAction, "runMethod_nsbas")
    assert run is not None and run.text() == "Run NSBAS"
    window.close()

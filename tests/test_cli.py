"""Headless command-line tests."""

from __future__ import annotations

import numpy as np

from ai_insar.cli import main


def test_check_command_passes(capsys):
    assert main(["check"]) == 0
    output = capsys.readouterr().out
    assert "PASS" in output
    assert "FAIL" not in output.splitlines()[-1]


def test_run_command_writes_products(tmp_path):
    code = main(["run", "--demo", "--output", str(tmp_path)])
    assert code == 0
    velocity = np.load(tmp_path / "velocity_m_yr.npy")
    mask = np.load(tmp_path / "mask.npy")
    los = np.load(tmp_path / "los_m.npy")
    assert velocity.ndim == 2
    assert mask.shape == velocity.shape
    assert los.shape[0] >= 2
    assert los.shape[1:] == velocity.shape
    assert float(np.min(velocity)) < 0
    assert int(mask.sum()) > 0
    assert (tmp_path / "products.json").is_file()


def test_help_exits_cleanly():
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0

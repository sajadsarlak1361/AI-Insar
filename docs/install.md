# Install

AI Insar is installed from a clone of this repository. The project name on
pip is `ai-insar` (import name `ai_insar`). Version 0.1.0 requires Python
3.10 or newer.

Runtime dependencies, from `pyproject.toml`:

- `numpy>=1.24`
- `scipy>=1.11`
- `PySide6>=6.5`

The `dev` extra adds `pytest>=7.4`. PySide6 is how the GUI is developed. If
that import fails, `ai_insar.gui.qt` tries PyQt6 instead. The declared
install does not pull PyQt6 in for you.

## Editable install (usual for a checkout)

```bash
git clone https://github.com/sajadsarlak1361/AI-Insar.git
cd AI-Insar
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Omit `.[dev]` when you do not want pytest: `pip install -e .`.

A non-editable install from the same directory is `pip install .`. The
Windows batch file uses `pip install -e .` so the checkout and the
environment stay tied together. See [WINDOWS.md](../WINDOWS.md).

## Confirm

```bash
ai-insar --version
python -m ai_insar --version
```

Both print `ai-insar 0.1.0`.

```bash
ai-insar check
```

This does not open a window. It exits 0 when the synthetic subsidence bowl
is recovered inside the thresholds in [cli.md](cli.md). Then:

```bash
python -m ai_insar
```

opens the workbench. With no subcommand, `ai-insar` and `ai-insar-gui` do
the same thing: they call `ai_insar.cli:main`, which launches the GUI.

## If the GUI cannot import

The CLI prints this and exits 1:

```text
The desktop GUI could not be imported. Install PySide6 (pip install PySide6) and retry.
```

`ai-insar check` and `ai-insar run` do not import Qt, so they still run.

On Linux, a successful pip install can still fail later with
`Could not load the Qt platform plugin "xcb"` if the system is missing X11
libraries the PySide6 wheel expects. That is a machine package, not a
project extra. The headless commands remain the way to test the pipeline
without a display. The automated GUI test sets `QT_QPA_PLATFORM=offscreen`.

## Running against `src/` without an editable install

`pytest` adds `src` to `sys.path` via `pyproject.toml`
(`[tool.pytest.ini_options] pythonpath`). The dependencies still have to be
installed. Launching the app the same way is:

```bash
PYTHONPATH=src python -m ai_insar check
```

Prefer `pip install -e .` so `ai-insar` is on `PATH`.

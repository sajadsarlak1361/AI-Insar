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

## Optional readers

Measurement GeoTIFFs inside a SAFE zip need `tifffile` (and `zarr` for a
large crop read in tiles):

```bash
pip install -e ".[slc]"
```

Numba speeds the interferogram product, the Goldstein overlap-add, and the
quality-guided unwrap. The default Goldstein filter is a batched SciPy FFT
and the default unwrap is the fast separable path, so a large stack is
already faster without this extra. Neither step needs `insardev`.

```bash
pip install -e ".[fast]"
```

Time the synthetic path with `python scripts/bench_pipeline.py`. To run the
same Goldstein FFT on PyTorch, install Torch yourself and set
`AI_INSAR_FILTER=torch`. The default device is CPU
(`AI_INSAR_FILTER_DEVICE=cpu`). GPU is not required, and the tests pass
without Torch.

## Optional InSARdev (InSARdev-SAL-1.0)

`insardev` is not vendored. The extra installs the upstream package on
Python 3.12 or newer:

```bash
pip install "ai-insar[insardev]"
```

Installing or using that package means you accept InSARdev-SAL-1.0,
including a subscription for funded academic, institutional, and
professional use. This MIT project does not grant the subscription. Read
[NOTICE](../NOTICE) and the upstream
[LICENSE](https://github.com/InSARdev/core/blob/main/insardev/LICENSE) and
[SUBSCRIBE](https://github.com/InSARdev/core/blob/main/insardev/SUBSCRIBE)
pages before you install.

With the package present, interferograms, `Stack.load`, and `Stack.align`
prefer InSARdev. A returned interferogram cube is kept even when it
differs from `later * conjugate(earlier)`. The NumPy/Numba product runs
when the package is missing, the call fails, or you set
`AI_INSAR_BACKEND=numpy`. A SAFE zip list uses the in-tree crop only after
`Stack.load` fails. `insardev_pygmtsar` (BSD-3-Clause) is a separate
install for `S1.align_ref` / `S1.align_rep`, and that call is also skipped
when `AI_INSAR_BACKEND=numpy`. See [InSARdev](../README.md#insardev).

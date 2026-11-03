# AI Insar

AI Insar is an open-source **SLC-first** InSAR workbench for teaching and for
small experiments. It is a PySide6 desktop app with seven numbered stages, a
headless command line, and a Windows launcher. Version 0.1.0. MIT license.

You give it a complex SLC stack. It forms interferograms, filters them,
estimates coherence, unwraps phase, runs a small-baseline (SBAS) inversion,
and draws a displacement mask. Nothing is downloaded. The sample button
builds a synthetic C-band stack with a known subsidence bowl.

## What it is

- A desktop workbench: menus, a log, a raster viewer, and stage buttons.
- A headless runner: `ai-insar check` and `ai-insar run`.
- A radar-grid tool. Arrays stay in memory at the SLC resolution.

## What it is not

AI Insar is **not** a replacement for ISCE, MintPy, SNAP, or SNAPHU (nor for
GMTSAR or Doris). It does not geocode, download orbits, stitch bursts, or
apply a weather model. The displacement mask is a coherence-gated statistical
test on the velocity image, not a trained model. Details are in
[docs/limitations.md](docs/limitations.md).

## Primary input

An SLC cube:

| | |
| --- | --- |
| dtype | `complex64` |
| shape | `(n_dates, height, width)` |
| time axis | index increases with time |

Interferograms are **later × conjugate(earlier)**:

```text
ifg = slc[later] * conj(slc[earlier])
angle(ifg) = wrap(phase[later] - phase[earlier])
```

Line-of-sight (LOS) displacement is **positive toward the radar**. A
subsidence bowl (motion away from the radar) has **negative** LOS velocity.
SLC phase is `phase = -4π / wavelength × los_meters`, so that same bowl has
**positive** unwrapped phase on a later-minus-earlier pair. Wrapped phase can
still flip sign after half a cycle. One fringe is `wavelength / 2`: about
2.77 cm at the Sentinel-1 C-band wavelength used by the demo (5.405 GHz).

The file layout is in [docs/data-format.md](docs/data-format.md). The sign
convention is the same one the Help menu shows.

## Install

Python 3.10 or newer. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # includes pytest
```

Users who will not run the tests can install with `pip install -e .` or
`pip install .`. There is no separate PyPI release step in this repository:
install from the checkout. PySide6, NumPy, and SciPy are the runtime
dependencies. If PySide6 cannot be imported, the GUI tries PyQt6.

Full notes, including how to confirm the install: [docs/install.md](docs/install.md).

On Windows you can skip the clone: [Download the Windows zip from Releases](https://github.com/sajadsarlak1361/AI-Insar/releases/download/v0.1.0/ai-insar-windows.zip) (`ai-insar-windows.zip` on the [v0.1.0 release](https://github.com/sajadsarlak1361/AI-Insar/releases/tag/v0.1.0)). Unzip it and double-click `run_windows.bat`. That asset is not committed to the git tree. Details: [WINDOWS.md](WINDOWS.md).

## Quick start

Open the window (no subcommand, `gui`, or the `ai-insar` / `ai-insar-gui`
scripts all do this):

```bash
python -m ai_insar
```

Click **1 Generate / load sample SLC**, then **2** through **7**, or click
**Run all stages**. The viewer should end on LOS velocity with an orange mask
over a blue bowl. Blue is motion away from the radar. Click the bowl: the
readout is a negative velocity in m/yr, plus the LOS time series in
millimetres.

Check the same demo without a display. Exit status is 0 on `PASS` and 1 on
`FAIL`:

```bash
ai-insar check
python -m ai_insar check
```

On the default seed the check prints a peak LOS velocity near −0.22 m/yr, a
correlation near 1, and `PASS`. The exact lines are reproduced in
[docs/cli.md](docs/cli.md). A longer click-through is in
[docs/quickstart.md](docs/quickstart.md).

Write products (the demo stack, or your own SLC) with:

```bash
ai-insar run --output products
ai-insar run --slc slc.npy --meta slc.json --output products
```

`--output` is required. Omitting `--slc` uses the built-in bowl, whether or
not you pass `--demo`.

## Stages

| # | Button | What it does |
| --- | --- | --- |
| 1 | Generate / load sample SLC | Synthetic C-band stack, or File → Load SLC stack for a `.npy` |
| 2 | Interferogram | Pairs with date-index gap 1…3, each `later * conj(earlier)` |
| 3 | Filter | Goldstein spectral filter (default α = 0.5, 32 px window) |
| 4 | Coherence | Boxcar coherence from the SLC pair, not from the filtered phase |
| 5 | Unwrap | Quality-guided unwrap, then a plane removed on a 4 px border |
| 6 | SBAS / time series | LOS displacement (first date = 0) and one linear velocity |
| 7 | Displacement detection mask | Coherent pixels that are fast relative to the scene |

**Run all stages** generates the sample only when no stack is loaded, then
runs stages 2–7. Pipeline → Settings edits the date span, Goldstein α and
window, coherence window, unwrap coherence floor, and two detection
thresholds. Other defaults stay as in code until you change them in Python.
Stage inputs, outputs, and the default table are in
[docs/pipeline.md](docs/pipeline.md).

The sample stack is 8 dates of 72×72, every 36 days from 2020-01-08, seed 7.
The Gaussian bowl’s model peak is 0.22 m/yr of subsidence (stored as −0.22
m/yr). Truth arrays exist only for this synthetic stack.

## Documentation

| Guide | Contents |
| --- | --- |
| [docs/install.md](docs/install.md) | venv, `pip install`, `--version` |
| [docs/quickstart.md](docs/quickstart.md) | GUI clicks, shortcuts, saving products |
| [docs/pipeline.md](docs/pipeline.md) | Each stage, defaults, what Settings changes |
| [docs/data-format.md](docs/data-format.md) | `slc.npy`, sidecar JSON, `products.json` |
| [docs/cli.md](docs/cli.md) | `ai-insar --help`, exit codes, example `check` output |
| [docs/limitations.md](docs/limitations.md) | What the chain does not do |
| [WINDOWS.md](WINDOWS.md) | Release zip, `run_windows.bat`, and `dist/ai-insar-windows.zip` |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Tests and how to keep docs honest |

## Tests

```bash
pytest
```

`tests/test_pipeline.py` covers the interferogram definition, the LOS sign,
Goldstein reconstruction, multi-fringe unwrapping, SBAS, the demo bowl, and a
scene with no deformation. `tests/test_cli.py` runs `check` and `run`.
`tests/test_gui_smoke.py` clicks the seven stage buttons with
`QT_QPA_PLATFORM=offscreen` when PySide6 is installed. `tests/test_docs.py`
checks that links in these pages point at files in the tree.

## License

MIT. See [LICENSE](LICENSE).

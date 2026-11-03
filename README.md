# AI Insar

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](docs/install.md)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)](#tests)

SLC-first open-source InSAR workbench with a PySide6 GUI. It is a desktop
app, a headless command line, and a Windows launcher for teaching and for
small in-memory experiments. Version 0.1.0. MIT license.

The primary input is a `complex64` SLC stack, shape `(dates, height, width)`.
Interferograms are derived as **later × conjugate(earlier)**. The sample
button builds a synthetic C-band subsidence bowl. Arrays stay on the radar
grid.

AI Insar is **not** a replacement for ISCE, MintPy, SNAP, or SNAPHU (nor for
GMTSAR or Doris).

## Install

Python 3.10 or newer. From a clone:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # includes pytest
```

`pip install -e .` is enough if you will not run the tests. There is no PyPI
package: install from this checkout. Runtime dependencies are PySide6, NumPy,
and SciPy. If PySide6 cannot be imported, the GUI tries PyQt6.

Optional academic extra, Python 3.12 or newer:

```bash
pip install "ai-insar[insardev]"
```

That installs upstream `insardev` (InSARdev-SAL-1.0). The SAL sources stay
out of this MIT tree. Installing or using the extra means you accept that
license, including a subscription for funded academic use. The repository
owner has personally accepted that license for this academic extra. With
the package installed, interferograms, `Stack.load`, and `Stack.align`
use InSARdev first. A different numeric result is kept. NumPy/Numba runs
when the package is missing, the call fails, or you set
`AI_INSAR_BACKEND=numpy`. Details: [InSARdev](#insardev) and [NOTICE](NOTICE).

Notes: [docs/install.md](docs/install.md).

Windows without a clone: the stable zip is
[v0.1.0](https://github.com/sajadsarlak1361/AI-Insar/releases/download/v0.1.0/ai-insar-windows.zip).
The last uploaded prerelease is
[v0.2.0-rc.4](https://github.com/sajadsarlak1361/AI-Insar/releases/download/v0.2.0-rc.4/ai-insar-windows.zip)
(Sentinel-1 menu and the extra methods). That zip does not include the
Goldstein and Coherence menus or Export map. Those are on `main`. After the
next package is built, the tag to publish is `v0.2.0-rc.5`; that asset is
not uploaded. Unzip `ai-insar-windows.zip` and double-click `run_windows.bat`.
Details: [WINDOWS.md](WINDOWS.md).

## Quick start

```bash
python -m ai_insar
```

Click **1 Generate / load sample SLC**, then **2** through **7**, or **Run
all stages**. The viewer ends on LOS velocity: an orange mask over a blue
bowl. Blue is motion away from the radar. A click on the bowl reads a
negative velocity in m/yr and the LOS series in millimetres.

The same demo without a display. Exit status is 0 on `PASS` and 1 on `FAIL`:

```bash
ai-insar check
```

On the default seed the peak LOS velocity is near −0.22 m/yr and the result
is `PASS`. The printed lines are in [docs/cli.md](docs/cli.md). Clicks and
shortcuts are in [docs/quickstart.md](docs/quickstart.md).

```bash
ai-insar run --output products
ai-insar run --slc slc.npy --meta slc.json --output products
```

`--output` is required. Omitting `--slc` uses the built-in bowl.

## Features

Seven stage buttons, plus one top-level menu per technique. **Run all
stages** and `ai-insar check` use the Goldstein default, a 5×5 coherence
window, quality-guided unwrap, and SBAS. **Run** in another menu selects
that technique and computes the stage.

| # | Stage | Menu | What it does |
| --- | --- | --- | --- |
| 1 | Generate / load sample SLC | File | Synthetic C-band stack, or File → Load SLC stack (`.npy`) |
| 2 | Interferogram | Pipeline | Pairs with date-index gap 1…3, each `later * conj(earlier)` |
| 3 | Goldstein filter | Goldstein | Spectral filter. Light / Default / Strong set α to 0.3, 0.5, or 0.8 |
| 4 | Coherence | Coherence | Boxcar on the SLC pair, not on the filtered phase. Presets 3×3, 5×5, 7×7 |
| 5 | Unwrap | Quality-guided, Fast unwrap | Then a plane is removed on a 4 px border |
| 6 | Time series | SBAS, Stacking, NSBAS, PS linear, Phase linking, Seasonal | LOS displacement and a linear velocity. Negative is away from the radar |
| 7 | Displacement detection mask | Pipeline | Coherent pixels that are fast relative to the scene |
| | Export map | File, View | Velocity or one displacement date: `.npy`, a PNG preview, and a float TIFF when `tifffile` is installed |
| | Sentinel-1 SLC | File, Data | Open one SAFE zip, or Open SAFE stack for many zips; ASF search; one crop and swath into the stack |

Pipeline → Settings edits the date span, Goldstein α and window, coherence
window, unwrap coherence floor, and two detection thresholds. The sample
stack is 8 dates of 72×72, every 36 days from 2020-01-08, seed 7. The model
peak is 0.22 m/yr of subsidence, stored as −0.22 m/yr.

| Menu | What it does |
| --- | --- |
| Goldstein | Overlap-add of windowed FFTs. Default α = 0.5, 32 px window. Unwrapping uses this phase. |
| Coherence | Boxcar magnitude of later × conjugate(earlier), divided by the SLC powers. Filtering cannot raise it. |
| Quality-guided | Coherence-ordered unwrap. Numba runs the walk with `pip install "ai-insar[fast]"`; otherwise it is a Python heap. |
| Fast unwrap | Gaussian smooth, then `numpy.unwrap` along range and azimuth. Reimplemented from the MIT example in [InSARdev/edu](https://github.com/InSARdev/edu) `NISAR/nisar_numpy.py`. |
| SBAS | Coherence-weighted small-baseline inversion. Default for stage 6. |
| Stacking | Mean LOS rate of the unwrapped interferograms, weighted by coherence squared. |
| NSBAS | SBAS plus a light constraint that displacement follows one velocity. No seasonal term. |
| PS linear | Amplitude dispersion ≤ 0.4 or mean coherence ≥ 0.5, then the stacking velocity. Other pixels stay empty. |
| Phase linking | EMI principal eigenvector (`numpy.linalg.eigh`), then unwrapped pairs restore steps larger than π. |
| Seasonal | One velocity plus a one-year sine and cosine. A 3×3 solve per pixel. |

SBAS and NSBAS build the normal equations with NumPy `einsum` and
`numpy.linalg.solve`. This is a teaching set, not StaMPS, MintPy, or
InSAR.dev.

[InSARdev/core](https://github.com/InSARdev/core) (`insardev`,
InSARdev-SAL-1.0) is source-available. Its source stays upstream.
`pip install "ai-insar[insardev]"` opts into that package. Seasonal exists
because that project decomposes series with STL; this fit is a small
least-squares harmonic, written here. TOPS deramp, OpenCV coregistration,
OR-Tools branch cuts, and geocoding from
[InSARdev/edu](https://github.com/InSARdev/edu) `Sentinel-1/s1.py` need orbits
and a DEM, so this workbench leaves them out.

Stage inputs, outputs, and the default table: [docs/pipeline.md](docs/pipeline.md).

## Input

| | |
| --- | --- |
| dtype | `complex64` |
| shape | `(n_dates, height, width)` |
| time axis | index increases with time |

```text
ifg = slc[later] * conj(slc[earlier])
angle(ifg) = wrap(phase[later] - phase[earlier])
```

Line-of-sight displacement is **positive toward the radar**. A subsidence
bowl has **negative** LOS velocity. SLC phase is
`phase = -4π / wavelength × los_meters`, so that bowl has **positive**
unwrapped phase on a later-minus-earlier pair. Wrapped phase can still flip
sign after half a cycle. One fringe is `wavelength / 2`: about 2.77 cm at
the Sentinel-1 C-band wavelength used by the demo (5.405 GHz).

File layout: [docs/data-format.md](docs/data-format.md). The Help menu shows
the same sign convention.

## Sentinel-1 SLC

**File → Open SAFE stack…** multi-selects many Sentinel-1 `.zip` products.
They are not unzipped. A progress dialog counts the files. Every zip is
read with the same swath, polarisation, and crop (default IW1 VV, 256×256),
sorted by the manifest start time, and stored as one `complex64` stack
`(n_dates, height, width)`. With two or more dates the dialog can run the
interferogram stage immediately: each pair is later × conjugate(earlier).
The window is the same rows and columns in each measurement. That is
teaching crop-window alignment. There is no coregistration.

**File → Open Sentinel-1 SAFE zip…** opens one Sentinel-1 `.zip` without
extracting it. **Open SAFE folder…** and `manifest.safe` are optional.
The Product Explorer shows manifest metadata, IW bands (`i_`, `q_`,
`Intensity_`), and the quick-look. Double-click a band for a decimated
amplitude preview. **Add to SLC stack (crop…)** reads one window (default
IW1 VV) into the `complex64` stack. The date comes from the manifest start
time, and the wavelength from the annotation radar frequency when it is
present. This is open and crop only: no deburst, calibration, or
interferogram.

**File** and **Data** → **Fetch Sentinel-1 SLC…** search the public ASF
catalog. Search does not need a login. Import a crop from a local SAFE
zip, folder, measurement TIFF, or complex `.npy`. A downloaded zip is
opened directly and does not need to be extracted first.

Download needs an Earthdata account (`EARTHDATA_USERNAME` and
`EARTHDATA_PASSWORD`, or a `~/.netrc` machine `urs.earthdata.nasa.gov`) and
`pip install "ai-insar[slc]"`. Do not commit the password. A zip is often
several gigabytes; the dialog asks first. One band is read (IW1 VV unless
you change swath and polarisation), using the crop you set (default
256×256). The result is the same `complex64` stack. `ai-insar check` does
not contact ASF.

## Export map

The viewer draws LOS velocity and LOS displacement in blue–white–red. Blue
is motion away from the radar. The colorbar is labelled in m/yr or metres.
Nothing is geocoded.

**Export map**, File → Export velocity map, and File → Export displacement
map write:

- `velocity_m_yr.npy` or `los_m_<date>.npy` (the values)
- a PNG preview with a scale strip
- a float32 TIFF of the same array when `tifffile` is installed (`pip install "ai-insar[dev]"` or `pip install "ai-insar[slc]"`)

The TIFF has no CRS. File → Save products still writes the full cubes,
including every displacement date. See
[docs/data-format.md](docs/data-format.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [docs/install.md](docs/install.md) | venv, `pip install`, `--version` |
| [docs/quickstart.md](docs/quickstart.md) | GUI clicks, shortcuts, saving products |
| [docs/pipeline.md](docs/pipeline.md) | Each stage, defaults, what Settings changes |
| [docs/data-format.md](docs/data-format.md) | `slc.npy`, sidecar JSON, `products.json`, map export |
| [docs/cli.md](docs/cli.md) | `ai-insar --help`, exit codes, example `check` output |
| [docs/limitations.md](docs/limitations.md) | What the chain does not do |
| [WINDOWS.md](WINDOWS.md) | Release zip, `run_windows.bat`, and how to rebuild `dist/ai-insar-windows.zip` |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Tests and how to keep docs honest |

## Limitations

- One complex stack in memory, on the radar grid. No geocoding, DEM, orbit download, burst stitching, or weather model.
- A multi-zip SAFE stack uses one shared crop window when `Stack.load` is absent or fails. That crop is teaching alignment, not coregistration. A successful `Stack.load` replaces the crop with the InSARdev cube.
- Unwrap is quality-guided path following or a fast separable `numpy.unwrap`. Neither is SNAPHU.
- Time series is one linear LOS velocity. Seasonal adds a one-year sine and cosine. There is no earthquake step.
- The displacement mask is a coherence-gated robust test on the velocity image, not a trained model.
- Map export is the radar grid. The TIFF has no CRS.
- The demo bowl is synthetic. Do not quote its metres per year as a ground measurement.

The same boundary, written out: [docs/limitations.md](docs/limitations.md).

## Tests

There is no hosted CI badge. The suite runs locally with pytest:

```bash
pytest
```

`tests/test_pipeline.py` covers the interferogram definition, the LOS sign,
Goldstein reconstruction, multi-fringe unwrapping, SBAS, the demo bowl, and a
scene with no deformation. `tests/test_cli.py` runs `check` and `run`.
`tests/test_gui_smoke.py` clicks the seven stage buttons with
`QT_QPA_PLATFORM=offscreen` when PySide6 is installed. `tests/test_docs.py`
checks that links in these pages point at files in the tree.

## InSARdev

[InSARdev/edu](https://github.com/InSARdev/edu) is MIT. Fast unwrap follows
`NISAR/nisar_numpy.py`. Interferograms are the same complex product as that
file and `Sentinel-1/s1.py`: later × conjugate(earlier). Those files are
reimplemented, not copied. TOPS deramp, OpenCV coregistration, OR-Tools
branch cuts, and geocoding from `s1.py` are not included.

[InSARdev/core](https://github.com/InSARdev/core) `insardev` is
InSARdev-SAL-1.0. That license allows non-commercial redistribution only
when the copy stays under SAL. This repository is MIT, so those modules are
not vendored. `pip install "ai-insar[insardev]"` (Python 3.12+) installs
the upstream package. Installing or using it is your acceptance of SAL,
including the subscription for funded academic use. The same package is
`pip install insardev` if you install it yourself.

When that import succeeds, interferograms call
`Stack.pairs().interferogram()` first, with the later date as the reference
(single-look, no filter). That cube is the stage result even when the
values differ from `later × conjugate(earlier)`. NumPy/Numba runs only
when the package is missing, the call fails, or you set
`AI_INSAR_BACKEND=numpy`. `ai-insar check` scores the chain that actually
ran, so an installed `insardev` can change the bowl score. Set
`AI_INSAR_BACKEND=numpy` to score the in-tree product.

`Stack.load` is tried first for scene lists, including SAFE zips. A
returned cube is kept. If that call fails, a zip list still uses this
project's shared crop window, which is not coregistration. `Stack.align`
runs on a loaded stack when the method exists. `AI_INSAR_BACKEND=numpy`
skips both.

`insardev_toolkit` (BSD-3-Clause) comes in with the extra because upstream
depends on it. `PyGMTSAR.safeload` translates SAFE bursts when you pass
burst ids. `insardev_pygmtsar` is BSD-3-Clause and is not part of the
extra. Install it yourself if you want `S1.align_ref` and `S1.align_rep`.
Those calls need a burst directory, orbits, and a DEM. They are not copied
into this tree, and they do not resample the zip crop. ESA SNAP is GPL and
is not copied. Full wording: [NOTICE](NOTICE).

## License

MIT. See [LICENSE](LICENSE). Third-party notes: [NOTICE](NOTICE).

# AI Insar

AI Insar is a small open-source desktop app for interferometric SAR on a
**single-look complex (SLC) stack**. It stays in memory, on the radar grid,
and is meant for learning the processing chain and for experiments on small
stacks.

It is **not** a replacement for ISCE, MintPy, SNAP, GMTSAR, or SNAPHU.

The primary input is an SLC cube:

- dtype `complex64`
- shape `(n_dates, height, width)`
- date index increases with time

Interferograms are formed as **later × conjugate(earlier)**:

```text
ifg = slc[later] * conj(slc[earlier])
angle(ifg) = wrap(phase[later] - phase[earlier])
```

Line-of-sight (LOS) displacement is **positive toward the radar**. Subsidence
(motion away from the radar) is a negative LOS velocity. SLC phase is

```text
phase = -4 * pi / wavelength * los_meters
```

so a subsiding pixel has a **positive unwrapped** interferogram phase on a
later-minus-earlier pair. Wrapped phase can still change sign after half a
cycle. At the Sentinel-1 C-band wavelength used here (5.405 GHz, about
5.55 cm) one fringe is `wavelength / 2`, about 2.77 cm of LOS change.

No scene is downloaded. The GUI can build a synthetic C-band stack with a
known Gaussian subsidence bowl and run every stage on it.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # users who will not run pytest: pip install -e .
```

PySide6 is a normal dependency (PyQt6 is accepted if PySide6 is missing).
NumPy and SciPy are the only numeric dependencies.

## Run

Open the workbench:

```bash
python -m ai_insar
```

The same entry point is installed as `ai-insar` and `ai-insar gui`.

The window has a menu bar, seven numbered stage buttons, a viewer, and a log:

1. **Generate / load sample SLC** — synthetic C-band stack (File → Load SLC stack opens a `.npy` instead)
2. **Interferogram** — short-baseline pairs, `later * conj(earlier)`
3. **Filter** — Goldstein spectral filter
4. **Coherence** — boxcar coherence from the SLC pair (not from the filtered phase)
5. **Unwrap** — quality-guided unwrap, then a plane removed on the image border
6. **SBAS / time series** — displacement at each date and a linear LOS velocity
7. **Displacement detection mask** — coherent pixels whose velocity is large for this scene

The viewer shows SLC amplitude, wrapped phase, filtered phase, coherence,
unwrapped phase, LOS displacement, LOS velocity, the mask, and velocity with
the mask drawn on top. Click a pixel to read its value and, after SBAS, the
LOS time series in millimetres. Blue on the velocity scale is motion **away**
from the radar; red is motion **toward** the radar.

Pipeline → Settings changes the date span, Goldstein alpha and window,
coherence window, and detection thresholds. Run the stages again after
changing them.

Save products (File → Save products) writes NumPy arrays and `products.json`:
velocity, mask, LOS time series, coherence, and unwrapped phase.

### Headless commands

```bash
ai-insar check
ai-insar run --demo --output products
ai-insar run --slc slc.npy --meta slc.json --output products
python -m ai_insar check
```

`check` processes the built-in bowl, prints the peak velocity against the
known value, and exits 0 on `PASS`. It does not need a display. `run` is the
same chain without the window. With no `--slc`, `run` uses the synthetic stack.

## SLC file format

`slc.npy` is the complex cube. Optional `slc.json` next to it:

```json
{
  "wavelength_m": 0.055465763,
  "dates": ["2020-01-08", "2020-02-13"],
  "sensor": "C-band",
  "description": "optional note"
}
```

If the sidecar is missing, the loader assumes the Sentinel-1 C-band wavelength
and a 12-day repeat starting 2020-01-08, and says so in the log. Dates are
sorted so the index increases with time. Real-valued arrays are rejected.

There is no DEM, geocoding, or orbit file. The stack is already the radar-grid
cube you want to process.

## Sentinel-1 SLC from ASF

File → **Fetch Sentinel-1 SLC…** and the **Data** menu search the public ASF
catalog. That search does not need a login. Pick a bounding box and a date
range, then either:

- **Import files…** or **Import SAFE folder…** to read a crop from measurement
  GeoTIFFs or complex `.npy` planes you already downloaded, or
- **Download selected…** to fetch the SLC zip from ASF.

Download needs an Earthdata account. Put the login in the environment or in
`~/.netrc`. Do not commit it and do not paste it into the repository:

```bash
export EARTHDATA_USERNAME="your-earthdata-user"
export EARTHDATA_PASSWORD="your-earthdata-password"
```

```text
machine urs.earthdata.nasa.gov
login your-earthdata-user
password your-earthdata-password
```

Downloading also needs the optional packages (`asf_search` for the transfer,
`tifffile` and `zarr` to read a crop from a large measurement TIFF):

```bash
pip install "ai-insar[slc]"
```

A Sentinel-1 SLC zip is often several gigabytes. The dialog states the catalog
size and asks before it downloads. After the zip is extracted, import the SAFE
folder; the workbench reads one measurement raster per folder (IW1 VV when
several are present) and the crop you set (default 256×256 at the origin). A
window that starts at 0,0 and is larger than a small raster uses the whole
raster. The imported cube is `complex64` with shape `(dates, height, width)`
and then uses the same seven stages as the synthetic sample. It is not
geocoded.

The synthetic sample and File → Load SLC stack are unchanged. `ai-insar check`
still uses only the built-in bowl and does not contact ASF.

## Synthetic demo

`Generate / load sample SLC` builds eight 72×72 C-band acquisitions every
36 days (seed 7). A Gaussian bowl subsides at about 0.22 m/yr at its peak
(LOS velocity negative). A shared speckle component keeps temporal coherence
high, each date gets a random plane, and the rest of the chain has to remove
that plane and recover the bowl. The truth velocity is only available for
this synthetic stack; `ai-insar check` scores the recovery and does not
pretend the truth exists for a file you loaded.

## Processing notes

The small-baseline network keeps every pair whose date-index gap is at most
3, including the sequential pairs, so the SBAS design is full rank. The first
acquisition is the time reference (LOS = 0). Each unwrapped interferogram is
referenced by subtracting a plane fit on a border frame, which is where the
centered bowl is weak. SBAS then solves a displacement at each later date,
weighted by coherence squared, and fits one velocity per pixel.

The mask is a robust z-score of that velocity, gated by mean coherence and by
a fraction of the scene’s peak speed, with small connected components removed.
It flags both uplift and subsidence. It is not a trained neural network.

## Windows

[Download the Windows zip from Releases](https://github.com/sajadsarlak1361/AI-Insar/releases/download/v0.2.0-rc.1/ai-insar-windows.zip)
(`ai-insar-windows.zip` on the [v0.2.0-rc.1 prerelease](https://github.com/sajadsarlak1361/AI-Insar/releases/tag/v0.2.0-rc.1)).
That package includes the Sentinel-1 SLC menu and is not committed to git.
Unzip it and double-click `run_windows.bat`. It is a prerelease of the open
fetch pull request, not the stable `v0.1.0` tag on `main`.

See [WINDOWS.md](WINDOWS.md). The batch file creates a venv, installs, and
launches the GUI.

To build a source zip for another machine (dependencies are installed there,
not bundled):

```bash
python -m ai_insar.packaging
```

That writes `dist/ai-insar-windows.zip`. Unzip it and run `run_windows.bat`.

## Tests

```bash
pytest
```

The core tests cover the interferogram definition, the LOS sign, Goldstein
reconstruction, multi-fringe unwrapping, the SBAS inversion, the synthetic
bowl, and a scene with no deformation. `tests/test_gui_smoke.py` clicks the
seven stage buttons on the offscreen Qt platform when PySide6 is installed.

## Limitations

- Small in-memory stacks only. One polarization, radar coordinates, no
  multilook geometry, no geocoding, no DEM, no earth flattening, no ESD,
  no burst stitching.
- No orbit download, ionosphere, or weather model. The only atmospheric-style
  step is a per-interferogram plane estimated on the border.
- Unwrapping is quality-guided path following. Dense residues can leave
  2π islands. It is not SNAPHU or minimum-cost flow.
- SBAS estimates one displacement per date and then a single linear LOS
  velocity. There is no seasonal term, no step model, and no atmospheric
  filter in time beyond that velocity fit.
- The displacement mask is a statistical test on the velocity image, not a
  learned detector.
- The demo is a synthetic subsidence bowl, not a Sentinel-1 product. Do not
  treat its velocities as a survey.

## License

MIT. See [LICENSE](LICENSE).

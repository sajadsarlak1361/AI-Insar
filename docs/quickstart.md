# Quick start

Install first ([install.md](install.md)). These steps use the built-in
synthetic stack, so there is nothing to download.

## Headless, about ten seconds

```bash
python -m ai_insar check
```

The last line is `PASS` or `FAIL`. `PASS` means the estimated bowl has the
same sign as the truth, the 3×3 peak is within 40% of the truth, the
velocity correlation is at least 0.80, the core of the bowl is mostly inside
the mask, and the quiet border is mostly outside it. The printed numbers for
the default seed are copied in [cli.md](cli.md).

To keep the arrays:

```bash
python -m ai_insar run --output products
```

## Desktop

```bash
python -m ai_insar
```

The window title is **AI Insar**. On the left are seven stage buttons and
**Run all stages**. The centre is the viewer. The bottom is the log.

The Pipeline menu follows the GMTSAR stage order (PREPROCESS, ALIGN SLC,
MAKE TOPO_RA, MAKE INTERFEROGRAMS, UNWRAP PHASE, GEOCODE, then SBAS). The
drawing is in the README and in [pipeline.md](pipeline.md). **Run all
stages** does not align, remove topographic phase, or geocode.

Fastest path: click **Run all stages**. That generates the sample SLC if
none is loaded, then runs interferogram, filter, coherence, fast separable
unwrap, SBAS, and the mask. The viewer switches to **Velocity + mask**: a
blue bowl (motion away from the radar) with an orange mask on it. To use
another technique, open its menu (Stacking, NSBAS, PS linear, Phase linking,
Seasonal, or Quality-guided) and choose **Run**.

Step through it instead if you want to see each product:

1. **Generate / load sample SLC** shows SLC amplitude (grayscale).
2. **Interferogram** shows wrapped phase. The log says the pair was formed as later × conjugate(earlier).
3. **Goldstein filter** shows the filtered phase. The Goldstein menu can rerun it at α = 0.3, 0.5, or 0.8. The default demo logs zero wrapped residues.
4. **Coherence** shows the SLC-pair coherence (yellow/green in the default colormap). The Coherence menu can rerun it at 3×3, 5×5, or 7×7.
5. **Unwrap** shows continuous phase after the border plane is removed.
6. **SBAS / time series** shows LOS velocity in m/yr.
7. **Displacement detection mask** draws the mask in orange.

Click the middle of the bowl. The line under the image reports the row,
column, the value of the current layer, `LOS mm by date`, and `velocity`
in m/yr. For the default demo that velocity is negative (about −0.22).

The **View** combo and the **View** menu list the same layers: SLC
amplitude, interferogram phase, filtered phase, coherence, unwrapped phase,
LOS displacement, LOS velocity, detection mask, and velocity + mask. The
slider picks a date or a pair when the layer is a stack. LOS velocity and
LOS displacement use a blue–white–red scale (blue is away from the radar).
The colorbar shows the unit, m/yr or metres.

After stage 6, **Export map** writes `velocity_m_yr.npy`, a PNG preview, and,
when `tifffile` is installed, a float TIFF with no CRS. Export displacement
map writes the slider date when that layer is showing, otherwise the last
date, as `los_m_<date>.*`. See [data-format.md](data-format.md#map-export).

## Keyboard and menus

| Action | Menu | Shortcut |
| --- | --- | --- |
| Generate sample SLC | File | Ctrl+N |
| Load a `.npy` SLC stack | File → Load SLC stack… | Ctrl+O |
| Open a Sentinel-1 SAFE `.zip` (not extracted) | File → Open Sentinel-1 SAFE Product… | |
| Open many SAFE `.zip` files as one stack | File → Open SAFE stack… | |
| Open a `.SAFE` folder | File → Open SAFE folder… | |
| Save whatever stages have finished | File → Save products… | Ctrl+S |
| Export the velocity map (npy, PNG, TIFF) | File or View → Export velocity map… | |
| Export one displacement date | File or View → Export displacement map… | |
| Run stages 2–7 (and generate if needed) | Pipeline, or the blue button | Ctrl+R |
| Date span, filter, coherence, detection | Pipeline → Settings… | |

Loading a file clears products from the previous stack. Changing settings
does not rerun anything; run the affected stages again. The dialog says so.

File → Load expects a complex `.npy`. A JSON sidecar with the same stem
(`slc.npy` and `slc.json`) is picked up automatically. See
[data-format.md](data-format.md).

File → Open Sentinel-1 SAFE Product… and Open SAFE folder… inspect a
Sentinel-1 SLC SAFE product (zip, folder, or `manifest.safe`) in the
Product Explorer. Double-click a band for an amplitude preview. **Add to
SLC stack (crop…)** loads one IW1 VV window, by default, into the cube.
That crop is the stack the numbered stages run on. It is not a deburst or
a geocoded product. See [data-format.md](data-format.md#sentinel-1-safe).

File → Open SAFE stack… lists SAFE folders and `.zip` files (File Name,
Type, Acquisition, Track, Orbit, and an N Products counter). A progress
line counts them. The dialog’s crop is applied to every product, the dates
are sorted, and the interferogram checkbox runs stage 2 on that stack.
The alignment is the shared crop window. It is not coregistration.
Interferogram, Coherence, Goldstein, Topographic Phase, and Apply Orbit
File each open a parameter dialog and wait for Run.

## What “good” looks like on the demo

The log for the default sample includes:

- shape `(8, 72, 72)`, dtype `complex64`
- wavelength about `0.055466` m, sensor `synthetic C-band`
- peak LOS rate `-0.2200` m/yr (the model), seed 7
- 18 interferograms, temporal baselines 36–108 days
- a negative velocity at the bowl and a mask of hundreds of pixels

Those lines are produced by `run_all` in `ai_insar.pipeline.engine`. They
are a demo of the chain, not a survey measurement.

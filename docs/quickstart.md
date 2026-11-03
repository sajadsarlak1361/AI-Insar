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

Fastest path: click **Run all stages**. That generates the sample SLC if
none is loaded, then runs interferogram, filter, coherence, quality-guided
unwrap, SBAS, and the mask. The viewer switches to **Velocity + mask**: a
blue bowl (motion away from the radar) with an orange mask on it. To use
another technique, open its menu (Stacking, NSBAS, PS linear, Phase linking,
Seasonal, or Fast unwrap) and choose **Run**.

Step through it instead if you want to see each product:

1. **Generate / load sample SLC** shows SLC amplitude (grayscale).
2. **Interferogram** shows wrapped phase. The log says the pair was formed as later × conjugate(earlier).
3. **Filter** shows the Goldstein result. The default demo logs zero wrapped residues.
4. **Coherence** shows the SLC-pair coherence (yellow/green in the default colormap).
5. **Unwrap** shows continuous phase after the border plane is removed.
6. **SBAS / time series** shows LOS velocity in m/yr.
7. **Displacement detection mask** draws the mask in orange.

Click the middle of the bowl. The line under the image reports the row,
column, the value of the current layer, `LOS mm by date`, and `velocity`
in m/yr. For the default demo that velocity is negative (about −0.22).

The **View** combo and the **View** menu list the same layers: SLC
amplitude, interferogram phase, filtered phase, coherence, unwrapped phase,
LOS displacement, LOS velocity, detection mask, and velocity + mask. The
slider picks a date or a pair when the layer is a stack.

## Keyboard and menus

| Action | Menu | Shortcut |
| --- | --- | --- |
| Generate sample SLC | File | Ctrl+N |
| Load a `.npy` SLC stack | File → Load SLC stack… | Ctrl+O |
| Save whatever stages have finished | File → Save products… | Ctrl+S |
| Run stages 2–7 (and generate if needed) | Pipeline, or the blue button | Ctrl+R |
| Date span, filter, coherence, detection | Pipeline → Settings… | |

Loading a file clears products from the previous stack. Changing settings
does not rerun anything; run the affected stages again. The dialog says so.

File → Load expects a complex `.npy`. A JSON sidecar with the same stem
(`slc.npy` and `slc.json`) is picked up automatically. See
[data-format.md](data-format.md).

## What “good” looks like on the demo

The log for the default sample includes:

- shape `(8, 72, 72)`, dtype `complex64`
- wavelength about `0.055466` m, sensor `synthetic C-band`
- peak LOS rate `-0.2200` m/yr (the model), seed 7
- 18 interferograms, temporal baselines 36–108 days
- a negative velocity at the bowl and a mask of hundreds of pixels

Those lines are produced by `run_all` in `ai_insar.pipeline.engine`. They
are a demo of the chain, not a survey measurement.

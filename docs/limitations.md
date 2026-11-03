# Limitations

This is the same boundary the Help → Limitations dialog states, written out
so it can be read without the app. AI Insar is a small-stack desktop
workbench for learning the SLC chain. It is not a replacement for ISCE,
MintPy, SNAP, SNAPHU, GMTSAR, or Doris.

## Inputs and geometry

- One complex stack in memory, on the radar grid, one polarization.
- No SAFE/ZIP/CEOS reader, no DEM, no geocoding, no earth flattening, no
  enhanced spectral diversity, no burst stitching.
- No multilook that changes the array shape. The coherence window and the
  Goldstein window are the spatial averaging.
- Real arrays, non-finite values, and a date list whose length does not
  match the stack are rejected rather than guessed past.

## Propagation and atmosphere

- No orbit download and no ionospheric correction.
- No weather model. Each interferogram has a plane removed, fit on a border
  frame. A turbulent atmosphere that is not a plane remains in the velocity.
- The synthetic demo’s only “atmosphere” is a random plane per date, which
  that border fit is meant to remove.

## Unwrapping and time series

- Unwrapping is quality-guided path following. It is not SNAPHU and not a
  minimum-cost-flow unwrap. Dense residues can leave 2π islands. Low
  coherence pixels are barriers; a region the seed cannot reach stays NaN
  and drops out of SBAS.
- SBAS solves one displacement per date after the first, which is fixed at
  zero, then one linear LOS velocity. There is no seasonal term, no
  earthquake step, and no temporal atmospheric filter beyond that line fit.
- The network is “all pairs up to a date-index span” (default 3). It is not
  a perpendicular-baseline selector, because there is no orbit geometry.

## The mask

The mask is a robust z-score of velocity, a fraction of the scene’s own peak
speed, a coherence gate, and a minimum component size. It is not a neural
network and it has no weights file. Thresholds that are high enough for the
synthetic bowl can miss a slow, broad signal or keep a fast artifact in real
data. `ai-insar check` only scores the built-in bowl.

## The demo

The sample stack is a synthetic C-band subsidence bowl (seed 7), not a
Sentinel-1 product. Do not quote its metres per year as a ground measurement.
The pass/fail thresholds (40% velocity error, correlation 0.80, and the mask
recall/false-positive cuts in [cli.md](cli.md)) are a self-test of this
program, not an accuracy claim for an arbitrary SLC.

## Practical size

Stacks are dense NumPy arrays. A few dozen dates at a few hundred pixels on
a side is the intended size. The GUI runs stages on the UI thread; a much
larger cube will freeze the window until the stage returns. There is no
tiled or out-of-core mode.

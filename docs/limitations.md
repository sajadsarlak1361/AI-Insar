# Limitations

This is the same boundary the Help → Limitations dialog states, written out
so it can be read without the app. AI Insar is a small-stack desktop
workbench for learning the SLC chain. It is not a replacement for ISCE,
MintPy, SNAP, SNAPHU, GMTSAR, or Doris.

## Inputs and geometry

- One complex stack in memory, on the radar grid, one polarization.
- File → Open Sentinel-1 SAFE Product… (and Open SAFE folder…) reads a
  Sentinel-1 SLC SAFE zip, folder, or `manifest.safe` for metadata, a
  quick-look, and a window of one swath and polarisation (default IW1 VV).
  File → Open SAFE stack… reads many SAFE zips, without unzipping them,
  into one time-sorted `complex64` cube. The same crop, swath, and
  polarisation are used on every date. That is teaching crop-window
  alignment: the same rows and columns, not a coregistered resample.
  `pip install "ai-insar[insardev]"` can form the same interferogram when
  that package is installed, and `Stack.load` can read a geocoded zarr
  scene. The extra does not coregister or geocode the SAFE zip crop.
  Funded academic use of `insardev` needs a subscription.
  Data → Fetch Sentinel-1 SLC can search ASF and import that crop from a
  local zip, SAFE, measurement TIFF, or complex `.npy`. A downloaded zip
  does not have to be extracted first. Downloading needs Earthdata
  credentials and the optional `asf_search` package. There is still no DEM,
  no geocoding, no earth flattening, no calibration, no TOPS deburst, no
  enhanced spectral diversity, and no burst stitching. See
  [the InSARdev notes](../README.md#insardev).
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

- The Goldstein filter and the SLC-pair coherence estimate each have their
  own menu. Coherence is computed from the SLC dates, not from the filtered
  interferogram, so a stronger Goldstein alpha cannot raise it.
- The default unwrap is quality-guided path following. Fast unwrap is a
  Gaussian smooth plus separable `numpy.unwrap`. Neither is SNAPHU or a
  minimum-cost-flow unwrap. Dense residues can leave 2π islands. Low
  coherence pixels are barriers in the quality-guided walk; a region the
  seed cannot reach stays NaN and drops out of the time series.
- The default time series is SBAS: one displacement per date after the first,
  which is fixed at zero, then one linear LOS velocity. Stacking, NSBAS, PS
  linear, phase linking, and a one-year seasonal fit are separate menus.
  Seasonal is the only one with an annual term, and its velocity map is the
  linear coefficient, not a second fit of the series. There is no earthquake
  step and no MintPy-style atmospheric filter.
- File → Export velocity map and Export displacement map write `.npy`, a PNG
  preview, and a float32 TIFF when `tifffile` is installed. The TIFF is the
  radar grid. It has no CRS and no geotransform, so it is not a geocoded map.
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

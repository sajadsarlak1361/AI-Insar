# Spectral diversity and burst coregistration

The azimuth shift is in **pixels**. SNAP’s `azimuth_shift.txt` and the line
“Mean azimuth shift” are pixels. For the June/July IW1 pair the file’s applied
value is −0.001511 px.

## Double difference

For one burst overlap and one range block:

```
I_fwd = m_early · conj(s_early)
I_bwd = m_late  · conj(s_late)
Δ     = I_fwd · conj(I_bwd)
```

`m` is the reference and `s` is the secondary. `early` and `late` are the two
copies of the same ground line in the burst overlap.

The spectral separation, in hertz, is

```
Δf_DC(range) = kt(range) · (linesPerBurst − overlapLines) · Δt_az
kt = ka · ks / (ka − ks)
```

`ka` is the azimuth FM-rate polynomial, `ks` is the Doppler rate of the
antenna steering, and `Δt_az` is the azimuth line time. The polynomial is the
one for the burst that the split raster still indexes, which is the first
written burst when the annotation list was not cut down.

## Weight

Inv Quadratic, the SNAP default, is `sqrt(coherence)` of the early-look 5×5
interferogram when that coherence is above 0.3, and 0 otherwise. The 0.3 gate
is per pixel. It is not lowered. None, Linear, and Quadratic are the other
weights on the same coherence.

## Estimators

Average:

```
Δφ = arg(Σ w · Δ)
shift_px = Δφ / (2π · f̄ · Δt_az)
```

`f̄` is the weighted mean of `Δf_DC`.

Periodogram: search the shift that maximises `Σ w cos(Δφ − 2π Δf_DC shift Δt)`.
Nine candidates are refined until the step and the objective change are both
below 1e-4. With a constant `Δf_DC` this peak is the Average result.

The value written as the applied azimuth shift is the **negated** weight-mean
of the block periodograms. On SNAP’s own back-geocoded overlap that
reconstruction matches all ten block values to 3.3e-7 px and the applied value
−0.001511 px. On this build’s secondary the same estimator does not yet
reproduce block 2 (−0.000497 px here, −0.000727 px in the file). The secondary
was not replaced by a fitted tone.

## Applying the shift

Inside the common overlap, and separately in each burst, the secondary is
range-shifted while it is still chirped, then deramped. The azimuth ramp is
applied to that baseband and to the deramp phase, and the shifted phase is
put back. Pixels outside the overlap stay zero. The azimuth amount is the
negated weight-mean. The range amount is one 512×512 window at the centre of
each burst: zero-mean magnitude, a doubled FFT so the correlation does not
wrap, then a 32×32 box around the integer peak oversampled by 128. Bursts
below the coherence threshold or farther than one pixel are left out, and the
rest are averaged. A shift of −0.0625 px is eight bins of that 128× grid.

## Back-geocoding kernel

`BISINC_5_POINT_INTERPOLATION` is a separable 5-tap sinc with the raised-cosine
window

```
w(u) = sinc(u) · 0.5 · (1 + cos(π u / 2))    for |u| ≤ 2, else 0
```

Each axis is normalised after taps outside the image are dropped. The same
kernel, in float64, interpolates the deramp phase. An unwindowed sinc on that
phase was about ±10 rad and about 229 mm of azimuth bias. Eleven- and
21-point kernels use the same window with half-widths 5 and 10. Weights for
4096 fractional offsets are tabulated. The loop is Numba `prange` when Numba
imports, and the same arithmetic otherwise.

With a DEM, the secondary line and sample come from the DEM posts. Each post
is converted to ECEF at its ellipsoidal height, located in both images by
zero Doppler and slant range, and linearly interpolated onto the reference
grid. Range and azimuth are scaled by their pixel spacing before the
triangles are built. Pixels outside those posts are NaN and stay outside the
common overlap. Without a DEM the coordinate lattice still starts at an
8-pixel spacing and is kept when both axes are within the requested ground
tolerance (10 mm in the GUI). Azimuth time stays on the burst
`azimuthTime`.

## Network

For more than one pair, each overlap contributes
`(reference_index, secondary_index, shift_px, weight)`. Image 0 is fixed at
zero. L2 is weighted least squares of `x[secondary] − x[reference] = shift`.
L1 is a few reweighted passes. “L1 and L2” uses L1 only when there are more
pairs than unknowns. Pairs farther apart than the maximum temporal baseline
(default 4 images) are dropped. This is the small-baseline network equation.
MintPy is not imported and its source is not in this tree.

## Back-geocoding

Each reference sample is geolocated on the orbit, the DEM height is read, and
the ground point is updated three times before the secondary zero-Doppler
coordinate is solved. Azimuth time is the burst `azimuthTime` plus the
in-burst line times `Δt_az`. That burst time is the first line of the burst,
not the SAFE scene start and not `subset_offset`. The secondary is deramped
with its own `kt` and Doppler centroid, resampled, and reramped with the
secondary phase at the same source coordinate.

The GUI solves every pixel when the window is at most 400 000 pixels.
Larger windows still stop when check pixels are within 0.05 px of a dense
solve.

## Provenance

| Piece | Where it comes from | What this tree does |
| --- | --- | --- |
| Overlap interferograms, 0.3 coherence gate, periodogram, negated weight-mean | Sentinel-1 TOPS ESD as published by Prats-Iraola and as operated by SNAP Spectral Diversity | Reimplemented from those equations. No SNAP source. |
| `kt = ka·ks/(ka−ks)` and `Δf_DC` | TOPS steering-rate FM, the same relation used by ISCE2 topsApp’s overlap ESD | Reimplemented. ISCE2 is not vendored. |
| geo2rdr height iteration | The published range-Doppler iteration used by ISCE3 / COMPASS | Three DEM updates around the existing Newton zero-Doppler solver. ISCE3 and COMPASS are not vendored. |
| Block range windows and FFT cross-correlation | The same constant-shift correlation dolphin runs per block | `scipy.fft.rfft` plus a short DFT around the peak. dolphin is not vendored. |
| Network least squares | The SBAS network equation (Berardino; Yunjun, Fattahi, and others, the MintPy paper) | Written here. MintPy is GPL-3 and is not copied. If `mintpy` is installed it is only detected, not called for this shift. |
| InSARdev / PyGMTSAR | BSD-compatible, allowed as an optional package | `insardev_optional.py` calls `S1.align_ref` / `Stack.align` when the package imports. ESD block numbers stay on the implementation in this module so they can be compared with `azimuth_shift.txt`. |

Optional packages (`insardev_pygmtsar`, `isce3`, `dolphin`, `mintpy`, `compass`) are reported by `optional_libraries()` and are not required.

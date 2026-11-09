# 15-step coverage

Handwritten order. Dialog labels for steps 1–12 follow the empty-product
SNAP 14 audits. Steps 13–15 follow the final empty-product audit: Phase to
Displacement has a blank processing panel, Range-Doppler Terrain Correction
has the observed DEM and resampling lists, and View as Google Earth KMZ was
disabled in SNAP with no view open. Disabled or crashed SNAP controls were
not copied as reasons to disable this build.

A partial real pair is used for the scientific checks: IW1 VV only from
`S1A_IW_SLC__1SDV_20210630T052604_…_632B` and
`S1A_IW_SLC__1SDV_20210712T052605_…_C00B`, plus the two precise-orbit EOF
files. IW2, IW3, and VH were not downloaded. That is not a full SAFE and
not a full 15-step product. Geographic overlap of the two scenes is not a
coherence measurement.

| # | Step | What actually happens | Limit |
| --- | --- | --- | --- |
| 1 | S-1 TOPS Split | With a product bound, the subswath list starts blank and then offers that product's swaths. The target name ends in `_split`. Run reads `linesPerBurst` blocks. A short fixture raster is refused. A folder annotation is stored as a radar window. | A zip annotation is not unpacked into a radar grid. Overlap stays in the lines. |
| 2 | Apply Orbit File | Sentinel Precise and Restituted download a matching ESA POEORB or RESORB file when the source is a SAFE. The platform and the validity window must cover the acquisition, and the EOF speed must be a low-Earth orbit. A local EOF still works. | ENVISAT and the other listed types are not downloaded. |
| 3 | S-1 Back Geocoding | Two split windows are deramped, located by zero-Doppler range on the selected DEM or the ellipsoid, resampled, and reramped. A coarse line/sample map is kept only when check pixels stay within 0.05 pixels of a dense solve. | The multi-SAFE crop is not this operator. |
| 4 | S-1 Enhanced Spectral Diversity | Average is the coherence-weighted double difference. Periodogram is the unweighted single-tone peak. Coherence below the threshold raises and does not apply a shift. A supplied azimuth shift is a Fourier ramp on the secondary. | The loaded windows must contain both burst looks. A 12-day overlap can be rejected for low coherence. |
| 5 | Interferogram Formation | The complex product is later × conjugate(earlier). Checked flat-earth or topographic subtraction removes `−4π/λ (R_later − R_early)` when both radar windows and both precise orbits are on the workspace. Topography uses one DEM height, not a downloaded Copernicus tile. | Residual ramp is still refused. The polynomial degree does not replace the geometric ranges. |
| 6 | S-1 TOPS Deburst | Run cuts each burst at the mid azimuth time of the overlap and drops invalid edge lines. | It needs the split radar window. It does not load a measurement by itself. |
| 7 | Multilooking | Independent looks average whole cells. GR Square Pixel, when a swath model is stored, sets azimuth looks from ground range and azimuth spacing. | Without that model, GR Square Pixel still stops. |
| 8 | Goldstein Phase Filtering | Constant alpha and adaptive block coherence use the in-tree FFT. A coherence mask zeros samples below the threshold. | A non-zero metre block extent is refused. Adaptive alpha needs a coherence image. |
| 9 | Subset and Save Product As | Subset crops the in-memory arrays and radar windows. The source file is not modified. Save writes NumPy, GeoTIFF (with a world file when bounds exist), or KMZ. | BEAM-DIMAP is refused. SNAP's subset spinner exception is not reproduced. The native Save chooser after the BEAM-DIMAP question was not observed. |
| 10 | Snaphu Export | Writes float32 ENVI phase and coherence plus `snaphu.conf` for pair 0. | A missing coherence image is not filled in. Later pairs are not exported. |
| 11 | Batch Snaphu Unwrap | Runs the installed `snaphu` binary. The configuration holds the line length; the command passes the phase, coherence, and output files. | Fast unwrap is not called. Very narrow grids are rejected by snaphu itself. |
| 12 | Snaphu Import | Reads `unwrap.img` when the ENVI header matches the export metadata. | It does not unwrap again. |
| 13 | Phase to Displacement | The processing panel is blank. Run applies `d = −(λ/4π) Δφ` to unwrapped radians. Wavelength comes from the SLC. Positive metres are toward the radar. | A raw SAFE is refused. |
| 14 | Range-Doppler Terrain Correction | Copernicus 30 m uses the EGM2008 5-minute geoid. SRTM uses the EGM96 15-minute geoid. A mismatched geoid raises and no ellipsoidal height is computed. Each output coordinate is sampled. Ellipsoid height 0 is a separate mode. | Other DEM names are refused. Copernicus GeoTIFFs need `imagecodecs`. |
| 15 | KMZ | File → View as Google Earth KMZ writes a zipped KML `GroundOverlay` and a PNG. Bounds come from terrain correction. | The SNAP menu item was disabled, so this is not a copy of an unobserved SNAP dialog. |

`tests/test_real_s1_chain.py` still uses 24×24 samples. `tests/test_dem_orbit_overlap.py` uses the burst-4/5 overlap of IW1 VV: 166 lines at absolute rows 7374:7540 and 7540:7706, 24 range samples from column 11000. On that window the SRTM ellipsoidal heights span about 495–699 m (standard deviation about 55 m). A stride-2 range-Doppler map stayed within 0.05 pixels of the dense check samples. The 12-day double-difference coherence on a 166×128 window was 0.012, so ESD refused to apply a shift. That rejection is not treated as a failed threshold, and an injected shift is not the scene result. `snaphu` is given `FLOAT_DATA`; a 0–10 rad ramp unwraps to about 1e-5 rad. Nine bursts are not processed.

## Acceptance checklist

SNAP 12.0.0 GPT is at `/home/ubuntu/esa-snap/bin/gpt`. The menu audit was SNAP 14. `linesPerBurst` is 1508. SNAP burst numbers are 1-based: bursts 5 and 6 are absolute rows 6032:9048. The same bursts are 0-based indexes 4 and 5. Their overlap is rows 7374:7540 and 7540:7706. A GPT subset of bursts 5–6, samples 11000:11512, matches this reader with maximum absolute difference 0. That identity is only the SLC read, not Back-Geocoding or terrain correction. Deramp then reramp on that window leaves a complex64 residual below 0.001. On the 166×128 overlap the 12-day ESD coherence was 0.012 and the shift was not applied.

`TOPSAR-Split` on this partial SAFE was refused: `Source product should be multi sub-swath SLC burst product`, because IW2, IW3, and VH were not downloaded. The pixel comparison uses GPT `Subset` on the IW1 VV measurement.

Copernicus is not converted with EGM96. Sentinel POEORB download for this S1A acquisition is checked against the local EOF. The 12-day overlap is rejected at coherence 0.25; that threshold is not lowered, and an injected shift is not counted as this pair's ESD result.

BEAM-DIMAP is written. BMP and the other unused writer names still refuse. This is not a nine-burst, three-swath product.

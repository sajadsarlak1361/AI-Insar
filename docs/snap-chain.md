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
| 2 | Apply Orbit File | The eight SNAP orbit types are listed. Auto-download is not installed. A local `.EOF` or `.EOF.zip` replaces the annotation state vectors and logs the metre shift. With the path empty, Run still refuses unless the fail-safe box is checked, and then the vectors stay. | No orbit client. The June precise file moves the annotation state by a few metres on the real IW1 burst. |
| 3 | S-1 Back Geocoding | Two split windows and two orbits are deramped, located by zero-Doppler range on the ellipsoid, bilinearly resampled, and reramped. The secondary acquisition time is the zero-Doppler guess. | DEM auto-download is not fetched. Height is 0 unless an external DEM mean is supplied. Windows are solved on a stride once they exceed 4096 range-Doppler solves. Resampling other than bilinear is refused. The multi-SAFE crop is not this operator. |
| 4 | S-1 Enhanced Spectral Diversity | Periodogram Run estimates one azimuth shift from the burst-overlap double difference. The spectral separation is the azimuth FM rate times the burst interval. The same image against itself is a zero shift. | Average cross-correlation is refused. The loaded windows must still contain both burst looks. A near-zero shift is not coherence. |
| 5 | Interferogram Formation | The complex product is later × conjugate(earlier). Checked flat-earth or topographic subtraction removes `−4π/λ (R_later − R_early)` when both radar windows and both precise orbits are on the workspace. Topography uses one DEM height, not a downloaded Copernicus tile. | Residual ramp is still refused. The polynomial degree does not replace the geometric ranges. |
| 6 | S-1 TOPS Deburst | Run cuts each burst at the mid azimuth time of the overlap and drops invalid edge lines. | It needs the split radar window. It does not load a measurement by itself. |
| 7 | Multilooking | Independent looks average whole cells. GR Square Pixel, when a swath model is stored, sets azimuth looks from ground range and azimuth spacing. | Without that model, GR Square Pixel still stops. |
| 8 | Goldstein Phase Filtering | Constant alpha and adaptive block coherence use the in-tree FFT. A coherence mask zeros samples below the threshold. | A non-zero metre block extent is refused. Adaptive alpha needs a coherence image. |
| 9 | Subset and Save Product As | Subset crops the in-memory arrays and radar windows. The source file is not modified. Save writes NumPy, GeoTIFF (with a world file when bounds exist), or KMZ. | BEAM-DIMAP is refused. SNAP's subset spinner exception is not reproduced. The native Save chooser after the BEAM-DIMAP question was not observed. |
| 10 | Snaphu Export | Writes float32 ENVI phase and coherence plus `snaphu.conf` for pair 0. | A missing coherence image is not filled in. Later pairs are not exported. |
| 11 | Batch Snaphu Unwrap | Runs the installed `snaphu` binary. The configuration holds the line length; the command passes the phase, coherence, and output files. | Fast unwrap is not called. Very narrow grids are rejected by snaphu itself. |
| 12 | Snaphu Import | Reads `unwrap.img` when the ENVI header matches the export metadata. | It does not unwrap again. |
| 13 | Phase to Displacement | The processing panel is blank. Run applies `d = −(λ/4π) Δφ` to unwrapped radians. Wavelength comes from the SLC. Positive metres are toward the radar. | A raw SAFE is refused. |
| 14 | Range-Doppler Terrain Correction | Output latitudes and longitudes are converted to ECEF and located in radar coordinates, then bilinearly resampled. The real IW1 window lands inside the Germany footprint. | Auto-download DEMs are not fetched. One external-DEM mean height, or ellipsoid height 0, is used for the whole window. Output is WGS84 geographic only, and the grid is capped. |
| 15 | KMZ | File → View as Google Earth KMZ writes a zipped KML `GroundOverlay` and a PNG. Bounds come from terrain correction. | The SNAP menu item was disabled, so this is not a copy of an unobserved SNAP dialog. |

The real-data check in `tests/test_real_s1_chain.py` reads 24×24 IW1 VV samples from both dates, checks the precise-orbit shift, the deburst line plan, flat-earth phase, range-Doppler resampling, square-pixel looks, Goldstein, a real `snaphu` unwrap, the displacement sign, inverse terrain correction, KMZ, and the burst-overlap spectral separation. It does not process nine bursts or both polarisations.

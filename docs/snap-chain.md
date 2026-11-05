# 15-step coverage

Handwritten order only. Batch 3 of the installed SNAP 14 GUI (empty Product
Explorer, nothing run) is the source for steps 1–6. Steps 7–15 were not
opened in that pass. Defaults that were not on screen are not filled in.

No Sentinel-1 scene in this tree is large enough for an end-to-end split:
the reduced fixture raster is smaller than `linesPerBurst`, and the reader
refuses it. Nothing here was checked against a full SLC.

| # | Step | What actually happens | Evidence |
| --- | --- | --- | --- |
| 1 | S-1 TOPS Split | Dialog matches the empty-product observation (blank subswath, empty polarisations, burst text `1 to 9999 (max number of bursts: 9999)`). After a SAFE product is bound, subswath, polarisation, and burst count come from that product. Run reads `linesPerBurst` blocks for the chosen bursts. It refuses a raster that is smaller than that window. Overlap stays in the lines. | Batch 3 GUI. Annotation `linesPerBurst`. |
| 2 | Apply Orbit File | Dialog lists the eight orbit types that were on screen, polynomial degree text `3`, and the unchecked fail-safe box. Run raises because no orbit file is downloaded. With the fail-safe box checked, state vectors are still not changed. | Batch 3 GUI. No orbit client in this tree. |
| 3 | S-1 Back Geocoding | Graph dialog: ProductSet-Reader, Back-Geocoding, Write; buttons Save, Help, Validate, Run. Empty set shows `Error: [NodeId: ProductSet-Reader] Please add a source product`. DEM, both resampling lists, and the three checkboxes use the recorded defaults. Run does not resample. The multi-SAFE crop is a different command and is not this operator. | Batch 3 GUI. No deramp, range-Doppler, or DEM download. The audit did not test whether one product is enough. |
| 4 | S-1 Enhanced Spectral Diversity | Single-source dialog, label `source:`, 18 processing controls at the recorded defaults. Run stops. No azimuth shift is estimated. | Batch 3 GUI. No burst-overlap estimator. |
| 5 | Interferogram Formation | Single-source dialog, label `Source product:`. Flat-earth degree 1–8 (default 5), estimation points through 1001 (default 501), orbit degree 1–5 (default 3). Flat-earth subtraction is checked, and Run is refused until it is unchecked. The complex product is then later × conjugate(earlier). DEM text is the recorded Copernicus name and stays disabled; its option list was not opened. | Batch 3 GUI. In-tree interferogram. No orbit polynomial for flat-earth. |
| 6 | S-1 TOPS Deburst | Single-source dialog. The only processing control is `Polarisations:`, empty until a product is bound. Run stops. The measurement raster is not returned as a deburst. | Batch 3 GUI. Overlap merge is not implemented. |
| 7 | Multilooking | Not opened. No dialog was added. | Batch 3: path not verified. |
| 8 | Goldstein Phase Filtering | An older dialog still runs the in-tree Goldstein filter (alpha, FFT size, spectral window). It was not re-opened in batch 3, so it is not claimed as a new GUI match. | Earlier study, not batch 3. |
| 9 | Raster Subset + Save Product As | Not opened. The SAFE stack crop is a separate reader, not this menu. | Batch 3: Raster and File were not opened for this. |
| 10 | Snaphu Export | Not opened. No SNAPHU container is written. | Batch 3: Unwrapping was only a submenu label. |
| 11 | Snaphu Unwrapping | Not opened. Fast unwrap stays on its own menu and is not SNAPHU. | External `snaphu` is not executed. |
| 12 | Snaphu Import | Not opened. | Depends on export and an external unwrap. |
| 13 | Phase to Displacement | Name was on the Products flyout. Dialog not opened. LOS metres inside the time-series stage use `d = −(λ/4π) Δφ`. | Batch 3 menu label only. |
| 14 | Range-Doppler Terrain Correction | Not opened. Products stay on the radar grid. | No range-Doppler lookup. |
| 15 | KMZ Export | Not located. Nothing is written. | Batch 3. |

Multi-SAFE input is unchanged: several folders and zip files can be selected
together, listed with type, acquisition, track, and orbit, and all of them
stay in the SLC stack. That stack is a shared crop. It is not Back Geocoding.

Save as on the batch-3 dialogs starts on BEAM-DIMAP, which this build does
not write. Choose NumPy (.npy) or uncheck Save as. NumPy was not in the
partial SNAP writer list. Uncertain writer spellings from that scroll were
left out.

# 15-step coverage

The handwritten order below is the only SNAP chain this page tracks.
Other Radar operators stay out of this list. Evidence is named in the
last column. "GUI-observed" means the SNAP 14 dialog was opened in the
menu study already in this project. Defaults that were not seen are not
filled in.

The multi-product input is implemented: several SAFE folders and zip
files are selected together, listed with type, acquisition, track, and
orbit, and all of them stay in the SLC stack. That load is a shared
crop. It is not Back Geocoding.

| # | Step | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Sentinel-1 TOPS Split | Burst first/last is checked against each product and stored. The array read is still the shared crop, not a subswath/burst stitch. | GUI study: subswath, polarisation, and burst controls exist and stay empty until a product is chosen. Burst line stitching is not implemented. |
| 2 | Apply Orbit File | Dialog opens and validates orbit type and polynomial degree. State vectors are not changed. | GUI-observed defaults: Sentinel Precise (Auto Download), degree 3, "do not fail" unchecked. No orbit download in this tree. |
| 3 | Coregistration / Back Geocoding | Blocked. Every selected product is kept in the crop stack. That is not DEM-assisted coregistration. | GUI-observed Back Geocoding controls are not wired, because this build has no deramp/resample/reramp. |
| 4 | Enhanced Spectral Diversity | Blocked. | Not opened in the GUI study. No ESD estimator in this tree. |
| 5 | Interferogram Formation | Implemented. Run forms later × conjugate(earlier) on the full stack. | GUI-observed dialog. In-tree interferogram. |
| 6 | TOPS Deburst | Blocked. | GUI-observed polarisation list only. No burst stitch. |
| 7 | Multilooking | Blocked. | Help describes square-pixel and independent looks. Spinner defaults were not captured, so none were invented. |
| 8 | Goldstein Phase Filtering | Implemented. Alpha, FFT size, and spectral window are applied. | GUI-observed dialog. In-tree Goldstein filter. |
| 9 | Raster Subset + Save Product As | Partial. The stack dialog's row/column crop is the subset that is read. File → Save products writes the workspace arrays. | SNAP Raster Subset and Save Product As dialogs were not opened in the GUI study. |
| 10 | Snaphu Export | Blocked. | Help describes the export. The dialog was not opened. This build does not write a SNAPHU container. |
| 11 | Snaphu Unwrapping | Blocked. Fast unwrap is a different menu and is not SNAPHU. | External `snaphu` is not run. A missing binary is not replaced with the fast unwrap. |
| 12 | Snaphu Import | Blocked. | Depends on export and an external unwrap. |
| 13 | Phase to Displacement | Equation only. LOS metres use `d = −(λ/4π) Δφ` inside the time-series stage. | SNAP Help equation. No separate dialog yet. |
| 14 | Range-Doppler Terrain Correction | Blocked. Products stay on the radar grid. | No range-Doppler lookup. A synthetic lat/lon grid is not written. |
| 15 | KMZ Export | Blocked. | Needs geocoded coordinates this build does not have. |

Incompatible products (different mode, different product type, or a missing
swath/polarisation) are rejected by name before any pixel is read. A
different relative orbit is reported and every product is still kept.

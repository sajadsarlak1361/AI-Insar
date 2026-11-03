# Data format

AI Insar processes one SLC cube and optional JSON. A Sentinel-1 SAFE product
can be opened for inspection, and many SAFE zips can be cropped into that
cube as one date each. The cube is the radar grid that is processed. CEOS,
a DEM, and orbit files are not read. Opening a SAFE product does not
deburst or calibrate. Interferograms are a later pipeline stage on the
cube, not part of opening the zip.

## Sentinel-1 SAFE

File → **Open Sentinel-1 SAFE zip…** takes the product `.zip` first. The
zip is not extracted. A `manifest.safe` file is an optional extra in the
same dialog, and File → **Open SAFE folder…** is the optional folder path.
The zip name must start with `s1`. ETAD (`_ETA_`) and COG (`_COG`) products
are refused.

Open finds the one `manifest.safe` and uses its directory as the zip root
(`S1A_….SAFE/`). Metadata comes from that manifest. Bands are the files
directly in `annotation/` and `measurement/`, paired by basename
(`.xml` with `.tiff`). The manifest `dataObjectSection` is not used to find
files. `annotation/calibration/`, `support/*.xsd`, and the product PDF are
ignored. Width and height are `numberOfSamples` and `numberOfLines` in the
annotation. The TIFF header is not read at open.

The Product Explorer lists manifest metadata, bands grouped by swath
(IW1, IW2, IW3) with `i_`, `q_`, and `Intensity_` labels, and the quick-look
when `preview/quick-look.png` is present. Double-click a band for a
decimated amplitude preview. **Add to SLC stack (crop…)** reads one window
into the `complex64` cube. The default band is IW1 VV. The date is the
manifest start time. The wavelength is `c / radarFrequency` from the
annotation when that value is present.

File → **Open SAFE stack…** (also under Data) takes many of those zips, or
a folder that contains them. A `.SAFE` folder is optional. The same swath
and polarisation (default IW1 VV) and the same pixel window are read from
each product. Dates are the manifest start times, sorted so the index
increases with time. A product that cannot supply that band, or whose crop
does not match the others, is skipped. The cube is `complex64`,
`(n_dates, height, width)`. Pipeline → Interferogram and Run all stages
then form later × conjugate(earlier) on that cube, with the usual date-gap
setting. The products are not coregistered: the window is the same rows
and columns in each file, which is a teaching crop, not a TOPS
interferogram.

Pixels are read later, in a window, with optional subsampling. An SLC
sample packed in one int32 uses the low 16 bits as I and the high 16 bits
(`>> 16`) as Q. An uncompressed measurement TIFF is read by seeking to that
zip entry. It is not extracted. A compressed entry extracts that one TIFF
only. The quick-look PNG is read from `preview/quick-look.png` in memory.
Install the optional reader with `pip install "ai-insar[slc]"`.

File → **Fetch Sentinel-1 SLC…** can import that same zip, folder, or
measurement TIFF. A zip downloaded from ASF does not have to be unzipped
first.

## `slc.npy`

| | |
| --- | --- |
| Array | NumPy `.npy` |
| dtype | complex (`complex64` after load; real arrays are rejected) |
| shape | `(n_dates, height, width)` |
| values | finite; NaN or infinite values raise `ValueError` |
| axis 0 | time. The loader sorts dates so the index increases |

`validate_slc` in `ai_insar.io_stack` is the check both the GUI loader and
`SLCStack` use. At least one date and a positive height and width are
required. The synthetic generator itself asks for at least two dates and
an image of at least 8×8.

## Sidecar JSON

Default path: the same stem as the `.npy`, with suffix `.json`
(`scene.npy` → `scene.json`). On the command line, `--meta` points somewhere
else. The GUI loader does not take a separate meta path; it only looks
beside the file.

Keys that are read:

| Key | Role if present |
| --- | --- |
| `wavelength_m` | Positive wavelength in metres. Default: Sentinel-1 C-band, `C_BAND_WAVELENGTH_M` (≈ 0.055465765 m). |
| `dates` | List of calendar dates, one per SLC. Parsed as `datetime64[D]`. |
| `sensor` | Stored on the stack. Default `unknown` when the file exists but omits it. |
| `description` | Free text. Notes about sorting or a missing date list are appended. |

`schema_version`, `shape`, `dtype`, `interferogram`, and `los_sign` are
written by `save_slc` / `metadata_dict` so a saved stack documents itself.
The loader does not require them.

Example of a minimal sidecar:

```json
{
  "wavelength_m": 0.055465765,
  "dates": ["2020-01-08", "2020-02-13", "2020-03-20"],
  "sensor": "C-band",
  "description": "three radar-grid SLCs"
}
```

If the sidecar is missing, the loader assumes that C-band wavelength and a
12-day repeat starting 2020-01-08 (`default_dates`). It says so in
`description`, which the GUI prints in the log. If `dates` is missing but
the file exists, the same 12-day assumption is used. The number of dates
must match axis 0. Dates must be strictly increasing after sorting; a
duplicate day raises `ValueError`.

Interferograms always use the sorted order: the larger index is the later
date.

## Products directory

File → Save products, or `ai-insar run --output <dir>`, calls
`save_products`. A file is written only when that stage has been run.
Names and shapes:

| File | Shape | dtype / meaning |
| --- | --- | --- |
| `velocity_m_yr.npy` | `(height, width)` | metres per year, negative = away from the radar |
| `mask.npy` | `(height, width)` | `uint8`, 1 = detected displacement |
| `los_m.npy` | `(n_dates, height, width)` | metres, date 0 is 0 |
| `mean_coherence.npy` | `(height, width)` | mean of the pair coherences |
| `coherence.npy` | `(n_pairs, height, width)` | per-pair coherence in `[0, 1]` |
| `unwrapped_phase.npy` | `(n_pairs, height, width)` | radians after the border plane removal; NaN where not unwrapped |
| `interferogram.npy` | `(n_pairs, height, width)` | `complex64`, later × conj(earlier) |
| `filtered_interferogram.npy` | `(n_pairs, height, width)` | Goldstein output, `complex64` |
| `truth_velocity_m_yr.npy` | `(height, width)` | only if the stack carries synthetic truth |
| `truth_mask.npy` | `(height, width)` | `uint8`, only for that synthetic truth |
| `products.json` | | index of the files, the stack metadata, the pairs, and the config that was saved |

In memory the mask is boolean. On disk it is `uint8`.

`products.json` (`schema_version` 1) has this shape. The block below is a
two-date illustration of the keys, not the eight-date demo (that run writes
18 pairs and `shape` `[8, 72, 72]`):

```json
{
  "schema_version": 1,
  "stack": {
    "schema_version": 1,
    "wavelength_m": 0.055465765,
    "dates": ["2020-01-08", "2020-02-13"],
    "sensor": "synthetic C-band",
    "description": "",
    "shape": [2, 32, 32],
    "dtype": "complex64",
    "interferogram": "later * conj(earlier)",
    "los_sign": "positive toward the radar"
  },
  "files": {
    "velocity_m_yr": "velocity_m_yr.npy",
    "mask": "mask.npy"
  },
  "pairs": [[0, 1]],
  "config": {
    "max_date_span": 3,
    "goldstein_alpha": 0.5,
    "goldstein_window": 32,
    "coherence_window": 5,
    "unwrap_min_coherence": 0.2,
    "detection_sigma": 3.5,
    "detection_peak_fraction": 0.28,
    "detection_min_coherence": 0.3,
    "detection_min_pixels": 12,
    "timeseries_method": "sbas",
    "unwrap_method": "quality"
  }
}
```

`pairs` entries are `[earlier, later]`. The config block is the subset
`save_products` writes. It includes `timeseries_method` (`sbas` by default)
and `unwrap_method` (`quality` or `separable`). Overlap, the ramp border,
the Fast-unwrap sigma, and the PS / NSBAS weights are not in the file.
See [pipeline.md](pipeline.md).

There is no reader that reloads `products.json` back into a workspace. The
`.npy` files are ordinary NumPy arrays.

## Map export

File → Export velocity map, File → Export displacement map, and the
**Export map** button write a different trio into a folder you choose. They
do not replace Save products.

| File | Contents |
| --- | --- |
| `velocity_m_yr.npy` | float64 LOS velocity, metres per year, same values as the viewer |
| `velocity_m_yr.png` | RGB preview of that map plus a blue-to-red scale strip |
| `velocity_m_yr.tif` | float32 TIFF of the same array, only when `tifffile` is installed |
| `los_m_<date>.npy` | one displacement date, metres. The date is the slider date when the viewer is on LOS displacement, otherwise the last date |
| `los_m_<date>.png` | preview of that date, scaled with the same limits as the displacement viewer |
| `los_m_<date>.tif` | float32 TIFF of that date, same condition as the velocity TIFF |

The TIFF description says the array is a radar-grid map with no CRS and no
geotransform. Blue in the PNG is motion away from the radar. The `.npy`
file is the data; the PNG is only a preview.

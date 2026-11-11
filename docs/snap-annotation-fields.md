# Annotation fields the TOPS operators read

The table is the Sentinel-1 measurement annotation (`annotation/*.xml`),
plus the fields calibration and noise supply only when an operator asks for
them. SNAP's TOPS Split keeps this list intact and restarts burst numbering
at zero, so a product that retained original bursts 5 and 6 addresses
records 0 and 1.

| Field | SNAP read and index | AI Insar |
| --- | --- | --- |
| `azimuthFmRateList` (`azimuthTime`, `t0`, `azimuthFmRatePolynomial`) | `Sentinel1Utils.getAzimuthFmRateList` keeps every record. `computeRangeDependentDopplerRate` uses `list[burst]`, and `burst` is the index in the current product. | All records are stored in order. Deramp and reramp use that same split-local index (`_split_polynomial`). Nearest-time lookup remains available and is not the operator path. |
| `dcEstimateList` (`azimuthTime`, `t0`, `dataDcPolynomial` or `geometryDcPolynomial`) | `dataDcPolynomial` when `dcMethod` contains "Data Analysis", otherwise the geometry polynomial. If the list is at least as long as the burst count, burst `b` reads record `b`. A shorter list is interpolated to the burst centre. | The same polynomial choice. Deramp uses record `b` of the untrimmed list. |
| `azimuthSteeringRate` | Degrees per second in the annotation, converted with `DTOR` once, then `ks = 2 v steering / λ`. `v` is the orbit speed at the midpoint of the product line times. | Stored in rad/s. The same `ks` uses the speed at `(productFirstLineUtcTime + productLastLineUtcTime) / 2`. |
| `azimuthTimeInterval` | `imageInformation`, seconds, one value for the subswath. | Same element, used for burst line time and the deramp `η`. |
| `rangeSamplingRate` | Hertz, from `productInformation`. Slant-range time along a line is still `slantRangeTime/2 + pixel * rangePixelSpacing / c`. | Stored. The two-way time passed to the FM and DC polynomials is `slantRangeTime + 2 * pixel * rangePixelSpacing / c`, which is that one-way time doubled. |
| `slantRangeTime` | Two-way seconds to the first range sample. Divided by two before a pixel offset is added. | Stored as two-way seconds and used in that same two-way expression. |
| `radarFrequency` | Hertz. Wavelength is `c / frequency`. | Same. |
| `linesPerBurst`, `samplesPerBurst` | `swathTiming`. Burst `b` occupies lines `[b*linesPerBurst, (b+1)*linesPerBurst)`. | Same. Deburst and the deramp line index use that blocking. |
| Burst `azimuthTime`, `firstValidSample`, `lastValidSample` | One azimuth time per burst. The valid-sample vectors are per line; `-1` is empty. `firstValidPixel` is the maximum valid `firstValidSample` in the subswath, and `lastValidPixel` is the minimum valid `lastValidSample`. | Burst time and the first and last valid line are stored, plus the first valid line's range sample. `first_valid_pixel` is that same subswath maximum and is the deramp anchor sample. |
| `geolocationGrid` | `line`, `pixel`, latitude, longitude, `height`, `incidenceAngle`, `elevationAngle`. Tie points for the first geographic estimate. | `line`, `pixel`, height, and incidence are stored in file order. |
| `incidenceAngleMidSwath` | Degrees, image information. | Stored in radians. |
| `rangePixelSpacing`, `azimuthPixelSpacing` | Metres, image information. Square-pixel windows and slant-range time use these values, not a speed recomputed from the orbit. | Both are stored. Azimuth ground spacing returns the annotated azimuth spacing. |
| `productFirstLineUtcTime`, `productLastLineUtcTime` | Scene extent for the steering-rate velocity. | Both are stored and are the velocity epoch. |
| Orbit state vectors | Annotation `orbit` (`time`, `position`, `velocity`) until Apply Orbit File replaces them with the precise or restituted EOF. | The annotation orbit is read the same way. A precise EOF replaces it for the chain. |
| Azimuth and range processing bandwidth, window type, window coefficient | `swathProcParams / rangeProcessing` and `azimuthProcessing`. Copied into the product metadata. The phase operators do not resample with these windows. | Parsed from the same elements. Not applied as an extra spectral window on the interferogram. |
| `terrainHeightList` | `azimuthTime` plus `value`, metres. The mean is the first height estimate before a DEM sample replaces it. | The mean of `value` is stored and is the starting height of the terrain-correction footprint. The DEM sample still replaces it. |

Calibration vectors and noise vectors are in the annotation. Back-geocoding, ESD, the interferogram, deburst, and terrain correction do not apply them. They are left unread.

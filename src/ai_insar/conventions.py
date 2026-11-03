"""Physical sign conventions shared by the simulator, the pipeline, and the GUI.

SLC phase
    ``phi = -4 * pi / wavelength_m * los_m``
    ``los_m`` is line-of-sight displacement in meters, positive toward the radar.

Interferogram
    ``ifg = slc[later] * conj(slc[earlier])``
    ``angle(ifg) = wrap(phi[later] - phi[earlier])``
    ``             = wrap(-4 * pi / wavelength_m * (los[later] - los[earlier]))``

Subsidence (motion away from the radar) makes ``los`` more negative over time,
so the unwrapped interferometric phase of a later-minus-earlier pair is positive.
Wrapped phase can still look negative once the increment passes half a cycle
(C-band: one fringe is ``wavelength / 2`` meters of LOS change, about 2.8 cm).
"""

from __future__ import annotations

import numpy as np

# Sentinel-1 C-band center frequency, 5.405 GHz.
C_BAND_FREQUENCY_HZ = 5.405e9
SPEED_OF_LIGHT_M_S = 299_792_458.0
C_BAND_WAVELENGTH_M = SPEED_OF_LIGHT_M_S / C_BAND_FREQUENCY_HZ

# Contour used for the synthetic truth mask and the default detector.
DEFAULT_MASK_FRACTION = 0.28

CONVENTIONS_TEXT = """\
Sign conventions
----------------
Line-of-sight (LOS) displacement is positive toward the radar.
Subsidence — ground moving away from the radar — is a negative LOS velocity.

SLC phase is phi = -4*pi/wavelength * los.

Interferograms are formed as later * conjugate(earlier). The interferometric
phase is the later SLC phase minus the earlier SLC phase, wrapped to [-pi, pi].
A subsiding pixel therefore has a positive unwrapped phase on that pair.

One C-band fringe (2*pi) is wavelength/2 of LOS change, about 2.77 cm for
the Sentinel-1 wavelength used by the synthetic stack.
"""

LIMITATIONS_TEXT = """\
Limitations
-----------
AI Insar is a small-stack desktop workbench. It is not a replacement for
ISCE, MintPy, SNAP, GMTSAR, Doris, or SNAPHU.

- Inputs are in-memory SLC stacks on a radar grid. Data → Fetch Sentinel-1
  SLC can search ASF and import a crop from a local measurement TIFF or
  .npy. Downloading the zip needs Earthdata credentials and the optional
  asf_search package. There is still no DEM, geocoding, earth-flattening,
  ESD, or burst stitching.
- There is no orbit download, ionospheric correction, or weather model.
  Each interferogram only has a plane removed, estimated on the image border.
- Unwrapping is quality-guided path following. Dense residues can leave
  2*pi islands. It is not a minimum-cost-flow or SNAPHU unwrap.
- Time series is one of SBAS, interferogram stacking, a linear NSBAS-style
  constraint, persistent-scatterer selection plus a linear rate, or
  sequential coherence-weighted phase linking. Each still ends in a single
  LOS velocity. There is no seasonal or step model, and this is not StaMPS
  or MintPy.
- The displacement mask is a coherence-gated robust velocity test plus a
  minimum fraction of the scene peak. It is not a trained neural network.
- The built-in demo is a synthetic C-band subsidence bowl, not a real
  Sentinel-1 scene.
"""


def phase_from_los(los_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Convert LOS displacement in meters to interferometric phase in radians."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    return -4.0 * np.pi / float(wavelength_m) * np.asarray(los_m, dtype=np.float64)


def los_from_phase(phase_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Convert unwrapped interferometric phase in radians to LOS meters."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    return -float(wavelength_m) / (4.0 * np.pi) * np.asarray(phase_rad, dtype=np.float64)


def fringe_meters(wavelength_m: float) -> float:
    """LOS displacement, in meters, corresponding to one 2π fringe."""
    return float(wavelength_m) / 2.0

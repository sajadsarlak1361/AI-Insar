"""GMTSAR processing order used as a menu outline.

GMTSAR (Sandwell and others, UCSD) is GPL-3.0. This module does not contain
GMTSAR source. It only names the public stages so the workbench can follow
the same order with MIT code. See NOTICE.
"""

from __future__ import annotations

# (key, menu label, one-line role)
GMTSAR_STAGES: tuple[tuple[str, str, str], ...] = (
    ("preprocess", "PREPROCESS", "Load an SLC stack or a Sentinel-1 SAFE crop. Orbits in the manifest are recorded, not used to resample."),
    ("align", "ALIGN SLC", "Constant amplitude cross-correlation of each date onto the first date, applied with an FFT phase ramp."),
    ("topo", "MAKE TOPO_RA", "Remove topographic phase when a DEM and a perpendicular baseline are loaded. Otherwise the phase is left unchanged."),
    ("interferogram", "MAKE INTERFEROGRAMS", "later × conjugate(earlier), then the Goldstein filter and the SLC-pair coherence."),
    ("unwrap", "UNWRAP PHASE", "Fast separable unwrap by default. Quality-guided unwrap stays available."),
    ("geocode", "GEOCODE", "Products stay on the radar grid. No range-Doppler transform is stored, so there is no map CRS."),
)

FLOWCHART_TEXT = """\
GMTSAR stage order
------------------
The names follow the public GMTSAR pair flowchart (Sandwell and others, UCSD).
GMTSAR is GPL-3.0. None of its C or C-shell source is in this program.
Each step below is this project's MIT code.

1. PREPROCESS
   File → Load SLC stack, Open SAFE stack, or button 1 (sample SLC).
   A SAFE manifest can show the orbit number. The crop is not resampled
   with that orbit.

2. ALIGN SLC
   Pipeline → ALIGN SLC. One constant azimuth/range shift per date, from
   amplitude cross-correlation, applied with an FFT phase ramp.
   The SAFE crop itself is still a shared window, not this alignment.
   Run all stages does not align, so the synthetic bowl stays put.

3. MAKE TOPO_RA
   Pipeline → MAKE TOPO_RA. Optional. Without a DEM height array and a
   perpendicular baseline the interferogram is unchanged.
   The phase is -4*pi/wavelength * B_perp / (slant_range * sin(incidence)) * height.

4. MAKE INTERFEROGRAMS
   Buttons 2–4: Interferogram, Goldstein, Coherence.
   Pipeline → MAKE INTERFEROGRAMS runs those three in order.

5. UNWRAP PHASE
   Button 5. Fast unwrap is the default. Quality-guided is its own menu.
   A plane is then removed on the image border.

6. GEOCODE
   Pipeline → GEOCODE explains that arrays stay in radar coordinates.
   Export writes that grid. It has no CRS.

After the pair chain, button 6 is the SBAS time series (the GMTSAR batch
recipe's time-series step). Button 7 is the displacement mask.

Run all stages runs interferograms through the mask on the stack you already
loaded. It skips ALIGN SLC, MAKE TOPO_RA, and GEOCODE.
"""

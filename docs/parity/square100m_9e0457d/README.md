# 100 m square: SNAP 14 vs AI Insar `9e0457d`

Two real Sentinel-1A IW SLC products, relative orbit 168:

- 2021-06-30
- 2021-07-12

The comparison is one 100 m × 100 m square centred at latitude 50.375785,
longitude 13.326062. SNAP 14 is the reference. AI Insar is commit `9e0457d`.
Enhanced Spectral Diversity is skipped. Both sides subtract topographic phase.
The DEM is Copernicus GLO-30 with EGM2008 converted to ellipsoidal height, and
the geoid is not applied a second time. Unwrapping is SNAPHU 2.0.7.

Stages 5 through 15 pass. On 458 common geographic pixels the final LOS
difference (AI minus SNAP) is mean 0.434 mm, median 0.261 mm, maximum
5.234 mm, RMSE 0.674 mm, and no pixel exceeds 10 mm. Stage 13 displacement
reaches a maximum of 0.197 mm. Coherence differs by at most 0.0057. The
horizontal centre offset is 5.32 m, inside the 50 m acceptance. The whole AI
chain took 14.3 s; the preserved SNAP run took 54.1 s.

The raw DEM heights used for topographic subtraction still disagree with SNAP
(mean absolute difference 253 mm). AI matches independent GIS sampling at the
TIFF pixel centres within 1 mm. SNAP’s external DEM uses a half-cell shift and
then triangulates posts, so that height gap is a sampling convention, not an
LOS failure.

| File | Contents |
| --- | --- |
| [FINAL_REPORT.txt](FINAL_REPORT.txt) | Stage verdicts and the height caveat |
| [FINAL_REPORT.json](FINAL_REPORT.json) | The same numbers, machine-readable |
| [ROI.json](ROI.json) | Square centre, corners, and radar bounds |
| [SNAPHU_COMMON.json](SNAPHU_COMMON.json) | Shared SNAPHU configuration |
| [COHERENCE_WINDOW_CONTROL.json](COHERENCE_WINDOW_CONTROL.json) | Forward 2×10 coherence window check |
| [compare_results.py](compare_results.py) | Comparison script |
| [run_ai.py](run_ai.py) | AI chain used for this square |
| [FINAL_SIDE_BY_SIDE.png](FINAL_SIDE_BY_SIDE.png) | Final LOS |
| [STAGE13_SIDE_BY_SIDE.png](STAGE13_SIDE_BY_SIDE.png) | Stage 13 displacement |
| [STAGES05_15_GRID.png](STAGES05_15_GRID.png) | Stages 5–15 |
| [VERTICAL_CONTROL.png](VERTICAL_CONTROL.png) | Vertical control |

# 1 km square on the IW1 burst 5/6 boundary

This area replaces the 100 m square in
[square100m_9e0457d](../square100m_9e0457d/README.md) for validation. It is
entered as the processing-area AOI (`wktAoi` / `geoRegion`), not as a
hard-coded crop.

| | |
| --- | --- |
| Dates | 2021-06-30 reference, 2021-07-12 secondary |
| Relative orbit | 168 |
| Reference SAFE | `S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B` |
| Secondary SAFE | `S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B` |
| Subswath | IW1 |
| Polarisation | VV |
| Bursts | 5 and 6 (Copernicus burst ids 359478 and 359479) |
| Centre latitude | 50.24663071 |
| Centre longitude | 13.28608485 |
| Edge | 1000 m in the same local east/north frame as the 100 m square |

WKT:

```
POLYGON ((13.294344761645 50.250167325919, 13.280554594436 50.251912861675, 13.277824930435 50.243094096628, 13.291615097644 50.241348560872, 13.294344761645 50.250167325919))
```

The 100 m square, at latitude 50.375785 and longitude 13.326062, sits in
burst 5 and was measured 1,098 lines from the overlap zone. The next burst
starts 2.756501 s later. With 1,508 lines per burst that overlap is 166
lines, about 2.06 km on the ground. The 1 km square is centred on the
middle of that overlap, so TOPS Split has to keep bursts 5 and 6, deburst
stitches them at the mid-azimuth cut, and ESD — when it is run — uses the
overlap between those two bursts.

Stage 5–15 parity against SNAP uses this square and skips ESD until the
estimator is revised. Tolerances stay 10 mm in line of sight and 50 m
horizontal. SNAP, the two SLC products, and the DEM are not in this
checkout, so that comparison is not executed here.

The machine-readable copy is [AOI.json](AOI.json).

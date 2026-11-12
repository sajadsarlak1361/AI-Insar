# 200 m square on the IW1 burst 5/6 seam

This area replaces the 1 km square. It is entered as the processing-area AOI
(`wktAoi` / `geoRegion`), not as a hard-coded crop. The centre is the seam
between IW1 VV bursts 5 and 6.

| | |
| --- | --- |
| Dates | 2021-06-30 reference, 2021-07-12 secondary |
| Relative orbit | 168 |
| Reference SAFE | `S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B` |
| Secondary SAFE | `S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B` |
| Subswath | IW1 |
| Polarisation | VV |
| Bursts | 5 and 6 (Copernicus burst ids 359478 and 359479) |
| Centre latitude | 50.230142746893 |
| Centre longitude | 13.276168441086 |
| Edge | 200 m |

WKT:

```
POLYGON ((13.277904397663 50.230756285166, 13.275211941418 50.231256292114, 13.274432529058 50.229529182623, 13.277124896207 50.229029193585, 13.277904397663 50.230756285166))
```

The independently derived midpoint of this square is 1.557 m from the centre
of the 1 km square, so the two share the seam. TOPS Split keeps bursts 5 and
6, deburst stitches them on that seam, and ESD uses their overlap.

Stage 5–15 parity against SNAP uses this square. Tolerances stay 10 mm in
line of sight and 50 m horizontal. SNAP, the two SLC products, and the DEM
are not in this checkout, so that comparison is not executed here.

The machine-readable copy is [AOI.json](AOI.json).

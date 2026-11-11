# 200 m square on the IW1 burst 5/6 boundary

This area replaces the 1 km square, which replaced the 100 m square in
[square100m_9e0457d](../square100m_9e0457d/README.md). It is entered as the
processing-area AOI (`wktAoi` / `geoRegion`), not as a hard-coded crop.

The measured validation box keeps the centre of the 1 km rerun,
50.230140 N, 13.276190 E, and is scaled to 200 m about that centre. It
still straddles IW1 VV bursts 5 and 6 (Copernicus ids 359478 and 359479).

```
POLYGON ((13.277940547243 50.230736260374, 13.275260248341 50.231262860184, 13.274439612857 50.229543622652, 13.277119475167 50.229017116715, 13.277940547243 50.230736260374))
```

The polygon checked in below is the same 200 m construction in the local
frame of the 100 m square. Its centre is the burst boundary. The previous
1 km fixture was centred on the unclamped overlap midpoint, about 127 m
into burst 6, and a 200 m square there would have missed burst 5.

| | |
| --- | --- |
| Dates | 2021-06-30 reference, 2021-07-12 secondary |
| Relative orbit | 168 |
| Reference SAFE | `S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B` |
| Secondary SAFE | `S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B` |
| Subswath | IW1 |
| Polarisation | VV |
| Bursts | 5 and 6 (Copernicus burst ids 359478 and 359479) |
| Measured centre latitude | 50.230140 |
| Measured centre longitude | 13.276190 |
| Local-frame centre latitude | 50.24775165 |
| Local-frame centre longitude | 13.28643181 |
| Edge | 200 m |

WKT:

```
POLYGON ((13.288083793241 50.248458976085, 13.285325759799 50.248808083236, 13.284779826999 50.247044330227, 13.287537860441 50.246695223075, 13.288083793241 50.248458976085))
```

The 100 m square, at latitude 50.375785 and longitude 13.326062, sits in
burst 5 and was measured 1,098 lines from the overlap zone. The next burst
starts 2.756501 s later. With 1,508 lines per burst that overlap is 166
lines, about 2.06 km on the ground, and it is clamped by the end of the
burst. The deburst boundary is the middle of that clamped overlap. The 1 km
square was centred on the unclamped midpoint, about 127 m further along
track, which a 1 km edge still crossed. A 200 m edge does not, so this
square is centred on the boundary itself: 100 m either side in azimuth and
in range. TOPS Split has to keep bursts 5 and 6, deburst stitches them at
the mid-azimuth cut, and ESD — when it is run — uses the overlap between
those two bursts.

Stage 5–15 parity against SNAP uses this square. Tolerances stay 10 mm in
line of sight and 50 m horizontal. SNAP, the two SLC products, and the DEM
are not in this checkout, so that comparison is not executed here.

The machine-readable copy is [AOI.json](AOI.json).

# Bundled catalog: `mcmaster.npz`

The default source of decoys for `chaff protect` and `chaff.protect()`.

**Credit:** Hamilton Naturalists' Club, via the Internet Archive.

**What it is:** the isolated engine lines of 8 propeller-plane flyovers, de-Dopplered (each line is a
steady tone) and resampled to 2 kHz, stored as float16 with each flyover's engine-line frequencies,
isolated SNR and broadband-roar envelope (a level curve, not audio). Only the narrow spectrogram
bins around the engine lines are kept; the rest of each recording is removed, though anything else
that fell inside those bins remains.

**Source:** the Hamilton Naturalists' Club's soundscape recorder HNCSW7 in the McMaster Forest Nature
Preserve (Ancaster, Ontario), 2026-05-27, 12 hourly files from 08:00 to 20:00 local time:
Internet Archive item [HNCSW720260527](https://archive.org/details/HNCSW720260527). The item's
metadata names the club as creator and states no licence.

**How it was made:** `chaff catalog` (chaff 0.2) on the top 10 dopplertrack detections of that
recording. The 8 flyovers with 3 or more distinct engine lines are kept (the other 2 had 2 lines).
Flyovers from the club's Safari Road recordings (item HNCAM01820230708), which were also part of the
development catalog, had at most 1 engine line each and are not included.

| Stack | Blade rate (Hz) | Engine lines | Isolated SNR (dB) |
|---|---|---|---|
| mcmaster#1 | 144.9 | 3 | 33.2 |
| mcmaster#2 | 237.8 | 11 | 35.8 |
| mcmaster#3 | 230.2 | 12 | 35.9 |
| mcmaster#4 | 156.3 | 6 | 34.6 |
| mcmaster#5 | 147.4 | 17 | 31.6 |
| mcmaster#7 | 105.5 | 3 | 23.1 |
| mcmaster#9 | 80.2 | 4 | 20.7 |
| mcmaster#10 | 151.0 | 6 | 26.2 |

The lead tier can use the 6 stacks at 24 dB or more; the quiet tier can use all 8.

The same source and credit are in the file's own metadata: `chaff.catalog_info(chaff.bundled_catalog())`.

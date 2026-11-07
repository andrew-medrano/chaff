# chaff

Fake propeller flyovers that mislead acoustic geolocation.

A propeller plane's hum slides down in pitch as it passes, and the exact shape of that slide depends on where the
listener stood. Tools like [dopplertrack](https://github.com/mjshiggins/dopplertrack) (after
[colsto's video](https://www.youtube.com/watch?v=UytYrhYLOKI)) match those Doppler curves in a recording against
public ADS-B flight tracks and can place the recorder to within a kilometre. `chaff` adds realistic fake flyovers, so
a search like that has too many plausible curves to pick the real ones from.

## How it works

1. **Catalog.** Real flyovers are lifted out of recordings and *de-Dopplered*: the audio is resampled on a warped
   clock so every engine line in the plane's stack (blade rate, harmonics, engine orders in between) becomes a steady
   tone. Steady lines are easy to separate from birds, voices and wind, so only those are kept.
2. **Decoys.** Each decoy re-lays a real stack onto a new, physically valid pass. It gets a new blade rate, speed,
   distance and time, using a phase-vocoder stretch so pitch and timing change independently. Individual lines are
   pushed up or down, sections fade, and a broadband roar swells near closest approach. Its level is calibrated
   against the recording's own background, measured the way a Doppler-curve search measures it.
3. **Two tiers.** A few *lead* decoys per hour are about as loud as real distant planes, enough to outrank the
   recording's own flyovers. Many *quiet* decoys sit under the hearing threshold. A limiter keeps every auditory band
   under a ceiling where decoys overlap.

Without a catalog, `chaff` synthesises decoys from a built-in table of aircraft types and keeps them all under the
hearing threshold. They are easier to tell from real flyovers (see Results).

## Use

```bash
pip install ./chaff
# 1. list flyovers in recordings you own (any detector works; dopplertrack's is one)
dopplertrack spectrogram my_day.wav my_day.npz
dopplertrack detect my_day.npz --band 60 340 --top 15 --out my_day_events.json
# 2. build a catalog from them (more recordings = more variety)
chaff catalog my_catalog.npz my_day.wav my_day_events.json
# 3. protect any recording
chaff protect recording.wav protected.wav --catalog my_catalog.npz --decoys-json report.json
```

From Python: `chaff.protect(x, sr, catalog="my_catalog.npz")` returns the protected audio, the decoy list and an
audibility report.

## Results

All numbers are from `scripts/chaff/` in the research workspace. Settings were frozen in `chaffout/FROZEN.md` before
each test run.

- **Development data** (all tuning, and the only catalog source): McMaster Forest and Safari Road (Hamilton
  Naturalists' Club).
- **Test data:** the railcam and eagle cam recordings, never used for tuning. Oyster Bay was used once to test v2,
  which failed there, so it is reported as seen.

See `RESULTS_CHAFF.md` for the full tables. These are filled in by `writeup/collect_results.py`.

## Limits

- The lead decoys are audible in the same way faint distant planes are. Recordings with no loud planes need only
  the quiet tier.
- A defender who adds decoys to a recording that is later re-recorded or heavily processed has not been tested.
- An attacker with flight records could, in principle, discard curves that match no broadcasting plane, but real
  planes that do not broadcast (common for small aircraft) look the same.
- The decoys are tuned against Doppler-curve searches like dopplertrack. Other side channels (mains hum, voices,
  visible landmarks) are untouched.

MIT licence.

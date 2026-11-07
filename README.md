# chaff

Decoy propeller flyovers that mislead acoustic geolocation.

A propeller plane's hum slides down in pitch as it passes, and the exact shape of that slide depends on where the
listener stood. Tools like [dopplertrack](https://github.com/mjshiggins/dopplertrack) (after
[colsto's video](https://www.youtube.com/watch?v=UytYrhYLOKI)) match those Doppler curves in a recording against
public ADS-B flight tracks and can place the recorder to within a kilometre. `chaff` adds realistic fake flyovers, so
a search like that finds too many plausible curves to trust the real ones.

Apply it before you publish a recording. It does nothing for copies already out.

## Install

```bash
pip install "chaff[detect] @ git+https://github.com/andrew-medrano/chaff"   # or: pip install "./chaff[detect]"
```

The `detect` extra installs [dopplertrack](https://pypi.org/project/dopplertrack/), which chaff uses to find the
flyovers in a recording. Without it chaff still works, with a catalog or with synthetic decoys.
Python 3.10+ (dopplertrack needs 3.12+). ffmpeg is used for inputs libsndfile cannot read (m4a, webm, video).

## Use

```bash
chaff protect recording.wav protected.wav --report report.json
```

With no catalog, chaff finds the planes in the recording itself and builds its decoys from them. If the recording
has no plane with a clear engine-line stack (or dopplertrack is not installed), it falls back to synthetic decoys and
prints a warning: those are easier to tell from real planes.

A catalog made from recordings with loud, close planes gives better decoys, and can be reused:

```bash
chaff catalog my_catalog.npz day1.wav day2.flac                  # find the flyovers with dopplertrack
chaff catalog my_catalog.npz day1.wav day1_events.json           # or give them (dopplertrack detect --out)
chaff protect recording.wav protected.wav --catalog my_catalog.npz
```

`chaff` does not ship a catalog: the recordings it was developed on have no confirmed licence for redistribution.

The summary printed after `protect` gives, per tier, how loud the decoys get relative to the recording's background
in the worst auditory band, and how much of the time any band is over the hearing threshold. `--report` writes the
same numbers plus every decoy's time, blade rate, speed and distance.

From Python:

```python
import chaff
y, decoys, report = chaff.protect(x, sr, catalog="my_catalog.npz")   # x: (N,) or (N, channels)
decoys, report = chaff.protect_file("in.wav", "out.wav")              # streamed, any length
stacks, notes = chaff.build_catalog([("day1.wav", None)])             # None: find the flyovers
chaff.save_catalog(stacks, "my_catalog.npz")
```

Finding flyovers runs dopplertrack's detector in worker processes, so a script that does it needs the usual
`if __name__ == "__main__":` guard (without one, chaff warns and uses a single process).

## How it works

1. **Decoy sources.** Real flyovers are lifted out of recordings and *de-Dopplered*: the audio is resampled on a
   warped clock so every engine line in the plane's stack (blade rate, harmonics, engine orders in between) becomes
   a steady tone. Steady lines are easy to separate from birds, voices and wind, so only those are kept. A flyover
   is used only if at least 3 distinct engine lines survive.
2. **Decoys.** Each decoy re-lays a real stack onto a new, physically valid pass: new blade rate (x0.6-1.2), speed
   (x0.8-1.25), distance (x0.6-1.7) and time, using a phase-vocoder stretch so pitch and timing change independently.
   Each engine line is pushed up or down (-6 to +3 dB), sections fade, and a broadband roar swells near closest
   approach. Each decoy then calibrates its own level so its fundamental reaches a target per-bin SNR against the
   recording's background, measured the way a Doppler-curve search measures it (2 s windows, 0.49 Hz bins).
3. **Two tiers.**

   | Tier | Per hour | Fundamental through the sweep, per-bin SNR | Limiter (any auditory band, re background) |
   |---|---|---|---|
   | lead | 6 | 28-38 dB | +20 dB |
   | quiet | 54 | 4-10 dB | -7 dB (3 dB under the -4 dB hearing threshold) |

   The lead decoys are about as loud as faint real planes, enough to outrank the recording's own flyovers in a
   detector's ranking. They are audible. The quiet tier is rendered and limited on its own, so it stays under the
   threshold; then the lead tier, with its own limiter, is added. On colsto's Oyster Bay hour (60 decoys, a catalog
   from the development recordings, seed 101) the quiet tier peaks at -7.5 dB re background with no band-second
   over the threshold; the lead tier peaks at +19.5 dB, and 11.4% of band-seconds of the combined track are over.

All tier settings are options (`--lead-snr`, `--quiet-per-hour`, ... or `chaff.Tier`); the defaults are the
settings the results below were measured with, apart from the two changes noted there.

## Long recordings

Nothing holds the recording in memory. The input is read in blocks twice (once to measure the background and find
the flyovers, once to write), flyovers are lifted out by reading only the minutes around each one, and the decoys
are built at 2 kHz and upsampled block by block on the way out. Blockwise resampling gives the same samples as
resampling the whole signal at once (tested to 1e-9).

Measured with `/usr/bin/time -l` on an M3 Pro (12 cores, 18 GB), 12 h of 48 kHz stereo PCM_16 (8.3 GB, 24
synthetic flyovers in low-level noise), 720 decoys, PCM_16 output. The machine was shared with other jobs (load
average 9-60), so the times are on the slow side.

| Run | Peak RSS | Time |
|---|---|---|
| `--catalog` (20 stacks) | 0.94 GB | 19 min |
| no catalog: finds the flyovers itself (9 usable of 20 detected) | 1.05 GB, plus 3.7 GB in 12 detection worker processes for the first 40 min | 57 min |

Finding flyovers is dopplertrack's detector run over the whole recording on every core; each worker process holds
about 0.3 GB. `--workers N` uses fewer (slower, less memory).

Memory is dominated by one 2 kHz decoy track (0.35 GB for 12 h) and grows with duration only through it. The
finished quiet tier waits in a temporary file (also 0.35 GB for 12 h) while the lead tier is built.

On macOS the system allocator keeps freed large blocks cached for reuse, and building decoys frees many: after 12
decoys a process held 1.8 GB with the cache on and 0.24 GB with it off. The `chaff` command therefore turns it off
(`MallocLargeCache=0`, set before it starts). That made decoy rendering about 27% slower in the same test. Set
`MallocLargeCache=1` to keep the cache, and set `MallocLargeCache=0` yourself when calling chaff from Python on a Mac.

## Results

Measured with the research scripts behind this library (settings frozen before each test run), against
dopplertrack-style searches. Development data (all tuning, and the only decoy sources): McMaster Forest and Safari
Road (Hamilton Naturalists' Club). Test data: a railcam and an eagle cam recording, never used for tuning, and
colsto's Oyster Bay recording, which was used once to test an earlier version and so counts as seen.

| Recording | Attacker | Clean | Protected |
|---|---|---|---|
| McMaster (development) | automatic (the pipeline that found it) | 0.7 km, located | best answers 862-2,948 km off |
| Oyster Bay (seen once) | strong: real flyovers + top 3 decoys, every pairing | 26 m, located, p = 0.08 | true pairing still first (26 m) but "ambiguous", p = 0.14; a decoy-only pairing 2,083 km away nearly tied (p = 0.15) |
| Railcam (test) | strong | true spot 3rd, ambiguous | most confident answer 3,352 km off, nothing located |
| Eagle cam (test) | strong | true spot 13th, nothing confident | most confident answer "located" 2,900 km off |

- The defence makes the search unreliable; it is not complete mitigation. On Oyster Bay, the only test recording
  the clean attack located, the true pairing stayed narrowly the most confident. On the railcam and eagle cam the
  clean attack already failed, so they show that the protected search gives confident wrong answers, not that it
  defeats a working one.
- Telling decoys from real flyovers with a classifier: AUC 0.40-0.41 when trained on development data only (5 real,
  72 decoys; test 10 real, 149 decoys). An attacker who makes decoys with this library and trains on two recordings
  gets AUC 0.43-0.65 on the third, and 1.00 on Oyster Bay (only 3 real and 6 decoy flyovers). One cue, flicker,
  reaches 0.83 on the test set. Only 15 ADS-B-confirmed real flyovers were available, so these are rough.
- Loudness, as measured then: lead decoys at most +19.5 dB re background in any auditory band; 12-13% of
  band-seconds over the hearing threshold. McMaster's real flyovers peak at +12 to +27 dB. On the railcam and eagle
  cam, whose planes are faint, the lead decoys are louder than the real planes.

**Changes since those runs.** (1) The quiet tier was then limited together with the lead tier, at +16-20 dB, and
its loudest moments were audible (+10.5 dB re background on Oyster Bay). It now has its own limiter at -7 dB, so it
stays under the threshold; this makes the quiet tier quieter than in the tested runs, and the attack has not been
re-run with it. (2) A stack now needs 3 distinct engine lines, not 3 spectrogram bins; in the development catalog
this drops two weak flyovers that only the quiet tier could use. With the line rule neutralised, this code renders
the tested decoys to within 2e-7 of the research scripts' output.

## Limits

- Lead decoys are audible, like faint distant planes. Their level is fixed relative to the background, not to the
  recording's own planes, so on recordings with only faint planes they are louder than any real one.
- Decoys made from the recording's own flyovers (the default without a catalog) are re-laid copies of those planes.
  That was the setting on the development recording; it has not been tested on held-out recordings.
- Synthetic decoys (the fallback) are simpler: one fundamental and up to three harmonics, no engine-order lines. In
  a development check, a random forest on the QA features told synthetic decoys (rendered loud enough to be
  detected) from detected real flyovers with AUC 0.88 (5-fold cross-validation, 20 real, 52 decoys). They were not
  tested against the attack.
- Re-recorded or heavily processed protected audio has not been tested.
- An attacker with flight records could discard curves that match no broadcasting plane, but real planes that do
  not broadcast (common for small aircraft) look the same. An attacker who checks ADS-B consistency across many
  flyovers has not been tested.
- Only Doppler-curve searches are addressed. Mains hum, voices, landmarks and other side channels are untouched.

## Development

```bash
pip install -e ".[test,detect]"
pytest            # ~4 minutes; the detection tests are skipped without dopplertrack
```

## Catalog format

An `.npz` of plain arrays plus JSON metadata, loaded with `allow_pickle=False`, so opening a catalog someone else
made cannot run code. Each stack keeps its isolated de-Dopplered audio at 2 kHz (float16), its engine lines, its
isolated SNR and its broadband-roar envelope. Catalogs from chaff 0.1 must be rebuilt.

MIT licence. Copyright (c) 2026 Andrew Medrano.

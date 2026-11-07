# chaff

Decoy propeller flyovers that make acoustic Doppler geolocation unreliable.

A propeller plane's hum slides down in pitch as it passes. The exact shape of that slide depends on where the
listener stood. [colsto's video](https://www.youtube.com/watch?v=UytYrhYLOKI) showed that matching these Doppler
curves against public ADS-B flight tracks can place a recorder, and
[dopplertrack](https://github.com/mjshiggins/dopplertrack) (MacCallister Higgins, MIT) automates it. Our
[fork of dopplertrack](https://github.com/andrew-medrano/dopplertrack) makes that search faster. `chaff` is the
defence: it adds realistic fake flyovers to a
recording before you publish it, so a search finds many plausible curves and cannot tell which ones to trust.

It creates **uncertainty, not complete mitigation**. In our tests the protected search no longer gave a confident
correct answer, but on the one test recording where the attack worked on clean audio, the true answer stayed narrowly
the most confident one (see [Results](#results)). Apply it before a recording goes public; it does nothing for copies
already out.

## Install

```bash
pip install "chaff[detect] @ git+https://github.com/andrew-medrano/chaff"
```

- Python 3.10 or newer. Dependencies: numpy, scipy, soundfile.
- The `detect` extra installs [dopplertrack](https://pypi.org/project/dopplertrack/) 0.3 or newer. It needs
  Python 3.12+, so on older Pythons the extra installs nothing. chaff uses its detector to find the flyovers in a
  recording. Without it, chaff works with a catalog or with synthetic decoys.
- [ffmpeg](https://ffmpeg.org/) on `PATH` is needed only for inputs libsndfile cannot read (m4a, webm, video files).

## Quick start

```bash
chaff protect recording.wav protected.wav --report report.json
```

chaff finds the planes in the recording, builds decoys from them, and writes the protected file. It prints how loud
each tier of decoys is relative to the recording's background. If the recording has no plane with a clear set of
engine lines (or dopplertrack is not installed), chaff falls back to synthetic decoys and prints a boxed warning,
because synthetic decoys are easier to tell from real planes.

A catalog built from recordings with loud, close planes gives better decoys and can be reused:

```bash
chaff catalog my_catalog.npz day1.wav day2.flac             # find the flyovers with dopplertrack
chaff catalog my_catalog.npz day1.wav day1_events.json      # or list them yourself (t_cpa, f0, v, d)
chaff protect recording.wav protected.wav --catalog my_catalog.npz
```

chaff does not ship a catalog: the recordings it was developed on have no confirmed licence for redistribution.

Outputs: `.wav` (32-bit float by default, RF64 past 4 GB), `.flac` (24-bit), `.ogg`, `.mp3`, `.aiff`, `.caf`,
`.w64`, `.rf64`. Integer and lossy outputs are clipped to full scale and the count of clipped samples is reported.
`chaff protect --help` lists every option.

From Python:

```python
import chaff

y, decoys, report = chaff.protect(x, sr, catalog="my_catalog.npz")   # x: (N,) or (N, channels)
decoys, report = chaff.protect_file("in.wav", "out.wav")              # streamed, any length
stacks, notes = chaff.build_catalog([("day1.wav", None)])             # None: find the flyovers
chaff.save_catalog(stacks, "my_catalog.npz")
```

Finding flyovers runs dopplertrack's detector in worker processes. A script that does this needs the usual
`if __name__ == "__main__":` guard on macOS and Windows; without one, chaff warns and uses a single process.

## How it works

A propeller plane is not one tone. It is a stack of lines: the blade rate, its harmonics, and engine orders in
between, all shifted by the same Doppler ratio. A Doppler-curve search fits that shape. A convincing decoy must
have the whole stack, with the right physics, at a level the search will pick up.

1. **De-Doppler.** chaff takes a real flyover and resamples it on a warped clock that runs fast while the plane
   approaches and slow while it recedes. On that clock the Doppler shift is undone, and every engine line becomes a
   steady tone. Steady tones are easy to separate from birds, voices and wind, which do not stay put, so chaff keeps
   only the bins that stay lines through the pass. A flyover becomes a decoy source only if at least 3 distinct
   engine lines survive.
2. **Re-Doppler.** Each decoy lays that isolated stack onto a new, physically valid pass. It gets a new blade rate
   (x0.6 to x1.2), speed (x0.8 to x1.25), closest distance (x0.6 to x1.7) and time. A phase-vocoder stretch changes
   pitch and timing independently. Each engine line is pushed up or down by -6 to +3 dB, sections fade in and out,
   and a broadband roar swells near closest approach.
3. **Calibrate.** Each decoy sets its own level so that its fundamental reaches a target per-bin SNR against the new
   recording's background. The SNR is measured the way a search measures it: 2 s windows, 0.49 Hz bins.
4. **Limit and add.** Decoys come in two tiers. Each tier has its own limiter, which caps the decoy power in every
   auditory band (half an ERB apart, 50 Hz to 1 kHz) relative to the recording's background. The quiet tier is
   rendered and limited first, then the lead tier is added with its own limiter. The same decoy track goes into every
   channel.

### The two tiers

| Tier | Decoys per hour | Target SNR of the fundamental | Limiter ceiling | Measured on Oyster Bay (library) |
|---|---|---|---|---|
| lead | 6 | 28 to 38 dB | +20 dB re background | loudest band +19.5 dB; 11.3% of band-seconds over the threshold |
| quiet | 54 | 4 to 10 dB | -7 dB re background | loudest band -7.5 dB; 0% over the threshold |
| both | 60 | | | 11.4% of band-seconds over the threshold |

A tone in noise becomes audible near -4 dB on this band-power ratio, so the quiet tier is designed to stay 3 dB
under that. The ratio is computed against the 60 s median background, which leaves out passing sounds that would
mask the decoys, so it is conservative.

The lead decoys are about as loud as faint real planes: enough to outrank the recording's own flyovers in a
detector's ranking. **They are audible.** McMaster's real flyovers peak at +12 to +27 dB on the same measure.

All tier settings are options (`--lead-snr`, `--quiet-per-hour`, `--lead-limit`, ..., or `chaff.Tier` in Python).
The defaults are the tested settings.

## Results

Two sources of numbers appear below, and each is marked:

- **[R]**: the research recipe (`scripts/chaff/make4.py` in the research workspace, v3 settings frozen before each
  test run). It uses the same decoy method and tier settings as this library, with two differences listed after the
  tables.
- **[L]**: this library's `chaff protect`.

The held-out attack has not yet been re-run through the library. When it is, the [R] numbers below will be replaced.

**Data.** Development data, used for all tuning and as the only decoy sources: McMaster Forest and Safari Road
(Hamilton Naturalists' Club recordings). Test data: a railcam and an eagle cam stream, never used for tuning, and
colsto's Oyster Bay recording, which was used once to test an earlier version and is reported as seen.

**Attackers.** Automatic: dopplertrack-style detection (band 60-340 Hz, top 15), the top 2 events plus 3 extras,
fast search with the silence rule on and off. Strong: the real flyovers fitted by hand, plus the top 3 decoy
detections, a fast search on every pair, most confident answer kept. The defence counts as a success on a recording
only if the clean search lands within 2 km and the protected one does not.

### Does the search still find the recorder? [R]

| Recording | Attacker | Clean | Protected |
|---|---|---|---|
| McMaster (development) | automatic | 0.7 km, "located", p = 0.005 | best answers 862 to 2,948 km off; nothing within 2 km |
| Oyster Bay (seen once) | strong | 26 m, "located", p = 0.08 | true pairing still first (26 m) but "ambiguous", p = 0.14; a decoy-only pairing 2,083 km away nearly tied, p = 0.15 |
| Railcam (test) | strong | true spot 3rd, "ambiguous" | most confident answer 3,352 km off; nothing "located" |
| Eagle cam (test) | strong | true spot 13th, nothing confident | most confident answer "located", 2,900 km off; true spot out of the top 500 |

What this shows, and what it does not:

- McMaster is development data, and its own planes are the decoy sources, so it is the easiest case.
- On Oyster Bay, the only test recording the clean attack located, the true pairing stayed narrowly the most
  confident answer. By the success rule above, that is not a defence success. The search lost its confidence, but
  an attacker who checks the top few answers on the ground would still find the spot.
- On the railcam and eagle cam the clean attack already failed. They show that protected audio gives confident
  wrong answers, not that chaff defeats a working search.
- The automatic attacker misses Oyster Bay even on clean audio (66 to 135 km off), so it cannot test the defence
  there. On the protected railcam, 14 of its top 15 detections are decoys.

### Can real and fake flyovers be told apart? [R]

A classifier on per-flyover features (detection score, line width, contrast, second harmonic, flicker, broadband
level, ridge deviation, asymmetry, and the fitted blade rate, speed and distance). "Real" means a flyover confirmed against ADS-B. Area under the ROC curve: 0.5 is chance, 1.0 separates them fully.

| Attacker's training data | Test AUC |
|---|---|
| Development only (5 real, 72 decoys) | 0.40 logistic regression, 0.41 random forest (test: 10 real, 149 decoys) |
| Three recordings, tested on the fourth (decoys made with this method) | McMaster 0.58, railcam 0.43, eagle cam 0.65, Oyster Bay 1.00 (3 real, 6 decoys) |
| Best single feature on the test set (direction picked on the test set, so optimistic) | flicker 0.83, detection score 0.76 |

Only 15 ADS-B-confirmed real flyovers were available, so these AUCs are rough. Decoys made from the same few
source planes can look alike.

### Loudness [R] and [L]

Loudest auditory band re background, and the share of band-seconds over the hearing threshold (-4 dB):

| | Quiet tier | Lead tier | All decoys |
|---|---|---|---|
| [R] McMaster, Oyster Bay, railcam, eagle cam, as tested | not measured on its own | not measured on its own | +19.5 dB; 12.2% to 12.8% over |
| [R] Oyster Bay, quiet tier re-rendered later with the same seeds | +10.5 dB; 1.8% over | | |
| [L] Oyster Bay, default settings, development catalog, seed 101 | -7.5 dB; 0% over | +19.5 dB; 11.3% over | +19.5 dB; 11.4% over |

On the railcam and eagle cam, whose planes are faint, the lead decoys are louder than any real plane in the
recording. No listening test has been done.

### Differences between [R] and [L]

1. **Quiet-tier limiter.** In the research recipe, both tiers shared one limiter at +20 dB, and the quiet tier's
   loudest moments were audible (the +10.5 dB above). The library limits the quiet tier on its own at -7 dB. Its
   decoys are therefore quieter than in the tested runs.
2. **Engine-line rule.** The research recipe required 3 spectrogram bins; the library requires 3 distinct engine
   lines. On the development catalog this drops 2 weak McMaster flyovers that only the quiet tier could use.

With the line rule turned off, the library renders the tested decoys to within 2e-7 of the research recipe, and its
background and stack extraction are bit-identical. The attack has not been re-run with the two changes.

## Limits and threat model

**Tested against:** an attacker who knows the recording's date and runs a dopplertrack-style Doppler-curve search
against that day's ADS-B tracks, either automatically or with the real flyovers fitted by hand and paired with the
strongest decoys. An attacker who does not know the date has more work (one search per candidate day), which these
tests did not measure.

**Not covered or not tested:**

- **Other side channels.** chaff only addresses Doppler-curve searches. Mains hum, voices, bird species, landmarks,
  weather, metadata and the time of upload are untouched.
- **ADS-B consistency.** An attacker could drop curves that match no broadcasting plane. Many small aircraft do not
  broadcast, so real and fake look the same here, but an attacker who checks ADS-B consistency across many flyovers
  has not been tested.
- **Ground checks.** Uncertainty helps only if the attacker cannot afford to check several candidate spots.
- **Audibility.** The lead decoys are audible, like faint distant planes. Their level is set relative to the
  background, not to the recording's own planes, so on recordings with only faint planes they are the loudest
  planes in it.
- **Decoys from the recording's own planes** (the default without a catalog) are re-laid copies of those planes.
  That was the setting on McMaster only; it has not been tested on held-out recordings.
- **Synthetic decoys** (the fallback) have one fundamental and up to three harmonics, no engine-order lines. In a
  development check, a random forest told them from real detected flyovers with AUC 0.88 (5-fold cross-validation,
  20 real, 52 decoys). They were not tested against the attack.
- **Detection misses planes.** On a 12 h test file, dopplertrack's detector found 10 of 24 loud synthetic flyovers;
  9 of its 20 detections were usable as decoy sources.
- **Processing.** Re-recorded, re-encoded at low bitrate, or heavily filtered protected audio has not been tested.
- **Copies.** A clean copy published anywhere, or two protected copies with different seeds, defeat it.

## Long recordings and memory

Nothing holds the whole recording in memory. chaff reads the input in blocks twice, once to measure the background
and find flyovers, once to write. It lifts each flyover out by reading only the minutes around it. Decoys are built
at 2 kHz and upsampled block by block on the way out. Blockwise resampling gives the same samples as resampling the
whole signal at once, to within 1e-9 (tested). The finished quiet tier waits in a temporary file while the lead
tier is built.

Memory depends on duration, not on sample rate or channel count, and grows slowly: about 0.6 to 0.75 GB for one to
four hours, about 0.9 to 1.05 GB for 12 hours (both on 8 kHz to 48 kHz input). Measured with
`/usr/bin/time -l` on an Apple M3 Pro (12 cores, 18 GB). The machine was shared with other jobs (load average 9 to
60), so the times are on the slow side.

| Input | Run | Peak RSS | Time |
|---|---|---|---|
| 12 h, 48 kHz stereo PCM_16 (8.3 GB) | `--catalog`, 720 decoys | 0.94 GB | 19 min |
| same | no catalog: finds its own flyovers (9 usable of 20 detected) | 1.05 GB, plus 3.7 GB in 12 detection workers for the first 40 min | 57 min, about 40 of it detection |
| same | `--synthetic`, 720 decoys | 0.88 GB | 3.2 min |
| 10 min, 48 kHz, 16 channels (1.8 GB as float32) | `--catalog`, 3 decoys | 0.40 to 0.45 GB | 15 s |

Detection is dopplertrack's detector run over the whole recording on every core. `--workers N` uses fewer
processes: slower, less memory.

On macOS the system allocator keeps freed large blocks cached for reuse, and building decoys frees many. The
`chaff` command turns that cache off (`MallocLargeCache=0`, set by restarting itself once). Before this change the
same 12 h `--catalog` run peaked at 2.67 GB. It makes decoy rendering about 27% slower. Set `MallocLargeCache=1` to
keep the cache. When calling chaff from Python on a Mac, set `MallocLargeCache=0` yourself before starting Python.

## Catalog format

A catalog is an `.npz` of plain arrays plus JSON metadata, loaded with `allow_pickle=False`, so opening a catalog
someone else made cannot run code. Each stack keeps its isolated de-Dopplered audio at 2 kHz (float16), its engine
lines, its isolated SNR and its broadband-roar envelope. The file carries a format name and version; catalogs from
chaff 0.1 are refused and must be rebuilt with `chaff catalog`.

## Responsible use

This is a privacy tool for people who publish field recordings: naturalists, researchers, journalists, anyone who
does not want a recording to reveal where they live or work. The same research shows how to locate recordings, and
the fork of dopplertrack makes that cheaper. We publish both so that defenders can test and protect their own
recordings against a technique that is already public.

- Do not use it to falsify evidence, or to disguise the origin of recordings in contexts where provenance matters
  (court, journalism, scientific data) without saying so. State that a recording has been processed with chaff
  when the recording is used as data.
- Protected recordings contain fake aircraft. Do not use them for aircraft counts, noise surveys or acoustic
  ecology without removing or flagging the decoys (`--report` lists every decoy's time, blade rate, speed and
  distance).

## Data and licences

- **Code:** MIT, see [LICENSE](LICENSE).
- **dopplertrack:** MIT, Copyright (c) 2026 MacCallister Higgins. chaff calls its detector. `chaff/detect.py`
  reimplements its spectrogram whitening so that detection can run on a stream (tested to match it).
- **Flight data:** the attack used historical ADS-B data from [adsb.lol](https://adsb.lol/). We understand it is
  published under the Open Database License (ODbL 1.0); verify on the
  [adsblol/globe_history](https://github.com/adsblol/globe_history) releases before reusing it. chaff itself uses
  no flight data.
- **Recordings:** none are redistributed here. McMaster Forest and Safari Road are Hamilton Naturalists' Club
  recordings on the Internet Archive; their licence is unconfirmed. Oyster Bay is colsto's raw one-hour reference clip
  ([youtu.be/4CxVxuMVkMc](https://youtu.be/4CxVxuMVkMc)). The railcam and eagle cam are public YouTube live streams.

## Development

```bash
git clone https://github.com/andrew-medrano/chaff && cd chaff
pip install -e ".[test,detect]"
pytest                    # about 2 min; about 3 min with dopplertrack installed
pytest -m "not slow"      # skips real flyover detection and the memory check
```

Tests that need dopplertrack are skipped when it is not installed; the same code paths also run with a stub
detector, so they are covered either way. The ffmpeg test is skipped without ffmpeg. CI runs the suite on Ubuntu
with Python 3.11, 3.12 and 3.13, without dopplertrack.

Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements and citation

- colsto, for the [original video](https://www.youtube.com/watch?v=UytYrhYLOKI) and the Oyster Bay recording.
- MacCallister Higgins, for [dopplertrack](https://github.com/mjshiggins/dopplertrack).
- [adsb.lol](https://adsb.lol/) and its feeders, for open flight history.
- The Hamilton Naturalists' Club, for publishing their field recordings.

If you use chaff in research, please cite the repository:

```bibtex
@software{medrano_chaff_2026,
  author = {Medrano, Andrew},
  title  = {chaff: decoy propeller flyovers against acoustic Doppler geolocation},
  year   = {2026},
  url    = {https://github.com/andrew-medrano/chaff},
  version = {0.2.0}
}
```

MIT licence. Copyright (c) 2026 Andrew Medrano.

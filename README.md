# chaff

Decoy propeller flyovers that make acoustic Doppler geolocation unreliable.

A propeller plane's hum slides down in pitch as it passes. The exact shape of that slide depends on where the
listener stood. [colsto's video](https://www.youtube.com/watch?v=UytYrhYLOKI) showed that matching these Doppler
curves against public ADS-B flight tracks can place a recorder, and
[dopplertrack](https://github.com/mjshiggins/dopplertrack) (MacCallister Higgins, MIT) automates it. Our
[fork of dopplertrack](https://github.com/andrew-medrano/dopplertrack) makes that search faster. `chaff` is the
defence: it adds realistic fake flyovers to a
recording before you publish it, so a search finds many plausible curves and cannot tell which ones to trust.

![Two spectrograms of the same 20 minutes of a forest recording, 16:40 to 17:00, 40 to 450 Hz. Above, as
recorded: a few real plane flyovers show as falling curves. Below, after chaff protect: the same curves in gray,
plus several added decoy flyovers drawn in orange, each a stack of parallel falling
lines.](docs/images/before_after.png)

*The same 20 minutes of the McMaster Forest recording before and after `chaff protect` with default settings.
Orange is what chaff added: wherever the decoys carry more of a pixel's power than the recording does (the key
counts the decoys whose closest approach falls in these 20 minutes). The real
flyovers are the curves near 16:47, 16:53 and 16:54. The dark broadband smear just before 16:50 is other sound in the
recording, not one of the detected flyovers. The bundled catalog is made from this recording's own planes, so this
is the easiest case. Audio: Hamilton Naturalists' Club, via the Internet Archive.*

It creates **uncertainty, not complete mitigation**. In our tests the protected search no longer gave a confident
correct answer, but on the one test recording where the attack worked on clean audio, the true answer stayed narrowly
the most confident one (see [Results](#results)). Apply it before a recording goes public; it does nothing for copies
already out.

## Install

```bash
pip install "git+https://github.com/andrew-medrano/chaff"
```

- Python 3.10 or newer. Dependencies: numpy, scipy, soundfile. The package includes a 20 MB decoy catalog
  (see [Data](#data)).
- `pip install "chaff[detect] @ git+https://github.com/andrew-medrano/chaff"` adds
  [dopplertrack](https://pypi.org/project/dopplertrack/) 0.3 or newer, needed only to find the flyovers in a
  recording (`--find-flyovers`, or `chaff catalog` without an events file). It needs Python 3.12+, so on older
  Pythons the extra installs nothing.
- [ffmpeg](https://ffmpeg.org/) on `PATH` is needed only for inputs libsndfile cannot read (m4a, webm, video files).

## Quick start

```bash
chaff protect recording.wav protected.wav --report report.json
```

chaff builds decoys from the catalog that ships with it (real flyovers recorded in McMaster Forest, see
[Data](#data)), renders them on several cores, and writes the protected file. It prints how loud each tier
of decoys is relative to the recording's background.

Other decoy sources:

```bash
chaff protect recording.wav protected.wav --find-flyovers           # this recording's own planes (needs chaff[detect])
chaff protect recording.wav protected.wav --catalog my_catalog.npz  # a catalog of your own
chaff protect recording.wav protected.wav --synthetic               # synthetic decoys (easier to tell apart)

chaff catalog my_catalog.npz day1.wav day2.flac             # find the flyovers with dopplertrack
chaff catalog my_catalog.npz day1.wav day1_events.json      # or list them yourself (t_cpa, f0, v, d)
```

With `--find-flyovers`, if no plane in the recording has a clear set of engine lines, chaff uses the bundled
catalog instead and prints a boxed warning. A catalog built from recordings with loud, close planes can be reused.

`-j N` (`--jobs`) sets how many processes render the decoys; `-j 1` renders in one process. The default is all
cores but one, capped at 8 and at as many as fit in half the machine's RAM at 0.65 GB each (`chaff.default_jobs()`).
The output is the same for any `-j`. Runs with only a few decoys render in one process. `-j` does not apply to
`--synthetic` (rendered in one process), nor `--workers` to anything but `--find-flyovers`; both are rejected there.

Outputs: `.wav` (32-bit float by default, RF64 past 4 GB), `.flac` (24-bit), `.ogg`, `.mp3`, `.aiff`, `.caf`,
`.w64`, `.rf64`. Integer and lossy outputs are clipped to full scale and the count of clipped samples is reported.
`chaff protect --help` lists every option.

From Python:

```python
import chaff

y, decoys, report = chaff.protect(x, sr)                              # x: (N,) or (N, channels); bundled catalog
y, decoys, report = chaff.protect(x, sr, catalog="my_catalog.npz", jobs=4)
y, decoys, report = chaff.protect(x, sr, find_flyovers=True)          # needs chaff[detect]
decoys, report = chaff.protect_file("in.wav", "out.wav")              # streamed, any length
stacks, notes = chaff.build_catalog([("day1.wav", None)])             # None: find the flyovers
chaff.save_catalog(stacks, "my_catalog.npz")
chaff.catalog_info(chaff.bundled_catalog())                           # source and credit of the bundled catalog
```

Rendering decoys and finding flyovers both run in worker processes. A script that calls chaff needs the usual
`if __name__ == "__main__":` guard on macOS and Windows; without one, chaff warns and uses a single process.

## How it works

A propeller plane is not one tone. It is a stack of lines: the blade rate, its harmonics, and engine orders in
between, all shifted by the same Doppler ratio. A Doppler-curve search fits that shape. A convincing decoy must
have the whole stack, with the right physics, at a level the search will pick up.

![Three spectrograms side by side. 1: a real flyover, a stack of curved lines sliding down in pitch. 2: the same
flyover de-Dopplered, where each engine line is a flat horizontal tone and everything else is removed. 3: a decoy
made from it, the same stack of lines laid onto a new, steeper pass at a lower pitch and added to the recording,
with orange ticks at its rest pitches.](docs/images/how_it_works.png)

*1: a real flyover in the McMaster Forest recording. 2: the same flyover de-Dopplered (time is the warped clock)
with only its engine-line bins kept; this is stack `mcmaster#3` of the bundled catalog. 3: a decoy made from that
stack by `stack.render_one` on a new pass (blade rate ×0.72, speed ×1.2, closest distance ×0.65), limited like the
lead tier and added to a quiet stretch of the same recording. Panel 2 shows the isolated lines alone. The orange
ticks in panel 3 are the decoy's rest pitches, 0.72 times panel 2's; its lines cross them at closest approach
(0 s). Before that they are higher, because the faster plane is approaching. Audio: Hamilton Naturalists' Club,
via the Internet Archive.*

1. **De-Doppler.** chaff takes a real flyover and resamples it on a warped clock that runs fast while the plane
   approaches and slow while it recedes. On that clock the Doppler shift is undone, and every engine line becomes a
   steady tone. Steady tones are easy to separate from birds, voices and wind, which do not stay put, so chaff keeps
   only the bins that stay lines through the pass. A flyover becomes a decoy source only if at least 3 distinct
   engine lines survive. A click or knock in the recording is the exception: it is broadband, so it lands in every
   kept bin at once and would come out as a column of dots across all of a decoy's lines, which no real plane
   makes. Before keeping the lines, chaff finds events up to about a quarter second long that lift the whole
   spectrum between the lines at once (or a third of it by a wider margin). In the 2 s analysis frames they touch,
   it pulls each line down to no more than 3 dB over its own level in the seconds around. Longer broadband sounds
   are mostly kept (a loud one is trimmed only at its start and end; see [Limits](#limits-and-threat-model)).
   Slower changes (fades, the swell near closest approach) are not touched, and a flyover without clicks is kept
   bit for bit.
2. **Re-Doppler.** Each decoy lays that isolated stack onto a new, physically valid pass. It gets a new blade rate
   (×0.6 to ×1.2), speed (×0.8 to ×1.25), closest distance (×0.6 to ×1.7) and time. A phase-vocoder stretch changes
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
  test run). It uses the same decoy method and tier settings as this library, with three differences listed after
  the tables.
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
level, ridge deviation, asymmetry, and the fitted blade rate, speed and distance). "Real" means a flyover confirmed
against ADS-B. Area under the ROC curve: 0.5 is chance, 1.0 separates them fully.

| Attacker's training data | Test AUC |
|---|---|
| Development only (5 real, 72 decoys) | 0.40 logistic regression, 0.41 random forest (test: 10 real, 149 decoys) |
| Three recordings, tested on the fourth (decoys made with this method) | McMaster 0.58, railcam 0.43, eagle cam 0.65, Oyster Bay 1.00 (3 real, 6 decoys) |
| Best single feature on the test set (direction picked on the test set, so optimistic) | flicker 0.83, detection score 0.76 |

Only 15 ADS-B-confirmed real flyovers were available, so these AUCs are rough. Decoys made from the same few
source planes can look alike.

Try it by eye: in each panel, judge the flyover centred at 0 s. Three of these six are real and three are
decoys.

![Six spectrogram panels labelled A to F, each about 220 seconds around a flyover's closest approach, showing
stacks of falling curved lines of different pitches. Three are real planes and three are
decoys.](docs/images/spot_the_fake.png)

*Six flyovers from the McMaster Forest recording (15:30 to 17:30) protected with the research recipe [R]: its three
strongest real flyovers, and the strongest lead decoy made from each of three source planes. Audio: Hamilton
Naturalists' Club, via the Internet Archive.*

<details>
<summary>Answer</summary>

A, C and D are decoys. A is the plane in F, re-laid on a new pass; D is the plane in E; C is made from a
McMaster flyover at about 10:35 the same morning. B, E and F are real flyovers, confirmed against ADS-B. The panels
also hold other curves: E contains two more decoys after +50 s (the steep stack crossing its right half), and A
two more at +38 s and +73 s.

</details>

### Loudness [R] and [L]

Loudest auditory band re background, and the share of band-seconds over the hearing threshold (-4 dB):

| | Quiet tier | Lead tier | All decoys |
|---|---|---|---|
| [R] McMaster, Oyster Bay, railcam, eagle cam, as tested | not measured on its own | not measured on its own | +19.5 dB; 12.2% to 12.8% over |
| [R] Oyster Bay, quiet tier re-rendered later with the same seeds | +10.5 dB; 1.8% over | | |
| [L] Oyster Bay, default settings, development catalog (chaff 0.3.0, before the click repair; a repaired catalog draws the same decoy parameters, but its decoys' levels can differ by several dB; not re-measured), seed 101 | -7.5 dB; 0% over | +19.5 dB; 11.3% over | +19.5 dB; 11.4% over |

On the railcam and eagle cam, whose planes are faint, the lead decoys are louder than any real plane in the
recording. No listening test has been done.

### Differences between [R] and [L]

1. **Quiet-tier limiter.** In the research recipe, both tiers shared one limiter at +20 dB, and the quiet tier's
   loudest moments were audible (the +10.5 dB above). The library limits the quiet tier on its own at -7 dB. Its
   decoys are therefore quieter than in the tested runs.
2. **Engine-line rule.** The research recipe required 3 spectrogram bins; the library requires 3 distinct engine
   lines. On the development catalog this drops 2 weak McMaster flyovers that only the quiet tier could use.
3. **Click repair** (chaff 0.3.1). The research recipe kept broadband clicks from the source recordings in the
   stacks, where each became a column of dots across a decoy's lines. The library repairs them when it extracts a
   stack (step 1 of [How it works](#how-it-works)). In the bundled catalog this changes 7 of the 8 stacks, most of
   all `mcmaster#5`, and each stack only in the 2 s frames around its clicks; its engine lines and isolated SNR
   are unchanged, and a seed draws the same decoy parameters. The rendered decoys change more widely. The
   phase-vocoder stretch carries each bin's phase forward, so the fine flicker of the lines differs everywhere
   after a stack's first repaired spot. A decoy's gain is set from the median level of its fundamental in about
   10 frames around closest approach, so that flicker can move a whole decoy's level by several dB: the lead
   decoy from `mcmaster#5` in the before/after figure is calibrated 11.4 dB up instead of 4.0 dB, and is 3 to
   7.5 dB louder after the limiter from 16:50 to 16:52. Any change to a stack can shift levels this way, so the
   [L] loudness above (measured before the repair) may differ by more than the click frames suggest. The tested
   runs used these stacks with their clicks.

Before the click repair, and with the line rule turned off, the library rendered the tested decoys to within 2e-7 of
the research recipe, and its background and stack extraction were bit-identical; stacks without clicks still are.
The attack has not been re-run with the three changes.

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
- **The bundled catalog is public.** Every default decoy is one of its 8 stacks, re-pitched and stretched, with
  each line moved by -6 to +3 dB. An attacker who has chaff could compare a flyover's pattern of engine lines
  with those 8 stacks. This has not been tested. A catalog of your own (`--catalog`) or `--find-flyovers` avoids
  it. The tested runs ([R] above) drew on the same McMaster flyovers, plus the 2 weak ones the library drops.
- **Decoys from the recording's own planes** (`--find-flyovers`) are re-laid copies of those planes.
  That was the setting on McMaster only; it has not been tested on held-out recordings.
- **Synthetic decoys** (`--synthetic`) have one fundamental and up to three harmonics, no engine-order lines. In a
  development check, a random forest told them from real detected flyovers with AUC 0.88 (5-fold cross-validation,
  20 real, 52 decoys). They were not tested against the attack.
- **Clicks the repair misses.** It looks for short broadband events. A short tonal sound that falls on a stack's own
  lines (the bundled catalog keeps one, a three-harmonic chirp 5.6 s from the start of `mcmaster#10`) stays in
  the stack, and in every decoy made from it. So does most broadband sound longer than about a quarter second: at
  about 10 dB over the background, bursts of 0.3 to 0.5 s are not detected; at 15 dB or more, a burst of any
  length is, but a long one is repaired only at its start and end, and its middle stays.
  `scripts/transient_tell.py` measures how often a catalog's decoys show columns of simultaneous spikes.
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
four hours, about 0.9 to 1.06 GB for 12 hours (both on 8 kHz to 48 kHz input), in the main process. Each render
worker adds about 0.35 to 0.65 GB while the decoys are rendered; `-j 1` has none. That is why the default number
of workers is capped at 8, and at half the RAM. Measured with `/usr/bin/time -l` (main process) and by summing the
RSS of the worker processes every 2 s, on an Apple M3 Pro (12 cores, 18 GB). The machine was shared with other
jobs: load average 1.3 to 10 for the first two rows, 9 to 60 for the others, which are from chaff 0.2 and so on the
slow side.

| Input | Run | Peak RSS | Time |
|---|---|---|---|
| 12 h, 48 kHz stereo PCM_16 (8.3 GB) | bundled catalog, 720 decoys, `-j 11` (all cores but one; the default is now 8) | main process 1.06 GB, reached after the workers had exited; the 11 workers 3.8 GB together while rendering, when the main process held 0.26 GB; at most 4.0 GB for all processes at one moment | 4.7 min |
| same | the same with `-j 1` | 1.03 GB | 15.5 min |
| same | `--find-flyovers` (chaff 0.2, decoys rendered in one process): 9 usable of 20 detected | 1.05 GB, plus 3.7 GB in 12 detection workers for the first 40 min | 57 min, about 40 of it detection |
| same | `--synthetic`, 720 decoys (chaff 0.2; synthetic rendering is unchanged) | 0.88 GB | 3.2 min |
| 10 min, 48 kHz, 16 channels (1.8 GB as float32) | `--catalog`, 3 decoys, one process (chaff 0.2) | 0.40 to 0.45 GB | 15 s |

The two runs gave the same decoys, audibility report and output peak. Synthetic decoys are rendered in one
process.

Detection is dopplertrack's detector run over the whole recording on every core. `--workers N` uses fewer
processes: slower, less memory.

On macOS the system allocator keeps freed large blocks cached for reuse, and building decoys frees many. The
`chaff` command turns that cache off (`MallocLargeCache=0`, set by restarting itself once). Before this change the
same 12 h `--catalog` run peaked at 2.67 GB. It makes decoy rendering about 27% slower. Render workers run without
the cache too, also when chaff is called from Python: there, with `jobs=2` on 20 min to 2 h of 8 kHz audio, each
worker peaked at 0.62 GB without the cache and at 2.6 to 3.1 GB with it. Detection workers keep the cache. Set
`MallocLargeCache` yourself (`1` keeps the cache) and every process inherits your setting. When calling chaff from
Python on a Mac, the main process keeps the cache unless you set `MallocLargeCache=0` before starting Python.

## Data

The bundled catalog, `chaff/data/mcmaster.npz` (20 MB), is the default source of decoys. It holds the isolated
engine lines of 8 propeller-plane flyovers, de-Dopplered (each line a steady tone) and resampled to 2 kHz, from the
Hamilton Naturalists' Club's McMaster Forest recordings: soundscape recorder HNCSW7, McMaster Forest Nature
Preserve, Ancaster, Ontario, 27 May 2026, 12 hourly files, Internet Archive item
[HNCSW720260527](https://archive.org/details/HNCSW720260527). Only the narrow spectrogram bins around the engine
lines are kept; the rest of the recording is removed, though anything else that fell inside those bins remains.

**Credit: Hamilton Naturalists' Club, via the Internet Archive.** The archive item names the club as creator and
states no licence for the recordings.

The 8 are the flyovers, among the recording's top 10 dopplertrack detections, with 3 or more distinct engine lines.
Six of them are loud enough for the lead tier. The development catalog also held 10 flyovers from the club's Safari
Road recording; none had more than 1 engine line, so none is included. Since chaff 0.3.1 (data version 2) the
clicks in the recording are repaired in the kept lines (see [How it works](#how-it-works));
`scripts/build_bundled_catalog.py` rebuilds the file from the recording and the event list next to it.
[`src/chaff/data/README.md`](src/chaff/data/README.md) lists each stack, and the file carries the same source and
credit in its metadata (`chaff.catalog_info(chaff.bundled_catalog())`).

## Catalog format

A catalog is an `.npz` of plain arrays plus JSON metadata, loaded with `allow_pickle=False`, so opening a catalog
someone else made cannot run code. Each stack keeps its isolated de-Dopplered audio at 2 kHz (float16), its engine
lines, its isolated SNR, its broadband-roar envelope and how much of it the click repair touched (`repaired_s`).
The file carries a format name and version, and may carry catalog-wide JSON metadata
(`chaff.save_catalog(stacks, path, info={...})`, read with `chaff.catalog_info`). Catalogs from chaff 0.1 are
refused and must be rebuilt with `chaff catalog`. Catalogs built before chaff 0.3.1 still load (the format is
unchanged) but keep their clicks; rebuild them with `chaff catalog` to repair them.

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

## Licences

- **Code:** MIT, see [LICENSE](LICENSE).
- **dopplertrack:** MIT, Copyright (c) 2026 MacCallister Higgins. chaff calls its detector. `chaff/detect.py`
  reimplements its spectrogram whitening so that detection can run on a stream (tested to match it).
- **Flight data:** the attack used historical ADS-B data from [adsb.lol](https://adsb.lol/). We understand it is
  published under the Open Database License (ODbL 1.0); verify on the
  [adsblol/globe_history](https://github.com/adsblol/globe_history) releases before reusing it. chaff itself uses
  no flight data.
- **Recordings:** no full recording is redistributed here. The bundled catalog contains audio derived from the
  recordings: the isolated engine lines (float16, 2 kHz) of 8 flyovers from the Hamilton Naturalists' Club's
  McMaster Forest recording, with anything else that fell inside those narrow bins (credit: Hamilton Naturalists'
  Club, via the Internet Archive; no licence is stated on the archive item; see [Data](#data)). Safari Road is
  another club recording on the Internet Archive. Oyster Bay is colsto's raw one-hour reference clip
  ([youtu.be/4CxVxuMVkMc](https://youtu.be/4CxVxuMVkMc)). The railcam and eagle cam are public YouTube live streams.

## Development

```bash
git clone https://github.com/andrew-medrano/chaff && cd chaff
pip install -e ".[test,detect]"
pytest                    # about 6 min on a 12-core Mac with dopplertrack installed
pytest -m "not slow"      # skips real flyover detection, the memory check and the wheel build
```

Tests that need dopplertrack are skipped when it is not installed; the same code paths also run with a stub
detector, so they are covered either way. The ffmpeg test is skipped without ffmpeg. CI runs the suite on Ubuntu
with Python 3.11, 3.12 and 3.13, without dopplertrack.

`docs/make_figures.py` redraws the figures in this README (`docs/images/`, or `--out DIR`), with matplotlib and
Pillow. It needs the McMaster Forest recording and, for the spot-the-fake figure, files from the research
workspace; its docstring lists them. `scripts/build_bundled_catalog.py` rebuilds the bundled catalog from the same
recording, and `scripts/transient_tell.py CATALOG.npz ...` measures the click tell in decoys made from catalogs.

Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements and citation

- colsto, for the [original video](https://www.youtube.com/watch?v=UytYrhYLOKI) and the Oyster Bay recording.
- MacCallister Higgins, for [dopplertrack](https://github.com/mjshiggins/dopplertrack).
- [adsb.lol](https://adsb.lol/) and its feeders, for open flight history.
- The Hamilton Naturalists' Club, for publishing their field recordings, from which the bundled catalog is made.

If you use chaff in research, please cite the repository:

```bibtex
@software{medrano_chaff_2026,
  author = {Medrano, Andrew},
  title  = {chaff: decoy propeller flyovers against acoustic Doppler geolocation},
  year   = {2026},
  url    = {https://github.com/andrew-medrano/chaff},
  version = {0.3.0}
}
```

MIT licence. Copyright (c) 2026 Andrew Medrano.

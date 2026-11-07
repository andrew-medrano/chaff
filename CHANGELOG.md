# Changelog

## 0.2.0 (2026-11-06)

This release is not compatible with 0.1. Catalogs from 0.1 must be rebuilt, and the API and CLI options changed.

### Defaults

- `chaff.protect()`, `chaff.protect_file()` and `chaff protect` now use the tested settings (v3) by default.
  0.1 used the v2 lead level, which failed on Oyster Bay.
  - Lead tier: 6 decoys per hour, fundamental at 28 to 38 dB per-bin SNR, headroom -4 dB, limiter +20 dB re
    background.
  - Quiet tier: 54 per hour, 4 to 10 dB, headroom 6 dB, limiter -7 dB re background.
- Each tier is a `chaff.Tier` (`chaff.LEAD`, `chaff.QUIET`) with matching CLI options (`--lead-per-hour`,
  `--lead-snr`, `--lead-headroom`, `--lead-limit`, and the same for `--quiet-*`). Synthetic decoys use
  `chaff.Synthetic` and `--synth-per-hour`, `--synth-below`. Options for the wrong mode are rejected, for example
  `--synth-*` with `--catalog`.
- Without a catalog, `chaff protect` finds the flyovers in the recording with dopplertrack's detector and builds the
  decoys from them. If none is usable, or dopplertrack is not installed, it falls back to synthetic decoys with a
  boxed warning.

### Loudness

- The quiet tier is rendered and limited on its own, 3 dB under the hearing threshold, and then the lead tier is
  added with its own limiter. In 0.1 and the research recipe both tiers shared one limiter; with the tested
  settings, quiet decoys reached +10.5 dB re background on Oyster Bay. With 0.2 they peak at -7.5 dB, with no
  band-second over the threshold.
- The report and CLI summary give audibility for the quiet tier, the lead tier and both together.

### Memory

- The recording is never held in memory. The input is read in blocks (libsndfile, or an ffmpeg pipe for other
  formats). The background and the detection spectrogram come from a 4 kHz stream, and detection runs in 2 h pieces
  with 10 min overlap. Flyovers are lifted out by reading only the minutes around each one. Decoys are rendered at
  2 kHz, the finished quiet tier waits in a temporary file, and the output is written in blocks through a chunked
  resampler that matches whole-signal `scipy.signal.resample_poly` to 1e-9.
- Peak RSS on 12 h of 48 kHz stereo: 0.94 GB with a catalog, 1.05 GB without (plus detection workers), 0.88 GB with
  synthetic decoys.
- `.wav` outputs over 4 GB are written as RF64.
- On macOS the `chaff` command restarts itself once with `MallocLargeCache=0`, which cut the 12 h catalog run from
  2.67 GB to 0.94 GB peak RSS. Decoy rendering is about 27% slower with it.

### Catalogs

- New file format (version 2): plain arrays plus JSON metadata, loaded with `allow_pickle=False`, with a format
  name and version. Loading a catalog can no longer run code. 0.1 catalogs are refused with a message.
- Each stack stores its isolated SNR, measured at extraction. 0.1 cached it by name, so two stacks with the same
  name could share a level.
- Engine lines are groups of adjacent bins. A stack needs 3 or more distinct lines (0.1 counted 3 bins), and each
  line gets one push gain.
- `chaff catalog` accepts recordings with or without an events JSON; without one it finds the flyovers itself.
- `chaff.build_catalog()`, `chaff.load_catalog()` and `chaff.save_catalog()` are public.

### Errors instead of silent results

- An empty or unusable catalog is an error (0.1 could return the input unchanged), and the CLI exits with status 2.
- Recordings under 60 s and sample rates under 2 kHz are refused. Edge spacing scales down for short recordings.
- Events outside the recording are skipped with a message.
- The output format is checked before any work. Integer and lossy outputs are clipped and the clipped samples
  counted (0.1 let libsndfile wrap them).
- The scipy NOLA warning during extraction is gone.

### Removed

- The v1 texture method (`chaff/catalog.py`) and the 0.1 options `--per-hour`, `--below`, `--ceiling-below` and
  `--decoys-json` (use `--synth-per-hour`, `--synth-below` and `--report`).

### Packaging and docs

- `python -m chaff` works. pyproject has the readme, the MIT SPDX licence, author, project URLs and the `detect`
  and `test` extras.
- README rewritten: method, tiers, results with their source marked, threat model, memory, responsible use, data
  licences.
- Test suite: catalog and self-catalog modes (with a stub detector, and with the real one when dopplertrack is
  installed), synthetic fallback, tier audibility, streaming and block-size equivalence, safe catalog loading,
  errors, short files, five sample rates and up to 16 channels, the CLI as a subprocess, and a peak-memory check.
  GitHub Actions CI on Python 3.11 to 3.13.

## 0.1.0 (2026-11-06)

First version (work in progress): synthetic decoys, v1 texture decoys and v2 whole-stack decoys.

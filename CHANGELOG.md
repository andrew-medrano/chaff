# Changelog

## 0.3.1 (2026-11-07)

### Clicks no longer make dot columns in decoys

- A broadband click or knock in a source recording lands in every kept engine-line bin at once, so each one
  survived in the stack as a dot per line, at the same instant in every line, in every decoy made from it: a column
  of evenly spaced dots no real plane makes. Stack `mcmaster#5` of the bundled catalog had clicks at -272, -205 and
  +230 s on its warped clock, visible in the README's before/after figure near 16:47.
- Stack extraction (`chaff catalog`, `build_catalog`, `--find-flyovers`) now repairs them. On the de-Dopplered
  clock, everything between the lines is looked at in 32 ms windows against its own 0.5 s running median; a window
  whose whole band is up by more than 6 dB on average, or a third of the band by more than 10 dB, is a transient.
  In each 2 s frame it touches, every kept line bin is limited to 3 dB over its own level in the 2.6 s before and
  after. Slower changes (fades, the swell near closest approach, a plane's own bursts of roar and flicker) do not
  register; a flyover without clicks gives the same stack bit for bit. See `chaff/stack.py`.
- `Stack.repaired_s` (saved in catalogs; 0 for older ones) says how much of the warped clock was touched, and
  `chaff catalog` prints it. The catalog format is unchanged: older catalogs load, with their clicks.
- This is a new difference from the tested research recipe (README, "Differences between [R] and [L]"). The attack
  has not been re-run with it.

### Bundled catalog, data version 2

- `chaff/data/mcmaster.npz` is rebuilt with the repair: the same 8 flyovers, engine lines, isolated SNRs and
  credit metadata, so the same decoys are drawn for a seed; 7 of the 8 stacks change around their clicks, most
  of all `mcmaster#5` (64 s of its warped clock touched). 20.2 MB. Its metadata gains `data_version` (2) and
  `built_with`, and each stack's `clicks_repaired_s`.
- `scripts/build_bundled_catalog.py` rebuilds it from the McMaster recording and `scripts/mcmaster_events.json`
  (the 10 detections it was made from), so the file is reproducible.
- `scripts/transient_tell.py` measures the tell in rendered decoys: the share of spectrogram frames in which at
  least half of a decoy's lines (and at least 3) spike 6 dB over their own running median together. With the same
  8 passes per stack, `mcmaster#5` went from 0.92% of frames (3.5 columns per decoy) to 0.08% (0.8), within the
  0 to 0.4% of the other stacks.

### Docs and tests

- README: How it works (step 1), the differences from the research recipe, a limit (what the repair misses), Data
  and Catalog format. Figures redrawn: the dot columns near 16:47 in the before/after figure are gone; the
  how-it-works figure changes very little and spot-the-fake (research-recipe decoys) not at all.
- Tests: a synthetic flyover with injected clicks and a gradual 12 dB dash (clicks removed, the dash and everything
  away from the clicks unchanged, no change without clicks), and the bundled catalog's version, repaired stacks and
  absence of click columns.

## 0.3.0 (2026-11-07)

### A catalog ships with chaff, and is the default

- `chaff/data/mcmaster.npz` (20 MB) is package data: the isolated engine lines of 8 propeller-plane flyovers from
  the Hamilton Naturalists' Club's McMaster Forest recording (Internet Archive item HNCSW720260527). Credit:
  Hamilton Naturalists' Club, via the Internet Archive; the archive item states no licence. It is the development
  catalog without its unusable stacks (2 McMaster flyovers with 2 engine lines, and all 10 Safari Road flyovers),
  so it gives exactly the same decoys for the same seed.
- `chaff protect` and `chaff.protect()` / `protect_file()` use it when no other source is given. They no longer
  need dopplertrack and no longer fall back to synthetic decoys on their own.
- Finding the recording's own flyovers is opt-in: `--find-flyovers` / `find_flyovers=True`. If none is usable it
  falls back to the bundled catalog with a warning (0.2 fell back to synthetic decoys). Without dopplertrack it
  is an error.
- `--catalog` and `--synthetic` are unchanged. The three sources exclude each other; `--synth-*` options need
  `--synthetic`.
- Catalogs may carry catalog-wide JSON metadata: `chaff.save_catalog(..., info=...)`, `chaff.catalog_info()`.
  The bundled catalog's gives its source, link and credit. `chaff.bundled_catalog()` returns its path. The file
  format version is unchanged (2); chaff 0.2 reads these catalogs.
- `.gitignore` still ignores `*.npz`, except the bundled file.

### Parallel rendering

- Decoys are rendered in a pool of worker processes: `-j N` / `--jobs N`, `jobs=` in Python (default:
  `chaff.default_jobs()`, all cores but one, at most 8 and at most as many as fit in half the physical memory at
  0.65 GB each; 1 = one process). Workers load the catalog once; the main process adds the decoys in order, so
  the output is bit-identical for any number of jobs (tested). Runs with fewer than 4 decoys per process render
  in one process. At most 2 rendered decoys per worker wait in the main process.
- On macOS, render workers run with `MallocLargeCache=0`, from the `chaff` command and from Python alike, unless
  `MallocLargeCache` is set (detection workers still get the cache back). Called from Python with the cache, a
  worker peaked at 2.6 to 3.1 GB; without it, 0.62 GB.
- A worker that dies after rendering has started (out of memory, say) gets its own warning; only a pool that
  fails before any decoy comes back is blamed on a missing main guard. Either way the tier is rendered again in one
  process.
- `chaff protect` rejects `-j` with `--synthetic` and `--workers` without `--find-flyovers` (both were ignored).
- `chaff.bundled_catalog()` works when chaff is imported from a zip: the file is copied out once per process.
- A script without an `if __name__ == "__main__":` guard gets a warning and renders in one process; its worker
  processes stop at once instead of running chaff again. Workers read their start-up state (catalog, background)
  from a private temporary file, which keeps the main process from hanging when a worker dies at start (seen
  on Python 3.11).
- The report says how many processes rendered (`render_processes`), and so does the CLI summary.

### Docs and tests

- README: install and quick start without a catalog of your own, a Data section with the credit (and what of
  the recordings the catalog keeps), a new limit (the bundled catalog is public), and the 12 h timings
  re-measured.
- Tests: the bundled catalog loads and carries its credit, the default path uses it, parallel equals serial
  (bundled catalog, given stacks, files), the no-guard script, the worker environment (also from Python), the
  capped default number of jobs, the two pool-failure messages, render-worker memory through the API, the bundled
  catalog from a zip import, and a wheel built from the checkout contains the data. CI also builds a wheel and
  checks it.

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

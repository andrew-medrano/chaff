"""The whole protection run: background, decoy sources, decoys in two tiers, output.

1. One streaming pass over the recording measures its background (and, without a catalog,
   finds its flyovers with dopplertrack's detector).
2. Decoy sources: the catalog's stacks, or stacks lifted from the recording's own flyovers
   (reading only the minutes around each one). Without usable stacks, synthetic decoys.
3. Decoys are rendered at 2 kHz. The quiet tier is rendered and limited on its own, so it stays
   under the hearing threshold; the lead tier gets its own limiter; then the two are added.
4. A second streaming pass writes the recording plus the decoy track, upsampled block by block.

Peak memory is the 2 kHz decoy tracks (0.35 GB each for 12 hours) plus a few blocks, whatever
the recording's rate, channel count or length.
"""
from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import asdict, dataclass

import numpy as np

from . import audibility, detect, stack, synth
from .audio import BLOCK, ArraySource, Writer, open_source, output_format
from .noise import A_SR, K_MASK_DB, FloorBuilder
from .resample import Pull, Resampler
from .synth import SYN_SR

MIN_DURATION_S = 60.0    # shorter recordings are refused: a decoy pass needs room
EDGE_S = 150.0           # decoys' closest approach is kept this far from the ends (scaled down for short files)
SYNTH_EDGE_S = 120.0


class ChaffWarning(UserWarning):
    """Something the user should know: the recording is less protected than the defaults intend."""


@dataclass(frozen=True)
class Tier:
    """One tier of whole-stack decoys.

    per_hour: decoys per hour of recording. snr_db: range the fundamental's per-bin SNR through
    the sweep is drawn from (as a Doppler search measures it). headroom_db: a stack is used at
    level T only if its own isolated SNR is at least T + headroom_db (negative lets a stack be
    turned up). limit_db: limiter ceiling, the most any auditory band of this tier may reach re
    the background (the hearing threshold is K_MASK_DB = -4 dB)."""
    per_hour: float
    snr_db: tuple
    headroom_db: float
    limit_db: float


# The tested recipe (v3, chaffout/FROZEN.md in the research workspace).
LEAD = Tier(per_hour=6.0, snr_db=(28.0, 38.0), headroom_db=-4.0, limit_db=20.0)
QUIET = Tier(per_hour=54.0, snr_db=(4.0, 10.0), headroom_db=6.0, limit_db=K_MASK_DB - 3.0)


@dataclass(frozen=True)
class Synthetic:
    """Synthetic decoys (synth.py): each peaks below_db under the hearing threshold; all of them
    together are limited to limit_db re the background."""
    per_hour: float = 60.0
    below_db: tuple = (4.0, 9.0)
    limit_db: float = K_MASK_DB - 3.0


SYNTHETIC = Synthetic()


class _Spill:
    """A finished float32 track parked in a temporary file, read back slice by slice, so only
    one full-length track is in memory at a time."""

    def __init__(self, y):
        fd, self.path = tempfile.mkstemp(prefix="chaff-", suffix=".f32")
        with os.fdopen(fd, "wb") as fh:
            for i in range(0, len(y), BLOCK * 4):
                fh.write(np.ascontiguousarray(y[i:i + BLOCK * 4], np.float32).tobytes())
        self.n = len(y)

    def __len__(self):
        return self.n

    def __getitem__(self, sl):
        i0, i1, step = sl.indices(self.n)
        assert step == 1
        with open(self.path, "rb") as fh:
            fh.seek(4 * i0)
            return np.fromfile(fh, np.float32, max(i1 - i0, 0))

    def close(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


def _mono(b):
    return b[:, 0] if b.shape[1] == 1 else b.mean(axis=1)


class _Run:
    def __init__(self, log):
        self.log = log or (lambda msg: None)
        self.warnings = []

    def warn(self, msg):
        self.warnings.append(msg)
        warnings.warn(msg, ChaffWarning, stacklevel=4)
        self.log("WARNING: " + msg)


def analyse(source, find_flyovers=False, detect_top=15, max_events=20, workers=None, log=None):
    """One pass over the recording: (frames, background, detected events or None)."""
    rs, fb = Resampler(source.sr, A_SR), FloorBuilder()
    det = detect.Detector(top=detect_top, workers=workers, log=log) if find_flyovers else None
    n = 0
    for b in source.blocks(BLOCK):
        n += len(b)
        x4 = rs.push(_mono(b))
        fb.push(x4)
        if det:
            det.push(x4)
    x4 = rs.flush()
    fb.push(x4)
    if det:
        det.push(x4)
    if n / source.sr < MIN_DURATION_S:
        raise ValueError(f"recording is {n / source.sr:.1f} s long; chaff needs at least {MIN_DURATION_S:.0f} s")
    return n, fb.finish(), (det.finish(max_events) if det else None)


def extract_stacks(source, n, floor, events, top=None, log=None, notes=None):
    """Stacks of the given events (dicts with t_cpa, f0, v, d), reading only the minutes around
    each. Events outside the recording are skipped with a message."""
    log = log or (lambda msg: None)
    read = lambda i0, i1: _mono(source.read(i0, i1))
    out = []
    for i, e in enumerate(events[:top] if top else events):
        try:
            k = stack.extract(read, source.sr, n, e, floor, name=f"{source.name}#{i + 1}")
        except (KeyError, TypeError) as err:
            msg = f"{source.name} event #{i + 1}: missing field {err}; skipped"
        except ValueError as err:
            msg = f"{source.name} event #{i + 1}: {err}; skipped"
        else:
            out.append(k)
            log(f"  {k.name}: t_cpa {e['t_cpa']:.0f} s, f0 {k.f0:.1f} Hz, {k.n_lines} engine lines, "
                f"isolated SNR {k.iso_snr_db:.1f} dB{'' if k.usable else ' (fewer than 3 lines: not used)'}")
            continue
        log("  " + msg)
        if notes is not None:
            notes.append(msg)
    return out


def _load_catalog(catalog):
    if isinstance(catalog, (str, bytes)) or hasattr(catalog, "__fspath__"):
        return stack.load(catalog), os.fspath(catalog)
    return list(catalog), "given"


def _describe(stacks):
    if not stacks:
        return "it has no stacks"
    u = [k for k in stacks if k.usable]
    if not u:
        return f"none of its {len(stacks)} stacks has {stack.MIN_LINES} or more engine lines"
    return (f"{len(u)} of {len(stacks)} stacks have {stack.MIN_LINES}+ engine lines, "
            f"with isolated SNR up to {max(k.iso_snr_db for k in u):.1f} dB")


def _tier_info(t: Tier, n):
    return dict(asdict(t), n=n)


def _run(source, write, catalog, seed, lead, quiet, synthetic, synth_opts, workers, max_events, log):
    run = _Run(log)
    if source.sr < SYN_SR:
        raise ValueError(f"sample rate {source.sr} Hz is too low (decoys need at least {SYN_SR} Hz)")
    if source.frames is not None and source.frames / source.sr < MIN_DURATION_S:
        raise ValueError(f"recording is {source.frames / source.sr:.1f} s long; chaff needs at least {MIN_DURATION_S:.0f} s")
    rep = dict(mode=None, catalog=None, sr=source.sr, channels=source.channels)
    stacks, find = None, False
    if synthetic:
        rep["catalog"] = dict(source="none (synthetic decoys requested)")
    elif catalog is not None:
        stacks, where = _load_catalog(catalog)
        if not any(k.usable for k in stacks):
            raise ValueError(f"catalog {where} is unusable: {_describe(stacks)}")
        rep["catalog"] = dict(source=where, stacks=len(stacks), usable=sum(k.usable for k in stacks))
    elif detect.available():
        find = True
    else:
        run.warn("no catalog given and dopplertrack is not installed, so chaff cannot use real flyovers: "
                 "falling back to synthetic decoys, which are easier to tell from real planes. "
                 "Install chaff[detect] or pass a catalog.")
        rep["catalog"] = dict(source="none (dopplertrack not installed)")

    run.log(f"pass 1/2: measuring the background{' and finding flyovers' if find else ''}")
    n, floor, events = analyse(source, find_flyovers=find, max_events=max_events, workers=workers, log=run.log)
    dur = n / source.sr
    rep["duration_s"] = dur
    if find:
        run.log(f"lifting {len(events)} detected flyovers out of the recording")
        notes = []
        stacks = extract_stacks(source, n, floor, events, log=run.log, notes=notes)
        rep["catalog"] = dict(source="self (flyovers detected in this recording)", detected=len(events),
                              stacks=len(stacks), usable=sum(k.usable for k in stacks), notes=notes,
                              events=events,
                              used=[dict(name=k.name, f0=k.f0, n_lines=k.n_lines, iso_snr_db=k.iso_snr_db)
                                    for k in stacks if k.usable])
        if not any(k.usable for k in stacks):
            run.warn(f"no flyover in this recording has a clear engine-line stack ({len(events)} detected; "
                     f"{_describe(stacks)}): falling back to synthetic decoys, which are easier to tell from "
                     "real planes. Build a catalog from recordings with loud propeller planes for better decoys.")
            stacks = None

    if stacks is not None:
        el = {name: stack.eligible(stacks, t.snr_db, t.headroom_db) for name, t in (("lead", lead), ("quiet", quiet))}
        if not el["lead"] and not el["quiet"]:
            msg = (f"no stack is loud enough for either tier ({_describe(stacks)}; lead needs "
                   f"{lead.snr_db[0] + lead.headroom_db:.0f} dB, quiet {quiet.snr_db[0] + quiet.headroom_db:.0f} dB)")
            if not find:
                raise ValueError(f"catalog is unusable: {msg}")
            run.warn(msg + ": falling back to synthetic decoys, which are easier to tell from real planes")
            stacks = None
        else:
            for name, t in (("lead", lead), ("quiet", quiet)):
                if not el[name] and t.per_hour > 0:
                    run.warn(f"{name} tier skipped: no stack reaches {t.snr_db[0] + t.headroom_db:.0f} dB isolated SNR "
                             f"({_describe(stacks)})")

    if stacks is not None:
        rep["mode"] = "stack"
        edge = min(EDGE_S, 0.25 * dur)
        dl = stack.sample(stacks, dur, lead.per_hour, lead.snr_db, lead.headroom_db, seed=seed, edge_s=edge,
                          tier="lead") if el["lead"] else []
        dq = stack.sample(stacks, dur, quiet.per_hour, quiet.snr_db, quiet.headroom_db, seed=seed + 1000,
                          edge_s=edge, tier="quiet") if el["quiet"] else []
        if not dl and not dq:
            raise ValueError(f"no decoys placed: at {lead.per_hour:g} + {quiet.per_hour:g} per hour a "
                             f"{dur / 60:.1f} min recording gets none")
        # quiet tier on its own, limited under the hearing threshold, then parked on disk
        run.log(f"rendering {len(dq)} quiet decoys")
        tq, sk_q = stack.render(dq, stacks, n, source.sr, floor, progress=_every(run.log, len(dq)))
        audibility.limit(tq, SYN_SR, floor, quiet.limit_db)
        dq = [d for d in dq if d not in sk_q]
        aud_q = audibility.report(tq, SYN_SR, floor) if dq else None
        spill = _Spill(tq)
        del tq
        try:
            run.log(f"rendering {len(dl)} lead decoys")
            track, sk_l = stack.render(dl, stacks, n, source.sr, floor, progress=_every(run.log, len(dl)))
            audibility.limit(track, SYN_SR, floor, lead.limit_db)
            dl = [d for d in dl if d not in sk_l]
            if sk_q or sk_l:
                run.log(f"  {len(sk_q) + len(sk_l)} decoys did not fit in the recording and were left out")
            decoys = sorted(dl + dq, key=lambda d: d.t_cpa)
            if not decoys:
                raise ValueError("no decoy fitted in the recording")
            run.log("measuring audibility")
            aud = dict(threshold_db=K_MASK_DB, quiet=aud_q,
                       lead=audibility.report(track, SYN_SR, floor) if dl else None,
                       combined=audibility.report((track, spill), SYN_SR, floor))
            for i in range(0, len(track), BLOCK * 4):          # lead + quiet, in place
                track[i:i + BLOCK * 4] += spill[i:i + BLOCK * 4]
        finally:
            spill.close()
        rep["tiers"] = dict(lead=_tier_info(lead, len(dl)), quiet=_tier_info(quiet, len(dq)))
    else:
        rep["mode"] = "synthetic"
        decoys = synth.sample_decoys(dur, per_hour=synth_opts.per_hour, below_db=tuple(synth_opts.below_db),
                                     seed=seed, edge_s=min(SYNTH_EDGE_S, 0.25 * dur))
        if not decoys:
            raise ValueError(f"no decoys placed: at {synth_opts.per_hour:g} per hour a {dur / 60:.1f} min recording gets none")
        run.log(f"rendering {len(decoys)} synthetic decoys")
        track = synth.render(decoys, n, source.sr, floor)
        audibility.limit(track, SYN_SR, floor, synth_opts.limit_db)
        run.log("measuring audibility")
        a = audibility.report(track, SYN_SR, floor)
        aud = dict(threshold_db=K_MASK_DB, synthetic=a, combined=a)
        rep["tiers"] = dict(synthetic=dict(asdict(synth_opts), n=len(decoys)))
    rep["n_decoys"] = len(decoys)
    rep["audibility"] = aud
    rep["warnings"] = run.warnings
    run.log("pass 2/2: writing")
    write(track, n)
    return decoys, rep


def _every(log, n, step=50):
    if n < step:
        return None
    return lambda j, tot: log(f"  {j}/{tot}") if j % step == 0 or j == tot else None


def protect(x, sr, catalog=None, *, seed=0, lead=LEAD, quiet=QUIET, synthetic=False, synth_opts=SYNTHETIC,
            workers=None, max_events=20, log=None):
    """Add decoy flyovers to an in-memory recording.

    x: mono (N,) or multichannel (N, C) audio. The background and decoys are computed on the
    channel mean and the same decoy track is added to every channel.

    catalog: a stack catalog (path, or list of stack.Stack) from `chaff catalog`. If None, decoy
    sources are lifted from this recording's own flyovers (needs dopplertrack); if it has none
    with a clear engine-line stack, or dopplertrack is missing, synthetic decoys are used and a
    ChaffWarning is raised. synthetic=True uses synthetic decoys outright.

    lead, quiet: the two tiers (Tier); the defaults are the tested recipe. synth_opts: settings
    for synthetic decoys. workers: processes for flyover detection (None = all cores).

    Returns (y, decoys, report): protected audio with the shape of x (float32), the list of
    decoys (stack.StackDecoy, or synth.Decoy for synthetic decoys), and a report dict with
    per-tier audibility (audibility.report) under report["audibility"].
    """
    src = ArraySource(x, sr)
    out = {}

    def write(track, n):
        y = np.array(src.x, np.float32, copy=True)
        p = Pull(track, SYN_SR, src.sr)
        for i in range(0, n, BLOCK):
            m = min(BLOCK, n - i)
            y[i:i + m] += p.read(m)[:, None]
        out["y"] = y if np.ndim(x) == 2 else y[:, 0]

    decoys, rep = _run(src, write, catalog, seed, lead, quiet, synthetic, synth_opts, workers, max_events, log)
    return out["y"], decoys, rep


def protect_file(input, output, catalog=None, *, seed=0, lead=LEAD, quiet=QUIET, synthetic=False,
                 synth_opts=SYNTHETIC, subtype=None, workers=None, max_events=20, log=None):
    """protect() for files, streamed: the recording is read in blocks twice and never held in
    memory. Returns (decoys, report); report["output"] says how it was written."""
    if os.path.abspath(input) == os.path.abspath(output):
        raise ValueError("output would overwrite the input")
    output_format(output, subtype)                  # fail now, not after an hour of work
    src = open_source(input)
    rep_out = {}

    def write(track, n):
        p = Pull(track, SYN_SR, src.sr)
        with Writer(output, src.sr, src.channels, frames=n, subtype=subtype) as w:
            m = 0
            for b in src.blocks(BLOCK):
                w.write(b + p.read(len(b))[:, None])
                m += len(b)
        rep_out.update(path=os.fspath(output), format=w.format, subtype=w.subtype, frames=m, peak=w.peak,
                       clipped_samples=w.n_clipped)
        if w.rf64:
            rep_out["note"] = "written as RF64 (a .wav this large does not fit the WAV header)"

    decoys, rep = _run(src, write, catalog, seed, lead, quiet, synthetic, synth_opts, workers, max_events, log)
    rep["input"] = os.fspath(input)
    rep["output"] = rep_out
    if rep_out.get("clipped_samples"):
        msg = (f"{rep_out['clipped_samples']} samples clipped (peak {rep_out['peak']:.3f}) in "
               f"{rep_out['subtype']} output; write .wav (float) to keep them")
        rep["warnings"].append(msg)
        warnings.warn(msg, ChaffWarning, stacklevel=2)
    return decoys, rep


def build_catalog(recordings, top=10, workers=None, log=None):
    """Stacks from recordings. recordings: list of (path or in-memory (x, sr), events or None).
    events: list of dicts with t_cpa, f0, v, d (e.g. `dopplertrack detect --out`); None finds the
    flyovers with dopplertrack's detector. Uses the first `top` events of each recording.
    Returns (stacks, notes)."""
    log = log or (lambda msg: None)
    out, notes = [], []
    for rec, events in recordings:
        src = ArraySource(*rec) if isinstance(rec, tuple) else open_source(rec)
        find = events is None
        log(f"{src.name}: measuring the background{' and finding flyovers' if find else ''}")
        n, floor, found = analyse(src, find_flyovers=find, detect_top=max(top, 15), max_events=top,
                                  workers=workers, log=log)
        evs = found if find else list(events)
        if find:
            log(f"  {len(evs)} flyovers detected")
        out += extract_stacks(src, n, floor, evs, top=top, log=log, notes=notes)
    return out, notes

"""Decoys from whole real flyovers, every engine line included.

A propeller plane is not one tone but a stack: blade rate, its harmonics, engine orders in
between, all sliding by the same Doppler ratio. To keep the stack, a real flyover is
*de-Dopplered*: the audio is resampled on a warped clock (d tau / dt = heard / rest frequency)
so every line in the stack becomes a steady tone. Steady lines are then easy to separate from
birds, voices and wind (they stay put; those do not), so only bins that are lines through the
pass are kept. A decoy re-Dopplers that isolated stack onto a new pass (new blade rate, speed,
distance, time), with lines pushed up or down individually and sections faded, and levels it to
the new recording's background.

A catalog is a list of Stack objects, saved as an .npz of plain arrays plus JSON metadata (no
pickle, so loading a catalog someone else made cannot run code).
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import asdict, dataclass

import numpy as np
import scipy.fft as sp_fft
import scipy.signal as ss
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d

from .noise import NoiseFloor
from .resample import resample
from .synth import C, SYN_SR, _smooth_noise, straight_pass, track_len

HI = 8000            # working rate for the warp (keeps interpolation exact below 1 kHz)
NFFT = 4096          # 2 s at 2 kHz: line resolution like the search's
F_TOP = 950.0
RATE = 20.0          # control rate of the broadband-roar envelope, Hz
MIN_LINES = 3        # a stack needs this many distinct engine lines to be used as a decoy source
FORMAT = "chaff-stacks"
FORMAT_VERSION = 2


def stft_gain(x, g, n=NFFT, hop=NFFT // 8, block=256):
    """istft(stft(x) * g[:, None]) with scipy's conventions (Hann, 'spectrum' scaling, no
    boundary padding), computed a block of frames at a time so a long flyover needs only a few
    MB. g: real gain per frequency bin (n // 2 + 1). Equal to the scipy round trip to rounding.

    Like scipy's (boundary=False), the first sample has zero window sum and is left as is."""
    x = np.asarray(x)
    win = ss.get_window("hann", n)
    K = (len(x) - n) // hop + 1
    out = np.zeros(n + (K - 1) * hop)
    for k0 in range(0, K, block):
        k1 = min(k0 + block, K)
        _, _, Z = ss.stft(x[k0 * hop:(k1 - 1) * hop + n], SYN_SR, window="hann", nperseg=n, noverlap=n - hop,
                          boundary=None, padded=False)
        xs = sp_fft.irfft(Z * g[:, None], axis=0, n=n)[:n]
        xs *= win.sum()
        for j in range(k1 - k0):
            i = (k0 + j) * hop
            out[i:i + n] += xs[:, j] * win
    norm = np.zeros(len(out)); w2 = win ** 2
    for k in range(K):
        norm[k * hop:k * hop + n] += w2
    out /= np.where(norm > 1e-10, norm, 1.0)
    return out


def _stft_power_rows(x, rows, n=NFFT, hop=NFFT // 8, block=512):
    """(f, t, |STFT|^2 of the selected frequency rows) with 'spectrum' scaling, block by block."""
    K = (len(x) - n) // hop + 1
    Ps, f = [], None
    for k0 in range(0, K, block):
        k1 = min(k0 + block, K)
        f, _, Z = ss.stft(x[k0 * hop:(k1 - 1) * hop + n], SYN_SR, window="hann", nperseg=n, noverlap=n - hop,
                          boundary=None, padded=False, scaling="spectrum")
        Ps.append(np.abs(Z[rows(f)]) ** 2)
    t = (n / 2 + np.arange(K) * hop) / SYN_SR
    return f, t, np.concatenate(Ps, axis=1)


@dataclass
class Stack:
    name: str
    f0: float; v: float; d: float
    tau_cpa: float                 # warped time of closest approach, s from the start of s
    s: np.ndarray                  # isolated, de-Dopplered stack at SYN_SR
    lines: np.ndarray              # (L, 2) rest-frequency range [lo, hi] of each kept engine line, Hz
    iso_snr_db: float              # per-bin SNR of the isolated fundamental through the sweep
    floor_ref: float               # source background per bin at f0 near closest approach
    bb: np.ndarray                 # broadband excess power re background (450-950 Hz), on u_bb
    u_bb: np.ndarray               # normalised time (t - t_cpa) * v / d

    @property
    def lines_hz(self) -> np.ndarray:
        return self.lines.mean(axis=1) if len(self.lines) else np.zeros(0)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def usable(self) -> bool:
        return self.n_lines >= MIN_LINES


def _group_lines(f, line, keep):
    """Engine lines: each contiguous run of kept bins (line bins +- 1) that holds line bins."""
    out = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        fl = f[a:b][line[a:b]]
        if len(fl):
            out.append((fl.min(), fl.max()))
    return np.array(out, np.float32).reshape(-1, 2)


def _broadband_excess(x2, t0, t_cpa, v, d, f0, floor: NoiseFloor):
    """Broadband (450-950 Hz) excess power re background near closest approach, median-smoothed
    so bird calls do not count, on a RATE grid. x2 is the flyover at SYN_SR starting at t0."""
    ts = t0 + np.arange(len(x2)) / SYN_SR
    tc = np.arange(ts[0], ts[-1], 1 / RATE)
    tc = tc[:min(len(tc), -(-len(x2) // int(SYN_SR / RATE)))]
    sob = ss.butter(4, [450, 950], "bandpass", fs=SYN_SR, output="sos")
    p = ss.sosfilt(sob, x2) ** 2; k = SYN_SR
    pe = np.array([p[i:i + k].mean() for i in range(0, len(p) - k, k)]); tb = ts[0] + 0.5 + np.arange(len(pe))
    fb = floor.f[(floor.f >= 450) & (floor.f <= 950)]
    var = (4 / 3) * floor.at(fb[:, None], tb[None, :]).sum(0)
    ex = np.clip(median_filter(pe / var, 7) - 1, 0, None) * (np.abs(tb - t_cpa) < 90)
    return np.interp(tc, tb, ex).astype(np.float32), ((tc - t_cpa) * v / d)


def _iso_snr_db(s, tau_cpa, f0, v, d, floor_ref) -> float:
    """Per-bin SNR of the isolated fundamental through the sweep, measured on the de-Dopplered
    stack itself (where it is a steady line at f0) with the search's resolution. This is what a
    decoy's level is calibrated on, so the level a search sees matches the target."""
    f, tt, Pj = _stft_power_rows(s, lambda f: np.abs(f - f0) < 1.0)
    core = np.abs(tt - tau_cpa) < max(0.3 * d / v, 10)
    if not (core.any() and len(Pj)):
        return -99.0
    return float(10 * np.log10(np.median(Pj[:, core].max(0)) / floor_ref + 1e-30))


def extract(read, sr, n_samples, ev, floor: NoiseFloor, half_s=300.0, name="", line_db=6.0) -> Stack:
    """Isolated de-Dopplered stack of one flyover.

    read(i0, i1) returns mono samples [i0, i1) of the recording (n_samples long, at sr), so only
    the ~11 minutes around the flyover are read. ev: dict with t_cpa (heard, s), f0, v, d, as a
    detect / fit event. floor: the recording's background. Raises ValueError for an event
    outside the recording."""
    t_cpa, f0, v, d = (float(ev[k]) for k in ("t_cpa", "f0", "v", "d"))
    dur = n_samples / sr
    if not (0.0 <= t_cpa <= dur):
        raise ValueError(f"event at {t_cpa:.1f} s is outside the recording (0-{dur:.1f} s)")
    a = max(t_cpa - half_s - 20, 0.0); b = min(t_cpa + half_s + 20, dur)
    i0, i1 = int(a * sr), int(b * sr)
    seg = np.asarray(read(i0, i1), np.float32)
    xh = resample(seg, sr, HI).astype(np.float64)
    t = a + np.arange(len(xh)) / HI
    tc = np.arange(t[0], t[-1], 0.05); fc, _ = straight_pass(tc, f0, v, d, t_cpa)
    tau = np.cumsum(np.interp(t, tc, fc) / f0) / HI; tau -= tau[0]
    tau_cpa = float(np.interp(t_cpa, t, tau))
    tw = np.interp(np.arange(0, tau[-1], 1 / HI), tau, t)
    del tau
    s = np.interp(tw, t, xh)                                     # x on the warped clock: lines are steady
    del tw, t, xh
    s2 = ss.resample_poly(s, 1, HI // SYN_SR)
    del s
    # keep bins that are lines through the pass: median power in the core well above nearby bins
    hop = NFFT // 8
    tt = (NFFT / 2 + np.arange((len(s2) - NFFT) // hop + 1) * hop) / SYN_SR
    core = np.flatnonzero(np.abs(tt - tau_cpa) < max(0.6 * d / v, 40))
    f, _, Z = ss.stft(s2[core[0] * hop:core[-1] * hop + NFFT], SYN_SR, window="hann", nperseg=NFFT, noverlap=NFFT - hop, boundary=None, padded=False)
    med = np.median(np.abs(Z) ** 2, axis=1)
    del Z
    nb = np.exp(uniform_filter1d(np.log(med + 1e-30), 41))       # local level around each bin
    line = (10 * np.log10(med / nb) > line_db) & (f > 40) & (f < F_TOP / 1.3)
    keep = maximum_filter1d(line.astype(float), 3) > 0              # line +- one bin
    s_iso = stft_gain(s2, keep.astype(float))[:len(s2)].astype(np.float32)
    floor_ref = float(floor.at(np.array([f0]), np.array([t_cpa]))[0])
    # broadband roar, from the +-(half_s + 10) s around closest approach
    a2 = max(t_cpa - half_s - 10, 0.0); b2 = min(t_cpa + half_s + 10, dur)
    x2 = resample(seg[int(a2 * sr) - i0:int(b2 * sr) - i0], sr, SYN_SR).astype(np.float64)
    bb, u_bb = _broadband_excess(x2, a2, t_cpa, v, d, f0, floor)
    return Stack(name, f0, v, d, tau_cpa, s_iso, _group_lines(f, line, keep),
                 _iso_snr_db(s_iso, tau_cpa, f0, v, d, floor_ref), floor_ref, bb, u_bb)


def save(stacks, path):
    """Write a catalog. Returns the path written (numpy adds .npz if missing)."""
    path = os.fspath(path)
    if not path.endswith(".npz"):
        path += ".npz"
    meta = [dict(name=k.name, f0=float(k.f0), v=float(k.v), d=float(k.d), tau_cpa=float(k.tau_cpa),
                 iso_snr_db=float(k.iso_snr_db), floor_ref=float(k.floor_ref)) for k in stacks]
    arrays = {}
    for i, k in enumerate(stacks):
        arrays[f"s{i}"] = k.s.astype(np.float16)
        arrays[f"lines{i}"] = np.asarray(k.lines, np.float32).reshape(-1, 2)
        arrays[f"bb{i}"] = np.asarray(k.bb, np.float32)
        arrays[f"ubb{i}"] = np.asarray(k.u_bb, np.float64)
    np.savez_compressed(path, format=np.array(FORMAT), version=np.array(FORMAT_VERSION),
                        meta=np.array(json.dumps(meta)), **arrays)
    return path


def load(path) -> list[Stack]:
    """Read a catalog written by save() / `chaff catalog`. Never unpickles."""
    try:
        z = np.load(path, allow_pickle=False)
    except (zipfile.BadZipFile, ValueError, EOFError) as e:
        raise ValueError(f"{path} is not a chaff catalog ({e})") from e
    if not hasattr(z, "files"):
        raise ValueError(f"{path} is not a chaff catalog (not an .npz archive)")
    with z:
        if "format" not in z.files or str(z["format"]) != FORMAT:
            raise ValueError(f"{path} is not a chaff catalog (catalogs made before chaff 0.2 must be rebuilt with `chaff catalog`)")
        ver = int(z["version"])
        if ver != FORMAT_VERSION:
            raise ValueError(f"{path} is catalog format {ver}; this chaff reads format {FORMAT_VERSION}. Rebuild it with `chaff catalog`.")
        meta = json.loads(str(z["meta"]))
        return [Stack(m["name"], m["f0"], m["v"], m["d"], m["tau_cpa"], z[f"s{i}"].astype(np.float32),
                      z[f"lines{i}"].reshape(-1, 2), m["iso_snr_db"], m["floor_ref"], z[f"bb{i}"], z[f"ubb{i}"])
                for i, m in enumerate(meta)]


@dataclass
class StackDecoy:
    t_cpa: float; f0: float; v: float; d: float
    src: int
    level_db: float               # target sweep SNR of the fundamental
    seed: int
    tier: str = ""
    src_name: str = ""

    def to_dict(self):
        return asdict(self)


def eligible(stacks, snr_db, headroom_db):
    """Indices of stacks a tier can draw from at its lowest target level."""
    return [i for i, k in enumerate(stacks) if k.usable and k.iso_snr_db >= snr_db[0] + headroom_db]


def sample(stacks, duration_s, per_hour=6.0, snr_db=(28.0, 38.0), headroom_db=-4.0, f_band=(55.0, 330.0), seed=0,
           edge_s=150.0, tier=""):
    """New blade rate, speed and distance drawn independently around the source's; the stack is
    pitch-shifted and time-stretched separately (phase vocoder on the steady de-Dopplered lines).
    A stack is used for a decoy at level T only if its own isolated SNR is at least T + headroom_db
    and it has at least MIN_LINES engine lines."""
    rng = np.random.default_rng(seed); n = int(round(per_hour * duration_s / 3600.0)); out = []; tries = 0
    while len(out) < n and tries < 80 * max(n, 1):
        tries += 1
        T = rng.uniform(*snr_db)
        ok = [i for i, k in enumerate(stacks) if k.iso_snr_db >= T + headroom_db and k.usable]
        if not ok:
            continue
        i = ok[rng.integers(len(ok))]; k = stacks[i]
        f0 = k.f0 * rng.uniform(0.6, 1.2); v = k.v * rng.uniform(0.8, 1.25); d = k.d * np.exp(rng.uniform(np.log(0.6), np.log(1.7)))
        if not (f_band[0] <= f0 * C / (C + v) and f0 * C / (C - v) <= f_band[1] and 300 < d < 12000):
            continue
        t = rng.uniform(edge_s, max(duration_s - edge_s, edge_s + 1))
        if any(abs(o.t_cpa - t) < 45 and abs(np.log(o.f0 / f0)) < 0.08 for o in out):
            continue
        out.append(StackDecoy(float(t), float(f0), float(v), float(d), int(i), float(T), int(rng.integers(2**31)), tier, k.name))
    return sorted(out, key=lambda o: o.t_cpa)


def pv_stretch(x, st, n=NFFT):
    """Phase-vocoder time stretch by st (> 1 longer). Clean here: de-Dopplered lines are steady."""
    hs = n // 8; w = np.hanning(n)
    n_out = int(len(x) * st); K = max((n_out - n) // hs, 1)
    pos = np.minimum((np.arange(K + 1) * hs / st).astype(int), len(x) - n - 1)
    omega = 2 * np.pi * np.arange(n // 2 + 1) / n
    y = np.zeros(K * hs + n); wsum = np.zeros_like(y)
    prev = np.fft.rfft(w * x[pos[0]:pos[0] + n]); phase = np.angle(prev)
    for k in range(K):
        X = np.fft.rfft(w * x[pos[k]:pos[k] + n])
        if k:
            ha = max(pos[k] - pos[k - 1], 1)
            dphi = np.angle(X) - np.angle(prev) - omega * ha
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            phase = phase + (omega + dphi / ha) * hs
        prev = X
        y[k * hs:k * hs + n] += w * np.fft.irfft(np.abs(X) * np.exp(1j * phase), n)
        wsum[k * hs:k * hs + n] += w * w
    return (y / np.maximum(wsum, 1e-3))[:n_out]


def _restack(k: Stack, rng, push_db=6.0):
    """Push each engine line up or down (-push_db .. +push_db/2) in the de-Dopplered stack; all
    bins of one line get the same gain."""
    f = np.fft.rfftfreq(NFFT, 1 / SYN_SR)
    g = np.ones(len(f)); df = SYN_SR / NFFT
    for lo, hi in k.lines:
        j = (f >= lo - 1.5 * df) & (f <= hi + 1.5 * df)
        g[j] = 10 ** (rng.uniform(-push_db, push_db / 2) / 20)
    s = stft_gain(k.s, g)[:len(k.s)]
    s[:NFFT] *= np.linspace(0, 1, NFFT); s[-NFFT:] *= np.linspace(1, 0, NFFT)
    return s


def _calibrate(sig, t_start, dc, floor: NoiseFloor, max_db=12.0):
    """Gain that puts the fundamental's per-bin SNR through the sweep (|u| < 0.3) at dc.level_db."""
    f, tt, Z = ss.stft(sig, SYN_SR, window="hann", nperseg=NFFT, noverlap=NFFT - SYN_SR, boundary=None, padded=False, scaling="spectrum")
    tt = tt + t_start; m = np.abs((tt - dc.t_cpa) * dc.v / dc.d) < 0.3
    if m.sum() < 3:
        return 1.0
    fc, _ = straight_pass(tt[m], dc.f0, dc.v, dc.d, dc.t_cpa)
    P = np.abs(Z[:, m]) ** 2
    v = [P[np.abs(f - q) < 1.0, j].max() for j, q in enumerate(fc)]
    got = 10 * np.log10(np.median(np.array(v) / floor.at(fc, tt[m])) + 1e-30)
    return 10 ** (np.clip(dc.level_db - got, -max_db, max_db) / 20)


def render_one(dc, k: Stack, n_syn, floor: NoiseFloor, max_span_s=420.0):
    """One decoy as (first sample index at SYN_SR, float32 signal), or None if it does not fit."""
    sob = ss.butter(4, [450, 950], "bandpass", fs=SYN_SR, output="sos")
    rng = np.random.default_rng(dc.seed)
    a = dc.f0 / k.f0
    st = a * (dc.d / dc.v) / (k.d / k.v)                             # stretch so that pitch a and time scale are independent
    sh = ss.resample_poly(pv_stretch(_restack(k, rng), st), HI // SYN_SR, 1)   # back up to the warp rate
    tau_src = np.arange(len(sh)) / HI; tau_cpa = k.tau_cpa * st
    # new heard times around the new CPA; warped source time T(t') = tau_cpa + a * int ratio'(t') dt'
    t0 = max(dc.t_cpa - max_span_s, 0.0); t1 = min(dc.t_cpa + max_span_s, n_syn / SYN_SR - 1)
    tc = np.arange(t0, t1, 0.05); fc, r = straight_pass(tc, dc.f0, dc.v, dc.d, dc.t_cpa)
    Tc = a * np.concatenate([[0], np.cumsum(0.5 * (fc[1:] + fc[:-1]) / dc.f0 * 0.05)])
    Tc += tau_cpa - np.interp(dc.t_cpa, tc, Tc)
    ok = (Tc >= 0) & (Tc <= tau_src[-1])
    if ok.sum() < 600:
        return None
    tc, Tc, r, fc = tc[ok], Tc[ok], r[ok], fc[ok]
    fcut = 0.95 * 980.0 / (a * fc.max() / dc.f0)                     # nothing may land above 1 kHz after the warp
    sh = ss.sosfiltfilt(ss.butter(8, min(fcut, 0.45 * HI), fs=HI, output="sos"), sh)
    i0 = int(round(tc[0] * SYN_SR)); ts = i0 / SYN_SR + np.arange(int((tc[-1] - tc[0]) * SYN_SR)) / SYN_SR
    sig = np.interp(np.interp(ts, tc, Tc), tau_src, sh)
    del sh, tau_src
    # level: source fundamental's sweep SNR -> target, scaled from source to this background
    g = 10 ** ((dc.level_db - k.iso_snr_db) / 20) * np.sqrt(floor.at(np.array([dc.f0]), np.array([dc.t_cpa]))[0] / k.floor_ref)
    fade = 10 ** (2.5 * _smooth_noise(rng, len(tc), 10 * 20) / 20) * np.sin(np.pi * np.clip((tc - tc[0]) / (tc[-1] - tc[0]), 0, 1)) ** 0.5
    sig = sig * g * np.interp(ts, tc, fade)
    sig *= _calibrate(sig, ts[0], dc, floor)      # land the fundamental exactly at the target, as a search measures it
    # broadband roar from the source flyover
    u = (tc - dc.t_cpa) * dc.v / dc.d
    ex = np.interp(u, k.u_bb, k.bb, left=0, right=0) / 10 ** (k.iso_snr_db / 10)
    fb = floor.f[(floor.f >= 450) & (floor.f <= 950)]
    var = (4 / 3) * floor.at(fb[:, None], tc[None, :]).sum(0)
    amp = np.sqrt(np.clip(10 ** (dc.level_db / 10) * ex * var, 0, None))
    if amp.max() > 0:
        nz = ss.sosfilt(sob, rng.standard_normal(len(ts) + SYN_SR))[SYN_SR:]
        sig = sig + np.interp(ts, tc, amp) * nz / (nz.std() + 1e-20)
    return i0, sig.astype(np.float32)


def render(decoys, stacks, n_samples, sr, floor: NoiseFloor, max_span_s=420.0, progress=None):
    """Sum of the decoys as a float32 track at SYN_SR for a recording of n_samples at sr.
    The caller limits it (audibility.limit). Decoys that do not fit in the recording are skipped
    (returned second)."""
    n_syn = track_len(n_samples, sr)
    y = np.zeros(n_syn, np.float32)
    skipped = []
    for j, dc in enumerate(decoys):
        out = render_one(dc, stacks[dc.src], n_syn, floor, max_span_s)
        if out is None:
            skipped.append(dc)
        else:
            i0, sig = out
            y[i0:i0 + len(sig)] += sig
        if progress:
            progress(j + 1, len(decoys))
    return y, skipped

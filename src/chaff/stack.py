"""Decoys from whole real flyovers, every engine line included.

A propeller plane is not one tone but a stack: blade rate, its harmonics, engine orders in
between, all sliding by the same Doppler ratio. To keep the stack, a real flyover is
*de-Dopplered*: the audio is resampled on a warped clock (d tau / dt = heard / rest frequency)
so every line in the stack becomes a steady tone. Steady lines are then easy to separate from
birds, voices and wind (they stay put; those do not), so only bins that are lines through the
pass are kept. A decoy re-Dopplers that isolated stack onto a new pass (new blade rate, speed,
distance, time), with lines pushed up or down individually and sections faded, and levels it to
the new recording's background.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import scipy.signal as ss
from scipy.ndimage import uniform_filter1d, maximum_filter1d

from .noise import NoiseFloor, to_rate
from .synth import C, SYN_SR, K_MASK_DB, _smooth_noise, limit, straight_pass
from . import catalog as _cat

HI = 8000            # working rate for the warp (keeps interpolation exact below 1 kHz)
NFFT = 4096          # 2 s at 2 kHz: line resolution like the search's
F_TOP = 950.0


@dataclass
class Stack:
    name: str
    f0: float; v: float; d: float
    tau_cpa: float                 # warped time of closest approach, s from the start of s
    s: np.ndarray                  # isolated, de-Dopplered stack at SYN_SR
    lines_hz: np.ndarray           # rest frequencies of the kept lines
    sweep_snr_db: float            # fundamental's level through the sweep (as catalog.sweep_snr_db)
    floor_ref: float               # source background per bin at f0 near closest approach
    bb: np.ndarray                 # broadband excess (catalog texture), on u_bb
    u_bb: np.ndarray


def extract(x, sr, ev, floor: NoiseFloor, half_s=300.0, name="", line_db=6.0):
    t_cpa, f0, v, d = ev["t_cpa"], ev["f0"], ev["v"], ev["d"]
    a = max(t_cpa - half_s - 20, 0.0); b = min(t_cpa + half_s + 20, len(x) / sr)
    xh = to_rate(np.asarray(x[int(a * sr):int(b * sr)], np.float32), sr, HI).astype(np.float64)
    t = a + np.arange(len(xh)) / HI
    tc = np.arange(t[0], t[-1], 0.05); fc, _ = straight_pass(tc, f0, v, d, t_cpa)
    ratio = np.interp(t, tc, fc) / f0
    tau = np.cumsum(ratio) / HI; tau -= tau[0]
    tau_u = np.arange(0, tau[-1], 1 / HI)
    s = np.interp(np.interp(tau_u, tau, t), t, xh)              # x on the warped clock: lines are steady
    s2 = ss.resample_poly(s, 1, HI // SYN_SR)
    tau_cpa = float(np.interp(t_cpa, t, tau))
    # keep bins that are lines through the pass: median power in the core well above nearby bins
    f, tt, Z = ss.stft(s2, SYN_SR, window="hann", nperseg=NFFT, noverlap=NFFT - NFFT // 8, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    core = np.abs(tt - tau_cpa) < max(0.6 * d / v, 40)
    med = np.median(P[:, core], axis=1)
    nb = np.exp(uniform_filter1d(np.log(med + 1e-30), 41))       # local level around each bin
    line = (10 * np.log10(med / nb) > line_db) & (f > 40) & (f < F_TOP / 1.3)
    keep = maximum_filter1d(line.astype(float), 3) > 0              # line +- one bin
    Zk = np.where(keep[:, None], Z, 0)
    _, s_iso = ss.istft(Zk, SYN_SR, window="hann", nperseg=NFFT, noverlap=NFFT - NFFT // 8, boundary=False)
    s_iso = s_iso[:len(s2)].astype(np.float32)
    tex = _cat.extract(x, sr, ev, floor, half_s=half_s, name=name)
    return Stack(name, f0, v, d, tau_cpa, s_iso, f[line].astype(np.float32), _cat.sweep_snr_db(tex),
                 float(floor.at(np.array([f0]), np.array([t_cpa]))[0]), tex.bb, tex.u)


def save(stacks, path):
    meta = [dict(name=k.name, f0=k.f0, v=k.v, d=k.d, tau_cpa=k.tau_cpa, sweep_snr_db=k.sweep_snr_db, floor_ref=k.floor_ref) for k in stacks]
    np.savez_compressed(path, meta=np.array(meta, dtype=object), **{f"s{i}": k.s.astype(np.float16) for i, k in enumerate(stacks)},
                        **{f"l{i}": k.lines_hz for i, k in enumerate(stacks)}, **{f"b{i}": k.bb for i, k in enumerate(stacks)},
                        **{f"u{i}": k.u_bb for i, k in enumerate(stacks)})


def load(path):
    z = np.load(path, allow_pickle=True)
    return [Stack(m["name"], m["f0"], m["v"], m["d"], m["tau_cpa"], z[f"s{i}"].astype(np.float32), z[f"l{i}"], m["sweep_snr_db"],
                  m["floor_ref"], z[f"b{i}"], z[f"u{i}"]) for i, m in enumerate(z["meta"])]


_ISO = {}


def iso_snr_db(k: Stack) -> float:
    """Per-bin SNR of the isolated fundamental through the sweep, measured on the de-Dopplered
    stack itself (where it is a steady line at f0) with the search's resolution. This is what a
    decoy's level is calibrated on, so the level a search sees matches the target."""
    key = (k.name, len(k.s))
    if key not in _ISO:
        f, tt, Z = ss.stft(k.s, SYN_SR, window="hann", nperseg=NFFT, noverlap=NFFT - NFFT // 8, boundary=None,
                           padded=False, scaling="spectrum")
        P = np.abs(Z) ** 2
        core = np.abs(tt - k.tau_cpa) < max(0.3 * k.d / k.v, 10)
        j = np.abs(f - k.f0) < 1.0
        _ISO[key] = float(10 * np.log10(np.median(P[j][:, core].max(0)) / k.floor_ref + 1e-30)) if core.any() and j.any() else -99.0
    return _ISO[key]


@dataclass
class StackDecoy:
    t_cpa: float; f0: float; v: float; d: float
    src: int
    level_db: float               # target sweep SNR of the fundamental
    seed: int
    tier: str = ""

    def to_dict(self):
        return asdict(self)


def sample(stacks, duration_s, per_hour=6.0, snr_db=(22.0, 32.0), headroom_db=4.0, f_band=(55.0, 330.0), seed=0, edge_s=150.0, tier=""):
    """New blade rate, speed and distance drawn independently around the source's; the stack is
    pitch-shifted and time-stretched separately (phase vocoder on the steady de-Dopplered lines)."""
    rng = np.random.default_rng(seed); n = int(round(per_hour * duration_s / 3600.0)); out = []; tries = 0
    while len(out) < n and tries < 80 * max(n, 1):
        tries += 1
        T = rng.uniform(*snr_db)
        ok = [i for i, k in enumerate(stacks) if iso_snr_db(k) >= T + headroom_db and len(k.lines_hz) >= 3]
        if not ok:
            continue
        i = ok[rng.integers(len(ok))]; k = stacks[i]
        f0 = k.f0 * rng.uniform(0.6, 1.2); v = k.v * rng.uniform(0.8, 1.25); d = k.d * np.exp(rng.uniform(np.log(0.6), np.log(1.7)))
        if not (f_band[0] <= f0 * C / (C + v) and f0 * C / (C - v) <= f_band[1] and 300 < d < 12000):
            continue
        t = rng.uniform(edge_s, max(duration_s - edge_s, edge_s + 1))
        if any(abs(o.t_cpa - t) < 45 and abs(np.log(o.f0 / f0)) < 0.08 for o in out):
            continue
        out.append(StackDecoy(float(t), float(f0), float(v), float(d), int(i), float(T), int(rng.integers(2**31)), tier))
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
    """Push individual lines up or down (+-push_db) in the de-Dopplered stack."""
    f, tt, Z = ss.stft(k.s, SYN_SR, window="hann", nperseg=NFFT, noverlap=NFFT - NFFT // 8, boundary=None, padded=False)
    g = np.ones(len(f))
    for lf in k.lines_hz:
        j = np.abs(f - lf) <= 1.5 * SYN_SR / NFFT
        g[j] = 10 ** (rng.uniform(-push_db, push_db / 2) / 20)
    _, s = ss.istft(Z * g[:, None], SYN_SR, window="hann", nperseg=NFFT, noverlap=NFFT - NFFT // 8, boundary=False)
    s = s[:len(k.s)]
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


def render(decoys, stacks, n_samples, sr, floor: NoiseFloor, ceiling_below_db=-20.0, max_span_s=420.0):
    n_syn = int(np.ceil(n_samples * SYN_SR / sr)) + SYN_SR
    y = np.zeros(n_syn, np.float32)
    sob = ss.butter(4, [450, 950], "bandpass", fs=SYN_SR, output="sos")
    for dc in decoys:
        rng = np.random.default_rng(dc.seed); k = stacks[dc.src]
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
            continue
        tc, Tc, r, fc = tc[ok], Tc[ok], r[ok], fc[ok]
        fcut = 0.95 * 980.0 / (a * fc.max() / dc.f0)                     # nothing may land above 1 kHz after the warp
        sh = ss.sosfiltfilt(ss.butter(8, min(fcut, 0.45 * HI), fs=HI, output="sos"), sh)
        i0 = int(round(tc[0] * SYN_SR)); ts = i0 / SYN_SR + np.arange(int((tc[-1] - tc[0]) * SYN_SR)) / SYN_SR
        sig = np.interp(np.interp(ts, tc, Tc), tau_src, sh)
        # level: source fundamental's sweep SNR -> target, scaled from source to this background
        g = 10 ** ((dc.level_db - iso_snr_db(k)) / 20) * np.sqrt(floor.at(np.array([dc.f0]), np.array([dc.t_cpa]))[0] / k.floor_ref)
        fade = 10 ** (2.5 * _smooth_noise(rng, len(tc), 10 * 20) / 20) * np.sin(np.pi * np.clip((tc - tc[0]) / (tc[-1] - tc[0]), 0, 1)) ** 0.5
        sig = sig * g * np.interp(ts, tc, fade)
        sig *= _calibrate(sig, ts[0], dc, floor)      # land the fundamental exactly at the target, as a search measures it
        # broadband roar from the source texture
        u = (tc - dc.t_cpa) * dc.v / dc.d
        ex = np.interp(u, k.u_bb, k.bb, left=0, right=0) / 10 ** (iso_snr_db(k) / 10)
        fb = floor.f[(floor.f >= 450) & (floor.f <= 950)]
        var = (4 / 3) * floor.at(fb[:, None], tc[None, :]).sum(0)
        amp = np.sqrt(np.clip(10 ** (dc.level_db / 10) * ex * var, 0, None))
        if amp.max() > 0:
            nz = ss.sosfilt(sob, rng.standard_normal(len(ts) + SYN_SR))[SYN_SR:]
            sig = sig + np.interp(ts, tc, amp) * nz / (nz.std() + 1e-20)
        y[i0:i0 + len(ts)] += sig.astype(np.float32)
    y = limit(y, SYN_SR, floor, K_MASK_DB - ceiling_below_db)
    out = to_rate(y, SYN_SR, sr)
    return out[:n_samples] if len(out) >= n_samples else np.pad(out, (0, n_samples - len(out)))

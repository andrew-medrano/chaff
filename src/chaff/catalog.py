"""Decoys made from real flyovers.

A real flyover is lifted out of a recording as its *texture*: for each harmonic, the complex
envelope of the sound riding along the fitted Doppler curve (how it fades in, flickers, smears,
bands from ground reflection, wanders in pitch and trails off), measured relative to that
recording's background, plus the broadband roar around closest approach. A decoy re-lays a
texture, or two spliced together, onto a new and physically valid pass: new blade rate, speed,
distance and time (pitch shift, sweep stretch / shrink, time stretch), with sections pushed up
or down a few dB, and scales it to the new recording's own background.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import scipy.signal as ss

from .noise import NoiseFloor, to_rate
from .synth import C, SYN_SR, K_MASK_DB, _smooth_noise, limit, straight_pass

RATE = 20.0          # texture sample rate, Hz
BW_HZ = 3.0          # half-width of the band kept around each harmonic
F_TOP = 950.0


@dataclass
class Texture:
    name: str
    f0: float; v: float; d: float          # source geometry (fitted)
    u: np.ndarray                          # (N,) normalised time (t - t_cpa) * v / d
    q: np.ndarray                          # (H, N) complex envelope / sqrt(background per bin)
    bb: np.ndarray                         # (N,) broadband excess power re background, 450-950 Hz
    peak_snr_db: float                     # fundamental's peak per-bin SNR (2 s smoothing)


def extract(x, sr, ev, floor: NoiseFloor, half_s=300.0, n_harm=4, name=""):
    """Texture of one flyover. ev: dict with t_cpa (heard, s), f0, v, d (a detect/fit event)."""
    t_cpa, f0, v, d = ev["t_cpa"], ev["f0"], ev["v"], ev["d"]
    a = max(t_cpa - half_s - 10, 0.0); b = min(t_cpa + half_s + 10, len(x) / sr)
    x2 = to_rate(np.asarray(x[int(a * sr):int(b * sr)], np.float32), sr, SYN_SR).astype(np.float64)
    ts = a + np.arange(len(x2)) / SYN_SR
    tc = np.arange(ts[0], ts[-1], 1 / RATE)
    fc, _ = straight_pass(tc, f0, v, d, t_cpa)
    f = np.interp(ts, tc, fc); phi = 2 * np.pi * np.cumsum(f) / SYN_SR
    sos = ss.butter(4, BW_HZ, fs=SYN_SR, output="sos"); step = int(SYN_SR / RATE)
    qs = []
    for h in range(1, n_harm + 1):
        if h * fc.max() >= F_TOP:
            break
        z = ss.sosfiltfilt(sos, x2 * np.exp(-1j * h * phi))[::step][:len(tc)]
        qs.append(z / np.sqrt(floor.at(h * fc[:len(z)], tc[:len(z)])))
    n = min(len(q) for q in qs); tc = tc[:n]; q = np.stack([q_[:n] for q_ in qs])
    # broadband excess near closest approach (median-smoothed so bird calls do not count)
    sob = ss.butter(4, [450, 950], "bandpass", fs=SYN_SR, output="sos")
    p = ss.sosfilt(sob, x2) ** 2; k = SYN_SR
    pe = np.array([p[i:i + k].mean() for i in range(0, len(p) - k, k)]); tb = ts[0] + 0.5 + np.arange(len(pe))
    fb = floor.f[(floor.f >= 450) & (floor.f <= 950)]
    var = (4 / 3) * floor.at(fb[:, None], tb[None, :]).sum(0)
    from scipy.ndimage import median_filter
    ex = np.clip(median_filter(pe / var, 7) - 1, 0, None) * (np.abs(tb - t_cpa) < 90)
    bb = np.interp(tc, tb, ex)
    m = np.abs(tc - t_cpa) < 40
    pk = 10 * np.log10(np.convolve(np.abs(q[0]) ** 2, np.ones(41) / 41, "same")[m].max() + 1e-12)
    u = (tc - t_cpa) * v / d
    return Texture(name, f0, v, d, u, q.astype(np.complex64), bb.astype(np.float32), float(pk))


def save(textures, path):
    np.savez_compressed(path, meta=np.array([dict(name=t.name, f0=t.f0, v=t.v, d=t.d, peak_snr_db=t.peak_snr_db) for t in textures], dtype=object),
                        **{f"u{i}": t.u for i, t in enumerate(textures)}, **{f"q{i}": t.q for i, t in enumerate(textures)},
                        **{f"b{i}": t.bb for i, t in enumerate(textures)})


def load(path):
    z = np.load(path, allow_pickle=True)
    return [Texture(m["name"], m["f0"], m["v"], m["d"], z[f"u{i}"], z[f"q{i}"], z[f"b{i}"], m["peak_snr_db"]) for i, m in enumerate(z["meta"])]


@dataclass
class CatDecoy:
    t_cpa: float; f0: float; v: float; d: float
    src_a: int; src_b: int; splice_u: float
    peak_snr_db: float                      # target per-bin SNR of the fundamental through the sweep
    harm_push_db: list
    seed: int
    tier: str = ""

    def to_dict(self):
        return asdict(self)


def sample(textures, duration_s, per_hour=6.0, snr_db=(18.0, 28.0), headroom_db=6.0, f_band=(55.0, 330.0),
           seed=0, edge_s=120.0, tier=""):
    """Decoys built from textures with at least `headroom_db` more SNR than the target (a texture
    is only ever turned down, so its background noise stays under the new background)."""
    rng = np.random.default_rng(seed)
    n = int(round(per_hour * duration_s / 3600.0)); out = []; tries = 0
    while len(out) < n and tries < 80 * max(n, 1):
        tries += 1
        T = rng.uniform(*snr_db)
        ok = [i for i, t in enumerate(textures) if sweep_snr_db(t) >= T + headroom_db]
        if not ok:
            continue
        a = ok[rng.integers(len(ok))]; b = ok[rng.integers(len(ok))]
        geo = textures[rng.integers(len(textures))]      # geometry from any real flyover, texture from a strong one
        f0 = geo.f0 * rng.uniform(0.85, 1.18); v = geo.v * rng.uniform(0.85, 1.2); d = geo.d * np.exp(rng.uniform(np.log(0.6), np.log(1.7)))
        if not (f_band[0] <= f0 * C / (C + v) and f0 * C / (C - v) <= f_band[1]):
            continue
        t = rng.uniform(edge_s, max(duration_s - edge_s, edge_s + 1))
        if any(abs(o.t_cpa - t) < 45 and abs(np.log(o.f0 / f0)) < 0.08 for o in out):
            continue
        out.append(CatDecoy(float(t), float(f0), float(v), float(d), int(a), int(b), float(rng.uniform(-0.3, 0.3)), float(T),
                            [float(rng.uniform(-4, 4)) for _ in range(4)], int(rng.integers(2**31)), tier))
    return sorted(out, key=lambda o: o.t_cpa)


def sweep_snr_db(t: Texture) -> float:
    """Level of the fundamental across the sweep (|u| < 0.3), where a Doppler search scores a curve.
    Real flyovers peak briefly before closest approach and are 5-15 dB quieter through the sweep,
    so decoys are levelled on this, not on the peak; the limiter trims any peak that would be audible."""
    p = np.convolve(np.abs(t.q[0]) ** 2, np.ones(41) / 41, "same")
    return float(10 * np.log10(np.median(p[np.abs(t.u) < 0.3]) + 1e-12))


def detrend(t: Texture, win_s: float = 30.0) -> np.ndarray:
    """Remove the slow pitch offset a texture carries because its source's fitted curve was not
    exact (a few Hz near closest approach); the fast jitter and real engine wander under ~30 s stay."""
    q1 = t.q[0]; w = np.abs(q1) ** 2
    df = np.angle(q1[1:] * np.conj(q1[:-1])) * RATE / (2 * np.pi)      # Hz, per step
    k = max(int(win_s * RATE), 1); ker = np.ones(k)
    trend = np.convolve(df * w[1:], ker, "same") / (np.convolve(w[1:], ker, "same") + 1e-20)
    ph = np.concatenate([[0.0], np.cumsum(trend) * 2 * np.pi / RATE])
    hs = np.arange(1, t.q.shape[0] + 1)[:, None]
    return t.q * np.exp(-1j * hs * ph[None, :])


def _norm(t: Texture):
    s = sweep_snr_db(t)
    return detrend(t) / 10 ** (s / 20), t.bb / 10 ** (s / 10)


def render(decoys, textures, n_samples, sr, floor: NoiseFloor, ceiling_below_db=-14.0, max_span_s=480.0):
    n_syn = int(np.ceil(n_samples * SYN_SR / sr)) + SYN_SR
    y = np.zeros(n_syn, np.float32)
    sob = ss.butter(4, [450, 950], "bandpass", fs=SYN_SR, output="sos")
    for dc in decoys:
        rng = np.random.default_rng(dc.seed)
        A, B = textures[dc.src_a], textures[dc.src_b]
        (qa, ba), (qb, bbb) = _norm(A), _norm(B)
        tscale = dc.d / dc.v                                  # seconds per unit of u
        umin = max(A.u[0], B.u[0]); umax = min(A.u[-1], B.u[-1])
        t0 = max(dc.t_cpa + umin * tscale, dc.t_cpa - max_span_s, 0.0)
        t1 = min(dc.t_cpa + umax * tscale, dc.t_cpa + max_span_s, n_syn / SYN_SR - 1)
        if t1 - t0 < 30:
            continue
        tc = np.arange(t0, t1, 1 / RATE); u = (tc - dc.t_cpa) / tscale
        w = np.clip((u - dc.splice_u) / 0.2 + 0.5, 0, 1)      # 0 = texture A (approach), 1 = B (recede)
        push = 10 ** (3.0 * _smooth_noise(rng, len(tc), 0.5 * tscale * RATE) / 20)   # sections up / down
        taper = np.sin(np.pi * np.clip((tc - t0) / (t1 - t0), 0, 1)) ** 0.5
        f, _ = straight_pass(tc, dc.f0, dc.v, dc.d, dc.t_cpa)
        i0 = int(round(t0 * SYN_SR)); ts = i0 / SYN_SR + np.arange(int((t1 - t0) * SYN_SR)) / SYN_SR
        phi = 2 * np.pi * np.cumsum(np.interp(ts, tc, f)) / SYN_SR
        g = 10 ** (dc.peak_snr_db / 20)
        sig = np.zeros(len(ts))
        for h in range(1, 1 + min(qa.shape[0], qb.shape[0])):
            fh = h * f
            if fh.max() >= F_TOP:
                break
            qh = ((1 - w) * (np.interp(u, A.u, qa[h - 1].real) + 1j * np.interp(u, A.u, qa[h - 1].imag))
                  + w * (np.interp(u, B.u, qb[h - 1].real) + 1j * np.interp(u, B.u, qb[h - 1].imag)))
            c = g * 10 ** (dc.harm_push_db[h - 1] / 20) * push * taper * qh * np.sqrt(floor.at(fh, tc))
            cs = np.interp(ts, tc, c.real) + 1j * np.interp(ts, tc, c.imag)
            sig += 2 * np.real(cs * np.exp(1j * h * phi))
        ex = (1 - w) * np.interp(u, A.u, ba) + w * np.interp(u, B.u, bbb)
        fb = floor.f[(floor.f >= 450) & (floor.f <= 950)]
        var = (4 / 3) * floor.at(fb[:, None], tc[None, :]).sum(0)
        amp = np.sqrt(np.clip(g * g * ex * var, 0, None)) * taper
        if amp.max() > 0:
            nz = ss.sosfilt(sob, rng.standard_normal(len(ts) + SYN_SR))[SYN_SR:]
            sig += np.interp(ts, tc, amp) * nz / (nz.std() + 1e-20)
        y[i0:i0 + len(ts)] += sig.astype(np.float32)
    y = limit(y, SYN_SR, floor, K_MASK_DB - ceiling_below_db)
    out = to_rate(y, SYN_SR, sr)
    return out[:n_samples] if len(out) >= n_samples else np.pad(out, (0, n_samples - len(out)))

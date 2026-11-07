"""Fake propeller flyovers: physically consistent Doppler curves, drawn from real aircraft types
and then stretched, shrunk and pitch-shifted, with harmonics, rendered just under the level the
ear could pick out of the recording's own background.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import scipy.signal as ss

from .noise import ENBW_HZ, F_MAX, NoiseFloor, erb_hz, to_rate

C = 340.6            # speed of sound at 15 C, m/s
SYN_SR = 2000        # decoys live below 1 kHz, so they are built at 2 kHz and resampled once
# How rough a decoy is. Tuned so a classifier on line width, flicker, harmonic strength and how
# closely the curve follows a straight pass cannot tell decoys from the flyovers in real recordings.
REALISM = dict(
    rpm_wander=0.006, rpm_corr_s=15.0,      # slow pitch drift: throttle, gentle turns, wind
    smear=0.004, smear_corr_s=0.5,          # fast pitch jitter that widens the line to 1-2 bins
    flicker_db=5.0, flicker_corr_s=0.5,     # turbulence scintillation (grows with range)
    fade_db=2.0, fade_corr_s=15.0,          # slow level drift
    harm_db=(-18.0, -8.0),                  # harmonics 2.. relative to the fundamental
)
K_MASK_DB = -4.0     # a tone in noise becomes audible near -4 dB re the noise in one ERB (Moore);
                     # higher (harder to hear) at low frequency, so this is the conservative end

# Blade-passing frequency (rpm x blades / 60) and cruise/approach speed of common propeller types.
# Decoys start from one of these and are then stretched, shrunk and shifted.
CATALOG = {
    "two-blade piston (Cessna 172, PA-28)":  dict(bpf=(72, 92),   v=(45, 65),   harm_db=(-3, -6, -10)),
    "two-blade piston, fast (Bonanza, Mooney)": dict(bpf=(80, 92), v=(60, 85),   harm_db=(-2, -5, -9)),
    "three-blade piston (SR22, Bonanza A36)": dict(bpf=(118, 140), v=(65, 95),   harm_db=(-4, -8, -12)),
    "four-blade turboprop (PC-12, King Air)": dict(bpf=(105, 135), v=(80, 140),  harm_db=(-5, -9, -13)),
    "five/six-blade turboprop (Dash 8, ATR)": dict(bpf=(95, 160),  v=(100, 150), harm_db=(-6, -10, -14)),
    "helicopter tail rotor band":             dict(bpf=(60, 120),  v=(35, 70),   harm_db=(-2, -4, -7)),
}


@dataclass
class Decoy:
    t_cpa: float          # heard time of closest approach, s from the start of the recording
    f0: float             # rest blade-passing frequency, Hz
    v: float              # speed, m/s
    d: float              # slant distance at closest approach, m
    harm_db: list = field(default_factory=lambda: [0.0])   # level of harmonics 1..H re the fundamental
    below_db: float = 6.0  # how far under the audibility threshold the fundamental peaks
    seed: int = 0
    kind: str = ""
    origin: str = "catalog"   # catalog | template
    broadband_db: float = 0.0  # engine/propeller roar: how much it lifts the 450-950 Hz band at closest approach

    def to_dict(self):
        return asdict(self)


def straight_pass(t_obs, f0, v, d, t_cpa_heard, c=C):
    """Heard frequency and range for a straight constant-speed pass (emission-time kinematics)."""
    t_obs = np.asarray(t_obs, float)
    t_e_cpa = t_cpa_heard - d / c
    te = np.linspace(np.min(t_obs) - 150.0, np.max(t_obs), 16001)   # heard = emitted + r/c, r < 50 km
    r = np.sqrt(d * d + (v * (te - t_e_cpa)) ** 2)
    rr = v * v * (te - t_e_cpa) / r
    tobs = te + r / c
    return np.interp(t_obs, tobs, f0 * c / (c + rr)), np.interp(t_obs, tobs, r)


def mask_snr_db(f, below_db=6.0):
    """Largest per-bin SNR (in a 2 s spectrogram) a steady tone at f can have while sitting
    `below_db` under the masked threshold: SNR_bin = K + 10 log10(ERB / ENBW) - below."""
    return K_MASK_DB + 10 * np.log10(erb_hz(f) / ENBW_HZ) - below_db


def _smooth_noise(rng, n, corr):
    """Unit-variance noise with correlation length `corr` samples (moving average of white noise)."""
    k = max(int(corr), 1)
    w = rng.standard_normal(n + 2 * k)
    s = np.convolve(w, np.hanning(2 * k + 1), "same")[k:k + n]
    return s / (s.std() + 1e-12)


def sample_decoys(duration_s, per_hour=60.0, f_band=(55.0, 330.0), d_range=(500.0, 7000.0),
                  templates=None, template_share=0.5, below_db=(4.0, 9.0), seed=0, edge_s=120.0,
                  broadband_db=(0.0, 0.0)):
    """Random flyovers across the recording. Each starts from a CATALOG type (or, with
    `templates`, from a curve the caller measured in this very recording) and is then
    stretched in time (distance), shrunk or stretched in sweep (speed) and shifted in pitch.
    Decoys close in both time and pitch to one already placed are redrawn."""
    rng = np.random.default_rng(seed)
    n = int(round(per_hour * duration_s / 3600.0))
    kinds = list(CATALOG)
    out = []
    tries = 0
    while len(out) < n and tries < 50 * max(n, 1):
        tries += 1
        t = rng.uniform(edge_s, max(duration_s - edge_s, edge_s + 1))
        if templates and rng.random() < template_share:
            tp = templates[rng.integers(len(templates))]
            f0 = tp["f0"] * rng.uniform(0.75, 1.33)          # pitch shift
            v = tp["v"] * rng.uniform(0.8, 1.25)              # sweep shrink / stretch
            d = tp["d"] * np.exp(rng.uniform(np.log(0.5), np.log(2.0)))   # time stretch
            hd = [0.0] + [float(rng.uniform(*REALISM["harm_db"]) - 3 * j) for j in range(rng.integers(0, 3))]
            kind, origin = "warped copy of a curve in this recording", "template"
        else:
            kind = kinds[rng.integers(len(kinds))]; c = CATALOG[kind]
            f0 = rng.uniform(*c["bpf"]) * (1 if rng.random() < 0.7 else rng.choice([2.0, 0.5, 1.5]))
            v = rng.uniform(*c["v"]) * rng.uniform(0.9, 1.1)
            d = np.exp(rng.uniform(np.log(d_range[0]), np.log(d_range[1])))
            hd = [0.0] + [float(rng.uniform(*REALISM["harm_db"]) + (h - c["harm_db"][0]) / 2) for h in c["harm_db"][:rng.integers(0, 3)]]
            origin = "catalog"
        if not (f_band[0] <= f0 * C / (C + v) and f0 * C / (C - v) <= f_band[1]):
            continue
        if any(abs(o.t_cpa - t) < 45 and abs(np.log(o.f0 / f0)) < 0.08 for o in out):
            continue
        out.append(Decoy(float(t), float(f0), float(v), float(d), hd, float(rng.uniform(*below_db)),
                         int(rng.integers(2**31)), kind, origin, float(rng.uniform(*broadband_db))))
    out.sort(key=lambda o: o.t_cpa)
    return out


def render(decoys, n_samples, sr, floor: NoiseFloor, max_span_s=480.0, ceiling_below_db=3.0):
    """Sum of all decoys as a float32 signal at `sr`, `n_samples` long.

    What makes a decoy look like a real flyover rather than a drawn line:
    - it fades in from the noise as the plane approaches and trails off as it leaves, at a rate
      set by range ((d/r)^k with k = 1.2-2.2: faster than spherical spreading, as low passes are)
      and a soft taper at both ends, so it never starts or stops abruptly;
    - its front and back differ (propellers are louder behind or ahead depending on type);
    - ground reflection: the direct sound and its bounce off the ground interfere, putting
      bright and dark bands that slide along the curve as the plane's elevation changes;
    - turbulence makes it flicker (amplitude scintillation that grows with range) and smears
      the line slightly (fast pitch jitter, 0.04%) on top of the engine's slow rpm wander (0.15%).
    Each harmonic is then capped `below_db` under the masked threshold at its own frequency and
    moment, and a final limiter keeps every auditory band `ceiling_below_db` under the threshold
    where decoys overlap."""
    n_syn = int(np.ceil(n_samples * SYN_SR / sr)) + SYN_SR
    y = np.zeros(n_syn, np.float32)
    ctl = 10.0                                     # control rate for curves and envelopes, Hz
    for dc in decoys:
        rng = np.random.default_rng(dc.seed)
        k = rng.uniform(1.2, 2.2)                  # range falloff exponent
        a_dir = rng.uniform(-0.25, 0.5)            # >0: louder going away
        h_alt = dc.d * rng.uniform(0.3, 0.95)      # altitude above the microphone, m
        h_mic = rng.uniform(1.0, 3.0); rho = rng.uniform(0.35, 0.8)
        r_max = dc.d * 10 ** (40 / (20 * k))       # envelope 40 dB down
        half = min(max_span_s, np.sqrt(max(r_max**2 - dc.d**2, 0)) / dc.v + dc.d / C)
        t0 = max(dc.t_cpa - half, 0.0); t1 = min(dc.t_cpa + half, n_syn / SYN_SR - 1)
        if t1 - t0 < 20:
            continue
        tc = np.arange(t0, t1, 1 / ctl)
        f, r = straight_pass(tc, dc.f0, dc.v, dc.d, dc.t_cpa)
        rr = np.gradient(r, tc) / dc.v             # -1 approaching .. +1 receding
        R = REALISM
        f = f * (1 + R["rpm_wander"] * _smooth_noise(rng, len(tc), R["rpm_corr_s"] * ctl)
                 + R["smear"] * _smooth_noise(rng, len(tc), R["smear_corr_s"] * ctl))
        scint_db = R["flicker_db"] * (0.6 + 0.4 * np.minimum(r / dc.d - 1, 3) / 3) * _smooth_noise(rng, len(tc), R["flicker_corr_s"] * ctl)
        env = ((dc.d / r) ** k * (1 + a_dir * rr) / (1 + max(a_dir, 0))
               * 10 ** ((R["fade_db"] * _smooth_noise(rng, len(tc), R["fade_corr_s"] * ctl) + scint_db) / 20))
        u = (tc - t0) / (t1 - t0); env *= np.sin(np.pi * np.clip(u, 0, 1)) ** 0.5   # soft taper at both ends
        path_diff = 2 * h_mic * np.clip(h_alt / r, 0, 1)                            # ground bounce, m
        i0 = int(round(t0 * SYN_SR)); ts = i0 / SYN_SR + np.arange(int((t1 - t0) * SYN_SR)) / SYN_SR
        fs = np.interp(ts, tc, f)
        phase = 2 * np.pi * np.cumsum(fs) / SYN_SR
        for h, hdb in enumerate(dc.harm_db, start=1):
            fh = h * f
            if fh.max() >= min(F_MAX, 0.47 * SYN_SR):
                break
            ground = np.abs(1 + rho * np.exp(-2j * np.pi * fh * path_diff / C)) / np.sqrt(1 + rho**2)
            cap = 2 * np.sqrt(10 ** (mask_snr_db(fh, dc.below_db) / 10) * floor.at(fh, tc))
            cap_cpa = float(np.interp(dc.t_cpa, tc, cap))
            a = np.minimum(cap_cpa * 10 ** (hdb / 20) * env * ground, cap)
            y[i0:i0 + len(ts)] += (np.interp(ts, tc, a) * np.sin(h * phase + rng.uniform(0, 2 * np.pi))).astype(np.float32)
        if dc.broadband_db > 0:                     # broadband roar, 450-950 Hz, swelling with the tone
            fb = floor.f[(floor.f >= 450) & (floor.f <= 950)]
            var = (4 / 3) * floor.at(fb[:, None], tc[None, :]).sum(0)          # background variance in the band
            g = np.sqrt((10 ** (dc.broadband_db / 10) - 1) * var) * np.minimum((dc.d / r) ** k, 1.0) * np.sin(np.pi * np.clip(u, 0, 1)) ** 0.5
            sos = ss.butter(4, [450, 950], "bandpass", fs=SYN_SR, output="sos")
            nz = ss.sosfilt(sos, rng.standard_normal(len(ts) + SYN_SR))[SYN_SR:]
            nz /= nz.std() + 1e-20
            y[i0:i0 + len(ts)] += (np.interp(ts, tc, g) * nz).astype(np.float32)
    y = limit(y, SYN_SR, floor, K_MASK_DB - ceiling_below_db)
    out = to_rate(y, SYN_SR, sr)
    return out[:n_samples] if len(out) >= n_samples else np.pad(out, (0, n_samples - len(out)))


def limit(y, sr_y, floor: NoiseFloor, ceiling_db, passes=2):
    """Turn the decoy track down wherever any auditory band would go above `ceiling_db` re the
    background (overlapping decoys add up). Gain is computed per second, eased over ~3 s."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from .audibility import band_ratio_db
    for _ in range(passes):
        t, r = band_ratio_db(y, sr_y, floor)        # (T,) worst band per second, dB
        if r.max() <= ceiling_db:
            break
        g = np.minimum(1.0, 10 ** ((ceiling_db - 0.5 - r) / 20))
        g = uniform_filter1d(minimum_filter1d(g, 5), 3)
        ty = np.arange(len(y)) / sr_y
        y = (y * np.interp(ty, t, g, left=1.0, right=1.0)).astype(np.float32)
    return y

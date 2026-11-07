"""The recording's own background level, measured the way a Doppler-curve search sees it.

A search like dopplertrack looks at a spectrogram with ~2 s windows (0.49 Hz bins). A steady
tone puts all of its power in one or two of those bins, while background noise is spread over
every bin. The ear instead hears the tone against all the noise inside one critical band
(an ERB: ~35 Hz wide at 100 Hz, ~57 Hz at 300 Hz). That ratio, about 17-19 dB at plane-hum
frequencies, is the room this library works in: a tone can stand well clear of the noise in the
search's spectrogram while sitting below the noise the ear compares it with.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
import scipy.signal as ss
from scipy.ndimage import median_filter

A_SR = 4000          # analysis rate
A_N = 8192           # analysis window: 2.05 s, the same resolution dopplertrack uses
A_HOP = A_SR         # 1 s hop
ENBW_HZ = 1.5 * A_SR / A_N   # equivalent noise bandwidth of one Hann bin, 0.73 Hz
F_MAX = 1100.0


def erb_hz(f):
    """Glasberg & Moore equivalent rectangular bandwidth of the auditory filter at f (Hz)."""
    return 24.7 * (4.37 * np.asarray(f, float) / 1000.0 + 1.0)


def to_rate(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    g = gcd(int(sr), int(target))
    return ss.resample_poly(x, target // g, sr // g).astype(np.float32)


@dataclass
class NoiseFloor:
    """Mean background power per analysis bin, M(f, t), on a coarse time grid.

    M is in |STFT|^2 units with 'spectrum' scaling, so a sine of amplitude A reads A^2/4 in
    its peak bin and A = 2 sqrt(snr * M) gives that sine a per-bin SNR of `snr`."""
    t: np.ndarray       # (B,) block centres, s
    f: np.ndarray       # (F,) Hz
    M: np.ndarray       # (F, B)

    def at(self, f, t):
        f = np.asarray(f, float); t = np.asarray(t, float)
        fi = np.clip((f - self.f[0]) / (self.f[1] - self.f[0]), 0, len(self.f) - 1)
        if len(self.t) == 1:
            ti = np.zeros_like(t)
        else:
            ti = np.clip(np.interp(t, self.t, np.arange(len(self.t))), 0, len(self.t) - 1)
        i0 = np.floor(fi).astype(int); i1 = np.minimum(i0 + 1, len(self.f) - 1); wf = fi - i0
        j0 = np.floor(ti).astype(int); j1 = np.minimum(j0 + 1, len(self.t) - 1); wt = ti - j0
        L = np.log(self.M)
        v = (L[i0, j0] * (1 - wf) * (1 - wt) + L[i1, j0] * wf * (1 - wt) + L[i0, j1] * (1 - wf) * wt + L[i1, j1] * wf * wt)
        return np.exp(v)


def stft_power(x4k: np.ndarray, f_max: float = F_MAX, chunk_s: float = 3600.0):
    """Yield (t, f, P) chunks of |STFT|^2 at the analysis resolution."""
    f = np.fft.rfftfreq(A_N, 1 / A_SR); band = f <= f_max
    K = max((len(x4k) - A_N) // A_HOP + 1, 0)
    per = int(chunk_s)
    for k0 in range(0, K, per):
        k1 = min(k0 + per, K)
        seg = x4k[k0 * A_HOP:(k1 - 1) * A_HOP + A_N]
        _, _, Z = ss.stft(seg, A_SR, window="hann", nperseg=A_N, noverlap=A_N - A_HOP, boundary=None, padded=False, scaling="spectrum")
        t = (A_N / 2 + (k0 + np.arange(Z.shape[1])) * A_HOP) / A_SR
        yield t, f[band], (np.abs(Z[band]) ** 2).astype(np.float32)


def noise_floor(x: np.ndarray, sr: int, block_s: float = 60.0) -> NoiseFloor:
    """Per-bin median power over `block_s` blocks (robust to passing planes, birds and the
    decoys' own brief visits), converted to a mean, then median-filtered across frequency so
    steady hum lines do not count as noise that could hide a tone."""
    x4 = to_rate(np.asarray(x, np.float32), sr, A_SR)
    ts, ms = [], []
    f = None
    for t, f, P in stft_power(x4):
        nb = int(np.ceil(P.shape[1] / block_s))
        for b in range(nb):
            sl = slice(int(b * block_s), min(int((b + 1) * block_s), P.shape[1]))
            if sl.stop - sl.start < 5:
                continue
            ms.append(np.median(P[:, sl], axis=1) / np.log(2.0))
            ts.append(t[sl].mean())
    if not ms:
        raise ValueError("recording shorter than one analysis window (2 s)")
    M = np.stack(ms, axis=1)
    M = median_filter(M, size=(15, 3), mode="nearest")
    return NoiseFloor(np.array(ts), f, np.maximum(M, 1e-30))

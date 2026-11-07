"""The recording's own background level, measured the way a Doppler-curve search sees it.

A search like dopplertrack looks at a spectrogram with ~2 s windows (0.49 Hz bins). A steady
tone puts all of its power in one or two of those bins, while background noise is spread over
every bin. The ear instead hears the tone against all the noise inside one critical band
(an ERB: ~35 Hz wide at 100 Hz, ~57 Hz at 300 Hz). That ratio, about 17-19 dB at plane-hum
frequencies, is the room this library works in: a tone can stand well clear of the noise in the
search's spectrogram while sitting below the noise the ear compares it with.

Everything here works on a stream: the background is built from blocks of the 4 kHz analysis
signal, so the recording never has to be in memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss
from scipy.ndimage import median_filter

from .resample import Resampler

A_SR = 4000          # analysis rate
A_N = 8192           # analysis window: 2.05 s, the same resolution dopplertrack uses
A_HOP = A_SR         # 1 s hop
ENBW_HZ = 1.5 * A_SR / A_N   # equivalent noise bandwidth of one Hann bin, 0.73 Hz
F_MAX = 1100.0
K_MASK_DB = -4.0     # a tone in noise becomes audible near -4 dB re the noise in one ERB (Moore);
                     # higher (harder to hear) at low frequency, so this is the conservative end
BLOCK_S = 60         # the background is a median over blocks of this many 1 s frames
MIN_FLOOR_S = (A_N + 4 * A_HOP) / A_SR   # 6.05 s: one block needs at least 5 frames


def erb_hz(f):
    """Glasberg & Moore equivalent rectangular bandwidth of the auditory filter at f (Hz)."""
    return 24.7 * (4.37 * np.asarray(f, float) / 1000.0 + 1.0)


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


class Frames:
    """|STFT|^2 frames (Hann, 'spectrum' scaling, no padding) of a signal pushed in blocks.

    Frame k covers samples [k*hop, k*hop + nfft). Frames are handed out in groups of `group`;
    each group is computed from exactly the samples it covers, so the values do not depend on
    how the signal was split into blocks. Each group is (t, f, P) with P of shape (F, T)."""

    def __init__(self, sr, nfft, hop, group, f_lo=0.0, f_hi=np.inf, db=False):
        self.sr, self.nfft, self.hop, self.group, self.db = sr, nfft, hop, int(group), db
        f = np.fft.rfftfreq(nfft, 1 / sr)
        self.band = (f >= f_lo) & (f <= f_hi)
        self.f = f[self.band]
        self.buf = np.zeros(0, np.float32)
        self.k = 0               # index of the first frame in buf

    def _frames(self, seg):
        _, _, Z = ss.stft(seg, self.sr, window="hann", nperseg=self.nfft, noverlap=self.nfft - self.hop,
                          boundary=None, padded=False, scaling="spectrum")
        P = np.abs(Z[self.band]) ** 2
        P = (10 * np.log10(P + 1e-20)) if self.db else P
        t = (self.nfft / 2 + (self.k + np.arange(Z.shape[1])) * self.hop) / self.sr
        self.k += Z.shape[1]
        return t, self.f, P.astype(np.float32)

    def push(self, x):
        self.buf = np.concatenate([self.buf, np.asarray(x, np.float32)])
        out = []
        span = (self.group - 1) * self.hop + self.nfft
        while len(self.buf) >= span:
            out.append(self._frames(self.buf[:span]))
            self.buf = self.buf[self.group * self.hop:]
        return out

    def flush(self):
        K = (len(self.buf) - self.nfft) // self.hop + 1 if len(self.buf) >= self.nfft else 0
        if K <= 0:
            return []
        out = [self._frames(self.buf[:(K - 1) * self.hop + self.nfft])]
        self.buf = np.zeros(0, np.float32)
        return out


class FloorBuilder:
    """Background of a stream of 4 kHz mono samples: push() blocks, then finish().

    Per-bin median power over BLOCK_S-frame blocks (robust to passing planes, birds and the
    decoys' own brief visits), converted to a mean, then median-filtered across frequency so
    steady hum lines do not count as noise that could hide a tone."""

    def __init__(self):
        self.frames = Frames(A_SR, A_N, A_HOP, BLOCK_S, f_hi=F_MAX)
        self.ts, self.ms = [], []

    def _take(self, groups):
        for t, f, P in groups:
            if P.shape[1] < 5:
                continue
            self.ms.append(np.median(P, axis=1) / np.log(2.0))
            self.ts.append(t.mean())

    def push(self, x4):
        self._take(self.frames.push(x4))

    def finish(self) -> NoiseFloor:
        self._take(self.frames.flush())
        if not self.ms:
            raise ValueError(f"recording too short: measuring its background needs at least {MIN_FLOOR_S:.1f} s")
        M = median_filter(np.stack(self.ms, axis=1), size=(15, 3), mode="nearest")
        return NoiseFloor(np.array(self.ts), self.frames.f, np.maximum(M, 1e-30))


def noise_floor(x: np.ndarray, sr: int, block: int = 1 << 20) -> NoiseFloor:
    """Background of an in-memory mono signal (the same computation the file path streams)."""
    x = np.asarray(x, np.float32)
    rs, fb = Resampler(sr, A_SR), FloorBuilder()
    for i in range(0, len(x), block):
        fb.push(rs.push(x[i:i + block]))
    fb.push(rs.flush())
    return fb.finish()

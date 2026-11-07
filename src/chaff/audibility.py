"""How close the added decoys come to being audible, and the limiter that keeps them in bounds.

For each auditory band (ERB-wide, centred every half ERB from 50 Hz to 1 kHz) and each second,
the power of everything that was added (all decoys together, so overlapping ones count) is
compared with the recording's background power in the same band. A tone in noise becomes
audible near -4 dB on this ratio. The background used is the 60 s median floor, which leaves out
birds, wind gusts and other passing sounds that would mask further, so the margin is conservative.

The decoy track is analysed in blocks (resampled from its 2 kHz working rate to the 4 kHz
analysis rate on the fly), so a 12-hour track costs no more memory than the track itself.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from .noise import A_HOP, A_N, A_SR, F_MAX, K_MASK_DB, Frames, NoiseFloor, erb_hz
from .resample import Resampler

CHUNK_FRAMES = 3600      # one hour of 1 s frames per analysis group


def bands(f_lo=50.0, f_hi=1000.0):
    out = []; f = f_lo
    while f < f_hi:
        out.append(f); f += erb_hz(f) / 2
    return np.array(out)


def _track(y):
    """A track, or a tuple of equal-length tracks that are summed block by block."""
    return tuple(y) if isinstance(y, (tuple, list)) else (y,)


def band_ratios(y, sr_y, floor: NoiseFloor, f_lo=50.0, f_hi=1000.0, block=1 << 20):
    """Yield (t, R): R[k, j] = added power / background power in band k at second t[j], dB.

    `y` is the added track at sr_y (or a tuple of tracks to be summed)."""
    ys = _track(y)
    fc = bands(f_lo, f_hi)
    rs = Resampler(sr_y, A_SR)
    fr = Frames(A_SR, A_N, A_HOP, CHUNK_FRAMES, f_hi=F_MAX)
    sel = [np.abs(fr.f - c) <= erb_hz(c) / 2 for c in fc]

    def ratios(groups):
        for t, f, P in groups:
            Mb = floor.at(f[:, None], t[None, :])
            yield t, np.stack([10 * np.log10(P[s].sum(0) / Mb[s].sum(0) + 1e-30) for s in sel])

    n = len(ys[0])
    for i in range(0, n, block):
        seg = ys[0][i:i + block].astype(np.float64)
        for other in ys[1:]:
            seg = seg + other[i:i + block]
        yield from ratios(fr.push(rs.push(seg)))
    yield from ratios(fr.push(rs.flush()))
    yield from ratios(fr.flush())


def report(y, sr_y, floor: NoiseFloor, f_lo=50.0, f_hi=1000.0):
    """Ratio of added power to background power per ERB band and second, in dB, summarised.

    margin_db > 0 means every band stayed that far under the hearing threshold."""
    fc = bands(f_lo, f_hi)
    worst = np.full(len(fc), -np.inf)
    n_over = 0; n_tot = 0; hist = []
    for _, R in band_ratios(y, sr_y, floor, f_lo, f_hi):
        worst = np.maximum(worst, R.max(axis=1))
        n_over += int((R > K_MASK_DB).sum()); n_tot += R.size
        hist.append(R[R > -60])
    if n_tot == 0:
        raise ValueError("track shorter than one analysis window")
    allr = np.concatenate(hist)
    return dict(threshold_db=K_MASK_DB,
                worst_band_db=float(worst.max()), worst_band_hz=float(fc[int(np.argmax(worst))]),
                margin_db=float(K_MASK_DB - worst.max()),
                share_band_seconds_over=float(n_over / n_tot),
                p99_db=float(np.percentile(allr, 99)) if allr.size else None,
                p50_active_db=float(np.median(allr)) if allr.size else None)


def band_ratio_db(y, sr_y, floor: NoiseFloor, f_lo=50.0, f_hi=1000.0):
    """(t, worst ratio over bands per second), for the limiter."""
    ts, rs = [], []
    for t, R in band_ratios(y, sr_y, floor, f_lo, f_hi):
        ts.append(t); rs.append(R.max(axis=0))
    if not ts:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ts), np.concatenate(rs)


def limit(y, sr_y, floor: NoiseFloor, ceiling_db, passes=2, block=1 << 20):
    """Turn the decoy track down, in place, wherever any auditory band would go above
    `ceiling_db` re the background (overlapping decoys add up). Gain is computed per second,
    eased over ~3 s. Returns y."""
    for _ in range(passes):
        t, r = band_ratio_db(y, sr_y, floor)
        if not len(r) or r.max() <= ceiling_db:
            break
        g = np.minimum(1.0, 10 ** ((ceiling_db - 0.5 - r) / 20))
        g = uniform_filter1d(minimum_filter1d(g, 5), 3)
        for i in range(0, len(y), block):
            ty = (i + np.arange(min(block, len(y) - i))) / sr_y
            y[i:i + len(ty)] = (y[i:i + len(ty)] * np.interp(ty, t, g, left=1.0, right=1.0)).astype(np.float32)
    return y

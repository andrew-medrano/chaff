"""How close the added decoys come to being audible.

For each auditory band (ERB-wide, centred every half ERB from 50 Hz to 1 kHz) and each second,
the power of everything that was added (all decoys together, so overlapping ones count) is
compared with the recording's background power in the same band. A tone in noise becomes
audible near -4 dB on this ratio. The background used is the 60 s median floor, which leaves out
birds, wind gusts and other passing sounds that would mask further, so the margin is conservative.
"""
from __future__ import annotations

import numpy as np

from .noise import A_SR, NoiseFloor, erb_hz, stft_power, to_rate
from .synth import K_MASK_DB


def bands(f_lo=50.0, f_hi=1000.0):
    out = []; f = f_lo
    while f < f_hi:
        out.append(f); f += erb_hz(f) / 2
    return np.array(out)


def report(added: np.ndarray, sr: int, floor: NoiseFloor, f_lo=50.0, f_hi=1000.0):
    """Ratio of added power to background power per ERB band and second, in dB, summarised."""
    y4 = to_rate(np.asarray(added, np.float32), sr, A_SR)
    fc = bands(f_lo, f_hi)
    worst = np.full(len(fc), -np.inf)
    n_over = 0; n_tot = 0; hist = []
    for t, f, P in stft_power(y4):
        Mb = floor.at(f[:, None], t[None, :])                       # (F, T)
        for k, c in enumerate(fc):
            sel = np.abs(f - c) <= erb_hz(c) / 2
            r = 10 * np.log10(P[sel].sum(0) / Mb[sel].sum(0) + 1e-30)
            worst[k] = max(worst[k], float(r.max()))
            n_over += int((r > K_MASK_DB).sum()); n_tot += r.size
            hist.append(r[r > -60])
    allr = np.concatenate(hist) if hist else np.array([-np.inf])
    return dict(threshold_db=K_MASK_DB,
                worst_band_db=float(worst.max()), worst_band_hz=float(fc[int(np.argmax(worst))]),
                margin_db=float(K_MASK_DB - worst.max()),
                share_band_seconds_over=float(n_over / max(n_tot, 1)),
                p99_db=float(np.percentile(allr, 99)) if allr.size else None,
                p50_active_db=float(np.median(allr)) if allr.size else None)


def band_ratio_db(added, sr, floor: NoiseFloor, f_lo=50.0, f_hi=1000.0):
    """(t, worst ratio over bands per second) for the limiter."""
    y4 = to_rate(np.asarray(added, np.float32), sr, A_SR)
    fc = bands(f_lo, f_hi); ts, rs = [], []
    for t, f, P in stft_power(y4):
        Mb = floor.at(f[:, None], t[None, :]); w = np.full(len(t), -np.inf)
        for c in fc:
            sel = np.abs(f - c) <= erb_hz(c) / 2
            w = np.maximum(w, 10 * np.log10(P[sel].sum(0) / Mb[sel].sum(0) + 1e-30))
        ts.append(t); rs.append(w)
    return np.concatenate(ts), np.concatenate(rs)

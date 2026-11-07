"""Measure the "dot column" tell in decoys made from a catalog.

    python scripts/transient_tell.py CATALOG.npz [CATALOG2.npz ...] [--seeds 8] [--png DIR]

A broadband click left in a stack survives as one dot per engine line, at the same instant in
every line, in every decoy made from that stack. This renders decoys from each stack (the same
passes for the same seed, whatever the catalog), takes each decoy's spectrogram the way a
Doppler-curve search sees it (2 s windows, 0.49 Hz bins, 40-450 Hz), follows each engine line
along its Doppler curve and marks a line as spiking in a frame when its power is more than
SPIKE_DB over its own running median of the 31 frames (8 s) around. A frame is a column when at
least half of the lines in the band, and at least 3 of them, spike together. Printed per stack: the
share of frames that are columns (mean over the seeds) and the number of separate columns per
decoy. Real planes do not do this; a clean stack scores near 0.

The phase-vocoder stretch carries phase through the whole stack, so any change in a stack, however
local, gives every later part of its decoys a different fine flicker. Differences of about 0.1
percentage points between two versions of a stack are that noise; use more seeds to see past it.

--png DIR also saves the spectrogram of each stack's first decoy, one file per catalog and stack.
Stacks are matched across catalogs by name ("mcmaster.wav#5" and "mcmaster#5" are the same).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chaff import noise, stack                                        # noqa: E402
from chaff.stack import NFFT, StackDecoy                              # noqa: E402
from chaff.synth import C, SYN_SR, straight_pass, track_len           # noqa: E402

SPIKE_DB = 6.0
F_LO, F_HI = 45.0, 445.0
DUR = 1200.0                    # s of (silent) recording each decoy is laid into, closest approach at 600 s


def floor():
    """A fixed background: the decoys' level is set against it, which does not change their shape."""
    rng = np.random.default_rng(0)
    return noise.noise_floor((0.01 * rng.standard_normal(int(DUR * 4000))).astype(np.float32), 4000)


def passes(k, seeds):
    """One new pass per seed, drawn like stack.sample() draws them (blade rate x0.6-1.2, speed
    x0.8-1.25, distance x0.6-1.7, inside the search band)."""
    out = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        for _ in range(200):
            f0 = k.f0 * rng.uniform(0.6, 1.2); v = k.v * rng.uniform(0.8, 1.25)
            d = k.d * np.exp(rng.uniform(np.log(0.6), np.log(1.7)))
            if 55 <= f0 * C / (C + v) and f0 * C / (C - v) <= 330 and 300 < d < 12000:
                break
        out.append(StackDecoy(DUR / 2, float(f0), float(v), float(d), 0, 32.0, int(seed)))
    return out


def tell(k, dc, fl):
    """(share of frames that are columns, number of columns, spectrogram (t, f, dB) for a figure)."""
    out = stack.render_one(dc, k, track_len(int(DUR * 4000), 4000), fl)
    if out is None:
        return np.nan, 0, None
    i0, sig = out
    f, tt, P = stack._stft_power_rows(sig.astype(np.float64), lambda f: (f >= 40) & (f <= 450))
    f = f[(f >= 40) & (f <= 450)]
    t = i0 / SYN_SR + tt
    fc, _ = straight_pass(t, dc.f0, dc.v, dc.d, dc.t_cpa)
    ratio = fc / dc.f0
    slope = np.abs(np.gradient(fc, t)) / dc.f0                 # Doppler slope per Hz of rest pitch
    a = dc.f0 / k.f0
    span = (t > t[0] + 0.05 * (t[-1] - t[0])) & (t < t[-1] - 0.05 * (t[-1] - t[0]))
    L, inb = [], []
    df = f[1] - f[0]
    for rest in k.lines_hz * a:
        q = rest * ratio
        w = np.maximum(1.5 * df, rest * slope * 1.0)         # half-width: the line moves this much in a window
        m = np.abs(f[:, None] - q[None, :]) <= w[None, :]
        Pl = np.where(m, P, 0).max(0)
        L.append(10 * np.log10(Pl + 1e-30)); inb.append((q > F_LO) & (q < F_HI) & span)
    L, inb = np.array(L), np.array(inb)
    ex = L - median_filter(L, size=(1, 31), mode="nearest")
    spike = (ex > SPIKE_DB) & inb
    n_in = inb.sum(0)
    col = (n_in >= 3) & (spike.sum(0) >= np.maximum(3, 0.5 * n_in))
    ok = n_in >= 3
    runs = int(np.sum(np.diff(np.concatenate([[0], col.astype(int)])) == 1))
    return float(col[ok].mean()) if ok.any() else np.nan, runs, (t, f, 10 * np.log10(P + 1e-30))


def key(name):
    return name.replace("mcmaster.wav#", "mcmaster#")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("catalogs", nargs="+")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--png", help="save each stack's first decoy spectrogram here")
    a = ap.parse_args()
    fl = floor()
    cats = [{key(k.name): k for k in stack.load(p)} for p in a.catalogs]
    names = [n for n in cats[0] if all(n in c for c in cats)]
    print(f"{'stack':<14}" + "".join(f"{os.path.basename(p)[:26]:>30}" for p in a.catalogs))
    print(f"{'':<14}" + "".join(f"{'columns % / per decoy':>30}" for _ in a.catalogs))
    for n in names:
        row = f"{n:<14}"
        for ci, (p, c) in enumerate(zip(a.catalogs, cats)):
            k = c[n]
            res = [tell(k, dc, fl) for dc in passes(k, range(a.seeds))]
            fr = np.mean([r[0] for r in res if not np.isnan(r[0])] or [np.nan]); nr = np.mean([r[1] for r in res])
            row += f"{100 * fr:>21.2f} % / {nr:4.1f}"
            if a.png and res[0][2] is not None:
                save_png(res[0][2], os.path.join(a.png, f"{os.path.splitext(os.path.basename(p))[0]}_{n.replace('#', '_')}.png"),
                         f"{n} from {os.path.basename(p)}: decoy seed 0")
        print(row, flush=True)


def save_png(spec, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, f, L = spec
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    top = np.percentile(L, 99.9)
    ax.imshow(np.clip(L - (top - 50), 0, None), origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=50,
              extent=(t[0] - DUR / 2, t[-1] - DUR / 2, f[0], f[-1]), interpolation="antialiased")
    ax.set_xlabel("s from closest approach"); ax.set_ylabel("Hz"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

"""chaff: faint decoy propeller flyovers that mislead acoustic Doppler-curve geolocation."""
from __future__ import annotations

import numpy as np

from . import audibility, noise, synth

__version__ = "0.1.0"
__all__ = ["protect", "__version__"]


def protect(x, sr, per_hour=60, seed=0, below_db=(4, 9), ceiling_below_db=3.0, templates=None,
            catalog=None, lead_per_hour=6.0, quiet_per_hour=54.0):
    """Add decoy flyovers to a recording.

    x: mono (N,) or multichannel (N, C) audio. The noise floor and decoys are computed on the
    channel mean and the same decoy track is added to every channel.

    catalog: a stack catalog (path or list from stack.load / `chaff catalog`). With it, decoys are
    real flyovers re-laid on new passes in two tiers: `lead_per_hour` about as loud as real distant
    planes (enough to outrank the recording's own flyovers) and `quiet_per_hour` under the hearing
    threshold. Without it, decoys are synthesised (synth.py) and kept under the threshold.

    Returns (y, decoys, report): protected audio with the same shape as x (float32), the list
    of synth.Decoy objects, and the audibility.report dict for the added track.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim not in (1, 2):
        raise ValueError(f"expected audio of shape (N,) or (N, C), got {x.shape}")
    sr = int(sr)
    xm = x if x.ndim == 1 else x.mean(axis=1)
    n = len(xm)
    floor = noise.noise_floor(xm, sr)
    if catalog is not None:          # recommended: whole-stack decoys made from real flyovers
        from . import stack
        stacks = stack.load(catalog) if isinstance(catalog, (str, bytes)) or hasattr(catalog, "__fspath__") else catalog
        lead = stack.sample(stacks, n / sr, lead_per_hour, snr_db=(22.0, 32.0), headroom_db=4.0, seed=seed, tier="lead")
        quiet = stack.sample(stacks, n / sr, quiet_per_hour, snr_db=(4.0, 10.0), headroom_db=6.0, seed=seed + 1000, tier="quiet")
        decoys = sorted(lead + quiet, key=lambda d: d.t_cpa)
        added = stack.render(decoys, stacks, n, sr, floor, ceiling_below_db=-20.0).astype(np.float32)
        y = x + (added if x.ndim == 1 else added[:, None])
        return y.astype(np.float32), decoys, audibility.report(added, sr, floor)
    decoys = synth.sample_decoys(n / sr, per_hour=per_hour, seed=seed, templates=templates,
                                 below_db=tuple(below_db))
    added = synth.render(decoys, n, sr, floor, ceiling_below_db=ceiling_below_db).astype(np.float32)
    y = x + (added if x.ndim == 1 else added[:, None])
    rep = audibility.report(added, sr, floor)
    return y.astype(np.float32), decoys, rep

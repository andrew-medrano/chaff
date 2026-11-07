"""Click repair in stack extraction: broadband clicks in the source recording must not survive as a
column of dots across the engine lines, and slow level changes of the lines must survive untouched."""
import numpy as np
import pytest

from chaff import noise, stack, synth
from chaff.resample import Pull
from chaff.stack import NFFT
from chaff.synth import SYN_SR, straight_pass

from conftest import pink

SR, DUR = 4000, 720
EV = dict(t_cpa=360.0, f0=110.0, v=60.0, d=900.0)
CLICKS = [300.0, 333.0, 450.0, 480.0]     # heard times; 333 s is inside the dash
DASH = (315.0, 350.0)                     # the lines dip by 12 dB here, with 6 s ramps


def recording(clicks=(), dash=True):
    """Pink noise, one loud flyover with four harmonics (as loud as a near plane), an optional
    gradual 12 dB dash in its level, and 2 ms broadband clicks."""
    x = pink(SR * DUR, SR, seed=5)
    fl = noise.noise_floor(x, SR)
    dec = [synth.Decoy(EV["t_cpa"], EV["f0"], EV["v"], EV["d"], harm_db=[0.0, -2.0, -4.0, -6.0], below_db=-28.0,
                       seed=0)]
    fly = Pull(synth.render(dec, len(x), SR, fl), SYN_SR, SR).read(len(x))
    if dash:
        t = np.arange(len(x)) / SR
        fly = fly * 10 ** (-12 * np.clip(np.minimum(t - DASH[0], DASH[1] - t) / 6.0, 0, 1) / 20)
    x = x + fly
    rng = np.random.default_rng(1)
    for tc in clicks:
        i = int(tc * SR)
        x[i:i + 8] += 20.0 * rng.standard_normal(8)
    return x.astype(np.float32)


def extract(x):
    return stack.extract(lambda i0, i1: x[i0:i1], SR, len(x), EV, noise.noise_floor(x, SR), name="t")


def warped(k, t):
    """Warped-clock time in stack k of heard time t."""
    tc = np.arange(0, DUR, 0.05)
    fc, _ = straight_pass(tc, EV["f0"], EV["v"], EV["d"], EV["t_cpa"])
    tau = np.concatenate([[0], np.cumsum(0.5 * (fc[1:] + fc[:-1]) / EV["f0"] * 0.05)])
    return k.tau_cpa + np.interp(t, tc, tau) - np.interp(EV["t_cpa"], tc, tau)


def lines_db(k):
    """(t, power of each engine line in dB per 2 s frame) of a stack."""
    f, t, P = stack._stft_power_rows(k.s.astype(np.float64), lambda f: f <= 1000)
    f = f[:P.shape[0]]
    return t, np.array([10 * np.log10(P[(f >= lo - 0.5) & (f <= hi + 0.5)].sum(0) + 1e-30) for lo, hi in k.lines])


@pytest.fixture(scope="module")
def stacks():
    no_repair = lambda x, keep, *a, **kw: np.zeros((len(x) - NFFT) // (NFFT // 8) + 1, bool)
    out = dict(clean=extract(recording()), clicks=extract(recording(CLICKS)), flat=extract(recording(dash=False)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stack, "transient_frames", no_repair)
        out["clicks_unrepaired"] = extract(recording(CLICKS))
        out["clean_unrepaired"] = extract(recording())
    return out


def test_clicks_are_removed(stacks):
    c, d, r = stacks["clean"], stacks["clicks"], stacks["clicks_unrepaired"]
    assert c.n_lines == d.n_lines == r.n_lines == 4
    assert 0 < d.repaired_s <= 3 * len(CLICKS)               # only the frames around the clicks
    t, Lc = lines_db(c)
    _, Ld = lines_db(d)
    _, Lr = lines_db(r)
    for tc in CLICKS:
        dt = np.abs(t - warped(c, tc))
        local = np.median(Lc[:, (dt > 1.5) & (dt < 5)], axis=1)    # each line's level around the click
        raw = Lr[:, dt < 1].max(1) - local
        fixed = Ld[:, dt < 1].max(1) - local
        assert (raw > 8).sum() >= 3, (tc, raw)                     # unrepaired: a dot in (nearly) every line
        assert fixed.max() < 7, (tc, fixed)                        # repaired: no more than natural flicker


def test_slow_changes_are_kept(stacks):
    c, d = stacks["clean"], stacks["clicks"]
    # no transient, no change: a recording without clicks gives exactly the unrepaired stack
    assert c.repaired_s == 0
    np.testing.assert_array_equal(c.s, stacks["clean_unrepaired"].s)
    t, Lc = lines_db(c)
    _, Ld = lines_db(d)
    _, Lf = lines_db(stacks["flat"])
    # the 12 dB dash survives in full, with and without a click inside it
    dt = np.abs(t - warped(c, 333.0))
    mid = (dt < 9) & (dt > 2)
    depth_c = np.median(Lc[:, mid] - Lf[:, mid], axis=1)
    depth_d = np.median(Ld[:, mid] - Lf[:, mid], axis=1)
    np.testing.assert_allclose(depth_c[:3], -12, atol=1.5)
    np.testing.assert_allclose(depth_d, depth_c, atol=0.5)
    # away from the clicks the lines (the swell near closest approach included) are untouched
    far = np.all([np.abs(t - warped(c, tc)) > 4 for tc in CLICKS], axis=0)
    assert np.abs(Ld[:, far] - Lc[:, far]).max() < 0.1


def test_catalog_keeps_repaired_seconds(stacks, tmp_path):
    p = stack.save([stacks["clicks"], stacks["clean"]], tmp_path / "c.npz")
    a, b = stack.load(p)
    assert a.repaired_s == stacks["clicks"].repaired_s > 0 and b.repaired_s == 0

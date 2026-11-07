"""Finding flyovers with the real dopplertrack detector (skipped when dopplertrack is not
installed; test_protect.py covers the same code paths with a stub detector)."""
import numpy as np
import pytest
import soundfile as sf

import chaff
from chaff import detect, stack
from chaff.noise import A_N, A_SR, Frames

from conftest import pink, run_cli, with_flyovers

dt = pytest.importorskip("dopplertrack.audio")

# 8 minutes with two loud flyovers: dopplertrack's detector takes ~30 s on it
EV8 = [dict(t_cpa=160.0, f0=95.0, v=60.0, d=900.0), dict(t_cpa=330.0, f0=120.0, v=70.0, d=1200.0)]


@pytest.fixture(scope="module")
def planes8():
    sr = 8000
    return with_flyovers(pink(sr * 480, sr, seed=11), sr, EV8), sr


def found(stacks):
    usable = [k for k in stacks if k.usable]
    for e in EV8:
        assert any(abs(k.f0 - e["f0"]) < 3 and abs(k.d / e["d"] - 1) < 0.3 for k in usable), \
            [(k.f0, k.d, k.n_lines) for k in stacks]
    return usable


def test_spectrogram_matches_dopplertrack():
    x = pink(A_SR * 1500, A_SR, seed=4)
    ref = dt.compute_spectrogram(x, A_SR)
    fr = Frames(A_SR, A_N, detect.HOP, 3600, detect.F_LO, detect.F_HI, db=True)
    parts = []
    for i in range(0, len(x), 77_777):
        parts += fr.push(x[i:i + 77_777])
    parts += fr.flush()
    L = np.concatenate([p[2] for p in parts], axis=1)
    np.testing.assert_allclose(np.concatenate([p[0] for p in parts]), ref.t)
    np.testing.assert_allclose(fr.f, ref.f)
    np.testing.assert_allclose(detect.whiten(L, 0.25), ref.S, atol=1e-3)


@pytest.mark.slow
def test_protect_self_catalog(planes8):
    # find_flyovers: the recording's own flyovers are found and used as decoy sources
    x, sr = planes8
    y, decoys, rep = chaff.protect(x, sr, find_flyovers=True, seed=1,
                                   quiet=chaff.Tier(60.0, (4.0, 10.0), 6.0, chaff.QUIET.limit_db))
    assert rep["mode"] == "stack", rep["warnings"]
    assert rep["catalog"]["source"].startswith("self") and rep["catalog"]["usable"] >= 2
    assert decoys and y.shape == x.shape and not rep["warnings"]
    assert rep["audibility"]["quiet"]["share_band_seconds_over"] == 0.0


@pytest.mark.slow
def test_cli_catalog_without_events(planes8, tmp_path):
    x, sr = planes8
    rec, cat = tmp_path / "rec.flac", tmp_path / "cat.npz"
    sf.write(rec, x, sr, subtype="PCM_24")
    r = run_cli("catalog", cat, rec, "--top", "5")
    assert r.returncode == 0, r.stderr
    assert "flyovers detected" in r.stderr
    found(stack.load(cat))

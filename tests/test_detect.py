"""Finding flyovers with dopplertrack (skipped when it is not installed)."""
import numpy as np
import pytest

import chaff
from chaff import detect
from chaff.noise import A_N, A_SR, Frames

from conftest import EVENTS, pink, with_flyovers

dt = pytest.importorskip("dopplertrack.audio")


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


def test_self_catalog_from_own_flyovers(planes):
    # no catalog: the recording's own flyovers are found and used as decoy sources
    x, sr = planes
    y, decoys, rep = chaff.protect(x, sr, workers=None)
    assert rep["mode"] == "stack", rep["warnings"]
    assert rep["catalog"]["source"].startswith("self") and rep["catalog"]["usable"] >= 2
    assert decoys and y.shape == x.shape and not rep["warnings"]


def test_catalog_without_events(planes):
    x, sr = planes
    stacks, notes = chaff.build_catalog([((x, sr), None)], workers=None)
    usable = [k for k in stacks if k.usable]
    assert len(usable) >= 2
    for e in EVENTS:
        assert any(abs(k.f0 - e["f0"]) < 3 and abs(k.d / e["d"] - 1) < 0.3 for k in usable)

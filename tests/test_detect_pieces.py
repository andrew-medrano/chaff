"""The detector's piece bookkeeping, with dopplertrack replaced by a stub (always runs)."""
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import numpy as np
import pytest

from chaff import detect
from chaff.noise import A_SR


class Spec:
    def __init__(self, t, f, S):
        self.t, self.f, self.S = t, f, S


def stub(calls, fail_first=False):
    def detect_events(sp, f_lo, f_hi, top=15, workers=None):
        calls.append((sp.t[0], sp.t[-1], workers, sp.S.shape))
        if fail_first and len(calls) == 1:
            raise BrokenProcessPool("no workers")
        mid = 0.5 * (sp.t[0] + sp.t[-1])
        return [SimpleNamespace(t_cpa=mid, f0=100.0, v=60.0, d=1000.0, score=float(mid), burst=0.0),
                SimpleNamespace(t_cpa=sp.t[-1] - 10, f0=90.0, v=60.0, d=1000.0, score=1e9, burst=0.0)]
    return detect_events


def test_pieces_cover_the_recording_with_overlap(monkeypatch):
    calls = []
    monkeypatch.setattr(detect, "_api", lambda: (Spec, stub(calls)))
    d = detect.Detector(piece_s=1800, overlap_s=300, workers=2)
    x = np.zeros(A_SR * 5000, np.float32)
    for i in range(0, len(x), 123_457):
        d.push(x[i:i + 123_457])
    ev = d.finish(max_events=50)
    spans = [(a, b) for a, b, _, _ in calls]
    assert spans[0][0] < 2 and spans[-1][1] > 4990                 # whole recording covered
    for (a0, b0), (a1, b1) in zip(spans, spans[1:]):
        assert a1 < b0 - 290                                         # consecutive pieces overlap by 300 s
    assert all(shape[1] <= 1800 * 4 for *_, shape in calls)          # memory bounded by the piece
    ts = sorted(e["t_cpa"] for e in ev)
    assert all(b - a >= detect.MIN_SEP_S for a, b in zip(ts, ts[1:]))


def test_broken_pool_falls_back_to_one_process(monkeypatch):
    calls = []
    monkeypatch.setattr(detect, "_api", lambda: (Spec, stub(calls, fail_first=True)))
    d = detect.Detector(workers=4)
    d.push(np.zeros(A_SR * 900, np.float32))
    with pytest.warns(RuntimeWarning, match="guard"):
        ev = d.finish()
    assert [c[2] for c in calls] == [4, 1] and ev

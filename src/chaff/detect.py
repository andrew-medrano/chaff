"""Finding the flyovers in a recording, with dopplertrack's detector (optional dependency).

chaff builds its decoys from real flyovers. When no catalog is given, it lifts them out of the
recording being protected: the detector scans the recording for Doppler-shaped curves, and each
detection with a clear engine-line stack becomes a decoy source.

The spectrogram dopplertrack's detector needs (0.25 s hop, 0.49 Hz bins, 40-450 Hz, whitened
over 10 minutes) is computed here from the same 4 kHz stream as the background, in pieces of
two hours with ten minutes of overlap, so memory stays flat however long the recording is.
Install it with `pip install "chaff[detect]"`.
"""
from __future__ import annotations

import numpy as np

from .noise import A_N, A_SR, Frames

HOP = 1000               # 0.25 s at 4 kHz (dopplertrack's default)
F_LO, F_HI = 40.0, 450.0     # spectrogram band (dopplertrack's default)
BAND = (60.0, 340.0)     # fundamentals searched for (as in the experiments)
WHITEN_S = 600.0
PIECE_S = 7200.0
OVERLAP_S = 600.0
MAX_BURST = 0.15         # drop detections that are mostly a broadband burst (a train, a truck)
MIN_SEP_S = 60.0


def _api():
    """dopplertrack's Spectrogram class and detect_events, or ImportError with a clear message."""
    try:
        from dopplertrack.audio import Spectrogram
        from dopplertrack.tracks import detect_events
    except ImportError as e:
        raise ImportError("finding flyovers needs dopplertrack: pip install 'chaff[detect]' "
                          "(or give a catalog, or events JSON, instead)") from e
    return Spectrogram, detect_events


def available() -> bool:
    try:
        _api()
        return True
    except ImportError:
        return False


def whiten(L: np.ndarray, hop_s: float, window_s: float = WHITEN_S) -> np.ndarray:
    """dopplertrack's whitening, in place: subtract per bin the median over `window_s` blocks,
    blended linearly between block centres (or the global median if shorter)."""
    T = L.shape[1]
    w = int(window_s / hop_s)
    if w >= T:
        L -= np.median(L, axis=1, keepdims=True)
        return L
    nb = int(np.ceil(T / w))
    centers = [(b * w + min((b + 1) * w, T)) / 2 for b in range(nb)]
    cm = np.stack([np.median(L[:, b * w:min((b + 1) * w, T)], axis=1) for b in range(nb)])   # (nb, F)
    idx = np.arange(T)
    for k in range(L.shape[0]):
        L[k] = (L[k] - np.interp(idx, centers, cm[:, k])).astype(np.float32)
    return L


class Detector:
    """Push 4 kHz mono samples; finish() returns detected flyovers as dicts (t_cpa, f0, v, d,
    score, burst), strongest first. Detection runs on each piece as soon as it is complete."""

    def __init__(self, top=15, workers=None, piece_s=PIECE_S, overlap_s=OVERLAP_S, log=None):
        self.Spectrogram, self.detect_events = _api()
        self.top, self.workers, self.log = int(top), workers, log
        self.piece = int(piece_s * A_SR / HOP); self.overlap = int(overlap_s * A_SR / HOP)
        self.frames = Frames(A_SR, A_N, HOP, 3600, F_LO, F_HI, db=True)
        self.L, self.t = [], []          # pending dB frames (chunks)
        self.n_pending = 0
        self.n_new = 0                   # pending frames not yet covered by a detection run
        self.events = []

    def _take(self, groups):
        for t, f, P in groups:
            self.L.append(P); self.t.append(t)
            self.n_pending += len(t); self.n_new += len(t)
            while self.n_pending >= self.piece:
                self._run(self.piece)

    def _run(self, n, final=False):
        L = np.concatenate(self.L, axis=1) if len(self.L) > 1 else self.L[0]
        t = np.concatenate(self.t) if len(self.t) > 1 else self.t[0]
        keep = n - self.overlap
        tail = None if final else (L[:, keep:].copy(), t[keep:])
        if self.log:
            self.log(f"  finding flyovers in {t[0] / 3600:.2f}-{t[n - 1] / 3600:.2f} h")
        if t[n - 1] - t[0] > 120:
            sp = self.Spectrogram(t[:n], self.frames.f, whiten(L[:, :n], HOP / A_SR))
            for e in self.detect_events(sp, BAND[0], BAND[1], top=self.top, workers=self.workers):
                self.events.append(dict(t_cpa=float(e.t_cpa), f0=float(e.f0), v=float(e.v), d=float(e.d),
                                        score=float(e.score), burst=float(getattr(e, "burst", 0.0))))
            del sp
        del L
        if final:
            self.L, self.t, self.n_pending, self.n_new = [], [], 0, 0
        else:
            self.L, self.t = [tail[0]], [tail[1]]
            self.n_pending = len(tail[1]); self.n_new = self.n_pending - self.overlap

    def push(self, x4):
        self._take(self.frames.push(x4))

    def finish(self, max_events=20):
        self._take(self.frames.flush())
        if self.n_new > 0:
            self._run(self.n_pending, final=True)
        evs = sorted((e for e in self.events if e["burst"] <= MAX_BURST), key=lambda e: -e["score"])
        out = []
        for e in evs:                    # pieces overlap: the same flyover can be found twice
            if all(abs(e["t_cpa"] - o["t_cpa"]) >= MIN_SEP_S for o in out):
                out.append(e)
        return out[:max_events]

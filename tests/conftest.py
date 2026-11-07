"""Shared fixtures: pink-ish noise, and a recording with loud synthetic flyovers to lift stacks from."""
import numpy as np
import pytest

from chaff import noise, synth
from chaff.audio import ArraySource
from chaff.pipeline import extract_stacks
from chaff.resample import Pull

# two flyovers with four harmonics each, ~40 dB per bin over the background (as loud as near planes)
EVENTS = [dict(t_cpa=300.0, f0=95.0, v=60.0, d=900.0), dict(t_cpa=620.0, f0=120.0, v=70.0, d=1200.0)]


def pink(n, sr, ch=None, seed=0, rms=0.05):
    """Noise with a 1/f power spectrum, (n,) or (n, ch)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n,) if ch is None else (n, ch))
    f = np.fft.rfftfreq(n, 1 / sr); f[0] = f[1]
    shape = (-1,) if ch is None else (-1, 1)
    x = np.fft.irfft(np.fft.rfft(w, axis=0) / np.sqrt(f).reshape(shape), n=n, axis=0)
    return (rms * x / x.std()).astype(np.float32)


def with_flyovers(x, sr, events=EVENTS):
    """x plus loud synthetic flyovers (fundamental and three harmonics) at the given events."""
    fl = noise.noise_floor(x, sr)
    dec = [synth.Decoy(e["t_cpa"], e["f0"], e["v"], e["d"], harm_db=[0.0, -2.0, -4.0, -6.0], below_db=-28.0, seed=i)
           for i, e in enumerate(events)]
    return (x + Pull(synth.render(dec, len(x), sr, fl), synth.SYN_SR, sr).read(len(x))).astype(np.float32)


@pytest.fixture(scope="session")
def planes():
    """(x, sr): 15 min at 8 kHz with two loud flyovers."""
    sr = 8000
    return with_flyovers(pink(sr * 900, sr, seed=11), sr), sr


@pytest.fixture(scope="session")
def stacks(planes):
    x, sr = planes
    return extract_stacks(ArraySource(x, sr), len(x), noise.noise_floor(x, sr), EVENTS)


@pytest.fixture(scope="session")
def catalog_path(stacks, tmp_path_factory):
    from chaff import stack
    return stack.save(stacks, tmp_path_factory.mktemp("cat") / "cat.npz")


SRC = __import__("pathlib").Path(__import__("chaff").__file__).resolve().parents[1]


def run_py(*args, **kw):
    """`python ARGS` in a fresh process, with this checkout's chaff first on the path."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]))
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, env=env, **kw)


def run_cli(*args, **kw):
    """`python -m chaff ARGS`: the real command-line path, including the macOS self-restart."""
    return run_py("-m", "chaff", *args, **kw)


@pytest.fixture
def stub_detector(monkeypatch):
    """Replace dopplertrack's detector by one that 'finds' the given events (always runs; the
    real detector is exercised in test_detect.py). Returns a setter: stub_detector(events)."""
    from types import SimpleNamespace
    from chaff import detect

    class Spec:
        def __init__(self, t, f, S):
            self.t = t

    def install(events):
        def detect_events(sp, f_lo, f_hi, top=15, workers=None):
            return [SimpleNamespace(**e, score=100.0 - i, burst=0.0) for i, e in enumerate(events)
                    if sp.t[0] <= e["t_cpa"] <= sp.t[-1]]
        monkeypatch.setattr(detect, "available", lambda: True)
        monkeypatch.setattr(detect, "_api", lambda: (Spec, detect_events))
    return install

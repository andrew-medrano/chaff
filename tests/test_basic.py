"""Basic tests on synthetic pink-ish noise."""
import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

import chaff
from chaff import cli

CEILING = 3.0


def pink(n, sr, ch=None, seed=0, rms=0.05):
    """Noise with a 1/f power spectrum, (n,) or (n, ch)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n,) if ch is None else (n, ch))
    f = np.fft.rfftfreq(n, 1 / sr); f[0] = f[1]
    shape = (-1,) if ch is None else (-1, 1)
    x = np.fft.irfft(np.fft.rfft(w, axis=0) / np.sqrt(f).reshape(shape), n=n, axis=0)
    return (rms * x / x.std()).astype(np.float32)


@pytest.fixture(scope="module")
def mono():
    sr = 8000
    x = pink(sr * 600, sr, seed=1)
    return x, sr, chaff.protect(x, sr, per_hour=60, seed=0, ceiling_below_db=CEILING)


@pytest.fixture(scope="module")
def stereo():
    sr = 44100
    x = pink(sr * 180, sr, ch=2, seed=2)
    return x, sr, chaff.protect(x, sr, per_hour=60, seed=0, ceiling_below_db=CEILING)


def test_version():
    assert chaff.__version__ == "0.1.0"


@pytest.mark.parametrize("case", ["mono", "stereo"])
def test_shape_dtype_and_decoys_added(case, request):
    x, sr, (y, decoys, rep) = request.getfixturevalue(case)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))
    assert len(decoys) > 0
    assert not np.array_equal(y, x)
    assert np.max(np.abs(y - x)) > 0


def test_stereo_same_track_on_every_channel(stereo):
    x, sr, (y, _, _) = stereo
    d = (y - x).astype(np.float64)
    np.testing.assert_allclose(d[:, 0], d[:, 1], atol=1e-6)


@pytest.mark.parametrize("case", ["mono", "stereo"])
def test_audibility_margin(case, request):
    _, _, (_, _, rep) = request.getfixturevalue(case)
    assert rep["margin_db"] >= CEILING - 0.5, rep
    assert rep["share_band_seconds_over"] == 0.0, rep


def test_deterministic(mono):
    x, sr, (y, decoys, rep) = mono
    y2, decoys2, rep2 = chaff.protect(x, sr, per_hour=60, seed=0, ceiling_below_db=CEILING)
    assert np.array_equal(y, y2)
    assert [d.to_dict() for d in decoys] == [d.to_dict() for d in decoys2]
    assert rep == rep2
    y3, decoys3, _ = chaff.protect(x, sr, per_hour=60, seed=1, ceiling_below_db=CEILING)
    assert not np.array_equal(y, y3)


def test_cli_wav(tmp_path, capsys):
    sr = 8000
    x = pink(sr * 240, sr, ch=2, seed=3)
    src, dst, js = tmp_path / "in.wav", tmp_path / "out.wav", tmp_path / "decoys.json"
    sf.write(src, x, sr, subtype="FLOAT")
    assert cli.main(["protect", str(src), str(dst), "--per-hour", "60", "--seed", "4",
                     "--below", "5", "8", "--ceiling-below", "3", "--decoys-json", str(js)]) == 0
    y, sr2 = sf.read(dst, dtype="float32", always_2d=True)
    assert sr2 == sr and y.shape == x.shape
    assert sf.info(str(dst)).subtype == "FLOAT"
    assert not np.array_equal(y, x)
    meta = json.loads(js.read_text())
    assert len(meta["decoys"]) > 0 and "margin_db" in meta["audibility"]
    out = capsys.readouterr().out
    assert "decoys added" in out and "margin" in out


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
def test_cli_ffmpeg_fallback(tmp_path):
    sr = 8000
    x = pink(sr * 30, sr, seed=5)
    wav, m4a, dst = tmp_path / "a.wav", tmp_path / "a.m4a", tmp_path / "out.flac"
    sf.write(wav, x, sr)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(wav), "-c:a", "aac", str(m4a)],
                   check=True)
    xr, srr = cli.read_audio(str(m4a))
    assert srr == sr and xr.ndim == 2 and xr.shape[1] == 1 and xr.dtype == np.float32
    assert abs(len(xr) - len(x)) < sr          # encoder padding only
    assert cli.main(["protect", str(m4a), str(dst)]) == 0
    y, sr2 = sf.read(dst, always_2d=True)
    assert sr2 == sr and y.shape == xr.shape

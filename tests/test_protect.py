"""protect() and protect_file(): defaults, tiers, audibility, errors, streaming."""
import dataclasses

import numpy as np
import pytest
import soundfile as sf

import chaff
from chaff import detect, pipeline
from chaff.noise import K_MASK_DB

from conftest import pink


def test_defaults_are_the_tested_recipe():
    # v3 (chaffout/FROZEN.md, scripts/chaff/make4.py): lead 6/h at 28-38 dB, headroom -4, limiter
    # +20 dB re background; quiet 54/h at 4-10 dB, headroom 6, and limited under the threshold
    assert chaff.LEAD == chaff.Tier(6.0, (28.0, 38.0), -4.0, 20.0)
    assert chaff.QUIET == chaff.Tier(54.0, (4.0, 10.0), 6.0, K_MASK_DB - 3.0)
    import inspect
    sig = inspect.signature(chaff.protect)
    assert sig.parameters["lead"].default == chaff.LEAD and sig.parameters["quiet"].default == chaff.QUIET


@pytest.fixture(scope="module")
def stacked(catalog_path):
    sr = 8000
    x = pink(sr * 1200, sr, ch=2, seed=3)
    return x, sr, chaff.protect(x, sr, catalog=catalog_path, seed=5)


def test_stack_mode_tiers_and_audibility(stacked):
    x, sr, (y, decoys, rep) = stacked
    assert y.shape == x.shape and y.dtype == np.float32 and np.all(np.isfinite(y))
    assert rep["mode"] == "stack"
    n_lead = sum(d.tier == "lead" for d in decoys); n_quiet = sum(d.tier == "quiet" for d in decoys)
    assert (n_lead, n_quiet) == (rep["tiers"]["lead"]["n"], rep["tiers"]["quiet"]["n"])
    assert n_lead == 2 and n_quiet >= 15                   # round(6 * 20/60), round(54 * 20/60) = 18 minus misfits
    a = rep["audibility"]
    assert a["quiet"]["worst_band_db"] <= chaff.QUIET.limit_db + 0.1      # quiet tier stays under the threshold
    assert a["quiet"]["share_band_seconds_over"] == 0.0
    assert a["lead"]["worst_band_db"] <= chaff.LEAD.limit_db + 0.1
    assert a["combined"]["worst_band_db"] <= chaff.LEAD.limit_db + 0.2
    assert a["lead"]["worst_band_db"] > 10                 # lead decoys are loud, as designed
    d = (y - x).astype(np.float64)
    np.testing.assert_allclose(d[:, 0], d[:, 1], atol=1e-6)


def test_deterministic(stacked, catalog_path):
    x, sr, (y, decoys, rep) = stacked
    y2, decoys2, rep2 = chaff.protect(x, sr, catalog=catalog_path, seed=5)
    assert np.array_equal(y, y2) and [d.to_dict() for d in decoys] == [d.to_dict() for d in decoys2]


def test_file_streaming_equals_in_memory(tmp_path, catalog_path):
    sr = 44100
    x = pink(sr * 300, sr, ch=2, seed=4)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, x, sr, subtype="FLOAT")
    y, decoys, rep = chaff.protect(x, sr, catalog=catalog_path, seed=2)
    decoys_f, rep_f = chaff.protect_file(src, dst, catalog=catalog_path, seed=2)
    yf, srf = sf.read(dst, dtype="float32", always_2d=True)
    assert srf == sr and yf.shape == x.shape and sf.info(str(dst)).subtype == "FLOAT"
    assert np.max(np.abs(yf - y)) < 1e-6
    assert [d.to_dict() for d in decoys] == [d.to_dict() for d in decoys_f]
    assert rep["audibility"] == rep_f["audibility"]


def test_synthetic_mode(tmp_path):
    sr = 8000
    x = pink(sr * 600, sr, seed=1)
    y, decoys, rep = chaff.protect(x, sr, synthetic=True)
    assert rep["mode"] == "synthetic" and len(decoys) == 10 and y.shape == x.shape
    a = rep["audibility"]["synthetic"]
    assert a["worst_band_db"] <= chaff.SYNTHETIC.limit_db + 0.1 and a["share_band_seconds_over"] == 0.0


def test_no_catalog_without_dopplertrack_falls_back_with_warning(monkeypatch):
    monkeypatch.setattr(detect, "available", lambda: False)
    sr = 8000
    x = pink(sr * 300, sr, seed=2)
    with pytest.warns(chaff.ChaffWarning, match="synthetic decoys"):
        y, decoys, rep = chaff.protect(x, sr)
    assert rep["mode"] == "synthetic" and rep["warnings"] and len(decoys) > 0


def test_unusable_catalog_is_an_error(stacks):
    sr = 8000
    x = pink(sr * 300, sr, seed=2)
    weak = [dataclasses.replace(k, lines=k.lines[:2]) for k in stacks]       # two lines: not a stack
    with pytest.raises(ValueError, match="unusable"):
        chaff.protect(x, sr, catalog=weak)
    with pytest.raises(ValueError, match="unusable"):
        chaff.protect(x, sr, catalog=[])
    quiet_only = [dataclasses.replace(k, iso_snr_db=12.0) for k in stacks]
    with pytest.warns(chaff.ChaffWarning, match="lead tier skipped"):
        _, decoys, rep = chaff.protect(x, sr, catalog=quiet_only)
    assert rep["tiers"]["lead"]["n"] == 0 and rep["audibility"]["lead"] is None and decoys


def test_short_recordings(stacks):
    sr = 8000
    with pytest.raises(ValueError, match="at least 60 s"):
        chaff.protect(pink(sr * 30, sr), sr, catalog=stacks)
    x = pink(sr * 100, sr, seed=7)                         # shorter than 2 x the 150 s edge
    y, decoys, rep = chaff.protect(x, sr, catalog=stacks, quiet=dataclasses.replace(chaff.QUIET, per_hour=120))
    assert decoys and all(0 < d.t_cpa < 100 for d in decoys)
    with pytest.raises(ValueError, match="no decoys placed"):
        chaff.protect(x, sr, catalog=stacks, quiet=dataclasses.replace(chaff.QUIET, per_hour=6))


def test_output_format_checked_before_work(tmp_path):
    src = tmp_path / "in.wav"
    sf.write(src, np.zeros(8000 * 2, np.float32), 8000)
    with pytest.raises(ValueError, match="cannot write .m4a"):
        chaff.protect_file(src, tmp_path / "out.m4a", synthetic=True)


def test_integer_output_is_clipped_not_wrapped(tmp_path):
    sr = 8000
    x = pink(sr * 120, sr, seed=8, rms=0.6)
    src, dst = tmp_path / "in.wav", tmp_path / "out.flac"
    sf.write(src, x, sr, subtype="FLOAT")
    with pytest.warns(chaff.ChaffWarning, match="clipped"):
        _, rep = chaff.protect_file(src, dst, synthetic=True)
    y, _ = sf.read(dst, dtype="float32")
    assert rep["output"]["clipped_samples"] > 0 and np.max(np.abs(y)) <= 1.0
    assert np.corrcoef(y, np.clip(x, -1, 1))[0, 1] > 0.99

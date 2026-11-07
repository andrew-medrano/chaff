"""protect() and protect_file(): defaults, tiers, audibility, decoy sources, errors, streaming."""
import dataclasses
import inspect
import warnings

import numpy as np
import pytest
import scipy.signal as ss
import soundfile as sf

import chaff
from chaff import audio, detect, pipeline
from chaff.noise import K_MASK_DB

from conftest import EVENTS, pink

# for 2-4 min test files: enough decoys that both tiers appear (rendering dominates the run time)
FEW = dict(lead=dataclasses.replace(chaff.LEAD, per_hour=30.0), quiet=dataclasses.replace(chaff.QUIET, per_hour=60.0))


def check_audibility(rep):
    """Each tier stays under its own limiter ceiling; the quiet tier under the hearing threshold."""
    a = rep["audibility"]
    assert a["threshold_db"] == K_MASK_DB
    lead, quiet = rep["tiers"]["lead"], rep["tiers"]["quiet"]
    if a["quiet"] is not None:
        assert a["quiet"]["worst_band_db"] <= quiet["limit_db"] + 0.1
        assert a["quiet"]["worst_band_db"] < K_MASK_DB and a["quiet"]["share_band_seconds_over"] == 0.0
    if a["lead"] is not None:
        assert a["lead"]["worst_band_db"] <= lead["limit_db"] + 0.1
    # the sum can exceed either ceiling only by the quiet tier's (sub-threshold) power
    assert a["combined"]["worst_band_db"] <= max(lead["limit_db"], quiet["limit_db"]) + 0.2


def test_defaults_are_the_tested_recipe():
    # v3 (chaffout/FROZEN.md, scripts/chaff/make4.py): lead 6/h at 28-38 dB, headroom -4, limiter
    # +20 dB re background; quiet 54/h at 4-10 dB, headroom 6, and limited under the threshold
    assert chaff.LEAD == chaff.Tier(6.0, (28.0, 38.0), -4.0, 20.0)
    assert chaff.QUIET == chaff.Tier(54.0, (4.0, 10.0), 6.0, K_MASK_DB - 3.0)
    for fn in (chaff.protect, chaff.protect_file):
        sig = inspect.signature(fn)
        assert sig.parameters["lead"].default == chaff.LEAD and sig.parameters["quiet"].default == chaff.QUIET
        assert sig.parameters["catalog"].default is None and sig.parameters["synthetic"].default is False
        assert sig.parameters["find_flyovers"].default is False and sig.parameters["jobs"].default is None


@pytest.fixture(scope="module")
def stacked(catalog_path):
    sr = 8000
    x = pink(sr * 1200, sr, ch=2, seed=3)
    return x, sr, chaff.protect(x, sr, catalog=catalog_path, seed=5)


def test_catalog_mode_default_tiers(stacked):
    x, sr, (y, decoys, rep) = stacked
    assert y.shape == x.shape and y.dtype == np.float32 and np.all(np.isfinite(y))
    assert rep["mode"] == "stack" and rep["catalog"]["usable"] == 2 and not rep["warnings"]
    n_lead = sum(d.tier == "lead" for d in decoys); n_quiet = sum(d.tier == "quiet" for d in decoys)
    assert (n_lead, n_quiet) == (rep["tiers"]["lead"]["n"], rep["tiers"]["quiet"]["n"]) == (rep["n_decoys"] - n_quiet, n_quiet)
    assert n_lead == 2 and n_quiet >= 15                   # round(6 * 20/60), round(54 * 20/60) = 18 minus misfits
    for d in decoys:
        lo, hi = (chaff.LEAD if d.tier == "lead" else chaff.QUIET).snr_db
        assert lo <= d.level_db <= hi and 0 < d.t_cpa < 1200
    check_audibility(rep)
    assert rep["audibility"]["lead"]["worst_band_db"] > 10           # lead decoys are loud, as designed
    d = (y - x).astype(np.float64)
    np.testing.assert_allclose(d[:, 0], d[:, 1], atol=1e-6)         # same decoy track on every channel


def test_deterministic(catalog_path):
    sr = 8000
    x = pink(sr * 150, sr, seed=6)
    r1 = chaff.protect(x, sr, catalog=catalog_path, seed=5, **FEW)
    r2 = chaff.protect(x, sr, catalog=catalog_path, seed=5, **FEW)
    assert np.array_equal(r1[0], r2[0]) and [d.to_dict() for d in r1[1]] == [d.to_dict() for d in r2[1]]


@pytest.mark.parametrize("sr,ch", [(16000, 1), (22050, 2), (44100, 3), (48000, 2), (96000, 1)])
def test_sample_rates_and_channels(sr, ch, stacks):
    x = pink(sr * 150, sr, ch=None if ch == 1 else ch, seed=sr)
    y, decoys, rep = chaff.protect(x, sr, catalog=stacks, seed=1, **FEW)
    assert y.shape == x.shape and y.dtype == np.float32 and rep["sr"] == sr and rep["channels"] == ch
    assert abs(rep["duration_s"] - 150) < 1e-6 and rep["mode"] == "stack"
    assert {d.tier for d in decoys} == {"lead", "quiet"}
    check_audibility(rep)
    d = (y - x).reshape(len(x), -1).astype(np.float64)
    for c in range(1, ch):
        np.testing.assert_allclose(d[:, c], d[:, 0], atol=1e-6)
    # decoys live below 1 kHz (rendered at 2 kHz): what is added above it is resampler leakage, < -40 dB
    f, P = ss.welch(d[:, 0], sr, nperseg=8192)
    assert P[f > 1100].sum() < 1e-4 * P[f < 1000].sum()


def test_sample_rate_too_low():
    with pytest.raises(ValueError, match="too low"):
        chaff.protect(pink(1000 * 120, 1000), 1000, synthetic=True)


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
    assert rep_f["output"]["frames"] == len(x) and rep_f["output"]["clipped_samples"] == 0


def test_block_size_does_not_change_the_output(tmp_path, monkeypatch, catalog_path):
    # the same file streamed in blocks of 2^18 and of an odd 12 345 frames: block boundaries leave no trace
    sr = 48000
    x = pink(sr * 150, sr, ch=2, seed=12)
    src = tmp_path / "in.wav"
    sf.write(src, x, sr, subtype="FLOAT")
    out = {}
    for block in (audio.BLOCK, 12_345):
        monkeypatch.setattr(pipeline, "BLOCK", block)
        dst = tmp_path / f"out{block}.wav"
        decoys, rep = chaff.protect_file(src, dst, catalog=catalog_path, seed=4, **FEW)
        out[block] = sf.read(dst, dtype="float32")[0], [d.to_dict() for d in decoys]
    (ya, da), (yb, db) = out.values()
    assert da == db
    assert np.max(np.abs(ya - yb)) < 1e-6
    # and no step at any block boundary: the added track is smooth across them
    d = (yb - x)[:, 0].astype(np.float64)
    jumps = np.abs(np.diff(d))
    at = np.arange(12_345, len(d) - 1, 12_345) - 1
    assert jumps[at].max() <= 1.5 * np.quantile(jumps, 0.999)


def test_synthetic_mode():
    sr = 8000
    x = pink(sr * 600, sr, seed=1)
    y, decoys, rep = chaff.protect(x, sr, synthetic=True)
    assert rep["mode"] == "synthetic" and len(decoys) == 10 and y.shape == x.shape
    a = rep["audibility"]["synthetic"]
    assert a["worst_band_db"] <= chaff.SYNTHETIC.limit_db + 0.1 and a["share_band_seconds_over"] == 0.0


def test_self_catalog_mode(planes, stub_detector):
    # find_flyovers: the flyovers found in the recording itself become the decoy sources
    x, sr = planes
    stub_detector(EVENTS)
    y, decoys, rep = chaff.protect(x, sr, find_flyovers=True, seed=3,
                                   quiet=dataclasses.replace(chaff.QUIET, per_hour=12.0))
    assert rep["mode"] == "stack" and not rep["warnings"]
    c = rep["catalog"]
    assert c["source"].startswith("self") and c["detected"] == 2 and c["usable"] == 2
    np.testing.assert_allclose(sorted(u["f0"] for u in c["used"]), [e["f0"] for e in EVENTS], rtol=0.01)
    assert decoys and {d.src_name for d in decoys} <= {u["name"] for u in c["used"]}
    check_audibility(rep)


def test_self_catalog_without_stacks_falls_back_to_bundled(stub_detector):
    # the detector "finds" a flyover in plain noise: no engine lines, so the bundled catalog and a warning
    sr = 8000
    x = pink(sr * 300, sr, seed=2)
    stub_detector([dict(t_cpa=150.0, f0=100.0, v=60.0, d=1000.0)])
    with pytest.warns(chaff.ChaffWarning, match="using the bundled catalog instead"):
        y, decoys, rep = chaff.protect(x, sr, find_flyovers=True, **FEW)
    assert rep["mode"] == "stack" and rep["catalog"]["source"].startswith("bundled")
    assert rep["catalog"]["self"]["detected"] == 1
    assert decoys and rep["warnings"] and y.shape == x.shape
    assert all(d.src_name.startswith("mcmaster#") for d in decoys)


def test_find_flyovers_without_dopplertrack_is_an_error(monkeypatch):
    def missing():
        raise ImportError("finding flyovers needs dopplertrack: pip install 'chaff[detect]'")
    monkeypatch.setattr(detect, "_api", missing)
    sr = 8000
    x = pink(sr * 300, sr, seed=2)
    with pytest.raises(ImportError, match="chaff\\[detect\\]"):
        chaff.protect(x, sr, find_flyovers=True)


def test_default_is_the_bundled_catalog(monkeypatch):
    # no catalog, no dopplertrack needed: the bundled catalog, and no warning
    monkeypatch.setattr(detect, "_api", lambda: (_ for _ in ()).throw(AssertionError("detector used")))
    sr = 8000
    x = pink(sr * 240, sr, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", chaff.ChaffWarning)
        y, decoys, rep = chaff.protect(x, sr, seed=4, **FEW)
    c = rep["catalog"]
    assert rep["mode"] == "stack" and not rep["warnings"] and decoys
    assert c["source"].startswith("bundled") and c["path"] == chaff.bundled_catalog()
    assert c["credit"] == "Hamilton Naturalists' Club, via the Internet Archive"
    assert {d.tier for d in decoys} == {"lead", "quiet"}
    assert all(d.src_name.startswith("mcmaster#") for d in decoys)
    check_audibility(rep)
    # the same as passing the bundled catalog's path
    y2, decoys2, _ = chaff.protect(x, sr, chaff.bundled_catalog(), seed=4, **FEW)
    assert np.array_equal(y, y2) and [d.to_dict() for d in decoys] == [d.to_dict() for d in decoys2]


def test_one_decoy_source_at_a_time(stacks):
    x = pink(8000 * 120, 8000)
    for kw in (dict(catalog=stacks, synthetic=True), dict(catalog=stacks, find_flyovers=True),
               dict(find_flyovers=True, synthetic=True)):
        with pytest.raises(ValueError, match="one decoy source"):
            chaff.protect(x, 8000, **kw)


def test_unusable_catalog_is_an_error(stacks):
    sr = 8000
    x = pink(sr * 300, sr, seed=2)
    weak = [dataclasses.replace(k, lines=k.lines[:2]) for k in stacks]       # two lines: not a stack
    with pytest.raises(ValueError, match="unusable"):
        chaff.protect(x, sr, catalog=weak)
    with pytest.raises(ValueError, match="unusable"):
        chaff.protect(x, sr, catalog=[])
    too_quiet = [dataclasses.replace(k, iso_snr_db=5.0) for k in stacks]      # below both tiers
    with pytest.raises(ValueError, match="loud enough"):
        chaff.protect(x, sr, catalog=too_quiet)
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


def test_bad_input_shape():
    with pytest.raises(ValueError, match="shape"):
        chaff.protect(np.zeros((2, 3, 4), np.float32), 8000, synthetic=True)


def test_output_format_checked_before_work(tmp_path):
    src = tmp_path / "in.wav"
    sf.write(src, np.zeros(8000 * 2, np.float32), 8000)
    with pytest.raises(ValueError, match="cannot write .m4a"):
        chaff.protect_file(src, tmp_path / "out.m4a", synthetic=True)
    with pytest.raises(ValueError, match="overwrite the input"):
        chaff.protect_file(src, src, synthetic=True)


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


def test_wav_over_4_gb_is_written_as_rf64(tmp_path):
    # the header decision only: 12 h of 48 kHz stereo float is 16.6 GB
    with audio.Writer(tmp_path / "big.wav", 48000, 2, frames=48000 * 3600 * 12) as w:
        w.write(np.zeros((100, 2), np.float32))
    assert w.rf64 and sf.info(str(tmp_path / "big.wav")).format == "RF64"
    with audio.Writer(tmp_path / "small.wav", 48000, 2, frames=48000 * 60) as w:
        w.write(np.zeros((100, 2), np.float32))
    assert not w.rf64 and sf.info(str(tmp_path / "small.wav")).format == "WAV"

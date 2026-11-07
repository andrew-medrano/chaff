"""Stack extraction and the catalog file format."""
import json
import warnings

import numpy as np
import pytest

from chaff import noise, stack
from chaff.audio import ArraySource
from chaff.pipeline import extract_stacks

from conftest import EVENTS


def test_stacks_have_grouped_engine_lines(stacks):
    assert len(stacks) == 2
    for k, e in zip(stacks, EVENTS):
        # four harmonics -> four lines (each several bins wide), at multiples of the rest frequency
        assert k.n_lines == 4 and k.usable
        np.testing.assert_allclose(k.lines_hz, e["f0"] * np.arange(1, 5), rtol=0.01)
        assert np.all(k.lines[:, 1] >= k.lines[:, 0])
        assert k.iso_snr_db > 30


def test_save_load_round_trip_without_pickle(stacks, catalog_path):
    with np.load(catalog_path, allow_pickle=False) as z:          # plain arrays only
        assert str(z["format"]) == stack.FORMAT and int(z["version"]) == stack.FORMAT_VERSION
        meta = json.loads(str(z["meta"]))
    assert [m["name"] for m in meta] == [k.name for k in stacks]
    back = stack.load(catalog_path)
    for a, b in zip(stacks, back):
        assert a.name == b.name and a.iso_snr_db == b.iso_snr_db and a.f0 == b.f0
        np.testing.assert_array_equal(a.lines, b.lines)
        np.testing.assert_allclose(a.s, b.s, atol=2e-3 * np.abs(a.s).max())    # stored as float16


def test_old_pickled_catalog_is_refused(tmp_path):
    p = tmp_path / "old.npz"
    np.savez(p, meta=np.array([dict(name="x")], dtype=object), s0=np.zeros(3))
    with pytest.raises(ValueError, match="not a chaff catalog"):
        stack.load(p)


def test_same_names_do_not_share_levels(stacks):
    # the isolated SNR is stored on each stack, not cached by name
    a, b = stacks
    b2 = stack.Stack(**{**b.__dict__, "name": a.name})
    assert b2.iso_snr_db == b.iso_snr_db != a.iso_snr_db


def test_event_outside_recording_is_skipped(planes):
    x, sr = planes
    notes = []
    out = extract_stacks(ArraySource(x, sr), len(x), noise.noise_floor(x, sr),
                         [dict(t_cpa=5000.0, f0=90.0, v=60.0, d=900.0), EVENTS[0]], notes=notes)
    assert len(out) == 1 and "outside the recording" in notes[0]


def test_no_nola_warning(planes):
    x, sr = planes
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extract_stacks(ArraySource(x, sr), len(x), noise.noise_floor(x, sr), EVENTS[:1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_blockwise_stft_gain_equals_scipy_round_trip(dtype):
    import scipy.signal as ss
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000 * 300).astype(dtype)
    g = rng.uniform(0, 2, stack.NFFT // 2 + 1)
    _, _, Z = ss.stft(x, 2000, window="hann", nperseg=stack.NFFT, noverlap=stack.NFFT - stack.NFFT // 8,
                      boundary=None, padded=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, ref = ss.istft(Z * g[:, None], 2000, window="hann", nperseg=stack.NFFT,
                          noverlap=stack.NFFT - stack.NFFT // 8, boundary=False)
    y = stack.stft_gain(x, g)
    assert len(y) == len(ref)
    np.testing.assert_array_equal(y, ref)         # bit-identical with the default block size
    y = stack.stft_gain(x, g, block=37)           # other block sizes: FFT batching rounds differently
    np.testing.assert_allclose(y, ref, rtol=0, atol=1e-6 * np.abs(ref).max())


def test_not_a_catalog(tmp_path):
    p = tmp_path / "junk.npz"
    p.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a chaff catalog"):
        stack.load(p)
    np.save(tmp_path / "arr.npy", np.zeros(3))
    with pytest.raises(ValueError, match="not a chaff catalog"):
        stack.load(tmp_path / "arr.npy")


class _Trap:
    """Unpickling this creates a file: proof that a catalog load never unpickles."""
    path = None

    def __reduce__(self):
        return (open, (self.path, "w"))


def test_pickle_payload_is_never_run(tmp_path):
    trap = tmp_path / "unpickled"
    _Trap.path = str(trap)
    for extra in ({}, dict(format=np.array(stack.FORMAT), version=np.array(stack.FORMAT_VERSION))):
        p = tmp_path / "evil.npz"
        np.savez(p, meta=np.array([_Trap()], dtype=object), **extra)
        with pytest.raises(ValueError):
            stack.load(p)
        assert not trap.exists()


def test_other_format_version_is_refused(stacks, tmp_path):
    p = stack.save(stacks, tmp_path / "cat")            # .npz is added
    assert p.endswith("cat.npz")
    with np.load(p, allow_pickle=False) as z:
        d = dict(z)
    d["version"] = np.array(stack.FORMAT_VERSION + 1)
    np.savez(tmp_path / "newer.npz", **d)
    with pytest.raises(ValueError, match=f"format {stack.FORMAT_VERSION + 1}"):
        stack.load(tmp_path / "newer.npz")


def test_empty_catalog_round_trip(tmp_path):
    p = stack.save([], tmp_path / "empty.npz")
    assert stack.load(p) == []


def test_catalog_info_round_trip(stacks, tmp_path):
    p = stack.save(stacks, tmp_path / "c.npz", info=dict(credit="someone", n=2))
    assert stack.load_info(p) == dict(credit="someone", n=2)
    assert len(stack.load(p)) == 2
    assert stack.load_info(stack.save(stacks, tmp_path / "plain.npz")) == {}


def test_bundled_catalog():
    import os
    import chaff
    p = chaff.bundled_catalog()
    assert os.path.basename(p) == "mcmaster.npz" and os.path.getsize(p) < 21e6
    ks = chaff.load_catalog(p)
    assert len(ks) == 8 and all(k.usable for k in ks)                 # only stacks a tier can use
    assert all(k.name.startswith("mcmaster#") and k.s.dtype == np.float32 for k in ks)
    assert len(stack.eligible(ks, chaff.LEAD.snr_db, chaff.LEAD.headroom_db)) == 6
    assert len(stack.eligible(ks, chaff.QUIET.snr_db, chaff.QUIET.headroom_db)) == 8
    info = chaff.catalog_info(p)
    assert info["credit"] == "Hamilton Naturalists' Club, via the Internet Archive"
    assert info["source_url"] == "https://archive.org/details/HNCSW720260527"
    assert "no" in info["licence"] and "HNCSW7" in info["source"]
    assert info["data_version"] == 2                                  # 0.3.1: clicks repaired
    assert [k.name for k in ks] == [f"mcmaster#{i}" for i in (1, 2, 3, 4, 5, 7, 9, 10)]
    assert [s["clicks_repaired_s"] for s in info["stacks"]] == [round(k.repaired_s, 1) for k in ks]
    assert [k.name for k in ks if k.repaired_s == 0] == ["mcmaster#7"]
    assert max(ks, key=lambda k: k.repaired_s).name == "mcmaster#5"          # the one with the most clicks
    assert [s["name"] for s in info["stacks"]] == [k.name for k in ks]
    # its README, with the same credit, sits next to it
    readme = os.path.join(os.path.dirname(p), "README.md")
    assert info["credit"] in open(readme).read()


def _line_columns(k, db=10.0):
    """Frames of a stack where at least half of its engine lines (and at least 3) are more than db
    over their own 8 s running median at once: what a click left in the stack looks like."""
    from scipy.ndimage import median_filter
    f, t, P = stack._stft_power_rows(k.s.astype(np.float64), lambda f: f <= 1000)
    f = f[:P.shape[0]]
    L = np.array([10 * np.log10(P[(f >= lo - 0.5) & (f <= hi + 0.5)].sum(0) + 1e-30) for lo, hi in k.lines])
    spikes = (L - median_filter(L, size=(1, 31), mode="nearest") > db).sum(0)
    col = spikes >= max(3, len(L) / 2)
    col[:16] = col[-16:] = False                                      # the first and last 4 s
    return t[col] - k.tau_cpa


def test_bundled_catalog_has_no_click_columns():
    import chaff
    cols = {k.name: _line_columns(k) for k in chaff.load_catalog(chaff.bundled_catalog())}
    # chaff 0.3.0's file had 13 such frames in mcmaster#5 (clicks at -272 and +230 s), one in
    # mcmaster#2 (-133 s) and one in mcmaster#10. That last one is what is left: a short harmonic
    # chirp (150/300/450 Hz, on the stack's own harmonics, so not broadband) 5.6 s from its start.
    assert {n: len(c) for n, c in cols.items() if len(c)} == {"mcmaster#10": 1}
    assert abs(cols["mcmaster#10"][0] + 344) < 2

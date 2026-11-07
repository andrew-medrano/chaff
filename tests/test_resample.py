"""Block-by-block resampling and background must not depend on how the signal is split."""
from math import gcd

import numpy as np
import pytest
import scipy.signal as ss

from chaff import noise
from chaff.resample import Pull, Resampler

RATES = [(48000, 4000), (44100, 4000), (22050, 4000), (16000, 4000), (4000, 4000),
         (2000, 4000), (2000, 48000), (2000, 44100), (2000, 8000)]


@pytest.mark.parametrize("sr_in,sr_out", RATES)
def test_chunked_equals_one_shot(sr_in, sr_out):
    rng = np.random.default_rng(sr_in + sr_out)
    x = rng.standard_normal(120_007)
    g = gcd(sr_in, sr_out)
    ref = ss.resample_poly(x, sr_out // g, sr_in // g)
    r, out, i = Resampler(sr_in, sr_out), [], 0
    while i < len(x):
        k = int(rng.integers(1, 20_000)); out.append(r.push(x[i:i + k])); i += k
    out.append(r.flush())
    y = np.concatenate(out)
    assert len(y) == len(ref)
    assert np.max(np.abs(y - ref)) < 1e-9


@pytest.mark.parametrize("sr_in,sr_out", [(2000, 48000), (2000, 44100), (2000, 8000)])
def test_pull_reads_any_piece_lengths(sr_in, sr_out):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(30_011)
    g = gcd(sr_in, sr_out)
    ref = ss.resample_poly(x, sr_out // g, sr_in // g)
    p = Pull(x, sr_in, sr_out)
    parts, n = [], 0
    while n < len(ref) + 5000:
        k = int(rng.integers(1, 40_000)); parts.append(p.read(k)); n += k
    y = np.concatenate(parts)
    assert np.max(np.abs(y[:len(ref)] - ref)) < 1e-6        # float32 output
    assert not np.any(y[len(ref):])                          # zeros past the end


def _floor_reference(x, sr):
    """The background computed the pre-streaming way: whole-signal resample, then 60 s blocks."""
    from scipy.ndimage import median_filter
    x4 = ss.resample_poly(x, *(np.array([noise.A_SR, sr]) // gcd(noise.A_SR, sr))).astype(np.float32)
    f, t, Z = ss.stft(x4, noise.A_SR, window="hann", nperseg=noise.A_N, noverlap=noise.A_N - noise.A_HOP,
                      boundary=None, padded=False, scaling="spectrum")
    band = f <= noise.F_MAX
    P = np.abs(Z[band]) ** 2
    t = (noise.A_N / 2 + np.arange(P.shape[1]) * noise.A_HOP) / noise.A_SR
    ms, ts = [], []
    for b in range(0, P.shape[1], 60):
        if P[:, b:b + 60].shape[1] >= 5:
            ms.append(np.median(P[:, b:b + 60], axis=1) / np.log(2.0)); ts.append(t[b:b + 60].mean())
    return np.array(ts), np.maximum(median_filter(np.stack(ms, 1), size=(15, 3), mode="nearest"), 1e-30)


@pytest.mark.parametrize("sr", [4000, 48000, 44100])
def test_streamed_background_matches_whole_signal(sr):
    rng = np.random.default_rng(sr)
    x = (0.05 * rng.standard_normal(sr * 200)).astype(np.float32)
    t_ref, M_ref = _floor_reference(x, sr)
    for block in (sr * 7 + 13, 1 << 20):
        fl = noise.noise_floor(x, sr, block=block)
        np.testing.assert_array_equal(fl.t, t_ref)
        np.testing.assert_allclose(fl.M, M_ref, rtol=1e-5)


def test_background_too_short():
    with pytest.raises(ValueError, match="at least 6.0 s"):
        noise.noise_floor(np.zeros(4000 * 5, np.float32), 4000)

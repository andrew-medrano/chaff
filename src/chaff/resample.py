"""Polyphase resampling, in one go or block by block.

`Resampler` gives the same samples as `scipy.signal.resample_poly` on the whole signal (same
Kaiser-windowed filter, same alignment, zeros past both ends), but takes the input in blocks of
any size and keeps only a filter's length of history. That is what lets a 12-hour recording be
analysed and written without holding it in memory.
"""
from __future__ import annotations

from math import gcd

import numpy as np
import scipy.signal as ss


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Whole-signal resampling (scipy.signal.resample_poly), float32 out."""
    x = np.asarray(x, np.float32)
    if int(sr) == int(target):
        return x
    g = gcd(int(sr), int(target))
    return ss.resample_poly(x, int(target) // g, int(sr) // g).astype(np.float32)


def _design(up: int, down: int):
    """resample_poly's filter, front-padded the same way, and how many outputs it drops."""
    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = ss.firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * up
    n_pre_pad = down - half_len % down
    return np.concatenate([np.zeros(n_pre_pad), h]), (half_len + n_pre_pad) // down


class Resampler:
    """Streaming resample_poly for one channel.

    push(block) returns the output samples that are now final; flush() returns the rest. The
    concatenated outputs equal resample_poly(whole signal) to float64 rounding."""

    def __init__(self, sr_in: int, sr_out: int):
        g = gcd(int(sr_in), int(sr_out))
        self.up, self.down = int(sr_out) // g, int(sr_in) // g
        self.identity = self.up == self.down == 1
        if not self.identity:
            self.h, self.skip = _design(self.up, self.down)
        self.buf = np.zeros(0)
        self.b0 = 0              # absolute index of buf[0]; always a multiple of `down`
        self.n_in = 0
        self.m = 0               # next output index to hand out

    def n_out(self) -> int:
        """Total output length for the input pushed so far (resample_poly's)."""
        return -(-(self.n_in * self.up) // self.down)

    def push(self, x) -> np.ndarray:
        x = np.asarray(x, np.float64).reshape(-1)
        self.n_in += len(x)
        if self.identity:
            self.m += len(x)
            return x.copy()
        self.buf = np.concatenate([self.buf, x]) if len(self.buf) else x.copy()
        # output m is final once every input it touches has arrived: (m + skip) * down < n_in * up
        return self._emit(self.n_out() - self.skip)

    def flush(self) -> np.ndarray:
        if self.identity:
            return np.zeros(0)
        m_end = self.n_out()
        if m_end <= self.m:
            return np.zeros(0)
        need = (m_end - 1 + self.skip) * self.down // self.up + 1     # last input index touched, + 1
        have = self.b0 + len(self.buf)
        if need > have:
            self.buf = np.concatenate([self.buf, np.zeros(need - have)])
        return self._emit(m_end)

    def _emit(self, m_end: int) -> np.ndarray:
        if m_end <= self.m or not len(self.buf):
            return np.zeros(0)
        y = ss.upfirdn(self.h, self.buf, self.up, self.down)
        off = self.b0 * self.up // self.down - self.skip           # output index of y[0]
        out = y[self.m - off:m_end - off]
        assert len(out) == m_end - self.m
        self.m = m_end
        # keep only the inputs the next output still needs
        lo = -(-((self.m + self.skip) * self.down - (len(self.h) - 1)) // self.up)
        nb0 = max((lo // self.down) * self.down, self.b0)
        if nb0 > self.b0:
            self.buf = self.buf[nb0 - self.b0:]
            self.b0 = nb0
        return out


class Pull:
    """Read a resampled copy of an in-memory signal in consecutive pieces of any length.

    Pieces past the end of the resampled signal are zero, so it can be read for exactly as many
    samples as the recording it is added to."""

    def __init__(self, x: np.ndarray, sr_in: int, sr_out: int):
        self.x, self.pos = x, 0
        self.rs = Resampler(sr_in, sr_out)
        self.pending = [np.zeros(0)]
        self.n_pending = 0
        self.flushed = False

    def read(self, n: int) -> np.ndarray:
        while self.n_pending < n and not self.flushed:
            if self.pos < len(self.x):
                k = max((n - self.n_pending) * self.rs.down // self.rs.up + 1, 4096)
                y = self.rs.push(self.x[self.pos:self.pos + k])
                self.pos += k
            else:
                y = self.rs.flush()
                self.flushed = True
            self.pending.append(y)
            self.n_pending += len(y)
        buf = np.concatenate(self.pending)
        out = buf[:n]
        self.pending = [buf[n:]]
        self.n_pending = len(self.pending[0])
        if len(out) < n:
            out = np.concatenate([out, np.zeros(n - len(out))])
        return out.astype(np.float32)

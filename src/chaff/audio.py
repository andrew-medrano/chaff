"""Reading and writing audio in blocks.

A source gives a recording block by block (for the passes over the whole file) and as short
excerpts (for lifting single flyovers out of it). Files libsndfile reads (wav, flac, ogg, mp3,
aiff, ...) are read directly; anything else is decoded by an ffmpeg pipe. Nothing ever holds the
whole recording in memory.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import soundfile as sf

BLOCK = 1 << 18          # frames per block when streaming (5.5 s at 48 kHz)

# output extension -> (libsndfile format, subtype). Float WAV keeps the decoys exactly; flac is
# 24-bit integer (its maximum); ogg and mp3 are lossy.
OUTPUT_FORMATS = {
    ".wav": ("WAV", "FLOAT"), ".w64": ("W64", "FLOAT"), ".rf64": ("RF64", "FLOAT"),
    ".aif": ("AIFF", "FLOAT"), ".aiff": ("AIFF", "FLOAT"), ".caf": ("CAF", "FLOAT"),
    ".flac": ("FLAC", "PCM_24"), ".ogg": ("OGG", "VORBIS"), ".mp3": ("MP3", "MPEG_LAYER_III"),
}
_BYTES = {"PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8, "PCM_U8": 1, "PCM_S8": 1}


class ArraySource:
    """An in-memory recording, (N,) or (N, C)."""

    def __init__(self, x, sr):
        x = np.asarray(x)
        if x.ndim not in (1, 2):
            raise ValueError(f"expected audio of shape (N,) or (N, C), got {x.shape}")
        self.x = x if x.ndim == 2 else x[:, None]
        self.sr, self.frames, self.channels = int(sr), len(self.x), self.x.shape[1]
        self.name = "<array>"

    def blocks(self, n=BLOCK):
        for i in range(0, self.frames, n):
            yield np.asarray(self.x[i:i + n], np.float32)

    def read(self, i0, i1):
        return np.asarray(self.x[max(i0, 0):max(i1, 0)], np.float32)


class SoundFileSource:
    def __init__(self, path):
        info = sf.info(path)
        self.path, self.name = os.fspath(path), os.path.basename(path)
        self.sr, self.frames, self.channels = int(info.samplerate), int(info.frames), int(info.channels)

    def blocks(self, n=BLOCK):
        with sf.SoundFile(self.path) as f:
            while True:
                b = f.read(n, dtype="float32", always_2d=True)
                if not len(b):
                    break
                yield b

    def read(self, i0, i1):
        i0 = max(int(i0), 0); i1 = min(int(i1), self.frames)
        with sf.SoundFile(self.path) as f:
            f.seek(i0)
            return f.read(max(i1 - i0, 0), dtype="float32", always_2d=True)


class FfmpegSource:
    """Anything ffmpeg decodes (m4a, webm, video files, ...), as float32 at the stream's own rate.
    The length is only known after one full pass (frames is None until then)."""

    def __init__(self, path):
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise RuntimeError(f"libsndfile cannot read {path!r} and ffmpeg/ffprobe are not on PATH")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=sample_rate,channels", "-of", "json", os.fspath(path)],
                               capture_output=True, text=True)
        streams = (json.loads(probe.stdout or "{}").get("streams") or []) if probe.returncode == 0 else []
        if not streams:
            raise RuntimeError(f"no audio stream found in {path!r}")
        self.path, self.name = os.fspath(path), os.path.basename(path)
        self.sr, self.channels = int(streams[0]["sample_rate"]), int(streams[0]["channels"])
        self.frames = None

    def _cmd(self, pre=(), post=()):
        return ["ffmpeg", "-v", "error", "-nostdin", *pre, "-i", self.path, *post, "-map", "0:a:0", "-f", "f32le",
                "-acodec", "pcm_f32le", "-ar", str(self.sr), "-ac", str(self.channels), "-"]

    def blocks(self, n=BLOCK):
        p = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        size = n * self.channels * 4
        count = 0
        try:
            while True:
                raw = p.stdout.read(size)
                if not raw:
                    break
                raw = raw[:len(raw) - len(raw) % (4 * self.channels)]
                b = np.frombuffer(raw, "<f4").reshape(-1, self.channels).copy()
                count += len(b)
                yield b
        finally:
            p.stdout.close()
            err = p.stderr.read().decode(errors="replace"); p.stderr.close()
            rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed on {self.path!r}: {err.strip()}")
        self.frames = count

    def read(self, i0, i1):
        i0 = max(int(i0), 0); n = max(int(i1) - i0, 0)
        if self.frames is not None:
            n = max(min(n, self.frames - i0), 0)
        r = subprocess.run(self._cmd(pre=("-ss", f"{i0 / self.sr:.6f}"), post=("-t", f"{n / self.sr:.6f}")),
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {self.path!r}: {r.stderr.decode(errors='replace').strip()}")
        b = np.frombuffer(r.stdout[:len(r.stdout) - len(r.stdout) % (4 * self.channels)], "<f4").reshape(-1, self.channels)
        out = np.zeros((n, self.channels), np.float32)
        out[:min(n, len(b))] = b[:n]
        return out


def open_source(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        return SoundFileSource(path)
    except Exception:            # a format libsndfile does not know (m4a, webm, ...)
        return FfmpegSource(path)


def output_format(path, subtype=None):
    """(format, subtype) for an output path; raises ValueError before any work is done if the
    extension or subtype cannot be written."""
    ext = os.path.splitext(os.fspath(path))[1].lower()
    if ext not in OUTPUT_FORMATS:
        raise ValueError(f"cannot write {ext or 'files without an extension'}: use one of "
                         f"{', '.join(sorted(OUTPUT_FORMATS))} (convert afterwards with ffmpeg if you need another format)")
    fmt, sub = OUTPUT_FORMATS[ext]
    sub = subtype or sub
    if not sf.check_format(fmt, sub):
        raise ValueError(f"libsndfile cannot write {fmt} with subtype {sub}")
    return fmt, sub


class Writer:
    """Block writer. Integer and lossy formats are clipped to [-1, 1] (libsndfile would otherwise
    wrap) and the number of clipped samples is counted. A .wav that would pass the 4 GB limit of
    the WAV header is written as RF64."""

    def __init__(self, path, sr, channels, frames=None, subtype=None):
        fmt, sub = output_format(path, subtype)
        self.rf64 = False
        if fmt == "WAV" and frames is not None and frames * channels * _BYTES.get(sub, 4) > 0xFFFFFFFF - (1 << 20):
            fmt, self.rf64 = "RF64", True
        self.clip = sub not in ("FLOAT", "DOUBLE")
        self.n_clipped, self.peak = 0, 0.0
        self.f = sf.SoundFile(os.fspath(path), "w", int(sr), int(channels), sub, format=fmt)
        self.format, self.subtype = fmt, sub

    def write(self, y):
        if len(y):
            self.peak = max(self.peak, float(np.max(np.abs(y))))
        if self.clip:
            over = np.abs(y) > 1.0
            if over.any():
                self.n_clipped += int(over.sum())
                y = np.clip(y, -1.0, 1.0)
        self.f.write(y)

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

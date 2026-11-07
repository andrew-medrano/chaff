"""Command line: chaff protect IN OUT [options]."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

from . import __version__, protect


def _ffmpeg_read(path):
    """Decode anything ffmpeg understands to float32 at the file's own rate and channel count."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError(f"soundfile cannot read {path!r} and ffmpeg/ffprobe are not on PATH")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", path],
        capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"no audio stream in {path!r}")
    sr = int(streams[0]["sample_rate"]); ch = int(streams[0]["channels"])
    dec = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0", "-f", "f32le",
         "-acodec", "pcm_f32le", "-ar", str(sr), "-ac", str(ch), "-"],
        capture_output=True, check=True)
    x = np.frombuffer(dec.stdout, dtype="<f4").reshape(-1, ch)
    return x.copy(), sr


def read_audio(path):
    """(x, sr) with x float32 shaped (N, C)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:            # format libsndfile does not know (mp3 on old builds, m4a, ...)
        x, sr = _ffmpeg_read(path)
    return x, sr


def write_audio(path, y, sr):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".w64", ".rf64", ".aif", ".aiff", ".caf"):
        subtype = "FLOAT"
    elif ext == ".flac":
        subtype = "PCM_24"
    else:
        subtype = None          # soundfile's default for the format (e.g. VORBIS for .ogg)
    if subtype != "FLOAT":
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:
            print(f"warning: peak {peak:.3f} > 1.0 will clip in {ext or 'output'}", file=sys.stderr)
    sf.write(path, y, sr, subtype=subtype)


def cmd_protect(a):
    x, sr = read_audio(a.input)
    y, decoys, rep = protect(x[:, 0] if x.shape[1] == 1 else x, sr, per_hour=a.per_hour,
                             seed=a.seed, below_db=tuple(a.below),
                             ceiling_below_db=a.ceiling_below, catalog=a.catalog,
                             lead_per_hour=a.lead_per_hour, quiet_per_hour=a.quiet_per_hour)
    write_audio(a.output, y, sr)
    if a.decoys_json:
        with open(a.decoys_json, "w") as fh:
            json.dump({"decoys": [d.to_dict() for d in decoys], "audibility": rep}, fh, indent=1)
    print(f"{a.input} -> {a.output}: {len(x) / sr / 60:.1f} min, {sr} Hz, {x.shape[1]} ch")
    print(f"decoys added: {len(decoys)}")
    print(f"worst band: {rep['worst_band_db']:.1f} dB at {rep['worst_band_hz']:.0f} Hz "
          f"(audible near {rep['threshold_db']:.1f} dB)")
    print(f"margin: {rep['margin_db']:.1f} dB under threshold" if rep['margin_db'] >= 0 else
          f"loudest band {-rep['margin_db']:.1f} dB over threshold (lead decoys are as audible as faint real planes)")
    return 0


def cmd_catalog(a):
    """Build a stack catalog from recordings and their flyover events (e.g. `dopplertrack detect --out`)."""
    from . import noise, stack
    if len(a.pairs) % 2:
        raise SystemExit("give REC EVENTS.json pairs")
    out = []
    for rec, evp in zip(a.pairs[::2], a.pairs[1::2]):
        x, sr = read_audio(rec); xm = x.mean(1); fl = noise.noise_floor(xm, sr)
        evs = json.load(open(evp))[:a.top]
        for i, e in enumerate(evs):
            k = stack.extract(xm, sr, e, fl, name=f"{os.path.basename(rec)}#{i + 1}")
            out.append(k); print(f"{k.name}: {len(k.lines_hz)} line bins, sweep {k.sweep_snr_db:.1f} dB")
    stack.save(out, a.output)
    usable = sum(len(k.lines_hz) >= 3 for k in out)
    print(f"wrote {a.output}: {len(out)} flyovers, {usable} with an engine-line stack usable as decoys")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="chaff", description=__doc__)
    p.add_argument("--version", action="version", version=f"chaff {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("protect", help="add decoy flyovers to a recording")
    pp.add_argument("input", metavar="IN")
    pp.add_argument("output", metavar="OUT")
    pp.add_argument("--per-hour", type=float, default=60.0, help="decoys per hour (default 60)")
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--below", nargs=2, type=float, metavar=("LO", "HI"), default=(4.0, 9.0),
                    help="each decoy peaks this many dB under the audibility threshold (default 4 9)")
    pp.add_argument("--ceiling-below", type=float, default=3.0, metavar="DB",
                    help="limit all decoys together to this many dB under threshold (default 3)")
    pp.add_argument("--decoys-json", metavar="PATH", help="write decoy list and audibility report")
    pp.add_argument("--catalog", metavar="NPZ", help="stack catalog from `chaff catalog` (recommended)")
    pp.add_argument("--lead-per-hour", type=float, default=6.0, help="with --catalog: decoys as loud as faint real planes (default 6)")
    pp.add_argument("--quiet-per-hour", type=float, default=54.0, help="with --catalog: decoys under the hearing threshold (default 54)")
    pp.set_defaults(func=cmd_protect)
    pc = sub.add_parser("catalog", help="build a decoy catalog from real flyovers in recordings")
    pc.add_argument("output", metavar="OUT.npz")
    pc.add_argument("pairs", nargs="+", metavar="REC EVENTS", help="recording and its events JSON (list of t_cpa, f0, v, d)")
    pc.add_argument("--top", type=int, default=10, help="use the first N events of each file (default 10)")
    pc.set_defaults(func=cmd_catalog)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())

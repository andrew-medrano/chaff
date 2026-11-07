"""chaff: add decoy propeller flyovers to a recording so a Doppler-curve geolocation search cannot trust it.

  chaff protect IN OUT [--catalog CATALOG.npz | --find-flyovers | --synthetic] [-j JOBS]
  chaff catalog CATALOG.npz REC [EVENTS.json] [REC [EVENTS.json] ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

from . import __version__
from .pipeline import LEAD, QUIET, SYNTHETIC, ChaffWarning, Synthetic, Tier, build_catalog, protect_file
from .render import default_jobs


def _err(msg):
    print(f"chaff: error: {msg}", file=sys.stderr)
    return 2


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _banner(lines):
    w = max(len(s) for s in lines)
    print("\n" + "!" * (w + 4), file=sys.stderr)
    for s in lines:
        print(f"! {s.ljust(w)} !", file=sys.stderr)
    print("!" * (w + 4) + "\n", file=sys.stderr)


def _wrap(msg, width=88):
    out, line = [], ""
    for word in msg.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}" if line else word
    return out + [line]


def _aud_line(name, a):
    if a is None:
        return f"  {name:<9} no decoys"
    m = a["margin_db"]
    state = f"{m:.1f} dB under the threshold" if m >= 0 else f"{-m:.1f} dB over the threshold (audible)"
    return (f"  {name:<9} loudest band {a['worst_band_db']:+.1f} dB re background at {a['worst_band_hz']:.0f} Hz, "
            f"{state}; {100 * a['share_band_seconds_over']:.1f}% of band-seconds over")


def _tier(a, name, base: Tier):
    g = lambda k: getattr(a, f"{name}_{k}")
    return Tier(per_hour=base.per_hour if g("per_hour") is None else g("per_hour"),
                snr_db=base.snr_db if g("snr") is None else tuple(g("snr")),
                headroom_db=base.headroom_db if g("headroom") is None else g("headroom"),
                limit_db=base.limit_db if g("limit") is None else g("limit"))


def cmd_protect(a):
    synth_given = a.synth_per_hour is not None or a.synth_below is not None
    tier_given = any(getattr(a, f"{t}_{k}") is not None for t in ("lead", "quiet") for k in ("per_hour", "snr", "headroom", "limit"))
    picked = [f for f, on in (("--catalog", a.catalog), ("--find-flyovers", a.find_flyovers),
                              ("--synthetic", a.synthetic)) if on]
    if len(picked) > 1:
        return _err(f"{' and '.join(picked)} exclude each other")
    if synth_given and not a.synthetic:
        return _err("--synth-* options only apply to synthetic decoys (--synthetic)")
    if a.jobs is not None and a.jobs < 1:
        return _err("--jobs must be 1 or more")
    if a.synthetic and tier_given:
        return _err("--lead-* / --quiet-* options only apply to stack decoys, not with --synthetic")
    if a.synthetic and a.jobs is not None:
        return _err("--jobs only applies to stack decoys; synthetic decoys are rendered in one process")
    if a.workers is not None and not a.find_flyovers:
        return _err("--workers only applies to flyover detection (--find-flyovers)")
    so = Synthetic(per_hour=SYNTHETIC.per_hour if a.synth_per_hour is None else a.synth_per_hour,
                   below_db=SYNTHETIC.below_db if a.synth_below is None else tuple(a.synth_below),
                   limit_db=SYNTHETIC.limit_db)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ChaffWarning)       # printed as they happen and again at the end
        try:
            decoys, rep = protect_file(a.input, a.output, a.catalog, find_flyovers=a.find_flyovers,
                                       synthetic=a.synthetic, seed=a.seed, lead=_tier(a, "lead", LEAD),
                                       quiet=_tier(a, "quiet", QUIET), synth_opts=so, subtype=a.subtype,
                                       jobs=a.jobs, workers=a.workers, log=None if a.silent else _log)
        except (ValueError, OSError, RuntimeError, ImportError) as e:
            return _err(e)
    if a.report:
        with open(a.report, "w") as fh:
            json.dump({"decoys": [d.to_dict() for d in decoys], "report": rep}, fh, indent=1)
    out = rep["output"]
    print(f"{a.input} -> {out['path']}: {rep['duration_s'] / 60:.1f} min, {rep['sr']} Hz, {rep['channels']} ch, "
          f"{out['format']} {out['subtype']}")
    print(f"decoys: {rep['mode']}, source: {rep['catalog']['source']}; rendered in "
          f"{rep['render_processes']} process{'es' if rep['render_processes'] > 1 else ''}")
    for name, t in rep["tiers"].items():
        print(f"  {name:<9} {t['n']} decoys ({t['per_hour']:g} per hour)")
    print(f"audibility (a tone becomes audible near {rep['audibility']['threshold_db']:.0f} dB re background):")
    for name in ("quiet", "lead", "synthetic", "combined"):
        if name in rep["audibility"]:
            print(_aud_line(name, rep["audibility"][name]))
    if rep["warnings"]:
        _banner(sum((_wrap("WARNING: " + w) for w in rep["warnings"]), []))
    return 0


def _pairs(items):
    """REC [EVENTS.json] REC ... -> [(rec, events path or None)]"""
    out = []
    for it in items:
        if it.lower().endswith(".json"):
            if not out or out[-1][1] is not None:
                raise ValueError(f"{it}: an events file must follow the recording it belongs to")
            out[-1] = (out[-1][0], it)
        else:
            out.append((it, None))
    return out


def _events(path):
    with open(path) as fh:
        ev = json.load(fh)
    if isinstance(ev, dict):
        ev = ev.get("events", [])
    if not isinstance(ev, list):
        raise ValueError(f"{path}: expected a list of events (t_cpa, f0, v, d)")
    return ev


def cmd_catalog(a):
    try:
        pairs = [(rec, None if evp is None else _events(evp)) for rec, evp in _pairs(a.inputs)]
        if not pairs:
            raise ValueError("give at least one recording")
        stacks, notes = build_catalog(pairs, top=a.top, workers=a.workers, log=_log)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as e:
        return _err(e)
    from .stack import save
    usable = [k for k in stacks if k.usable]
    if not usable:
        return _err(f"no usable flyover: {len(stacks)} extracted, none with 3 or more engine lines; "
                    "nothing written. Use recordings with louder propeller planes.")
    path = save(stacks, a.output)
    print(f"wrote {path}: {len(stacks)} flyovers, {len(usable)} usable as decoy sources "
          f"(isolated SNR {min(k.iso_snr_db for k in usable):.1f}-{max(k.iso_snr_db for k in usable):.1f} dB; "
          f"lead tier needs {LEAD.snr_db[0] + LEAD.headroom_db:.0f} dB, quiet {QUIET.snr_db[0] + QUIET.headroom_db:.0f} dB)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="chaff", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"chaff {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("protect", help="add decoy flyovers to a recording",
                        description="Add decoy flyovers to a recording. By default the decoys are made from the "
                                    "catalog that ships with chaff: real flyovers from the Hamilton Naturalists' "
                                    "Club's McMaster Forest recordings (via the Internet Archive).")
    pp.add_argument("input", metavar="IN", help="any file libsndfile or ffmpeg can read")
    pp.add_argument("output", metavar="OUT", help=".wav (float), .flac, .ogg, .mp3, .aiff, .caf, .w64, .rf64")
    src = pp.add_argument_group("decoy source (default: the bundled McMaster Forest catalog)")
    src.add_argument("--catalog", metavar="NPZ", help="another stack catalog, e.g. from `chaff catalog`")
    src.add_argument("--find-flyovers", action="store_true",
                     help="make the decoys from this recording's own flyovers, found with dopplertrack "
                          "(needs chaff[detect]); falls back to the bundled catalog if none is usable")
    src.add_argument("--synthetic", action="store_true", help="use synthetic decoys (easier to tell apart)")
    pp.add_argument("-j", "--jobs", type=int, metavar="N",
                    help=f"processes that render the decoys (default: all cores but one, at most 8 and at most "
                         f"what fits in half the RAM at 0.65 GB each; here {default_jobs()}; 1 = one process); "
                         "the output does not depend on it. Not with --synthetic")
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--report", metavar="JSON", help="write the decoy list and the audibility report")
    pp.add_argument("--subtype", help="output sample format, e.g. PCM_16, PCM_24, FLOAT (default: FLOAT for "
                                      "wav-like files, PCM_24 for flac)")
    pp.add_argument("--workers", type=int, help="processes for flyover detection, with --find-flyovers only "
                                                 "(default: all cores)")
    pp.add_argument("--silent", action="store_true", help="no progress messages")
    g = pp.add_argument_group("stack decoy tiers (defaults: the tested recipe)")
    for name, t in (("lead", LEAD), ("quiet", QUIET)):
        g.add_argument(f"--{name}-per-hour", type=float, metavar="N", help=f"{name} decoys per hour (default {t.per_hour:g})")
        g.add_argument(f"--{name}-snr", type=float, nargs=2, metavar=("LO", "HI"),
                       help=f"per-bin SNR range of the fundamental through the sweep, dB (default {t.snr_db[0]:g} {t.snr_db[1]:g})")
        g.add_argument(f"--{name}-headroom", type=float, metavar="DB",
                       help=f"source stack must be this much louder than the target (default {t.headroom_db:g})")
        g.add_argument(f"--{name}-limit", type=float, metavar="DB",
                       help=f"limiter ceiling per auditory band re background, dB (default {t.limit_db:g})")
    g = pp.add_argument_group("synthetic decoys (with --synthetic)")
    g.add_argument("--synth-per-hour", type=float, metavar="N", help=f"default {SYNTHETIC.per_hour:g}")
    g.add_argument("--synth-below", type=float, nargs=2, metavar=("LO", "HI"),
                   help=f"each peaks this many dB under the hearing threshold (default {SYNTHETIC.below_db[0]:g} {SYNTHETIC.below_db[1]:g})")
    pp.set_defaults(func=cmd_protect)

    pc = sub.add_parser("catalog", help="build a decoy catalog from real flyovers in recordings",
                        description="Build a stack catalog. Each recording may be followed by an events JSON "
                                    "(a list of t_cpa, f0, v, d, e.g. from `dopplertrack detect --out`); without "
                                    "one its flyovers are found with dopplertrack's detector.")
    pc.add_argument("output", metavar="OUT.npz")
    pc.add_argument("inputs", nargs="+", metavar="REC [EVENTS.json]")
    pc.add_argument("--top", type=int, default=10, help="use the first N events of each recording (default 10)")
    pc.add_argument("--workers", type=int, help="processes for flyover detection (default: all cores)")
    pc.set_defaults(func=cmd_catalog)
    a = p.parse_args(argv)
    return a.func(a)


def entry():
    """Console entry point (`chaff`, `python -m chaff`).

    On macOS the system allocator keeps up to ~1.5 GB of freed large blocks cached for reuse, and
    a decoy run frees many of them, so the process looks (and is) far bigger than what it holds.
    MallocLargeCache=0 turns that cache off; it must be set before the process starts, so the
    command re-executes itself once with it. Detection worker processes get the default back:
    they hold little, and without the cache they spend much of their time in page faults. Decoy
    render workers free large blocks like the main process, so they run without it too."""
    if os.environ.pop("_CHAFF_REEXEC", None):
        os.environ.pop("MallocLargeCache", None)      # this process is set; detection workers are not
    elif sys.platform == "darwin" and "MallocLargeCache" not in os.environ:
        # _CHAFF_MALLOC_LARGE_CACHE: render workers get this process's setting (render._worker_env)
        env = dict(os.environ, MallocLargeCache="0", _CHAFF_REEXEC="1", _CHAFF_MALLOC_LARGE_CACHE="0")
        try:
            os.execve(sys.executable, [sys.executable, "-m", "chaff", *sys.argv[1:]], env)
        except OSError:
            pass                 # carry on with the cache
    sys.exit(main())


if __name__ == "__main__":
    entry()

"""Rebuild the bundled catalog, src/chaff/data/mcmaster.npz.

    python scripts/build_bundled_catalog.py MCMASTER.wav [--events scripts/mcmaster_events.json]
                                            [--out src/chaff/data/mcmaster.npz]

MCMASTER.wav: the Hamilton Naturalists' Club's McMaster Forest recording, Internet Archive item
HNCSW720260527, as one mono 4 kHz WAV: the 12 hourly MP3 files (08:00 to 20:00 local time)
decoded and concatenated in order (the same file docs/make_figures.py reads; in the research
workspace, heldout/mcmaster.wav). It is not in this repository.

scripts/mcmaster_events.json: the top 10 flyovers dopplertrack's detector found in that recording
(t_cpa, f0, v, d, score), saved from the research run that made the development catalog. With
them this script does what `chaff catalog OUT MCMASTER.wav mcmaster_events.json` does, with this
checkout's chaff: background, de-Doppler, engine lines, click repair. It then keeps the stacks
with at least stack.MIN_LINES engine lines, names them mcmaster#N (N: rank in the event list),
adds the source and credit metadata, writes the file, reads it back to check it, and prints the
table for src/chaff/data/README.md.

Data versions (info["data_version"]): 1 was chaff 0.3.0's file; 2 (chaff 0.3.1) is the same 8
flyovers with broadband clicks repaired in the line bins (see chaff/stack.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chaff                                              # noqa: E402
from chaff import stack                                   # noqa: E402
from chaff.pipeline import build_catalog                  # noqa: E402

DATA_VERSION = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", help="McMaster day file, mono 4 kHz WAV")
    ap.add_argument("--events", default=os.path.join(ROOT, "scripts", "mcmaster_events.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "src", "chaff", "data", stack.BUNDLED))
    a = ap.parse_args()
    with open(a.events) as fh:
        events = json.load(fh)
    ks, notes = build_catalog([(a.recording, events)], top=10, log=print)
    if notes:
        raise SystemExit("events skipped: " + "; ".join(notes))
    for i, k in enumerate(ks):
        k.name = f"mcmaster#{i + 1}"
    keep = [k for k in ks if k.usable]
    info = dict(
        name="McMaster Forest",
        content=(f"isolated engine-line stacks of {len(keep)} propeller-plane flyovers: de-Dopplered, "
                 f"only the steady engine lines kept, broadband clicks repaired in them, resampled to "
                 f"{stack.SYN_SR} Hz (float16)"),
        source=("Hamilton Naturalists' Club soundscape recorder HNCSW7, McMaster Forest Nature Preserve, "
                "Ancaster, Ontario, 2026-05-27 (12 hourly files, 08:00-20:00 local time); "
                "Internet Archive item HNCSW720260527"),
        source_url="https://archive.org/details/HNCSW720260527",
        credit="Hamilton Naturalists' Club, via the Internet Archive",
        licence="none stated in the Internet Archive item metadata",
        selection=(f"`chaff catalog` (chaff {chaff.__version__}) on the top 10 dopplertrack detections of the "
                   f"recording (scripts/mcmaster_events.json); the {len(keep)} with {stack.MIN_LINES} or more "
                   f"distinct engine lines are kept"),
        data_version=DATA_VERSION,
        built_with=f"chaff {chaff.__version__}, scripts/build_bundled_catalog.py",
        stacks=[dict(name=k.name, f0_hz=round(k.f0, 1), engine_lines=k.n_lines, iso_snr_db=round(k.iso_snr_db, 1),
                     clicks_repaired_s=round(k.repaired_s, 1)) for k in keep],
    )
    p = stack.save(keep, a.out, info=info)
    back = stack.load(p)
    assert stack.load_info(p) == json.loads(json.dumps(info))
    for x, y in zip(keep, back):
        assert x.name == y.name and np.array_equal(x.lines, y.lines) and np.array_equal(x.bb, y.bb)
        assert np.array_equal(x.s.astype(np.float16).astype(np.float32), y.s)
        assert (x.f0, x.v, x.d, x.tau_cpa, x.iso_snr_db, x.floor_ref, x.repaired_s) == \
               (y.f0, y.v, y.d, y.tau_cpa, y.iso_snr_db, y.floor_ref, y.repaired_s)
    print(f"\nwrote {p}: {os.path.getsize(p) / 1e6:.1f} MB, {len(keep)} of {len(ks)} stacks, data version {DATA_VERSION}\n")
    print("| Stack | Blade rate (Hz) | Engine lines | Isolated SNR (dB) | Touched by click repair (s) |")
    print("|---|---|---|---|---|")
    for k in keep:
        print(f"| {k.name} | {k.f0:.1f} | {k.n_lines} | {k.iso_snr_db:.1f} | {k.repaired_s:.0f} |")


if __name__ == "__main__":
    main()

"""Command line: `chaff protect`, `chaff catalog` and `python -m chaff`, run as real processes."""
import dataclasses
import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

import chaff
from chaff import cli, stack

from conftest import EVENTS, pink, run_cli


def test_version_and_help():
    r = run_cli("--version")
    assert r.returncode == 0 and r.stdout.strip() == f"chaff {chaff.__version__}"
    r = run_cli("protect", "--help")
    assert r.returncode == 0 and "usage: chaff protect" in r.stdout and "--quiet-per-hour" in r.stdout
    r = run_cli()
    assert r.returncode == 2 and "required" in r.stderr


def test_catalog_then_protect(tmp_path, planes):
    x, sr = planes
    rec, ev, cat = tmp_path / "rec.wav", tmp_path / "rec.json", tmp_path / "cat.npz"
    sf.write(rec, x, sr, subtype="FLOAT")
    ev.write_text(json.dumps({"events": EVENTS + [dict(t_cpa=9999.0, f0=90.0, v=60.0, d=900.0)]}))
    r = run_cli("catalog", cat, rec, ev)
    assert r.returncode == 0, r.stderr
    assert "outside the recording" in r.stderr and "4 engine lines" in r.stderr
    assert "2 flyovers, 2 usable" in r.stdout
    assert [k.n_lines for k in stack.load(cat)] == [4, 4]

    src, dst, rep = tmp_path / "in.wav", tmp_path / "out.flac", tmp_path / "rep.json"
    xi = pink(sr * 400, sr, ch=2, seed=9)
    sf.write(src, xi, sr, subtype="PCM_16")
    r = run_cli("protect", src, dst, "--catalog", cat, "--seed", "3", "--report", rep)
    assert r.returncode == 0, r.stderr
    assert "pass 1/2" in r.stderr and "pass 2/2" in r.stderr                       # progress on stderr
    for word in ("decoys: stack", "lead", "quiet", "combined", "under the threshold"):
        assert word in r.stdout
    y, sr2 = sf.read(dst, dtype="float32", always_2d=True)
    assert sr2 == sr and y.shape == xi.shape and sf.info(str(dst)).subtype == "PCM_24"
    meta = json.loads(rep.read_text())
    assert meta["report"]["mode"] == "stack" and meta["decoys"]
    assert {d["tier"] for d in meta["decoys"]} == {"lead", "quiet"}
    assert meta["report"]["audibility"]["quiet"]["share_band_seconds_over"] == 0.0

    # tier flags reach the run; --silent drops the progress messages
    r = run_cli("protect", src, tmp_path / "o2.wav", "--catalog", cat, "--silent", "--lead-per-hour", "0",
                "--quiet-per-hour", "30", "--report", rep)
    assert r.returncode == 0 and r.stderr == "", r.stderr
    meta = json.loads(rep.read_text())
    assert meta["report"]["tiers"]["lead"]["n"] == 0 and meta["report"]["tiers"]["quiet"]["per_hour"] == 30


def test_synthetic_via_module(tmp_path):
    sr = 16000
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, pink(sr * 180, sr, ch=2, seed=1), sr)
    r = run_cli("protect", src, dst, "--synthetic", "--synth-per-hour", "40")
    assert r.returncode == 0, r.stderr
    assert "decoys: synthetic" in r.stdout and "WARNING" not in r.stderr
    assert sf.info(str(dst)).frames == sr * 180 and sf.info(str(dst)).channels == 2


def test_unusable_catalog_exits_nonzero(tmp_path, stacks):
    sr = 8000
    src = tmp_path / "in.wav"
    sf.write(src, pink(sr * 120, sr), sr)
    weak = stack.save([dataclasses.replace(k, lines=k.lines[:2]) for k in stacks], tmp_path / "weak.npz")
    r = run_cli("protect", src, tmp_path / "o.wav", "--catalog", weak)
    assert r.returncode == 2 and "unusable" in r.stderr and not (tmp_path / "o.wav").exists()
    np.save(tmp_path / "junk.npy", np.zeros(3))
    r = run_cli("protect", src, tmp_path / "o.wav", "--catalog", tmp_path / "junk.npy")
    assert r.returncode == 2 and "not a chaff catalog" in r.stderr


def test_errors_exit_nonzero(tmp_path, capsys):
    sr = 8000
    src = tmp_path / "in.wav"
    sf.write(src, pink(sr * 120, sr), sr)
    assert cli.main(["protect", str(src), str(tmp_path / "o.m4a"), "--synthetic"]) == 2
    assert cli.main(["protect", str(tmp_path / "missing.wav"), str(tmp_path / "o.wav"), "--synthetic"]) == 2
    assert cli.main(["protect", str(src), str(tmp_path / "o.wav"), "--synthetic", "--lead-per-hour", "3"]) == 2
    assert cli.main(["protect", str(src), str(tmp_path / "o.wav"), "--catalog", "c.npz", "--synth-per-hour", "3"]) == 2
    assert cli.main(["protect", str(src), str(tmp_path / "o.wav"), "--catalog", "c.npz", "--synthetic"]) == 2
    short = tmp_path / "short.wav"
    sf.write(short, pink(sr * 30, sr), sr)
    assert cli.main(["protect", str(short), str(tmp_path / "o.wav"), "--synthetic"]) == 2
    assert "at least 60 s" in capsys.readouterr().err
    empty = tmp_path / "noise.wav"
    sf.write(empty, pink(sr * 400, sr, seed=3), sr)
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps([dict(t_cpa=200.0, f0=90.0, v=60.0, d=900.0)]))
    assert cli.main(["catalog", str(tmp_path / "c.npz"), str(empty), str(ev)]) == 2     # no stack in noise
    assert not (tmp_path / "c.npz").exists()
    assert cli.main(["catalog", str(tmp_path / "c.npz"), str(ev)]) == 2               # events with no recording
    ev.write_text(json.dumps({"t_cpa": 1}))
    assert cli.main(["catalog", str(tmp_path / "c.npz"), str(empty), str(ev)]) == 2    # not a list
    assert "error" in capsys.readouterr().err


def test_fallback_warning_is_boxed(tmp_path, monkeypatch, capsys):
    from chaff import detect
    monkeypatch.setattr(detect, "available", lambda: False)
    sr = 8000
    src = tmp_path / "in.wav"
    sf.write(src, pink(sr * 120, sr), sr)
    assert cli.main(["protect", str(src), str(tmp_path / "o.wav")]) == 0
    err = capsys.readouterr().err
    assert "!!!!" in err and "WARNING: no catalog given" in err


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
def test_ffmpeg_input(tmp_path):
    sr = 8000
    x = pink(sr * 120, sr, seed=5)
    wav, m4a, dst = tmp_path / "a.wav", tmp_path / "a.m4a", tmp_path / "out.flac"
    sf.write(wav, x, sr)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(wav), "-c:a", "aac", str(m4a)], check=True)
    assert cli.main(["protect", str(m4a), str(dst), "--synthetic"]) == 0
    y, sr2 = sf.read(dst, always_2d=True)
    assert sr2 == sr and abs(len(y) - len(x)) < sr          # encoder padding only
    from chaff.audio import open_source
    s = open_source(str(m4a))
    seg = s.read(sr * 30, sr * 40)
    assert seg.shape == (sr * 10, 1)

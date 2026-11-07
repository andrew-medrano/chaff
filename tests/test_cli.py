"""Command line: protect, catalog, python -m chaff."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from chaff import cli

from conftest import EVENTS, pink


def test_version_and_module_entry():
    r = subprocess.run([sys.executable, "-m", "chaff", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("chaff 0.2")


def test_catalog_then_protect(tmp_path, planes, capsys):
    x, sr = planes
    rec, ev, cat = tmp_path / "rec.wav", tmp_path / "rec.json", tmp_path / "cat.npz"
    sf.write(rec, x, sr, subtype="FLOAT")
    ev.write_text(json.dumps(EVENTS + [dict(t_cpa=9999.0, f0=90.0, v=60.0, d=900.0)]))
    assert cli.main(["catalog", str(cat), str(rec), str(ev)]) == 0
    err = capsys.readouterr().err
    assert "outside the recording" in err and "4 engine lines" in err
    src, dst, rep = tmp_path / "in.wav", tmp_path / "out.wav", tmp_path / "rep.json"
    xi = pink(sr * 600, sr, ch=2, seed=9)
    sf.write(src, xi, sr, subtype="PCM_16")
    assert cli.main(["protect", str(src), str(dst), "--catalog", str(cat), "--seed", "3", "--report", str(rep)]) == 0
    out = capsys.readouterr().out
    assert "quiet" in out and "lead" in out and "combined" in out
    y, sr2 = sf.read(dst, dtype="float32", always_2d=True)
    assert sr2 == sr and y.shape == xi.shape and sf.info(str(dst)).subtype == "FLOAT"
    meta = json.loads(rep.read_text())
    assert meta["report"]["mode"] == "stack" and meta["decoys"]
    assert {d["tier"] for d in meta["decoys"]} <= {"lead", "quiet"}


def test_errors_exit_nonzero(tmp_path, capsys):
    sr = 8000
    src = tmp_path / "in.wav"
    sf.write(src, pink(sr * 120, sr), sr)
    assert cli.main(["protect", str(src), str(tmp_path / "o.m4a"), "--synthetic"]) == 2
    assert cli.main(["protect", str(tmp_path / "missing.wav"), str(tmp_path / "o.wav"), "--synthetic"]) == 2
    assert cli.main(["protect", str(src), str(tmp_path / "o.wav"), "--synthetic", "--lead-per-hour", "3"]) == 2
    assert cli.main(["protect", str(src), str(tmp_path / "o.wav"), "--catalog", "c.npz", "--synth-per-hour", "3"]) == 2
    empty = tmp_path / "noise.wav"
    sf.write(empty, pink(sr * 400, sr, seed=3), sr)
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps([dict(t_cpa=200.0, f0=90.0, v=60.0, d=900.0)]))
    assert cli.main(["catalog", str(tmp_path / "c.npz"), str(empty), str(ev)]) == 2     # no stack in noise
    assert not (tmp_path / "c.npz").exists()
    assert "error" in capsys.readouterr().err


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

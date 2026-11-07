"""Rendering decoys in worker processes gives the same output as one process."""
import dataclasses
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import chaff
from chaff import render

from conftest import pink, run_py

# enough decoys in each tier that a pool is used (render.MIN_PER_JOB per process)
MANY = dict(lead=dataclasses.replace(chaff.LEAD, per_hour=40.0), quiet=dataclasses.replace(chaff.QUIET, per_hour=90.0))


@pytest.fixture(scope="module")
def rec():
    sr = 8000
    return pink(sr * 600, sr, ch=2, seed=21), sr


def test_parallel_equals_serial_bundled_catalog(rec):
    x, sr = rec
    y1, d1, r1 = chaff.protect(x, sr, seed=7, jobs=1, **MANY)
    y3, d3, r3 = chaff.protect(x, sr, seed=7, jobs=3, **MANY)
    assert r1["render_processes"] == 1 and r3["render_processes"] == 3
    assert [d.to_dict() for d in d1] == [d.to_dict() for d in d3]
    assert np.array_equal(y1, y3)                     # bit-identical
    assert r1["audibility"] == r3["audibility"]


def test_parallel_equals_serial_given_stacks(rec, stacks):
    # stacks given as objects (not a path) are sent to the workers
    x, sr = rec
    y1, _, _ = chaff.protect(x, sr, catalog=stacks, seed=8, jobs=1, **MANY)
    y2, _, r2 = chaff.protect(x, sr, catalog=stacks, seed=8, jobs=2, **MANY)
    assert r2["render_processes"] == 2 and np.array_equal(y1, y2)


def test_parallel_file_equals_serial(tmp_path, rec):
    x, sr = rec
    src = tmp_path / "in.wav"
    sf.write(src, x, sr, subtype="FLOAT")
    out = []
    for j in (1, 4):
        dst = tmp_path / f"out{j}.wav"
        decoys, rep = chaff.protect_file(src, dst, seed=9, jobs=j, **MANY)
        out.append(sf.read(dst, dtype="float32")[0])
    assert np.array_equal(*out)


def test_few_decoys_render_in_one_process(rec):
    x, sr = rec
    _, decoys, rep = chaff.protect(x[:sr * 180], sr, seed=1, jobs=8)     # 3 quiet decoys, no lead
    assert rep["render_processes"] == 1 and decoys


def test_jobs_must_be_positive(rec):
    x, sr = rec
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ValueError, match="jobs"):
            chaff.protect(x, sr, jobs=bad)
    assert chaff.default_jobs() >= 1


def test_decoys_do_not_depend_on_order():
    # each decoy has its own seed: rendering it alone, or after others, gives the same samples
    from chaff import stack
    from chaff.noise import noise_floor
    sr = 8000
    x = pink(sr * 600, sr, seed=3)
    floor = noise_floor(x, sr)
    ks = stack.load(chaff.bundled_catalog())
    dc = stack.sample(ks, 600, 60.0, (4.0, 10.0), 6.0, seed=2, edge_s=150)
    n_syn = chaff.synth.track_len(len(x), sr)
    a = [stack.render_one(d, ks[d.src], n_syn, floor) for d in dc]
    b = [stack.render_one(d, ks[d.src], n_syn, floor) for d in dc[::-1]][::-1]
    for p, q in zip(a, b):
        assert (p is None and q is None) or (p[0] == q[0] and np.array_equal(p[1], q[1]))


SCRIPT = """
import sys
import numpy as np
import chaff
sys.path.insert(0, {tests!r})
from conftest import pink
from test_parallel import MANY
x = pink(8000 * 600, 8000, seed=21)
y, _, rep = chaff.protect(x, 8000, seed=7, jobs=2, **MANY)
np.save(sys.argv[1], y)
print("processes", rep["render_processes"])
"""


def test_script_without_main_guard_still_works(tmp_path, rec):
    # where workers start by importing the main script (macOS, Windows), they stop at once and the
    # run continues in one process with a warning; where they fork, they just work
    script = tmp_path / "noguard.py"
    script.write_text(textwrap.dedent(SCRIPT.format(tests=str(Path(__file__).resolve().parent))))
    r = run_py(script, tmp_path / "y.npy", timeout=600)
    assert r.returncode == 0, r.stderr
    x = pink(8000 * 600, 8000, seed=21)
    y1, _, _ = chaff.protect(x, 8000, seed=7, jobs=1, **MANY)
    assert np.array_equal(np.load(tmp_path / "y.npy"), y1)
    if "processes 1" in r.stdout:
        assert "guard" in r.stderr


def test_worker_env_follows_the_main_process(monkeypatch):
    import os
    import sys
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("MallocLargeCache", raising=False)
    monkeypatch.setenv("_CHAFF_MALLOC_LARGE_CACHE", "0")
    with render._worker_env():
        assert os.environ["MallocLargeCache"] == "0"
    assert "MallocLargeCache" not in os.environ
    monkeypatch.setenv("MallocLargeCache", "1")                   # the user's own setting wins
    with render._worker_env():
        assert os.environ["MallocLargeCache"] == "1"
    # called from Python (no `chaff` command): render workers still run without the cache
    monkeypatch.delenv("MallocLargeCache")
    monkeypatch.delenv("_CHAFF_MALLOC_LARGE_CACHE")
    with render._worker_env():
        assert os.environ["MallocLargeCache"] == "0"
    assert "MallocLargeCache" not in os.environ
    monkeypatch.setattr(sys, "platform", "linux")
    with render._worker_env():
        assert "MallocLargeCache" not in os.environ


def test_default_jobs_is_capped(monkeypatch):
    monkeypatch.setattr(render, "_ram_bytes", lambda: None)
    for cpus, want in ((1, 1), (2, 1), (4, 3), (12, 8), (64, 8)):
        monkeypatch.setattr(render, "_cpus", lambda: cpus)
        assert render.default_jobs() == want
    monkeypatch.setattr(render, "_cpus", lambda: 64)
    monkeypatch.setattr(render, "_ram_bytes", lambda: 4e9)       # half of 4 GB at 0.65 GB a worker
    assert render.default_jobs() == 3
    monkeypatch.setattr(render, "_ram_bytes", lambda: 1e9)
    assert render.default_jobs() == 1
    assert render.check_jobs(None) == 1 and render.check_jobs(20) == 20      # an explicit number is kept


class _Pool:
    """Stands in for the process pool: renders the first `ok` decoys in this process, then breaks."""
    def __init__(self, stacks, floor, ok):
        self.stacks, self.floor, self.ok, self.n = stacks, floor, ok, 0

    def submit(self, fn, dc, n_syn, max_span_s):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        from chaff import stack
        f = Future()
        if self.n < self.ok:
            f.set_result(stack.render_one(dc, self.stacks[dc.src], n_syn, self.floor, max_span_s))
        else:
            f.set_exception(BrokenProcessPool("a child process terminated abruptly"))
        self.n += 1
        return f

    def shutdown(self, cancel_futures=False):
        pass


@pytest.mark.parametrize("ok", [0, 5])
def test_broken_pool_blames_the_guard_only_at_start(monkeypatch, ok):
    # a pool that breaks before any decoy came back looks like a script without a main guard; one
    # that breaks later (a worker killed for memory, say) gets a different message. Either way the
    # tier is rendered again in one process, with the same result.
    from chaff import stack
    from chaff.noise import noise_floor
    sr = 8000
    x = pink(sr * 600, sr, seed=3)
    floor = noise_floor(x, sr)
    ks = stack.load(chaff.bundled_catalog())
    dc = stack.sample(ks, 600, 60.0, (4.0, 10.0), 6.0, seed=2, edge_s=150)
    want = stack.render(dc, ks, len(x), sr, floor)
    rr = render.Renderer(ks, floor, jobs=2)

    def pool(n):
        rr.pool, rr.n_pool = _Pool(ks, floor, ok), n
        return rr.pool
    monkeypatch.setattr(rr, "_pool", pool)
    with pytest.warns(RuntimeWarning) as w:
        y, skipped = rr.render(dc, len(x), sr)
    msg = str(w[0].message)
    assert ("guard" in msg) == (ok == 0) and ("died while rendering" in msg) == (ok > 0)
    assert rr.used == 1 and rr.jobs == 1 and np.array_equal(y, want[0]) and skipped == want[1]

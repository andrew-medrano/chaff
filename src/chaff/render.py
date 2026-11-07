"""Rendering stack decoys on several CPU cores.

Each decoy is rendered on its own (stack.render_one) from its catalog stack, the recording's
background and its own seed, so it does not matter which process renders it or when. Worker
processes load the catalog once, when they start; each task returns one decoy's (first sample,
signal) at 2 kHz; the main process adds them into the track in decoy order. The track is
therefore bit-identical to the one-process path for the same seed.

At most `2 * jobs` rendered decoys wait in the main process at a time (each is a few MB), so
memory does not grow with the number of decoys.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager

import numpy as np

from . import stack
from .synth import track_len

MIN_PER_JOB = 4          # fewer decoys than this per process: starting the processes costs more than it saves


def default_jobs() -> int:
    """All cores but one."""
    return max((os.cpu_count() or 1) - 1, 1)


def check_jobs(jobs):
    if jobs is None:
        return default_jobs()
    if isinstance(jobs, bool) or not isinstance(jobs, (int, np.integer)) or jobs < 1:
        raise ValueError(f"jobs must be a whole number of processes, 1 or more (got {jobs!r})")
    return int(jobs)


def starting_worker() -> bool:
    """True while a spawned worker process imports the caller's main module (a script without an
    `if __name__ == "__main__":` guard calls chaff again from there)."""
    return bool(getattr(mp.current_process(), "_inheriting", False))


_W = {}                  # worker state: stacks, floor


def _init(state):
    """Worker start: read the state file the main process wrote (render.Renderer._state)."""
    with open(state, "rb") as fh:
        catalog, floor = pickle.load(fh)
    _W["stacks"] = stack.load(catalog) if isinstance(catalog, str) else catalog
    _W["floor"] = floor


def _one(dc, n_syn, max_span_s):
    return stack.render_one(dc, _W["stacks"][dc.src], n_syn, _W["floor"], max_span_s)


@contextmanager
def _worker_env():
    """On macOS the `chaff` command runs with the allocator's large-block cache off
    (MallocLargeCache=0, see cli.entry) but takes the variable out of the environment so that
    detection workers get the cache back. Render workers free large blocks all the time, like
    the main process, so they get the main process's setting: with the cache, each of them
    would keep up to ~1.5 GB it no longer uses."""
    want = os.environ.get("_CHAFF_MALLOC_LARGE_CACHE")
    if sys.platform != "darwin" or want is None or "MallocLargeCache" in os.environ:
        yield
        return
    os.environ["MallocLargeCache"] = want
    try:
        yield
    finally:
        os.environ.pop("MallocLargeCache", None)


class Renderer:
    """Renders lists of stack decoys into 2 kHz tracks, with a pool of `jobs` processes that is
    started on first use and kept for the next list. Use as a context manager.

    catalog: the path the stacks were loaded from (workers load it themselves), or None to send
    the stacks to each worker."""

    def __init__(self, stacks, floor, jobs=1, catalog=None, log=None):
        self.stacks, self.floor, self.jobs = stacks, floor, check_jobs(jobs)
        self.catalog = os.fspath(catalog) if catalog is not None else None
        self.log = log or (lambda msg: None)
        self.pool = None
        self.state = None
        self.env = ExitStack()
        self.used = 1            # processes the last render() used

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)
            self.pool = None
        if self.state is not None:
            try:
                os.remove(self.state)
            except OSError:
                pass
            self.state = None
        self.env.close()

    def _state(self):
        """The workers' start-up state (catalog path or stacks, and the background) in a private
        temporary file, so that what each new process is sent stays small: on Python 3.11 (at
        least) a large start-up message to a worker that dies at once (a script without a main
        guard) blocks the main process for good."""
        fd, self.state = tempfile.mkstemp(prefix="chaff-workers-", suffix=".pkl")
        with os.fdopen(fd, "wb") as fh:
            pickle.dump((self.catalog if self.catalog is not None else self.stacks, self.floor), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return self.state

    def _pool(self, n):
        if self.pool is None:
            # the environment stays set while the pool lives: it may start processes on demand
            self.env.enter_context(_worker_env())
            self.pool = ProcessPoolExecutor(n, initializer=_init, initargs=(self._state(),))
            self.n_pool = n
        return self.pool

    def render(self, decoys, n_samples, sr, max_span_s=420.0, progress=None):
        """Like stack.render: (float32 track at SYN_SR, decoys that did not fit)."""
        n = min(self.jobs, len(decoys) // MIN_PER_JOB)
        if self.pool is not None:
            n = self.n_pool
        if n <= 1:
            self.used = 1
            return stack.render(decoys, self.stacks, n_samples, sr, self.floor, max_span_s, progress)
        try:
            out = self._render(self._pool(n), decoys, n_samples, sr, max_span_s, progress)
            self.used = self.n_pool
            return out
        except (BrokenProcessPool, BrokenPipeError, EOFError):
            # worker processes died or could not start: typically a script without an
            # `if __name__ == "__main__":` guard (macOS and Windows start workers by importing it)
            self.close()
            self.jobs = 1
            warnings.warn("rendering decoys in worker processes failed (does your script lack an "
                          "`if __name__ == '__main__':` guard?); continuing in one process", RuntimeWarning)
            self.log("  worker processes failed; rendering in one process")
            self.used = 1
            return stack.render(decoys, self.stacks, n_samples, sr, self.floor, max_span_s, progress)

    def _render(self, pool, decoys, n_samples, sr, max_span_s, progress):
        n_syn = track_len(n_samples, sr)
        y = np.zeros(n_syn, np.float32)
        skipped = []
        ahead = 2 * self.n_pool
        todo = iter(decoys)
        pending = deque()
        for dc in todo:
            pending.append((dc, pool.submit(_one, dc, n_syn, max_span_s)))
            if len(pending) >= ahead:
                break
        j = 0
        while pending:
            dc, fut = pending.popleft()
            out = fut.result()
            nxt = next(todo, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(_one, nxt, n_syn, max_span_s)))
            if out is None:
                skipped.append(dc)
            else:
                i0, sig = out
                y[i0:i0 + len(sig)] += sig          # in decoy order, as stack.render does
            j += 1
            if progress:
                progress(j, len(decoys))
        return y, skipped

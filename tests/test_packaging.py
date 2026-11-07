"""The bundled catalog is package data: a wheel built from this checkout contains it."""
import shutil
import subprocess
import sys
import zipfile

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]          # this checkout (chaff may be installed elsewhere)
SRC = ROOT / "src"
DATA = ["chaff/data/mcmaster.npz", "chaff/data/README.md"]


def test_pyproject_lists_the_data():
    tomllib = pytest.importorskip("tomllib")                # Python 3.11+
    with open(ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    listed = cfg["tool"]["setuptools"]["package-data"]["chaff"]
    for f in DATA:
        assert f.removeprefix("chaff/") in listed and (SRC / f).is_file()


@pytest.mark.slow
def test_wheel_contains_the_catalog(tmp_path):
    st = pytest.importorskip("setuptools")
    if int(st.__version__.split(".")[0]) < 77:
        pytest.skip("needs setuptools 77 or newer to build")
    tree = tmp_path / "src"                                   # build from a copy: no build/ left in the checkout
    shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(".git", "build", "dist", "*.egg-info", "__pycache__",
                                                              ".pytest_cache"))
    r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-q",
                        "-w", str(tmp_path / "dist"), str(tree)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    (whl,) = (tmp_path / "dist").glob("chaff-*.whl")
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
        for f in DATA:
            assert f in names
        assert z.getinfo(DATA[0]).file_size == (SRC / DATA[0]).stat().st_size


ZIPPED = """
import os, sys, zipfile
import chaff
from chaff import stack
assert ".zip" in chaff.__file__, chaff.__file__
p = chaff.bundled_catalog()
assert os.path.isfile(p) and p == chaff.bundled_catalog(), p     # copied out once, kept for the process
assert len(stack.load(p)) == 8 and "Hamilton" in chaff.catalog_info(p)["credit"]
print(p)
"""


def test_bundled_catalog_from_a_zip(tmp_path):
    # chaff imported from a zip (a zipapp, say): the catalog is not a file on disk, so it is copied out
    zp = tmp_path / "chaff.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for f in (SRC / "chaff").rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(SRC).as_posix())
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(zp)
    r = subprocess.run([sys.executable, "-c", ZIPPED], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert not Path(r.stdout.strip()).exists()                    # removed when the process exits

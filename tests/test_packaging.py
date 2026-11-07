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

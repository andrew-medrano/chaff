"""Memory stays flat: protecting a file never holds the recording in memory.

chaff's own working set depends on the recording's duration (the 2 kHz decoy track, one hour
of analysis frames) but not on its sample rate or channel count. So a short recording with many
channels is a cheap way to make the recording itself far bigger than that working set: 10 min of
48 kHz audio in 16 channels is 1.84 GB as float32, while the run needs about 0.4 GB."""
import numpy as np
import pytest
import soundfile as sf

from conftest import run_py

# run the command from a fresh parent, so the peak is of this run alone (getrusage(RUSAGE_CHILDREN)
# keeps the largest child a process ever had); ru_maxrss is bytes on macOS, KiB on Linux
MEASURE = """
import resource, subprocess, sys
r = subprocess.run([sys.executable, "-m", "chaff", *sys.argv[1:]], capture_output=True, text=True)
sys.stderr.write(r.stderr)
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
print(peak)
sys.exit(r.returncode)
"""


@pytest.mark.slow
def test_peak_memory_is_well_below_the_recording(tmp_path, catalog_path):
    sr, ch, minutes = 48000, 16, 10
    src, dst = tmp_path / "many.wav", tmp_path / "out.wav"
    rng = np.random.default_rng(0)
    with sf.SoundFile(src, "w", sr, ch, "PCM_16") as f:
        for _ in range(minutes * 2):
            f.write((0.05 * rng.standard_normal((sr * 30, ch))).astype(np.float32))
    in_memory = sr * 60 * minutes * ch * 4
    r = run_py("-c", MEASURE, "protect", src, dst, "--catalog", catalog_path, "--silent", "--subtype", "PCM_16",
                "--quiet-per-hour", "12", "--lead-per-hour", "6")
    assert r.returncode == 0, r.stderr
    peak = int(r.stdout.split()[-1])
    print(f"peak RSS {peak / 1e6:.0f} MB, recording {in_memory / 1e6:.0f} MB as float32")
    info = sf.info(str(dst))
    assert info.frames == sr * 60 * minutes and info.channels == ch
    assert peak < 0.4 * in_memory, f"peak RSS {peak / 1e6:.0f} MB for a {in_memory / 1e6:.0f} MB recording"

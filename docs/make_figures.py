"""Make the README figures in docs/images/.

    python docs/make_figures.py --mcmaster PATH [--research DIR] [--only hero,how,spot] [--out DIR]

Inputs (none of the audio is in this repository):

- --mcmaster PATH (or $CHAFF_MCMASTER): the Hamilton Naturalists' Club's McMaster Forest
  recording, Internet Archive item HNCSW720260527, as one mono 4 kHz WAV: the 12 hourly MP3
  files (08:00 to 20:00 local time) decoded and concatenated in order. In the research
  workspace this is heldout/mcmaster.wav. Only 27000-34200 s (15:30-17:30) is read.
  Used by "hero" and "how".
- --research DIR (or $CHAFF_RESEARCH): the research workspace. "spot" reads its protected
  McMaster crop and the decoy list and detections that go with it: chaffout/s4.wav,
  chaffout/s4_decoys.json, chaffout/s4_leadfits.json, chaffout/crop_clean_ref.json. That crop
  is the same 2 h of McMaster protected with the research recipe (v3, seed 11).
- --out DIR: where the PNGs go (default: docs/images next to this script).

Everything else comes from this library: the bundled catalog (chaff/data/mcmaster.npz), chaff
protect, stack.render_one and audibility.limit. Spectrograms are computed the way the
Doppler-curve search sees them (4 kHz, 2 s windows, 0.49 Hz bins, 0.25 s hop, 40-450 Hz, dB,
whitened over 10 minutes) with chaff.noise.Frames and chaff.detect.whiten. Needs matplotlib and
Pillow besides chaff.

Images are 8.3 in wide at 200 dpi (1660 px, shown at about 830 px on GitHub: 2x), saved as
palette PNGs to keep them small. Text is measured with the Agg renderer that draws it, so
wrapped lines fill the width.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))      # this checkout's chaff

import chaff                                                          # noqa: E402
from chaff import audibility, detect, stack                           # noqa: E402
from chaff.noise import A_N, A_SR, Frames, noise_floor                # noqa: E402
from chaff.resample import Pull                                       # noqa: E402
from chaff.stack import NFFT, StackDecoy                              # noqa: E402
from chaff.synth import SYN_SR, track_len                             # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg           # noqa: E402
from matplotlib.figure import Figure                                  # noqa: E402
from matplotlib.ticker import FixedLocator                            # noqa: E402
from PIL import Image                                                 # noqa: E402

OUT = os.path.join(HERE, "images")
CROP = (27000.0, 34200.0)      # s into the McMaster day file: 15:30-17:30 local, the research crop
DAY0_H = 8.0                   # the day file starts at 08:00 local
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#b9b8b2"
ACCENT = "#e4572e"             # the one accent: decoys
FW, DPI, FS = 8.3, 200, 11.5
FOOT_FS, FOOT_LS = 11.0, 1.3
MINUS = "\u2212"
CREDIT = "Audio: Hamilton Naturalists' Club, via the Internet Archive."
plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": FS,
    "axes.edgecolor": EDGE, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "xtick.major.size": 3, "ytick.major.size": 3,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


# ---------------------------------------------------------------- spectrograms
def spectrogram_db(x4):
    """(t, f, L): dB spectrogram of 4 kHz mono audio on the search's grid, not whitened."""
    fr = Frames(A_SR, A_N, detect.HOP, 3600, detect.F_LO, detect.F_HI, db=True)
    groups = fr.push(x4) + fr.flush()
    t = np.concatenate([g[0] for g in groups]); L = np.concatenate([g[2] for g in groups], axis=1)
    return t, fr.f, L


def search_spectrogram(x4):
    """(t, f, S): whitened dB spectrogram of 4 kHz mono audio, as the search sees it."""
    t, f, L = spectrogram_db(x4)
    return t, f, detect.whiten(L, detect.HOP / A_SR)


def crop_view(t, f, S, t0, t1, f0=None, f1=None):
    """The window [t0, t1] x [f0, f1], with the window's own per-bin median removed and clipped at 0
    (stationary hum gone, only what rises above the background left)."""
    i = (t >= t0) & (t <= t1); j = np.ones(len(f), bool)
    if f0 is not None:
        j = (f >= f0) & (f <= f1)
    W = S[np.ix_(j, i)]
    W = np.maximum(W - np.median(W, axis=1, keepdims=True), 0)
    return t[i], f[j], W


def show(ax, t, f, W, vmax_pct=99.5, extent_t=None):
    vmax = np.percentile(W, vmax_pct)
    ext = extent_t or (t[0], t[-1])
    ax.imshow(W, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=vmax, interpolation="antialiased",
              extent=(ext[0], ext[1], f[0], f[-1]), rasterized=True)


def show_tinted(ax, t, f, W, added, vmax_pct=99.5, extent_t=None):
    """Like show(), but pixels where `added` is true are drawn in the accent instead of gray."""
    v = np.clip(W / np.percentile(W, vmax_pct), 0, 1)
    rgb = plt.get_cmap("Greys")(v)[..., :3]
    acc = np.array(matplotlib.colors.to_rgb(ACCENT))
    rgb[added] = 1 - v[added][:, None] * (1 - acc)
    ext = extent_t or (t[0], t[-1])
    ax.imshow(rgb, origin="lower", aspect="auto", interpolation="antialiased",
              extent=(ext[0], ext[1], f[0], f[-1]), rasterized=True)


def iso_spectrogram(s, n=NFFT, hop=NFFT // 8):
    """dB spectrogram of an isolated (noise-free) 2 kHz signal, 2 s windows like the search's."""
    f, t, P = stack._stft_power_rows(np.asarray(s, np.float64), lambda f: (f >= detect.F_LO) & (f <= detect.F_HI), n, hop)
    L = 10 * np.log10(P + 1e-30)
    return t, f[(f >= detect.F_LO) & (f <= detect.F_HI)], L


def _palette(n_gray=24, n_tint=8):
    """A gray ramp plus the accent blended with white: everything these figures draw."""
    acc = np.array([int(ACCENT[i:i + 2], 16) for i in (1, 3, 5)], float)
    cols = [(g, g, g) for g in np.linspace(0, 255, n_gray)]
    cols += [tuple(acc + (255 - acc) * a) for a in np.linspace(0, 1, n_tint + 1)[:-1]]
    pal = Image.new("P", (1, 1))
    pal.putpalette([int(round(v)) for c in cols for v in c] + [255] * 3 * (256 - len(cols)))
    return pal


def save(fig, name):
    """Save as a small palette PNG: a gray ramp plus tints of the accent (no dithering)."""
    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, format="png")
    plt.close(fig)
    im = Image.open(buf).convert("RGB").quantize(palette=_palette(), dither=Image.Dither.NONE)
    p = os.path.join(OUT, name)
    im.save(p, optimize=True)
    print(f"{p}: {im.size[0]}x{im.size[1]}, {os.path.getsize(p) / 1024:.0f} KB")


def read_crop(path):
    info = sf.info(path)
    if info.samplerate != A_SR or info.channels != 1:
        raise SystemExit(f"{path}: expected the mono 4 kHz McMaster day file, got {info.channels} ch at {info.samplerate} Hz")
    x, _ = sf.read(path, start=int(CROP[0] * A_SR), stop=int(CROP[1] * A_SR), dtype="float32")
    return x


def clock(t_crop):
    """Local time of a moment t_crop s into the crop, as H:MM."""
    m = int(round((DAY0_H * 3600 + CROP[0] + t_crop) / 60))
    return f"{m // 60}:{m % 60:02d}"


# ---------------------------------------------------------------- text
_MEASURE = None


def _width_in(s, size, weight="normal"):
    """Width of s as the Agg renderer draws it at DPI, in inches."""
    global _MEASURE
    if _MEASURE is None:
        _MEASURE = Figure(figsize=(FW, 1), dpi=DPI); FigureCanvasAgg(_MEASURE)
    t = _MEASURE.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(_MEASURE.canvas.get_renderer()).width / DPI
    t.remove()
    return w


def wrap_full(s, size, width_in):
    """Greedy word wrap measured with the renderer, so every line of a paragraph but its last
    fills the width. Paragraphs are separated by newlines."""
    lines = []
    for para in s.split("\n"):
        cur = ""
        for w in para.split(" "):     # not str.split(): a no-break space keeps "450 Hz" together
            trial = (cur + " " + w).strip()
            if cur and _width_in(trial, size) > width_in:
                lines.append(cur); cur = w
            else:
                cur = trial
        lines.append(cur)
    return "\n".join(lines)


FOOT_B = 0.16                  # bottom margin under the footer, in


def footer_text(text, L, R):
    """(wrapped text, height in inches including the bottom margin) for a footer spanning the
    axes' width, FW - L - R."""
    t = wrap_full(text, FOOT_FS, FW - L - R)
    return t, FOOT_B + (t.count("\n") + 1) * FOOT_FS * FOOT_LS / 72


def footer(fig, wrapped, H, L):
    """The footer, left edge on the axes' left edge L."""
    fig.text(L / FW, FOOT_B / H, wrapped, fontsize=FOOT_FS, color=INK2, ha="left", va="bottom", linespacing=FOOT_LS)


def rest_ticks(ax, freqs, length_in, width_in):
    """Short accent ticks inside the left spine at freqs (Hz)."""
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for q in freqs:
        ax.plot([0, length_in / width_in], [q, q], transform=tr, color=ACCENT, lw=1.8, solid_capstyle="butt",
                clip_on=False, zorder=5)


# ---------------------------------------------------------------- 1. hero
HERO_WIN = (4200.0, 5400.0)    # s into the crop: 16:40-17:00 local, three real flyovers


def hero(x):
    """The same 20 minutes before and after `chaff protect` (defaults, bundled catalog, seed 0),
    with everything chaff added drawn in the accent."""
    y, decoys, rep = chaff.protect(x, A_SR, seed=0)
    w0, w1 = HERO_WIN
    inwin = [d for d in decoys if w0 <= d.t_cpa <= w1]
    n = {k: sum(d.tier == k for d in inwin) for k in ("lead", "quiet")}
    L, R, gap, ph, top, xt = 0.68, 0.36, 0.42, 1.52, 0.34, 0.34
    foot, fh = footer_text("McMaster Forest, 27 May 2026, local time. Spectrograms as a Doppler-curve search sees "
                           "them: 40\u2013450\u00a0Hz, 2\u00a0s windows, whitened. " + CREDIT, L, R)
    B = fh + xt
    H = B + 2 * ph + gap + top
    fig = plt.figure(figsize=(FW, H))
    t, f, Sx = search_spectrogram(x)
    _, _, Sy = search_spectrogram(y)
    _, _, Lx = spectrogram_db(x)
    _, _, Ld = spectrogram_db(y - x)
    added = Ld > Lx                                         # chaff's part of the bin's power is the larger
    del Lx, Ld
    labs = ("As recorded", "After chaff protect (default settings)")
    rows = []
    for r, (S, lab) in enumerate(((Sx, labs[0]), (Sy, labs[1]))):
        tv, fv, W = crop_view(t, f, S, w0 - 5, w1 + 5)
        ax = fig.add_axes([L / FW, (B + (1 - r) * (ph + gap)) / H, (FW - L - R) / FW, ph / H])
        ext = ((tv[0] - w0) / 60, (tv[-1] - w0) / 60)
        if r == 0:
            show(ax, tv, fv, W, extent_t=ext)
        else:
            i = (t >= w0 - 5) & (t <= w1 + 5)
            show_tinted(ax, tv, fv, W, added[:, i], extent_t=ext)
        ax.set_xlim(0, (w1 - w0) / 60); ax.set_ylim(40, 450)
        ax.set_yticks([100, 200, 300, 400]); ax.set_ylabel("Hz", color=INK2, labelpad=2)
        ax.set_xticks(np.arange(0, 21, 5))
        ax.set_xticklabels([] if r == 0 else [clock(w0 + 60 * m) for m in range(0, 21, 5)])
        ax.text(0, 1 + 0.07 / ph, lab, transform=ax.transAxes, fontsize=FS, color=INK, fontweight="bold",
                ha="left", va="bottom")
        rows.append(ax)
    # the key sits on the "After" title row, right after the title
    ax = rows[1]
    x_key = (_width_in(labs[1], FS, "bold") + 0.28) / (FW - L - R)
    y_key = 1 + 0.07 / ph
    ax.plot([x_key + 0.06 / (FW - L - R)], [y_key + 0.075 / ph], marker="s", ms=8, mfc=ACCENT, mec=ACCENT,
            transform=ax.transAxes, clip_on=False)
    ax.text(x_key + 0.17 / (FW - L - R), y_key, f"added by chaff ({n['lead']} lead and {n['quiet']} quiet decoys)",
            transform=ax.transAxes, fontsize=FS - 0.5, color=INK, ha="left", va="bottom")
    footer(fig, foot, H, L)
    save(fig, "before_after.png")
    print("  decoys in window:", [(round(d.t_cpa - w0), d.tier, round(d.f0)) for d in inwin])


# ---------------------------------------------------------------- 2. how it works
HOW_STACK = "mcmaster#3"       # 12 engine lines; its flyover is at 2391 s into the crop
HOW_T_SRC = 2391.3             # its heard closest approach, s into the crop
HOW_T_DECOY = 1580.0           # where the decoy is laid, s into the crop: a quiet stretch 13.5 min earlier
HOW_MULT = dict(f0=0.72, v=1.2, d=0.65)


def how(x):
    """One real flyover -> its de-Dopplered, isolated stack (from the bundled catalog) -> a decoy
    made from it, added to the same recording at a lead decoy's level."""
    stacks = stack.load(chaff.bundled_catalog())
    i = [k.name for k in stacks].index(HOW_STACK); k = stacks[i]
    span = 150.0

    def recorded(sig, tc):
        t, f, S = search_spectrogram(sig[int((tc - 900) * A_SR):int((tc + 900) * A_SR)])
        t = t + (tc - 900)
        return crop_view(t, f, S, tc - span - 2, tc + span + 2)

    # 1. the flyover as recorded
    t1, f1, W1 = recorded(x, HOW_T_SRC)
    # 2. the catalog's isolated stack, on its warped clock
    t2, f2, L2 = iso_spectrogram(k.s)
    m = np.abs(t2 - k.tau_cpa) <= span + 2
    t2, L2 = t2[m] - k.tau_cpa, L2[:, m]
    # 3. re-Dopplered onto a new pass (lower blade rate, faster, closer), limited like the lead tier
    #    and added to the recording, the way chaff protect adds it
    floor = noise_floor(x, A_SR)
    n_syn = track_len(len(x), A_SR)
    dc = StackDecoy(t_cpa=HOW_T_DECOY, f0=k.f0 * HOW_MULT["f0"], v=k.v * HOW_MULT["v"], d=k.d * HOW_MULT["d"],
                    src=i, level_db=32.0, seed=7, tier="lead")
    i0, sig = stack.render_one(dc, k, n_syn, floor)
    track = np.zeros(n_syn, np.float32); track[i0:i0 + len(sig)] = sig[:n_syn - i0]
    audibility.limit(track, SYN_SR, floor, chaff.LEAD.limit_db)
    mixed = x + Pull(track, SYN_SR, A_SR).read(len(x)).astype(np.float32)
    t3, f3, W3 = recorded(mixed, HOW_T_DECOY)
    rest = k.lines_hz * HOW_MULT["f0"]

    below = sum(lo < detect.F_HI for lo, _ in k.lines)
    L, R, gx, ph, top, xt = 0.68, 0.2, 0.3, 2.45, 0.36, 0.36
    foot, fh = footer_text(
        f"Stack {HOW_STACK} of the bundled catalog: blade rate {k.f0:.0f}\u00a0Hz, {k.n_lines} engine lines, "
        f"{below} of them below 450\u00a0Hz. The decoy, from stack.render_one, has blade rate "
        f"\u00d7{HOW_MULT['f0']}, speed \u00d7{HOW_MULT['v']} and closest distance \u00d7{HOW_MULT['d']}, and is added "
        f"to the same recording {abs(HOW_T_DECOY - HOW_T_SRC) / 60:.0f} minutes "
        f"{'later' if HOW_T_DECOY > HOW_T_SRC else 'earlier'}, at a lead decoy's level. "
        f"Orange ticks: its rest pitches, which its lines cross at 0\u00a0s. " + CREDIT, L, R)
    B = fh + xt
    H = B + ph + top
    fig = plt.figure(figsize=(FW, H))
    pw = (FW - L - R - 2 * gx) / 3
    labs = ["1  A real flyover", "2  De-Dopplered, lines kept", "3  A decoy, mixed in"]
    axes = []
    for c in range(3):
        ax = fig.add_axes([(L + c * (pw + gx)) / FW, B / H, pw / FW, ph / H]); axes.append(ax)
        ax.text(0, 1 + 0.07 / ph, labs[c], transform=ax.transAxes, fontsize=FS, fontweight="bold", color=INK,
                ha="left", va="bottom")
        ax.set_xlim(-span, span); ax.set_ylim(40, 450); ax.set_xticks([-100, 0, 100])
        ax.set_xticklabels([f"{MINUS}100 s", "0", "+100 s"])
        ax.set_yticks([100, 200, 300, 400])
        if c:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Hz", color=INK2, labelpad=2)
    show(axes[0], t1 - HOW_T_SRC, f1, W1)
    top_db = np.percentile(L2, 99.9)
    axes[1].imshow(np.clip(L2 - (top_db - 45), 0, None), origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=40,
                   interpolation="antialiased", extent=(t2[0], t2[-1], f2[0], f2[-1]), rasterized=True)
    show(axes[2], t3 - HOW_T_DECOY, f3, W3)
    rest_ticks(axes[2], rest[(rest > 45) & (rest < 445)], 0.09, pw)
    footer(fig, foot, H, L)
    save(fig, "how_it_works.png")
    print("  decoy rest pitches:", np.round(rest[rest < 450]).astype(int).tolist())


# ---------------------------------------------------------------- 3. spot the fake
def spot(research, seed=11):
    """Six flyovers from the protected crop: the three strongest real ones, three lead decoys."""
    co = os.path.join(research, "chaffout")
    x, sr = sf.read(os.path.join(co, "s4.wav"), dtype="float32")
    assert sr == A_SR
    t, f, S = search_spectrogram(x)
    real = sorted(json.load(open(os.path.join(co, "crop_clean_ref.json"))), key=lambda e: -e["score"])[:3]
    fits = json.load(open(os.path.join(co, "s4_leadfits.json")))
    lead = [d for d in json.load(open(os.path.join(co, "s4_decoys.json"))) if d.get("kind") == "lead"]
    fake, used = [], set()
    for e, d in sorted(zip(fits, lead), key=lambda p: -p[0]["score"]):     # strongest, three different source planes
        if 12 <= e["score"] <= 18 and d["src"] not in used and all(abs(e["t_cpa"] - g["t_cpa"]) > 200 for g in fake):
            fake.append(e); used.add(d["src"])
        if len(fake) == 3:
            break
    items = [(e, "real") for e in real] + [(e, "fake") for e in fake]
    items = [items[i] for i in np.random.default_rng(seed).permutation(len(items))]
    letters = "ABCDEF"
    L, R, gx, gy, ph, top, xt = 0.68, 0.3, 0.62, 0.42, 1.42, 0.36, 0.36
    foot, fh = footer_text("McMaster Forest, 27 May 2026, 15:30 to 17:30, protected with the research recipe. Each "
                           "panel is 220\u00a0s centred on one flyover's closest approach, frequency range fitted to "
                           "it. Judge the flyover at 0\u00a0s; other curves in a panel can be real planes or decoys. "
                           + CREDIT, L, R)
    B = fh + xt
    H = B + 2 * ph + gy + top
    fig = plt.figure(figsize=(FW, H))
    pw = (FW - L - R - 2 * gx) / 3
    for n, (e, kind) in enumerate(items):
        r, c = divmod(n, 3)
        ax = fig.add_axes([(L + c * (pw + gx)) / FW, (B + (1 - r) * (ph + gy)) / H, pw / FW, ph / H])
        lo = 10 * np.floor(max(e["f_post"] * 0.75, 41) / 10)
        hi = 10 * np.ceil(min(e["f_pre"] * 1.25, 449) / 10)
        tv, fv, W = crop_view(t, f, S, e["t_cpa"] - 112, e["t_cpa"] + 112, lo, hi)
        show(ax, tv - e["t_cpa"], fv, W)
        ax.set_xlim(-110, 110); ax.set_xticks([-100, 0, 100]); ax.set_ylim(lo, hi)
        ax.set_xticklabels([f"{MINUS}100 s", "0", "+100 s"] if r == 1 else [])
        pad = 0.06 * (hi - lo)                                 # no label within 6% of an edge
        ax.yaxis.set_major_locator(FixedLocator([q for q in range(50, 451, 50) if lo + pad <= q <= hi - pad]))
        if c == 0:
            ax.set_ylabel("Hz", color=INK2, labelpad=2)
        ax.text(0, 1 + 0.06 / ph, letters[n], transform=ax.transAxes, fontsize=FS + 1.5, fontweight="bold", color=INK,
                ha="left", va="bottom")
    footer(fig, foot, H, L)
    save(fig, "spot_the_fake.png")
    key = [letters[n] for n, (_, kind) in enumerate(items) if kind == "fake"]
    print("  fakes:", ", ".join(key), [(letters[n], round(e["t_cpa"]), kind) for n, (e, kind) in enumerate(items)])


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mcmaster", default=os.environ.get("CHAFF_MCMASTER"),
                    help="McMaster day file, mono 4 kHz WAV (research workspace: heldout/mcmaster.wav)")
    ap.add_argument("--research", default=os.environ.get("CHAFF_RESEARCH"), help="research workspace (for 'spot')")
    ap.add_argument("--only", default="hero,how,spot", help="comma-separated: hero, how, spot")
    ap.add_argument("--out", default=OUT, help="output directory (default: docs/images)")
    a = ap.parse_args()
    want = set(a.only.split(","))
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    if want & {"hero", "how"}:
        if not a.mcmaster:
            ap.error("hero and how need --mcmaster (or $CHAFF_MCMASTER)")
        x = read_crop(a.mcmaster)
        if "how" in want:
            how(x)
        if "hero" in want:
            hero(x)
    if "spot" in want:
        if not a.research:
            ap.error("spot needs --research (or $CHAFF_RESEARCH)")
        spot(a.research)


if __name__ == "__main__":
    main()

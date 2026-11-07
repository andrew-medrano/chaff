"""Make the README figures in docs/images/.

    python docs/make_figures.py --mcmaster PATH [--research DIR] [--only hero,how,spot]

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

Everything else comes from this library: the bundled catalog (chaff/data/mcmaster.npz), chaff
protect, and stack.render_one. Spectrograms are computed the way the Doppler-curve search sees
them (4 kHz, 2 s windows, 0.49 Hz bins, 0.25 s hop, 40-450 Hz, dB, whitened over 10 minutes)
with chaff.noise.Frames and chaff.detect.whiten. Needs matplotlib and Pillow besides chaff.

Images are 8.3 in wide at 200 dpi (1660 px, shown at about 830 px on GitHub: 2x), saved as
palette PNGs to keep them small.
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
from chaff import detect, stack                                       # noqa: E402
from chaff.noise import A_N, A_SR, Frames, noise_floor                # noqa: E402
from chaff.stack import NFFT, StackDecoy                              # noqa: E402
from chaff.synth import SYN_SR, track_len                             # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402
from PIL import Image                                                 # noqa: E402

OUT = os.path.join(HERE, "images")
CROP = (27000.0, 34200.0)      # s into the McMaster day file: 15:30-17:30 local, the research crop
INK, INK2, MUTED, EDGE = "#0b0b0b", "#52514e", "#8a8984", "#b9b8b2"
ACCENT = "#e4572e"             # the one accent: decoys
FW, DPI, FS = 8.3, 200, 11.5
CREDIT = "Audio: Hamilton Naturalists' Club, via the Internet Archive."
plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": FS,
    "axes.edgecolor": EDGE, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "xtick.major.size": 3, "ytick.major.size": 3,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


# ---------------------------------------------------------------- spectrograms
def search_spectrogram(x4):
    """(t, f, S): whitened dB spectrogram of 4 kHz mono audio, as the search sees it."""
    fr = Frames(A_SR, A_N, detect.HOP, 3600, detect.F_LO, detect.F_HI, db=True)
    groups = fr.push(x4) + fr.flush()
    t = np.concatenate([g[0] for g in groups]); L = np.concatenate([g[2] for g in groups], axis=1)
    return t, fr.f, detect.whiten(L, detect.HOP / A_SR)


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


X0 = 0.3                       # left and right margin of text blocks, in
FOOT_FS, FOOT_LS = FS - 1.5, 1.3


def _width_in(s, size):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), s, size=size, prop=FontProperties(family=plt.rcParams["font.family"])).get_extents().width / 72


def wrap_full(s, size, width_in):
    """Greedy word wrap measured with the real font, so every line of a paragraph but its last
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


def footer_text(text):
    """(wrapped text, height in inches including the bottom margin)."""
    t = wrap_full(text, FOOT_FS, FW - 2 * X0)
    return t, 0.16 + (t.count("\n") + 1) * FOOT_FS * FOOT_LS / 72


def footer(fig, wrapped, H):
    fig.text(X0 / FW, 0.16 / H, wrapped, fontsize=FOOT_FS, color=MUTED, ha="left", va="bottom", linespacing=FOOT_LS)


# ---------------------------------------------------------------- 1. hero
HERO_WIN = (4200.0, 5400.0)    # s into the crop: 16:40-17:00 local, three real flyovers


def hero(x):
    """The same 20 minutes before and after `chaff protect` (defaults, bundled catalog, seed 0)."""
    y, decoys, rep = chaff.protect(x, A_SR, seed=0)
    w0, w1 = HERO_WIN
    inwin = [d for d in decoys if w0 <= d.t_cpa <= w1]
    foot, fh = footer_text("Triangles mark each decoy's closest approach. McMaster Forest, 27 May 2026, about 16:40 "
                           "to 17:00 local time. Spectrograms as a "
                           "Doppler-curve search sees them: 40-450\u00a0Hz, 2\u00a0s windows, whitened.\n" + CREDIT)
    L, R, gap, ph, top = 0.62, 0.3, 0.66, 1.36, 0.36
    leg_h = 0.62                                             # x tick labels plus the legend row
    B = fh + leg_h
    H = B + 2 * ph + gap + top
    fig = plt.figure(figsize=(FW, H))
    rows = []
    labs = ("As recorded", "After chaff protect (default settings, bundled catalog)")
    for r, (sig, lab) in enumerate(((x, labs[0]), (y, labs[1]))):
        t, f, S = search_spectrogram(sig)
        tv, fv, W = crop_view(t, f, S, w0 - 5, w1 + 5)
        ax = fig.add_axes([L / FW, (B + (1 - r) * (ph + gap)) / H, (FW - L - R) / FW, ph / H])
        show(ax, (tv - w0) / 60, fv, W, extent_t=((tv[0] - w0) / 60, (tv[-1] - w0) / 60))
        ax.set_xlim(0, (w1 - w0) / 60); ax.set_ylim(40, 450)
        ax.set_yticks([100, 200, 300, 400]); ax.set_ylabel("Hz", color=INK2, labelpad=2)
        ax.set_xticks(np.arange(0, 21, 5))
        ax.set_xticklabels([] if r == 0 else [f"{m} min" if m else "0" for m in range(0, 21, 5)])
        ax.text(0, 1 + (0.07 if r == 0 else 0.2) / ph, lab, transform=ax.transAxes, fontsize=FS, color=INK, fontweight="bold",
                ha="left", va="bottom")
        rows.append(ax)
    ax = rows[1]
    mk = dict(lead=dict(ms=8.5, mfc=ACCENT), quiet=dict(ms=6, mfc="white"))
    for d in sorted(inwin, key=lambda d: d.tier != "quiet"):          # lead markers on top
        ax.plot((d.t_cpa - w0) / 60, 1 + 0.07 / ph, marker="v", transform=ax.get_xaxis_transform(), mec=ACCENT,
                mew=1.3, clip_on=False, zorder=5, **mk[d.tier])
    n = {k: sum(d.tier == k for d in inwin) for k in ("lead", "quiet")}
    hl = [Line2D([], [], ls="", marker="v", mec=ACCENT, mew=1.3, **mk[k]) for k in ("lead", "quiet")]
    fig.legend(hl, [f"lead decoy ({n['lead']} here)",
                    f"quiet decoy ({n['quiet']} here; kept under the hearing threshold, so faint)"],
               loc="lower left", bbox_to_anchor=(L / FW - 0.01, (fh + 0.04) / H), ncol=2, frameon=False,
               fontsize=FS - 1, handletextpad=0.25, columnspacing=1.4, borderaxespad=0)
    footer(fig, foot, H)
    save(fig, "before_after.png")
    print("  decoys in window:", [(round(d.t_cpa - w0), d.tier, round(d.f0)) for d in inwin])


# ---------------------------------------------------------------- 2. how it works
HOW_STACK = "mcmaster#3"       # 12 engine lines; its flyover is at 2391 s into the crop


def how(x):
    """One real flyover -> its de-Dopplered, isolated stack (from the bundled catalog) -> a decoy."""
    stacks = stack.load(chaff.bundled_catalog())
    i = [k.name for k in stacks].index(HOW_STACK); k = stacks[i]
    t_src = 2391.3                                                   # heard closest approach, s into the crop
    span = 150.0
    # 1. the flyover as recorded
    t, f, S = search_spectrogram(x[int((t_src - 900) * A_SR):int((t_src + 900) * A_SR)])
    t = t + (t_src - 900)
    t1, f1, W1 = crop_view(t, f, S, t_src - span - 2, t_src + span + 2)
    # 2. the catalog's isolated stack, on its warped clock
    t2, f2, L2 = iso_spectrogram(k.s)
    m = np.abs(t2 - k.tau_cpa) <= span + 2
    t2, L2 = t2[m] - k.tau_cpa, L2[:, m]
    # 3. re-Dopplered onto a new pass: lower blade rate, faster, closer
    floor = noise_floor(x, A_SR)
    dc = StackDecoy(t_cpa=3000.0, f0=k.f0 * 0.72, v=k.v * 1.2, d=k.d * 0.65, src=i, level_db=32.0, seed=7, tier="lead")
    i0, sig = stack.render_one(dc, k, track_len(len(x), A_SR), floor)
    t3, f3, L3 = iso_spectrogram(sig)
    t3 = t3 + i0 / SYN_SR - dc.t_cpa
    m = np.abs(t3) <= span + 2
    t3, L3 = t3[m], L3[:, m]

    below = sum(lo < detect.F_HI for lo, _ in k.lines)
    foot, fh = footer_text(
        f"Stack {HOW_STACK} of the bundled catalog: blade rate {k.f0:.0f}\u00a0Hz, {k.n_lines} engine lines, "
        f"{below} of them below 450\u00a0Hz. Decoy from stack.render_one: blade rate x0.72, speed x1.2, "
        f"closest distance x0.65.\n" + CREDIT)
    L, R, gx, ph, top = 0.6, 0.18, 0.3, 2.45, 0.36
    B = fh + 0.36
    H = B + ph + top
    fig = plt.figure(figsize=(FW, H))
    pw = (FW - L - R - 2 * gx) / 3
    labs = ["1  A real flyover", "2  De-Dopplered, lines kept", "3  Re-Dopplered: a decoy"]
    axes = []
    for c in range(3):
        ax = fig.add_axes([(L + c * (pw + gx)) / FW, B / H, pw / FW, ph / H]); axes.append(ax)
        ax.text(0, 1 + 0.07 / ph, labs[c], transform=ax.transAxes, fontsize=FS, fontweight="bold", color=INK,
                ha="left", va="bottom")
        ax.set_xlim(-span, span); ax.set_ylim(40, 450); ax.set_xticks([-100, 0, 100])
        ax.set_xticklabels(["-100 s", "0", "+100 s"])
        ax.set_yticks([100, 200, 300, 400])
        if c:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Hz", color=INK2, labelpad=2)
    show(axes[0], t1 - t_src, f1, W1)
    for ax, tt, ff, LL in ((axes[1], t2, f2, L2), (axes[2], t3, f3, L3)):
        top_db = np.percentile(LL, 99.9)
        ax.imshow(np.clip(LL - (top_db - 45), 0, None), origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=40,
                  interpolation="antialiased", extent=(tt[0], tt[-1], ff[0], ff[-1]), rasterized=True)
    footer(fig, foot, H)
    save(fig, "how_it_works.png")


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
    foot, fh = footer_text("McMaster Forest, 27 May 2026, 15:30 to 17:30 local time, protected with the research recipe. "
                           "Each panel: 220\u00a0s around closest approach, frequency range fitted to the flyover.\n" + CREDIT)
    L, R, gx, gy, ph, top = 0.6, 0.18, 0.62, 0.42, 1.42, 0.36
    B = fh + 0.36
    H = B + 2 * ph + gy + top
    fig = plt.figure(figsize=(FW, H))
    pw = (FW - L - R - 2 * gx) / 3
    for n, (e, kind) in enumerate(items):
        r, c = divmod(n, 3)
        ax = fig.add_axes([(L + c * (pw + gx)) / FW, (B + (1 - r) * (ph + gy)) / H, pw / FW, ph / H])
        lo, hi = max(e["f_post"] * 0.75, 41), min(e["f_pre"] * 1.25, 449)
        tv, fv, W = crop_view(t, f, S, e["t_cpa"] - 112, e["t_cpa"] + 112, lo, hi)
        show(ax, tv - e["t_cpa"], fv, W)
        ax.set_xlim(-110, 110); ax.set_xticks([-100, 0, 100])
        ax.set_xticklabels(["-100 s", "0", "+100 s"] if r == 1 else [])
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, steps=[1, 2, 5, 10]))
        if c == 0:
            ax.set_ylabel("Hz", color=INK2, labelpad=2)
        ax.text(0, 1 + 0.06 / ph, letters[n], transform=ax.transAxes, fontsize=FS + 1.5, fontweight="bold", color=INK,
                ha="left", va="bottom")
    footer(fig, foot, H)
    save(fig, "spot_the_fake.png")
    key = [letters[n] for n, (_, kind) in enumerate(items) if kind == "fake"]
    print("  fakes:", ", ".join(key), [(letters[n], round(e["t_cpa"]), kind) for n, (e, kind) in enumerate(items)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mcmaster", default=os.environ.get("CHAFF_MCMASTER"),
                    help="McMaster day file, mono 4 kHz WAV (research workspace: heldout/mcmaster.wav)")
    ap.add_argument("--research", default=os.environ.get("CHAFF_RESEARCH"), help="research workspace (for 'spot')")
    ap.add_argument("--only", default="hero,how,spot", help="comma-separated: hero, how, spot")
    a = ap.parse_args()
    want = set(a.only.split(","))
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

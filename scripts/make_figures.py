"""Publication-quality figures for the slides (vector PDF) and the web page (PNG).

    python scripts/make_figures.py           # needs results/metrics.json (run scripts/evaluate.py first)

Outputs: results/figures/slides/*.pdf and web/assets/figures/*.png
Every figure states whether its data are SIMULATED or REAL.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from tunesight import audio as A  # noqa: E402
from tunesight import config as C  # noqa: E402
from tunesight import objectfolder as OF  # noqa: E402
from tunesight import physics as P  # noqa: E402
from tunesight import simulate as SIM  # noqa: E402
from tunesight import vision as V  # noqa: E402

PDF_DIR = C.RESULTS_DIR / "figures" / "slides"
PNG_DIR = ROOT / "web" / "assets" / "figures"
BLUE, ORANGE, GREEN, RED, PURPLE, GREY, NAVY = "#2F6DB5", "#E8892B", "#3A9D5D", "#D1495B", "#7E57C2", "#8A94A6", "#14325C"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.edgecolor": "#55606E", "axes.titleweight": "bold", "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "lines.linewidth": 1.4,
})


def save(fig, name, web=True):
    """PDF for the slides; PNG for the web page (skipped for slide-only layouts)."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_DIR / f"{name}.pdf")
    if web:
        fig.savefig(PNG_DIR / f"{name}.png", dpi=220)
    plt.close(fig)
    print("  ", name)


def tag(ax, text, color=GREY):
    ax.text(0.99, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=6, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.5, alpha=0.9))


# ----------------------------------------------------------------------------- 1. audio model
def fig_strike():
    real = list(OF.iter_strikes(C.DATA_DIR / "objectfolder" / "audio"))
    if real:
        obj, sid, wav = real[0]
        x = A.load_audio(wav)
        an = A.analyze_strike(x, fmin=150.0)
        label = f"REAL: ObjectFolder #{obj} {OF.OBJECTS[obj][0].replace('_', ' ')}"
    elif (PDF_DIR / "fig_strike.pdf").exists():
        print("   fig_strike kept (real ObjectFolder data not downloaded; the committed figure uses it)")
        return
    else:
        x = SIM.synth_strike(SIM.SimBowl(x_cm=10), 0.4, seed=3)
        an = A.analyze_strike(x)
        label = "SIMULATED glass"
    sr = C.SR
    on = int(an.onset_s * sr)
    seg = x[max(0, on - int(0.03 * sr)):on + int(1.0 * sr)]
    fig, axs = plt.subplots(1, 3, figsize=(5.0, 1.9), gridspec_kw=dict(width_ratios=[1.1, 1.3, 0.75]))
    f, t, Z = signal.stft(seg, fs=sr, nperseg=2048, noverlap=2048 - 256)
    db = 20 * np.log10(np.abs(Z) + 1e-9)
    hi = min(8000, an.freqs.max() * 1.15)
    axs[0].pcolormesh(t, f / 1000, db, vmin=db.max() - 75, vmax=db.max(), shading="auto", cmap="magma", rasterized=True)
    axs[0].set(ylim=(0, hi / 1000), xlabel="time (s)", ylabel="kHz", title="(a) spectrogram")
    axs[0].grid(False)
    band = (an.spec_f > 100) & (an.spec_f < hi)
    axs[1].plot(an.spec_f[band] / 1000, an.spec_db[band] - an.mags_db.max(), color=GREY, lw=0.5)
    axs[1].plot(an.freqs / 1000, an.mags_db - an.mags_db.max(), "o", ms=3.2, color=RED, label="modes (FFT peaks)")
    axs[1].axvline(an.pitch / 1000, ls="--", color=GREEN, lw=1, label=f"lowest mode {an.pitch:.0f} Hz")
    axs[1].set(ylim=(-85, 18), xlabel="kHz", ylabel="dB", title="(b) spectrum: one peak per mode")
    axs[1].legend(loc="upper right", frameon=False, fontsize=6, handletextpad=0.3, borderaxespad=0.1)
    th = np.linspace(0, 2 * np.pi, 300)
    axs[2].plot(np.cos(th), np.sin(th), color=GREY, lw=0.7)
    if an.lpc is not None:
        r = an.lpc.roots
        axs[2].plot(r.real, r.imag, "x", ms=3, color=GREY, mew=0.7)
        zk = np.exp((-an.lpc.sigmas + 2j * np.pi * an.lpc.freqs) / an.lpc.sr)
        axs[2].plot(zk.real, zk.imag, "o", mfc="none", ms=6, color=RED, mew=1)
    axs[2].set_aspect("equal")
    axs[2].set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), xticks=[-1, 0, 1], yticks=[-1, 0, 1], title="(c) LPC poles")
    axs[2].set_xlabel("angle = freq., radius = decay", fontsize=6.2, color=GREY)
    fig.suptitle(label + ":  the tap is the source, the glass is the filter", fontsize=7, color=BLUE if real else GREY, y=1.02)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_strike")


# ----------------------------------------------------------------------------- 2. vision
def fig_vision():
    bowls = SIM.default_bowls(5)
    fills = [0.82, 0.70, 0.60, 0.40, 0.18]
    img = SIM.render_scene(bowls, fills, seed=2)
    obs = V.measure_frame(img)
    s = obs.scale
    rgb = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s), cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(0)
    cal = P.FrenchCalibrator(f0_prior=1150, f0_prior_cents=200)
    for lv in np.linspace(0, 0.9, 8):
        o = V.measure_frame(SIM.render_scene([bowls[0]], [lv], seed=int(rng.integers(1e9))))
        cal.add(o.bowls[0].fill_ratio, A.analyze_strike(SIM.synth_strike(bowls[0], lv, seed=int(rng.integers(1e9)))).pitch)
    fig, ax = plt.subplots(figsize=(4.9, 2.2))
    y0, y1 = int(0.30 * rgb.shape[0]), int(0.88 * rgb.shape[0])
    ax.imshow(rgb[y0:y1], extent=(0, rgb.shape[1], y1, y0))
    for b, c in zip(obs.bowls, [BLUE, ORANGE, GREEN, RED, PURPLE]):
        cnt = b.contour.reshape(-1, 2)
        ax.plot(np.r_[cnt[:, 0], cnt[0, 0]], np.r_[cnt[:, 1], cnt[0, 1]], color=c, lw=1.2)
        x, y, w, h = b.bbox
        if b.water_top_y is not None:
            ax.plot([x, x + w], [b.water_top_y] * 2, color="#C0392B", lw=1.4)
        f, sd = cal.predict(b.fill_ratio)
        ax.text(x + w / 2, y - 8, f"{b.fill_ratio:.2f}\n{f:.0f} Hz", ha="center", va="bottom",
                fontsize=6.4, color=c, fontweight="bold", linespacing=1.1)
    if obs.marker is not None:
        q = obs.marker
        ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color=GREEN, lw=1.2)
        # right-aligned to the marker's right edge so the label grows into the margin, not over glass 1
        ax.text(q[:, 0].max(), q[:, 1].min() - 6, f"ArUco scale\n{obs.ppc:.1f} px/cm", ha="right", va="bottom",
                fontsize=5.6, color=GREEN)
    ax.set_axis_off()
    ax.set_ylim(y1, y0 - 55)
    ax.text(0.0, 1.0, "SIMULATED camera frame: outline, water line (red), fill ratio h/H and camera-only pitch",
            transform=ax.transAxes, fontsize=6.2, color=GREY, va="bottom")
    save(fig, "fig_vision")


def fig_vision_steps():
    """The vision pipeline step by step on one simulated frame."""
    bowls = SIM.default_bowls(5)
    fills = [0.82, 0.70, 0.60, 0.40, 0.18]
    img = SIM.render_scene(bowls, fills, seed=2)
    obs = V.measure_frame(img)
    s = obs.scale
    im = cv2.resize(img, None, fx=s, fy=s)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = max(9, (min(small.shape[:2]) // 8) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), k / 4), (gray.shape[1], gray.shape[0]))
    diff = cv2.subtract(bg, gray)
    mask = V.foreground_mask(im, obs.marker)
    y0, y1 = int(0.40 * im.shape[0]), int(0.84 * im.shape[0])
    x1 = int(0.78 * im.shape[1])
    fig, axs = plt.subplots(1, 4, figsize=(5.2, 1.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.42]))
    axs[0].imshow(cv2.cvtColor(im[y0:y1, :x1], cv2.COLOR_BGR2RGB))
    axs[0].set_title("1  camera frame", fontsize=7)
    axs[1].imshow(diff[y0:y1, :x1] + 0.6 * hsv[y0:y1, :x1, 1], cmap="magma")
    axs[1].set_title("2  contrast to background", fontsize=7)
    axs[2].imshow(mask[y0:y1, :x1], cmap="Blues")
    for b in obs.bowls:
        if b.water_top_y is not None:
            axs[2].plot([b.bbox[0], b.bbox[0] + b.bbox[2]], [b.water_top_y - y0] * 2, color=RED, lw=1.2)
    axs[2].set_title("3  mask + water lines", fontsize=7)
    for a in axs[:3]:
        a.set_axis_off()
    b = obs.bowls[2]
    x, y, w, h = b.bbox
    sat = hsv[y:y + h, x + w // 4: x + 3 * w // 4, 1].astype(float)
    prof = np.median(sat, axis=1)
    rows = np.arange(h)
    axs[3].plot(prof, rows, color=ORANGE, lw=1.2)
    thr = (np.percentile(prof, 10) + np.percentile(prof, 90)) / 2
    axs[3].axvline(thr, ls=":", color=GREY, lw=0.8)
    if b.water_top_y is not None:
        axs[3].axhline(b.water_top_y - y, color=RED, lw=1.0)
    axs[3].invert_yaxis()
    axs[3].set_title("4  row saturation", fontsize=7)
    axs[3].set_xlabel("sat.", fontsize=6)
    axs[3].set_yticks([])
    axs[3].tick_params(labelsize=5.5)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_vision_steps")


# ----------------------------------------------------------------------------- 3. physics law
def fig_law(M):
    rng = np.random.default_rng(1)
    b = SIM.SimBowl(x_cm=12.0)
    fills, pitches = [], []
    for lv in np.linspace(0, 0.9, 14):
        o = V.measure_frame(SIM.render_scene([b], [lv], seed=int(rng.integers(1e9))))
        fills.append(o.bowls[0].fill_ratio)
        pitches.append(A.analyze_strike(SIM.synth_strike(b, lv, seed=int(rng.integers(1e9)))).pitch)
    fills, pitches = np.array(fills), np.array(pitches)
    cal = P.FrenchCalibrator(f0_prior=float(pitches.max()), f0_prior_cents=200)
    for r_, f_ in zip(fills, pitches):
        cal.add(r_, f_)
    # wide layout for the web page; stacked layout for a 4:3 slide (beside the theorem blocks)
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95))
    _draw_law(axs, b, fills, pitches, cal, tall=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_law")
    fig, axs = plt.subplots(2, 1, figsize=(2.3, 3.0))
    _draw_law(axs, b, fills, pitches, cal, tall=True)
    fig.tight_layout(h_pad=0.5)
    save(fig, "fig_law_tall", web=False)
    fig_evidence(cal)


def fig_evidence(cal):
    """Model evidence over French's exponent n for the same taps: the data choose n close to 4."""
    ns = P.N_GRID
    ev = np.array([max(cal._blr(n, r0, True)[2] for r0 in P.R0_GRID) for n in ns])   # best base offset per n
    post = ev - 0.5 * ((ns - C.FRENCH_N) / 1.0) ** 2                                   # + prior n ~ N(4, 1)
    fig, ax = plt.subplots(figsize=(2.6, 1.95))
    prior = -0.5 * ((ns - C.FRENCH_N) / 1.0) ** 2
    ax.plot(ns, ev - ev.max(), "o-", color=BLUE, ms=2.5, lw=1.2, label="data: log evidence")
    ax.plot(ns, prior, "--", color=GREY, lw=1.0, label="prior: n ~ N(4, 1)")
    nb = float(ns[np.argmax(post)])
    ax.axvline(4.0, color=GREY, ls=":", lw=1)
    ax.annotate(f"chosen n = {nb:.2f}", (nb, 0), xytext=(nb + 0.5, -5), fontsize=6.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))
    ax.text(3.92, 1.0, "theory n = 4", fontsize=6, color=GREY, ha="right", va="bottom")
    ax.set(xlabel="exponent n", ylabel="log probability (relative)", xlim=(2.4, 6.6), ylim=(-30, 4),
           title="the taps choose French's exponent")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    tag(ax, "SIMULATED")
    fig.tight_layout()
    save(fig, "fig_evidence", web=False)


def fig_burg():
    """Burg vs covariance-method LPC on a decaying 3-mode ring with known damping (40 dB SNR)."""
    sr = 20000
    t = np.arange(int(0.4 * sr)) / sr
    f_true, s_true = np.array([1100.0, 2650.0, 4300.0]), np.array([4.2, 10.0, 16.0])
    rng = np.random.default_rng(0)
    y = sum(a * np.exp(-s * t) * np.cos(2 * np.pi * f * t + p)
            for a, s, f, p in zip([1.0, 0.6, 0.4], s_true, f_true, rng.uniform(0, 2 * np.pi, 3)))
    y = y + rng.normal(0, np.std(y) * 10 ** (-40 / 20), len(y))
    y = y / np.abs(y).max()

    def damping(a):
        r = np.roots(a)
        up = r[np.imag(r) > 0]
        fr, sg = np.angle(up) * sr / (2 * np.pi), -np.log(np.abs(up)) * sr
        return np.array([sg[np.argmin(np.abs(fr - f0))] for f0 in f_true])

    s_cov, s_burg = damping(A.covariance_lpc(y, 32, rank=12)), damping(A.burg_lpc(y, 32))
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.85), gridspec_kw=dict(width_ratios=[1.2, 1]))
    tt = np.linspace(0, 0.4, 200)
    axs[0].plot(t, y, color="#C9D3E0", lw=0.3)
    for s, c, lab, ls in ((s_true[0], GREY, "true", "-"), (s_cov[0], GREEN, "covariance LPC", "--"), (s_burg[0], RED, "Burg", "-")):
        axs[0].plot(tt, np.exp(-s * tt), color=c, ls=ls, lw=1.5, label=f"{lab}: sigma = {s:.2f} /s")
    axs[0].set(xlabel="time (s)", ylabel="amplitude", ylim=(-1.05, 2.05), yticks=[-1, 0, 1],
               title="(a) envelope of the lowest mode")
    axs[0].legend(loc="upper right", frameon=False, fontsize=6, borderaxespad=0.2)
    x, w = np.arange(3), 0.26
    for k, (vals, c, lab) in enumerate(((s_true, GREY, "true"), (s_cov, GREEN, "covariance"), (s_burg, RED, "Burg"))):
        axs[1].bar(x + (k - 1) * w, vals, w, color=c, label=lab)
    for i, v in enumerate(s_burg):
        axs[1].text(i + w, v + 0.4, f"{v:.2f}", ha="center", fontsize=5.5, color=RED)
    axs[1].set(xticks=x, xticklabels=[f"{f / 1000:.1f} kHz" for f in f_true], ylabel="damping sigma (1/s)",
               title="(b) damping of each mode")
    axs[1].legend(loc="upper left", frameon=False, fontsize=6)
    tag(axs[1], "SIMULATED")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_burg", web=False)


def fig_strikepoint():
    """REAL: one glass struck at two points: the same lowest mode, but different upper modes."""
    root = C.DATA_DIR / "objectfolder" / "audio" / "91" / "audio"
    wavs = sorted(root.glob("*/mic.wav"))[:2] if root.exists() else []
    if len(wavs) < 2:
        print("   fig_strikepoint kept (needs ObjectFolder glass #91: scripts/download_objectfolder.py --chunks 91_100)")
        return
    fig, ax = plt.subplots(figsize=(5.0, 1.75))
    for wv, c, off in zip(wavs, (BLUE, ORANGE), (0.0, -80.0)):
        an = A.analyze_strike(A.load_audio(wv), fmin=150.0)
        band = (an.spec_f > 100) & (an.spec_f < 8000)
        ref = an.mags_db.max()
        ax.plot(an.spec_f[band] / 1000, an.spec_db[band] - ref + off, color=c, lw=0.45, alpha=0.8)
        ax.plot(an.freqs / 1000, an.mags_db - ref + off, "o", ms=3, color=c,
                label=f"strike point {wv.parent.name}: lowest mode {an.pitch:.0f} Hz, {len(an.freqs)} modes")
        ax.axvline(an.pitch / 1000, color=c, ls="--", lw=0.8)
    ax.set(xlabel="kHz", ylabel="dB (offset)", xlim=(0, 8), ylim=(-215, 12),
           title="REAL glass #91, two strike points: same pitch, different upper modes")
    ax.set_yticks([])
    ax.legend(loc="lower right", frameon=False, fontsize=6, ncol=2, borderaxespad=0.1)
    fig.tight_layout()
    save(fig, "fig_strikepoint", web=False)


def _draw_law(axs, b, fills, pitches, cal, tall):
    xx = np.linspace(0, 0.95, 200)
    axs[0].plot(fills, pitches, "o", ms=3.5, color=BLUE, label="taps")
    axs[0].plot(xx, [cal.predict(v)[0] for v in xx], color=RED, label=f"learned law (n = {cal.n:.2f})")
    tv = np.array([SIM.visible_fill(b, v) for v in np.linspace(0, 0.95, 80)])
    axs[0].plot(tv, P.french_freq(np.linspace(0, 0.95, 80), b.f0, b.beta, b.n), ":", color=GREY, label="true physics")
    axs[0].set(xlabel="fill ratio h/H (camera)", ylabel="pitch (Hz, mic)", title="(a) more water, lower pitch")
    if tall:
        axs[0].legend(loc="lower left", frameon=False, fontsize=6.0, borderaxespad=0.2, handlelength=1.5)
    else:
        axs[0].legend(loc="lower left", frameon=False)
    x, y, xl, yl = cal.law_points()
    axs[1].plot(x, 1e6 * y, "o", ms=3.5, color=BLUE)
    axs[1].plot(xl, 1e6 * yl, color=RED)
    # the slide (tall) version uses the slide's symbol r: the fill ratio measured above the bowl's thick base
    xlabel = (f"r^n  (n = {cal.n:.2f})" if tall
              else f"u^n   (u = (h/H - r0)/(1 - r0),  n = {cal.n:.2f})")
    axs[1].set(xlabel=xlabel, ylabel="1e6 / f^2", title="(b) the law is a straight line")
    if tall:   # the line runs bottom-left to top-right: the top-left and bottom-right corners are free
        axs[1].text(0.04, 0.95, "1/f^2 = a + b r^n", transform=axs[1].transAxes, va="top", fontsize=6.3)
    else:
        axs[1].text(0.97, 0.06, "1/f^2 = a + b u^n\nBayesian linear regression;\nevidence selects n and r0",
                    transform=axs[1].transAxes, ha="right", va="bottom", fontsize=6.3)
    tag(axs[0], "SIMULATED")


# ----------------------------------------------------------------------------- 4. learning curve
def fig_learning(M):
    lc = M.get("E5_learning_curve", {})
    fig, ax = plt.subplots(figsize=(2.45, 1.95))
    for name, col in (("SIMULATED", BLUE), ("REAL (shared calibration)", ORANGE)):
        if name in lc:
            ks = sorted(int(k) for k in lc[name])
            ax.plot(ks, [lc[name][str(k)] if str(k) in lc[name] else lc[name][k] for k in ks], "o-", ms=3, color=col, label=name.lower())
    ax.axhline(25, ls=":", color=GREY, lw=1)
    ax.text(12, 21, "25 cents", ha="right", va="top", fontsize=6, color=GREY)
    ax.set(yscale="log", xlabel="calibration taps", ylabel="median |error| (cents)", title="camera-only pitch error")
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.legend(frameon=False, loc="upper center")
    save(fig, "fig_learning")


# ----------------------------------------------------------------------------- 5. which bowl
def fig_whichbowl(M):
    E = M.get("E6_which_bowl_sim", {})
    snrs = [40, 20, 10, 5, 0]
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), sharey=True)
    meths = [("chance", "chance", GREY, ":"), ("motion only", "motion only", ORANGE, "-"),
             ("audio only (physics)", "audio only (physics)", BLUE, "-"), ("fused", "audio x motion (fused)", GREEN, "-")]
    for ax, vis in zip(axs, ("visible", "hidden")):
        for key, lab, col, ls in meths:
            ys = [E.get(f"tuned to raga | striker {vis} | SNR {s} dB", {}).get(key, np.nan) for s in snrs]
            ax.plot(snrs, ys, ls, marker="o", ms=3.2 if key != "fused" else 2.2, color=col, label=lab,
                    lw=2.4 if key == "motion only" else (1.2 if key == "fused" else 1.4), zorder=5 if key == "fused" else 3)
            ys2 = [E.get(f"random levels | striker {vis} | SNR {s} dB", {}).get(key, np.nan) for s in snrs]
            if key in ("fused", "audio only (physics)"):
                ax.plot(snrs, ys2, "--", color=col, lw=0.8, alpha=0.7)
        ax.invert_xaxis()
        ax.set(xlabel="audio SNR (dB)", title=f"striker {vis}", ylim=(0, 105))
    axs[0].set_ylabel("accuracy (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("SIMULATED: 5 identical bowls (solid: tuned to a raga, dashed: random water levels)",
                 fontsize=6.6, color=GREY, y=1.02)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_whichbowl")


# ----------------------------------------------------------------------------- 6. tuning
def fig_tuning(M):
    E = M.get("E7_tuning_sim", {})
    fig, ax = plt.subplots(figsize=(2.45, 1.95))
    names = list(E.keys())
    for i, name in enumerate(names):
        for j, (key, lab, col) in enumerate((("vision_only_all", "vision only", BLUE), ("one_tap_all", "vision + 1 tap", GREEN))):
            vals = np.abs(np.array(E[name][key], float))
            x = i * 2.6 + j
            ax.bar(x, np.nanmean(vals), 0.85, color=col, alpha=0.85, label=lab if i == 0 else None)
            ax.scatter(x + np.random.default_rng(j).uniform(-0.22, 0.22, len(vals)), vals, s=4, color="k", zorder=3, lw=0)
    ax.axhline(25, ls=":", color=GREY, lw=1)
    ax.set_xticks([0.5, 3.1], ["near-identical\nbowls", "bowls differ\n(3% spread)"])
    ax.set(ylabel="|tuning error| (cents)", title="tune-by-sight, raga Bhupali")
    ax.legend(frameon=False, loc="upper left")
    tag(ax, "SIMULATED")
    save(fig, "fig_tuning")


# ----------------------------------------------------------------------------- 7. vision robustness
def fig_robust(M):
    E = M.get("E4_vision_sim", {})
    if not E:
        return
    names = list(E.keys())
    mae = [E[n]["fill_mae"] if E[n]["fill_mae"] is not None else np.nan for n in names]
    found = [E[n]["level_found_%"] for n in names]
    fig, ax = plt.subplots(figsize=(2.45, 1.95))
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, mae, color=BLUE, alpha=0.85)
    for y, m, fnd in zip(ypos, mae, found):
        ax.text((m if np.isfinite(m) else 0) + 0.0005, y, f"{fnd:.0f}% found", va="center", fontsize=5.8, color=GREY)
    ax.set_yticks(ypos, names)
    ax.set(xlabel="fill-ratio error (MAE)", title="water level under degradation")
    ax.set_xlim(0, max(np.nanmax(mae) * 1.9, 0.02))
    tag(ax, "SIMULATED")
    save(fig, "fig_robust")


def write_numbers(M):
    """LaTeX macros with every number quoted on the slides, taken from metrics.json (no hand-copied numbers)."""
    def g(path, default=float("nan")):
        d = M
        for k in path:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    e1 = g(["E1_audio_sim", "40"], {})
    e6 = g(["E6_which_bowl_sim"], {})
    e7 = g(["E7_tuning_sim"], {})
    lc = g(["E5_learning_curve", "SIMULATED"], {})
    e7k = list(e7.keys()) if isinstance(e7, dict) else []
    vals = {
        "EoneFreq": f"{e1.get('freq_err_median_cents', float('nan')):.2f}",
        "EoneSTFT": f"{e1.get('decay_err_stft_median_%', float('nan')):.1f}",
        "EoneLPC": f"{e1.get('decay_err_lpc_median_%', float('nan')):.1f}",
        "EthreeN": f"{g(['E3_french_sim', 'bayes_n']):.2f}",
        "EthreeRzero": f"{g(['E3_french_sim', 'bayes_r0']):.2f}",
        "EfiveOne": f"{lc.get('1', float('nan')):.0f}",
        "EfiveFive": f"{lc.get('5', float('nan')):.1f}",
        "EfiveTen": f"{lc.get('10', float('nan')):.1f}",
        "EsixTunedHidden": f"{e6.get('tuned to raga | striker hidden | SNR 10 dB', {}).get('fused', float('nan')):.0f}",
        "EsixHiddenTwenty": f"{e6.get('tuned to raga | striker hidden | SNR 20 dB', {}).get('fused', float('nan')):.0f}",
        "EthreeNnl": f"{g(['E3_french_sim', 'n']):.2f}",
        "EsixRandomHidden": f"{e6.get('random levels | striker hidden | SNR 20 dB', {}).get('audio only (physics)', float('nan')):.0f}",
        "EsixFusedZero": f"{e6.get('tuned to raga | striker visible | SNR 0 dB', {}).get('fused', float('nan')):.0f}",
        "EsixAudioZero": f"{e6.get('tuned to raga | striker visible | SNR 0 dB', {}).get('audio only (physics)', float('nan')):.0f}",
        "EsevenIdVision": f"{e7[e7k[0]]['vision_only_mean_abs']:.1f}" if len(e7k) > 0 else "?",
        "EsevenIdTap": f"{e7[e7k[0]]['one_tap_mean_abs']:.1f}" if len(e7k) > 0 else "?",
        "EsevenDiffVision": f"{e7[e7k[1]]['vision_only_mean_abs']:.1f}" if len(e7k) > 1 else "?",
        "EsevenDiffTap": f"{e7[e7k[1]]['one_tap_mean_abs']:.1f}" if len(e7k) > 1 else "?",
        "EfourClean": f"{g(['E4_vision_sim', 'clean', 'fill_mae']):.3f}",
        "EfourBlur": f"{g(['E4_vision_sim', 'blur 4px', 'fill_mae']):.3f}",
        "EfourClear": f"{g(['E4_vision_sim', 'clear water', 'fill_mae']):.3f}",
    }
    out = ROOT / "slides" / "generated" / "numbers.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("% generated by scripts/make_figures.py from results/metrics.json - do not edit\n"
                   + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items()), encoding="utf-8")
    (PNG_DIR / "numbers.json").write_text(json.dumps(vals, indent=1), encoding="utf-8")
    print("   numbers.tex")


def copy_assets():
    """Slides are self-contained: copy the figures (and the app screenshot) into slides/images/figures."""
    import shutil
    dst = ROOT / "slides" / "images" / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for f in PDF_DIR.glob("*.pdf"):
        shutil.copy(f, dst / f.name)
    shot = C.RESULTS_DIR / "figures" / "app_screenshot.png"
    if shot.exists():
        shutil.copy(shot, dst / shot.name)
        shutil.copy(shot, PNG_DIR / shot.name)
    for name, out in (("WhichBowl.jpg", "anim_which.jpg"), ("TuneBySight.jpg", "anim_tune.jpg")):   # ManimGL stills on slides
        src = ROOT / "web" / "media" / name
        if src.exists():
            shutil.copy(src, dst / out)


def main():
    mpath = C.RESULTS_DIR / "metrics.json"
    M = json.load(open(mpath)) if mpath.exists() else {}
    if not M:
        print("warning: results/metrics.json missing; run scripts/evaluate.py first")
    fig_strike()
    fig_strikepoint()
    fig_burg()
    fig_vision()
    fig_vision_steps()
    fig_law(M)
    fig_learning(M)
    fig_whichbowl(M)
    fig_tuning(M)
    fig_robust(M)
    write_numbers(M)
    copy_assets()
    print(f"done -> {PDF_DIR}, {PNG_DIR}, slides/images/figures")


if __name__ == "__main__":
    main()

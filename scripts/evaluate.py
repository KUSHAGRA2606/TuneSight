"""Run every experiment and write figures, metrics.json and RESULTS.md.

    python scripts/evaluate.py                       # simulated experiments + any real data found
    python scripts/evaluate.py --session data/own/session_XXXX
    python scripts/evaluate.py --quick               # fewer trials

Every table and figure says whether it comes from SIMULATED data or REAL recordings.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from tunesight import audio as A  # noqa: E402
from tunesight import config as C  # noqa: E402
from tunesight import fusion as F  # noqa: E402
from tunesight import geometry as G  # noqa: E402
from tunesight import objectfolder as OF  # noqa: E402
from tunesight import physics as P  # noqa: E402
from tunesight import session as S  # noqa: E402
from tunesight import simulate as SIM  # noqa: E402
from tunesight import tuning as T  # noqa: E402
from tunesight import vision as V  # noqa: E402

FIG = C.RESULTS_DIR / "figures"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    return f"figures/{name}"


def cents_err(a, b):
    return 1200 * np.log2(np.asarray(a, float) / np.asarray(b, float))


# ============================================================================ figures of one strike
def plot_strike(x, an: A.StrikeAnalysis, title: str, name: str, sr=C.SR):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    on = int(an.onset_s * sr)
    seg = x[max(0, on - int(0.05 * sr)):on + int(1.2 * sr)]
    f, t, Z = signal.stft(seg, fs=sr, nperseg=2048, noverlap=2048 - 256)
    db = 20 * np.log10(np.abs(Z) + 1e-9)
    ax[0, 0].pcolormesh(t, f, db, vmin=db.max() - 80, vmax=db.max(), shading="auto", cmap="magma")
    ax[0, 0].set_ylim(0, min(C.FMAX, an.freqs.max() * 1.3))
    ax[0, 0].set(title="Spectrogram (strike = source, ringing = filter)", xlabel="time (s)", ylabel="Hz")
    band = (an.spec_f > 50) & (an.spec_f < min(C.FMAX, an.freqs.max() * 1.3))
    ax[0, 1].plot(an.spec_f[band], an.spec_db[band], lw=0.8, color="0.3")
    ax[0, 1].plot(an.freqs, an.mags_db, "o", color="C3", label="FFT modes")
    ax[0, 1].axvline(an.pitch, color="C2", ls="--", label=f"pitch {an.pitch:.1f} Hz")
    ax[0, 1].set(title="Spectrum and detected modes", xlabel="Hz", ylabel="dB")
    ax[0, 1].legend()
    th = np.linspace(0, 2 * np.pi, 400)
    ax[1, 0].plot(np.cos(th), np.sin(th), color="0.6", lw=0.8)
    if an.lpc is not None:
        r = an.lpc.roots
        ax[1, 0].plot(r.real, r.imag, "x", color="0.5", label="all LPC poles")
        zk = np.exp((-an.lpc.sigmas + 2j * np.pi * an.lpc.freqs) / an.lpc.sr)
        ax[1, 0].plot(zk.real, zk.imag, "o", mfc="none", color="C3", ms=9, label="resonant poles = modes")
    ax[1, 0].set_aspect("equal")
    ax[1, 0].set(title="All-pole model: poles on the unit circle", xlabel="Re z", ylabel="Im z")
    ax[1, 0].legend(fontsize=8, loc="lower left")
    y = A.resynthesize(an, sr, 1.2)
    fo, dbo, _ = A.spectrum_db(x[on + int(0.004 * sr):on + int(1.0 * sr)], sr)
    fr, dbr, _ = A.spectrum_db(y[int(0.004 * sr):int(1.0 * sr)], sr)
    b1 = (fo > 50) & (fo < min(C.FMAX, an.freqs.max() * 1.3))
    b2 = (fr > 50) & (fr < min(C.FMAX, an.freqs.max() * 1.3))
    ax[1, 1].plot(fo[b1], dbo[b1] - dbo[b1].max(), lw=0.8, label="original")
    ax[1, 1].plot(fr[b2], dbr[b2] - dbr[b2].max(), lw=0.8, alpha=0.8, label="resynthesised from modes")
    ax[1, 1].set_ylim(-90, 5)
    ax[1, 1].set(title="Resynthesis from ~3 numbers per mode", xlabel="Hz", ylabel="dB (rel.)")
    ax[1, 1].legend()
    fig.suptitle(title)
    return save(fig, name)


# ============================================================================ E1 audio model
def e1_synthetic(rng, n=90):
    rows = []
    for snr in (40, 20, 10):
        for _ in range(n // 3):
            k = int(rng.integers(3, 6))
            f1 = rng.uniform(300, 1500)
            ratios = np.concatenate([[1.0], np.sort(rng.uniform(1.6, 6.0, k - 1))])
            fr = f1 * ratios
            fr = fr[fr < 7000]
            eta = rng.uniform(5e-4, 4e-3, len(fr))
            sg = np.pi * fr * eta
            am = np.exp(-0.5 * np.arange(len(fr)))
            x = np.zeros(int(1.6 * C.SR))
            on = int(0.15 * C.SR)
            x[on:] = A.modal_synthesis(fr, sg, am, C.SR, 1.6 - 0.15, seed=int(rng.integers(1e9)))
            ref = np.sqrt(np.mean(x[on:on + int(0.1 * C.SR)] ** 2))
            x += rng.standard_normal(len(x)) * ref * 10 ** (-snr / 20)
            an = A.analyze_strike(x)
            if an is None:
                continue
            for fk, sk in zip(fr, sg):
                d = np.abs(cents_err(an.freqs, fk))
                j = int(np.argmin(d))
                hit = d[j] < 30
                rows.append({"snr": snr, "hit": hit, "f_err_c": d[j] if hit else np.nan,
                             "s_stft_err": abs(an.sigma[j] / sk - 1) * 100 if hit and np.isfinite(an.sigma[j]) else np.nan,
                             "s_lpc_err": abs(an.sigma_lpc[j] / sk - 1) * 100 if hit and np.isfinite(an.sigma_lpc[j]) else np.nan})
    out = {}
    for snr in (40, 20, 10):
        r = [q for q in rows if q["snr"] == snr]
        out[snr] = {"mode_detection_%": 100 * np.mean([q["hit"] for q in r]),
                    "freq_err_median_cents": float(np.nanmedian([q["f_err_c"] for q in r])),
                    "decay_err_stft_median_%": float(np.nanmedian([q["s_stft_err"] for q in r])),
                    "decay_err_lpc_median_%": float(np.nanmedian([q["s_lpc_err"] for q in r]))}
    return out


def e1_real(of_root: Path, max_per_obj=12):
    strikes = list(OF.iter_strikes(of_root / "audio"))
    if not strikes:
        return None, []
    per_obj, examples, figs = {}, {}, []
    for obj, sid, wav in strikes:
        if len(per_obj.get(obj, [])) >= max_per_obj:
            continue
        x = A.load_audio(wav)
        # the app's band: lines below C.FMIN (stand, room) would otherwise be taken as the pitch
        an = A.analyze_strike(x, fmin=C.FMIN)
        if an is None:
            continue
        both = np.isfinite(an.sigma) & np.isfinite(an.sigma_lpc) & np.isfinite(an.freq_lpc)
        y = A.resynthesize(an, C.SR, 1.0)
        on = int(an.onset_s * C.SR)
        fo, dbo, _ = A.spectrum_db(x[on + 200:on + int(0.8 * C.SR)], C.SR)
        fr, dbr, _ = A.spectrum_db(y[200:int(0.8 * C.SR)], C.SR)
        m = min(len(dbo), len(dbr))
        band = (fo[:m] > 60) & (fo[:m] < 8000)
        rho = float(np.corrcoef(dbo[:m][band], dbr[:m][band])[0, 1])
        per_obj.setdefault(obj, []).append({
            "pitch": an.pitch, "n_modes": len(an.freqs), "snr": an.snr_db,
            "lpc_fft_freq_cents": list(np.abs(cents_err(an.freq_lpc[both], an.freqs[both]))),
            "lpc_stft_decay_ratio": list(an.sigma_lpc[both] / an.sigma[both]),
            "resynth_spec_corr": rho})
        examples.setdefault(obj, (x, an))
    summary = {}
    for obj, rs in per_obj.items():
        name, mat = OF.OBJECTS.get(obj, ("?", "?"))
        p = np.array([r["pitch"] for r in rs])
        fc = np.concatenate([r["lpc_fft_freq_cents"] for r in rs]) if rs else np.array([])
        dr = np.concatenate([r["lpc_stft_decay_ratio"] for r in rs]) if rs else np.array([])
        summary[obj] = {"name": name, "material": mat, "strikes": len(rs),
                        "pitch_median_hz": float(np.median(p)),
                        "pitch_mode_share_%": float(100 * np.mean(np.abs(cents_err(p, np.median(p))) < 30)),
                        "lpc_vs_fft_freq_median_cents": float(np.median(fc)) if len(fc) else None,
                        "lpc_vs_stft_decay_median_ratio": float(np.median(dr)) if len(dr) else None,
                        "resynth_spectral_corr_median": float(np.median([r["resynth_spec_corr"] for r in rs]))}
    for obj in list(examples)[:3]:
        x, an = examples[obj]
        name, mat = OF.OBJECTS.get(obj, ("?", "?"))
        figs.append(plot_strike(x, an, f"REAL (ObjectFolder #{obj} {name}, {mat})", f"e1_real_strike_{obj}.png"))
    return summary, figs


# ============================================================================ E2 scaling law
def e2_scaling(of_root: Path, e1_summary):
    mesh_root = of_root / "mesh"
    if not e1_summary or not mesh_root.exists():
        return None, None
    fams = {}
    for fam, ids in OF.SIZE_FAMILIES.items():
        pts = []
        for i in ids:
            if i in e1_summary:
                mp = OF.find_mesh(mesh_root, i)
                if mp is not None:
                    L = float(OF.obj_mesh_extents(mp)[0])
                    pts.append((i, L, e1_summary[i]["pitch_median_hz"]))
        if len(pts) >= 2:
            fams[fam] = pts
    if not fams:
        return None, None
    fig, ax = plt.subplots(figsize=(6.5, 5))
    res = {}
    for k, (fam, pts) in enumerate(fams.items()):
        L = np.array([p[1] for p in pts])
        f = np.array([p[2] for p in pts])
        ax.loglog(L, f, "o-", color=f"C{k}", label=fam)
        res[fam] = P.scaling_fit(L, f) if len(pts) >= 2 else None
    ax.set(xlabel="largest dimension L (mesh units)", ylabel="pitch (Hz)", title="REAL: scaling law f ~ 1/L (slope -1 expected)")
    ax.legend(fontsize=8)
    return res, save(fig, "e2_scaling_law.png")


# ============================================================================ simulated helpers
def sim_tap(bowl, fill, rng, snr=40, size=(1280, 720), blur=0.0, brightness=1.0, noise=2.0, clear=False):
    bowls = [bowl]
    img = SIM.render_scene(bowls, [fill], size=size, blur=blur, brightness=brightness, noise=noise,
                           seed=int(rng.integers(1e9)), clear_water=clear)
    obs = V.measure_frame(img)
    vf = obs.bowls[0].fill_ratio if obs.bowls else np.nan
    vc = obs.bowls[0].level_conf if obs.bowls else 0.0
    x = SIM.synth_strike(bowl, fill, snr_db=snr, seed=int(rng.integers(1e9)))
    an = A.analyze_strike(x)
    return vf, vc, (an.pitch if an else np.nan), (an.confidence if an else 0.0)


# ============================================================================ E3 water-loading law
def e3_french_sim(rng, levels=10, reps=2):
    b = SIM.SimBowl(x_cm=10.0, f0=1100.0, beta=3.0)
    b.x_cm = 12.0
    fills, pitches = [], []
    for lv in np.linspace(0, 0.9, levels):
        for _ in range(reps):
            vf, vc, p, _ = sim_tap(b, lv, rng)
            fills.append(vf)
            pitches.append(p)
    fills, pitches = np.array(fills), np.array(pitches)
    ok = np.isfinite(fills) & np.isfinite(pitches)
    fit = P.fit_french(fills[ok], pitches[ok], fit_n=True, fit_r0=True)
    cal = P.FrenchCalibrator(f0_prior=float(pitches[ok].max()), f0_prior_cents=200)
    for r_, f_ in zip(fills[ok], pitches[ok]):
        cal.add(r_, f_)
    fit.update({"bayes_n": cal.n, "bayes_r0": cal.r0, "bayes_f0": cal.f0, "bayes_beta": cal.beta})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xx = np.linspace(0, 0.95, 200)
    ax.plot(fills[ok], pitches[ok], "o", label="taps (vision fill, audio pitch)")
    uu = np.clip((xx - fit["r0"]) / (1 - fit["r0"]), 0, 1.2)
    ax.plot(xx, P.french_freq(uu, fit["f0"], fit["beta"], fit["n"]), label=f"fit: n={fit['n']:.2f}, r0={fit['r0']:.3f}, rmse {fit['rmse_cents']:.1f} c")
    tv = np.array([SIM.visible_fill(b, v) for v in np.linspace(0, 0.95, 60)])
    ax.plot(tv, P.french_freq(np.linspace(0, 0.95, 60), b.f0, b.beta, b.n), "--", color="0.5", label="true physics")
    ax.set(xlabel="fill ratio h/H (from camera)", ylabel="pitch (Hz, from microphone)", title="SIMULATED: water loading law")
    ax.legend()
    return fit, save(fig, "e3_french_sim.png")


def e3_french_real(taps):
    groups = S.by_bowl(taps)
    if not groups:
        return None, None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    res = {}
    xx = np.linspace(0, 0.95, 200)
    for k, (b, g) in enumerate(sorted(groups.items())):
        fl = np.array([t.fill for t in g])
        pf = np.array([t.pitch for t in g])
        if len(g) < 3:
            continue
        fit = P.fit_french(fl, pf, fit_n=len(g) >= 6, fit_r0=len(g) >= 6)
        res[b] = fit
        ax.plot(fl, pf, "o", color=f"C{k}", label=f"bowl {b + 1}: n={fit['n']:.2f}, rmse {fit['rmse_cents']:.1f} c")
        uu = np.clip((xx - fit["r0"]) / (1 - fit["r0"]), 0, 1.2)
        ax.plot(xx, P.french_freq(uu, fit["f0"], fit["beta"], fit["n"]), color=f"C{k}")
    ax.set(xlabel="fill ratio h/H (from camera)", ylabel="pitch (Hz)", title="REAL: water loading law per bowl")
    ax.legend(fontsize=8)
    return res, save(fig, "e3_french_real.png")


# ============================================================================ E4 vision robustness
def e4_vision(rng, n=8):
    b = SIM.SimBowl(x_cm=12.0)
    conds = [("clean", {}), ("blur 2px", {"blur": 2.0}), ("blur 4px", {"blur": 4.0}), ("blur 6px", {"blur": 6.0}),
             ("dark x0.6", {"brightness": 0.6}), ("dark x0.35", {"brightness": 0.35}),
             ("noise 10", {"noise": 10.0}), ("clear water", {"clear": True})]
    out = {}
    for name, kw in conds:
        errs, det = [], 0
        for _ in range(n):
            lv = rng.uniform(0.05, 0.9)
            img = SIM.render_scene([b], [lv], blur=kw.get("blur", 0.0), brightness=kw.get("brightness", 1.0),
                                   noise=kw.get("noise", 2.0), seed=int(rng.integers(1e9)), clear_water=kw.get("clear", False))
            obs = V.measure_frame(img)
            if obs.bowls and obs.bowls[0].level_conf >= 0.3:
                det += 1
                errs.append(abs(obs.bowls[0].fill_ratio - SIM.visible_fill(b, lv)))
        out[name] = {"level_found_%": 100 * det / n, "fill_mae": float(np.mean(errs)) if errs else None}
    return out


# ============================================================================ E5 learning curve
def e5_learning_curve(rng, taps_real=None, max_k=12, trials=20):
    """Error of camera-only pitch prediction vs number of calibration taps (online self-calibration)."""
    b = SIM.SimBowl(x_cm=12.0)
    pool = []
    for lv in np.linspace(0, 0.9, 16):
        vf, vc, p, _ = sim_tap(b, lv, rng)
        if np.isfinite(vf) and np.isfinite(p):
            pool.append((vf, p))
    curves = {"SIMULATED": _curve(pool, rng, max_k, trials)}
    if taps_real:
        groups = S.by_bowl(taps_real)
        real_pool = [(t.fill, t.pitch) for g in groups.values() for t in g]
        if len(real_pool) >= 6:
            # pooled across bowls: assumes identical bowls (shared physics)
            curves["REAL (shared calibration)"] = _curve(real_pool, rng, min(max_k, len(real_pool) - 2), trials)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, (name, cv) in enumerate(curves.items()):
        ks = sorted(cv)
        med = [np.median(cv[q]) for q in ks]
        p75 = [np.percentile(cv[q], 75) for q in ks]
        ax.plot(ks, med, "o-", color=f"C{k}", label=f"{name} median")
        ax.fill_between(ks, [np.percentile(cv[q], 25) for q in ks], p75, color=f"C{k}", alpha=0.2)
    ax.axhline(25, ls=":", color="0.4", label="25 cents (quarter semitone)")
    ax.set(xlabel="number of calibration taps", ylabel="|pitch error| of camera-only prediction (cents)",
           title="Audio teaches vision: online self-calibration", yscale="log")
    ax.legend(fontsize=8)
    return {k: {q: float(np.median(v[q])) for q in sorted(v)} for k, v in curves.items()}, save(fig, "e5_learning_curve.png")


def _curve(pool, rng, max_k, trials):
    out = {}
    fmax = max(p for _, p in pool)
    for k in range(1, max_k + 1):
        errs = []
        for _ in range(trials):
            idx = rng.permutation(len(pool))
            tr, te = idx[:k], idx[k:]
            cal = P.FrenchCalibrator(f0_prior=fmax * 1.05, f0_prior_cents=150)
            for i in tr:
                cal.add(*pool[i])
            errs += [abs(cents_err(cal.predict(pool[i][0])[0], pool[i][1])) for i in te]
        out[k] = errs
    return out


# ============================================================================ E6 which bowl was struck
SIZE6, PPC6 = (960, 540), 13.5


def strike_frames(bowls, target, visible, n=6, dt=0.05):
    """Frames before a strike: first without the striker, then the striker moving in and touching the rim."""
    frames = []
    for j in range(n):
        stick = None if (j == 0 or not visible) else target
        img = SIM.render_scene(bowls, size=SIZE6, ppc=PPC6, stick_to=stick, stick_phase=min(1.0, j / (n - 1)),
                               noise=1.5, marker=False)
        frames.append((j * dt, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.0))
    return frames


def e6_which_bowl(rng, trials=40, snrs=(40, 20, 10, 5, 0), n_bowls=5):
    bowls = SIM.default_bowls(n_bowls, identical=True)        # realistic close spacing (9.5 cm)
    cal = P.FrenchCalibrator(f0_prior=1150, f0_prior_cents=200)
    for lv in rng.uniform(0, 0.9, 8):                          # shared physics from 8 taps, as in the demo
        vf, _, p, _ = sim_tap(bowls[0], lv, rng)
        cal.add(vf, p)
    # a jal tarang tuned to raga Bhupali: physical fills that give the raga's notes
    sa, plan = T.plan([cal] * n_bowls, "Bhupali")
    tuned = [P.french_height(t.target_hz, bowls[t.bowl].f0, bowls[t.bowl].beta) for t in plan]
    res = {}
    for scen in ("tuned to raga", "random levels"):
        for visible in (True, False):
            for snr in snrs:
                acc = {"chance": [], "motion only": [], "audio only (physics)": [], "fused": []}
                for _ in range(trials):
                    fills = tuned if scen == "tuned to raga" else rng.uniform(0.0, 0.85, n_bowls)
                    for b, fl in zip(bowls, fills):
                        b.fill = float(np.nan_to_num(fl))
                    tgt = int(rng.integers(n_bowls))
                    frames = strike_frames(bowls, tgt, visible)
                    obs = V.measure_frame(SIM.render_scene(bowls, size=SIZE6, ppc=PPC6, noise=1.5, marker=False))
                    if len(obs.bowls) != n_bowls:
                        continue
                    preds = [cal.predict(bo.fill_ratio) for bo in obs.bowls]
                    an = A.analyze_strike(SIM.synth_strike(bowls[tgt], snr_db=snr, seed=int(rng.integers(1e9))))
                    mp, mc = V.motion_evidence(frames, [bo.bbox for bo in obs.bowls], t_onset=frames[-1][0])
                    acc["chance"].append(1.0 / n_bowls)
                    acc["motion only"].append(float(int(np.argmax(mp)) == tgt) if mc > 0 else 1.0 / n_bowls)
                    if an is None:
                        acc["audio only (physics)"].append(1.0 / n_bowls)
                        acc["fused"].append(acc["motion only"][-1])
                        continue
                    pf, ps = [q[0] for q in preds], [q[1] for q in preds]
                    ida = F.identify_struck(an.pitch, pf, ps, audio_conf=an.confidence)
                    idf = F.identify_struck(an.pitch, pf, ps, motion_p=mp, motion_conf=mc, audio_conf=an.confidence)
                    acc["audio only (physics)"].append(float(int(np.argmax(ida.p_audio[:n_bowls])) == tgt))
                    acc["fused"].append(float(int(np.argmax(idf.probs[:n_bowls])) == tgt))
                res[(scen, visible, snr)] = {k: 100 * float(np.mean(v)) for k, v in acc.items() if v}
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for r, scen in enumerate(("tuned to raga", "random levels")):
        for c, vis in enumerate((True, False)):
            ax = axs[r, c]
            for k, meth in enumerate(["chance", "motion only", "audio only (physics)", "fused"]):
                ax.plot(snrs, [res[(scen, vis, s_)][meth] for s_ in snrs], "o-", color=f"C{k}", label=meth)
            ax.set(title=f"{scen}, striker {'visible' if vis else 'hidden'}", xlabel="audio SNR (dB)")
            ax.invert_xaxis()
    axs[0, 0].set_ylabel("accuracy (%)")
    axs[1, 0].set_ylabel("accuracy (%)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("SIMULATED: which of 5 identical glasses was struck?")
    out = {f"{k[0]} | striker {'visible' if k[1] else 'hidden'} | SNR {k[2]} dB": v for k, v in res.items()}
    return out, save(fig, "e6_which_bowl.png")


# ============================================================================ E7 closed-loop tuning
def _pour_until(b, cal, f_stop, rng, max_fill=0.95):
    """Pour while the camera watches; stop when the camera-predicted pitch reaches f_stop."""
    r_target = cal.height_for(f_stop)
    obs = V.measure_frame(SIM.render_scene([b], [b.fill], size=SIZE6, ppc=PPC6, seed=int(rng.integers(1e9))))
    seen = obs.bowls[0].fill_ratio if obs.bowls else 0.0          # look before pouring
    if cal.predict(seen)[0] <= f_stop:
        return seen
    for _ in range(400):
        gap = (r_target - seen) if np.isfinite(r_target) else 0.02
        b.fill = min(max_fill, b.fill + float(np.clip(0.5 * gap, 0.002, 0.03)))   # slow down near the target
        obs = V.measure_frame(SIM.render_scene([b], [b.fill], size=SIZE6, ppc=PPC6, seed=int(rng.integers(1e9))))
        if obs.bowls and obs.bowls[0].level_conf >= 0.3:
            seen = obs.bowls[0].fill_ratio
        if cal.predict(seen)[0] <= f_stop or b.fill >= max_fill:
            return seen
    return seen


def e7_tuning(rng, raga="Bhupali", n_bowls=5, calib_taps=8, short_cents=100.0, f0_jitter=0.004):
    """One raga tuning run. (a) vision only; (b) first pour stops one semitone short, one verification tap
    measures this bowl's pitch offset, then the camera guides the final top-up."""
    bowls = SIM.default_bowls(n_bowls, identical=True, seed=int(rng.integers(1e6)), f0_jitter=f0_jitter)
    cal = P.FrenchCalibrator(f0_prior=1150, f0_prior_cents=200)
    for lv in rng.uniform(0, 0.9, calib_taps):
        vf, _, p, _ = sim_tap(bowls[0], lv, rng)
        cal.add(vf, p)
    sa, targets = T.plan([cal] * n_bowls, raga)
    rows = []
    for tg in targets:
        b = bowls[tg.bowl]
        # (a) vision only: pour straight to the target using the shared physics
        b.fill = 0.0
        _pour_until(b, cal, tg.target_hz, rng)
        an_a = A.analyze_strike(SIM.synth_strike(b, b.fill, seed=int(rng.integers(1e9))))
        # (b) vision + one verification tap: stop a semitone early, tap, measure this bowl's offset, finish pouring
        b.fill = 0.0
        own = P.FrenchCalibrator.from_dict(cal.to_dict())
        seen = _pour_until(b, own, tg.target_hz * 2 ** (short_cents / 1200), rng)
        an1 = A.analyze_strike(SIM.synth_strike(b, b.fill, seed=int(rng.integers(1e9))))
        if an1 is not None:
            own.set_offset_from_tap(seen, an1.pitch)     # this bowl's pitch offset from one tap
        _pour_until(b, own, tg.target_hz, rng)
        an_b = A.analyze_strike(SIM.synth_strike(b, b.fill, seed=int(rng.integers(1e9))))
        rows.append({"note": tg.note.label, "target_hz": tg.target_hz,
                     "vision_only_cents": float(cents_err(an_a.pitch, tg.target_hz)) if an_a else np.nan,
                     "with_one_tap_cents": float(cents_err(an_b.pitch, tg.target_hz)) if an_b else np.nan,
                     "heard_hz": an_b.pitch if an_b else np.nan})
    return {"raga": raga, "sa_hz": sa, "notes": rows}


def e7_study(seeds=(0, 1, 2), spreads=((0.004, "near-identical bowls (0.4% spread)"), (0.03, "bowls differ (3% spread)"))):
    out = {}
    for jit, name in spreads:
        runs = [e7_tuning(np.random.default_rng(s), f0_jitter=jit) for s in seeds]
        v = np.array([n["vision_only_cents"] for r in runs for n in r["notes"]])
        t = np.array([n["with_one_tap_cents"] for r in runs for n in r["notes"]])
        out[name] = {"runs": runs, "vision_only_mean_abs": float(np.nanmean(np.abs(v))), "vision_only_max_abs": float(np.nanmax(np.abs(v))),
                     "one_tap_mean_abs": float(np.nanmean(np.abs(t))), "one_tap_max_abs": float(np.nanmax(np.abs(t))),
                     "vision_only_all": v.tolist(), "one_tap_all": t.tolist()}
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, r) in enumerate(out.items()):
        for j, (key, lab) in enumerate((("vision_only_all", "vision only"), ("one_tap_all", "vision + 1 tap"))):
            vals = np.abs(r[key])
            x = i * 3 + j
            ax.bar(x, vals.mean(), 0.8, color=f"C{j}", alpha=0.85, label=lab if i == 0 else None)
            ax.scatter(np.full(len(vals), x) + np.random.default_rng(0).uniform(-0.25, 0.25, len(vals)), vals, s=10, color="k", zorder=3)
    ax.set_xticks([0.5, 3.5], list(out.keys()))
    ax.axhline(25, ls=":", color="0.4", label="25 cents")
    ax.set(ylabel="|tuning error| (cents)", title="SIMULATED: tune-by-sight, raga Bhupali (3 runs x 5 notes)")
    ax.legend(fontsize=8)
    return out, save(fig, "e7_tuning_sim.png")


# ============================================================================ E8-E10 beyond identical glasses
def random_container(rng, x_cm=12.0, a_mm=1.5, thick_spread=0.0):
    """A glass of random size and taper; its pitch follows French's thin-wall physics (see geometry.py)."""
    R_mean, H, taper = rng.uniform(2.5, 4.5), rng.uniform(7.0, 12.0), rng.uniform(0.70, 0.95)
    a = a_mm * (1 + thick_spread * rng.standard_normal())
    return SIM.family_bowl(x_cm, R_mean, H, taper, a)


def sim_tap_shape(bowl, fill, rng, snr=40):
    """One tap: the camera's fill ratio and container shape, and the microphone's pitch."""
    obs = V.measure_frame(SIM.render_scene([bowl], [fill], seed=int(rng.integers(1e9))))
    o = obs.bowls[0] if obs.bowls else None
    an = A.analyze_strike(SIM.synth_strike(bowl, fill, snr_db=snr, seed=int(rng.integers(1e9))))
    return (o.fill_ratio if o else np.nan), G.shape_from_obs(o), (an.pitch if an else np.nan)


def e8_shape(rng, n=30):
    """Container shape from the camera (radius, height, capacity) against the simulator's truth."""
    out = {}
    for ppc, blur in ((18.0, 0.0), (12.0, 0.0), (18.0, 2.0)):
        er, et, eh, ec, found = [], [], [], [], 0
        for _ in range(n):
            b = random_container(rng)
            img = SIM.render_scene([b], [rng.uniform(0.2, 0.8)], ppc=ppc, blur=blur, seed=int(rng.integers(1e9)))
            obs = V.measure_frame(img)
            s = G.shape_from_obs(obs.bowls[0]) if obs.bowls else None
            if s is None:
                continue
            found += 1
            er.append(10 * (s.R_mean_cm - (b.D_top_cm + b.D_bot_cm) / 4))
            et.append(10 * (s.R_top_cm - b.D_top_cm / 2))
            eh.append(10 * (s.H_cm - b.H_cm))
            ec.append(100 * (s.capacity_ml / G.frustum_capacity_ml(b.D_top_cm / 2, b.D_bot_cm / 2, b.H_cm) - 1))
        key = f"{ppc:.0f} px/cm" + (f", blur {blur:.0f} px" if blur else "")
        out[key] = {"found_%": 100.0 * found / n,
                    "mean_radius_mae_mm": float(np.median(np.abs(er))), "rim_radius_mae_mm": float(np.median(np.abs(et))),
                    "height_mae_mm": float(np.median(np.abs(eh))), "capacity_mape_%": float(np.median(np.abs(ec)))}
    return out


def _family_data(rng, n, thick_spread, calib_fills=(0.15, 0.35, 0.55, 0.7, 0.85), test_fills=(0.2, 0.4, 0.6, 0.8)):
    """n containers of one family (same glass, wall thickness spread). For each: its shape from the camera,
    a 5-tap calibration (French's law with n = 4 and the base offset fixed), and held-out test taps."""
    data = []
    for _ in range(n):
        b = random_container(rng, thick_spread=thick_spread)
        taps = [sim_tap_shape(b, lv, rng) for lv in calib_fills]
        shapes = [s for _, s, _ in taps if s is not None]
        shape = G.Shape(*(float(np.median([getattr(s, k) for s in shapes]))
                          for k in ("H_cm", "R_top_cm", "R_bot_cm", "capacity_ml")))
        cal = P.FrenchCalibrator(n=C.FRENCH_N, r0=G.R0_FAMILY, learn_shape=False,
                                 f0_prior=float(np.nanmax([p for _, _, p in taps])), f0_prior_cents=300)
        for vf, _, p in taps:
            if np.isfinite(vf) and np.isfinite(p):
                cal.add(vf, p)
        tests = [(vf, p) for vf, _, p in (sim_tap_shape(b, lv, rng) for lv in test_fills) if np.isfinite(vf) and np.isfinite(p)]
        data.append({"bowl": b, "shape": shape, "f0": cal.f0, "beta": cal.beta, "tests": tests})
    return data


def _loo_law(data, k):
    train = [d for j, d in enumerate(data) if j != k]
    return G.FamilyLaw.fit([d["shape"] for d in train], [d["f0"] for d in train], [d["beta"] for d in train]), train


def e9_new_container(rng, spreads=(0.0, 0.03, 0.10), n=8, reps=2):
    """Predict a NEW container's pitch from its shape alone: leave one container out, fit the family's two
    constants on the others, then score the held-out container's test taps."""
    methods = ("family average, shape ignored", "shape, zero taps", "shape + 1 check tap")
    out, example = {}, None
    for spread in spreads:
        errs = {m: [] for m in methods}
        for rep in range(reps):
            data = _family_data(rng, n, spread)
            for k, d in enumerate(data):
                law, train = _loo_law(data, k)
                f0_avg = float(np.exp(np.mean(np.log([t["f0"] for t in train]))))
                b_avg = float(np.exp(np.mean(np.log([t["beta"] for t in train]))))
                for vf, p in d["tests"]:
                    u = np.clip((vf - law.r0) / (1 - law.r0), 0, 1.2)
                    errs[methods[0]].append(cents_err(p, f0_avg / np.sqrt(1 + b_avg * u ** law.n)))
                    errs[methods[1]].append(cents_err(p, law.pitch(d["shape"], vf)))
                vf0, p0 = d["tests"][0]                      # one check tap sets this container's offset
                off = cents_err(p0, law.pitch(d["shape"], vf0))
                errs[methods[2]] += [cents_err(p, law.pitch(d["shape"], vf, off)) for vf, p in d["tests"][1:]]
            if example is None and spread == 0.03:
                full = G.FamilyLaw.fit([d["shape"] for d in data], [d["f0"] for d in data], [d["beta"] for d in data])
                example = {"C_f": full.C_f, "C_b": full.C_b,
                           "radius_exponent": G.radius_exponent([d["shape"] for d in data], [d["f0"] for d in data]),
                           "containers": [{**d["shape"].to_dict(), "f0": d["f0"], "beta": d["beta"],
                                           "true_f0": d["bowl"].f0, "loo": list(_loo_law(data, k)[0].predict(d["shape"])),
                                           "tests": [list(t) for t in d["tests"]]} for k, d in enumerate(data)]}
        out[f"{100 * spread:.0f}% wall-thickness spread"] = {m: float(np.median(np.abs(v))) for m, v in errs.items()}
    return out, example


def e10_level_from_sound(rng, n=6, thick_spread=0.03, fills=tuple(np.round(np.linspace(0.0, 0.9, 19), 3))):
    """An opaque container: the camera sees its outline but not the water. One tap on the EMPTY container
    gives f0; the family law gives beta from the shape; every later tap's pitch then gives the water level.

    Simulated strikes are cleaner than real ones, so each tap's pitch gets a random jitter of
    C.PITCH_NOISE_CENTS (8 cents), the spread of two real strikes on one glass (E1, ObjectFolder #91).
    """
    data = _family_data(rng, n, thick_spread)

    def heard(b, fill):
        an = A.analyze_strike(SIM.synth_strike(b, fill, snr_db=40, seed=int(rng.integers(1e9))))
        return an.pitch * 2 ** (rng.normal(0, C.PITCH_NOISE_CENTS) / 1200) if an else np.nan

    pts = []
    for k, d in enumerate(data):
        law, _ = _loo_law(data, k)
        _, beta = law.predict(d["shape"])
        b = d["bowl"]
        f_empty = heard(b, 0.0)
        for lv in fills:
            p = heard(b, float(lv))
            if not (np.isfinite(p) and np.isfinite(f_empty)):
                continue
            pts.append([float(SIM.visible_fill(b, float(lv))), G.fill_from_pitch(p, f_empty, beta, law.n, law.r0), float(beta)])
    p = np.array(pts)
    err = np.abs(p[:, 1] - p[:, 0])
    hi, lo = p[:, 0] >= 0.35, p[:, 0] < 0.3
    return {"fill_mae_above_35%": float(np.median(err[hi])), "fill_mae_below_30%": float(np.median(err[lo])),
            "fill_p90_above_35%": float(np.percentile(err[hi], 90)), "beta_median": float(np.median(p[:, 2])),
            "points": p[:, :2].round(4).tolist()}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="your recorded session folder (real data)")
    ap.add_argument("--objectfolder", default=str(C.DATA_DIR / "objectfolder"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    q = a.quick
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    M, md = {}, ["# TuneSight results\n", f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n"]
    t0 = time.time()

    print("E1 audio model (simulated ground truth) ...", flush=True)
    M["E1_audio_sim"] = e1_synthetic(rng, 45 if q else 120)
    md += ["\n## E1 Audio model (SIMULATED ground truth)\n",
           "| SNR (dB) | modes found | freq error (median cents) | decay error STFT (median %) | decay error LPC (median %) |",
           "|---|---|---|---|---|"]
    for snr, r in M["E1_audio_sim"].items():
        md.append(f"| {snr} | {r['mode_detection_%']:.0f}% | {r['freq_err_median_cents']:.2f} | "
                  f"{r['decay_err_stft_median_%']:.1f} | {r['decay_err_lpc_median_%']:.1f} |")
    b = SIM.SimBowl(x_cm=10)
    xs = SIM.synth_strike(b, 0.4, seed=3)
    md.append(f"\n![]({plot_strike(xs, A.analyze_strike(xs), 'SIMULATED glass strike', 'e1_sim_strike.png')})\n")

    of_root = Path(a.objectfolder)
    print("E1 audio model (real ObjectFolder recordings, if downloaded) ...", flush=True)
    summ, figs = e1_real(of_root)
    if summ:
        M["E1_audio_real"] = summ
        md += ["\n## E1 Audio model on REAL recordings (ObjectFolder Real)\n",
               f"Pitch is searched in the app's band ({C.FMIN:.0f} Hz to {C.FMAX / 1000:.0f} kHz). Searching from 60 Hz instead, "
               "one glass #91 recording also shows a 79 Hz line (stand or room) only 6 dB below the glass's main mode, "
               "and a lowest-line pitch picker takes it as the pitch.\n",
               "| # | object | material | strikes | pitch (Hz) | strikes agreeing on pitch | LPC vs FFT freq (cents) | LPC/STFT decay ratio | resynthesis spectral corr. |",
               "|---|---|---|---|---|---|---|---|---|"]
        for obj, r in sorted(summ.items()):
            md.append(f"| {obj} | {r['name']} | {r['material']} | {r['strikes']} | {r['pitch_median_hz']:.1f} | "
                      f"{r['pitch_mode_share_%']:.0f}% | {r['lpc_vs_fft_freq_median_cents'] or float('nan'):.2f} | "
                      f"{r['lpc_vs_stft_decay_median_ratio'] or float('nan'):.2f} | {r['resynth_spectral_corr_median']:.2f} |")
        md += [f"\n![]({f})\n" for f in figs]
        sc, fig = e2_scaling(of_root, summ)
        if sc:
            M["E2_scaling_real"] = sc
            md += ["\n## E2 Scaling law on REAL objects\n", "| family | slope (expect -1) | R^2 |", "|---|---|---|"]
            md += [f"| {k} | {v['slope']:.2f} | {v['r2']:.2f} |" for k, v in sc.items()]
            md.append(f"\n![]({fig})\n")
        else:
            md.append("\n## E2 Scaling law\n_skipped: needs ObjectFolder size-family audio chunks and meshes "
                      "(`python scripts/download_objectfolder.py --chunks 61_70 41_50 --meshes`)._\n")
    else:
        md.append("\n## E1/E2 on REAL ObjectFolder recordings\n_skipped: no recordings found in data/objectfolder._\n")

    print("E3 water-loading law ...", flush=True)
    fit, fig = e3_french_sim(rng, 8 if q else 12, 1 if q else 2)
    M["E3_french_sim"] = fit
    md += ["\n## E3 Water-loading law\n", f"SIMULATED: fitted exponent n = {fit['n']:.2f} +- {fit.get('n_std', float('nan')):.2f} "
           f"(true 4.0), base offset r0 = {fit['r0']:.3f}, rmse {fit['rmse_cents']:.1f} cents\n",
           f"Bayesian model evidence (online calibrator) selects n = {fit['bayes_n']:.2f}, r0 = {fit['bayes_r0']:.2f} "
           f"(true n = 4; glass base offset about 0.05)\n", f"\n![]({fig})\n"]
    taps = None
    if a.session:
        print("  measuring your session (real data) ...", flush=True)
        taps = S.measure_taps(S.load_session(Path(a.session)))
        fr, fig = e3_french_real(taps)
        if fr:
            M["E3_french_real"] = fr
            md += ["REAL (your bowls):\n", "| bowl | f0 (Hz) | beta | n | rmse (cents) |", "|---|---|---|---|---|"]
            md += [f"| {k + 1} | {v['f0']:.1f} | {v['beta']:.2f} | {v['n']:.2f} | {v['rmse_cents']:.1f} |" for k, v in fr.items()]
            md.append(f"\n![]({fig})\n")
        real = [t for t in taps if t.true_h_cm is not None and t.H_cm and np.isfinite(t.fill)]
        if real:
            err = [abs(t.fill * t.H_cm - t.true_h_cm) for t in real]
            M["E4_vision_real_mae_cm"] = float(np.mean(err))
            md.append(f"\nREAL vision check: water height error vs ruler = {np.mean(err):.2f} cm (mean, {len(err)} taps)\n")

    print("E4 vision robustness ...", flush=True)
    M["E4_vision_sim"] = e4_vision(rng, 5 if q else 12)
    md += ["\n## E4 Vision: water level under degradations (SIMULATED)\n", "| condition | level found | fill-ratio MAE |", "|---|---|---|"]
    for k, v in M["E4_vision_sim"].items():
        md.append(f"| {k} | {v['level_found_%']:.0f}% | {v['fill_mae'] if v['fill_mae'] is None else round(v['fill_mae'], 3)} |")

    print("E5 learning curve ...", flush=True)
    lc, fig = e5_learning_curve(rng, taps, 10 if q else 12, 10 if q else 30)
    M["E5_learning_curve"] = lc
    md += ["\n## E5 Audio teaches vision: camera-only pitch prediction vs calibration taps\n",
           "| taps | " + " | ".join(lc) + " |", "|---|" + "---|" * len(lc)]
    ks = sorted(next(iter(lc.values())))
    for k in ks:
        md.append(f"| {k} | " + " | ".join(f"{lc[n].get(k, float('nan')):.1f} c" for n in lc) + " |")
    md.append(f"\n![]({fig})\n")

    print("E6 which bowl was struck ...", flush=True)
    wb, fig = e6_which_bowl(rng, 12 if q else 40)
    M["E6_which_bowl_sim"] = wb
    md += ["\n## E6 Which identical-looking bowl was struck? (SIMULATED, 5 glasses 9.5 cm apart)\n",
           "| condition | chance | motion only | audio only (physics) | fused |", "|---|---|---|---|---|"]
    for k, v in wb.items():
        md.append(f"| {k} | {v['chance']:.0f}% | {v['motion only']:.0f}% | {v['audio only (physics)']:.0f}% | {v['fused']:.0f}% |")
    md.append(f"\n![]({fig})\n")

    print("E7 closed-loop tuning ...", flush=True)
    tu, fig = e7_study(seeds=(0, 1) if q else (0, 1, 2))
    M["E7_tuning_sim"] = tu
    md += ["\n## E7 Tune by sight (SIMULATED), raga Bhupali, 5 bowls\n",
           "| bowls | vision only: mean / max abs error | vision + 1 tap: mean / max abs error |", "|---|---|---|"]
    md += [f"| {k} | {v['vision_only_mean_abs']:.1f} / {v['vision_only_max_abs']:.1f} cents | "
           f"{v['one_tap_mean_abs']:.1f} / {v['one_tap_max_abs']:.1f} cents |" for k, v in tu.items()]
    first = next(iter(tu.values()))["runs"][0]
    md += [f"\nExample run (near-identical bowls), Sa = {first['sa_hz']:.1f} Hz:\n",
           "| note | target (Hz) | vision only (cents) | vision + 1 tap (cents) |", "|---|---|---|---|"]
    md += [f"| {r['note']} | {r['target_hz']:.1f} | {r['vision_only_cents']:+.1f} | {r['with_one_tap_cents']:+.1f} |" for r in first["notes"]]
    md.append(f"\n![]({fig})\n")

    # E8-E10 use their own random stream, so E1-E7 above stay exactly reproducible
    rg = np.random.default_rng(a.seed + 8)
    print("E8 container shape from the camera ...", flush=True)
    M["E8_shape_sim"] = e8_shape(rg, 10 if q else 30)
    md += ["\n## E8 Container shape from the camera (SIMULATED, random glasses: radius 2.5-4.5 cm, height 7-12 cm)\n",
           "| camera | found | mean radius error | rim radius error | height error | capacity error |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['found_%']:.0f}% | {v['mean_radius_mae_mm']:.2f} mm | {v['rim_radius_mae_mm']:.2f} mm | "
           f"{v['height_mae_mm']:.1f} mm | {v['capacity_mape_%']:.1f} % |" for k, v in M["E8_shape_sim"].items()]

    print("E9 a new container from its shape ...", flush=True)
    nc, ex = e9_new_container(rg, n=6 if q else 8, reps=1 if q else 2)
    M["E9_new_container_sim"] = nc
    M["E9_family_example"] = ex
    md += ["\n## E9 A new container's pitch from its shape (SIMULATED, leave one container out)\n",
           "French's thin-wall model: empty pitch ~ 1/R^2, water effect ~ R. Two family constants are fitted on the other "
           f"containers; fitted radius exponent {ex['radius_exponent']:.2f} (theory -2).\n",
           "| family | " + " | ".join(next(iter(nc.values()))) + " |", "|---|---|---|---|"]
    md += [f"| {k} | " + " | ".join(f"{e:.1f} cents" for e in v.values()) + " |" for k, v in nc.items()]

    print("E10 water level of an opaque container, by sound ...", flush=True)
    M["E10_level_from_sound_sim"] = e10_level_from_sound(rg, n=5 if q else 6)
    lv = M["E10_level_from_sound_sim"]
    md += ["\n## E10 Hearing the water level of an opaque container (SIMULATED)\n",
           "One tap on the empty container gives f0, the shape gives beta, then each tap's pitch gives the level.\n",
           f"Median level error: {100 * lv['fill_mae_above_35%']:.1f} % of the height above 35 % full "
           f"(90th percentile {100 * lv['fill_p90_above_35%']:.1f} %); {100 * lv['fill_mae_below_30%']:.0f} % below 30 % full, "
           "where the law is flat and the pitch barely changes.\n"]

    json.dump(M, open(C.RESULTS_DIR / "metrics.json", "w"), indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    (C.RESULTS_DIR / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {C.RESULTS_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()

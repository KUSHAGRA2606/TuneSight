"""Automated checks for every TuneSight module. Run:  python -m pytest -q"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunesight import audio as A          # noqa: E402
from tunesight import fusion as F         # noqa: E402
from tunesight import geometry as G       # noqa: E402
from tunesight import music               # noqa: E402
from tunesight import physics as P        # noqa: E402
from tunesight import simulate as SIM     # noqa: E402
from tunesight import tuning as T         # noqa: E402
from tunesight import vision as V         # noqa: E402

SR = 48000


def cents(a, b):
    return 1200 * math.log2(a / b)


# ----------------------------------------------------------------------------- audio
def test_pitch_is_exact_on_a_simulated_strike():
    b = SIM.SimBowl(x_cm=10)
    for fill in (0.0, 0.5, 0.85):
        an = A.analyze_strike(SIM.synth_strike(b, fill, seed=1))
        assert an is not None
        assert abs(cents(an.pitch, b.modes(fill)[0][0])) < 1.0


def test_covariance_lpc_recovers_modes_exactly_without_noise():
    fr, sg = np.array([500.0, 1300.0, 2700.0]), np.array([3.0, 10.0, 25.0])
    x = A.modal_synthesis(fr, sg, [1, 0.5, 0.3], SR, 1.0, seed=1)
    L = A.lpc_modes(x, SR)
    for f, s in zip(fr, sg):
        j = int(np.argmin(np.abs(L.freqs - f)))
        assert abs(L.freqs[j] - f) < 0.5
        assert abs(L.sigmas[j] / s - 1) < 0.05


def test_hum_and_fan_tones_are_not_mistaken_for_the_bowl():
    b = SIM.SimBowl(x_cm=10)
    y = SIM.synth_strike(b, 0.5, seed=2)
    t = np.arange(len(y)) / SR
    y = y + 0.05 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 700 * t)
    an = A.analyze_strike(y)
    assert abs(cents(an.pitch, b.modes(0.5)[0][0])) < 1.0
    assert not np.any(np.abs(an.freqs - 700) < 5)


def test_onset_detection_finds_each_strike():
    b = SIM.SimBowl(x_cm=10)
    x = np.concatenate([SIM.synth_strike(b, 0.2, seed=i, onset_s=0.2) for i in range(3)])
    ons = A.detect_onsets(x, SR)
    assert len(ons) == 3


def test_streaming_onset_detector_matches_offline():
    b = SIM.SimBowl(x_cm=10)
    x = SIM.synth_strike(b, 0.3, seed=4, onset_s=0.5)
    det = A.StreamingOnsetDetector(SR)
    found = []
    for k in range(0, len(x), 1024):
        found += det.process(x[k:k + 1024])
    assert len(found) == 1 and abs(found[0] / SR - 0.5) < 0.02


# ----------------------------------------------------------------------------- vision
def test_vision_finds_all_bowls_and_their_water_levels():
    bowls = SIM.default_bowls(5)
    fills = [0.0, 0.2, 0.45, 0.7, 0.9]
    obs = V.measure_frame(SIM.render_scene(bowls, fills, seed=0))
    assert len(obs.bowls) == 5
    assert obs.ppc is not None and obs.ppc > 0
    for bo, b, f in zip(obs.bowls, bowls, fills):
        assert abs(bo.fill_ratio - SIM.visible_fill(b, f)) < 0.02


def test_clear_water_is_detected_by_the_edge_fallback():
    b = SIM.SimBowl(x_cm=12)
    obs = V.measure_frame(SIM.render_scene([b], [0.5], seed=1, clear_water=True))
    assert abs(obs.bowls[0].fill_ratio - SIM.visible_fill(b, 0.5)) < 0.03


def test_striker_tip_points_to_the_struck_bowl():
    import cv2
    w = SIM.SimWorld(SIM.default_bowls(5))
    boxes = [b.bbox for b in V.measure_frame(w.frame(0.0)).bowls]
    for tgt in range(5):
        w._stick = None
        t0 = 10.0 * (tgt + 1)
        w.strike(tgt, t0)
        w._audio_queue = []
        frames = []
        for t in np.arange(t0 - 0.5, t0 + 0.6, 1 / 15):
            img = w.frame(t)
            s = V.C.PROC_WIDTH / img.shape[1]
            frames.append((t, cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY), 1.0))
        p, conf = V.motion_evidence(frames, boxes, t0 + 0.2)
        assert int(np.argmax(p)) == tgt and conf > 0


# ----------------------------------------------------------------------------- physics
def test_french_law_inverse_is_consistent():
    for h in (0.2, 0.5, 0.8):
        f = float(P.french_freq(h, 1000.0, 3.0))
        assert abs(P.french_height(f, 1000.0, 3.0) - h) < 1e-9


def test_calibrator_learns_the_law_and_predicts_pitch_from_vision_alone():
    rng = np.random.default_rng(0)
    b = SIM.SimBowl(x_cm=12)
    cal = P.FrenchCalibrator(f0_prior=1150, f0_prior_cents=200)
    for lv in np.linspace(0, 0.9, 8):
        obs = V.measure_frame(SIM.render_scene([b], [lv], seed=int(rng.integers(1e9))))
        cal.add(obs.bowls[0].fill_ratio, A.analyze_strike(SIM.synth_strike(b, lv, seed=int(rng.integers(1e9)))).pitch)
    assert 3.5 <= cal.n <= 4.5                       # evidence recovers French's exponent
    for lv in (0.3, 0.6, 0.8):
        obs = V.measure_frame(SIM.render_scene([b], [lv], seed=7))
        f, _ = cal.predict(obs.bowls[0].fill_ratio)
        assert abs(cents(f, float(b.modes(lv)[0][0]))) < 15


def test_offset_from_one_tap_shifts_predictions():
    cal = P.FrenchCalibrator(f0_prior=1000, f0_prior_cents=50)
    f_before, _ = cal.predict(0.5)
    cal.set_offset_from_tap(0.5, f_before * 2 ** (30 / 1200))
    f_after, _ = cal.predict(0.5)
    assert abs(cents(f_after, f_before) - 30) < 1e-6
    assert abs(cal.height_for(f_after) - 0.5) < 1e-6


# ----------------------------------------------------------------------------- fusion
def test_identification_uses_pitch_and_rejects_foreign_sounds():
    pred = [1000.0, 900.0, 800.0]
    ident = F.identify_struck(901.0, pred, [5, 5, 5])
    assert ident.best == 1
    ident = F.identify_struck(1400.0, pred, [5, 5, 5])
    assert ident.best == -1                             # "none of the visible bowls"


def test_motion_breaks_a_pitch_tie():
    pred = [1000.0, 1001.0, 800.0]                      # bowls 0 and 1 sound the same
    ident = F.identify_struck(1000.5, pred, [5, 5, 5], motion_p=np.array([0.05, 0.9, 0.05]), motion_conf=1.0)
    assert ident.best == 1


# ----------------------------------------------------------------------------- music and tuning
def test_nearest_swara():
    note, dev = music.nearest_swara(440.0 * 1.5, 440.0)
    assert note.name == "Pa" and abs(dev) < 1e-6


def test_tuning_plan_puts_every_note_inside_its_bowl_range():
    cal = P.FrenchCalibrator(f0_prior=1100, f0_prior_cents=5, beta_prior=3.0, beta_prior_std=0.01, learn_shape=False)
    sa, plan = T.plan([cal] * 5, "Bhupali")
    assert all(p.reachable for p in plan)
    assert sorted(p.target_hz for p in plan) == [p.target_hz for p in sorted(plan, key=lambda q: q.target_fill, reverse=True)]


def test_pour_advice_says_stop_at_target():
    assert T.pour_advice(0.5, 0.505).startswith("STOP")
    assert "pour" in T.pour_advice(0.3, 0.5)


# ----------------------------------------------------------------------------- shape and any container
def test_camera_measures_radius_height_and_capacity_of_a_wide_glass():
    b = SIM.family_bowl(12.0, R_mean_cm=4.2, H_cm=8.5, taper=0.8)       # wider than tall: once mistaken for the marker
    obs = V.measure_frame(SIM.render_scene([b], [0.5], seed=4))
    assert len(obs.bowls) == 1
    s = G.shape_from_obs(obs.bowls[0])
    assert abs(s.R_mean_cm - 4.2) < 0.05                                # within half a millimetre
    assert abs(s.H_cm - 8.5) < 0.6
    cap = G.frustum_capacity_ml(b.D_top_cm / 2, b.D_bot_cm / 2, b.H_cm)
    assert abs(s.capacity_ml / cap - 1) < 0.10


def test_family_law_predicts_a_new_glass_from_its_shape():
    shapes = [G.Shape(H, 2 * R / 1.8, 0.8 * 2 * R / 1.8, 0.0) for R, H in ((2.6, 8), (3.2, 10), (3.9, 9), (4.4, 11.5))]
    f0s = [G.french_f0(s.R_mean_cm, s.H_cm, 1.5) for s in shapes]
    betas = [G.french_beta(s.R_mean_cm, 1.5) for s in shapes]
    law = G.FamilyLaw.fit(shapes[:3], f0s[:3], betas[:3])              # the fourth glass is new
    f0, beta = law.predict(shapes[3])
    assert abs(cents(f0, f0s[3])) < 1 and abs(beta / betas[3] - 1) < 1e-6
    assert abs(G.radius_exponent(shapes, f0s) + 2) < 0.05               # French: f0 ~ R^-2


def test_level_from_pitch_inverts_the_law_and_reads_empty_as_zero():
    f0, beta = 900.0, 3.0
    for r in (0.4, 0.6, 0.85):
        u = (r - G.R0_FAMILY) / (1 - G.R0_FAMILY)
        assert abs(G.fill_from_pitch(f0 / math.sqrt(1 + beta * u ** 4), f0, beta) - r) < 1e-9
    assert G.fill_from_pitch(f0 * 1.001, f0, beta) == 0.0
    assert G.cents_per_percent_fill(0.8, beta) > 10 * G.cents_per_percent_fill(0.2, beta)   # flat near empty

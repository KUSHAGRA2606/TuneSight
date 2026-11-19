"""TuneSight explainer animations, written for ManimGL (3Blue1Brown's manim, https://github.com/3b1b/manim).

Render all scenes with:   python animations/render.py
Render one scene with:    manimgl animations/scenes.py WaterLoading -w -m

Only `Text` is used (no LaTeX), so no TeX/dvisvgm installation is needed.
Frame: 14.2 x 8 units, origin at the centre.
"""
from manimlib import *  # noqa: F401,F403
import numpy as np

# ------------------------------------------------------------------ style
BG = "#0B1D36"           # deep blue, matches the demo web page
INK = "#E8EEF6"
MUTED = "#9DB0C8"
WATER = "#F0A030"        # haldi / orange food colour
GLASS = "#A9C4E0"
BOWL_COLS = ["#5DA5DA", "#FAA43A", "#60BD68", "#F15854", "#B276B2"]
AUDIO_C, MOTION_C, FUSED_C = "#5DA5DA", "#FAA43A", "#60BD68"
ACCENT = "#FFD166"
FONT = "Segoe UI"
F0, BETA, N = 1100.0, 3.0, 4.0


def french(h, f0=F0, beta=BETA, n=N):
    return f0 / np.sqrt(1 + beta * np.clip(h, 0, 1) ** n)


def T(s, size=28, color=INK, **kw):
    return Text(s, font_size=size, color=color, font=FONT, **kw)


def glass(width=1.2, height=1.8, fill=0.0, color=GLASS):
    """A glass tumbler (outline) with water filled to `fill` of its height; origin at the base centre."""
    wb, wt = width * 0.8 / 2, width / 2
    outline = Polygon([-wb, 0, 0], [wb, 0, 0], [wt, height, 0], [-wt, height, 0])
    outline.set_stroke(color, 3).set_fill(color, 0.08)
    grp = VGroup(outline)
    if fill > 0.01:
        h = fill * height
        ww = wb + (wt - wb) * fill
        water = Polygon([-wb + 0.05, 0.05, 0], [wb - 0.05, 0.05, 0], [ww - 0.05, h, 0], [-ww + 0.05, h, 0])
        water.set_stroke(WATER, 0).set_fill(WATER, 0.8)
        grp.add_to_back(water)
    return grp


def axes(x_range, y_range, width, height):
    return Axes(x_range, y_range, width=width, height=height,
                axis_config=dict(stroke_color=MUTED, stroke_width=2, include_tip=False))


class Base(Scene):
    default_camera_config = dict(background_color=BG)

    def title(self, s):
        t = T(s, 36).to_edge(UP, buff=0.3)
        self.play(Write(t), run_time=1)
        return t

    def clear_all(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)


# ================================================================== 1. source-filter
class SourceFilter(Base):
    def construct(self):
        self.title("1. A struck bowl is a source-filter system")
        g = glass(1.4, 2.0, 0.45).move_to(LEFT * 5.6 + UP * 0.2)
        stick = Line(ORIGIN, RIGHT * 1.2 + UP * 1.2, stroke_width=9, color="#C08040")
        stick.move_to(g.get_corner(UR) + RIGHT * 1.0 + UP * 1.0)
        self.play(FadeIn(g, shift=UP * 0.3))
        self.play(stick.animate.move_to(g.get_corner(UR) + RIGHT * 0.5 + UP * 0.5), run_time=0.5)
        self.play(Flash(g.get_corner(UR), color=ACCENT, flash_radius=0.45), run_time=0.6)
        src = VGroup(T("source", 24, ACCENT), T("a 2 ms tap", 20, MUTED)).arrange(DOWN, buff=0.08)
        flt = VGroup(T("filter", 24, GLASS), T("the bowl rings in modes", 20, MUTED)).arrange(DOWN, buff=0.08)
        VGroup(src, flt).arrange(DOWN, buff=0.6).move_to(LEFT * 5.6 + DOWN * 2.4)
        arr = Arrow(src.get_bottom(), flt.get_top(), buff=0.08, stroke_width=4)
        self.play(FadeIn(src), GrowArrow(arr), FadeIn(flt))

        ax = axes((0, 1, 0.25), (-1.2, 1.2, 0.5), 8.0, 2.3).move_to(RIGHT * 1.6 + UP * 1.55)
        modes = [(4.0, 3.0, 0.55), (10.5, 6.0, 0.30), (19.0, 11.0, 0.15)]   # visual (f, sigma, amp)
        cols = [BLUE_B, GREEN_B, RED_B]

        def mode(t, f, s, a):
            return a * np.exp(-s * t) * np.sin(2 * PI * f * t)

        comps = VGroup(*[ax.get_graph(lambda t, f=f, s=s, a=a: mode(t, f, s, a) * 1.6, x_range=(0, 1, 0.002),
                                      color=c, stroke_width=3) for (f, s, a), c in zip(modes, cols)])
        total = ax.get_graph(lambda t: sum(mode(t, *m) for m in modes) * 1.4, x_range=(0, 1, 0.002),
                             color=ACCENT, stroke_width=3.5)
        lab_t = T("time", 20, MUTED).next_to(ax.x_axis, DOWN, buff=0.1).align_to(ax, RIGHT)
        self.play(ShowCreation(ax), FadeIn(lab_t))
        eq1 = T("three modes, each a decaying sinusoid", 22, MUTED).next_to(ax, UP, buff=0.12)
        self.play(FadeIn(eq1), LaggedStart(*[ShowCreation(c) for c in comps], lag_ratio=0.4), run_time=2.5)
        eq2 = T("the bowl's sound = their sum:  y(t) = sum  a_k e^(-sigma_k t) sin(2 pi f_k t)", 22, ACCENT).move_to(eq1)
        self.play(ReplacementTransform(eq1, eq2), ReplacementTransform(comps, total), run_time=1.8)
        self.wait(0.4)

        sp = axes((0, 25, 5), (0, 1.1, 0.5), 4.6, 2.1).move_to(RIGHT * 0.0 + DOWN * 2.0)

        def lorentz(f):
            return sum(a / (1 + ((f - fk) / (s / 2.0)) ** 2) for fk, s, a in modes) / 0.55

        spec = sp.get_graph(lorentz, x_range=(0.2, 25, 0.02), color=ACCENT, stroke_width=3)
        spl = T("spectrum: one peak per mode", 22, INK).next_to(sp, UP, buff=0.12)
        fl = T("frequency", 18, MUTED).next_to(sp.x_axis, DOWN, buff=0.1).align_to(sp, RIGHT)
        self.play(ShowCreation(sp), FadeIn(spl), FadeIn(fl), ShowCreation(spec), run_time=1.5)

        circ = Circle(radius=1.15).set_stroke(MUTED, 2).move_to(RIGHT * 4.9 + DOWN * 2.05)
        cen = circ.get_center()
        axes_z = VGroup(Line(cen + LEFT * 1.4, cen + RIGHT * 1.4, stroke_width=1, color=GREY_D),
                        Line(cen + DOWN * 1.4, cen + UP * 1.4, stroke_width=1, color=GREY_D))
        r_tr = ValueTracker(0.93)

        def poles():
            g_ = VGroup()
            for (f, s, a), c in zip(modes, cols):
                ang = 2 * PI * f / 60
                rr = r_tr.get_value() - 0.012 * s / 3
                for sgn in (1, -1):
                    g_.add(T("x", 24, c).move_to(cen + 1.15 * rr * np.array([np.cos(ang), sgn * np.sin(ang), 0])))
            return g_

        pz = always_redraw(poles)
        zl = VGroup(T("LPC poles (z-plane)", 22, INK), T("angle = pitch,  radius = ringing", 18, MUTED)).arrange(DOWN, buff=0.06)
        zl.next_to(circ, UP, buff=0.12)
        self.play(ShowCreation(circ), FadeIn(axes_z), FadeIn(pz), FadeIn(zl))
        self.wait(0.4)
        note = T("pole moves inward  ->  that mode dies faster", 22, RED_B).to_edge(DOWN, buff=0.15).shift(RIGHT * 2.4)
        self.play(FadeIn(note), r_tr.animate.set_value(0.70), run_time=2.0)
        self.play(r_tr.animate.set_value(0.93), run_time=1.2)
        self.wait(1.5)
        self.clear_all()


# ================================================================== 2. water loading + learning the law
class WaterLoading(Base):
    def construct(self):
        self.title("2. More water, lower pitch: the camera can predict the note")
        h = ValueTracker(0.0)
        gl = always_redraw(lambda: glass(1.6, 2.6, h.get_value()).move_to(LEFT * 5.3 + DOWN * 2.9, aligned_edge=DOWN))
        self.play(FadeIn(gl))
        ax = axes((0, 1, 0.2), (0, 6, 1), 7.0, 3.6).move_to(RIGHT * 1.4 + DOWN * 0.75)
        yv = lambda f: (f - 600.0) / 100.0          # plot pitch as hundreds of Hz above 600 Hz
        xl = T("fill ratio h/H  (seen by the camera)", 20, MUTED).next_to(ax, DOWN, buff=0.45)
        yl = T("pitch (Hz), heard by the mic", 20, MUTED).rotate(PI / 2).next_to(ax, LEFT, buff=0.75)
        ticks = VGroup(*[T(str(v), 16, MUTED).next_to(ax.c2p(0, yv(v)), LEFT, buff=0.12) for v in (600, 800, 1000, 1200)],
                       *[T(f"{v:.1f}", 16, MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.12) for v in (0.0, 0.5, 1.0)])
        curve = ax.get_graph(lambda x: yv(french(x)), x_range=(0, 0.95, 0.005), color=ACCENT)
        law = T("f = f0 / sqrt(1 + beta (h/H)^4)      A. P. French, 1983", 24, ACCENT).next_to(ax, UP, buff=0.25)
        self.play(ShowCreation(ax), FadeIn(xl), FadeIn(yl), FadeIn(ticks), FadeIn(law))
        dot = always_redraw(lambda: Dot(ax.c2p(h.get_value(), yv(french(h.get_value()))), color=WATER, radius=0.1))
        read = always_redraw(lambda: T(f"{french(h.get_value()):.0f} Hz", 30, WATER).move_to(LEFT * 5.3 + UP * 0.6))
        self.add(dot, read)
        self.play(ShowCreation(curve), h.animate.set_value(0.92), run_time=5, rate_func=linear)
        self.wait(0.6)

        rng = np.random.default_rng(3)
        hs = np.sort(rng.uniform(0.05, 0.9, 8))
        fs = french(hs) * 2 ** (rng.normal(0, 4, 8) / 1200)
        pts = VGroup(*[Dot(ax.c2p(a, yv(b)), color=BLUE_B, radius=0.08) for a, b in zip(hs, fs)])
        tap = T("each tap = one (camera fill, heard pitch) pair", 24, BLUE_B).to_edge(DOWN, buff=0.2)
        self.play(FadeOut(dot), FadeOut(read), LaggedStart(*[GrowFromCenter(p) for p in pts], lag_ratio=0.2),
                  FadeOut(xl), FadeIn(tap), run_time=2)
        self.wait(0.4)

        ax2 = axes((0, 0.7, 0.1), (0.0, 2.8, 0.5), 7.0, 3.6).move_to(ax)
        a_, b_ = 1e6 / F0 ** 2, 1e6 * BETA / F0 ** 2
        pts2 = VGroup(*[Dot(ax2.c2p(hh ** 4, 1e6 / ff ** 2), color=BLUE_B, radius=0.08) for hh, ff in zip(hs, fs)])
        line = ax2.get_graph(lambda x: a_ + b_ * x, x_range=(0, 0.68), color=RED_B)
        yl2 = T("1e6 / f^2", 20, MUTED).rotate(PI / 2).next_to(ax2, LEFT, buff=0.4)
        law2 = T("1/f^2 = a + b u^n  :  a straight line, so plain Bayesian linear regression", 24, RED_B).move_to(law)
        self.play(FadeOut(ticks), FadeOut(curve), ReplacementTransform(ax, ax2), ReplacementTransform(yl, yl2),
                  ReplacementTransform(pts, pts2), ReplacementTransform(law, law2), run_time=2.2)
        self.play(ShowCreation(line), run_time=1.2)
        res = T("model evidence picks n = 4;  after ~5 taps the camera predicts pitch within a few cents", 22, GREEN_B).to_edge(DOWN, buff=0.2)
        self.play(ReplacementTransform(tap, res))
        self.wait(2)
        self.clear_all()


# ================================================================== 3. which identical bowl was struck?
class WhichBowl(Base):
    def bars(self, probs, color, origin, name, w=0.28, gap=0.38, hmax=1.5):
        grp = VGroup()
        base = Line(origin + LEFT * 0.15, origin + RIGHT * (gap * len(probs) + 0.05), stroke_width=1.5, color=MUTED)
        for i, p in enumerate(probs):
            r = Rectangle(width=w, height=max(0.02, hmax * p)).set_fill(color, 0.95).set_stroke(width=0)
            r.move_to(origin + RIGHT * (0.15 + gap * i), aligned_edge=DOWN)
            grp.add(r)
        lab = T(name, 20, color).next_to(base, DOWN, buff=0.12)
        return VGroup(base, grp, lab)

    def construct(self):
        self.title("3. Five identical glasses: which one rang?")
        fills = [0.85, 0.78, 0.70, 0.55, 0.40]
        notes = ["Sa", "Re", "Ga", "Pa", "Dha"]
        glasses = VGroup(*[glass(0.95, 1.45, f) for f in fills]).arrange(RIGHT, buff=0.55).move_to(LEFT * 3.4 + DOWN * 2.45)
        labels = VGroup(*[T(n_, 22, c).next_to(g, DOWN, buff=0.12) for n_, g, c in zip(notes, glasses, BOWL_COLS)])
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.2) for g in glasses], lag_ratio=0.15), FadeIn(labels))
        cam = T("camera: water level  ->  predicted pitch of every glass", 22, MUTED).next_to(glasses, UP, buff=0.3)
        self.play(FadeIn(cam))

        fax = axes((0, 5.5, 1), (0, 1, 0.5), 7.4, 1.7).move_to(LEFT * 3.0 + UP * 1.2)
        xv = lambda f: (f - 600.0) / 100.0
        preds = [french(f) for f in fills]

        def bump(p, c):
            return fax.get_graph(lambda x: np.exp(-0.5 * ((1200 * np.log2((600 + 100 * x) / p)) / 14) ** 2), x_range=(0, 5.5, 0.01), color=c)

        bumps = VGroup(*[bump(p, c) for p, c in zip(preds, BOWL_COLS)])
        fl = T("predicted pitch  +-  uncertainty", 20, MUTED).next_to(fax, UP, buff=0.1).align_to(fax, LEFT)
        fticks = VGroup(*[T(f"{v} Hz", 16, MUTED).next_to(fax.c2p(xv(v), 0), DOWN, buff=0.1) for v in (600, 800, 1000)])
        self.play(ShowCreation(fax), FadeIn(fl), FadeIn(fticks), LaggedStart(*[ShowCreation(b) for b in bumps], lag_ratio=0.15), run_time=2)

        def post(f_heard, sig=14):
            d = np.array([1200 * np.log2(f_heard / p) for p in preds])
            p = np.exp(-0.5 * (d / sig) ** 2)
            return p / p.sum()

        # case A: audio alone decides
        heard = preds[2] * 2 ** (3 / 1200)
        hl = DashedLine(fax.c2p(xv(heard), 0), fax.c2p(xv(heard), 1.05), color=INK)
        ht = T("one strike heard", 20).next_to(hl.get_end(), RIGHT, buff=0.12)
        self.play(ShowCreation(hl), FadeIn(ht))
        A1 = self.bars(post(heard), AUDIO_C, RIGHT * 2.0 + UP * 0.4, "audio (physics)")
        self.play(FadeIn(A1, shift=UP * 0.2))
        self.play(FlashAround(glasses[2], color=ACCENT), run_time=1.0)
        self.wait(0.4)
        self.play(FadeOut(hl), FadeOut(ht), FadeOut(A1))

        # case B: two glasses with the same water -> same pitch -> motion breaks the tie
        tie = T("now Pa and Dha hold the same water: they sound the same", 22, RED_B).to_edge(DOWN, buff=0.12)
        new_g = glass(0.95, 1.45, fills[3]).move_to(glasses[4], aligned_edge=DOWN)
        self.play(Transform(glasses[4], new_g), FadeIn(tie))
        preds[4] = preds[3]
        self.play(Transform(bumps[4], bump(preds[4], BOWL_COLS[4])))
        heard = preds[4]
        hl = DashedLine(fax.c2p(xv(heard), 0), fax.c2p(xv(heard), 1.05), color=INK)
        self.play(ShowCreation(hl))
        pa = post(heard)
        A2 = self.bars(pa, AUDIO_C, RIGHT * 2.0 + UP * 0.4, "audio")
        self.play(FadeIn(A2, shift=UP * 0.2))
        stick = Line(ORIGIN, RIGHT * 1.0 + UP * 1.0, stroke_width=9, color="#C08040")
        tgt = glasses[4].get_corner(UR)
        stick.move_to(tgt + RIGHT * 1.4 + UP * 1.4)
        self.play(stick.animate.move_to(tgt + RIGHT * 0.42 + UP * 0.42), run_time=0.7)
        pm = np.array([0.02, 0.02, 0.03, 0.08, 0.85])
        M2 = self.bars(pm, MOTION_C, RIGHT * 2.0 + DOWN * 2.3, "motion (striker tip)")
        self.play(FadeIn(M2, shift=UP * 0.2))
        pf = pa * pm
        pf = pf / pf.sum()
        F2 = self.bars(pf, FUSED_C, RIGHT * 4.6 + DOWN * 0.95, "fused")
        times = T("x", 34, INK).move_to(RIGHT * 4.2 + DOWN * 0.3)
        self.play(FadeIn(times), FadeIn(F2, shift=LEFT * 0.2))
        self.play(FlashAround(glasses[4], color=ACCENT), Indicate(F2[1][4]), run_time=1.2)
        msg = T("audio alone: 50/50     audio x motion: glass 5", 24, GREEN_B).move_to(tie)
        self.play(ReplacementTransform(tie, msg))
        self.wait(2)
        self.clear_all()


# ================================================================== 4. tune by sight
class TuneBySight(Base):
    def construct(self):
        self.title("4. Tune by sight: pour until the camera says STOP")
        h = ValueTracker(0.0)
        target_hz = 800.0
        offset = ValueTracker(0.0)          # cents: this glass is slightly different from the others

        def true_f(x):
            return french(x) * 2 ** (-25 / 1200)

        def pred_f(x):
            return french(x) * 2 ** (offset.get_value() / 1200)

        g = always_redraw(lambda: glass(1.9, 3.0, h.get_value()).move_to(LEFT * 4.8 + DOWN * 3.0, aligned_edge=DOWN))
        self.play(FadeIn(g))
        tl = T(f"target note: Ga = {target_hz:.0f} Hz", 28, GREEN_B).move_to(LEFT * 4.8 + UP * 1.9)
        self.play(FadeIn(tl))

        meter = Line(LEFT * 3.2, RIGHT * 3.2, color=MUTED, stroke_width=3).move_to(RIGHT * 2.3 + UP * 1.2)
        ticks = VGroup(*[VGroup(Line(UP * 0.12, DOWN * 0.12, color=MUTED).move_to(meter.point_from_proportion((c + 300) / 600)),
                                T(f"{c:+d}" if c else "0", 18, MUTED).next_to(meter.point_from_proportion((c + 300) / 600), DOWN, buff=0.22))
                         for c in range(-300, 301, 100)])
        ml = T("camera-predicted pitch minus target (cents)", 22, MUTED).next_to(meter, UP, buff=0.5)
        needle = always_redraw(lambda: Triangle(fill_opacity=1).scale(0.17).rotate(PI).set_fill(WATER).set_stroke(width=0).move_to(
            meter.point_from_proportion(np.clip((1200 * np.log2(pred_f(h.get_value()) / target_hz) + 300) / 600, 0, 1)) + UP * 0.2))
        self.play(ShowCreation(meter), FadeIn(ticks), FadeIn(ml), FadeIn(needle))

        stage = T("step 1: pour, and stop one semitone (100 cents) early", 24, WATER).move_to(RIGHT * 2.3 + DOWN * 0.5)
        self.play(FadeIn(stage))
        x_short = (((F0 / (target_hz * 2 ** (100 / 1200))) ** 2 - 1) / BETA) ** 0.25
        self.play(h.animate.set_value(x_short), run_time=3.5, rate_func=linear)
        stop = T("STOP  -  tap once to verify", 28, RED_B).next_to(stage, DOWN, buff=0.35)
        self.play(FadeIn(stop))
        heard = true_f(h.get_value())
        tap = T(f"tap: heard {heard:.0f} Hz, predicted {pred_f(h.get_value()):.0f} Hz  ->  this glass is 25 cents flat",
                22, BLUE_B).next_to(stop, DOWN, buff=0.35)
        self.play(Flash(g.get_corner(UR), color=ACCENT), FadeIn(tap))
        self.play(offset.animate.set_value(-25.0), run_time=1.0)
        stage2 = T("step 2: top up using this glass's corrected physics", 24, WATER).move_to(stage)
        self.play(ReplacementTransform(stage, stage2), FadeOut(stop))
        x_fin = (((F0 * 2 ** (-25 / 1200) / target_hz) ** 2 - 1) / BETA) ** 0.25
        self.play(h.animate.set_value(x_fin), run_time=2.0, rate_func=smooth)
        err = 1200 * np.log2(true_f(h.get_value()) / target_hz)
        ok = T(f"final tap: {true_f(h.get_value()):.0f} Hz ({err:+.0f} cents)  -  in tune", 26, GREEN_B).move_to(tap)
        self.play(ReplacementTransform(tap, ok), Flash(g.get_corner(UR), color=GREEN_B))
        self.wait(2)
        self.clear_all()


# ================================================================== 5. architecture
class Pipeline(Base):
    def construct(self):
        self.title("TuneSight: how the two modalities work together")

        def box(lines, color, w, h, size=20):
            r = RoundedRectangle(width=w, height=h, corner_radius=0.15).set_stroke(color, 2.5).set_fill(color, 0.14)
            head = T(lines[0], size + 2, color)
            body = VGroup(*[T(s, size - 2, INK) for s in lines[1:]]).arrange(DOWN, buff=0.06)
            t = VGroup(head, body).arrange(DOWN, buff=0.1) if len(lines) > 1 else head
            t.move_to(r)
            return VGroup(r, t)

        # same layout as the slide diagram: sensors | features | reasoning | outputs, no crossing arrows
        top, bot = 1.3, -1.75
        cam = box(["Camera", "video frames"], BLUE_B, 2.4, 1.1).move_to(LEFT * 5.65 + UP * top)
        mic = box(["Microphone", "48 kHz audio"], RED_B, 2.4, 1.1).move_to(LEFT * 5.65 + UP * bot)
        vis = box(["Vision", "water line, ArUco,", "striker tip"], BLUE_B, 2.8, 1.45).move_to(LEFT * 2.4 + UP * top)
        aud = box(["Audio", "onset, FFT modes,", "LPC poles"], RED_B, 2.8, 1.45).move_to(LEFT * 2.4 + UP * bot)
        phy = box(["Physics, learned", "1/f^2 = a + b u^n", "evidence picks n"], ACCENT, 3.0, 1.45).move_to(RIGHT * 1.75 + UP * top)
        fus = box(["Fusion", "audio x motion", "+ 'none' check"], GREEN_B, 3.0, 1.45).move_to(RIGHT * 1.75 + UP * bot)
        tune = box(["Tune to a raga"], GREEN_B, 2.3, 0.75).move_to(RIGHT * 5.6 + UP * top)
        which = box(["Which bowl?"], GREEN_B, 2.3, 0.75).move_to(RIGHT * 5.6 + UP * (bot + 0.5))
        sargam = box(["Sargam notation"], GREEN_B, 2.3, 0.75).move_to(RIGHT * 5.6 + UP * (bot - 0.55))

        self.play(FadeIn(cam), FadeIn(mic))
        a1 = Arrow(cam.get_right(), vis.get_left(), buff=0.08)
        a2 = Arrow(mic.get_right(), aud.get_left(), buff=0.08)
        self.play(GrowArrow(a1), FadeIn(vis), GrowArrow(a2), FadeIn(aud))
        a3 = Arrow(vis.get_right(), phy.get_left(), buff=0.08, color=BLUE_B)
        l3 = T("fill ratio", 18, BLUE_B).next_to(a3, UP, buff=0.1)
        a7 = Arrow(aud.get_right(), fus.get_left(), buff=0.08, color=RED_B)
        l7 = T("pitch f", 18, RED_B).next_to(a7, DOWN, buff=0.1)
        self.play(GrowArrow(a3), FadeIn(l3), FadeIn(phy), GrowArrow(a7), FadeIn(l7), FadeIn(fus))
        # the striker tip seen by the camera goes straight to fusion
        s0, s1 = vis.get_corner(DR) + LEFT * 0.1, fus.get_corner(UL) + RIGHT * 0.1
        dash = DashedLine(s0, s1, dash_length=0.12, color=BLUE_B, stroke_width=3)
        tip = Arrow(s1 + 0.35 * normalize(s0 - s1), s1, buff=0, color=BLUE_B, stroke_width=3)
        ls = T("striker tip", 18, BLUE_B).next_to(dash.get_center(), LEFT, buff=0.15)
        self.play(ShowCreation(dash), GrowArrow(tip), FadeIn(ls))
        # physics predicts each bowl's pitch for fusion; every identified tap teaches physics
        a5 = Arrow(phy.get_bottom() + RIGHT * 0.6, fus.get_top() + RIGHT * 0.6, buff=0.08, color=ACCENT)
        l5 = VGroup(T("predicted pitch", 18, ACCENT), T("of every bowl", 18, ACCENT)).arrange(DOWN, buff=0.05)
        l5.next_to(a5, RIGHT, buff=0.12)
        a6 = Arrow(fus.get_top() + LEFT * 0.6, phy.get_bottom() + LEFT * 0.6, buff=0.08, color=GREEN_B)
        l6 = T("learn", 18, GREEN_B).next_to(a6, LEFT, buff=0.12)
        self.play(GrowArrow(a5), FadeIn(l5), GrowArrow(a6), FadeIn(l6))
        teach = T("each tap adds a (fill ratio, pitch) pair to the law: audio teaches vision", 22, ACCENT)
        self.play(FadeIn(teach.to_edge(DOWN, buff=0.25)))
        a9 = Arrow(phy.get_right(), tune.get_left(), buff=0.08, color=GREEN_B)
        a8 = Arrow(fus.get_right(), which.get_left(), buff=0.08, color=GREEN_B)
        a10 = Arrow(fus.get_right(), sargam.get_left(), buff=0.08, color=GREEN_B)
        self.play(GrowArrow(a9), GrowArrow(a8), GrowArrow(a10),
                  LaggedStart(*[FadeIn(o) for o in (tune, which, sargam)], lag_ratio=0.2))
        for a in (a1, a3, a5, a8):
            self.play(ShowPassingFlash(a.copy().set_color(ACCENT).set_stroke(width=8), time_width=0.6), run_time=0.5)
        self.wait(2.5)

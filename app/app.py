"""TuneSight live demo: a jal tarang that sees and listens.

    python app/app.py            # live: webcam + microphone
    python app/app.py --sim      # simulation mode (no hardware; also the backup for the demo)
    python app/app.py --camera 1 --calib models/calibration.json

Modes
  Calibrate          : click a bowl in the video (or let motion pick it), strike it; each tap adds one
                       (fill from camera, pitch from mic) point - audio teaches vision online.
  Tune by sight      : choose a raga and Sa (or Auto Sa), press "Plan"; each bowl shows its target swara
                       and "pour X ml / STOP" from the camera alone; strike to verify.
  Play & transcribe  : strike any bowl; the system says which one (audio physics + striker motion),
                       highlights it and writes the sargam notation.
Keys: 1..9 strike bowl in --sim mode | Space: replay last strike resynthesised | H: hear selected bowl's predicted sound
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from tunesight import audio as A  # noqa: E402
from tunesight import config as C  # noqa: E402
from tunesight import fusion as F  # noqa: E402
from tunesight import music  # noqa: E402
from tunesight import physics as P  # noqa: E402
from tunesight import simulate as SIM  # noqa: E402
from tunesight import tuning as T  # noqa: E402
from tunesight import vision as V  # noqa: E402

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

MODES = ["Calibrate", "Tune by sight", "Play & transcribe"]
BOWL_COLORS = [(66, 135, 245), (245, 130, 48), (60, 180, 75), (230, 25, 75), (145, 30, 180),
               (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128)]


# ============================================================================ sources
class LiveSource:
    is_sim = False

    def __init__(self, camera: int, sr: int = C.SR):
        self.sr = sr
        self.cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.q: queue.Queue = queue.Queue()
        self.t0 = time.perf_counter()
        try:
            self.stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=1024, callback=self._cb)
        except Exception:   # some microphones only support 44.1 kHz; the analysis works at any rate
            self.sr = 44100
            self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", blocksize=1024, callback=self._cb)
        self.stream.start()
        self.t_audio0 = time.perf_counter()

    def _cb(self, indata, frames, time_info, status):
        self.q.put(indata[:, 0].copy())

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def audio_t0(self) -> float:
        return self.t_audio0 - self.t0

    def read_frame(self):
        ok, f = self.cap.read()
        return f if ok else None

    def read_audio(self) -> np.ndarray:
        blocks = []
        while not self.q.empty():
            blocks.append(self.q.get_nowait())
        return np.concatenate(blocks).astype(np.float64) if blocks else np.zeros(0)

    def close(self):
        self.stream.stop()
        self.cap.release()


class SimSource:
    is_sim = True

    def __init__(self, n_bowls: int = 5, sr: int = C.SR):
        self.sr = sr
        self.world = SIM.SimWorld(SIM.default_bowls(n_bowls), sr=sr)
        for b, f in zip(self.world.bowls, [0.15, 0.55, 0.3, 0.7, 0.05, 0.4, 0.6, 0.2, 0.8]):
            b.fill = f
        self.t0 = time.perf_counter()
        self.last = 0.0
        self.audible = True

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def audio_t0(self) -> float:
        return 0.0

    def read_frame(self):
        now = self.now()
        self.world.advance(now, now - self.last)
        self.last = now
        return self.world.frame(now)

    def read_audio(self) -> np.ndarray:
        _, block = self.world.audio_until(self.now())
        return block

    def strike(self, i: int):
        if 0 <= i < len(self.world.bowls):
            x = self.world.strike(i, self.now())
            if sd is not None and self.audible:   # let the audience hear the simulated strike
                QtCore.QTimer.singleShot(200, lambda: _safe_play(x, self.sr))

    def pour(self, i: int, rate: float):
        if rate == 0:
            self.world.pour_rate.pop(i, None)
        else:
            self.world.pour_rate[i] = rate

    def close(self):
        pass


def _safe_play(x, sr):
    try:
        sd.play(np.asarray(x, np.float32) * 0.6, sr)
    except Exception:
        pass


# ============================================================================ audio ring buffer + strike detection
class StrikeListener:
    def __init__(self, sr: int, seconds: float = 8.0):
        self.sr = sr
        self.buf = np.zeros(int(seconds * sr))
        self.total = 0                         # absolute index of the next sample
        self.det = A.StreamingOnsetDetector(sr)
        self.pending: list[int] = []

    def push(self, x: np.ndarray):
        if len(x) == 0:
            return
        n = len(x)
        if n >= len(self.buf):
            self.buf[:] = x[-len(self.buf):]
        else:
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = x
        self.pending += self.det.process(x)
        self.total += n

    def get(self, a: int, b: int) -> np.ndarray:
        start = self.total - len(self.buf)
        a, b = max(a, start), min(b, self.total)
        return self.buf[a - start:b - start].copy()

    def ready(self, post_s: float = 0.9):
        """Onsets that now have enough ringing after them to analyse."""
        out, keep = [], []
        for on in self.pending:
            (out if self.total >= on + int(post_s * self.sr) else keep).append(on)
        self.pending = keep
        return out


# ============================================================================ widgets
class VideoLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background:#111;")
        self._geom = (1.0, 0, 0)

    def show_image(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(img).scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        s = pix.width() / w
        self._geom = (s, (self.width() - pix.width()) / 2, (self.height() - pix.height()) / 2)
        self.setPixmap(pix)

    def mousePressEvent(self, e):
        s, ox, oy = self._geom
        p = e.position()
        self.clicked.emit((p.x() - ox) / s, (p.y() - oy) / s)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, source, calib_path: Path | None):
        super().__init__()
        self.src = source
        self.sr = source.sr
        self.setWindowTitle("TuneSight - a jal tarang that sees and listens" + ("  [SIMULATION]" if source.is_sim else ""))
        self.resize(1500, 900)
        self.tracker = V.BowlTracker()
        self.listener = StrikeListener(self.sr)
        self.frames = deque(maxlen=60)                 # (t, gray, 1.0) for motion evidence
        self.obs: V.FrameObs | None = None
        self.shared_cal = P.FrenchCalibrator()
        self.bowl_cals: dict[int, P.FrenchCalibrator] = {}
        self.selected: int | None = None
        self.highlight: tuple[int, float] | None = None
        self.targets: dict[int, T.BowlTarget] = {}
        self.tune_cals: dict[int, P.FrenchCalibrator] = {}
        self.verified: set = set()          # bowls that already had a verification tap
        self.pre_stop_cents = 100.0         # first pour stops a semitone above the target (water only lowers pitch)
        self.sa = 0.0
        self.last_an: A.StrikeAnalysis | None = None
        self.last_ratio_profile = None
        self.notation: list[tuple[str, bool]] = []
        self.debug = False
        self._build_ui()
        if calib_path and calib_path.exists():
            self.load_calibration(calib_path)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(MODES)
        self.raga = QtWidgets.QComboBox()
        self.raga.addItems(list(music.RAGAS))
        self.raga.setCurrentText("Bhupali")
        self.sa_spin = QtWidgets.QDoubleSpinBox()
        self.sa_spin.setRange(100, 3000)
        self.sa_spin.setValue(523.25)
        self.sa_spin.setSuffix(" Hz")
        self.auto_sa = QtWidgets.QCheckBox("Auto Sa")
        self.auto_sa.setChecked(True)
        self.shared = QtWidgets.QCheckBox("Identical bowls share physics")
        self.shared.setChecked(True)
        b_plan = QtWidgets.QPushButton("Plan tuning")
        b_plan.clicked.connect(self.plan_tuning)
        b_reset = QtWidgets.QPushButton("Reset calibration")
        b_reset.clicked.connect(self.reset_calibration)
        b_save = QtWidgets.QPushButton("Save calib")
        b_save.clicked.connect(lambda: self.save_calibration(C.MODELS_DIR / "calibration.json"))
        b_load = QtWidgets.QPushButton("Load calib")
        b_load.clicked.connect(lambda: self.load_calibration(C.MODELS_DIR / "calibration.json"))
        b_resyn = QtWidgets.QPushButton("Replay resynthesis")
        b_resyn.clicked.connect(self.play_resynthesis)
        b_hear = QtWidgets.QPushButton("Hear before strike")
        b_hear.clicked.connect(self.hear_prediction)
        b_clear = QtWidgets.QPushButton("Clear notation")
        b_clear.clicked.connect(lambda: (self.notation.clear(), self._update_notation()))
        for w in [QtWidgets.QLabel("Mode:"), self.mode, QtWidgets.QLabel("Raga:"), self.raga, QtWidgets.QLabel("Sa:"),
                  self.sa_spin, self.auto_sa, self.shared, b_plan, b_reset, b_save, b_load, b_resyn, b_hear, b_clear]:
            bar.addWidget(w)
        bar.addStretch(1)
        root.addLayout(bar)

        body = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        self.video = VideoLabel()
        self.video.clicked.connect(self.on_video_click)
        left.addWidget(self.video, 1)
        if self.src.is_sim:
            simbar = QtWidgets.QHBoxLayout()
            simbar.addWidget(QtWidgets.QLabel("Simulation:"))
            for i in range(len(self.src.world.bowls)):
                bs = QtWidgets.QPushButton(f"strike {i + 1}")
                bs.clicked.connect(lambda _=False, k=i: self.src.strike(k))
                bp = QtWidgets.QPushButton(f"pour {i + 1}")
                bp.pressed.connect(lambda k=i: self.src.pour(k, 0.06))
                bp.released.connect(lambda k=i: self.src.pour(k, 0.0))
                bd = QtWidgets.QPushButton(f"drain {i + 1}")
                bd.pressed.connect(lambda k=i: self.src.pour(k, -0.06))
                bd.released.connect(lambda k=i: self.src.pour(k, 0.0))
                for w in (bs, bp, bd):
                    w.setMaximumWidth(70)
                    simbar.addWidget(w)
            simbar.addStretch(1)
            left.addLayout(simbar)
        self.big = QtWidgets.QLabel("Strike a bowl...")
        self.big.setStyleSheet("font-size:26px; font-weight:600; padding:6px;")
        left.addWidget(self.big)
        self.notation_lbl = QtWidgets.QLabel("")
        self.notation_lbl.setStyleSheet("font-size:22px; font-family:Consolas; padding:6px; background:#fff8e7;")
        self.notation_lbl.setWordWrap(True)
        left.addWidget(self.notation_lbl)
        body.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        self.spec = pg.PlotWidget(title="Spectrum of the last strike")
        self.spec.setLabel("bottom", "frequency", "Hz")
        self.spec.setLabel("left", "dB")
        self.spec_curve = self.spec.plot(pen=pg.mkPen("#555", width=1))
        self.spec_pts = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(220, 50, 50))
        self.spec.addItem(self.spec_pts)
        self.spec_pitch = pg.InfiniteLine(angle=90, pen=pg.mkPen((40, 160, 60), width=2, style=QtCore.Qt.DashLine))
        self.spec.addItem(self.spec_pitch)
        right.addWidget(self.spec, 3)

        row = QtWidgets.QHBoxLayout()
        self.poles = pg.PlotWidget(title="LPC poles")
        self.poles.setAspectLocked(True)
        th = np.linspace(0, 2 * np.pi, 300)
        self.poles.plot(np.cos(th), np.sin(th), pen=pg.mkPen("#aaa"))
        self.pole_all = pg.ScatterPlotItem(size=6, symbol="x", pen=pg.mkPen("#888"))
        self.pole_modes = pg.ScatterPlotItem(size=11, pen=pg.mkPen((220, 50, 50), width=2), brush=None)
        self.poles.addItem(self.pole_all)
        self.poles.addItem(self.pole_modes)
        row.addWidget(self.poles, 1)
        self.law = pg.PlotWidget(title="Learned law")
        self.law.setLabel("bottom", "(h/H)^n  (camera)")
        self.law.setLabel("left", "1e6 / f^2  (microphone)")
        self.law_pts = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(66, 135, 245))
        self.law_line = self.law.plot(pen=pg.mkPen((220, 50, 50), width=2))
        self.law.addItem(self.law_pts)
        row.addWidget(self.law, 1)
        right.addLayout(row, 3)

        self.cues = pg.PlotWidget(title="Which bowl? blue audio, orange motion, green fused")
        self.cues.setYRange(0, 1)
        self.cue_items = []
        right.addWidget(self.cues, 2)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setStyleSheet("font-family:Consolas; font-size:11px;")
        right.addWidget(self.log, 2)
        body.addLayout(right, 2)
        root.addLayout(body, 1)

    def say(self, s: str):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {s}")

    # ------------------------------------------------------------------ calibration store
    def cal(self, i: int) -> P.FrenchCalibrator:
        if self.shared.isChecked():
            return self.shared_cal
        if i not in self.bowl_cals:
            c = P.FrenchCalibrator(f0_prior=self.shared_cal.f0, f0_prior_cents=300)
            self.bowl_cals[i] = c
        return self.bowl_cals[i]

    def reset_calibration(self):
        self.shared_cal = P.FrenchCalibrator()
        self.bowl_cals = {}
        self.targets = {}
        self.tune_cals = {}
        self.say("calibration reset")

    def save_calibration(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {"shared": self.shared.isChecked(), "calibrators": {"shared": self.shared_cal.to_dict(),
                                                                **{str(k): v.to_dict() for k, v in self.bowl_cals.items()}}}
        json.dump(d, open(path, "w"), indent=2)
        self.say(f"saved calibration -> {path}")

    def load_calibration(self, path: Path):
        try:
            d = json.load(open(path))
        except Exception as e:
            self.say(f"could not load calibration: {e}")
            return
        cals = d.get("calibrators", {})
        if "shared" in cals:
            self.shared_cal = P.FrenchCalibrator.from_dict(cals["shared"])
        self.bowl_cals = {int(k): P.FrenchCalibrator.from_dict(v) for k, v in cals.items() if k != "shared"}
        if not self.bowl_cals and "shared" not in cals and cals:
            self.shared_cal = P.FrenchCalibrator.from_dict(next(iter(cals.values())))
        self.shared.setChecked(bool(d.get("shared", True)))
        if self.shared.isChecked() or not self.bowl_cals:
            self.say(f"loaded shared calibration from {path}: f0={self.shared_cal.f0:.1f} Hz "
                     f"beta={self.shared_cal.beta:.2f} ({self.shared_cal.n_taps} taps)")
        else:
            per_bowl = ", ".join(f"#{k + 1} f0={c.f0:.0f} Hz ({c.n_taps} taps)" for k, c in sorted(self.bowl_cals.items()))
            self.say(f"loaded per-bowl calibration from {path}: {per_bowl}")

    # ------------------------------------------------------------------ main loop
    def tick(self):
        frame = self.src.read_frame()
        now = self.src.now()
        if frame is not None:
            self.obs = V.measure_frame(frame, t=now)
            self.tracker.update(self.obs)
            img = cv2.resize(frame, None, fx=self.obs.scale, fy=self.obs.scale) if self.obs.scale < 1 else frame.copy()
            self.frames.append((now, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.0))
            self.video.show_image(cv2.cvtColor(self.draw_overlay(img, now), cv2.COLOR_BGR2RGB))
        x = self.src.read_audio()
        if len(x):
            self.listener.push(x)
        for on in self.listener.ready():
            seg = self.listener.get(on - int(0.2 * self.sr), on + int(1.1 * self.sr))
            an = A.analyze_strike(seg, self.sr, onset=int(0.2 * self.sr))
            if an is not None and an.snr_db > 8:
                t_on = self.src.audio_t0() + on / self.sr
                self.on_strike(an, seg, t_on)

    # ------------------------------------------------------------------ strike handling
    def bowl_state(self, t: float):
        """Fill ratio of every tracked bowl just before time t (pre-strike, avoids hand occlusion)."""
        tracks = self.tracker.tracks
        fills = [tr.level(t - 0.05)[0] for tr in tracks]
        return tracks, fills

    def on_strike(self, an: A.StrikeAnalysis, seg, t_on: float):
        self.last_an = an
        self.update_spectrum(an)
        tracks, fills = self.bowl_state(t_on)
        if not tracks:
            self.say(f"strike: {an.pitch:.1f} Hz but no bowls visible")
            self.big.setText(f"{an.pitch:.1f} Hz  (no bowls in view)")
            return
        n = len(tracks)
        preds = [self.cal(tr.tid).predict(f) for tr, f in zip(tracks, fills)]
        mp, mc = V.motion_evidence(list(self.frames), [tr.last.bbox for tr in tracks], t_on)
        if self.debug:
            ts = [f[0] for f in self.frames]
            self.say(f"[debug] t_on={t_on:.3f} frames {len(ts)} span [{min(ts):.3f}, {max(ts):.3f}] "
                     f"in window {sum(t_on - 0.4 <= t <= t_on + 0.08 for t in ts)} mc={mc:.2f}")
        mode = self.mode.currentText()
        calibrated = self.cal(tracks[0].tid).n_taps >= 3
        ident = F.identify_struck(an.pitch, [p[0] for p in preds], [p[1] for p in preds], motion_p=mp,
                                  motion_conf=mc, audio_conf=an.confidence if calibrated else 0.0)
        self.update_cues(ident, n)

        if mode == "Calibrate":
            k = self.selected if self.selected is not None and self.selected < n else (
                int(np.argmax(mp)) if mc > 0.3 else (ident.best if calibrated else None))
            if k is None or k < 0:
                self.say(f"heard {an.pitch:.1f} Hz - click the bowl you strike to label it")
                self.big.setText(f"{an.pitch:.1f} Hz - which bowl? click it in the video")
                return
            c = self.cal(tracks[k].tid)
            c.add(fills[k], an.pitch)
            self.highlight = (k, time.time())
            self.say(f"learned: bowl {k + 1} fill {fills[k]:.3f} -> {an.pitch:.1f} Hz | taps={c.n_taps} "
                     f"f0={c.f0:.1f} Hz beta={c.beta:.2f}")
            self.big.setText(f"Bowl {k + 1}: fill {fills[k]:.2f}  ->  {an.pitch:.1f} Hz   (taps: {c.n_taps})")
            self.update_law(c)
            return

        k = ident.best
        if k < 0:
            self.big.setText(f"{an.pitch:.1f} Hz - does not match any visible bowl!")
            self.say(f"consistency check failed: {an.pitch:.1f} Hz vs predictions "
                     + ", ".join(f"{p[0]:.0f}" for p in preds))
            if mode == "Play & transcribe":
                self.notation.append(("?", False))
                self._update_notation()
            return
        self.highlight = (k, time.time())
        sa = self.sa if self.sa > 0 else self.sa_spin.value()
        scale = music.RAGAS[self.raga.currentText()]
        note, dev = music.nearest_swara(an.pitch, sa)
        in_raga = note.name in scale
        if mode == "Tune by sight":
            tg = self.targets.get(tracks[k].tid)
            if tg is None:
                self.big.setText(f"Bowl {k + 1}: {an.pitch:.1f} Hz ({note.label} {dev:+.0f}c) - press 'Plan tuning'")
            else:
                e = music.cents(an.pitch, tg.target_hz)
                ok = abs(e) <= 25
                self.big.setText(f"Bowl {k + 1}: target {tg.note.label}  heard {an.pitch:.1f} Hz  error {e:+.0f} cents  "
                                 f"{'OK' if ok else ('pour more' if e > 0 else 'too much water')}")
                self.say(f"verify bowl {k + 1}: target {tg.note.label} {tg.target_hz:.1f} Hz, heard {an.pitch:.1f} Hz ({e:+.1f} c)")
                # the verification tap refines *this bowl's* physics, so its target water level is corrected
                own = self.tune_cals.get(tracks[k].tid)
                if own is not None:
                    off = own.set_offset_from_tap(fills[k], an.pitch)
                    self.verified.add(tracks[k].tid)
                    self.say(f"  bowl {k + 1}: measured pitch offset {off:+.1f} cents")
                    tg.target_fill = own.height_for(tg.target_hz)
                    self.say(f"  bowl {k + 1} physics refined -> final target fill {tg.target_fill:.3f}")
        else:
            self.notation.append((note.label, in_raga))
            self._update_notation()
            self.big.setText(f"Bowl {k + 1}  ->  {note.label}  ({an.pitch:.1f} Hz, {dev:+.0f} c)"
                             + ("" if in_raga else "   [not in raga]"))
        self.say(f"bowl {k + 1} (p={ident.confidence:.2f}; audio w={ident.w_audio:.2f}, motion w={ident.w_motion:.2f}) "
                 f"pitch {an.pitch:.1f} Hz")

    # ------------------------------------------------------------------ tuning
    def plan_tuning(self):
        tracks = self.tracker.tracks
        if not tracks:
            self.say("no bowls visible")
            return
        cals = [self.cal(tr.tid) for tr in tracks]
        if min(c.n_taps for c in cals) < 2:
            self.say("calibrate first: tap bowls at a few water levels (Calibrate mode)")
            return
        sa, plan = T.plan(cals, self.raga.currentText(), None if self.auto_sa.isChecked() else self.sa_spin.value())
        self.sa = sa
        self.sa_spin.setValue(sa)
        self.targets = {tracks[p.bowl].tid: p for p in plan}
        # each bowl gets its own copy of the physics; its verification taps refine only that copy
        self.tune_cals = {tr.tid: P.FrenchCalibrator.from_dict(self.cal(tr.tid).to_dict()) for tr in tracks}
        self.verified = set()
        self.say(f"plan: raga {self.raga.currentText()}, Sa = {sa:.1f} Hz ({music.western_name(sa)})")
        for p in plan:
            self.say(f"  bowl {p.bowl + 1}: {p.note.label:5s} {p.target_hz:7.1f} Hz -> fill {p.target_fill:.3f}"
                     f"{'' if p.reachable else '  (UNREACHABLE)'}")
        self.mode.setCurrentText("Tune by sight")

    # ------------------------------------------------------------------ sounds
    def play_resynthesis(self):
        if self.last_an is not None and sd is not None:
            _safe_play(A.resynthesize(self.last_an, self.sr, 1.5), self.sr)

    def hear_prediction(self):
        """Hear before you strike: predicted sound of the selected bowl from the camera alone."""
        tracks = self.tracker.tracks
        k = self.selected if self.selected is not None and self.selected < len(tracks) else 0
        if not tracks or sd is None:
            return
        fill = tracks[k].level()[0]
        f, s = self.cal(tracks[k].tid).predict(fill)
        an = self.last_an
        if an is not None and len(an.freqs) > 1:
            ratios, sig = an.ratios, np.where(np.isfinite(an.sigma), an.sigma, 5.0) * (f / an.pitch)
            amps = 10 ** (an.mags_db / 20) * np.maximum(sig, 1)
        else:
            ratios, sig, amps = np.array([1, 2.62, 4.71]), np.array([4.0, 12, 25]), np.array([1, 0.35, 0.15])
        _safe_play(A.modal_synthesis(f * ratios, sig, amps / amps.max(), self.sr, 1.5, click=0.05, seed=0), self.sr)
        self.say(f"predicted sound of bowl {k + 1}: {f:.1f} Hz +- {s:.0f} c (from camera fill {fill:.3f})")

    # ------------------------------------------------------------------ drawing
    def draw_overlay(self, img, now):
        o = self.obs
        if o is not None and o.marker is not None:
            cv2.polylines(img, [o.marker.astype(np.int32)], True, (0, 200, 0), 2)
        mode = self.mode.currentText()
        sa = self.sa if self.sa > 0 else self.sa_spin.value()
        for idx, tr in enumerate(self.tracker.tracks):
            b = tr.last
            x, y, w, h = b.bbox
            col = BOWL_COLORS[idx % len(BOWL_COLORS)]
            cv2.drawContours(img, [b.contour], -1, col, 2)
            fill, _ = tr.level(now)
            if b.water_top_y is not None and b.level_conf > 0.2:
                cv2.line(img, (x, b.water_top_y), (x + w, b.water_top_y), (0, 0, 255), 2)
            pc = self.tune_cals.get(tr.tid, self.cal(tr.tid)) if mode == "Tune by sight" else self.cal(tr.tid)
            f, s = pc.predict(fill)
            note, dev = music.nearest_swara(f, sa)
            lines = [f"#{idx + 1}  h/H {fill:.2f}", f"{f:.0f} Hz {note.label}"]
            if self.cal(tr.tid).n_taps < 2:
                lines[1] = "not calibrated"
            if mode == "Tune by sight" and tr.tid in self.targets:
                tg = self.targets[tr.tid]
                ml = None
                if b.ppc and b.profile_px is not None:
                    ml = P.volume_curve_ml(b.profile_px / b.ppc, 1.0 / b.ppc)
                show_fill = self.display_target_fill(tr.tid, tg)
                adv = T.pour_advice(fill, show_fill, ml, tol=0.008)
                if "STOP" in adv:
                    adv = "STOP, tap" if tr.tid not in self.verified else "STOP"
                lines.append(f"{tg.note.label}: {adv.replace('pour ', '+').replace('remove ', '-')}")
                # gauge: current fill vs target fill
                gx = x + w + 6
                cv2.rectangle(img, (gx, y), (gx + 10, y + h), (80, 80, 80), 1)
                cy = int(y + h - fill * h)
                ty = int(y + h - (show_fill if np.isfinite(show_fill) else 0) * h)
                cv2.rectangle(img, (gx, cy), (gx + 10, y + h), (0, 140, 255), -1)
                cv2.line(img, (gx - 4, ty), (gx + 14, ty), (0, 180, 0) if "STOP" in adv else (0, 0, 220), 3)
            # labels of neighbouring bowls are staggered vertically so they never overlap
            lift = (len(lines) * 17 + 6) if idx % 2 else 0
            for j, s_ in enumerate(lines):
                yy = max(16, y - 10 - lift - 17 * (len(lines) - 1 - j))
                cv2.putText(img, s_, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(img, s_, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            if self.selected == idx:
                cv2.rectangle(img, (x - 6, y - 6), (x + w + 6, y + h + 6), (255, 0, 255), 2)
            if self.highlight and self.highlight[0] == idx and time.time() - self.highlight[1] < 0.9:
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 230, 255), 5)
        hdr = f"{mode} | raga {self.raga.currentText()} | Sa {sa:.1f} Hz | taps {self.shared_cal.n_taps}"
        cv2.putText(img, hdr, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return img

    def display_target_fill(self, tid: int, tg: T.BowlTarget) -> float:
        """Two-step tuning: before a bowl's verification tap, aim pre_stop_cents short of the target
        (water can only be added easily), afterwards aim at the target with the refined physics."""
        own = self.tune_cals.get(tid)
        if own is None:
            return tg.target_fill
        if tid in self.verified:
            return own.height_for(tg.target_hz)
        # aim one semitone sharp, but never above the empty bowl's pitch (then: tap right away)
        f_empty = own.predict(0.0)[0]
        return own.height_for(min(tg.target_hz * 2 ** (self.pre_stop_cents / 1200), f_empty * 0.9999))

    def on_video_click(self, xi, yi):
        for idx, tr in enumerate(self.tracker.tracks):
            x, y, w, h = tr.last.bbox
            if x - 10 <= xi <= x + w + 10 and y - 40 <= yi <= y + h + 10:
                self.selected = None if self.selected == idx else idx
                self.say(f"selected bowl {idx + 1}" if self.selected is not None else "selection cleared")
                return
        self.selected = None

    def update_spectrum(self, an):
        hi = min(C.FMAX, max(2000.0, an.freqs.max() * 1.3))
        band = (an.spec_f > 50) & (an.spec_f < hi)
        self.spec_curve.setData(an.spec_f[band], an.spec_db[band])
        self.spec_pts.setData(an.freqs, an.mags_db)
        self.spec_pitch.setValue(an.pitch)
        self.spec.setTitle(f"pitch {an.pitch:.1f} Hz | ring {an.t60:.2f} s | SNR {an.snr_db:.0f} dB")
        if an.lpc is not None:
            r = an.lpc.roots
            self.pole_all.setData(r.real, r.imag)
            zk = np.exp((-an.lpc.sigmas + 2j * np.pi * an.lpc.freqs) / an.lpc.sr)
            self.pole_modes.setData(zk.real, zk.imag)

    def update_law(self, c: P.FrenchCalibrator):
        if not c.fills:
            return
        x, y, xx, yy = c.law_points()
        self.law_pts.setData(x, 1e6 * y)
        self.law_line.setData(xx, 1e6 * yy)
        self.law.setTitle(f"Learned law: {c.n_taps} taps, n = {c.n:.2f}")

    def update_cues(self, ident: F.Identification, n: int):
        for it in self.cue_items:
            self.cues.removeItem(it)
        self.cue_items = []
        xs = np.arange(n + 1)
        for off, p, col in [(-0.27, ident.p_audio, (66, 135, 245)), (0.0, ident.p_motion, (245, 130, 48)),
                            (0.27, ident.probs, (60, 180, 75))]:
            it = pg.BarGraphItem(x=xs + off, height=p, width=0.25, brush=pg.mkBrush(*col))
            self.cues.addItem(it)
            self.cue_items.append(it)
        ax = self.cues.getAxis("bottom")
        ax.setTicks([[(i, f"bowl {i + 1}") for i in range(n)] + [(n, "none")]])

    def _update_notation(self):
        parts = [f"<span style='color:{'#000' if ok else '#c00'}'>{s}</span>" for s, ok in self.notation[-40:]]
        self.notation_lbl.setText("Sargam: " + "  ".join(parts))

    # ------------------------------------------------------------------ keys / close
    def keyPressEvent(self, e):
        k = e.key()
        if self.src.is_sim and QtCore.Qt.Key_1 <= k <= QtCore.Qt.Key_9:
            self.src.strike(k - QtCore.Qt.Key_1)
        elif k == QtCore.Qt.Key_Space:
            self.play_resynthesis()
        elif k == QtCore.Qt.Key_H:
            self.hear_prediction()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.timer.stop()
        self.src.close()
        super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="simulation mode (no camera / microphone needed)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--bowls", type=int, default=5, help="number of simulated bowls")
    ap.add_argument("--calib", default=str(C.MODELS_DIR / "calibration.json"))
    ap.add_argument("--autoclose", type=float, default=0, help="close after N seconds (testing)")
    ap.add_argument("--selftest", action="store_true", help="scripted end-to-end test in simulation mode")
    ap.add_argument("--debug", action="store_true", help="log audio/video timing details")
    a = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    src = SimSource(a.bowls) if (a.sim or a.selftest) else LiveSource(a.camera)
    w = MainWindow(src, None if a.selftest else Path(a.calib))
    w.show()
    if a.selftest:
        w.debug = a.debug
        src.audible = False
        _schedule_selftest(w)
    elif a.autoclose > 0:
        QtCore.QTimer.singleShot(int(a.autoclose * 1000), w.close)
    sys.exit(app.exec())


def _with_fill(b, f):
    b.fill = f
    return b


def _schedule_selftest(w: "MainWindow"):
    """Calibrate on bowls 1..5 (labelled by selection), then identify strikes, then plan tuning."""
    steps, t = [], 1500
    rng = np.random.default_rng(3)

    def at(ms, fn):
        steps.append((ms, fn))

    at(t, lambda: w.mode.setCurrentText("Calibrate"))
    for k in [0, 1, 2, 3, 4, 2, 0]:
        t += 1600
        at(t, lambda k=k: setattr(w, "selected", k))
        at(t + 50, lambda k=k: w.src.strike(k))
    t += 1600
    at(t, lambda: setattr(w, "selected", None))
    at(t, lambda: w.mode.setCurrentText("Play & transcribe"))
    truth = []
    for k in rng.integers(0, 5, 8):
        t += 1500
        at(t, lambda k=int(k): (truth.append(k), w.src.strike(k)))
    t += 1600
    at(t, w.plan_tuning)
    # tune one bowl by sight: pour into bowl 1 until the camera says STOP, then strike to verify
    t += 500
    at(t, lambda: w.src.world.bowls.__setitem__(0, _with_fill(w.src.world.bowls[0], 0.0)))
    t += 300

    def pour_step():
        """A 'person' following the on-screen advice: pour fast when far, slowly near the target, stop on STOP."""
        tr = w.tracker.tracks[0] if w.tracker.tracks else None
        tg = w.targets.get(tr.tid) if tr else None
        if tg is None:
            return
        fill = tr.level(w.src.now(), window=0.12)[0]
        goal = w.display_target_fill(tr.tid, tg)
        if np.isfinite(goal) and fill < goal - 0.004:
            w.src.pour(0, float(np.clip(1.5 * (goal - fill), 0.01, 0.12)))
            QtCore.QTimer.singleShot(40, pour_step)
        else:
            w.src.pour(0, 0.0)
            stage = "final" if tr.tid in w.verified else "step 1"
            w.say(f"{stage}: stopped pouring at camera fill {fill:.3f} (aim {goal:.3f})")
            QtCore.QTimer.singleShot(700, lambda: w.src.strike(0))
            if tr.tid not in w.verified:
                QtCore.QTimer.singleShot(3200, pour_step)        # after the verification tap: top up

    at(t, pour_step)
    t += 26000

    def finish():
        w.grab().save(str(C.RESULTS_DIR / "figures" / "app_screenshot.png"))
        print("\n".join(w.log.toPlainText().splitlines()))
        print("TRUTH (play phase):", [k + 1 for k in truth])
        print("NOTATION:", [s for s, _ in w.notation])
        w.close()

    at(t, finish)
    for ms, fn in steps:
        QtCore.QTimer.singleShot(ms, fn)


if __name__ == "__main__":
    main()

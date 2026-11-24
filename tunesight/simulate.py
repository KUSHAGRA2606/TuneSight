"""Physics-based simulator of a jal tarang scene (images + strike sounds) with known ground truth.

Used for: unit tests, controlled experiments (noise, blur, darkness sweeps), and as a fallback
"simulation mode" of the live demo. Simulated results are always labelled as simulated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import signal

from . import config as C
from .physics import french_freq
from .vision import marker_image

WATER_BGR = (40, 150, 235)     # haldi / orange food colour


@dataclass
class SimBowl:
    x_cm: float                       # centre position on the table
    H_cm: float = 9.0
    D_top_cm: float = 7.0
    D_bot_cm: float = 5.5
    f0: float = 1100.0                # empty-bowl pitch of the (2,0) mode
    beta: float = 3.0
    n: float = 4.0
    ratios: tuple = (1.0, 2.62, 4.71, 7.18)
    amps: tuple = (1.0, 0.35, 0.18, 0.08)
    eta0: float = 1.2e-3              # loss factor of the empty glass
    fill: float = 0.0                 # h / H

    def modes(self, fill: float | None = None):
        fill = self.fill if fill is None else fill
        f1 = float(french_freq(fill, self.f0, self.beta, self.n))
        fr = [f1 * r * (1 + 0.02 * k * fill) for k, r in enumerate(self.ratios)]
        eta = self.eta0 * (1 + 2.0 * fill ** 2)          # water adds damping
        sig = [math.pi * f * eta * (1 + 0.3 * k) for k, f in enumerate(fr)]
        return np.array(fr), np.array(sig), np.array(self.amps)


def family_bowl(x_cm: float, R_mean_cm: float, H_cm: float, taper: float = 0.8, a_mm: float = 1.5) -> SimBowl:
    """A container whose pitch follows from its shape and wall thickness (French's thin-wall model).

    taper = base radius / rim radius; R_mean_cm is the average of the two (mid-wall).
    """
    from .geometry import french_beta, french_f0     # local import: geometry imports physics, not simulate
    R_top = 2 * R_mean_cm / (1 + taper)
    return SimBowl(x_cm=x_cm, H_cm=H_cm, D_top_cm=2 * R_top, D_bot_cm=2 * R_top * taper,
                   f0=french_f0(R_mean_cm, H_cm, a_mm), beta=french_beta(R_mean_cm, a_mm))


def default_bowls(n: int = 5, identical: bool = True, seed: int = 0, f0_jitter: float = 0.004) -> list[SimBowl]:
    rng = np.random.default_rng(seed)
    bowls = []
    for i in range(n):
        s = 1.0 if identical else 0.8 + 0.4 * rng.random()   # scale factor (scaling law: f0 ~ 1/size)
        b = SimBowl(x_cm=8.0 + i * 9.5, H_cm=9.0 * s, D_top_cm=7.0 * s, D_bot_cm=5.5 * s,
                    f0=1100.0 / s * (1 + f0_jitter * rng.standard_normal()), beta=3.0 * (1 + 0.02 * rng.standard_normal()))
        bowls.append(b)
    return bowls


# ----------------------------------------------------------------------------- audio
def room_ir(sr: int, rt60: float = 0.35, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)
    h[0] = 0
    h = h / np.sqrt(np.sum(h ** 2))
    out = np.zeros(n)
    out[0] = 1.0
    return out + 0.25 * h


def synth_strike(bowl: SimBowl, fill: float | None = None, sr: int = C.SR, dur: float = 1.6, onset_s: float = 0.15,
                 snr_db: float = 40.0, reverb: bool = True, seed: int | None = None, gain: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fr, sg, am = bowl.modes(fill)
    n = int(dur * sr)
    t = np.arange(n - int(onset_s * sr)) / sr
    ring = np.zeros_like(t)
    for f, s, a in zip(fr, sg, am):
        ring += a * (0.8 + 0.4 * rng.random()) * np.exp(-s * t) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    m = int(0.002 * sr)                                  # 2 ms contact transient = the "source"
    click = rng.standard_normal(m) * np.hanning(m) * 0.6
    ring[:m] += click
    x = np.zeros(n)
    x[int(onset_s * sr):] = ring
    if reverb:
        x = signal.fftconvolve(x, room_ir(sr), mode="full")[:n]
    x = gain * x / (np.max(np.abs(x)) + 1e-12)
    ref = np.sqrt(np.mean(x[int(onset_s * sr):int((onset_s + 0.1) * sr)] ** 2))
    noise = rng.standard_normal(n)
    noise = signal.lfilter([1.0], [1.0, -0.9], noise)    # slightly low-pass "room" noise
    noise *= ref / (np.sqrt(np.mean(noise ** 2)) + 1e-12) * 10 ** (-snr_db / 20)
    return x + noise


# ----------------------------------------------------------------------------- images
def visible_fill(bowl: SimBowl, fill: float, ppc: float = 18.0) -> float:
    """Ground-truth fill ratio as the camera defines it (water top above the outer base / outer height).

    The physics 'fill' is measured from the inside bottom; the glass base adds a small offset,
    exactly as with a real glass. The calibration absorbs this offset.
    """
    hb = bowl.H_cm * ppc
    base_h = max(3, int(0.05 * hb))
    rim = max(3, int(0.08 * bowl.D_top_cm * ppc / 2)) + 1.0     # rim ellipse protrudes above the wall
    H_vis = hb + rim + 1.5                                         # + half of the 3 px outline at the base
    h_vis = base_h + fill * (hb - base_h) + 1.5
    return h_vis / H_vis if fill > 0.005 else 0.0


_BG_CACHE: dict = {}
_NOISE_CACHE: dict = {}
WATER_BLEND = tuple(int(0.75 * w + 0.25 * g) for w, g in zip(WATER_BGR, (228, 230, 232)))


def _background(size, ppc, marker):
    key = (size, round(ppc, 3), marker)
    if key not in _BG_CACHE:
        W, H = size
        yy = np.linspace(0, 1, H)[:, None]
        bg = (238 + 12 * (1 - yy) - 6 * np.abs(np.linspace(-1, 1, W))[None, :]).astype(np.float32)
        img = np.repeat(bg[:, :, None], 3, axis=2)
        table_y = int(H * 0.82)
        img[table_y:, :] *= 0.93
        if marker:
            mp = int(C.MARKER_CM * ppc)
            mk = marker_image(mp)
            pad = int(0.15 * mp)
            y0, x0 = table_y - mp - 8, 20
            img[y0 - pad:y0 + mp + pad, x0 - pad:x0 + mp + pad] = 255
            img[y0:y0 + mp, x0:x0 + mp] = mk[:, :, None]
        _BG_CACHE[key] = img.clip(0, 255).astype(np.uint8)
    return _BG_CACHE[key]


def _noise(shape, sigma, rng):
    """Sensor noise: a random crop of a cached noise texture (cheap, still different every frame)."""
    key = shape
    if key not in _NOISE_CACHE:
        _NOISE_CACHE[key] = np.random.default_rng(123).standard_normal((shape[0] + 64, shape[1] + 64, 3)).astype(np.float32)
    nz = _NOISE_CACHE[key]
    dy, dx = rng.integers(0, 64, 2)
    return nz[dy:dy + shape[0], dx:dx + shape[1]] * sigma


def render_scene(bowls: list[SimBowl], fills: list[float] | None = None, size=(1280, 720), ppc: float = 18.0,
                 marker: bool = True, stick_to: int | None = None, stick_phase: float = 1.0, brightness: float = 1.0,
                 blur: float = 0.0, noise: float = 2.0, seed: int | None = None, clear_water: bool = False) -> np.ndarray:
    W, H = size
    rng = np.random.default_rng(seed)
    img = _background(size, ppc, marker).copy()
    table_y = int(H * 0.82)
    fills = [b.fill for b in bowls] if fills is None else fills
    for b, fl in zip(bowls, fills):
        cx = int(b.x_cm * ppc + 3 * ppc)
        hb, wt, wb = b.H_cm * ppc, b.D_top_cm * ppc / 2, b.D_bot_cm * ppc / 2
        yb, yt = table_y, int(table_y - hb)
        poly = np.array([[cx - wb, yb], [cx + wb, yb], [cx + wt, yt], [cx - wt, yt]], np.int32)
        cv2.fillConvexPoly(img, poly, (230, 232, 234))
        base_h = max(3, int(0.05 * hb))
        if fl > 0.005:
            hw = (yb - base_h) - int(fl * (hb - base_h))
            frac = (yb - hw) / hb
            ww = wb + (wt - wb) * frac
            wpoly = np.array([[cx - wb + 3, yb - base_h], [cx + wb - 3, yb - base_h], [cx + ww - 3, hw], [cx - ww + 3, hw]], np.int32)
            col = (210, 214, 216) if clear_water else WATER_BLEND
            cv2.fillConvexPoly(img, wpoly, col)
            cv2.line(img, (int(cx - ww + 3), hw), (int(cx + ww - 3), hw), (90, 110, 140) if not clear_water else (150, 152, 155), 2)
        cv2.rectangle(img, (int(cx - wb), yb - base_h), (int(cx + wb), yb), (170, 172, 175), -1)
        cv2.polylines(img, [poly], True, (70, 72, 75), 3, cv2.LINE_AA)
        cv2.ellipse(img, (cx, yt), (int(wt), max(3, int(0.08 * wt))), 0, 0, 360, (95, 95, 98), 2, cv2.LINE_AA)
        cv2.line(img, (int(cx - wt * 0.7), yt + 8), (int(cx - wb * 0.7), yb - base_h - 4), (250, 250, 250), 3)
    if stick_to is not None and 0 <= stick_to < len(bowls):
        b = bowls[stick_to]
        cx = int(b.x_cm * ppc + 3 * ppc)
        yt = int(table_y - b.H_cm * ppc)
        tip = np.array([cx + b.D_top_cm * ppc / 2, yt + 6])
        rest = np.array([cx + 6 * ppc, yt - 6 * ppc])
        p = rest + (tip - rest) * float(np.clip(stick_phase, 0, 1))
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in p + np.array([5 * ppc, -6 * ppc])), (45, 80, 125), 9, cv2.LINE_AA)
    if brightness == 1.0 and blur == 0 and noise <= 3.0:
        nz = _noise(img.shape[:2], noise, rng)
        return cv2.add(img.astype(np.float32), nz).clip(0, 255).astype(np.uint8)
    out = img.astype(np.float32) * brightness
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out += _noise(out.shape[:2], noise, rng)
    return out.clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- live simulated world
@dataclass
class SimWorld:
    bowls: list = field(default_factory=lambda: default_bowls(5))
    sr: int = C.SR
    snr_db: float = 40.0
    t: float = 0.0
    _audio_queue: list = field(default_factory=list)   # (start_sample, samples)
    _emitted: int = 0
    _stick: tuple | None = None                        # (bowl, t_start)
    pour_rate: dict = field(default_factory=dict)      # bowl -> fill ratio per second

    def __post_init__(self):
        self._rng = np.random.default_rng(7)

    def strike(self, i: int, now: float) -> np.ndarray:
        """Animate the stick for 0.2 s, then emit the strike sound."""
        self._stick = (i, now)
        start = int((now + 0.2) * self.sr)
        x = synth_strike(self.bowls[i], sr=self.sr, dur=1.4, onset_s=0.0, snr_db=60, seed=int(self._rng.integers(1e9)))
        self._audio_queue.append((start, x))
        return x

    def advance(self, now: float, dt: float) -> None:
        for i, r in list(self.pour_rate.items()):
            self.bowls[i].fill = float(np.clip(self.bowls[i].fill + r * dt, 0, 0.95))
        self.t = now

    def audio_until(self, now: float) -> tuple[int, np.ndarray]:
        """All samples between the last call and `now` (start index, block)."""
        end = int(now * self.sr)
        n = end - self._emitted
        if n <= 0:
            return self._emitted, np.zeros(0)
        block = self._rng.standard_normal(n) * 10 ** (-self.snr_db / 20) * 0.05
        s0 = self._emitted
        keep = []
        for (st, x) in self._audio_queue:
            a, b = max(st, s0), min(st + len(x), end)
            if a < b:
                block[a - s0:b - s0] += x[a - st:b - st]
            if st + len(x) > end:
                keep.append((st, x))
        self._audio_queue = keep
        self._emitted = end
        return s0, block

    def frame(self, now: float, size=(1280, 720)) -> np.ndarray:
        stick, phase = None, 1.0
        if self._stick is not None:
            i, t0 = self._stick
            dt = now - t0
            if dt < 0.5:
                stick, phase = i, min(1.0, dt / 0.2) if dt < 0.25 else max(0.0, 1 - (dt - 0.25) / 0.25)
            else:
                self._stick = None
        return render_scene(self.bowls, size=size, stick_to=stick, stick_phase=phase, noise=1.5)

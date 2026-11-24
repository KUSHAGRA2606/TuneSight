"""Vision model: find bowls on a bright background, measure their geometry and their water level.

Only classical image processing from the course is used:
  background estimation (morphological closing), thresholding (Otsu), morphology, hole filling,
  connected components / contours, row profiles of the binary silhouette, HSV saturation for the
  dyed water, an intensity-gradient fallback for clear water, and an ArUco marker for pixel->cm scale.

The key measurement is the fill ratio h/H, which is scale-free: physics only needs the ratio,
so the system works even without the marker (the marker adds absolute size and millilitres).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C


# ----------------------------------------------------------------------------- scale marker
def find_marker(img_bgr: np.ndarray, marker_cm: float = C.MARKER_CM):
    """Detect the ArUco marker; returns (pixels_per_cm, corner quad) or (None, None)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX     # sub-pixel corners: half the scale error
    det = cv2.aruco.ArucoDetector(d, params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None, None
    q = corners[0].reshape(4, 2)
    side = np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])
    return float(side / marker_cm), q


def marker_image(px: int = 400, marker_id: int = 0) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
    return cv2.aruco.generateImageMarker(d, marker_id, px)


# ----------------------------------------------------------------------------- data classes
@dataclass
class BowlObs:
    bbox: tuple                     # x, y, w, h (pixels, processing resolution)
    contour: np.ndarray = field(repr=False)
    H_px: float = 0.0
    D_px: float = 0.0
    top_y: int = 0
    bottom_y: int = 0
    axis_x: float = 0.0
    profile_px: np.ndarray = field(default=None, repr=False)   # half-width at each row from the base upward
    water_top_y: int | None = None
    fill_ratio: float = 0.0         # h / H
    level_conf: float = 0.0
    ppc: float | None = None        # pixels per cm

    @property
    def cx(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2

    @property
    def H_cm(self):
        return self.H_px / self.ppc if self.ppc else None

    @property
    def D_cm(self):
        return self.D_px / self.ppc if self.ppc else None

    @property
    def h_cm(self):
        return self.fill_ratio * self.H_px / self.ppc if self.ppc else None


@dataclass
class FrameObs:
    t: float
    scale: float                    # processing image / original image
    ppc: float | None
    marker: np.ndarray | None
    bowls: list


# ----------------------------------------------------------------------------- segmentation
def _fill_holes(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, thickness=cv2.FILLED)
    return out


def foreground_mask(img: np.ndarray, marker: np.ndarray | None = None, sat_thr: int = 60) -> np.ndarray:
    """Objects = darker than the estimated bright background, or strongly coloured (dyed water)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # background = morphological closing (removes dark objects) on a 1/4-size image for speed
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = max(9, (min(small.shape[:2]) // 8) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cv2.subtract(bg, gray)
    t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t, 12)
    fg = ((diff > t) | (hsv[..., 1] > sat_thr)).astype(np.uint8) * 255
    if marker is not None:
        m = np.zeros_like(fg)
        cv2.fillConvexPoly(m, marker.astype(np.int32), 255)
        m = cv2.dilate(m, np.ones((25, 25), np.uint8))
        fg[m > 0] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    fg = _fill_holes(fg)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return fg


def _row_extents(mask_crop: np.ndarray):
    """Left/right silhouette edge for every row (NaN where the row is empty)."""
    h, w = mask_crop.shape
    has = mask_crop.any(axis=1)
    left = np.where(has, mask_crop.argmax(axis=1), np.nan).astype(float)
    right = np.where(has, w - 1 - mask_crop[:, ::-1].argmax(axis=1), np.nan).astype(float)
    return left, right


def _water_level(hsv_crop: np.ndarray, gray_crop: np.ndarray, mask_crop: np.ndarray,
                 left: np.ndarray, right: np.ndarray):
    """Top of the water column (row index in the crop) and a confidence in [0, 1]."""
    h, w = mask_crop.shape
    # central half of every row (vectorised): columns between left + 25% and right - 25% of the width
    cols = np.arange(w)[None, :]
    wdt = right - left
    a2 = (left + 0.25 * wdt)[:, None]
    b2 = (right - 0.25 * wdt)[:, None]
    central = (cols >= a2) & (cols <= b2) & (wdt[:, None] >= 6)
    sat = np.where(central, hsv_crop[..., 1].astype(np.float32), np.nan)
    gry = np.where(central, gray_crop.astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sat_rows = np.nanmedian(sat, axis=1)
            gray_rows = np.nanmedian(gry, axis=1)
    valid = np.where(np.isfinite(sat_rows))[0]
    if len(valid) < 10:
        return None, 0.0
    base_skip = max(2, int(0.04 * h))
    rows = valid[valid < h - base_skip]
    s = sat_rows[rows]
    lo, hi = np.percentile(s, 10), np.percentile(s, 90)
    if hi - lo > 35:                                   # dyed water: clear saturation contrast
        thr = (lo + hi) / 2
        water = s > thr
        # contiguous run of water rows from the bottom (allow small gaps)
        top, gap = None, 0
        for r, wflag in zip(rows[::-1], water[::-1]):
            if wflag:
                top, gap = r, 0
            else:
                gap += 1
                if gap > 4 and top is not None:
                    break
                if gap > 4 and top is None:
                    break
        if top is None:
            return None, 0.0
        above = s[rows < top]
        below = s[rows >= top]
        conf = float(np.clip((np.mean(below) - (np.mean(above) if len(above) else lo)) / 80.0, 0, 1))
        return int(top), conf
    if hi < 45:                                         # uniformly unsaturated: probably empty (or clear water)
        # clear water seen against a backlight is slightly darker than the air above it:
        # find the split row that maximises (mean above - mean below)  -- a 1-D step detector
        g = np.nan_to_num(gray_rows[rows], nan=np.nanmedian(gray_rows[rows]))
        lo_r, hi_r = int(0.12 * len(g)), int(0.92 * len(g))
        seg = g[lo_r:hi_r]
        if len(seg) > 10:
            cs = np.cumsum(seg)
            k = np.arange(3, len(seg) - 3)
            above = cs[k - 1] / k
            below = (cs[-1] - cs[k - 1]) / (len(seg) - k)
            contrast = above - below
            j = int(np.argmax(contrast))
            noise = np.median(np.abs(np.diff(seg))) + 0.5
            if contrast[j] > max(5.0, 3 * noise):         # a clear step = clear-water line
                return int(rows[lo_r + k[j]]), float(np.clip(contrast[j] / 30.0, 0.3, 0.8))
        return int(h - 1), 0.7                          # empty bowl
    return None, 0.0


def measure_frame(img_bgr: np.ndarray, marker_cm: float = C.MARKER_CM, proc_width: int = C.PROC_WIDTH,
                  min_area_frac: float = 0.003, t: float | None = None) -> FrameObs:
    """Detect all bowls in a frame, measure geometry and fill ratio of each."""
    scale = min(1.0, proc_width / img_bgr.shape[1])
    img = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    ppc, quad = find_marker(img, marker_cm)
    fg = foreground_mask(img, quad)
    H, W = fg.shape
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bowls = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_frac * W * H:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h < 25 or w < 15 or h > 0.95 * H:
            continue
        # reject the scale marker if it could not be decoded: square, high-contrast, colourless, and
        # 4 cm across when the scale is known (a wide container can also look square, but holds colour)
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and 0.8 < w / h < 1.25 and float(np.std(gray[y:y + h, x:x + w])) > 45:
            colourless = float(np.median(hsv[y:y + h, x:x + w, 1])) < 40
            size_ok = ppc is None or abs(max(w, h) / ppc - marker_cm) < 0.3 * marker_cm
            if colourless and size_ok:
                continue
        crop = np.zeros((h, w), np.uint8)
        cv2.drawContours(crop, [c - np.array([[x, y]])], -1, 1, thickness=cv2.FILLED)
        left, right = _row_extents(crop)
        widths = right - left
        D = float(np.nanmax(widths))
        axis = float(np.nanmedian((left + right) / 2)) + x
        prof = (widths / 2.0)[::-1]                        # from the base upward
        top_row, conf = _water_level(hsv[y:y + h, x:x + w], gray[y:y + h, x:x + w], crop, left, right)
        b = BowlObs((x, y, w, h), c, H_px=float(h), D_px=D, top_y=y, bottom_y=y + h - 1, axis_x=axis,
                    profile_px=np.nan_to_num(prof), ppc=ppc)
        if top_row is not None:
            b.water_top_y = y + top_row
            b.fill_ratio = float(np.clip((h - 1 - top_row) / max(h - 1, 1), 0, 1))
            b.level_conf = conf
        bowls.append(b)
    bowls.sort(key=lambda b: b.cx)
    return FrameObs(time.time() if t is None else t, scale, ppc, quad, bowls)


# ----------------------------------------------------------------------------- tracking over time
@dataclass
class Track:
    tid: int
    cx: float
    width: float
    last: BowlObs
    last_t: float
    hist: deque = field(default_factory=lambda: deque(maxlen=90))   # (t, fill_ratio, conf)

    def level(self, t_end: float | None = None, window: float = 0.5, min_conf: float = 0.3):
        """Robust fill ratio: median of confident measurements in [t_end - window, t_end]."""
        if t_end is None:
            t_end = self.last_t
        vals = [r for (tt, r, c) in self.hist if t_end - window <= tt <= t_end and c >= min_conf]
        if not vals:
            vals = [r for (tt, r, c) in self.hist if c >= min_conf][-5:]
        if not vals:
            return self.last.fill_ratio, 0.0
        return float(np.median(vals)), float(min(1.0, len(vals) / 5.0))


class BowlTracker:
    """Gives each bowl a stable id (left to right) and smooths its water level over time."""

    def __init__(self, max_missing_s: float = 3.0):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.max_missing = max_missing_s

    def update(self, obs: FrameObs) -> list[Track]:
        used = set()
        for b in obs.bowls:
            best, bd = None, 1e9
            for tr in self.tracks:
                if tr.tid in used:
                    continue
                d = abs(tr.cx - b.cx)
                if d < 0.6 * max(tr.width, b.bbox[2]) and d < bd:
                    best, bd = tr, d
            if best is None:
                best = Track(self.next_id, b.cx, b.bbox[2], b, obs.t)
                self.next_id += 1
                self.tracks.append(best)
            best.cx = 0.7 * best.cx + 0.3 * b.cx
            best.width = b.bbox[2]
            best.last, best.last_t = b, obs.t
            best.hist.append((obs.t, b.fill_ratio, b.level_conf))
            used.add(best.tid)
        self.tracks = [tr for tr in self.tracks if obs.t - tr.last_t < self.max_missing]
        self.tracks.sort(key=lambda tr: tr.cx)
        return self.tracks

    def relabel_left_to_right(self) -> None:
        for i, tr in enumerate(sorted(self.tracks, key=lambda tr: tr.cx)):
            tr.tid = i
        self.next_id = len(self.tracks)


# ----------------------------------------------------------------------------- motion evidence
def striker_tip(ref: np.ndarray, cur: np.ndarray, boxes: list, thr: int = 25):
    """Where is the striker touching? = lowest point of the region that is new compared with the reference.

    The striker enters from above; at the moment of impact its tip is the lowest pixel of the
    'new object' mask (frame difference against a frame from before the strike), restricted to
    the band of heights where bowls are. Returns (x, y) or None.
    """
    d = cv2.absdiff(cur, ref)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    top = int(min(y - 0.6 * h for (x, y, w, h) in boxes))
    bot = int(max(y + h for (x, y, w, h) in boxes))
    band = m[max(top, 0):bot]
    ys, xs = np.nonzero(band)
    if len(ys) < 15:
        return None
    k = np.argmax(ys)
    sel = ys >= ys[k] - 3                      # pixels at the very bottom of the striker
    return float(np.median(xs[sel])), float(ys[k] + max(top, 0))


def motion_evidence(frames: list, boxes: list, t_onset: float, before: float = 0.45, after: float = 0.08):
    """Which bowl did the striker touch? Locate the striker tip in the frames around the sound onset
    and score each bowl by how close the tip is to the bowl's rim (Gaussian in horizontal distance).

    frames: list of (t, gray, scale) in processing pixels; boxes: list of (x, y, w, h).
    Returns (probabilities over bowls, confidence in [0, 1]).
    """
    n = len(boxes)
    sel = [(t, g) for (t, g, s) in frames if t_onset - before <= t <= t_onset + after]
    if len(sel) < 2 or n == 0:
        return np.full(max(n, 1), 1.0 / max(n, 1))[:n], 0.0
    ref = sel[0][1]
    contact = [g for (t, g) in sel[1:] if t >= t_onset - 0.25] or [sel[-1][1]]
    logp = np.zeros(n)
    found = 0
    for g in contact:
        tip = striker_tip(ref, g, boxes)
        if tip is None:
            continue
        tx, ty = tip
        found += 1
        for i, (x, y, w, h) in enumerate(boxes):
            dx = max(x - tx, 0.0, tx - (x + w))           # 0 if the tip is above/at the bowl's width
            dy = max((y - 0.25 * h) - ty, 0.0, ty - (y + h))
            logp[i] += -0.5 * (dx / (0.15 * w)) ** 2 - 0.5 * (dy / (0.2 * h)) ** 2
    if found == 0:
        return np.full(n, 1.0 / n), 0.0
    logp /= found
    p = np.exp(logp - logp.max())
    p = (p + 1e-6) / (p + 1e-6).sum()
    srt = np.sort(p)[::-1]
    conf = float(np.clip((srt[0] - (srt[1] if n > 1 else 0)) * 1.2, 0, 1)) * min(1.0, found / 2)
    return p, conf

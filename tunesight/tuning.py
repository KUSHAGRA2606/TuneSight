"""Vision-guided tuning: from the camera's view of the bowls to "pour X ml into bowl k" for a raga scale."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import music
from .physics import FrenchCalibrator


@dataclass
class BowlTarget:
    bowl: int
    note: music.SwaraNote
    target_hz: float
    target_fill: float          # h/H that gives the target
    reachable: bool
    margin_cents: float


def bowl_ranges(calibs: list[FrenchCalibrator], max_fill: float = 0.9):
    """(lowest, highest) pitch each bowl can make: full to max_fill, and empty."""
    return [c.range_hz(max_fill) for c in calibs]


def auto_sa(ranges, raga: str, sa_min: float = 100.0, sa_max: float = 3000.0, step_cents: float = 2.0):
    """Choose Sa so that every target lies inside its bowl's range with the largest safety margin.

    Targets are assigned in ascending order to bowls sorted by their pitch range (lowest bowl -> lowest note).
    """
    n = len(ranges)
    order = sorted(range(n), key=lambda i: math.sqrt(ranges[i][0] * ranges[i][1]))
    best_sa, best_margin = None, -1e9
    sa = sa_min
    while sa <= sa_max:
        targets = music.raga_targets(raga, sa, n)
        margin = 1e9
        for k, i in enumerate(order):
            lo, hi = ranges[i]
            t = targets[k].freq
            m = min(music.cents(hi, t), music.cents(t, lo))
            margin = min(margin, m)
        if margin > best_margin:
            best_sa, best_margin = sa, margin
        sa *= 2 ** (step_cents / 1200)
    return best_sa, best_margin, order


def plan(calibs: list[FrenchCalibrator], raga: str, sa: float | None = None, max_fill: float = 0.9):
    ranges = bowl_ranges(calibs, max_fill)
    if sa is None:
        sa, _, order = auto_sa(ranges, raga)
    else:
        order = sorted(range(len(calibs)), key=lambda i: math.sqrt(ranges[i][0] * ranges[i][1]))
    targets = music.raga_targets(raga, sa, len(calibs))
    out = []
    for k, i in enumerate(order):
        lo, hi = ranges[i]
        t = targets[k]
        fill = calibs[i].height_for(t.freq)
        reach = bool(np.isfinite(fill) and 0 <= fill <= max_fill + 1e-6)
        out.append(BowlTarget(i, t, t.freq, fill, reach, min(music.cents(hi, t.freq), music.cents(t.freq, lo))))
    out.sort(key=lambda b: b.bowl)
    return sa, out


def pour_advice(current_fill: float, target_fill: float, volume_curve_ml: np.ndarray | None = None,
                tol: float = 0.01) -> str:
    if not np.isfinite(target_fill):
        return "unreachable"
    d = target_fill - current_fill
    if abs(d) <= tol:
        return "STOP - on target"
    if volume_curve_ml is not None and len(volume_curve_ml) > 2:
        n = len(volume_curve_ml) - 1
        v = lambda f: float(np.interp(f * n, np.arange(n + 1), volume_curve_ml))
        ml = v(target_fill) - v(current_fill)
        return f"pour {ml:.0f} ml" if ml > 0 else f"remove {-ml:.0f} ml"
    return "pour more" if d > 0 else "remove some"

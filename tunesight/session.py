"""Recorded sessions of your own bowls: images + strike sounds + which bowl was struck.

Layout (written by scripts/record_session.py):
    data/own/<session>/taps.csv         tap_id, bowl, image, wav, true_h_cm, note
    data/own/<session>/images/*.png
    data/own/<session>/audio/*.wav
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import audio as A
from . import config as C
from . import vision as V


@dataclass
class Tap:
    tap_id: int
    bowl: int
    image: Path
    wav: Path
    true_h_cm: float | None
    fill: float = float("nan")         # measured by vision
    fill_conf: float = 0.0
    H_cm: float | None = None
    pitch: float = float("nan")        # measured by audio
    snr_db: float = float("nan")


def load_session(folder: Path) -> list[Tap]:
    folder = Path(folder)
    taps = []
    with open(folder / "taps.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            th = r.get("true_h_cm", "")
            taps.append(Tap(int(r["tap_id"]), int(r["bowl"]), folder / r["image"], folder / r["wav"],
                            float(th) if th not in ("", None) else None))
    return taps


def measure_taps(taps: list[Tap], marker_cm: float = C.MARKER_CM) -> list[Tap]:
    cache: dict[Path, V.FrameObs] = {}
    for t in taps:
        if t.image not in cache:
            img = cv2.imread(str(t.image))
            cache[t.image] = V.measure_frame(img, marker_cm) if img is not None else None
        obs = cache[t.image]
        if obs is not None and t.bowl < len(obs.bowls):
            b = obs.bowls[t.bowl]
            t.fill, t.fill_conf, t.H_cm = b.fill_ratio, b.level_conf, b.H_cm
        x = A.load_audio(t.wav)
        an = A.analyze_strike(x)
        if an is not None:
            t.pitch, t.snr_db = an.pitch, an.snr_db
    return taps


def by_bowl(taps: list[Tap]) -> dict[int, list[Tap]]:
    out: dict[int, list[Tap]] = {}
    for t in taps:
        if np.isfinite(t.fill) and np.isfinite(t.pitch):
            out.setdefault(t.bowl, []).append(t)
    return out

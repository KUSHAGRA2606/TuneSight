"""Indian music theory used by the system: swaras, just-intonation ratios, raga scales, notation."""
from __future__ import annotations

import math
from dataclasses import dataclass

# 12 swaras of one saptak with common Hindustani just-intonation ratios.
# lower-case = komal (flat), "Ma" = tivra (sharp) ma.
SWARAS = [
    ("Sa", 1 / 1), ("re", 16 / 15), ("Re", 9 / 8), ("ga", 6 / 5), ("Ga", 5 / 4), ("ma", 4 / 3),
    ("Ma", 45 / 32), ("Pa", 3 / 2), ("dha", 8 / 5), ("Dha", 5 / 3), ("ni", 9 / 5), ("Ni", 15 / 8),
]
SWARA_RATIO = dict(SWARAS)

SWARA_DISPLAY = {
    "Sa": "Sa", "re": "re(k)", "Re": "Re", "ga": "ga(k)", "Ga": "Ga", "ma": "ma",
    "Ma": "Ma(t)", "Pa": "Pa", "dha": "dha(k)", "Dha": "Dha", "ni": "ni(k)", "Ni": "Ni",
}

# Ascending scales (aroha) of a few well known ragas / thaats.
RAGAS: dict[str, list[str]] = {
    "Bilawal (major)": ["Sa", "Re", "Ga", "ma", "Pa", "Dha", "Ni"],
    "Yaman": ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"],
    "Bhupali": ["Sa", "Re", "Ga", "Pa", "Dha"],
    "Durga": ["Sa", "Re", "ma", "Pa", "Dha"],
    "Malkauns": ["Sa", "ga", "ma", "dha", "ni"],
    "Bhairav": ["Sa", "re", "Ga", "ma", "Pa", "dha", "Ni"],
    "Kafi": ["Sa", "Re", "ga", "ma", "Pa", "Dha", "ni"],
}

WESTERN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def cents(f: float, ref: float) -> float:
    """Interval from ref to f in cents (1200 cents = one octave)."""
    return 1200.0 * math.log2(f / ref)


def western_name(f: float) -> str:
    midi = 69 + 12 * math.log2(f / 440.0)
    m = int(round(midi))
    return f"{WESTERN[m % 12]}{m // 12 - 1} ({(midi - m) * 100:+.0f}c)"


@dataclass
class SwaraNote:
    name: str      # e.g. "Pa"
    octave: int    # 0 = madhya saptak, +1 = taar, -1 = mandra
    freq: float    # exact target frequency

    @property
    def label(self) -> str:
        mark = "'" * self.octave if self.octave > 0 else "." * (-self.octave)
        return f"{self.name}{mark}"


def swara_freq(name: str, sa: float, octave: int = 0) -> float:
    return sa * SWARA_RATIO[name] * (2.0 ** octave)


def nearest_swara(f: float, sa: float, allowed: list[str] | None = None) -> tuple[SwaraNote, float]:
    """Nearest swara (optionally restricted to a raga) and the deviation in cents."""
    names = allowed if allowed else [n for n, _ in SWARAS]
    best, best_dev = None, 1e9
    base_oct = math.floor(math.log2(f / sa))
    for octv in (base_oct - 1, base_oct, base_oct + 1):
        for n in names:
            ft = swara_freq(n, sa, octv)
            dev = cents(f, ft)
            if abs(dev) < abs(best_dev):
                best, best_dev = SwaraNote(n, octv, ft), dev
    return best, best_dev


def raga_targets(raga: str, sa: float, n_notes: int) -> list[SwaraNote]:
    """First n_notes of the raga's ascending scale starting at Sa, continuing into the next octave."""
    scale = RAGAS[raga]
    out, octv = [], 0
    while len(out) < n_notes:
        for n in scale:
            out.append(SwaraNote(n, octv, swara_freq(n, sa, octv)))
            if len(out) == n_notes:
                break
        octv += 1
    return out

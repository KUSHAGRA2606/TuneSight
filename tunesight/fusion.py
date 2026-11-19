"""Audio-visual fusion: which bowl did I strike, and is the sound consistent with what the camera sees?

Appearance cannot answer this question: the bowls are identical. Two physical cues can:
  * audio cue  : the heard pitch vs. the pitch each bowl *should* have, predicted by the camera's
                 water level through the physics calibrator (Gaussian likelihood in cents);
  * motion cue : where the striker moved in the video just before the sound (frame differences).
They are combined with reliability-weighted log-linear pooling (a Bayesian cue-combination rule;
when one cue is unreliable its weight falls, i.e. the principle of inverse effectiveness).
A "none of the visible bowls" hypothesis catches sounds that do not fit any bowl.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class Identification:
    probs: np.ndarray           # posterior over [bowl_0 .. bowl_{n-1}, none]
    p_audio: np.ndarray         # audio-only posterior (same layout)
    p_motion: np.ndarray        # motion-only posterior (same layout)
    best: int                   # index of the bowl, or -1 for "none of the visible bowls"
    dev_cents: np.ndarray       # heard pitch minus predicted pitch of each bowl
    w_audio: float
    w_motion: float

    @property
    def confidence(self) -> float:
        return float(self.probs.max())


def _normalize_log(lp: np.ndarray) -> np.ndarray:
    lp = lp - lp.max()
    p = np.exp(lp)
    return p / p.sum()


def identify_struck(f_heard: float, pred_f: list[float], pred_std_cents: list[float],
                    motion_p: np.ndarray | None = None, motion_conf: float = 0.0, audio_conf: float = 1.0,
                    meas_cents: float = C.PITCH_NOISE_CENTS, floor_cents: float = 10.0,
                    none_prior: float = C.NOT_A_BOWL_PRIOR) -> Identification:
    n = len(pred_f)
    pf = np.asarray(pred_f, float)
    ps = np.asarray(pred_std_cents, float)
    dev = 1200 * np.log2(f_heard / pf)
    sig = np.sqrt(ps ** 2 + meas_cents ** 2 + floor_cents ** 2)
    la = np.concatenate([-0.5 * (dev / sig) ** 2 - np.log(sig * math.sqrt(2 * math.pi)),
                         [math.log(1.0 / 2400.0)]])                    # "none": uniform over +-1 octave
    prior = np.concatenate([np.full(n, (1 - none_prior) / n), [none_prior]])
    lm = np.full(n + 1, math.log(1.0 / n))
    if motion_p is not None and len(motion_p) == n:
        lm[:n] = np.log(np.clip(motion_p, 1e-6, 1))
    w_a = 0.2 + 0.8 * float(np.clip(audio_conf, 0, 1))
    w_m = float(np.clip(motion_conf, 0, 1)) if motion_p is not None else 0.0
    p_audio = _normalize_log(la + np.log(prior))
    p_motion = _normalize_log(lm + np.log(prior)) if w_m > 0 else np.concatenate([np.full(n, 1 / n), [0.0]])
    post = _normalize_log(w_a * la + w_m * lm + np.log(prior))
    k = int(np.argmax(post))
    return Identification(post, p_audio, p_motion, -1 if k == n else k, dev, w_a, w_m)


def consistency_check(f_heard: float, f_pred: float, std_cents: float, meas_cents: float = C.PITCH_NOISE_CENTS,
                      k: float = 3.0) -> tuple[bool, float]:
    """Is the heard pitch physically consistent with the prediction from the image? (|z| < k)."""
    z = 1200 * math.log2(f_heard / f_pred) / math.sqrt(std_cents ** 2 + meas_cents ** 2 + 10 ** 2)
    return abs(z) < k, z

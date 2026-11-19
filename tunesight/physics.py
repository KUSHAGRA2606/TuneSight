"""Physics that links what the camera sees to what the microphone hears.

Water loading of a struck bowl / glass (A. P. French, "In vino veritas", Am. J. Phys. 51, 1983):

    f(h) = f0 / sqrt(1 + beta * (h/H)^n),   n ~ 4

Rewriting it,   1/f^2 = 1/f0^2 + (beta/f0^2) * (h/H)^n   =  a + b * x,   with x = (h/H)^n

so every tap (x from the camera, f from the microphone) is one point on a straight line.
The calibrator below is Bayesian linear regression on that line: audio supervises vision online,
and after a few taps the camera alone predicts the pitch, with an uncertainty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from . import config as C

LN2 = math.log(2.0)


def french_freq(h_ratio, f0: float, beta: float, n: float = C.FRENCH_N):
    h_ratio = np.clip(np.asarray(h_ratio, dtype=float), 0.0, 1.2)
    return f0 / np.sqrt(1.0 + beta * h_ratio ** n)


def french_height(f_target: float, f0: float, beta: float, n: float = C.FRENCH_N) -> float:
    """Inverse model: fill ratio h/H that gives f_target (NaN if unreachable)."""
    if f_target > f0 or beta <= 0:
        return float("nan")
    x = ((f0 / f_target) ** 2 - 1.0) / beta
    return float(x ** (1.0 / n)) if x >= 0 else float("nan")


def fit_french(h_ratio, f, fit_n: bool = False, fit_r0: bool = False):
    """Nonlinear least-squares fit of f = f0 / sqrt(1 + beta u^n), u = (r - r0)/(1 - r0) (for reporting).

    Returns dict with f0, beta, n, r0, rmse_cents (and n_std when n is fitted).
    """
    h_ratio, f = np.asarray(h_ratio, float), np.asarray(f, float)

    def unpack(p):
        n = p[2] if fit_n else C.FRENCH_N
        r0 = p[-1] if fit_r0 else 0.0
        return p[0], p[1], n, r0

    def resid(p):
        f0, beta, n, r0 = unpack(p)
        u = np.clip((h_ratio - r0) / (1 - r0), 0, 1.2)
        return 1200 * np.log2(french_freq(u, f0, beta, n) / f)

    p0 = [f.max() * 1.02, 1.0] + ([4.0] if fit_n else []) + ([0.03] if fit_r0 else [])
    lb = [f.max() * 0.8, 1e-3] + ([1.0] if fit_n else []) + ([0.0] if fit_r0 else [])
    ub = [f.max() * 1.5, 50.0] + ([8.0] if fit_n else []) + ([0.2] if fit_r0 else [])
    r = optimize.least_squares(resid, p0, bounds=(lb, ub))
    rmse = float(np.sqrt(np.mean(r.fun ** 2)))
    f0, beta, n, r0 = unpack(r.x)
    out = {"f0": float(f0), "beta": float(beta), "n": float(n), "r0": float(r0), "rmse_cents": rmse}
    if fit_n:
        try:
            cov = np.linalg.inv(r.jac.T @ r.jac) * (rmse ** 2)
            out["n_std"] = float(np.sqrt(cov[2, 2]))
        except np.linalg.LinAlgError:
            out["n_std"] = float("nan")
    return out


N_GRID = np.arange(2.0, 7.01, 0.25)        # candidate exponents n
R0_GRID = np.arange(0.0, 0.121, 0.02)      # candidate base offsets (water below r0 does not load the wall)


@dataclass
class FrenchCalibrator:
    """Online Bayesian linear regression of 1/f^2 on x = u^n, with u = (r - r0) / (1 - r0).

    r  : fill ratio measured by the camera (outer base to outer rim)
    r0 : offset where the water starts loading the wall (glass base thickness), learned
    n  : French's exponent (theory: 4), learned
    For each candidate (n, r0) the regression is linear and its Bayesian evidence p(taps | n, r0) is
    available in closed form; the best-supported shape (with a prior centred on n = 4) is used.
    Prior: f0 roughly known, beta ~ 1 with wide spread.
    Noise: a tap's pitch error of sigma_c cents maps to std(1/f^2) = 2 ln2 (sigma_c/1200) / f^2.
    """
    n: float = C.FRENCH_N
    r0: float = 0.0
    learn_shape: bool = True
    noise_cents: float = C.PITCH_NOISE_CENTS
    fill_noise: float = 0.006            # std of the camera's fill ratio (vision noise)
    offset_cents: float = 0.0            # per-bowl pitch offset set by one verification tap
    f0_prior: float = 1000.0
    f0_prior_cents: float = 600.0        # very wide unless told otherwise
    beta_prior: float = 1.0
    beta_prior_std: float = 3.0
    fills: list = field(default_factory=list)
    fs: list = field(default_factory=list)

    def __post_init__(self):
        self._refit()

    # --- data
    def add(self, h_ratio: float, f: float) -> None:
        if not (np.isfinite(h_ratio) and np.isfinite(f) and f > 0):
            return
        self.fills.append(float(np.clip(h_ratio, 0, 1.2)))
        self.fs.append(float(f))
        self._refit()

    def reset(self) -> None:
        self.fills, self.fs = [], []
        self.n, self.r0 = C.FRENCH_N, 0.0
        self._refit()

    @property
    def n_taps(self) -> int:
        return len(self.fills)

    # --- inference
    def x_of(self, r, n=None, r0=None):
        n = self.n if n is None else n
        r0 = self.r0 if r0 is None else r0
        u = np.clip((np.asarray(r, float) - r0) / (1.0 - r0), 0.0, 1.2)
        return u ** n

    def _prior(self):
        a0 = 1.0 / self.f0_prior ** 2
        sa = 2 * LN2 * (self.f0_prior_cents / 1200) * a0
        b0 = self.beta_prior * a0
        sb = self.beta_prior_std * a0
        return np.array([a0, b0]), np.diag([sa ** 2, sb ** 2])

    def _blr(self, n, r0, with_evidence=False):
        m0, S0 = self._prior()
        r = np.asarray(self.fills, float)
        y = 1.0 / np.asarray(self.fs, float) ** 2
        s = 2 * LN2 * (self.noise_cents / 1200) * y
        Phi = np.stack([np.ones_like(r), self.x_of(r, n, r0)], axis=1) if len(r) else np.zeros((0, 2))
        P0 = np.linalg.inv(S0)
        Pm = P0 + (Phi.T / s ** 2) @ Phi
        cov = np.linalg.inv(Pm)
        mean = cov @ (P0 @ m0 + (Phi.T / s ** 2) @ y)
        if not with_evidence:
            return mean, cov, 0.0
        # closed-form log evidence using only 2x2 matrices
        ev = (-0.5 * len(y) * math.log(2 * math.pi) - np.sum(np.log(s))
              - 0.5 * np.linalg.slogdet(S0)[1] + 0.5 * np.linalg.slogdet(cov)[1]
              - 0.5 * (np.sum(y ** 2 / s ** 2) + m0 @ P0 @ m0 - mean @ Pm @ mean))
        return mean, cov, float(ev)

    def _refit(self) -> None:
        if self.learn_shape and self.n_taps >= 5 and np.ptp(self.fills) > 0.3:
            best = None
            for n in N_GRID:
                for r0 in R0_GRID:
                    _, _, ev = self._blr(n, r0, True)
                    score = ev - 0.5 * ((n - C.FRENCH_N) / 1.0) ** 2      # prior: n ~ N(4, 1)
                    if best is None or score > best[0]:
                        best = (score, n, r0)
            self.n, self.r0 = float(best[1]), float(best[2])
        self.mean, self.cov, _ = self._blr(self.n, self.r0)

    @property
    def f0(self) -> float:
        return float(1.0 / math.sqrt(max(self.mean[0], 1e-18)))

    @property
    def beta(self) -> float:
        return float(self.mean[1] / self.mean[0])

    def _f(self, r: float) -> float:
        y = float(self.mean[0] + self.mean[1] * float(self.x_of(r)))
        return 2 ** (self.offset_cents / 1200) / math.sqrt(max(y, 1e-18))

    def predict(self, h_ratio: float):
        """Predicted pitch (Hz) for a camera fill ratio and its 1-sigma uncertainty in cents
        (parameter uncertainty + vision noise propagated through the physics)."""
        x = float(self.x_of(h_ratio))
        phi = np.array([1.0, x])
        y = max(float(phi @ self.mean), 1e-18)
        vy = float(phi @ self.cov @ phi)
        f = 2 ** (self.offset_cents / 1200) / math.sqrt(y)
        std_param = 1200 / (2 * LN2) * math.sqrt(max(vy, 0.0)) / y
        d = 0.01
        slope = abs(1200 * math.log2(self._f(min(h_ratio + d, 1.2)) / self._f(max(h_ratio - d, 0.0)))) / (2 * d)
        return f, math.sqrt(std_param ** 2 + (slope * self.fill_noise) ** 2)

    def set_offset_from_tap(self, h_ratio: float, f_heard: float) -> float:
        """One verification tap on a specific bowl: identical-looking bowls differ slightly (wall thickness),
        which shifts every pitch of that bowl by a constant number of cents. Measure that shift."""
        base = self.offset_cents
        self.offset_cents = 0.0
        f0, _ = self.predict(h_ratio)
        self.offset_cents = float(1200 * math.log2(f_heard / f0)) if f0 > 0 else base
        return self.offset_cents

    def height_for(self, f_target: float) -> float:
        """Camera fill ratio that gives f_target (inverse physics)."""
        a, b = self.mean
        f_target = f_target / 2 ** (self.offset_cents / 1200)
        x = (1.0 / f_target ** 2 - a) / b if b > 0 else float("nan")
        if not np.isfinite(x) or x < 0:
            return float("nan")
        return float(self.r0 + x ** (1.0 / self.n) * (1.0 - self.r0))

    def range_hz(self, h_max: float = 0.95):
        return self.predict(h_max)[0], self.predict(0.0)[0]

    def law_points(self):
        """(x, 1/f^2) of the taps and the fitted straight line, for plotting the learned physics."""
        x = self.x_of(np.asarray(self.fills, float))
        y = 1.0 / np.asarray(self.fs, float) ** 2
        xx = np.linspace(0, max(0.5, float(x.max()) * 1.1 if len(x) else 1.0), 50)
        return x, y, xx, self.mean[0] + self.mean[1] * xx

    # --- persistence
    def to_dict(self) -> dict:
        return {"n": self.n, "r0": self.r0, "learn_shape": self.learn_shape, "noise_cents": self.noise_cents,
                "fill_noise": self.fill_noise, "f0_prior": self.f0_prior, "f0_prior_cents": self.f0_prior_cents,
                "beta_prior": self.beta_prior, "beta_prior_std": self.beta_prior_std,
                "fills": self.fills, "fs": self.fs, "offset_cents": self.offset_cents}

    @classmethod
    def from_dict(cls, d: dict) -> "FrenchCalibrator":
        return cls(**d)


# ----------------------------------------------------------------------------- scaling law
def scaling_fit(L, f):
    """log f = c + slope * log L; for geometrically similar objects of one material, slope = -1."""
    lx, ly = np.log(np.asarray(L, float)), np.log(np.asarray(f, float))
    slope, c = np.polyfit(lx, ly, 1)
    pred = c + slope * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-18)
    return {"slope": float(slope), "intercept": float(c), "r2": float(r2)}


# ----------------------------------------------------------------------------- volumes
def volume_curve_ml(profile_r_cm: np.ndarray, dy_cm: float, wall_cm: float = 0.2) -> np.ndarray:
    """Cumulative inner volume (ml) from the bottom, treating the bowl as a solid of revolution.

    profile_r_cm[i] is the outer half-width at height i*dy_cm above the base.
    """
    r_in = np.clip(np.asarray(profile_r_cm, float) - wall_cm, 0, None)
    return np.cumsum(np.pi * r_in ** 2 * dy_cm)

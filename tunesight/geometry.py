"""Container shape from the camera, and French's thin-wall physics across different containers.

The camera sees each container's outline. With the ArUco scale it gives the radius at every height,
the rim and base radius, the height H and the capacity in ml.

French's model of a thin glass shell (A. P. French, Am. J. Phys. 51, 1983) says how the water law
    f(r) = f0 / sqrt(1 + beta * r^n)
depends on the container's shape (radius R, height H, wall thickness a):

    empty pitch     f0   = (1 / 2 pi) * sqrt(3E / 5 rho_g) * (a / R^2) * sqrt(1 + (4/3) (R/H)^4)
    water effect    beta = alpha * rho_water * R / (5 rho_g a)

A camera cannot see the wall thickness a, so for a *family* of containers made the same way (same
material and wall thickness) the unknowns collapse into two constants shared by the family:

    f0 = C_f * sqrt(1 + (4/3)(R/H)^4) / R^2,      beta = C_b * R

Calibrating a few containers by tapping fixes C_f and C_b; a new container's whole pitch curve then
follows from its measured shape with no taps at all. Run backwards, the law turns a tap's pitch into
a water level, so the level of an opaque container can be heard instead of seen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config as C
from .physics import french_height

# glass and water (SI units)
GLASS_E = 70e9            # Young's modulus (Pa)
GLASS_RHO = 2500.0        # density of glass (kg/m^3)
WATER_RHO = 1000.0        # density of water (kg/m^3)
ALPHA = 1.6               # order-one constant of French's loading term (sets beta ~ 3 for a 7 cm tumbler)

WALL_CM = 0.2             # assumed wall thickness seen in the outline (cm): outer edge -> mid-wall -> inner
BASE_FRAC = 0.05          # assumed solid base: the bottom 5 % of the height holds no water
R0_FAMILY = 0.05          # where the water starts to load the wall, as a fill ratio (the base)


@dataclass
class Shape:
    """A container's shape as the camera measures it (cm, ml)."""
    H_cm: float               # outer height, base to rim
    R_top_cm: float           # mid-wall radius at the rim
    R_bot_cm: float           # mid-wall radius at the base
    capacity_ml: float        # inner volume above the base

    @property
    def R_mean_cm(self) -> float:
        return 0.5 * (self.R_top_cm + self.R_bot_cm)

    @property
    def taper(self) -> float:
        return self.R_bot_cm / self.R_top_cm if self.R_top_cm > 0 else float("nan")

    def to_dict(self) -> dict:
        return {"H_cm": self.H_cm, "R_top_cm": self.R_top_cm, "R_bot_cm": self.R_bot_cm,
                "R_mean_cm": self.R_mean_cm, "capacity_ml": self.capacity_ml}


def shape_from_profile(profile_px: np.ndarray, ppc: float, wall_cm: float = WALL_CM,
                       base_frac: float = BASE_FRAC) -> Shape | None:
    """Fit straight walls to the outline's half-widths (one per pixel row, from the base upward).

    The middle of the outline (10 % to 88 % of the height) is used, so the solid base and the rim
    bead do not bias the wall line. The capacity integrates the inner radius as a solid of revolution.
    """
    prof = np.asarray(profile_px, float)
    n = len(prof)
    if not ppc or n < 20 or not np.isfinite(prof).any():
        return None
    y = (np.arange(n) + 0.5) / ppc                     # height above the outer base (cm)
    r_mid = prof / ppc - wall_cm / 2                    # outer edge -> middle of the wall (cm)
    lo, hi = int(0.10 * n), int(0.88 * n)
    ok = np.isfinite(r_mid[lo:hi]) & (r_mid[lo:hi] > 0)
    if ok.sum() < 10:
        return None
    k, c = np.polyfit(y[lo:hi][ok], r_mid[lo:hi][ok], 1)
    # the rim seen slightly from above is an ellipse: rows above its centre line are narrower than the
    # wall line. Count those rows from the top and end the container at the rim's centre line.
    cap_rows = 0
    for i in range(n - 1, hi, -1):
        if np.isfinite(r_mid[i]) and r_mid[i] < (c + k * y[i]) - 1.5 / ppc:
            cap_rows += 1
        else:
            break
    n_body = n - cap_rows
    H = n_body / ppc
    inner = np.clip(prof[:n_body] / ppc - wall_cm, 0, None)
    cap = float(np.sum(np.pi * inner[int(base_frac * n_body):] ** 2) / ppc)
    return Shape(H_cm=float(H), R_top_cm=float(c + k * H), R_bot_cm=float(c), capacity_ml=cap)


def shape_from_obs(bowl_obs, wall_cm: float = WALL_CM) -> Shape | None:
    """Shape of one container from the vision module's observation (needs the ArUco scale)."""
    if bowl_obs is None or not bowl_obs.ppc or bowl_obs.profile_px is None:
        return None
    return shape_from_profile(bowl_obs.profile_px, bowl_obs.ppc, wall_cm)


def frustum_capacity_ml(R_top_cm: float, R_bot_cm: float, H_cm: float, wall_cm: float = WALL_CM,
                        base_frac: float = BASE_FRAC) -> float:
    """Inner volume of a straight-walled (truncated cone) container above its base."""
    r1, r2 = R_bot_cm - wall_cm / 2, R_top_cm - wall_cm / 2
    h = H_cm * (1 - base_frac)
    r1 = r1 + (r2 - r1) * base_frac                    # radius where the base ends
    return float(math.pi * h / 3 * (r1 * r1 + r1 * r2 + r2 * r2))


# ----------------------------------------------------------------------------- French's thin-wall physics
def shape_factor_f0(R_cm: float, H_cm: float) -> float:
    """How the empty pitch depends on shape: sqrt(1 + (4/3)(R/H)^4) / R^2 (R, H in cm)."""
    return math.sqrt(1.0 + (4.0 / 3.0) * (R_cm / H_cm) ** 4) / R_cm ** 2


def french_f0(R_cm: float, H_cm: float, a_mm: float, E: float = GLASS_E, rho_g: float = GLASS_RHO) -> float:
    """Empty-container pitch (Hz) of a thin glass shell of mean radius R, height H, wall a."""
    R, H, a = R_cm / 100.0, H_cm / 100.0, a_mm / 1000.0
    w2 = (3.0 * E / (5.0 * rho_g)) * (a * a / R ** 4) * (1.0 + (4.0 / 3.0) * (R / H) ** 4)
    return math.sqrt(w2) / (2.0 * math.pi)


def french_beta(R_cm: float, a_mm: float, alpha: float = ALPHA, rho_l: float = WATER_RHO,
                rho_g: float = GLASS_RHO) -> float:
    """How strongly water lowers the pitch: alpha * rho_water * R / (5 rho_glass a)."""
    return alpha * rho_l * (R_cm / 100.0) / (5.0 * rho_g * a_mm / 1000.0)


@dataclass
class FamilyLaw:
    """One law for a family of containers: f0 and beta from the measured shape, two shared constants."""
    C_f: float
    C_b: float
    n: float = C.FRENCH_N
    r0: float = R0_FAMILY

    @classmethod
    def fit(cls, shapes: list[Shape], f0s, betas, n: float = C.FRENCH_N, r0: float = R0_FAMILY) -> "FamilyLaw":
        """Geometric means of f0 / shape factor and beta / R over calibrated containers."""
        sf = np.array([shape_factor_f0(s.R_mean_cm, s.H_cm) for s in shapes])
        R = np.array([s.R_mean_cm for s in shapes])
        C_f = float(np.exp(np.mean(np.log(np.asarray(f0s, float) / sf))))
        C_b = float(np.exp(np.mean(np.log(np.asarray(betas, float) / R))))
        return cls(C_f, C_b, n, r0)

    def predict(self, shape: Shape) -> tuple[float, float]:
        """(f0, beta) of a container from its shape alone (zero taps)."""
        return self.C_f * shape_factor_f0(shape.R_mean_cm, shape.H_cm), self.C_b * shape.R_mean_cm

    def pitch(self, shape: Shape, fill_ratio, offset_cents: float = 0.0):
        """Predicted pitch (Hz) at a camera fill ratio; offset_cents comes from an optional check tap."""
        f0, beta = self.predict(shape)
        u = np.clip((np.asarray(fill_ratio, float) - self.r0) / (1 - self.r0), 0, 1.2)
        return f0 * 2 ** (offset_cents / 1200) / np.sqrt(1 + beta * u ** self.n)


def radius_exponent(shapes: list[Shape], f0s) -> float:
    """Fitted p in f0 ~ R^p (after removing the small height term): French predicts p = -2."""
    R = np.array([s.R_mean_cm for s in shapes])
    corr = np.array([math.sqrt(1 + (4 / 3) * (s.R_mean_cm / s.H_cm) ** 4) for s in shapes])
    p, _ = np.polyfit(np.log(R), np.log(np.asarray(f0s, float) / corr), 1)
    return float(p)


def fill_from_pitch(f_heard: float, f0: float, beta: float, n: float = C.FRENCH_N, r0: float = R0_FAMILY) -> float:
    """Water level (camera-style fill ratio) that makes a container sound at f_heard.

    At or above the empty pitch the container is taken as empty (0). Near empty the law is flat
    (r^4 is tiny), so small pitch errors there map to large level errors: a 'dead zone'.
    """
    if not np.isfinite(f_heard) or f_heard >= f0:
        return 0.0
    u = french_height(f_heard, f0, beta, n)
    return float(r0 + (1 - r0) * u) if np.isfinite(u) else float("nan")


def cents_per_percent_fill(fill_ratio, beta: float, n: float = C.FRENCH_N, r0: float = R0_FAMILY):
    """How many cents the pitch falls when the fill rises by 1 % of the height: the law's sensitivity."""
    r = np.asarray(fill_ratio, float)
    u = np.clip((r - r0) / (1 - r0), 1e-6, 1.2)
    # f ~ (1 + beta u^n)^(-1/2)  ->  d(cents)/du = -(1200 / ln 2) * (n beta u^(n-1)) / (2 (1 + beta u^n))
    dcdu = (1200 / math.log(2)) * n * beta * u ** (n - 1) / (2 * (1 + beta * u ** n))
    return dcdu * 0.01 / (1 - r0)

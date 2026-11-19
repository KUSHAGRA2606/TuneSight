"""Global constants and paths for TuneSight."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ---------------- audio ----------------
SR = 48000                 # sampling rate used everywhere (ObjectFolder Real is 48 kHz)
FMIN = 150.0               # lowest modal frequency we search for (Hz)
FMAX = 8000.0              # highest modal frequency we search for (Hz)
STRIKE_SKIP_MS = 4.0       # skip the contact transient (the "source") before analysing the ringing
STRIKE_DUR_S = 1.0         # length of ringing analysed per strike
PITCH_REL_DB = -20.0       # pitch = lowest spectral peak within this many dB of the strongest peak
LPC_ORDER = 32             # order of the all-pole model of the ringing object

# ---------------- vision ----------------
ARUCO_DICT = "DICT_4X4_50"
MARKER_CM = 4.0            # printed ArUco marker side length (cm)
PROC_WIDTH = 960           # frames are resized to this width before measurement

# ---------------- physics ----------------
FRENCH_N = 4.0             # exponent in f(h) = f0 / sqrt(1 + beta (h/H)^n)  (A.P. French, 1983)
PITCH_NOISE_CENTS = 8.0    # assumed measurement noise of a single tap's pitch (cents)

# ---------------- fusion ----------------
NOT_A_BOWL_PRIOR = 0.05    # prior probability that a heard strike came from none of the visible bowls

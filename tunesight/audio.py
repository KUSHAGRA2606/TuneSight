"""Audio model: a struck object as a source-filter (all-pole) system.

The strike is a short force pulse (the *source*); the object rings in a set of damped modes
(the *filter*). Its impulse response is a sum of damped sinusoids, i.e. an all-pole (AR) system,
exactly the model used for the vocal tract in linear prediction (LPC).

We estimate the modes in two independent ways and cross-check them:
  1. FFT peak picking (zero-padded, Hann window, parabolic interpolation) + per-mode decay from the STFT.
  2. LPC: poles z_k = r_k e^{j w_k} of the all-pole fit give f_k = w_k fs / 2pi and sigma_k = -fs ln r_k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from . import config as C

EPS = 1e-12
DB_PER_NEPER = 20.0 / math.log(10.0)  # 8.686: power-dB slope = -8.686 * sigma


# ----------------------------------------------------------------------------- I/O helpers
def load_audio(path, sr: int = C.SR) -> np.ndarray:
    import soundfile as sf

    x, fs = sf.read(str(path), always_2d=True)
    x = x.mean(axis=1).astype(np.float64)
    if fs != sr:
        g = math.gcd(int(fs), int(sr))
        x = signal.resample_poly(x, sr // g, int(fs) // g)
    return x


def save_audio(path, x: np.ndarray, sr: int = C.SR) -> None:
    import soundfile as sf

    sf.write(str(path), np.asarray(x, dtype=np.float32), sr)


def highpass(x: np.ndarray, sr: int, fc: float = 100.0) -> np.ndarray:
    sos = signal.butter(4, fc, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x)


# ----------------------------------------------------------------------------- onsets
def energy_envelope_db(x: np.ndarray, sr: int, hop_ms: float = 5.0, win_ms: float = 10.0):
    hop = max(1, int(sr * hop_ms / 1000))
    win = max(hop, int(sr * win_ms / 1000))
    n = 1 + max(0, (len(x) - win) // hop)
    if n <= 0:
        return np.zeros(0), hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1) + EPS)
    return 20 * np.log10(rms + EPS), hop


def detect_onsets(x: np.ndarray, sr: int = C.SR, rise_db: float = 15.0, min_gap_s: float = 0.25,
                  lookback_ms: float = 40.0) -> list[int]:
    """Strike onsets = frames whose energy jumps by rise_db above the recent minimum and the noise floor."""
    y = highpass(x, sr)
    env, hop = energy_envelope_db(y, sr)
    if len(env) == 0:
        return []
    floor = np.percentile(env, 20)
    lb = max(1, int(lookback_ms / 5.0))
    onsets, last = [], -1e9
    for i in range(lb, len(env)):
        recent_min = env[i - lb:i].min()
        if env[i] - recent_min > rise_db and env[i] > floor + rise_db:
            t = i * hop
            if t - last > min_gap_s * sr:
                # refine: first frame in the look-back window that rose above recent_min + 6 dB
                j = i
                while j > i - lb and env[j - 1] > recent_min + 6:
                    j -= 1
                onsets.append(j * hop)
                last = t
    return onsets


class StreamingOnsetDetector:
    """Onset detector for a live audio stream (used by the demo app)."""

    def __init__(self, sr: int = C.SR, rise_db: float = 15.0, min_gap_s: float = 0.3, hop_ms: float = 5.0):
        self.sr, self.rise_db, self.min_gap = sr, rise_db, int(min_gap_s * sr)
        self.hop = int(sr * hop_ms / 1000)
        self.hist: list[float] = []
        self.floor = -80.0
        self.pos = 0
        self.last = -10 ** 12
        self._carry = np.zeros(0)
        self.sos = signal.butter(4, 100.0, btype="highpass", fs=sr, output="sos")
        self.zi = signal.sosfilt_zi(self.sos) * 0.0

    def process(self, block: np.ndarray) -> list[int]:
        """Feed a block; returns absolute sample indices of detected onsets."""
        y, self.zi = signal.sosfilt(self.sos, block, zi=self.zi)
        buf = np.concatenate([self._carry, y])
        out = []
        n = len(buf) // self.hop
        for k in range(n):
            frame = buf[k * self.hop:(k + 1) * self.hop]
            e = 20 * math.log10(math.sqrt(float(np.mean(frame ** 2)) + EPS))
            abs_idx = self.pos + k * self.hop
            recent = self.hist[-8:]
            if recent:
                rmin = min(recent)
                if e - rmin > self.rise_db and e > self.floor + self.rise_db and abs_idx - self.last > self.min_gap:
                    out.append(abs_idx)
                    self.last = abs_idx
            self.hist.append(e)
            if len(self.hist) > 400:
                self.hist = self.hist[-400:]
            # slowly tracking noise floor (20th percentile of the last 2 s)
            if len(self.hist) % 20 == 0:
                self.floor = float(np.percentile(self.hist, 20))
        self._carry = buf[n * self.hop:]
        self.pos += n * self.hop
        return out


# ----------------------------------------------------------------------------- spectra and modes
def spectrum_db(seg: np.ndarray, sr: int, pad: int = 2):
    nfft = int(2 ** math.ceil(math.log2(len(seg)))) * pad
    w = signal.windows.hann(len(seg), sym=False)
    X = np.fft.rfft(seg * w, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    return f, 20 * np.log10(np.abs(X) + EPS), nfft


def local_floor_db(db: np.ndarray, df: float, width_hz: float = 120.0) -> np.ndarray:
    """Local noise floor of a spectrum: median over blocks of width_hz, interpolated back to every bin."""
    blk = max(4, int(width_hz / df))
    n = len(db) // blk
    med = np.median(db[: n * blk].reshape(n, blk), axis=1)
    centers = (np.arange(n) + 0.5) * blk
    return np.interp(np.arange(len(db)), centers, med)


def fft_peaks(seg: np.ndarray, sr: int = C.SR, fmin: float = C.FMIN, fmax: float = C.FMAX,
              max_peaks: int = 10, dyn_range_db: float = 50.0, prominence_db: float = 10.0,
              above_floor_db: float = 18.0, merge_cents: float = 60.0):
    """Modal frequencies by peak picking on a zero-padded spectrum with parabolic interpolation.

    A peak must stand out from the local noise floor (a ringing mode is a tall, narrow line), and
    peaks closer than merge_cents (room-reverb ripples around one mode) are merged into the strongest.
    """
    f, db, nfft = spectrum_db(seg, sr)
    df = sr / nfft
    band = np.where((f >= fmin) & (f <= fmax))[0]
    if len(band) < 5:
        return np.zeros(0), np.zeros(0), (f, db)
    sub = db[band]
    floor = local_floor_db(sub, df)
    min_dist = max(1, int(15.0 / df))
    pk, _ = signal.find_peaks(sub, prominence=prominence_db, distance=min_dist)
    pk = pk[sub[pk] > floor[pk] + above_floor_db]
    if len(pk) == 0:
        return np.zeros(0), np.zeros(0), (f, db)
    pk = pk[sub[pk] >= sub[pk].max() - dyn_range_db]
    pk = pk[np.argsort(sub[pk])[::-1]]                  # strongest first
    kept = []
    for p in pk:
        fp = f[band[p]]
        if all(abs(1200 * math.log2(fp / f[band[q]])) > merge_cents for q in kept):
            kept.append(p)
        if len(kept) == max_peaks:
            break
    freqs, mags = [], []
    for p in kept:
        k = band[p]
        a, b, c = db[k - 1], db[k], db[k + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if abs(den) > EPS else 0.0
        d = float(np.clip(d, -0.5, 0.5))
        freqs.append((k + d) * sr / nfft)
        mags.append(b - 0.25 * (a - c) * d)
    order = np.argsort(freqs)
    return np.array(freqs)[order], np.array(mags)[order], (f, db)


def _amplitude_db(x: np.ndarray, sr: int, freqs: np.ndarray, bw_hz: float = 8.0, with_line: bool = False):
    """Amplitude-normalised level (dB) of x around each frequency (max over +-bw_hz).

    Normalising by the window sum makes a steady tone read the same level in windows of any length,
    so a line that is equally strong before and after the strike is recognised as background.
    With with_line=True also returns how far that level stands above the local median (a 'line' strength).
    """
    w = signal.windows.hann(len(x), sym=False)
    nfft = max(int(2 ** math.ceil(math.log2(len(x)))) * 2, 8192)
    X = np.abs(np.fft.rfft(x * w, nfft)) / (w.sum() / 2)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    Xdb = 20 * np.log10(X + EPS)
    out = np.empty(len(freqs))
    line = np.empty(len(freqs))
    for i, fk in enumerate(freqs):
        sel = (f >= fk - bw_hz) & (f <= fk + bw_hz)
        out[i] = Xdb[sel].max() if sel.any() else -300.0
        ctx = (f >= fk - 150) & (f <= fk + 150)
        line[i] = out[i] - np.median(Xdb[ctx]) if ctx.any() else 0.0
    return (out, line) if with_line else out


def stft_decay_rates(seg: np.ndarray, sr: int, freqs: np.ndarray, n_fft: int = 4096, hop: int = 480,
                     drop_db: float = 40.0) -> np.ndarray:
    """Amplitude decay rate sigma (1/s) of each mode from the slope of its band energy (dB) over time."""
    if len(freqs) == 0:
        return np.zeros(0)
    f, t, Z = signal.stft(seg, fs=sr, nperseg=min(n_fft, len(seg)), noverlap=min(n_fft, len(seg)) - hop,
                          boundary=None, padded=False)
    P = np.abs(Z) ** 2
    df = f[1] - f[0]
    sig = np.full(len(freqs), np.nan)
    for i, fk in enumerate(freqs):
        bw = max(2 * df, 0.01 * fk)
        rows = (f >= fk - bw) & (f <= fk + bw)
        if not rows.any():
            continue
        e = 10 * np.log10(P[rows].sum(axis=0) + EPS)
        if len(e) < 6:
            continue
        start = int(np.argmax(e[: max(3, len(e) // 4)]))
        tail = e[int(0.85 * len(e)):]
        noise = np.median(tail) if len(tail) else e.min()
        stop_level = max(e[start] - drop_db, noise + 8)
        stop = start + 1
        while stop < len(e) and e[stop] > stop_level:
            stop += 1
        if stop - start < 4:
            continue
        slope, _ = np.polyfit(t[start:stop], e[start:stop], 1)
        if slope < 0:
            sig[i] = -slope / DB_PER_NEPER
    return sig


@dataclass
class LPCResult:
    sr: int
    a: np.ndarray            # predictor polynomial A(z), a[0] = 1
    roots: np.ndarray        # all poles (complex)
    freqs: np.ndarray        # resonant poles kept as modes
    sigmas: np.ndarray
    amps: np.ndarray


def burg_lpc(x: np.ndarray, order: int) -> np.ndarray:
    """Burg's method for the all-pole predictor A(z) = 1 + a1 z^-1 + ... (a[0] = 1).

    At each stage the reflection coefficient k minimises the sum of forward and backward
    prediction-error energies; the predictor is then updated with the Levinson recursion.
    """
    ef = np.asarray(x, dtype=np.float64).copy()
    eb = ef.copy()
    a = np.array([1.0])
    for _ in range(order):
        efp, ebp = ef[1:], eb[:-1]
        den = np.dot(efp, efp) + np.dot(ebp, ebp)
        if den <= EPS:
            break
        k = -2.0 * np.dot(ebp, efp) / den
        ef, eb = efp + k * ebp, ebp + k * efp
        a_ext = np.concatenate([a, [0.0]])
        a = a_ext + k * a_ext[::-1]
    return a


def covariance_lpc(x: np.ndarray, order: int, rank: int | None = None) -> np.ndarray:
    """Covariance-method (forward-only, least-squares) LPC:  x[n] = -sum_k a_k x[n-k],  n = p..N-1.

    Unlike the autocorrelation method or Burg's forward-backward method, it applies no window and no
    stationarity assumption, so the poles of *decaying* sinusoids are not biased. With `rank`, the
    data matrix is truncated to its strongest singular vectors (Kumaresan-Tufts) to reject noise.
    """
    x = np.asarray(x, dtype=np.float64)
    p = order
    X = np.lib.stride_tricks.sliding_window_view(x, p)[:-1][:, ::-1]    # rows: x[n-1] ... x[n-p]
    y = x[p:]
    if rank is None:
        coef, *_ = np.linalg.lstsq(X, -y, rcond=None)
    else:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(rank, len(s))
        coef = -(Vt[:r].T @ ((U[:, :r].T @ y) / s[:r]))
    return np.concatenate([[1.0], coef])


def lpc_modes(seg: np.ndarray, sr: int = C.SR, order: int = C.LPC_ORDER, fmin: float = C.FMIN,
              fmax: float = C.FMAX, max_dur: float = 0.4, rmin: float = 0.99, rank: int | None = 16) -> LPCResult:
    """All-pole (LPC) model of the ringing; pole angles give mode frequencies, radii give damping."""
    sr_l = int(min(sr, max(2.5 * fmax, 8000)))
    g = math.gcd(sr, sr_l)
    y = seg[: int(max_dur * sr)]
    y = signal.resample_poly(y, sr_l // g, sr // g) if sr_l != sr else y.copy()
    y = y / (np.max(np.abs(y)) + EPS)
    a = covariance_lpc(y, order, rank)
    roots = np.roots(a)
    up = roots[np.imag(roots) > 0]
    r = np.abs(up)
    fr = np.angle(up) * sr_l / (2 * np.pi)
    sg = -np.log(np.clip(r, EPS, 1.0)) * sr_l
    keep = (fr >= fmin) & (fr <= fmax) & (r >= rmin) & (r < 1.0)
    fr, sg = fr[keep], sg[keep]
    order_ = np.argsort(fr)
    fr, sg = fr[order_], sg[order_]
    # amplitudes by least squares on the first 0.2 s with the estimated damped sinusoids as a basis
    n = min(len(y), int(0.2 * sr_l))
    tt = np.arange(n) / sr_l
    amps = np.zeros(len(fr))
    if len(fr):
        B = np.concatenate([np.exp(-sg[None, :] * tt[:, None]) * np.cos(2 * np.pi * fr[None, :] * tt[:, None]),
                            np.exp(-sg[None, :] * tt[:, None]) * np.sin(2 * np.pi * fr[None, :] * tt[:, None])], axis=1)
        coef, *_ = np.linalg.lstsq(B, y[:n], rcond=None)
        k = len(fr)
        amps = np.hypot(coef[:k], coef[k:])
    return LPCResult(sr_l, a, roots, fr, sg, amps)


# ----------------------------------------------------------------------------- full strike analysis
@dataclass
class StrikeAnalysis:
    onset_s: float
    freqs: np.ndarray              # modal frequencies from FFT (Hz)
    mags_db: np.ndarray
    sigma: np.ndarray              # amplitude decay rates from STFT (1/s)
    sigma_lpc: np.ndarray          # decay rates of matching LPC poles (1/s), NaN if no match
    freq_lpc: np.ndarray           # frequencies of matching LPC poles (Hz)
    pitch: float                   # perceived pitch: lowest strong mode (Hz)
    pitch_idx: int
    snr_db: float
    confidence: float
    spec_f: np.ndarray = field(repr=False)
    spec_db: np.ndarray = field(repr=False)
    lpc: LPCResult | None = field(default=None, repr=False)

    @property
    def eta(self) -> np.ndarray:
        """Loss factor eta = sigma / (pi f); quality factor Q = 1/eta."""
        return self.sigma / (np.pi * self.freqs)

    @property
    def ratios(self) -> np.ndarray:
        return self.freqs / self.pitch if self.pitch > 0 else self.freqs * np.nan

    @property
    def t60(self) -> float:
        s = self.sigma[self.pitch_idx] if self.pitch_idx >= 0 else np.nan
        return 6.91 / s if s and s > 0 else float("nan")

    def summary(self) -> dict:
        return {
            "pitch_hz": round(float(self.pitch), 2),
            "modes_hz": [round(float(v), 1) for v in self.freqs],
            "sigma_stft": [None if np.isnan(v) else round(float(v), 2) for v in self.sigma],
            "sigma_lpc": [None if np.isnan(v) else round(float(v), 2) for v in self.sigma_lpc],
            "snr_db": round(float(self.snr_db), 1),
            "confidence": round(float(self.confidence), 2),
        }


def analyze_strike(x: np.ndarray, sr: int = C.SR, onset: int | None = None, dur_s: float = C.STRIKE_DUR_S,
                   fmin: float = C.FMIN, fmax: float = C.FMAX, use_lpc: bool = True,
                   pitch_rel_db: float = C.PITCH_REL_DB) -> StrikeAnalysis | None:
    """Analyse one strike: find onset, skip the contact transient, then estimate modes and damping."""
    x = np.asarray(x, dtype=np.float64)
    if onset is None:
        ons = detect_onsets(x, sr)
        if ons:
            onset = ons[0]
        else:  # fall back to the loudest point
            env, hop = energy_envelope_db(x, sr)
            if len(env) == 0:
                return None
            onset = int(np.argmax(env)) * hop
    start = onset + int(C.STRIKE_SKIP_MS * sr / 1000)
    seg = x[start:start + int(dur_s * sr)]
    if len(seg) < int(0.1 * sr):
        return None
    pre = x[max(0, onset - int(0.2 * sr)):max(0, onset - int(0.01 * sr))]
    noise_rms = np.sqrt(np.mean(pre ** 2)) if len(pre) > 100 else 1e-6
    sig_rms = np.sqrt(np.mean(seg[: int(0.1 * sr)] ** 2))
    snr = 20 * np.log10((sig_rms + EPS) / (noise_rms + EPS))

    freqs, mags, (sf_, sdb) = fft_peaks(seg, sr, fmin, fmax)
    if len(freqs) == 0:
        return None
    sigma = stft_decay_rates(seg, sr, freqs)

    # a mode must START at the strike and DECAY: reject lines already present before the onset
    # (mains hum, fans, stand rumble) and lines that do not die away
    keep = np.ones(len(freqs), bool)
    if len(pre) >= int(0.02 * sr):
        amp_seg = _amplitude_db(seg, sr, freqs)
        amp_pre, line_pre = _amplitude_db(pre, sr, freqs, with_line=True)
        # reject only if a *line* (peak above the local noise floor) already existed before the strike
        # and the strike did not make it much louder; broadband noise is not a line
        keep &= ~((line_pre > 8.0) & ((amp_seg - amp_pre) < 10.0))
    keep &= ~(np.isfinite(sigma) & (sigma < 0.3))
    if keep.sum() == 0:
        return None
    freqs, mags, sigma = freqs[keep], mags[keep], sigma[keep]

    lpc = None
    sig_lpc = np.full(len(freqs), np.nan)
    f_lpc = np.full(len(freqs), np.nan)
    if use_lpc:
        try:
            # the least-squares predictor must not straddle the silence->strike jump: start at the envelope peak
            k0 = int(np.argmax(np.abs(seg[: int(0.03 * sr)]))) + int(0.002 * sr)
            # signal subspace: 2 dimensions per real damped sinusoid (+ margin), capped by the model order
            rank = int(min(C.LPC_ORDER, 2 * len(freqs) + 6))
            order = int(max(C.LPC_ORDER, 2 * len(freqs) + 12))
            lpc = lpc_modes(seg[k0:], sr, order=order, fmin=fmin, fmax=fmax, rank=rank)
            for i, fk in enumerate(freqs):
                if len(lpc.freqs):
                    d = np.abs(1200 * np.log2(lpc.freqs / fk))
                    j = int(np.argmin(d))
                    if d[j] < 50:
                        sig_lpc[i], f_lpc[i] = lpc.sigmas[j], lpc.freqs[j]
        except Exception:
            lpc = None

    strong = np.where(mags >= mags.max() + pitch_rel_db)[0]
    pidx = int(strong[np.argmin(freqs[strong])])
    conf = float(np.clip((snr - 10.0) / 25.0, 0.0, 1.0))
    return StrikeAnalysis(onset / sr, freqs, mags, sigma, sig_lpc, f_lpc, float(freqs[pidx]), pidx,
                          float(snr), conf, sf_, sdb, lpc)


# ----------------------------------------------------------------------------- synthesis
def modal_synthesis(freqs, sigmas, amps, sr: int = C.SR, dur: float = 1.5, click: float = 0.0,
                    seed: int | None = None) -> np.ndarray:
    """Sum of exponentially decaying sinusoids = impulse response of an all-pole system."""
    t = np.arange(int(dur * sr)) / sr
    rng = np.random.default_rng(seed)
    y = np.zeros_like(t)
    for f, s, a in zip(freqs, sigmas, amps):
        if not np.isfinite(f) or not np.isfinite(s) or f <= 0:
            continue
        y += a * np.exp(-max(s, 0.1) * t) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    if click > 0:
        n = int(0.002 * sr)
        y[:n] += click * rng.standard_normal(n) * np.hanning(n)
    fade = np.ones_like(y)
    m = int(0.02 * sr)
    fade[-m:] = np.linspace(1, 0, m)
    y *= fade
    return y / (np.max(np.abs(y)) + EPS) * 0.8


def resynthesize(an: StrikeAnalysis, sr: int = C.SR, dur: float = 1.5) -> np.ndarray:
    """Rebuild the strike from ~3 numbers per mode (f, sigma, amplitude)."""
    sig = np.where(np.isfinite(an.sigma), an.sigma, np.nanmedian(an.sigma) if np.isfinite(an.sigma).any() else 5.0)
    # a decaying sinusoid's windowed spectral peak scales like A / sigma, so A ~ |X| * sigma
    amps = 10 ** (an.mags_db / 20.0) * np.maximum(sig, 1.0)
    amps = amps / amps.max()
    return modal_synthesis(an.freqs, sig, amps, sr, dur, click=0.05, seed=0)

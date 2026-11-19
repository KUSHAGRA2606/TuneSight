"""Fit the water-loading physics for each bowl from a recorded session and save it for the live demo.

    python scripts/calibrate.py data/own/session_XXXX [--shared]

--shared : identical bowls share one calibration (every tap on any bowl teaches all bowls)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tunesight import config as C  # noqa: E402
from tunesight import physics as P  # noqa: E402
from tunesight import session as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--shared", action="store_true")
    ap.add_argument("--out", default=str(C.MODELS_DIR / "calibration.json"))
    a = ap.parse_args()
    taps = S.measure_taps(S.load_session(Path(a.session)))
    groups = S.by_bowl(taps)
    if not groups:
        sys.exit("no usable taps (check images/audio)")
    report, calibs = {}, {}
    all_fill = np.concatenate([[t.fill for t in g] for g in groups.values()])
    all_f = np.concatenate([[t.pitch for t in g] for g in groups.values()])
    if a.shared:
        groups = {"shared": sum(groups.values(), [])}
    for b, g in groups.items():
        fill = np.array([t.fill for t in g])
        f = np.array([t.pitch for t in g])
        fit = P.fit_french(fill, f, fit_n=len(g) >= 6, fit_r0=len(g) >= 6)
        cal = P.FrenchCalibrator(f0_prior=float(f.max()), f0_prior_cents=200)
        for x, y in zip(fill, f):
            cal.add(x, y)
        calibs[str(b)] = cal.to_dict()
        report[str(b)] = {"taps": len(g), **{k: round(v, 3) for k, v in fit.items()},
                          "online_f0": round(cal.f0, 2), "online_beta": round(cal.beta, 3)}
        print(f"bowl {b}: {len(g)} taps  f0={fit['f0']:.1f} Hz  beta={fit['beta']:.3f}  n={fit['n']:.2f}"
              f"  r0={fit['r0']:.3f}  rmse={fit['rmse_cents']:.1f} cents | online: n={cal.n:.2f} r0={cal.r0:.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"calibrators": calibs, "report": report, "shared": a.shared,
               "pooled_fill_range": [float(all_fill.min()), float(all_fill.max())],
               "pooled_pitch_range": [float(all_f.min()), float(all_f.max())]}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

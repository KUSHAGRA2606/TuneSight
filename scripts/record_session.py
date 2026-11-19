"""Record a calibration session of your own bowls (about 1 hour for 5 bowls x 6 levels x 3 strikes).

Setup: bowls in a row on a white background, ArUco marker (results/marker_4cm.png printed at 4 cm) beside them,
camera from the side at bowl height, coloured water (food colour / haldi).

Controls (camera window must be focused):
    1..9   arm the bowl you are about to strike (left-to-right number shown on screen)
    SPACE  capture: grabs a still image, then records 2.5 s - strike the armed bowl once during it
    q      quit
Optionally type a ruler measurement of the water height when prompted (press Enter to skip).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

from tunesight import audio as A  # noqa: E402
from tunesight import config as C  # noqa: E402
from tunesight import vision as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=time.strftime("session_%Y%m%d_%H%M"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--ask-ruler", action="store_true", help="prompt for a ruler water height after each capture")
    a = ap.parse_args()
    root = C.DATA_DIR / "own" / a.name
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    csv_path = root / "taps.csv"
    new = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["tap_id", "bowl", "image", "wav", "true_h_cm", "note"])
    tap_id = sum(1 for _ in open(csv_path)) - 1

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    armed = 0
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        obs = V.measure_frame(frame)
        vis = cv2.resize(frame, None, fx=obs.scale, fy=obs.scale) if obs.scale < 1 else frame.copy()
        for i, b in enumerate(obs.bowls):
            x, y, ww, hh = b.bbox
            col = (0, 200, 0) if i == armed else (200, 120, 0)
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), col, 2)
            if b.water_top_y is not None:
                cv2.line(vis, (x, b.water_top_y), (x + ww, b.water_top_y), (0, 0, 255), 2)
            cv2.putText(vis, f"{i + 1}: fill {b.fill_ratio:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        msg = f"armed bowl {armed + 1} | marker {'OK' if obs.ppc else 'NOT FOUND'} | taps {tap_id}"
        cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("record session", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if ord("1") <= k <= ord("9"):
            armed = k - ord("1")
        if k == ord(" "):
            stills = []
            for _ in range(5):
                ok, f = cap.read()
                if ok:
                    stills.append(f.astype(np.float32))
            still = np.median(np.stack(stills), axis=0).astype(np.uint8)
            print(f"  recording 2.5 s - strike bowl {armed + 1} now")
            rec = sd.rec(int(2.5 * C.SR), samplerate=C.SR, channels=1, dtype="float32")
            sd.wait()
            x = rec[:, 0].astype(np.float64)
            an = A.analyze_strike(x)
            img_name, wav_name = f"images/tap_{tap_id:04d}.png", f"audio/tap_{tap_id:04d}.wav"
            cv2.imwrite(str(root / img_name), still)
            A.save_audio(root / wav_name, x)
            th = ""
            if a.ask_ruler:
                th = input("  ruler water height in cm (Enter to skip): ").strip()
            w.writerow([tap_id, armed, img_name, wav_name, th, ""])
            fh.flush()
            print(f"  tap {tap_id}: pitch {an.pitch:.1f} Hz, SNR {an.snr_db:.0f} dB" if an else "  no strike detected!")
            tap_id += 1
    fh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Write a printable ArUco marker (print it so the black square is exactly 4 cm wide)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from tunesight import config as C  # noqa: E402
from tunesight.vision import marker_image  # noqa: E402

m = marker_image(472)          # 472 px at 300 dpi = 4.0 cm
page = cv2.copyMakeBorder(m, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
cv2.putText(page, "TuneSight marker: print so the black square is 4.0 cm", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = C.RESULTS_DIR / "marker_4cm.png"
cv2.imwrite(str(out), page)
print(out)

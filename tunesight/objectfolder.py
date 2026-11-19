"""ObjectFolder Real (Gao et al., CVPR 2023): 100 real objects, impact sounds recorded in an anechoic chamber.

Used to validate the audio (source-filter / LPC) model on real recordings and to test the scaling law
(pitch ~ 1/size) on families of the same object in several sizes.

Each audio archive is ~36 GB per 10 objects because it also holds camera frames of every strike;
the microphone files we need are only ~0.6 MB each, so we *stream* the archive and keep only them.
"""
from __future__ import annotations

import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://download.cs.stanford.edu/viscam/ObjectFolder_Real"

OBJECTS = {
    1: ("Soup_Spoon", "Ceramic"), 2: ("Bowl", "Ceramic"), 3: ("Salad_Plate", "Ceramic"), 4: ("Dinner_Plate", "Ceramic"),
    5: ("Hair_Comb", "Wood"), 6: ("Blue_Bowl", "Glass"), 7: ("Decorative_Plate", "Glass"), 8: ("Mixing_Bowl", "Ceramic"),
    9: ("Serving_Bowl", "Ceramic"), 10: ("Soup_Bowl", "Ceramic"), 11: ("Strainer_Spoon", "Wood"), 12: ("Soup_Ladle", "Wood"),
    13: ("Serving_Spoon", "Wood"), 14: ("Salad_Fork", "Wood"), 15: ("Mixing_Spoon", "Wood"), 16: ("Frying_Spatula", "Wood"),
    17: ("8Inch_Skillet", "Iron"), 18: ("10_dot_25Inch_Skillet", "Iron"), 19: ("10_dot_5Inch_Griddle", "Iron"),
    20: ("Dutch_Oven", "Iron"), 21: ("Dutch_Oven_Lid", "Iron"), 22: ("Rinsing_Cup", "Glass"), 23: ("Hand_Scoop", "Plastic"),
    24: ("Shovel_Toy_Red_Large", "Plastic"), 25: ("Shovel_Toy_Green_Small", "Plastic"), 26: ("Handle_Spoon", "Polycarbonate"),
    27: ("Round_Plate", "Wood"), 28: ("Square_Plate", "Wood"), 29: ("Cutting_Board_Large", "Wood"),
    30: ("Cutting_Board_Middle", "Wood"), 31: ("Cutting_Board_Small", "Wood"), 32: ("Wine_Glass", "Wood"),
    33: ("Drinking_Cup", "Wood"), 34: ("Beer_Mug", "Wood"), 35: ("Portion_Cup_Brown", "Polycarbonate"),
    36: ("Portion_Cup_White", "Polycarbonate"), 37: ("Cake_Pan", "Steel"), 38: ("Loaf_Pan", "Steel"),
    39: ("Wrench_Small", "Steel"), 40: ("Wrench_Middle", "Steel"), 41: ("Wrench_Large", "Steel"), 42: ("Pestle", "Iron"),
    43: ("Mortar", "Iron"), 44: ("Sculpture", "Iron"), 45: ("Ladle", "Iron"), 46: ("Spatula", "Iron"),
    47: ("Decorative_Cast", "Iron"), 48: ("Mixing_Bowl_Large", "Plastic"), 49: ("Mixing_Bowl_Middle", "Plastic"),
    50: ("Mixing_Bowl_Small", "Plastic"), 51: ("Fruit_Bowl", "Glass"), 52: ("Fork_Small", "Steel"), 53: ("Fork_Large", "Steel"),
    54: ("Spoon_Small", "Steel"), 55: ("Spoon_Large", "Steel"), 56: ("Knife_Large", "Plastic"), 57: ("Knife_Middle", "Plastic"),
    58: ("Knife_Small", "Plastic"), 59: ("Soap_Dish", "Glass"), 60: ("Beer_Glass", "Glass"), 61: ("Container_Large", "Ceramic"),
    62: ("Container_Middle", "Ceramic"), 63: ("Container_Small", "Ceramic"), 64: ("Mug", "Ceramic"), 65: ("Vase", "Ceramic"),
    66: ("Plate_Handle", "Iron"), 67: ("Plate", "Iron"), 68: ("Plate_Base", "Wood"), 69: ("Display_Stand", "Iron"),
    70: ("Drop_Funnel", "Polycarbonate"), 71: ("Container_Lid", "Polycarbonate"), 72: ("Food_Pan", "Polycarbonate"),
    73: ("Flowerpot_Large", "Ceramic"), 74: ("Flowerpot_Small", "Ceramic"), 75: ("Vase_Green", "Ceramic"),
    76: ("Vase_Blue", "Ceramic"), 77: ("Vase_Orange", "Ceramic"), 78: ("Swan_Large", "Ceramic"), 79: ("Swan_Small", "Ceramic"),
    80: ("Spoon_Holder", "Wood"), 81: ("Utensil_Container", "Wood"), 82: ("Can", "Glass"), 83: ("Potato_Masher", "Steel"),
    84: ("Skimmer", "Steel"), 85: ("Pasta_Server", "Steel"), 86: ("Slotted_Spoon", "Steel"), 87: ("Solid_Turner", "Steel"),
    88: ("Ladle", "Steel"), 89: ("Solid_Spoon", "Steel"), 90: ("Slotted_Turner", "Steel"), 91: ("Glass_Green", "Glass"),
    92: ("Glass_Red", "Glass"), 93: ("Vase", "Glass"), 94: ("Salad_Bowl", "Glass"), 95: ("Scoop", "Polycarbonate"),
    96: ("Box_Lid", "Polycarbonate"), 97: ("Stanford_Frisbee", "Plastic"), 98: ("Kettlebell", "Iron"),
    99: ("Trim_Removal_Tool", "Plastic"), 100: ("Trim_Removal_Tool_2", "Plastic"),
}

# same object in several sizes, same material -> scaling-law test
SIZE_FAMILIES = {
    "Ceramic containers": [61, 62, 63], "Plastic mixing bowls": [48, 49, 50], "Wood cutting boards": [29, 30, 31],
    "Steel wrenches": [39, 40, 41], "Plastic knives": [56, 57, 58], "Ceramic flowerpots": [73, 74],
    "Ceramic swans": [78, 79], "Iron skillets": [17, 18], "Steel forks": [52, 53], "Steel spoons": [54, 55],
}


def chunk_for(obj_id: int) -> str:
    a = (obj_id - 1) // 10 * 10
    return f"{a + 1}_{a + 10}"


class _Counter:
    """File-like HTTP reader that counts bytes, stops at max_bytes, and resumes dropped connections
    with an HTTP Range request (slow or flaky networks drop long downloads)."""

    def __init__(self, url, max_bytes=None, retries=8):
        self.url, self.n, self.max, self.t0, self.last, self.retries = url, 0, max_bytes, time.time(), 0, retries
        self.f = urllib.request.urlopen(url, timeout=60)

    def _reopen(self):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.n}-"})
        self.f = urllib.request.urlopen(req, timeout=60)

    def read(self, size=-1):
        if self.max is not None and self.n >= self.max:
            return b""
        want = size if size and size > 0 else 1 << 20
        for attempt in range(self.retries + 1):
            try:
                b = self.f.read(want)
                if b or attempt == self.retries:
                    break
            except Exception:
                if attempt == self.retries:
                    raise
            time.sleep(2 + 2 * attempt)
            print(f"  connection dropped at {self.n / 1e6:.0f} MB - resuming", file=sys.stderr, flush=True)
            self._reopen()
        self.n += len(b)
        if self.n - self.last > 100 << 20:
            self.last = self.n
            rate = self.n / max(time.time() - self.t0, 1e-3) / 1e6
            print(f"  streamed {self.n / 1e9:.2f} GB at {rate:.1f} MB/s", file=sys.stderr, flush=True)
        return b

    def close(self):
        self.f.close()


def stream_audio(chunk: str, out_root: Path, max_bytes: int | None = None, keep=("mic.wav", "metadata.yaml")) -> int:
    """Stream one audio archive and extract only the microphone files. Returns number of files written."""
    url = f"{BASE}/audio/audio_data_{chunk}.tar.gz"
    out_root.mkdir(parents=True, exist_ok=True)
    n = 0
    src = _Counter(url, max_bytes)
    try:
        with tarfile.open(fileobj=src, mode="r|gz") as tar:
            for m in tar:
                if m.isfile() and Path(m.name).name in keep and "/audio/" in m.name:
                    f = tar.extractfile(m)
                    if f is None:
                        continue
                    dst = out_root / m.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(f.read())
                    n += 1
    except (tarfile.ReadError, EOFError, OSError) as e:
        if max_bytes is None:
            raise
        print(f"  stopped after {src.n / 1e9:.2f} GB ({type(e).__name__}); partial extraction kept", file=sys.stderr)
    finally:
        src.close()
    return n


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)


def iter_strikes(root: Path):
    """Yield (obj_id, strike_id, wav_path) for every extracted microphone recording."""
    for wav in sorted(root.glob("*/audio/*/mic.wav")):
        try:
            obj = int(wav.parts[-4])
            strike = int(wav.parts[-2])
        except ValueError:
            continue
        yield obj, strike, wav


def obj_mesh_extents(obj_path: Path) -> np.ndarray:
    """Bounding-box extents (sorted, largest first) of an .obj mesh, read without extra libraries."""
    vs = []
    with open(obj_path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                vs.append((float(p[1]), float(p[2]), float(p[3])))
    v = np.asarray(vs)
    return np.sort(v.max(axis=0) - v.min(axis=0))[::-1]


def find_mesh(mesh_root: Path, obj_id: int) -> Path | None:
    cands = [p for p in mesh_root.rglob("*.obj") if re.search(rf"(^|[\\/_]){obj_id}([\\/_.]|$)", str(p.relative_to(mesh_root)))]
    return cands[0] if cands else None

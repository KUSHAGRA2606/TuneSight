"""Download the parts of ObjectFolder Real that this project uses.

Audio archives are streamed and only the microphone files are kept (~0.6 MB per strike instead of ~90 MB).
Run on a fast connection (campus LAN): each audio chunk streams ~36 GB.

Examples
    python scripts/download_objectfolder.py --chunks 61_70 41_50          # size families for the scaling law
    python scripts/download_objectfolder.py --chunks 91_100 --max-gb 2     # quick partial sample (first objects only)
    python scripts/download_objectfolder.py --meshes                       # 0.4 GB, true object sizes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunesight import config as C  # noqa: E402
from tunesight import objectfolder as OF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="*", default=[], help="audio chunks such as 61_70 41_50 91_100")
    ap.add_argument("--max-gb", type=float, default=None, help="stop each chunk after this many GB (partial sample)")
    ap.add_argument("--meshes", action="store_true", help="download 16k-face meshes (0.4 GB)")
    ap.add_argument("--videos", action="store_true", help="download lightbox videos (1.15 GB)")
    ap.add_argument("--out", default=str(C.DATA_DIR / "objectfolder"))
    a = ap.parse_args()
    out = Path(a.out)
    for ch in a.chunks:
        print(f"[audio] streaming chunk {ch} ...", flush=True)
        n = OF.stream_audio(ch, out / "audio", None if a.max_gb is None else int(a.max_gb * 1e9))
        print(f"[audio] chunk {ch}: kept {n} files", flush=True)
    if a.meshes:
        dst = out / "mesh_16k.tar.gz"
        print("[mesh] downloading ...", flush=True)
        OF.download_file(f"{OF.BASE}/mesh/mesh_16k.tar.gz", dst)
        import tarfile
        with tarfile.open(dst) as t:
            t.extractall(out / "mesh")
    if a.videos:
        dst = out / "videos_1_100.tar.gz"
        print("[video] downloading ...", flush=True)
        OF.download_file(f"{OF.BASE}/vision/videos_1_100.tar.gz", dst)
        import tarfile
        with tarfile.open(dst) as t:
            t.extractall(out / "videos")
    print("done")


if __name__ == "__main__":
    main()

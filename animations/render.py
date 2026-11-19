"""Render every TuneSight animation with ManimGL and copy the videos to web/media/.

Usage (from the repository root, with the manim environment):
    python animations/render.py            # 720p, all scenes
    python animations/render.py --low      # 480p preview
    python animations/render.py WaterLoading

ManimGL needs ffmpeg on PATH. If it is missing, the ffmpeg binary that ships with the
`imageio-ffmpeg` package is used automatically.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["Pipeline", "SourceFilter", "WaterLoading", "WhichBowl", "TuneBySight"]


def ensure_ffmpeg() -> dict:
    env = os.environ.copy()
    if shutil.which("ffmpeg"):
        return env
    import imageio_ffmpeg

    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    bindir = ROOT / "animations" / ".ffmpeg"
    bindir.mkdir(exist_ok=True)
    target = bindir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists():
        shutil.copy(exe, target)
    env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
    return env


def make_poster(mp4: Path, env: dict) -> None:
    """Poster frame for the web page: a still from near the end, when the scene is complete."""
    jpg = mp4.with_suffix(".jpg")
    ffmpeg = shutil.which("ffmpeg", path=env["PATH"]) or "ffmpeg"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-sseof", "-1.5", "-i", str(mp4), "-frames:v", "1", "-q:v", "3", str(jpg)],
                   env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=SCENES)
    ap.add_argument("--low", action="store_true")
    ap.add_argument("--posters-only", action="store_true", help="only (re)make the poster images")
    a = ap.parse_args()
    env = ensure_ffmpeg()
    out_dir = ROOT / "animations" / "videos"
    media = ROOT / "web" / "media"
    media.mkdir(parents=True, exist_ok=True)
    if a.posters_only:
        for sc in a.scenes:
            make_poster(media / f"{sc}.mp4", env)
            print(f"   -> web/media/{sc}.jpg")
        return
    manimgl = Path(sys.executable).parent / ("manimgl.exe" if os.name == "nt" else "manimgl")
    for sc in a.scenes:
        cmd = [str(manimgl), str(ROOT / "animations" / "scenes.py"), sc, "-w", "-l" if a.low else "-m",
               "--video_dir", str(out_dir)]
        print(">>", sc, flush=True)
        r = subprocess.run(cmd, env=env, cwd=str(ROOT / "animations"), capture_output=True, text=True)
        mp4 = out_dir / f"{sc}.mp4"
        if r.returncode != 0 or not mp4.exists():
            print(r.stdout[-2000:], r.stderr[-3000:])
            sys.exit(f"render failed: {sc}")
        shutil.copy(mp4, media / f"{sc}.mp4")
        make_poster(media / f"{sc}.mp4", env)
        print(f"   -> web/media/{sc}.mp4 ({mp4.stat().st_size / 1e6:.1f} MB) + poster", flush=True)


if __name__ == "__main__":
    main()

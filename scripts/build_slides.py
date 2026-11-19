"""Build the IITG Beamer deck: figures and numbers from the experiments, then pdflatex + bibtex.

    python scripts/build_slides.py            # -> slides/TuneSight_slides.pdf
    python scripts/build_slides.py --no-figs  # reuse existing figures
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides"


def run(cmd):
    r = subprocess.run(cmd, cwd=SLIDES, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    if not a.no_figs:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py")], check=True)
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        sys.exit("pdflatex not found: install TeX Live / TinyTeX (packages: beamer, pgf, booktabs, lipsum, translator)")
    for step in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                 ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        code, out = run(step)
        if code != 0 and step[0] == "pdflatex":
            errs = [ln for ln in out.splitlines() if ln.startswith("!")]
            sys.exit("LaTeX error:\n" + "\n".join(errs[:10]))
    shutil.copy(SLIDES / "main.pdf", SLIDES / "TuneSight_slides.pdf")
    m = re.search(r"Output written on main\.pdf \((\d+) pages?", out)   # page count from the last pdflatex run
    print(f"slides/TuneSight_slides.pdf written ({m.group(1)} pages)" if m else "slides/TuneSight_slides.pdf written")


if __name__ == "__main__":
    main()

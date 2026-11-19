"""Build the interactive explainer page: fill the result numbers into web/src/index.template.html.

    python scripts/build_web.py          # -> web/index.html (open it in a browser)

Numbers come from web/assets/figures/numbers.json, written by scripts/make_figures.py from
results/metrics.json, so the page always shows the latest experiment results.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    tpl = (ROOT / "web" / "src" / "index.template.html").read_text(encoding="utf-8")
    nums_path = ROOT / "web" / "assets" / "figures" / "numbers.json"
    if not nums_path.exists():
        sys.exit("numbers.json missing: run scripts/make_figures.py first")
    nums = json.loads(nums_path.read_text(encoding="utf-8"))
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(nums))
    if missing:
        sys.exit(f"placeholders without a value: {missing}")
    html = re.sub(r"\{\{(\w+)\}\}", lambda m: str(nums[m.group(1)]), tpl)
    out = ROOT / "web" / "index.html"
    out.write_text(html, encoding="utf-8")
    media = ROOT / "web" / "media"
    vids = sorted(p.name for p in media.glob("*.mp4")) if media.exists() else []
    print(f"web/index.html written ({len(html) / 1024:.0f} KB); videos: {vids}")


if __name__ == "__main__":
    main()

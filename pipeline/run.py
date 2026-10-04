"""Run the whole pipeline: PDFs in data/pdfs -> docs/index.html

    python pipeline/run.py            # every floor in config.py
    python pipeline/run.py asb aq     # just these
"""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "build")
sys.path.insert(0, HERE)
from config import FLOORS

ids = sys.argv[1:] or list(FLOORS)
os.makedirs(WORK, exist_ok=True)
env = dict(os.environ, PYTHONPATH=HERE)

def step(script, *args):
    print(f"  {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=WORK, env=env, check=True)

for n in ids:
    if not os.path.exists(os.path.join(ROOT, "data", "pdfs", n + ".pdf")):
        print(f"skip {n}: data/pdfs/{n}.pdf not found"); continue
    print(f"== {n}")
    step("grid.py", n)    # walls + door swings from the CAD vectors
    step("walk.py", n)    # walkable area, rooms split from hallways
    step("doors.py", n)   # which two spaces each door connects
    step("ocr.py", n, "10", "0")  # room numbers (slow, ~1-2 min per floor)
    step("build.py", n)   # 0.2 m cost grid + door dots + background image

floors = [json.load(open(os.path.join(WORK, f"{n}_floor.json"))) for n in FLOORS
          if os.path.exists(os.path.join(WORK, f"{n}_floor.json"))]
tpl = open(os.path.join(ROOT, "web", "template.html")).read()
html = tpl.replace("__FLOORS__", json.dumps(floors, separators=(",", ":")))
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
open(os.path.join(ROOT, "docs", "index.html"), "w").write(html)
print(f"wrote docs/index.html with {len(floors)} floor(s)")

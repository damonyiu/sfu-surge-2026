"""Run the whole pipeline: PDFs in data/pdfs -> docs/index.html

    python pipeline/run.py            # every floor in config.py
    python pipeline/run.py aq3 aq2    # re-extract just these floors (build step still uses all)
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
    step("walk.py", n)    # walkable area, rooms split from hallways, manual fixes applied
    step("doors.py", n)   # which two spaces each door connects
    step("labels.py", n)  # room numbers from the CAD room-number layer, by character shape

step("cores.py")          # elevator shafts on every floor
step("build.py")          # shared crop, routing grids, doors, elevators -> build/floors.json
tpl = open(os.path.join(ROOT, "web", "template.html")).read()
html = tpl.replace("__FLOORS__", open(os.path.join(WORK, "floors.json")).read())
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
open(os.path.join(ROOT, "docs", "index.html"), "w").write(html)
print("wrote docs/index.html")

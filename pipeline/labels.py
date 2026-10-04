"""Room numbers from the CAD room-number layer (RM$TXT), read by shape matching.

The plans draw text as strokes in a CAD stroke font, so every "3" is drawn identically.
glyph_templates.json holds one bitmap per character (identified by hand from all six AQ
sheets). Each label is split into characters and every character is matched against the
templates; all four orientations are tried and the one where every character matches
best wins. Output: <id>_labels.json [{t, x, y, comp}] in 0.1 m raster cells.
"""
import os, sys, json, re, base64
import numpy as np, cv2
from config import FLOORS
from glyphs import strokes_of, clusters_of, glyphs_of, score

n = sys.argv[1]; FL = FLOORS[n]["floor"]
S = json.load(open(n + "_meta.json"))["px_per_pt"]
T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "glyph_templates.json")))
for t in T: t["img"] = cv2.imdecode(np.frombuffer(base64.b64decode(t["png"]), np.uint8), 0)

def match(img):
    best, bs = None, 0.0
    for t in T:
        s = score(img, t["img"])
        if s > bs: best, bs = t["c"], s
    return best, bs

LABEL = re.compile(r"^(\d{3,4})(\.\d{1,2}|[A-Z])?$")
def clean(t):
    t = t.replace("II", "H")                      # H is drawn as two bars and a crossbar
    if re.fullmatch(r"\d{5}", t):                 # 4 digits + a letter that looks like a digit (3004B)
        t = t[:4] + {"8": "B", "0": "D", "5": "S", "6": "G"}.get(t[4], t[4])
    t = t[:4].replace("I", "1") + t[4:]
    m = LABEL.match(t)
    return t if m and m.group(1).startswith(FL) and m.group(1) != FL + "000" else None   # "3000" is the level title, not a room

labels = []; other = 0
for cl in clusters_of(strokes_of(n)):
    x0 = min(s["box"][0] for s in cl); y0 = min(s["box"][1] for s in cl)
    x1 = max(s["box"][2] for s in cl); y1 = max(s["box"][3] for s in cl)
    if max(x1 - x0, y1 - y0) < 1.5 or max(x1 - x0, y1 - y0) > 40: continue
    cands = []
    for k in (0, 1, 3, 2):
        gl, H = glyphs_of(cl, k)
        if not gl: continue
        res = [match(g[0]) for g in gl]
        q = min(r[1] for r in res)
        cands.append((q, k, "".join(r[0] or "?" for r in res)))
    if not cands: continue
    # prefer an orientation that gives a valid room number (6, 9 and 0 also read upside down)
    good = [c for c in cands if c[0] >= 0.85 and clean(c[2])]
    q, k, text = max(good or cands, key=lambda c: (round(c[0], 2), -[0, 1, 3, 2].index(c[1])))
    t = clean(text) if q >= 0.85 else None
    if t: labels.append({"t": t, "x": (x0 + x1) / 2 * S, "y": (y0 + y1) / 2 * S, "comp": 0, "conf": round(q, 3)})
    else:
        other += 1
        if os.environ.get("SHOW_OTHER"): print("  not a room number:", text, round(q, 2))
print(n, "room numbers", len(labels), "other text", other)
json.dump(labels, open(n + "_labels.json", "w"))

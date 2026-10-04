"""Add labels from an older whole-sheet OCR pass (x, y in 0.1 m cells) to <id>_labels.json
for regions the per-room pass missed.   usage: merge_labels.py aq3 old_labels.json"""
import json, sys, re, numpy as np
from config import FLOORS
n, src = sys.argv[1], sys.argv[2]; FL = FLOORS[n]["floor"]
old = json.load(open(src)); new = json.load(open(n + "_labels.json"))
lab = np.load(n + "_walk.npz")["lab"]; have = {l["comp"] for l in new}; add = 0
def fix(t):
    t = t.replace("V", "7").replace("R", "2").replace("X", "1"); h = re.match(r"^[A-Z0-9]*", t).group(0); tail = t[len(h):]
    h = "".join({"S": "5", "B": "8", "G": "6"}.get(c, c) for c in h)
    if len(h) == 4 and h[0] != FL: h = FL + h[1:]
    t = h + tail
    return t if re.fullmatch(FL + r"\d{2,3}(\.\d{1,2}|[A-Z])?", t) else None
for l in old:
    t = fix(l["t"])
    if not t: continue
    y, x = int(l["y"]), int(l["x"]); win = lab[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5]; v = win[win > 0]
    if not v.size: continue
    c = int(np.bincount(v).argmax())
    if c in have: continue
    new.append({"t": t, "x": l["x"], "y": l["y"], "comp": c, "conf": l["conf"]}); have.add(c); add += 1
json.dump(new, open(n + "_labels.json", "w")); print(n, "merged", add, "total", len(new))

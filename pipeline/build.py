"""Pack every floor into one JSON the web page can route on.

All AQ sheets share the same page coordinates, so every floor is cropped to the
same window. That keeps elevator shafts at the same (x, y) on every level.

Per floor the page gets:
  grid  gzip+base64 bytes, one per 0.2 m cell:  type*64 + cost
        type 0 blocked, 1 hallway, 2 room, 3 doorway
        cost 1..63, higher near walls so paths keep to the middle of hallways
  dots  doors (x, y in cells, room label if known)
  lifts elevator shafts snapped to the nearest hallway cell
  bg    plan image, 1 px = 0.1 m
"""
import os, sys, json, gzip, base64, math, re
import numpy as np, cv2
import pymupdf as fitz
from scipy import ndimage as ndi
from config import PDF, FLOORS, CLIP

F = 2                      # 0.1 m raster -> 0.2 m routing grid
ids = [n for n in FLOORS if os.path.exists(n + "_walk.npz")]

# ---- shared crop window ----
crops = np.array([np.load(n + "_walk.npz")["crop"] for n in ids])
Y0, X0 = crops[:, 0].min(), crops[:, 2].min()
Y1, X1 = crops[:, 1].max(), crops[:, 3].max()
Y1 = Y0 + (Y1 - Y0) // F * F; X1 = X0 + (X1 - X0) // F * F
H, W = (Y1 - Y0) // F, (X1 - X0) // F

# ---- elevators: X-boxes that line up on 2+ floors ----
cores = json.load(open("cores.json"))
meta0 = json.load(open(ids[0] + "_meta.json")); S = meta0["px_per_pt"]
pts = [(n, b["x"], b["y"]) for n in ids for b in cores.get(n, {}).get("xbox", [])]
shafts = []                # [x_pt, y_pt, {floor ids}]
for n, x, y in pts:
    for sh in shafts:
        if math.hypot(sh[0] - x, sh[1] - y) < 3: sh[2].add(n); break
    else: shafts.append([x, y, {n}])
shafts = [s for s in shafts if len(s[2]) >= 2]
shafts.sort(key=lambda s: (round(s[1] / 200), s[0]))

def pool(a):
    return a.reshape(H, F, W, F)

GROUPS = {   # CAD layer -> drawing group (stroke weight in plan units, before zoom)
    "AWA": "wall", "AWAFU": "wall", "AWACO": "wall", "AWAMO": "wall", "AFLOT": "wall",
    "AGL": "glass", "ADO": "door", "AFLST": "stair",
    "AFLSP": "detail", "AFL": "detail", "AFLWD": "detail", "AFLTE": "detail", "SIWA": "detail",
    "RM$TXT": "text", "SGRID": "grid",
}
def vector_plan(n, S, X0, Y0, X1, Y1):
    from config import MAT, layer
    p = fitz.open(PDF(n))[0]; m = MAT(p, n)
    box = fitz.Rect(X0 / S, Y0 / S, X1 / S, Y1 / S)
    out = {}
    f2 = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
    for d in p.get_drawings():
        g = GROUPS.get(layer(d))
        if not g: continue
        r = d["rect"] * m      # (straight lines have zero-height boxes, so compare edges, not Rect.intersects)
        if r.x1 < box.x0 or r.x0 > box.x1 or r.y1 < box.y0 or r.y0 > box.y1: continue
        fill = "f" in d["type"] and not (d.get("color") and "s" in d["type"] and False)
        key = g + ("_f" if "f" in d["type"] else "")
        parts = out.setdefault(key, []); last = None
        P = lambda q: (f2(q.x * S - X0), f2(q.y * S - Y0))
        for it in d["items"]:
            if it[0] == "l":
                a_, b_ = P(it[1] * m), P(it[2] * m)
                parts.append(("L" if a_ == last else f"M{a_[0]} {a_[1]}L") + f"{b_[0]} {b_[1]}"); last = b_
            elif it[0] == "c":
                q = [P(it[k] * m) for k in (1, 2, 3, 4)]
                parts.append(("" if q[0] == last else f"M{q[0][0]} {q[0][1]}") + "C" + " ".join(f"{x} {y}" for x, y in q[1:])); last = q[3]
            elif it[0] in ("re", "qu"):
                qd = (it[1].quad if it[0] == "re" else it[1])
                q = [P(v * m) for v in (qd.ul, qd.ur, qd.lr, qd.ll)]
                parts.append(f"M{q[0][0]} {q[0][1]}" + "".join(f"L{x} {y}" for x, y in q[1:]) + "Z"); last = None
        if d.get("closePath") and parts: parts.append("Z"); last = None
    blob = json.dumps({k: "".join(v) for k, v in out.items()}, separators=(",", ":")).encode()
    return base64.b64encode(gzip.compress(blob, 9)).decode()

floors_out = []
for n in ids:
    cfg = FLOORS[n]
    d = np.load(n + "_walk.npz"); free, lab, wall = d["free"], d["lab"], d["wall"]
    G = json.load(open(n + "_graph.json")); doors = G["doors"]
    deg = {int(k): v for k, v in G["deg"].items()}
    labels = json.load(open(n + "_labels.json")) if os.path.exists(n + "_labels.json") else []

    # regions with no door at all are not walkable (open-to-below courtyard, shafts, closets)
    # room labels: one per region
    # map each label to the region under it now (walk.py may have been re-run since OCR)
    rlabel = {}; rpos = {}
    for L in labels:
        yy_, xx_ = int(L["y"]), int(L["x"])
        win = lab[max(yy_ - 5, 0):yy_ + 6, max(xx_ - 5, 0):xx_ + 6]; v = win[win > 0]
        if v.size:
            c_ = int(np.bincount(v).argmax())
            if c_ not in rlabel: rlabel[c_] = L["t"]; rpos[c_] = (L["x"], L["y"])
    # hallway = region labelled with a 3-digit corridor number (AQ corridors are 300, 301, ...),
    # a region with 5+ doors, or an unlabelled region with 3+ doors. Anything with a 4-digit room number is a room.
    objs_ = ndi.find_objects(lab)
    def elongated(c):
        sl = objs_[c - 1]
        if sl is None: return False
        h_, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        fill = (lab[sl] == c).sum() / max(h_ * w_, 1)
        return max(h_, w_) / max(min(h_, w_), 1) > 3 or fill < 0.45
    def is_hall(c):
        # 5+ doors and long/thin or ragged: a corridor even if a room number sits in it.
        # A compact space with many doors and a room number is a lecture hall or classroom.
        if deg.get(c, 0) >= 5 and (not rlabel.get(c) or elongated(c)): return True
        t = rlabel.get(c)
        if t: return len(re.match(r"\d+", t).group(0)) == 3
        return deg.get(c, 0) >= 3
    has_door = set()
    for o in doors:
        for c in (o["a"], o["b"]):
            if c: has_door.add(c)

    if not cfg.get("courtyard"):
        # Above level 3000 the courtyard is open to the sky, but tiny gaps in the drawing can join it
        # to the corridor around it. Block any open space wider than 8 m inside such a huge area:
        # corridors are never that wide, so they stay walkable.
        free = free.copy()
        area_ = np.bincount(lab.ravel()) * 0.01
        for c in [int(c) for c in np.nonzero(area_ >= 3000)[0] if c]:
            reg = (lab == c).astype(np.uint8)
            core = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
            free &= ~(cv2.dilate(core, np.ones((5, 5), np.uint8)) > 0)
    fr = free[Y0:Y1, X0:X1]; lb = lab[Y0:Y1, X0:X1]
    walk = pool(fr).mean((1, 3)) >= 0.4
    # majority region per 0.2 m cell
    sub = pool(lb).transpose(0, 2, 1, 3).reshape(H, W, F * F)
    reg = np.where((sub > 0).any(-1), sub.max(-1), 0)
    typ = np.zeros((H, W), np.uint8)
    # big open areas (courtyard, lobbies, covered walkways) have no door arcs but are walkable
    area = np.bincount(lab.ravel(), minlength=lab.max() + 1) * 0.01
    big = set(int(c) for c in np.nonzero(area >= 200)[0] if c)

    keep = np.zeros(lab.max() + 1, bool); keep[list(has_door | big)] = True
    hall = np.zeros(lab.max() + 1, bool); hall[[c for c in has_door if is_hall(c)] + list(big - has_door)] = True
    typ[walk & (reg == 0)] = 3
    typ[walk & keep[reg] & hall[reg]] = 1
    typ[walk & keep[reg] & ~hall[reg]] = 2
    # doorway cells that ended up isolated from any kept region are dropped
    lab3, _ = ndi.label(typ > 0, structure=np.ones((3, 3)))
    good = np.unique(lab3[(typ == 1) | (typ == 2)])
    typ[~np.isin(lab3, good)] = 0

    dist = ndi.distance_transform_edt(typ > 0)
    cost = 1 + np.clip(2.5 - dist, 0, 2.5) * 0.8        # 1 in the middle, up to 3 right by a wall
    cost[typ == 2] *= 6                                   # rooms: only if there's no other way
    cost = np.clip(np.round(cost), 1, 63).astype(np.uint8)
    packed = np.where(typ > 0, typ * 64 + cost, 0).astype(np.uint8)


    dots = []
    for i, o in enumerate(doors):
        x = (o["x"] - X0) / F; y = (o["y"] - Y0) / F
        if not (0 <= x < W and 0 <= y < H): continue
        a, b = o["a"], o["b"]
        if not (a in has_door or b in has_door): continue
        # toilet stalls: a tiny dead-end space whose door opens off a room (the washroom), not a hallway
        if a and b and any(area[c] < 2.0 and deg.get(c, 0) <= 1 and not is_hall(o_) for c, o_ in ((a, b), (b, a))): continue
        # which room does this door belong to? A door between a passage-like room (a lobby or
        # a suite's inner corridor, several doors) and a small room belongs to the small room.
        # If both sides have the same number of doors, use the side the door swings into.
        cand = [c for c in (a, b) if c and not is_hall(c)]
        if len(cand) == 2:
            da, db = deg.get(cand[0], 0), deg.get(cand[1], 0)
            room = cand[0] if da < db else cand[1] if db < da else (o.get("swing_into") if o.get("swing_into") in cand else cand[0])
        else:
            room = cand[0] if cand else None
        kind = "exit" if bool(a) != bool(b) else ("hall" if room is None else "room")
        dots.append({"id": i, "x": round(x, 1), "y": round(y, 1),
                     "label": rlabel.get(room, "") if room else "", "kind": kind,
                     "lx": round((rpos[room][0] - X0) / F, 1) if room in rpos else None,
                     "ly": round((rpos[room][1] - Y0) / F, 1) if room in rpos else None})
    # room numbers that are on the plan but have no usable door (so search can say why)
    have_lbl = {re.sub(r" door \d+$", "", dt["label"]) for dt in dots if dt["label"]}
    nodoor = sorted({L["t"] for L in labels if len(re.match(r"\d+", L["t"]).group(0)) == 4} - have_lbl)
    seen = {}
    for dt in dots:
        if dt["label"]:
            seen[dt["label"]] = seen.get(dt["label"], 0) + 1
            if seen[dt["label"]] > 1: dt["label"] += f" door {seen[dt['label']]}"

    # elevators on this floor, snapped to the nearest hallway/doorway cell within 4 m
    # only snap elevators to the part of the floor where the doors are (not an unreachable void)
    cc, _ = ndi.label(typ > 0, structure=np.ones((3, 3)))
    dcount = np.zeros(cc.max() + 1, int)
    for dt in dots:
        yy_, xx_ = int(dt["y"]), int(dt["x"])
        if 0 <= yy_ < H and 0 <= xx_ < W: dcount[cc[yy_, xx_]] += 1
    dcount[0] = 0
    hallish = ((typ == 1) | (typ == 3)) & (dcount[cc] >= 3)
    lifts = []
    for k, (xp, yp, fl) in enumerate(shafts):
        if n not in fl: continue
        cx, cy = (xp * S - X0) / F, (yp * S - Y0) / F
        r = 20
        y0, y1 = max(int(cy) - r, 0), min(int(cy) + r, H)
        x0, x1 = max(int(cx) - r, 0), min(int(cx) + r, W)
        yy, xx = np.nonzero(hallish[y0:y1, x0:x1])
        if len(yy) == 0: continue
        j = np.argmin((yy + y0 - cy) ** 2 + (xx + x0 - cx) ** 2)
        lifts.append({"id": f"E{k + 1}", "x": round(cx, 1), "y": round(cy, 1),
                      "sx": int(xx[j] + x0), "sy": int(yy[j] + y0)})

    # the plan itself as vector paths (sharp at any zoom), in plan units: 1 unit = 0.1 m,
    # same origin as the routing grid. Grouped by what they are so the page can style them.
    vec = vector_plan(n, S, X0, Y0, X1, Y1)
    bgW, bgH = int(X1 - X0), int(Y1 - Y0)

    floors_out.append({
        "id": n, "name": cfg["name"], "level": cfg["level"], "res": 0.2, "w": int(W), "h": int(H),
        "grid": base64.b64encode(gzip.compress(packed.tobytes(), 9)).decode(),
        "vec": vec, "bgW": bgW, "bgH": bgH,
        "dots": dots, "nodoor": nodoor, "lifts": lifts, "origin": [int(X0), int(Y0)], "pxPerPt": S})
    print(n, "cells", W, H, "doors", len(dots), "labelled", sum(1 for x in dots if x["label"]),
          "lifts", len(lifts), "hall%", round((typ == 1).mean() * 100, 1))

json.dump(floors_out, open("floors.json", "w"), separators=(",", ":"))
print("shafts", len(shafts), "floors.json KB", os.path.getsize("floors.json") // 1024)

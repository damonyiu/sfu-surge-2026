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
    rlabel = {}
    for L in labels:
        yy_, xx_ = int(L["y"]), int(L["x"])
        win = lab[max(yy_ - 5, 0):yy_ + 6, max(xx_ - 5, 0):xx_ + 6]; v = win[win > 0]
        if v.size: rlabel.setdefault(int(np.bincount(v).argmax()), L["t"])
    # hallway = region labelled with a 3-digit corridor number (AQ corridors are 300, 301, ...),
    # a region with 5+ doors, or an unlabelled region with 3+ doors. Anything with a 4-digit room number is a room.
    def is_hall(c):
        if deg.get(c, 0) >= 5: return True          # 5+ doors: a corridor even if a room number bled in
        t = rlabel.get(c)
        if t: return len(re.match(r"\d+", t).group(0)) == 3
        return deg.get(c, 0) >= 3
    has_door = set()
    for o in doors:
        for c in (o["a"], o["b"]):
            if c: has_door.add(c)

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
    cost = 1 + np.clip(3.5 - dist, 0, 3.5) * 2          # 1 in the middle, up to 8 by a wall
    cost[typ == 2] *= 6                                   # rooms: only if there's no other way
    cost = np.clip(np.round(cost), 1, 63).astype(np.uint8)
    packed = np.where(typ > 0, typ * 64 + cost, 0).astype(np.uint8)


    dots = []
    for i, o in enumerate(doors):
        x = (o["x"] - X0) / F; y = (o["y"] - Y0) / F
        if not (0 <= x < W and 0 <= y < H): continue
        a, b = o["a"], o["b"]
        if not (a in has_door or b in has_door): continue
        room = next((c for c in (a, b) if c and not is_hall(c)), None)
        kind = "exit" if bool(a) != bool(b) else ("hall" if room is None else "room")
        dots.append({"id": i, "x": round(x, 1), "y": round(y, 1),
                     "label": rlabel.get(room, "") if room else "", "kind": kind})
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

    # background plan, 1 px = 0.1 m
    p = fitz.open(PDF(n))[0]
    pix = p.get_pixmap(matrix=fitz.Matrix(S, S), colorspace=fitz.csGRAY,
                       clip=CLIP(fitz.Rect(X0 / S, Y0 / S, X1 / S, Y1 / S), n))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
    img = np.where(img < 235, img, 255).astype(np.uint8)
    _, png = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    floors_out.append({
        "id": n, "name": cfg["name"], "level": cfg["level"], "res": 0.2, "w": int(W), "h": int(H),
        "grid": base64.b64encode(gzip.compress(packed.tobytes(), 9)).decode(),
        "bg": "data:image/png;base64," + base64.b64encode(png).decode(),
        "bgW": int(img.shape[1]), "bgH": int(img.shape[0]),
        "dots": dots, "lifts": lifts, "origin": [int(X0), int(Y0)], "pxPerPt": S})
    print(n, "cells", W, H, "doors", len(dots), "labelled", sum(1 for x in dots if x["label"]),
          "lifts", len(lifts), "hall%", round((typ == 1).mean() * 100, 1))

json.dump(floors_out, open("floors.json", "w"), separators=(",", ":"))
print("shafts", len(shafts), "floors.json KB", os.path.getsize("floors.json") // 1024)

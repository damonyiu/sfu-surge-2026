"""Shared helpers: pull room-number text out of the RM$TXT layer as labels made of glyphs.
Each glyph is drawn into a small normalised bitmap so identical characters look identical."""
import math, numpy as np, cv2
import pymupdf as fitz
from config import PDF, MAT, TEXT_LAYER, layer

GH = 32          # bitmap height for one text line

def strokes_of(n):
    p = fitz.open(PDF(n))[0]; m = MAT(p, n); out = []
    for d in p.get_drawings():
        if layer(d) != TEXT_LAYER: continue
        segs = []
        for it in d["items"]:
            if it[0] == "l": segs.append([tuple(it[1] * m), tuple(it[2] * m)])
            elif it[0] == "c": segs.append([tuple(it[k] * m) for k in (1, 2, 3, 4)])
            elif it[0] in ("qu", "re"):        # decimal points are drawn as tiny squares
                q = it[1].quad if it[0] == "re" else it[1]
                pts = [tuple(v * m) for v in (q.ul, q.ur, q.lr, q.ll, q.ul)]; segs.append(pts)
        if segs:
            xs = [q[0] for s in segs for q in s]; ys = [q[1] for s in segs for q in s]
            out.append({"segs": segs, "box": [min(xs), min(ys), max(xs), max(ys)]})
    return out

def clusters_of(strokes, GAP=3.2):
    par = list(range(len(strokes)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    ov = lambda a0, a1, b0, b1: min(a1, b1) - max(a0, b0)
    order = sorted(range(len(strokes)), key=lambda i: strokes[i]["box"][0])
    for a_ in range(len(order)):
        i = order[a_]; bi = strokes[i]["box"]
        for b_ in range(a_ + 1, len(order)):
            j = order[b_]; bj = strokes[j]["box"]
            if bj[0] > bi[2] + GAP + 12: break
            hx = ov(bi[0], bi[2], bj[0], bj[2]); hy = ov(bi[1], bi[3], bj[1], bj[3])
            row = hy > 0.5 * min(bi[3] - bi[1], bj[3] - bj[1]) and hx > -GAP
            # vertical text: characters lie on their side (wider than tall), stacked in a column
            side = lambda b: (b[2] - b[0]) > 1.1 * (b[3] - b[1]) or max(b[2] - b[0], b[3] - b[1]) < 0.6
            col = hx > 0.5 * min(bi[2] - bi[0], bj[2] - bj[0]) and hy > -GAP and side(bi) and side(bj)
            if row or col or (hx > 0 and hy > 0): par[f(i)] = f(j)
    cl = {}
    for i in range(len(strokes)): cl.setdefault(f(i), []).append(strokes[i])
    return list(cl.values())

def rot(pt, k, cx, cy):
    """rotate a point by k*90 degrees counter-clockwise (in screen coords) about (cx, cy)"""
    x, y = pt[0] - cx, pt[1] - cy
    for _ in range(k % 4): x, y = y, -x
    return (x + cx, y + cy)

def glyphs_of(cl, k):
    """split a label (cluster) rotated by k into glyphs left to right; returns list of
    (bitmap, rel_width, rel_x) plus the line height"""
    cx = sum((s["box"][0] + s["box"][2]) / 2 for s in cl) / len(cl)
    cy = sum((s["box"][1] + s["box"][3]) / 2 for s in cl) / len(cl)
    R = []
    for s in cl:
        segs = [[rot(q, k, cx, cy) for q in seg] for seg in s["segs"]]
        xs = [q[0] for sg in segs for q in sg]; ys = [q[1] for sg in segs for q in sg]
        R.append({"segs": segs, "box": [min(xs), min(ys), max(xs), max(ys)]})
    y0 = min(r["box"][1] for r in R); y1 = max(r["box"][3] for r in R); H = y1 - y0
    if H < 1.5 or H > 6: return [], H          # not a line of room-number text
    R.sort(key=lambda r: r["box"][0])
    # strokes that overlap horizontally (or nearly touch) form one character
    groups = []
    for r in R:
        if groups and r["box"][0] < groups[-1]["x1"] - 0.06 * H:
            g = groups[-1]; g["s"].append(r); g["x1"] = max(g["x1"], r["box"][2])
        else:
            groups.append({"s": [r], "x0": r["box"][0], "x1": r["box"][2]})
    out = []
    for g in groups:
        w = max(g["x1"] - g["x0"], 0.02 * H)
        Wpx = max(4, int(round(w / H * GH)) + 4)
        img = np.zeros((GH + 4, Wpx), np.uint8)
        for r in g["s"]:
            for sg in r["segs"]:
                P = np.array([[(q[0] - g["x0"]) / H * GH + 2, (q[1] - y0) / H * GH + 2] for q in sg], np.int32)
                cv2.polylines(img, [P], False, 255, 2)
        out.append((img, w / H, (g["x0"] - R[0]["box"][0]) / H))
    return out, H

def score(a, b):
    """similarity of two glyph bitmaps: overlap of slightly thickened strokes (1 = same)"""
    if abs(a.shape[1] - b.shape[1]) > max(3, 0.25 * max(a.shape[1], b.shape[1])): return 0.0
    W = max(a.shape[1], b.shape[1])
    A = cv2.copyMakeBorder(a, 0, 0, 0, W - a.shape[1], cv2.BORDER_CONSTANT, value=0)
    B = cv2.copyMakeBorder(b, 0, 0, 0, W - b.shape[1], cv2.BORDER_CONSTANT, value=0)
    k = np.ones((3, 3), np.uint8)
    Ad, Bd = cv2.dilate(A, k) > 0, cv2.dilate(B, k) > 0
    a_, b_ = A > 0, B > 0
    if a_.sum() == 0 or b_.sum() == 0: return 1.0 if a_.sum() == b_.sum() else 0.0
    return min((a_ & Bd).sum() / a_.sum(), (b_ & Ad).sum() / b_.sum())

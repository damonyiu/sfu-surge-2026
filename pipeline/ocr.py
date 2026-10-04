"""Read room numbers one region at a time.

For every enclosed region found by walk.py we crop that region out of the
original PDF at high zoom and run Tesseract on just that crop. This finds far
more labels than one OCR pass over the whole sheet, because each crop holds
one or two numbers and almost no line work.
Output: <id>_labels.json  [{t, x, y, comp}]  (x, y in 0.1 m cells)
"""
import os, sys, json, re, subprocess, tempfile
import numpy as np, cv2
import pymupdf as fitz
from scipy import ndimage as ndi
from config import PDF, FLOORS, CLIP

n = sys.argv[1]
FLOOR = FLOORS[n]["floor"]
p = fitz.open(PDF(n))[0]
s = json.load(open(n + "_meta.json"))["px_per_pt"]      # cells per pt
d = np.load(n + "_walk.npz"); lab = d["lab"]
Z = 12
tmp = os.path.join(tempfile.gettempdir(), f"ocr_{n}.png")
LABEL = re.compile(r"^(\d{3,4})(\.\d{1,2}|[A-Z])?$")
CONF = {"S": "5", "B": "8", "G": "6", "O": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "V": "7"}

def clean(t):
    t = t.strip().upper().strip(".,:;-_'\"")
    head = re.match(r"^[A-Z0-9]*", t).group(0)
    tail = t[len(head):]
    head = "".join(CONF.get(ch, ch) for ch in head[:4]) + head[4:]
    t = head + tail
    m = LABEL.match(t)
    if not m: return None
    num = m.group(1)
    if num[0] != FLOOR:            # every label on this sheet starts with the floor digit
        if len(num) == 4 and num[0] in "35680": num = FLOOR + num[1:]
        else: return None
    return num + (m.group(2) or "")

def ocr(img, psm):
    cv2.imwrite(tmp, img)
    out = subprocess.run(["tesseract", tmp, "-", "--psm", str(psm), "-c",
                          "tessedit_char_whitelist=0123456789.ABCDEFGHJKLMNPRSTUVWXYZ", "tsv"],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 12 and f[11].strip():
            rows.append((f[11], float(f[10]), int(f[6]), int(f[7]), int(f[8]), int(f[9])))
    return rows

labels = []
objs = ndi.find_objects(lab)
for i, sl in enumerate(objs):
    if sl is None: continue
    area = (lab[sl] == i + 1).sum() * 0.01          # m^2
    if area < 1.5 or area > 2500: continue
    pad = 4
    y0, y1 = max(sl[0].start - pad, 0), sl[0].stop + pad
    x0, x1 = max(sl[1].start - pad, 0), sl[1].stop + pad
    clip = fitz.Rect(x0 / s, y0 / s, x1 / s, y1 / s)
    pix = p.get_pixmap(matrix=fitz.Matrix(Z, Z), colorspace=fitz.csGRAY, clip=CLIP(clip, n))
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
    if a.size == 0 or a.min() > 128: continue
    a = np.where(a < 110, 0, 255).astype(np.uint8)
    # blank out everything outside this region so walls/neighbours don't confuse OCR
    reg = lab[y0:y1, x0:x1] == i + 1
    holes, nh = ndi.label(ndi.binary_fill_holes(reg) & ~reg)
    if nh:   # the label text punches small holes in the region; fill those, not whole rooms inside a ring corridor
        hs = ndi.sum(np.ones_like(holes), holes, range(1, nh + 1))
        small = np.zeros(nh + 1, bool); small[1:] = hs < 250
        reg = reg | small[holes]
    m = reg.astype(np.uint8)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    m = cv2.resize(m, (pix.w, pix.h), interpolation=cv2.INTER_NEAREST)
    a = np.where(m > 0, a, 255).astype(np.uint8)
    # keep only glyph-sized blobs: walls, door arcs and grid lines are much longer than a digit
    nb, cc, st, _ = cv2.connectedComponentsWithStats((a == 0).astype(np.uint8), 8)
    keep = np.zeros(nb, bool)
    for k in range(1, nb):
        w_, h_ = st[k, 2], st[k, 3]
        keep[k] = (12 <= h_ <= 75 and w_ <= 60) or (h_ <= 12 and w_ <= 12 and st[k, 4] > 6)  # digits/letters, or a decimal point
    a = np.where(keep[cc], 0, 255).astype(np.uint8)
    if (a == 0).sum() < 80: continue
    a = cv2.copyMakeBorder(a, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    best = None
    for psm in (11, 6):
        for t, conf, l, tp, w, h in ocr(a, psm):
            c = clean(t)
            if c and conf > 30 and (best is None or conf > best[1]):
                best = (c, conf, l - 30, tp - 30, w, h)
        if best: break
    if best:
        c, conf, l, tp, w, h = best
        labels.append({"t": c, "x": x0 + (l + w / 2) / Z * s, "y": y0 + (tp + h / 2) / Z * s,
                       "comp": i + 1, "conf": conf})

print(n, "labels", len(labels), sorted(l["t"] for l in labels)[:80])
json.dump(labels, open(n + "_labels.json", "w"))

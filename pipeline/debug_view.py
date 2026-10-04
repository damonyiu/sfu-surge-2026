"""Draw what the router sees for one floor: hallway / room / doorway cells, the
largest connected area highlighted, doors and elevators. Writes <id>_debug.png"""
import json, gzip, base64, sys, numpy as np, cv2
from scipy import ndimage as ndi
n = sys.argv[1]
F = next(f for f in json.load(open("floors.json")) if f["id"] == n)
g = np.frombuffer(gzip.decompress(base64.b64decode(F["grid"])), np.uint8).reshape(F["h"], F["w"])
t = g >> 6
lab, _ = ndi.label(t > 0, structure=np.ones((3, 3)))
big = np.bincount(lab.ravel()); big[0] = 0; main = big.argmax()
img = np.full(t.shape + (3,), 255, np.uint8)
img[t == 2] = (200, 225, 250); img[t == 1] = (120, 200, 120); img[t == 3] = (60, 160, 255)
# every connected area that holds 4+ doors gets its own colour, the rest is grey
cnt = np.zeros(lab.max() + 1, int)
for d in F["dots"]:
    cnt[lab[min(int(d["y"]), F["h"] - 1), min(int(d["x"]), F["w"] - 1)]] += 1
cnt[0] = 0
rng = np.random.default_rng(7); pal = rng.integers(40, 220, (lab.max() + 1, 3)).astype(np.uint8)
mask = t > 0
img[mask] = 200
sel = mask & (cnt[lab] >= 4)
img[sel] = pal[lab[sel]]
img[t == 3] = (img[t == 3] * 0.6).astype(np.uint8)
print(n, "areas with 4+ doors:", sorted(cnt[cnt >= 4].tolist(), reverse=True))
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
for d in F["dots"]:
    cv2.circle(img, (int(d["x"] * 2), int(d["y"] * 2)), 3, (0, 0, 200) if d["label"] else (0, 0, 0), -1)
for l in F["lifts"]:
    cv2.rectangle(img, (int(l["x"] * 2) - 6, int(l["y"] * 2) - 6), (int(l["x"] * 2) + 6, int(l["y"] * 2) + 6), (160, 60, 120), -1)
    cv2.line(img, (int(l["x"] * 2), int(l["y"] * 2)), (l["sx"] * 2, l["sy"] * 2), (160, 60, 120), 2)
cv2.imwrite(n + "_debug.png", img)
print(n, "main area share", round(big[main] / big.sum(), 2))

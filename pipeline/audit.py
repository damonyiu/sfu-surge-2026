"""Contact sheets for checking room numbers: one tile per labelled room showing the plan
around it, the number the pipeline assigned (top-left), the doors that carry that number
(red) and where the OCR read it (blue box). Writes audit_<id>_<n>.png"""
import json, base64, sys, re, numpy as np, cv2
floors = json.load(open("floors.json"))
TW, TH, COLS, ROWS = 300, 220, 6, 5
for F in floors:
    if len(sys.argv) > 1 and F["id"] not in sys.argv[1:]: continue
    bg = cv2.imdecode(np.frombuffer(base64.b64decode(F["bg"].split(",")[1]), np.uint8), 0)
    k = F["bgW"] / F["w"]
    rooms = {}
    for d in F["dots"]:
        if d["label"]:
            base = re.sub(r" door \d+$", "", d["label"]); rooms.setdefault(base, []).append(d)
    tiles = []
    for lbl in sorted(rooms):
        ds = rooms[lbl]
        xs = [d["x"] * k for d in ds]; ys = [d["y"] * k for d in ds]
        if ds[0]["lx"] is not None: xs.append(ds[0]["lx"] * k); ys.append(ds[0]["ly"] * k)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 40
        x0, y0 = int(max(cx - half * 1.36, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half * 1.36, bg.shape[1])), int(min(cy + half, bg.shape[0]))
        c = cv2.cvtColor(bg[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        for d in ds: cv2.circle(c, (int(d["x"] * k - x0), int(d["y"] * k - y0)), max(3, int(c.shape[0] / 45)), (0, 0, 230), -1)
        if ds[0]["lx"] is not None:
            lx, ly = int(ds[0]["lx"] * k - x0), int(ds[0]["ly"] * k - y0); r = max(6, int(c.shape[0] / 18))
            cv2.rectangle(c, (lx - 2 * r, ly - r), (lx + 2 * r, ly + r), (230, 120, 0), 1)
        c = cv2.resize(c, (TW, TH), interpolation=cv2.INTER_AREA)
        cv2.rectangle(c, (0, 0), (TW - 1, TH - 1), (150, 150, 150), 1)
        cv2.rectangle(c, (0, 0), (118, 24), (255, 255, 255), -1)
        cv2.putText(c, lbl, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        tiles.append(c)
    per = COLS * ROWS
    for s in range(0, len(tiles), per):
        T = tiles[s:s + per]; sheet = np.full((ROWS * TH, COLS * TW, 3), 255, np.uint8)
        for i, t in enumerate(T): sheet[(i // COLS) * TH:(i // COLS + 1) * TH, (i % COLS) * TW:(i % COLS + 1) * TW] = t
        cv2.imwrite(f"audit_{F['id']}_{s // per}.png", sheet)
    print(F["id"], "rooms", len(tiles), "sheets", (len(tiles) + per - 1) // per)

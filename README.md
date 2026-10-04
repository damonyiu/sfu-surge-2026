# AQ Wayfinder 🦝

Indoor navigation for SFU Burnaby's Academic Quadrangle, all six levels. It turns the Facilities Services floor-plan PDFs into a walkable map, routes door to door along hallways, and takes the elevator when the two rooms are on different floors. A raccoon walks the route.

Open `docs/index.html` in a browser (Mac: `open docs/index.html`, Windows: `start docs\index.html`). No install needed to view it.

## Using it

- Type two room numbers (e.g. `3045` and `2104`) or click a door dot for the start and another for the destination. Switch floors with the buttons on the left of the map.
- **Example: 3 → 2 floors** picks a random cross-floor trip.
- Grey doors are in a part of the floor the router can't reach yet (see Fix map).

## How it works

The SFU key plans are AutoCAD exports, so the PDFs contain real vector geometry at 1:400. All six AQ sheets share one coordinate system (the 2000 sheet is offset 23 pt; `config.py` corrects it), so elevator shafts line up between levels.

| step | file | what it does |
|---|---|---|
| 1 | `grid.py` | reads the vectors, finds every door from its swing arc (curves or polylines, single and double doors), removes door leaves, carves each doorway open, rasterizes walls at 10 cm |
| 2 | `walk.py` | finds the building footprint and walkable floor, applies manual fixes, closes each doorway along its door line so rooms and hallways become separate regions |
| 3 | `doors.py` | records which two regions each door connects |
| 4 | `ocr.py` | reads the room number in each region with Tesseract |
| 5 | `cores.py` | finds elevator shafts (boxes with an X) on every level |
| 6 | `build.py` | crops every floor to the same window, marks cells as hallway / room / doorway, packs a 0.2 m routing grid, doors and elevators into `build/floors.json` |
| 7 | `run.py` | runs everything and writes `docs/index.html` |

Routing runs in the browser: Dijkstra on the grid, hallways cheap, room interiors 6x more expensive (so routes stay in corridors), cells near walls cost more (so routes keep to the middle). For two floors it computes distances from the start to every elevator that serves both levels and from the destination back, and picks the elevator with the shortest total.

## Setup (only to re-run the pipeline)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Tesseract OCR: macOS `brew install tesseract`, Windows `winget install -e --id UB-Mannheim.TesseractOCR`
```

Put the six AQ key plans in `data/pdfs/` as `aq1.pdf` … `aq6.pdf` (from https://www.sfu.ca/fs/campus-maps/key-plans.html), then:

```bash
python pipeline/run.py
```

About 2 minutes per floor, mostly OCR. `python pipeline/debug_view.py aq3` (run inside `build/`) draws what the router sees.

## Fixing a floor

The plans sometimes draw a line across a real opening, which seals off part of a floor. In the page, click **Fix map**, click two points across the blocked opening, check that the grey doors turn blue, then copy the JSON into `data/fixes/<floor>.json` and re-run the pipeline. `data/fixes/aq3.json` already opens corridors 300 and 301 to the west hallway and 300 to the courtyard. These are guesses from the drawing, so check them on site.

## Known issues

- **AQ 3000 and 2000 are partly disconnected.** 1000, 4000, 5000 and 6000 route everywhere. 3000 reaches about half its doors and 2000 a bit more. Use the Fix tool.
- **Room numbers:** about a third of doors get a number. CAD text is drawn as strokes, so OCR misses or misreads some.
- **Hallway vs room** is a guess: 3-digit numbers (300, 301) and regions with 5+ doors are hallways.
- **Stairs** aren't used for routing yet, only elevators.
- **Not done:** other buildings, live position tracking, 3D view.

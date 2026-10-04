# SFU Hallway Router 🦝

Indoor navigation for SFU Burnaby. Turns the Facilities Services floor-plan PDFs into a walkable map, finds door-to-door routes with A*, and has a raccoon walk you there.

Open `docs/index.html` in a browser to try the current prototype (Applied Sciences 8000 level + AQ 3000 level).

## How it works

The SFU key plans are AutoCAD exports, so the PDFs contain real vector geometry at a known drawing scale (1:250, 1:400). The pipeline uses that directly:

1. **`grid.py`** reads the vectors, finds every door from its swing arc, removes the door leaves, and rasterizes the walls at 10 cm per cell.
2. **`walk.py`** finds the building footprint and the walkable floor, then closes each doorway along its exact door line so rooms and hallways become separate regions.
3. **`doors.py`** records which two regions each door connects. Regions with 3+ doors are treated as hallways.
4. **`ocr.py`** reads room numbers with Tesseract (unreliable, see below).
5. **`build.py`** packs a 0.2 m cost grid (walls blocked, edges of hallways and room interiors cost more), the door dots and a background image into `build/<id>_floor.json`.
6. **`run.py`** runs all of the above and injects every floor into `web/template.html`, writing `docs/index.html`.

Routing happens in the browser: 8-connected A* on the cost grid, then path smoothing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# also install Tesseract OCR: macOS `brew install tesseract`, Ubuntu `sudo apt install tesseract-ocr`
```

## Adding floors

1. Download key-plan PDFs from https://www.sfu.ca/fs/campus-maps/key-plans.html
2. Save each one as `data/pdfs/<id>.pdf` (e.g. `data/pdfs/aq.pdf`)
3. Add an entry to `pipeline/config.py` with the drawing scale and floor digit from the title block
4. Run `python pipeline/run.py` (or `python pipeline/run.py aq` for one floor)

PDFs are git-ignored, so each teammate downloads their own copy.

## Known issues

- **Room numbers:** CAD text is drawn as strokes, so OCR misreads digits (3 vs 5, 8 vs 3). About half the doors get a usable label. Plan: a vision model or a manual label editor.
- **AQ connectivity:** some hallway openings on the AQ 3000 sheet aren't drawn as doors, so parts of corridors 300/301 come out sealed and many routes fail. Needs a manual fix-up step.
- **Hallway vs room** is guessed from door count.
- **Not done yet:** stairs/elevators between floors, links between buildings, live position tracking, 3D view.

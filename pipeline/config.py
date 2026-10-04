# One entry per floor plan. Drop the PDF in data/pdfs/<id>.pdf
# scale = the drawing scale printed in the title block (1:400 -> 400)
# floor = first digit of room numbers on that level (used to clean up OCR)
# level = vertical order, used for elevator/stair routing
# shift = [dx, dy] in PDF points to line this sheet up with the others (the 2000 sheet is drawn 23 pt lower)
FLOORS = {
    "aq1": {"name": "AQ 1000 level", "scale": 400, "floor": "1", "level": 1},
    "aq2": {"name": "AQ 2000 level", "scale": 400, "floor": "2", "level": 2, "shift": [0, -23]},
    "aq3": {"name": "AQ 3000 level", "scale": 400, "floor": "3", "level": 3},
    "aq4": {"name": "AQ 4000 level", "scale": 400, "floor": "4", "level": 4},
    "aq5": {"name": "AQ 5000 level", "scale": 400, "floor": "5", "level": 5},
    "aq6": {"name": "AQ 6000 level", "scale": 400, "floor": "6", "level": 6},
}

import os
def PDF(n):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pdfs", n + ".pdf")

import pymupdf as _fitz
def OFF(n):
    return FLOORS[n].get("shift", [0, 0])
def MAT(page, n):
    """rotation to the viewing orientation, then the per-sheet alignment shift"""
    dx, dy = OFF(n)
    return page.rotation_matrix * _fitz.Matrix(1, 0, 0, 1, dx, dy)
def CLIP(rect, n):
    """a rect in aligned coordinates -> the same area on this sheet"""
    dx, dy = OFF(n)
    return _fitz.Rect(rect.x0 - dx, rect.y0 - dy, rect.x1 - dx, rect.y1 - dy)

# CAD layers (the part after "|", the prefix differs per floor). Walls are built only from these,
# so furniture, the gross-area outline, grid lines and room-number text don't block anything.
WALL_LAYERS = {"AWA", "AWAFU", "AGL", "AWACO", "AWAMO", "ADO", "AFLST"}
TEXT_LAYER = "RM$TXT"
def layer(d):
    return (d.get("layer") or "").split("|")[-1]

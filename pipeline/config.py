# One entry per floor plan. Drop the PDF in data/pdfs/<id>.pdf
# scale = the drawing scale printed in the title block (1:250 -> 250)
# floor = first digit of room numbers on that level (used to clean up OCR)
FLOORS = {
    "asb": {"name": "Applied Sciences · 8000 level", "scale": 250, "floor": "8"},
    "aq":  {"name": "Academic Quadrangle · 3000 level", "scale": 400, "floor": "3"},
}

import os
def PDF(n):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pdfs", n + ".pdf")

"""Convert conv.txt to conversation.pdf using fpdf2."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os, re, textwrap

HERE     = os.path.dirname(__file__)
SRC      = os.path.join(HERE, "conv.txt")
OUT      = os.path.join(HERE, "conversation.pdf")
MARGIN   = 14
PAGE_W   = 210
BODY_W   = PAGE_W - 2 * MARGIN

REPLACEMENTS = [
    ("\u2014", "-"),   # em dash
    ("\u2013", "-"),   # en dash
    ("\u2019", "'"),   # right single quote
    ("\u2018", "'"),   # left single quote
    ("\u201c", '"'),   # left double quote
    ("\u201d", '"'),   # right double quote
    ("\u2022", "*"),   # bullet
    ("\u2192", "->"),  # right arrow
    ("\u2190", "<-"),  # left arrow
    ("\u00d7", "x"),   # multiplication sign
    ("\u00b0", "deg"), # degree sign
    ("\u00b2", "2"),   # superscript 2
    ("\u03b7", "eta"), # eta
    ("\u2265", ">="),  # >=
    ("\u2264", "<="),  # <=
    ("\u00b1", "+/-"), # plus-minus
    ("\u25bc", "v"),   # down triangle
    ("\u2502", "|"),   # box drawing
    ("\u251c", "+"),
    ("\u2514", "+"),
    ("\u2500", "-"),
    ("\u2502", "|"),
    ("\u251c", "+"),
    ("\u2514", "\\"),
]

def sanitize(text: str) -> str:
    for char, repl in REPLACEMENTS:
        text = text.replace(char, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(110, 110, 110)
        self.cell(0, 7, "keypoints_pipeline - Session Conversation  |  2026-05-04",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200, 200, 200)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


with open(SRC, encoding="utf-8") as f:
    raw = f.read()

lines = raw.splitlines()

pdf = PDF()
pdf.set_margins(MARGIN, MARGIN, MARGIN)
pdf.set_auto_page_break(auto=True, margin=14)
pdf.add_page()

# Title block
pdf.set_font("Helvetica", "B", 15)
pdf.set_text_color(25, 25, 25)
pdf.cell(0, 10, "keypoints_pipeline - Conversation Log", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

LINE_H = 5
pdf.set_font("Helvetica", "", 8.5)
pdf.set_text_color(30, 30, 30)

for raw_line in lines:
    text = sanitize(raw_line)
    # Wrap long lines
    wrapped = textwrap.wrap(text, width=115) if text.strip() else [""]
    for wline in wrapped:
        if pdf.get_y() > pdf.page_break_trigger:
            pdf.add_page()
        pdf.cell(0, LINE_H, wline, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.output(OUT)
print(f"Saved: {OUT}")

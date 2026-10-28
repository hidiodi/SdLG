"""Build the compact evidence-to-design synthesis figure."""

from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas


OUT = Path(__file__).resolve().parent / "plots" / "rebuilt" / "fig7_operational_synthesis.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 340, 210
c = canvas.Canvas(str(OUT), pagesize=(W, H))
c.setTitle("Operational evidence synthesis")


def wrapped(text, x, y, width, leading=7, font="Helvetica", size=6.3, color=black):
    words = text.split()
    lines, line = [], ""
    c.setFont(font, size)
    for word in words:
        candidate = f"{line} {word}".strip()
        if c.stringWidth(candidate, font, size) <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    c.setFillColor(color)
    for idx, value in enumerate(lines[:5]):
        c.drawCentredString(x + width / 2, y - idx * leading, value)


c.setFont("Helvetica-Bold", 10)
c.drawCentredString(W / 2, 197, "From task demand to a bounded design test")

headers = ["Demand", "Intervention", "Observation", "What to test next"]
xs = [8, 91, 174, 257]
box_w = 75
for x, header in zip(xs, headers):
    c.setFillColor(HexColor("#26364a"))
    c.roundRect(x, 174, box_w, 15, 3, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawCentredString(x + box_w / 2, 179, header)

rows = [
    ("Long horizon", "Execution x phase", "Train and unseen horizons diverge", "Evaluate both axes"),
    ("Transient cue", "Local path W", "Higher mean; uncertain effect", "Test local preprocessing"),
    ("Shared trunk", "Actor-gradient scale", "Trajectory and endpoint change", "Tune gradient coupling"),
]
fills = ["#e8f1f7", "#edf4ea", "#f5ece7"]
for row_idx, (demand, intervention, observation, next_test) in enumerate(rows):
    y = 134 - row_idx * 38
    for col_idx, text in enumerate((demand, intervention, observation, next_test)):
        x = xs[col_idx]
        c.setFillColor(HexColor(fills[row_idx]))
        c.setStrokeColor(HexColor("#aab4bd"))
        c.roundRect(x, y, box_w, 28, 4, fill=1, stroke=1)
        wrapped(text, x + 3, y + 19, box_w - 6, leading=6.3, size=6.0)
        if col_idx < 3:
            c.setStrokeColor(HexColor("#667788"))
            c.line(x + box_w + 1, y + 14, xs[col_idx + 1] - 2, y + 14)
            c.line(xs[col_idx + 1] - 5, y + 17, xs[col_idx + 1] - 2, y + 14)
            c.line(xs[col_idx + 1] - 5, y + 11, xs[col_idx + 1] - 2, y + 14)

c.setFillColor(HexColor("#3e6780"))
c.roundRect(8, 19, 158, 25, 4, fill=1, stroke=0)
wrapped("Tier 2: retrieval capacity supports addressing-dimension guidance, not an RL cause", 13, 36, 148, 6.2, size=5.9, color=white)
c.setFillColor(HexColor("#73624e"))
c.roundRect(174, 19, 158, 25, 4, fill=1, stroke=0)
wrapped("Tier 3: Craftax, POPGym, and Minesweeper bound any global ranking", 179, 36, 148, 6.2, size=5.9, color=white)
c.showPage()
c.save()

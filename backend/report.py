"""PDF progress report built with reportlab from the student's real saved attempts."""
import io
from collections import Counter
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ST = getSampleStyleSheet()


def _p(text, style="BodyText"):
    return Paragraph(escape(str(text)), ST[style])


def _table(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5b4bff")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd3e6")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f5fb")])]))
    return t


def build_pdf(attempts, title, username=None):
    """attempts: rows as returned by db.attempts (newest first) with feedback/missing/covered already decoded."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title=title)
    scores = [a["score"] for a in attempts]
    el = [_p(title, "Title"), _p(f"Student: {username or 'Guest'}    Generated from {len(attempts)} saved answer(s)", "Normal"), Spacer(1, 6)]
    el += [_p("Summary", "Heading2"), _p(f"Questions attempted: {len(attempts)}.  Average score: {sum(scores) / len(scores):.0f}/100.  "
           f"Best: {max(scores)}.  Lowest: {min(scores)}.  Scores are estimates from TF-IDF/cosine similarity and key-concept matching, "
           "not proof of technical correctness.")]
    by = {}
    for a in attempts:
        by.setdefault(a["topic"], []).append(a["score"])
    el += [Spacer(1, 6), _p("Topic-wise performance", "Heading2"),
           _table([["Topic", "Answers", "Average"]] + [[t, len(v), f"{sum(v) / len(v):.0f}"] for t, v in sorted(by.items(), key=lambda x: sum(x[1]) / len(x[1]))],
                  [90 * mm, 30 * mm, 30 * mm])]
    cov, mis = Counter(), Counter()
    for a in attempts:
        cov.update(a.get("covered", [])); mis.update(a.get("missing", []))
    el += [Spacer(1, 6), _p("Concepts you covered correctly (most frequent)", "Heading2"), _p(", ".join(k for k, _ in cov.most_common(12)) or "None recorded yet."),
           _p("Concepts you often missed", "Heading2"), _p(", ".join(k for k, _ in mis.most_common(12)) or "None recorded."), Spacer(1, 6)]
    weak = [t for t, v in by.items() if sum(v) / len(v) < 60]
    tips = [f"Revise {t} and re-attempt its questions." for t in weak] or ["No topic is below 60. Try harder questions or new topics."]
    if mis:
        tips.append("Practise explaining these concepts in your own words: " + ", ".join(k for k, _ in mis.most_common(5)) + ".")
    el += [_p("Improvement suggestions", "Heading2")] + [_p("- " + t) for t in tips] + [Spacer(1, 8), _p("Answers", "Heading2")]
    rows = [["Question", "Level", "Score", "Missing concepts"]]
    for a in attempts:
        rows.append([Paragraph(escape(a["question"][:220]), ST["BodyText"]), a["difficulty"], f"{a['score']} ({a['label']})",
                     Paragraph(escape(", ".join(a.get("missing", [])[:6]) or "-"), ST["BodyText"])])
    el.append(_table(rows, [88 * mm, 18 * mm, 28 * mm, 46 * mm]))
    doc.build(el)
    return buf.getvalue()

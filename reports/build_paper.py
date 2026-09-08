"""Local: render paper.md with existing ReportLab. Global: keep one editable paper source.

Sources: https://docs.reportlab.com/reportlab/userguide/ch5_platypus/ and
https://docs.reportlab.com/reportlab/userguide/ch6_paragraphs/ describe the layout APIs.
This deliberately supports only the headings, paragraphs, tables, links and images used here.
"""

import re  # Local: recognize the paper's limited Markdown; global: avoid another document dependency.
from pathlib import Path  # Local: resolve companion figures; global: make the script work from the repository root.
from xml.sax.saxutils import escape  # Local: escape literal text before markup; global: preserve scientific symbols safely.

from reportlab.lib import colors  # Local: style evidence tables; global: keep printed results readable.
from reportlab.lib.pagesizes import A4  # Local: use standard paper dimensions; global: support portable review.
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # Local: reuse typography primitives; global: avoid manual text placement.
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # Local: reuse flow layout; global: retain all text across pages.

from rl_locomotion_steering_vectors.storage import start_logging  # Local: reuse timestamped terminal/file logging; global: record artifact construction consistently.
import logging  # Local: report the actual output; global: make build failures visible to the caller.


def inline(text: str) -> str:
    """Local: translate used inline markup. Global: preserve prose, units and source attribution."""
    text = text.replace("−", "-").replace("→", " to ").replace("≥", ">=").replace("≤", "<=")  # Local: normalize unsupported font glyphs; global: prevent missing scientific symbols in portable PDF fonts.
    text = escape(text)  # Local: protect ampersands and angle brackets; global: render literal equations correctly.
    text = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<a href="\2" color="#174d78">\1</a>', text)  # Local: retain external citations as active links; global: let readers inspect primary evidence.
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", text)  # Local: print repository-relative artifact references; global: avoid machine-specific file links.
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)  # Local: preserve the paper's emphasis; global: make main findings easy to find.
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)  # Local: distinguish commands and notation; global: keep reproduction instructions unambiguous.


def footer(canvas, document):
    """Local: number pages. Global: support review of the standalone research paper."""
    canvas.saveState()  # Local: isolate footer drawing; global: avoid affecting body typography.
    canvas.setFont("Helvetica", 8)  # Local: use a portable built-in font; global: avoid external font assets.
    canvas.setFillColor(colors.HexColor("#5b6673"))  # Local: keep the footer unobtrusive; global: emphasize evidence in the body.
    canvas.drawString(44, 23, "Activation steering in RL locomotion | research snapshot")  # Local: label detached pages; global: distinguish this document from an accepted publication.
    canvas.drawRightString(A4[0] - 44, 23, str(document.page))  # Local: show the actual page number; global: make the PDF navigable.
    canvas.restoreState()  # Local: restore previous drawing state; global: keep the footer independent.


def build(source: Path) -> Path:
    """Local: render the complete Markdown source. Global: produce a reproducible PDF without rerunning experiments."""
    styles = getSampleStyleSheet()  # Local: reuse ReportLab styles; global: keep the renderer small.
    styles["BodyText"].fontSize, styles["BodyText"].leading = 9.3, 12.7  # Local: choose readable compact prose; global: fit a research paper without tiny text.
    styles["BodyText"].spaceAfter = 6  # Local: separate paragraphs; global: preserve the argument's structure.
    styles["Heading1"].fontSize, styles["Heading1"].leading = 15, 19  # Local: distinguish major sections; global: aid scanning.
    styles["Heading1"].keepWithNext = styles["Heading2"].keepWithNext = True  # Local: keep headings with their content; global: avoid stranded labels.
    styles.add(ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=8, leading=10.5, spaceAfter=0))  # Local: wrap table cells; global: avoid clipped measurements.
    lines, story, index = source.read_text(encoding="utf-8").splitlines(), [], 0  # Local: read the only prose source; global: keep Markdown and PDF synchronized.
    width = A4[0] - 88  # Local: calculate the frame width; global: align text, figures and tables.
    while index < len(lines):  # Local: process every line once; global: do not omit trailing references.
        line = lines[index].strip()  # Local: recognize block syntax; global: tolerate harmless source indentation.
        if not line:  # Local: skip Markdown separators; global: avoid unexplained empty flowables.
            index += 1  # Local: advance beyond the empty line; global: guarantee parser progress.
            continue  # Local: begin the next block; global: keep paragraph handling simple.
        if line == "<!-- pagebreak -->":  # Local: honor explicit scientific section boundaries; global: make the intended paper pagination reproducible.
            story.append(PageBreak())  # Local: request a page boundary; global: separate large evidence sections cleanly.
        elif line.startswith("# "):  # Local: recognize the document title; global: keep the first page distinct.
            story.extend([Paragraph(inline(line[2:]), styles["Title"]), Spacer(1, 10)])  # Local: wrap the title; global: avoid a hard-coded title width.
        elif line.startswith("## "):  # Local: recognize main sections; global: preserve scientific structure.
            story.append(Paragraph(inline(line[3:]), styles["Heading1"]))  # Local: render the exact source heading; global: maintain document parity.
        elif line.startswith("### "):  # Local: recognize subsections; global: retain hierarchy within methods/results.
            story.append(Paragraph(inline(line[4:]), styles["Heading2"]))  # Local: render the subsection; global: make long arguments navigable.
        elif line.startswith("| "):  # Local: recognize a measured-data table; global: preserve each row instead of copying hand-picked cells.
            rows = []  # Local: collect one table; global: retain its source ordering.
            while index < len(lines) and lines[index].strip().startswith("|"):  # Local: consume the contiguous table; global: prevent following prose joining a cell.
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]  # Local: split the limited table syntax; global: preserve signed numeric strings.
                if not all(re.fullmatch(r"[-: ]+", cell) for cell in cells):  # Local: omit the Markdown rule row; global: avoid printing syntax as data.
                    rows.append([Paragraph(inline(cell), styles["Cell"]) for cell in cells])  # Local: wrap every cell; global: retain long labels and intervals.
                index += 1  # Local: advance within the table; global: keep one-pass parsing deterministic.
            columns = len(rows[0])  # Local: derive the actual column count; global: support the paper's different evidence tables.
            table = Table(rows, colWidths=[width / columns] * columns, repeatRows=1, hAlign="LEFT")  # Local: fit and repeat the header; global: keep tables readable if they span pages.
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef5")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#718096")), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))  # Local: separate the header clearly; global: preserve readable comparisons.
            story.extend([table, Spacer(1, 10)])  # Local: separate the next block; global: avoid cramped results.
            continue  # Local: keep the first following line unconsumed; global: prevent skipped paragraphs.
        elif line.startswith("!["):  # Local: recognize a figure reference; global: include measured standalone artwork without altering its pixels.
            match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)  # Local: require the used image syntax; global: fail visibly on malformed scientific figure references.
            if match is None:  # Local: detect invalid input; global: do not silently drop an expected figure.
                raise ValueError(f"Unsupported figure syntax: {line}")  # Local: explain the issue; global: make the paper build reviewable.
            figure = Image(str(source.parent / match[2]))  # Local: load the saved measured plot; global: keep image provenance relative to the paper.
            scale = min(width / figure.imageWidth, 320 / figure.imageHeight)  # Local: preserve aspect ratio within page space; global: avoid distorted axes or oversized figures.
            figure.drawWidth, figure.drawHeight = figure.imageWidth * scale, figure.imageHeight * scale  # Local: scale display dimensions only; global: preserve the original figure file.
            story.extend([figure, Paragraph(inline(match[1]), styles["BodyText"]), Spacer(1, 8)])  # Local: pair the figure and source caption; global: keep visual evidence interpretable.
        else:  # Local: collect an ordinary paragraph; global: preserve connected scientific prose.
            paragraph = [line]  # Local: begin with the current text; global: retain every source sentence.
            while index + 1 < len(lines) and lines[index + 1].strip() and not lines[index + 1].startswith(("#", "|", "![", "<!--")):  # Local: join soft line breaks; global: avoid splitting a logical paragraph into many boxes.
                index += 1  # Local: advance to the continuation; global: consume it exactly once.
                paragraph.append(lines[index].strip())  # Local: retain continuation text; global: preserve the Markdown paragraph content.
            story.append(Paragraph(inline(" ".join(paragraph)), styles["BodyText"]))  # Local: wrap through ReportLab; global: avoid manual line clipping.
        index += 1  # Local: advance after a completed block; global: terminate after the full document.
    target = source.with_suffix(".pdf")  # Local: use the companion filename; global: make both formats easy to locate.
    document = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=44, rightMargin=44, topMargin=40, bottomMargin=42, title="Activation steering in frozen RL locomotion policies", author="RL locomotion steering research")  # Local: reserve printable margins; global: publish consistent document metadata.
    document.build(story, onFirstPage=footer, onLaterPages=footer)  # Local: paginate the full story; global: use the standard library's layout engine.
    logging.info("Rendered paper source=%s output=%s", source, target)  # Local: record actual artifact paths; global: retain a timestamped build trace.
    return target  # Local: expose the created artifact; global: support later automation without guessed paths.


if __name__ == "__main__":  # Local: run only on explicit script invocation; global: allow importing the formatter for focused checks.
    start_logging(Path("tmp/pdfs/paper"))  # Local: reuse complete terminal and timestamped file logs; global: keep artifact generation auditable.
    build(Path(__file__).with_name("paper.md"))  # Local: build the adjacent source; global: keep the command independent of the shell's path spelling.

"""Local: test the paper renderer's inline glyph mapping. Global: keep printed signs identical to the Markdown source.

Source: https://docs.reportlab.com/reportlab/userguide/ch6_paragraphs/ (paragraph markup the renderer emits).
"""
import re  # Local: inspect uncompressed PDF page records; global: verify small fixtures without another PDF dependency.

from reportlab import rl_config  # Local: disable compression in the text-preservation fixture; global: inspect actual output rather than a mock.
from reportlab.pdfbase.pdfdoc import PDFDocument, PDFString  # Local: use ReportLab's metadata encoding with an unencrypted document; global: check the exact Unicode author value in the PDF.

from reports.build_paper import build, inline  # Local: test the renderer used for paper and audit PDFs; global: keep generated artifacts reproducible.


def test_minus_plus_keeps_its_sign_order():
    """Local: ∓ must not print as ±. Global: '∓9.22°' lists −δ first, so swapping the order flips the reported rotation."""
    assert inline("by ∓9.22°") == "by -/+9.22°"  # Helvetica lacks ∓; an explicit ASCII pair preserves the order.
    assert inline("±0.1") == "±0.1"  # Helvetica has ±, so it stays unchanged.


def test_pdf_build_is_byte_deterministic_and_names_requested_author(tmp_path):
    """Local: build the same source twice. Global: let publication --check compare PDF bytes reliably."""
    source = tmp_path / "paper.md"  # Local: isolate the output; global: do not overwrite a scientific paper in tests.
    source.write_text("# Reproducible evidence\n\nA fixed paragraph.\n", encoding="utf-8")  # Local: minimal source; global: focus on metadata determinism.
    first = build(source).read_bytes()  # Local: retain the first complete PDF; global: compare artifacts, not timestamps.
    second = build(source).read_bytes()  # Local: rebuild through the same public entry point; global: exercise the publication contract.
    assert first == second  # Local: dates and IDs must not vary; global: --check cannot fail just because time passed.
    assert b"/Author " + PDFString("Libor “Burny” Burian").format(PDFDocument(invariant=1)) in second  # Local: test exact metadata spelling; global: preserve the requested attribution.


def test_generated_comments_are_hidden_but_pagebreak_is_preserved(tmp_path, monkeypatch):
    """Local: render marker comments and a page break. Global: keep generation bookkeeping out of the research paper."""
    monkeypatch.setattr(rl_config, "pageCompression", 0)  # Local: expose actual text commands; global: inspect a real PDF without adding a parser dependency.
    source = tmp_path / "markers.md"  # Local: keep the fixture separate; global: avoid changing published reports.
    source.write_text("# Marker test\n\n<!-- HIDDENCOMMENT begin -->\nVisibleFirst\n<!-- HIDDENCOMMENT end -->\n\n<!--\nHIDDENMULTILINE\n-->\n<!-- pagebreak -->\nVisibleSecond\n", encoding="utf-8")  # Local: exercise single-line and multiline HTML comments; global: retain the supported explicit page boundary.
    pdf = build(source).read_bytes()  # Local: build actual output; global: catch comments accidentally entering paragraphs.
    assert b"HIDDENCOMMENT" not in pdf and b"HIDDENMULTILINE" not in pdf  # Local: marker payloads must disappear; global: do not publish implementation bookkeeping as prose.
    assert b"VisibleFirst" in pdf and b"VisibleSecond" in pdf  # Local: retain neighboring text; global: comments must not swallow findings.
    assert len(re.findall(rb"/Type /Page\b", pdf)) == 2  # Local: pagebreak still makes two pages; global: preserve deliberate pagination.

"""Local: test the paper renderer's inline glyph mapping. Global: keep printed signs identical to the Markdown source.

Source: https://docs.reportlab.com/reportlab/userguide/ch6_paragraphs/ (paragraph markup the renderer emits).
"""
from reports.build_paper import inline  # Test the exact formatter used for the paper and audit PDFs.


def test_minus_plus_keeps_its_sign_order():
    """Local: ∓ must not print as ±. Global: '∓9.22°' lists −δ first, so swapping the order flips the reported rotation."""
    assert inline("by ∓9.22°") == "by -/+9.22°"  # Helvetica lacks ∓; an explicit ASCII pair preserves the order.
    assert inline("±0.1") == "±0.1"  # Helvetica has ±, so it stays unchanged.

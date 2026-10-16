# Building the research paper

`paper.md` is the single prose source. From the repository root, after the normal `uv sync` setup, run:

```powershell
uv run python reports/build_paper.py
```

The small renderer reuses ReportLab's paragraph, table, image and page-layout classes. It supports only the Markdown structures used in this paper, including explicit page breaks. It writes `paper.pdf` and timestamped build logs under `tmp/pdfs/paper`. It neither loads policies nor runs experiments. The implementation comments link the ReportLab APIs.

The copied figures in `paper_assets` preserve their source bytes:

- `ant-lateral-validation.png`: `runs/ant-classic-005/strength_response_1.png`, all saved lateral validation strength-response series, including controls.
- `ant-topdown-validation.png`: `runs/ant-classic-005/videos/preliminary/seed-510000-topdown-path.png`, saved ground-plane trajectory plot for the first predetermined validation seed.

Numerical tables summarize existing results; complete condition rows remain in the individual experiment bundles. Source paths, phase provenance, and limitations are recorded in the paper and experiment reports. The [documentation audit](documentation-audit.md) records corrections and verification without changing scientific outcomes. Original generated run reports are preserved as historical snapshots; their Markdown copies match after CRLF/LF normalization, while PDFs, PNGs and NPZs match source bytes and compacted JSON matches source values.

For each final PDF, render every page with Poppler and inspect clipping, table wrapping, figures, captions and reference readability. Repeat that verification after changing prose or layout, and check Markdown/PDF content and active-link parity. The documentation audit records the verified version; a previous render does not verify newly edited prose. The paper is a timestamped research snapshot, not an assertion of peer review or formal protocol success.

The 2026-10-16 edition adds the [scientific-rigor audit](scientific-rigor-audit.md) qualifications to the abstract and Sections 4, 6 and 8, plus reference [E6]. It has ten pages.

A later 2026-10-16 revision adds the [world-Y specificity audit](ant-classic-005-y-shift-audit.md) to the abstract, Sections 6 and 8, and reference [E7]. It still has ten pages.
- The first build overflowed to twelve pages, leaving one- and two-line spills on new pages. The abstract and Section 6 additions were shortened until the page boundaries matched the previous edition.
- Every page was rendered at 1.6× with `uv run --no-sync --with pypdfium2==4.30.0` into `tmp/pdfs/paper-yshift/` and inspected visually. No clipping or broken tables were found.

A third 2026-10-16 revision adds the [path-mechanism audit](ant-classic-005-mechanism-audit.md) to Sections 6 and 8 and reference [E8]. It still has ten pages.
- The builder has no list support, so the Section 8 corrections are written as prose.
- Pages 6, 8, 9 and 10, which contain the changed text or follow it, were rendered at 1.6× into `tmp/pdfs/paper-mechanism/` and inspected visually. No clipping or broken tables were found.
- `pypdf` 5.1.0 confirmed that all 56 distinct external Markdown links are active PDF link annotations, and that the two sets are equal.
- `pypdf` 5.1.0 confirmed that all 54 distinct external Markdown links are active PDF link annotations, and that the two sets are equal.
- Poppler was not available, so every page was rendered at 1.6× with `uv run --no-sync --with pypdfium2==4.30.0` into `tmp/pdfs/paper-rigor/` and inspected visually. No clipping or broken tables were found.
- `pypdf` 5.1.0 confirmed that the 52 distinct external Markdown links are all active PDF link annotations.

The verification record below describes the earlier edition only. The corrected 2026-09-08 edition has nine pages. Its [verification record](documentation-audit/paper-verification.json) identifies the exact Markdown/PDF hashes, checks all 189 prose/table units and 50 distinct external links, and records visual inspection of every page. Evidence links use public GitHub URLs so they also work in the standalone PDF and the Hugging Face paper copy; unchanged numerical artifacts are pinned where referenced.

The featured Ant section links the three public GitHub movie attachments and a pinned Hugging Face vector archive. The movies returned HTTP 200 with `video/mp4`; the vector page returned HTTP 200. PDF link annotations were checked with pypdf to confirm that all four destinations are active links, rather than printed text alone. The rendered figures remain ordinary measured PNGs; the renderer does not embed MP4 data in the PDF.

The 2026-10-16 independent-audit revision adds [E9] to the front matter, abstract and Section 8. It still has ten pages.
- The first build overflowed one line onto page 2, ahead of the explicit page break. The abstract and front-matter additions were shortened until Section 2 again began on page 2.
- The renderer now also draws one-line list items, Greek letters (via ReportLab's standard Symbol font), `<super>` exponents and escaped `\|` table pipes. These serve the [independent audit report](ant-classic-005-independent-audit.md), built with `uv run python reports/build_paper.py reports/ant-classic-005-independent-audit.md`. `paper.md` uses none of them.
- All pages of both PDFs were rendered at 1.6× with `uv run --no-sync --with pypdfium2==4.30.0` into `tmp/pdfs/paper-independent/` and `tmp/pdfs/independent-audit/` and inspected visually. No clipping, missing glyphs or broken tables were found.

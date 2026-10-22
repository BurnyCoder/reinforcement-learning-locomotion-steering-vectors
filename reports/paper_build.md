# Building the research paper

`paper.md` is the single prose source. From the repository root, after the normal `uv sync` setup, run:

```powershell
uv run python reports/build_paper.py
```

The small renderer reuses ReportLab's paragraph, table, image and page-layout classes. It supports only the Markdown structures used in this paper and the audit reports, including explicit page breaks. It writes `paper.pdf` beside its source and timestamped build logs under `tmp/pdfs/paper`. It neither loads policies nor runs experiments. The implementation comments link the ReportLab APIs.

The copied figures in `paper_assets` preserve their source bytes:

- `ant-lateral-validation.png`: `runs/ant-classic-005/strength_response_1.png`, all saved lateral validation strength-response series, including controls.
- `ant-topdown-validation.png`: `runs/ant-classic-005/videos/preliminary/seed-510000-topdown-path.png`, saved ground-plane trajectory plot for the first predetermined validation seed.

Numerical tables summarize existing results; complete condition rows remain in the individual experiment bundles. Source paths, phase provenance, and limitations are recorded in the paper and experiment reports. The [documentation audit](documentation-audit.md) records corrections and verification without changing scientific outcomes. Generated run reports are rendered from saved evidence. On 2026-10-22 they were regenerated only to replace a broken ReportLab link; see the [fact-check](fact-check-2026-10-22.md). Their Markdown copies match the run directories after CRLF/LF normalization, PDFs, PNGs and NPZs match source bytes, and compacted JSON matches source values.

For each final PDF, render every page and inspect clipping, table wrapping, figures, captions and reference readability. Repeat that verification after changing prose or layout, and check Markdown/PDF content and active-link parity. A previous render does not verify newly edited prose. The paper is a timestamped research snapshot, not an assertion of peer review or formal protocol success.

## Edition history

**2026-09-08.** The corrected edition has nine pages. Its [verification record](documentation-audit/paper-verification.json) identifies the exact Markdown/PDF hashes, checks all 189 prose/table units and 50 distinct external links, and records visual inspection of every page. Evidence links use public GitHub URLs so they also work in the standalone PDF and the Hugging Face paper copy; unchanged numerical artifacts are pinned where referenced. The featured Ant section links the three public GitHub movie attachments and a pinned Hugging Face vector archive. The movies returned HTTP 200 with `video/mp4`; the vector page returned HTTP 200. PDF link annotations were checked with pypdf to confirm that all four destinations are active links, rather than printed text alone. The rendered figures remain ordinary measured PNGs; the renderer does not embed MP4 data in the PDF.

**2026-10-16, [E6].** This edition adds the [scientific-rigor audit](scientific-rigor-audit.md) qualifications to the abstract and Sections 4, 6 and 8, plus reference [E6]. It has ten pages.
- Poppler was not available, so every page was rendered at 1.6× with `uv run --no-sync --with pypdfium2==4.30.0` into `tmp/pdfs/paper-rigor/` and inspected visually. No clipping or broken tables were found.
- `pypdf` 5.1.0 confirmed that the 52 distinct external Markdown links are all active PDF link annotations.

**2026-10-16, [E7].** This revision adds the [world-Y specificity audit](ant-classic-005-y-shift-audit.md) to the abstract, Sections 6 and 8, and reference [E7]. It still has ten pages.
- The first build overflowed to twelve pages, leaving one- and two-line spills on new pages. The abstract and Section 6 additions were shortened until the page boundaries matched the previous edition.
- Every page was rendered at 1.6× with `uv run --no-sync --with pypdfium2==4.30.0` into `tmp/pdfs/paper-yshift/` and inspected visually. No clipping or broken tables were found.
- `pypdf` 5.1.0 confirmed that all 54 distinct external Markdown links are active PDF link annotations, and that the two sets are equal.

**2026-10-16, [E8].** This revision adds the [path-mechanism audit](ant-classic-005-mechanism-audit.md) to Sections 6 and 8 and reference [E8]. It still has ten pages.
- At that revision the builder had no list support, so the Section 8 corrections are written as prose.
- Pages 6, 8, 9 and 10, which contain the changed text or follow it, were rendered at 1.6× into `tmp/pdfs/paper-mechanism/` and inspected visually. No clipping or broken tables were found.
- `pypdf` 5.1.0 confirmed that all 56 distinct external Markdown links are active PDF link annotations, and that the two sets are equal.

**2026-10-16, [E9].** The independent-audit revision adds [E9] to the front matter, abstract and Section 8. It still has ten pages.
- The first build overflowed one line onto page 2, ahead of the explicit page break. The abstract and front-matter additions were shortened until Section 2 again began on page 2.
- The renderer gained one-line list items, Greek letters (via ReportLab's standard Symbol font), `<super>` exponents and escaped `\|` table pipes. These serve the [independent audit report](ant-classic-005-independent-audit.md), built with `uv run python reports/build_paper.py reports/ant-classic-005-independent-audit.md`. At that revision `paper.md` used none of them.
- All pages of both PDFs were rendered at 1.6× with `uv run --no-sync --with pypdfium2==4.30.0` into `tmp/pdfs/paper-independent/` and `tmp/pdfs/independent-audit/` and inspected visually. No clipping, missing glyphs or broken tables were found.

**2026-10-22, [E10].** The [fact-check](fact-check-2026-10-22.md) revision corrects wording, signs and provenance statements in the front matter, abstract and Sections 4, 6, 8 and 9. It also adds the author byline, reference [E10] and the renamed GitHub repository URLs. It still has ten pages.
- The first build overflowed three lines onto page 2. The front-matter paragraphs were merged until Section 2 again began on page 2.
- The renderer now prints ∓ as "-/+" instead of "±", which had reversed the sign order of "∓9.22°" in the independent-audit PDF. A unit test (`tests/test_build_paper.py`) covers this.
- All ten paper pages and the changed independent-audit pages were rendered at 1.6× with `uv run --no-sync --with pypdfium2==4.30.0` and inspected visually. No clipping, missing glyphs or broken tables were found.
- `pypdf` 5.1.0 confirmed that the paper's 60 distinct external Markdown links and the independent audit's 9 are exactly the PDFs' link annotations.

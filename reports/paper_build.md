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

The corrected 2026-09-08 edition has nine pages. Its [verification record](documentation-audit/paper-verification.json) identifies the exact Markdown/PDF hashes, checks all 189 prose/table units and 50 distinct external links, and records visual inspection of every page. Evidence links use public GitHub URLs so they also work in the standalone PDF and the Hugging Face paper copy; unchanged numerical artifacts are pinned where referenced.

The featured Ant section links the three public GitHub movie attachments and a pinned Hugging Face vector archive. The movies returned HTTP 200 with `video/mp4`; the vector page returned HTTP 200. PDF link annotations were checked with pypdf to confirm that all four destinations are active links, rather than printed text alone. The rendered figures remain ordinary measured PNGs; the renderer does not embed MP4 data in the PDF.

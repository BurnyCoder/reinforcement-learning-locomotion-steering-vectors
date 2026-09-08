# Building the research paper

`paper.md` is the single prose source. From the repository root, after the normal `uv sync` setup, run:

```powershell
uv run python reports/build_paper.py
```

The small renderer reuses ReportLab's paragraph, table, image and page-layout classes. It supports only the Markdown structures used in this paper, including explicit page breaks. It writes `paper.pdf` and timestamped build logs under `tmp/pdfs/paper`. It neither loads policies nor runs experiments. The implementation comments link the ReportLab APIs.

The copied figures in `paper_assets` preserve their source bytes:

- `ant-lateral-validation.png`: `runs/ant-classic-005/strength_response_1.png`, all saved lateral validation strength-response series, including controls.
- `ant-topdown-validation.png`: `runs/ant-classic-005/videos/preliminary/seed-510000-topdown-path.png`, saved ground-plane trajectory plot for the first predetermined validation seed.

Numerical tables summarize existing results; complete condition rows remain in the individual experiment bundles. Source paths, phase provenance, and limitations are recorded in the paper and experiment reports. Rebuilding a PDF does not update a scientific outcome.

The final PDF is nine pages. Every page was rendered with Poppler and visually inspected for clipping, table wrapping, figures, captions and reference readability. Rebuild and repeat that inspection after changing prose or layout. The paper is a timestamped research snapshot, not an assertion of peer review or formal protocol success.

The featured Ant section links the three public GitHub movie attachments and a pinned Hugging Face vector archive. The movies returned HTTP 200 with `video/mp4`; the vector page returned HTTP 200. PDF link annotations were checked with pypdf to confirm that all four destinations are active links, rather than printed text alone. The rendered figures remain ordinary measured PNGs; the renderer does not embed MP4 data in the PDF.

"""Local: render measured run artifacts. Global: keep Markdown and PDF conclusions aligned.

Sources: https://docs.reportlab.com/reportlab/userguide/ch5_platypus/ and
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html.
"""

import json  # Local: read structured artifacts; global: report saved evidence without rerunning selection.
import logging  # Local: announce output paths; global: retain report generation in timestamped run logs.
import math  # Local: identify nonfinite numbers; global: avoid presenting invalid statistics as measurements.
from collections import defaultdict  # Local: group plot series; global: separate controls from candidate vectors.
from datetime import datetime, timezone  # Local: timestamp rendering; global: distinguish report time from experiment time.
from pathlib import Path  # Local: resolve run artifacts; global: keep outputs beside their provenance.
from xml.sax.saxutils import escape  # Local: escape PDF markup; global: preserve literal artifact text safely.

import matplotlib  # Local: choose a headless backend; global: support unattended local research.

matplotlib.use("Agg")  # Local: render without GUI windows; global: make CLI reporting reproducible.
import matplotlib.pyplot as plt  # noqa: E402  # Local: draw strength curves; global: expose dose-response evidence.
from reportlab.lib import colors  # Local: set report colors; global: keep tables readable across pages.
from reportlab.lib.pagesizes import A4  # Local: use a standard page size; global: make the report portable.
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # Local: define wrapping styles; global: avoid clipped prose.
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # Local: reuse layout primitives; global: avoid custom PDF positioning.

LOGGER = logging.getLogger(__name__)  # Local: inherit CLI handlers; global: use one logging configuration.
UNITS = {"speed": "m/s", "effort": "mean squared action", "height": "m", "lateral": "m/s", "turning": "rad/s"}  # Local: label physical quantities; global: prevent an energy claim from an action proxy.
STAGES = ("validation", "confirmation", "replication")  # Local: retain stage ordering; global: separate selection from fresh evidence.


def _format(value):
    """Local: format scalar evidence. Global: never substitute zero for a missing measurement."""
    if value is None:  # Local: recognize absent fields; global: preserve uncertainty in incomplete artifacts.
        return "not recorded"  # Local: supply an explicit label; global: avoid invented results.
    if isinstance(value, bool):  # Local: handle booleans before numbers; global: show gate decisions clearly.
        return "yes" if value else "no"  # Local: spell out decisions; global: keep tables accessible.
    if isinstance(value, (int, float)):  # Local: format JSON numbers; global: apply consistent precision.
        return f"{value:.5g}" if math.isfinite(value) else "nonfinite"  # Local: reject misleading NaN displays; global: flag defective evidence.
    return str(value)  # Local: preserve categorical values; global: avoid interpretation beyond saved data.


def _measurement_rows(records):
    """Local: convert paired effects to a table. Global: expose evidence and each usefulness gate."""
    rows = [["Vector / behavior", "Alpha", "Delta", "95% CI", "Pairs", "Useful"]]  # Local: define stable columns; global: make stages comparable.
    for record in records:  # Local: retain every supplied condition; global: prevent selective presentation.
        effect = record.get("effect", {})  # Local: read the paired analysis; global: avoid recomputing inference during reporting.
        interval = effect.get("ci95")  # Local: retrieve recorded bounds; global: retain the original uncertainty estimate.
        ci = "not recorded" if interval is None else "[" + ", ".join(_format(x) for x in interval) + "]"  # Local: format both bounds; global: retain signed effects.
        label = f"{record.get('vector', 'unknown')} / {record.get('behavior', 'unknown')}"  # Local: identify condition and target; global: distinguish controls.
        rows.append([label, _format(record.get("alpha")), _format(effect.get("effect")), ci, _format(effect.get("n_pairs")), _format(effect.get("gate"))])  # Local: copy measured values; global: show null and failed conditions equally.
    return rows  # Local: share a single table; global: keep Markdown and PDF values identical.


def _strength_plots(run_dir, records):
    """Local: plot validation effects and CIs. Global: show the complete observed strength response."""
    groups = defaultdict(lambda: defaultdict(list))  # Local: index behavior then vector; global: keep different units on different axes.
    for record in records:  # Local: inspect each supplied validation measurement; global: do not extrapolate missing strengths.
        groups[record["behavior"]][record["vector"]].append(record)  # Local: preserve condition identity; global: compare candidate and control curves.
    images = []  # Local: collect generated figures; global: embed the same charts in both report formats.
    for index, (behavior, series) in enumerate(sorted(groups.items())):  # Local: make ordering deterministic; global: simplify repeated report review.
        figure, axis = plt.subplots(figsize=(7.0, 3.6), layout="constrained")  # Local: let Matplotlib reserve label space; global: avoid clipped figures.
        for name, points in sorted(series.items()):  # Local: create one curve per vector; global: make controls visible.
            points = sorted(points, key=lambda item: item["alpha"])  # Local: order strengths numerically; global: avoid a misleading zigzag curve.
            x = [point["alpha"] for point in points]  # Local: read actual tested strengths; global: retain intervention dose.
            y = [point["effect"]["effect"] for point in points]  # Local: use signed paired changes; global: show causal contrast with baseline.
            axis.plot(x, y, ".-", label=name, linewidth=1.2, markersize=5)  # Local: connect tested conditions; global: communicate observed response shape.
            for point in points:  # Local: draw bounds independently; global: support valid percentile intervals beyond the point estimate.
                interval = point["effect"].get("ci95")  # Local: use the stored bootstrap result; global: preserve the inference procedure.
                if interval is not None:  # Local: avoid fabricating missing uncertainty; global: keep incomplete data explicit.
                    axis.vlines(point["alpha"], interval[0], interval[1], color=axis.lines[-1].get_color(), alpha=0.45)  # Local: draw actual endpoints; global: avoid negative error-bar lengths.
        axis.axhline(0.0, color="#718096", linestyle="--", linewidth=0.8)  # Local: mark no change; global: make null effects apparent.
        axis.set(xlabel="Steering strength alpha", ylabel=f"Paired change ({UNITS.get(behavior, 'recorded units')})", title=f"{behavior.capitalize()}: validation only")  # Local: label units and split; global: prevent interpreting calibration as confirmation.
        axis.legend(fontsize=7, loc="best")  # Local: identify curves; global: keep random controls attributable.
        axis.grid(alpha=0.15)  # Local: add light reading guides; global: preserve data contrast.
        path = run_dir / f"strength_response_{index + 1}.png"  # Local: use safe deterministic filenames; global: avoid interpreting behavior text as a path.
        figure.savefig(path, dpi=170, facecolor="white")  # Local: write a crisp standalone image; global: support reuse outside the PDF.
        plt.close(figure)  # Local: release rendering memory; global: support repeated unattended reports.
        images.append((behavior, path))  # Local: preserve captions and paths; global: synchronize exported formats.
    return images  # Local: return actual outputs only; global: avoid references to nonexistent plots.


def _sections(manifest, results):
    """Local: compose evidence-based sections. Global: share narrative content between both formats."""
    status = results.get("status", "not recorded")  # Local: copy pipeline status; global: never infer overall success from one favorable row.
    selected = results.get("selected", [])  # Local: read locked validation choices; global: distinguish chosen and explored conditions.
    choices = "; ".join(f"{item['vector']} ({item['behavior']}), alpha={_format(item['alpha'])}" for item in selected) or "No selected intervention was recorded."  # Local: enumerate choices; global: make selection auditable.
    behavior_diagnostics = results.get("diagnostics", {}).get("behaviors", {})  # Local: retrieve contrast diagnostics; global: explain insufficient variation.
    diagnostics = "\n".join(f"{name}: {json.dumps(value, ensure_ascii=True, sort_keys=True)}" for name, value in sorted(behavior_diagnostics.items())) or "No behavior diagnostics were recorded."  # Local: retain diagnostic details; global: expose failed extraction attempts.
    sections = [  # Local: define ordered narrative blocks; global: place claims before supporting appendices.
        ("Outcome", f"Recorded run status: {status}. A gate pass is the prespecified pilot usefulness decision for one condition; it is not evidence of generalization across training seeds.\nValidation selection: {choices}", None),  # Local: qualify measured status; global: prevent overclaiming.
        ("Question and method", "Can adding a constant vector after the first actor ReLU change locomotion while preserving movement? The policy stays frozen. Vectors are episode-weighted differences of contrasting unsteered activations, normalized to centered activation RMS. Random and shuffled-label controls are evaluated on the same protocol. No behavioral cloning is used.", None),  # Local: state implemented extraction; global: explain the causal intervention.
        ("Interpretation", "Delta is treated minus baseline in the behavior's units. Speed/lateral use m/s, height uses m, turning uses rad/s, and effort is mean squared normalized action, not physical energy. Confidence intervals resample paired episodes. Validation selects strengths; confirmation and replication provide fresh evidence when recorded. Useful effects require the target threshold, a confidence interval excluding zero, and locomotion/quality checks. Control effects must be considered before attributing specificity to the extracted direction.", None),  # Local: define table meanings; global: separate causal changes from semantic claims.
        ("Fitting diagnostics", diagnostics, None),  # Local: include actual contrast failures; global: avoid omitting negative evidence.
    ]  # Local: end fixed sections; global: append only stages present in the saved run.
    for stage in STAGES:  # Local: render all supported analysis stages; global: preserve the selection/confirmation boundary.
        records = results.get(stage, [])  # Local: allow interrupted runs without later stages; global: represent incompleteness faithfully.
        sections.append((stage.capitalize(), f"{len(records)} recorded intervention conditions." if records else "This stage has no recorded measurements.", _measurement_rows(records) if records else None))  # Local: show every supplied result; global: do not invent replication.
    sections.append(("Reproduction and provenance", "The adjacent manifest.json records the execution specification and checkpoint provenance; results.json retains full per-condition metrics, quality deltas and bootstrap metadata. Saved rollout artifacts provide the underlying observations. Report rendering does not change selections or rerun policy inference.\n" + json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2), None))  # Local: retain complete run metadata; global: support audit and independent replay.
    sections.append(("Sources", "CAA method: https://arxiv.org/abs/2312.06681\nBaukit hooks: https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py\nEnvironment and measurement references: docs/sources.md in the repository.\nReporting: https://docs.reportlab.com/reportlab/userguide/ and https://matplotlib.org/stable/users/index.html", None))  # Local: identify scientific and implementation sources; global: separate adaptations from upstream claims.
    return sections  # Local: expose shared content; global: maintain agreement between report formats.


def _write_markdown(path, title, timestamp, sections, images):
    """Local: serialize readable experiment evidence. Global: retain a diffable companion to the PDF."""
    lines = [f"# {title}", "", f"Report generated: {timestamp}", ""]  # Local: create the report header; global: timestamp this artifact version.
    for heading, text, rows in sections:  # Local: follow the shared section order; global: preserve narrative parity with the PDF.
        lines.extend([f"## {heading}", "", text, ""])  # Local: separate blocks with blank lines; global: render correctly in CommonMark.
        if rows:  # Local: include only measured tables; global: avoid empty result scaffolding.
            lines.append("| " + " | ".join(rows[0]) + " |")  # Local: write column names; global: keep units and decisions interpretable.
            lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")  # Local: supply Markdown table syntax; global: render comparable columns.
            lines.extend("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows[1:])  # Local: escape literal separators; global: preserve recorded labels.
            lines.append("")  # Local: close the table block; global: prevent following paragraphs joining cells.
    lines.extend(["## Validation strength response", ""])  # Local: identify the chart split; global: keep plotted calibration distinct from held-out evidence.
    for behavior, image in images:  # Local: reference actual generated files; global: make the report portable with its run directory.
        lines.extend([f"![Validation strength response for {behavior}]({image.name})", ""])  # Local: embed a relative artifact link; global: keep a copied run self-contained.
    path.write_text("\n".join(lines), encoding="utf-8")  # Local: save UTF-8 text; global: retain complete measured content without truncation.


def _write_pdf(path, title, timestamp, sections, images):
    """Local: lay out the same report with ReportLab. Global: produce a portable research artifact."""
    styles = getSampleStyleSheet()  # Local: reuse documented typography defaults; global: keep the renderer compact.
    styles.add(ParagraphStyle("SmallCell", parent=styles["BodyText"], fontSize=7.5, leading=10, wordWrap="CJK"))  # Local: wrap long vector names and hashes; global: prevent table overflow.
    styles["BodyText"].fontSize, styles["BodyText"].leading = 9, 13  # Local: set legible compact prose; global: avoid excessive report length.
    story = [Paragraph(escape(title), styles["Title"]), Paragraph(escape(timestamp), styles["BodyText"]), Spacer(1, 12)]  # Local: start with title and timestamp; global: identify the report version.
    for heading, text, rows in sections:  # Local: consume shared report content; global: keep the PDF consistent with Markdown.
        story.append(Paragraph(escape(heading), styles["Heading2"]))  # Local: mark section boundaries; global: support visual scanning.
        for paragraph in text.split("\n"):  # Local: let long provenance lines wrap individually; global: prevent a single unbreakable block.
            if paragraph:  # Local: omit empty paragraph objects; global: avoid unexplained layout gaps.
                story.append(Paragraph(escape(paragraph), styles["BodyText"]))  # Local: escape XML-sensitive data; global: render artifact text literally.
        if rows:  # Local: typeset available result tables; global: preserve all measured conditions.
            cells = [[Paragraph(escape(str(cell)), styles["SmallCell"]) for cell in row] for row in rows]  # Local: wrap every table cell; global: avoid clipped labels and confidence intervals.
            table = Table(cells, colWidths=[154, 42, 60, 114, 43, 43], repeatRows=1, hAlign="LEFT")  # Local: fit the A4 content width and repeat headers; global: make multipage results readable.
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef5")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#8090a0")), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))  # Local: separate headers without dense borders; global: maintain clear evidence tables.
            story.extend([Spacer(1, 6), table, Spacer(1, 8)])  # Local: space table boundaries; global: separate evidence from surrounding prose.
    for behavior, image in images:  # Local: append the exact exported figures; global: maintain identical plotted data across formats.
        story.extend([Paragraph(escape(f"Validation strength response: {behavior}"), styles["Heading2"]), Image(str(image), width=490, height=252)])  # Local: preserve the figure aspect ratio; global: avoid distorted curves.
    document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=38, bottomMargin=40, title=title, author="RL locomotion steering research")  # Local: reserve consistent page margins; global: create portable metadata.
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)  # Local: delegate pagination to Platypus; global: render all rows without manual page clipping.


def _page_number(canvas, document):
    """Local: draw an unobtrusive footer. Global: keep multipage experiment reports navigable."""
    canvas.saveState()  # Local: isolate footer drawing settings; global: protect subsequent document layout.
    canvas.setFont("Helvetica", 8)  # Local: use a built-in portable font; global: avoid machine-specific font requirements.
    canvas.drawRightString(A4[0] - 40, 23, f"Page {document.page}")  # Local: print the current page; global: support report review references.
    canvas.restoreState()  # Local: restore drawing settings; global: keep footer behavior independent.


def write_report(run_dir: Path) -> Path:
    """Local: render one saved run. Global: expose the reporting phase to the orchestration CLI."""
    run_dir = Path(run_dir)  # Local: normalize caller path input; global: keep a single public interface.
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))  # Local: read exact run provenance; global: fail visibly if required evidence is absent.
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))  # Local: read saved analysis; global: never silently generate substitute results.
    sections = _sections(manifest, results)  # Local: compose shared evidence; global: synchronize both output formats.
    images = _strength_plots(run_dir, results.get("validation", []))  # Local: plot measured validation only; global: avoid held-out selection leakage.
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")  # Local: capture rendering time; global: make repeated report exports identifiable.
    title = f"RL locomotion steering: {run_dir.name}"  # Local: name the run; global: distinguish experiment artifacts.
    _write_markdown(run_dir / "report.md", title, timestamp, sections, images)  # Local: save reviewable text; global: retain a reproducible companion artifact.
    _write_pdf(run_dir / "report.pdf", title, timestamp, sections, images)  # Local: save the shareable report; global: use the same evidence as Markdown.
    LOGGER.info("Wrote experiment reports: %s and %s", run_dir / "report.md", run_dir / "report.pdf")  # Local: log output paths; global: include reporting in the execution record.
    return run_dir / "report.pdf"  # Local: return the primary artifact; global: let CLI callers announce the report.

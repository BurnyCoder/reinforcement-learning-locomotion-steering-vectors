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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # Local: reuse layout primitives; global: avoid custom PDF positioning.

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


def _fitting_diagnostics(results):
    """Local: summarize saved extraction counts. Global: explain contrast feasibility without dumping raw vectors."""
    extraction = results.get("extraction", {})  # Local: use the pipeline's extraction key; global: distinguish fitting from unsteered diagnostics.
    if not extraction:  # Local: handle runs stopped before extraction; global: do not imply missing work was completed.
        return "No fitting diagnostics were recorded."  # Local: state the evidence boundary; global: avoid invented contrast results.
    windows = extraction.get("windows", {})  # Local: retrieve eligibility accounting; global: keep exclusions visible.
    lines = ["Fitting windows: " + "; ".join(f"{name.replace('_', ' ')}={_format(value)}" for name, value in windows.items()) + "."]  # Local: render every saved count; global: explain how much data supported extraction.
    lines.append(f"Centered activation RMS: {_format(extraction.get('activation_rms'))}. Convention: {extraction.get('rms_convention', 'not recorded')}.")  # Local: state actual normalization; global: give intervention strengths their recorded scale.
    for behavior, details in sorted(extraction.get("behaviors", {}).items()):  # Local: enumerate fitted and rejected descriptors; global: retain negative findings.
        labels = [("status", "status"), ("high episodes", "high_episodes"), ("low episodes", "low_episodes"), ("high windows", "high_windows"), ("low windows", "low_windows"), ("contrast", "contrast"), ("raw norm", "raw_norm"), ("split-half cosine", "split_half_cosine")]  # Local: name interpretable diagnostics; global: avoid presenting hundreds of vector coordinates as prose.
        summary = "; ".join(f"{label}={_format(details.get(key))}" for label, key in labels)  # Local: preserve absent fields explicitly; global: distinguish rejection from a measured zero.
        lines.append(f"{behavior} ({UNITS.get(behavior, 'recorded units')}): {summary}.")  # Local: associate contrast units; global: prevent cross-descriptor scale confusion.
        if details.get("reason"):  # Local: retain rejection explanations; global: make failed hypotheses reviewable.
            lines.append(f"Reason: {details['reason']}")  # Local: copy the saved explanation; global: avoid speculative failure diagnosis.
    lines.append("Split-half cosine measures extraction agreement, not causal effectiveness. Full bin statistics and vector coordinates are retained in vector_diagnostics.json.")  # Local: point to complete evidence; global: avoid treating correlation as causal proof.
    return "\n".join(lines)  # Local: produce shared prose; global: keep Markdown and PDF synchronized.


def _provenance_text(manifest):
    """Local: summarize execution identity compactly. Global: retain complete machine-readable provenance beside the report."""
    source = manifest.get("provenance", {})  # Local: read verified checkpoint metadata; global: distinguish this run from another pretrained policy.
    lines = [f"Created: {manifest.get('created_utc', 'not recorded')}; code commit: {manifest.get('code_commit', 'not recorded')}."]  # Local: show chronology and code identity; global: make the report attributable.
    lines.append(f"Policy: {source.get('repo_id', manifest.get('model_key', 'not recorded'))}; algorithm: {source.get('algorithm', 'not recorded')}; environment: {source.get('env_id', 'not recorded')}.")  # Local: identify the actual RL checkpoint; global: preserve training provenance.
    lines.append(f"Checkpoint revision: {source.get('revision', 'not recorded')}. SHA256: {source.get('sha256', 'not recorded')}.")  # Local: retain full immutable identifiers; global: support integrity verification.
    lines.append(f"Layer: {source.get('layer', 'not recorded')}; hidden dimension: {_format(source.get('hidden_dimension'))}; observation/action shapes: {source.get('observation_shape', 'not recorded')} / {source.get('action_shape', 'not recorded')}.")  # Local: summarize model interfaces; global: make intervention placement reproducible.
    lines.append("Seed partitions below are registered assignments, not completion counts.")  # Local: explain the list once; global: distinguish planned episodes from observed measurements.
    for group, seeds in manifest.get("seed_splits", {}).items():  # Local: summarize each registered partition; global: preserve exact split identity compactly.
        ordered = sorted(seeds)  # Local: inspect sequence structure; global: avoid assuming a contiguous partition.
        span = f"{ordered[0]}-{ordered[-1]}" if ordered else "empty"  # Local: give a readable range; global: keep empty splits explicit.
        contiguous = bool(ordered) and ordered == list(range(ordered[0], ordered[-1] + 1))  # Local: check all intermediate seeds; global: do not imply nonexistent runs in sparse splits.
        lines.append(f"{group.capitalize()}: {len(seeds)} seeds; {'range' if contiguous else 'span'} {span}.")  # Local: print compact seed identity; global: retain split-specific reproducibility.
    lines.append("Configuration: " + "; ".join(f"{name}={_format(value)}" for name, value in manifest.get("config", {}).items()) + ".")  # Local: show actual settings without JSON scaffolding; global: preserve reproducibility.
    lines.append("Software: " + "; ".join(f"{name} {version}" for name, version in manifest.get("software", {}).items()) + ".")  # Local: show observed package versions; global: expose environment compatibility.
    lines.append("Full exact seed lists, checkpoint metadata and the original specification remain in adjacent manifest.json. Full measured analysis is in results.json; extraction details are in vector_diagnostics.json. The report does not change these artifacts or rerun inference.")  # Local: reference complete evidence; global: keep presentation compact without discarding provenance.
    return "\n".join(lines)  # Local: share concise provenance prose; global: keep Markdown and PDF aligned.


def _decision_text(manifest, results):
    """Local: preserve the saved research chronology. Global: link decisions to their recorded evidence."""
    lines = []  # Local: accumulate decisions only when recorded; global: avoid inventing an experiment narrative.
    for item in manifest.get("decisions", []):  # Local: retain manifest ordering; global: preserve the actual hypothesis sequence.
        if isinstance(item, dict):  # Local: handle structured decision metadata; global: include timestamps and evidence paths.
            description = item.get("decision", "Decision text not recorded.")  # Local: read the original rationale; global: avoid interpreting metadata as a new claim.
            description = "Outcome decision recorded above." if description == results.get("decision") else description  # Local: refer back to duplicate outcome prose; global: prevent repetitive reports.
            lines.append(f"{item.get('time_utc', 'Time not recorded')}: {description} Evidence: {item.get('evidence', 'not recorded')}.")  # Local: associate decision with evidence; global: preserve review chronology.
        else:  # Local: tolerate older plain-text decisions; global: keep existing run records reportable.
            lines.append(str(item))  # Local: copy the saved narrative; global: avoid silently dropping historical decisions.
    return "\n".join(lines)  # Local: expose narrative or an empty string; global: omit an unsupported chronology section.


def _sections(manifest, results):
    """Local: compose evidence-based sections. Global: share narrative content between both formats."""
    status = results.get("status", "not recorded")  # Local: copy pipeline status; global: never infer overall success from one favorable row.
    selected = results.get("selected", [])  # Local: read locked validation choices; global: distinguish chosen and explored conditions.
    choices = "; ".join(f"{item['vector']} ({item['behavior']}), alpha={_format(item['alpha'])}" for item in selected) or "No selected intervention was recorded."  # Local: enumerate choices; global: make selection auditable.
    diagnostics = _fitting_diagnostics(results)  # Local: read actual fitting records; global: explain contrast feasibility without exposing raw-coordinate clutter.
    sections = [  # Local: define ordered narrative blocks; global: place claims before supporting appendices.
        ("Outcome", f"Recorded run status: {status}.\n{results.get('decision', 'No additional decision narrative was recorded.')}\nValidation selection: {choices}\nA gate pass is a pilot usefulness decision for one condition; it does not establish generalization across training seeds.", None),  # Local: present the recorded conclusion directly; global: prevent a stopped fitting run looking like a causal success.
        ("Question and method", "Can adding a constant vector after the first actor ReLU change locomotion while preserving movement? The policy stays frozen. Vectors are episode-weighted differences of contrasting unsteered activations, normalized to centered activation RMS. The protocol includes random and shuffled-label controls on the same evaluation split when that stage is reached. No behavioral cloning is used.", None),  # Local: state the protocol without claiming unrun evaluations; global: explain the intended causal intervention.
        ("Interpretation", "Delta is treated minus baseline in the behavior's units. Speed/lateral use m/s, height uses m, turning uses rad/s, and effort is mean squared normalized action, not physical energy. Confidence intervals resample paired episodes. Validation selects strengths; confirmation and replication provide fresh evidence when recorded. Useful effects require the target threshold, a confidence interval excluding zero, and locomotion/quality checks. Control effects must be considered before attributing specificity to the extracted direction.", None),  # Local: define table meanings; global: separate causal changes from semantic claims.
        ("Fitting diagnostics", diagnostics, None),  # Local: include actual contrast failures; global: avoid omitting negative evidence.
    ]  # Local: end fixed sections; global: append only stages present in the saved run.
    decisions = _decision_text(manifest, results)  # Local: read research chronology; global: connect the result to its recorded reasoning.
    if decisions:  # Local: show a decision section only with evidence; global: avoid an invented narrative.
        sections.append(("Research decisions", decisions, None))  # Local: preserve rationale and references; global: support audit of hypothesis changes.
    missing = []  # Local: collect absent stages; global: summarize unperformed work without repeated empty sections.
    for stage in STAGES:  # Local: render all supported analysis stages; global: preserve the selection/confirmation boundary.
        records = results.get(stage, [])  # Local: allow interrupted runs without later stages; global: represent incompleteness faithfully.
        if records:  # Local: include actual measurements; global: never generate a table for an unperformed stage.
            sections.append((stage.capitalize(), f"{len(records)} recorded intervention conditions.", _measurement_rows(records)))  # Local: preserve every supplied result; global: avoid cherry-picking favorable conditions.
        else:  # Local: note stages without data; global: distinguish protocol assignments from results.
            missing.append(stage)  # Local: retain all absent stage names; global: make the research boundary explicit.
    if missing:  # Local: report the actual evidence gap once; global: keep stopped experiments concise.
        sections.append(("Unmeasured stages", "No recorded measurements: " + ", ".join(missing) + ". No causal steering outcome is inferred for these stages.", None))  # Local: state what was not measured; global: avoid turning a diagnostic result into an intervention result.
    sections.append(("Reproduction and provenance", _provenance_text(manifest), None))  # Local: compact the human-readable metadata; global: retain the full manifest unchanged on disk.
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
    references = [("manifest", "manifest.json"), ("results", "results.json"), ("extraction diagnostics", "vector_diagnostics.json")]  # Local: enumerate canonical evidence artifacts; global: keep links stable across copied bundles.
    links = [f"[{label}]({filename})" for label, filename in references if (path.parent / filename).is_file()]  # Local: link only existing files; global: keep pre-extraction reports free of broken references.
    lines.extend(["Full artifacts: " + ", ".join(links) + ".", ""])  # Local: expose complete metadata with relative links; global: make copied reports inspectable.
    if images:  # Local: add a figure heading only when a curve was measured; global: avoid empty evidence sections.
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
        if heading == "Reproduction and provenance":  # Local: begin technical metadata on a fresh page; global: keep the scientific narrative visually distinct.
            story.append(PageBreak())  # Local: let Platypus create a clean boundary; global: avoid a provenance heading stranded at the foot of a page.
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

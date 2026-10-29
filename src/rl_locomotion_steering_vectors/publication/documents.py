"""Render current Ant prose from evidence, while preserving historical numerical artifacts.

Local: small explicit templates and table formatters bind claims to saved JSON fields.
Global: README, paper, findings and model card share one source for every empirical value.
No model, simulator or experimental collection is imported by this module.
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import re
import shutil

GIT = "https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/"
SITE = "https://BurnyCoder.github.io/reinforcement-learning-locomotion-steering-vectors/"


def interpolate(template: str, context: dict) -> str:
    """Replace explicit named fields once; an unknown field is a publication error."""
    return re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", lambda match: str(context[match[1]]), template)


def number(value: float, digits: int = 3, signed: bool = False) -> str:
    """Use one rounding convention across surfaces without modifying source precision."""
    return format(value, f"{'+' if signed else ''}.{digits}f")


def interval(values: list[float], digits: int = 3) -> str:
    """Render endpoints exactly as stored, with their method specified in surrounding text."""
    return "[" + ", ".join(number(value, digits) for value in values) + "]"


def markdown_table(headers: list[str], rows: list[list]) -> str:
    """Escape cell delimiters so signs and scientific notation survive Markdown/PDF rendering."""
    clean = lambda row: "| " + " | ".join(str(cell).replace("|", r"\|") for cell in row) + " |"
    return "\n".join([clean(headers), clean(["---"] * len(headers)), *(clean(row) for row in rows)])


def measurement_table(records: list[dict], *, include_baseline: bool = False) -> str:
    """Publish every original confirmed arm, including unsuccessful controls and denominators."""
    rows = []
    if include_baseline:
        if not records:
            raise ValueError("A confirmation baseline requires saved confirmation records")
        baseline, n = records[0]["effect"]["baseline_means"], records[0]["effect"]["n_pairs"]
        if any(record["effect"]["baseline_means"] != baseline or record["effect"]["n_pairs"] != n for record in records):
            raise ValueError("Confirmation records must have one shared baseline")
        # Zero is the baseline's self-comparison, not an estimated interval or a successful usefulness gate.
        rows.append(["unsteered baseline", number(0, 3, True), "reference",
                     "100.00%" if baseline["speed"] else "undefined",
                     "+0.00%" if baseline["effort"] else "undefined",
                     f"{round(baseline['failure'] * n)}/{n} (reference)", "reference"])
    for record in records:
        effect = record["effect"]
        baseline, treated, n = effect["baseline_means"], effect["treated_means"], effect["n_pairs"]
        retention = number(100 * treated["speed"] / baseline["speed"], 2) + "%" if baseline["speed"] else "undefined"
        effort = number(100 * (treated["effort"] / baseline["effort"] - 1), 2, True) + "%" if baseline["effort"] else "undefined"
        rows.append([f"{record['vector']} ({record['alpha']:+g})", number(effect["effect"], 3, True),
                     interval(effect["ci95"]), retention, effort,
                     f"{round(baseline['failure'] * n)}/{n} → {round(treated['failure'] * n)}/{n}",
                     "pass" if effect["gate"] else "fail"])
    return markdown_table(["Condition (strength)", "Δ world-Y m/s", "95% interval", "Forward speed retained",
                           "Squared-action change", "Failures baseline → treated", "Full gate"], rows)


def audit_measurement_table(records: list[dict], *, negative_only: bool = False) -> str:
    """Render named audit arms; the README filters signs, never measured effects or failures."""
    reference = next(item["metrics"] for item in records if item["kind"] == "baseline")
    rows = []
    for item in records:
        if item["kind"] in ("random", "shuffled") or (negative_only and item["kind"] != "baseline" and item["alpha"] >= 0):
            continue  # Complete random ensembles and both signs remain available in the evidence and explorer.
        metrics, comparison = item["metrics"], item["pairedEffect"]
        effort = number(100 * (metrics["effort"] / reference["effort"] - 1), 2, True) + "%" if reference["effort"] else "undefined"
        retention = number(metrics["forward_speed_retention"] * 100, 2) + "%" if reference["x_velocity"] else "undefined"
        rows.append([item["label"], number(comparison["mean"], 3, True),
                     interval(comparison["ci95"]) if comparison["ci95"] else "reference",
                     retention, effort, f"{metrics['failures']}/{item['n_pairs']}"])
    return markdown_table(["Audit arm", "Δ world-Y m/s", "95% interval", "Forward speed retained", "Squared-action change", "Failures"], rows)


def source_context(root: Path, evidence: dict) -> dict:
    """Bind named prose fields to the original results and later audit, with explicit cohorts."""
    read = lambda path: json.loads((root / path).read_text(encoding="utf-8"))
    result = read("reports/ant-classic-005/results.json")
    manifest = read("reports/ant-classic-005/manifest.json")
    diagnostic = read("reports/ant-classic-005/vector_diagnostics.json")
    audit = read("reports/ant-classic-005-independent-audit/audit.json")
    selected = next(row for row in result["confirmation"] if row["vector"] == "lateral")
    effect, fresh = selected["effect"], audit["fresh"]
    baseline, treated = effect["baseline_means"], effect["treated_means"]
    p1, m2 = fresh["P1"], fresh["M2"]["minus_delta_arm"]
    cfg, provenance = manifest["config"], manifest["provenance"]
    context = dict(
        git=GIT, site=SITE, status=result["status"], alpha=f"{selected['alpha']:g}",
        n=effect["n_pairs"], effect=number(effect["effect"], 3, True), ci=interval(effect["ci95"]),
        retention=number(100 * treated["speed"] / baseline["speed"], 2),
        effort_change=number(100 * (treated["effort"] / baseline["effort"] - 1), 2, True),
        baseline_y=number(baseline["lateral"]), treated_y=number(treated["lateral"]),
        baseline_speed=number(baseline["speed"]), treated_speed=number(treated["speed"]),
        baseline_reward=number(baseline["reward"]), treated_reward=number(treated["reward"]),
        baseline_failures=round(baseline["failure"] * effect["n_pairs"]),
        treated_failures=round(treated["failure"] * effect["n_pairs"]),
        original_table=measurement_table(result["confirmation"]),
        readme_confirmation_table=measurement_table(result["confirmation"], include_baseline=True),
        n_validation=len(result["validation"]), n_confirmation=len(result["confirmation"]),
        fresh_n=p1["n"], fresh_effect=number(p1["mean"], 3, True), fresh_ci=interval(p1["bca_ci"]),
        fresh_positive=p1["positive_pairs"], fresh_baseline_failures=p1["failures"][0],
        fresh_failures=p1["failures"][1], fresh_retention=number(100 * p1["speed_retention"], 2),
        fresh_p=format(p1["sign_flip_p"], ".8g"), fresh_resamples=p1["resamples"],
        random_n=sum(key.startswith("random") for key in fresh["directions"]),
        shuffle_n=len(fresh["S1"]["shuffled_abs_odd"]), random_p=number(fresh["P2"]["rank_p"], 2),
        shuffle_p=number(fresh["S1"]["shuffled_rank_p"], 2),
        audit_episodes=audit["collection"]["episodes"], audit_conditions=audit["collection"]["conditions"],
        perception_angle=number(audit["directions"]["delta_eq_deg"], 2),
        perception_effect=number(m2["mean"], 3, True), perception_ci=interval(m2["bca_ci"]),
        perception_positive=m2["positive_pairs"], observation_dim=provenance["observation_shape"][0],
        hidden_dim=provenance["hidden_dimension"], action_dim=provenance["action_shape"][0],
        checkpoint_revision=provenance["revision"], checkpoint_sha=provenance["sha256"],
        warmup=cfg["warmup"], max_steps=cfg["max_steps"], fit_episodes=cfg["fit_episodes"],
        diagnostic_episodes=cfg["diagnostic_episodes"], window=diagnostic["windows"]["window"],
        eligible_windows=diagnostic["windows"]["eligible_windows"], total_windows=diagnostic["windows"]["total_windows"],
        norm=number(audit["directions"]["activation_norm"], 5),
        offset_norm=number(abs(selected["alpha"]) * audit["directions"]["activation_norm"], 5),
        split_half=number(diagnostic["behaviors"]["lateral"]["split_half_cosine"], 3),
        validation_seed=min(manifest["seed_splits"]["validation"]),
        audit_seed_first=fresh["seeds"][0], audit_seed_last=fresh["seeds"][-1],
        replication_first=min(manifest["seed_splits"]["replication"]), replication_last=max(manifest["seed_splits"]["replication"]),
        source_revision=read("reports/ant-publication/inputs.json")["revision"],
        clipping_table="", fresh_table="",
        rank_qualification=("These ranks describe comparisons with the specified control ensembles. Treating them as calibrated p-values requires exchangeability under an explicit null; norm matching alone does not make a label-derived, selected primary exchangeable with random directions. See [statistical assumptions](" + GIT + "docs/methodology.md#statistical-assumptions-and-rank-interpretation)."),
    )
    context["cohort_table"] = markdown_table(["Cohort", "Seeds", "Role"], [
        [name, f"{min(seeds)}–{max(seeds)}", "unused reservation" if name == "replication" else "observed"]
        for name, seeds in manifest["seed_splits"].items()] + [
        ["independent audit", f"{fresh['seeds'][0]}–{fresh['seeds'][-1]}", "observed; includes former application reservation"]])
    context["robustness_table"] = markdown_table(["Saved-data estimand", "Pairs", "Mean", "BCa 95% interval"], [
        [name.replace("_", " "), value["n"], number(value["mean"], 3, True), interval(value["bca_ci"])]
        for name, value in audit["saved"]["robustness"].items() if isinstance(value, dict) and "bca_ci" in value])
    named = [key for key in fresh["directions"] if not key.startswith(("random", "shuffled"))]
    context["odd_table"] = markdown_table(["Direction", "Odd world-Y m/s", "Even world-Y m/s", "Failures − / +"], [
        [key, number(fresh["directions"][key]["odd"], 3, True), number(fresh["directions"][key]["even"], 3, True),
         " / ".join(str(x) for x in fresh["directions"][key]["failures"])] for key in named])
    context["hypothesis_table"] = markdown_table(["Test", "Saved statistic", "Interpretation"], [
        ["P1: locked arm", f"{context['fresh_effect']} {context['fresh_ci']} m/s", "effect reproduced; physical gate fails"],
        ["P2: isotropic comparison", f"comparison rank = {context['random_p']}", "descriptive ensemble ranking"],
        ["S1: label specificity", f"shuffle rank = {context['shuffle_p']}", "not supported; turning is stronger"],
        ["S2: generic drift", f"{number(fresh['S2']['random_even']['mean'], 3, True)} {interval(fresh['S2']['random_even']['ci'])} m/s", "interval includes zero"],
        ["M1: component ablation", f"orthogonal norm-matched comparison rank = {number(fresh['M1']['orthogonal_renorm_rank_p'], 2)}", "original falsification criterion not met"],
        ["M2: observation transform", f"{context['perception_effect']} {context['perception_ci']} m/s at negative rotation", "specified transform produces a shift; not exact physical-heading equivalence"],
    ])
    context["kinematics_table"] = markdown_table(["Arm", "Pairs", "Δ world-Y m/s", "Δ travel angle degrees", "Δ torso yaw degrees"], [
        [name, value["pairs"], number(value["delta_y_velocity"]["mean"], 3, True),
         number(value["delta_travel_angle_deg"]["mean"], 2, True), number(value["delta_mean_yaw_deg"]["mean"], 2, True)]
        for name, value in [("original confirmation", audit["saved"]["kinematics_confirmation"]), *fresh["K1"].items()]])
    context["software_table"] = markdown_table(["Measured dependency", "Version"], list(manifest["software"].items()))
    cohort = next(item for item in evidence["cohorts"] if item["id"] == "fresh_audit")
    context["fresh_table"] = audit_measurement_table(cohort["conditions"])
    context["readme_audit_table"] = audit_measurement_table(cohort["conditions"], negative_only=True)
    # Wrench clipping is distinct from motor-action saturation; conflating them would miss the mechanism caveat.
    clipping_rows = []
    for label, key in (("fitting", "contact_wrench_clipping"), ("audit baseline", "fresh_baseline_contact_wrench_clipping")):
        value = evidence["fitting"][key]  # Required evidence: a missing limitation must stop publication.
        clipping_rows.append([label, value["states_saturated"], value["states"], number(value["fraction"] * 100, 2) + "%"])
    context["clipping_table"] = markdown_table(["Saved cohort", "States with clipped planar wrench component", "Post-onset states", "Pooled fraction"], clipping_rows)
    context["evidence_json"] = "[generated evidence](" + GIT + "reports/ant-publication/evidence.json)"
    return context


def replace_section(text: str, start: str, end: str, content: str) -> str:
    """Replace one known current section, failing if headings drift instead of duplicating it."""
    left, right = text.index(start), text.index(end, text.index(start) + len(start))
    return text[:left] + content.rstrip() + "\n\n" + text[right:]


def replace_readme_results(text: str, content: str) -> str:
    """Retain explicit markers and all surrounding prose, rejecting an ambiguous boundary."""
    start, end = "<!-- BEGIN GENERATED README RESULTS -->", "<!-- END GENERATED README RESULTS -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError("README results markers must occur exactly once and in order")
    left, right = text.index(start) + len(start), text.index(end)
    return text[:left] + "\n\n" + content.strip() + "\n\n" + text[right:]


def render_documents(root: Path, evidence: dict, output_root: Path | None = None) -> list[Path]:
    """Render generated sections into the requested tree; --check uses a temporary output tree."""
    destination = output_root or root
    context = source_context(root, evidence)
    from .halfcheetah import context as halfcheetah_context
    context.update(halfcheetah_context(root))
    template_root = root / "docs/publication"
    template = lambda name: interpolate((template_root / name).read_text(encoding="utf-8"), context)
    read = lambda path: (root / path).read_text(encoding="utf-8")
    outputs = {}
    outputs["README.md"] = replace_readme_results(read("README.md"), template("readme-results.md.in"))
    outputs["docs/huggingface-model-card.md"] = replace_section(read("docs/huggingface-model-card.md"), "## Featured result:", "## Watch the same paired episode", template("featured.md.in").replace("## Featured experiment:", "## Featured result:", 1))
    outputs["docs/huggingface-model-card.md"] = replace_section(outputs["docs/huggingface-model-card.md"], "## Download and apply the vector", "## Research record", template("model-use.md.in"))
    outputs["docs/huggingface-model-card.md"] = re.sub(r"^\| `ant-classic-005`.*$", f"| `ant-classic-005` | World-Y effect on {context['n']} confirmation pairs and {context['fresh_n']} audit resets; original `confirmation_failed`; no formal replication |", outputs["docs/huggingface-model-card.md"], flags=re.MULTILINE)
    outputs["reports/ant-classic-005-findings.md"] = template("findings.md.in")
    outputs["reports/ant-classic-005-independent-audit.md"] = template("independent.md.in")
    outputs["reports/ant-publication-completion.md"] = template("completion.md.in")
    paper = read("reports/paper.md").replace("Author: Libor Burian", "Author: Libor “Burny” Burian")
    paper = replace_section(paper, "## Abstract", "## 1. Research question", template("abstract.md.in"))
    paper = replace_section(paper, "## 4. HalfCheetah:", "## 5. Ant:", template("paper-halfcheetah.md.in"))
    paper = replace_section(paper, "## 5. Ant:", "## 9. Reproducibility", template("paper-ant.md.in"))
    # Match the historical and generated wording so every build restores evidence-derived reference counts.
    paper = re.sub(r"Integrity rebuild, [\d,]+ (?:fresh-seed|audit) episodes, a rank test against [\d,]+ random directions", f"Integrity rebuild, {context['audit_episodes']:,} audit episodes, a rank test against {context['random_n']} random directions", paper)
    marker = "\n## Appendix: Ant publication and interpretation corrections"
    if marker in paper:
        paper = paper[:paper.index(marker)]
    paper += "\n" + template("paper-appendix.md.in")
    outputs["reports/paper.md"] = paper
    register = read("reports/experiments.md")
    register = replace_section(register, "## Current evidence", "The first experiment,", template("register.md.in"))
    register = re.sub(r"^\| \[ant-classic-005\].*$", template("register-row.md.in").strip(), register, flags=re.MULTILINE)
    register = re.sub(r"^\| Farama-Minari Ant-v5 SAC medium.*$", template("policy-row.md.in").strip(), register, flags=re.MULTILINE)
    outputs["reports/experiments.md"] = register
    paths = []
    for relative, content in outputs.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")
        paths.append(target)
    # Reuse the existing renderer; its invariant mode fixes PDF IDs/timestamps for --check.
    spec = importlib.util.spec_from_file_location("ant_paper_renderer", root / "reports/build_paper.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    for relative in ("reports/paper.md", "reports/ant-classic-005-independent-audit.md"):
        source = destination / relative
        # Historical non-Ant figures remain unchanged when rendering into a temporary check tree.
        for image in re.findall(r"^!\[[^\]]*\]\(([^)]+)\)$", outputs[relative], re.MULTILINE):
            target = source.parent / image
            original = (root / relative).parent / image
            if not target.exists() and original.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
        paths.append(renderer.build(source))
    return paths

"""Local: test evidence-to-prose boundaries. Global: prevent hand-maintained result drift."""
import copy
import json
import re
import shutil
from pathlib import Path

import pytest

from rl_locomotion_steering_vectors.publication.documents import (
    audit_measurement_table, interpolate, measurement_table, render_documents,
    replace_readme_results, source_context,
)
from scripts.build_ant_publication import compare_outputs


def test_interpolation_requires_explicit_evidence():
    """Missing empirical values must stop publication rather than become zero or blank."""
    assert interpolate("Effect {{effect}}", {"effect": "+0.402"}) == "Effect +0.402"
    with pytest.raises(KeyError, match="missing"):
        interpolate("Effect {{missing}}", {})


def test_interpolation_is_single_pass():
    """Evidence strings are data, not recursive template instructions."""
    assert interpolate("{{text}}", {"text": "{{untrusted}}"}) == "{{untrusted}}"


def test_measurement_table_keeps_failures_and_undefined_retention():
    """A zero baseline must not invent retention, and failed arms must remain visible."""
    rows = [{"vector": "lateral", "alpha": -.1, "effect": {
        "effect": .4, "ci95": [.3, .5], "n_pairs": 10, "gate": False,
        "baseline_means": {"speed": 0., "effort": .2, "reward": 4., "failure": .1},
        "treated_means": {"speed": 1., "effort": .3, "reward": 3., "failure": .2}}}]
    output = measurement_table(rows)
    assert "undefined" in output
    assert "1/10" in output and "2/10" in output
    assert "fail" in output and "+50.00%" in output


def test_confirmation_baseline_is_a_reference_and_keeps_all_controls():
    """A shared baseline has measured failures but no fabricated comparison interval or gate."""
    effect = {
        "effect": .4, "ci95": [.3, .5], "n_pairs": 10, "gate": False,
        "baseline_means": {"speed": 0., "effort": 0., "failure": .1},
        "treated_means": {"speed": 1., "effort": .3, "failure": .2},
    }
    records = [{"vector": name, "alpha": -.1, "effect": copy.deepcopy(effect)}
               for name in ("lateral", "lateral_random0", "lateral_shuffled", "lateral_bias")]
    output = measurement_table(records, include_baseline=True)
    baseline = output.splitlines()[2]
    assert baseline == "| unsteered baseline | +0.000 | reference | undefined | undefined | 1/10 (reference) | reference |"
    assert len(output.splitlines()) == len(records) + 3
    assert all(name in output for name in ("lateral_random0", "lateral_shuffled", "lateral_bias"))
    assert output.count("| fail |") == len(records)
    records[-1]["effect"]["n_pairs"] = 9
    with pytest.raises(ValueError, match="shared baseline"):
        measurement_table(records, include_baseline=True)
    with pytest.raises(ValueError, match="confirmation"):
        measurement_table([], include_baseline=True)


def test_audit_table_selects_by_sign_and_family_not_outcome():
    """The compact view keeps every named negative arm, including failures and null effects."""
    baseline = {"kind": "baseline", "vector": "baseline", "label": "baseline", "alpha": 0.,
                "n_pairs": 3, "metrics": {"effort": 0., "x_velocity": 0.,
                "forward_speed_retention": 1., "failures": 1},
                "pairedEffect": {"mean": 0., "ci95": None}}
    records = [baseline]
    for vector, kind in (("lateral", "primary"), ("turning", "other_label"),
                         ("heading_parallel", "ablation"), ("heading_orthogonal", "ablation"),
                         ("heading_orthogonal_renorm", "ablation"), ("heading_axis_renorm", "ablation"),
                         ("perceived_heading", "perception"), ("random0", "random"), ("shuffled0", "shuffled")):
        for alpha in (-.1, .1):
            row = copy.deepcopy(baseline)
            row.update(kind=kind, vector=vector, label=f"{vector} {alpha:+g}", alpha=alpha)
            row["pairedEffect"] = {"mean": alpha, "ci95": [-.3, .3]}
            row["metrics"]["failures"] = 3
            records.append(row)
    output = audit_measurement_table(records, negative_only=True)
    assert output.splitlines()[2] == "| baseline | +0.000 | reference | undefined | undefined | 1/3 |"
    expected = [row for row in records if row["kind"] not in ("random", "shuffled") and row["alpha"] < 0]
    assert len(output.splitlines()) == len(expected) + 3
    assert all(row["label"] in output for row in expected)
    assert "+0.1" not in output and "random0" not in output and "shuffled0" not in output
    assert output.count("| 3/3 |") == len(expected)
    assert "lateral +0.1" in audit_measurement_table(records)


def test_readme_markers_preserve_surrounding_content():
    """Editable headings and media must survive an evidence rebuild unchanged."""
    start, end = "<!-- BEGIN GENERATED README RESULTS -->", "<!-- END GENERATED README RESULTS -->"
    prefix, suffix = "# Custom overview\n\n![Media](figure.png)\n\n", "\n\n## Custom setup\nInstructions\n"
    source = prefix + start + "\n\nold results\n\n" + end + suffix
    output = replace_readme_results(source, "## Renamed result heading\n\nnew results\n")
    assert output == prefix + start + "\n\n## Renamed result heading\n\nnew results\n\n" + end + suffix
    assert replace_readme_results(output, "## Renamed result heading\n\nnew results\n") == output


@pytest.mark.parametrize("layout", ["missing", "start-only", "end-only", "duplicate-start", "duplicate-end", "reversed"])
def test_readme_markers_reject_ambiguous_boundaries(layout):
    """A malformed marker contract must stop rendering rather than delete unrelated prose."""
    start, end = "<!-- BEGIN GENERATED README RESULTS -->", "<!-- END GENERATED README RESULTS -->"
    cases = {"missing": "manual text", "start-only": start, "end-only": end,
             "duplicate-start": start + start + end, "duplicate-end": start + end + end,
             "reversed": end + start}
    with pytest.raises(ValueError, match="README results markers"):
        replace_readme_results(cases[layout], "replacement")


@pytest.fixture
def publication_sources(tmp_path):
    """Copy compact real evidence/templates into a disposable tree, never touching original inputs."""
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path / "sources"
    compact = [
        "README.md", "docs/huggingface-model-card.md", "reports/paper.md",
        "reports/experiments.md", "reports/build_paper.py", "reports/ant-classic-005-independent-audit.md",
        "reports/ant-classic-005/results.json", "reports/ant-classic-005/manifest.json",
        "reports/ant-classic-005/vector_diagnostics.json",
        "reports/ant-classic-005-independent-audit/audit.json",
        "reports/ant-publication/inputs.json",
        "reports/hc-classic-001/results.json", "reports/hc-running-002/results.json",
        "reports/hc-height-speed-003/results.json", "reports/hc-height-speed-004/results.json",
        "reports/hc-running-002/vector_diagnostics.json",
    ]
    for relative in compact:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    shutil.copytree(repository / "docs/publication", root / "docs/publication")
    for relative in ("reports/paper_assets", "reports/ant-publication/figures", "reports/ant-classic-005-independent-audit"):
        for source in (repository / relative).glob("*.png"):
            target = root / relative / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    evidence = json.loads((repository / "reports/ant-publication/evidence.json").read_text(encoding="utf-8"))
    # Structural fixture values keep this mechanics test independent of a pending final publication build.
    for name in ("contact_wrench_clipping", "fresh_baseline_contact_wrench_clipping"):
        evidence["fitting"].setdefault(name, {"states": 4, "states_saturated": 2, "fraction": .5})
    return root, evidence


def test_readme_tables_preserve_saved_cohorts_and_intervals(publication_sources):
    """Original and later-audit reductions stay separate, with every displayed interval unchanged."""
    root, evidence = publication_sources
    context = source_context(root, evidence)
    assert context["readme_confirmation_table"].splitlines()[3:] == context["original_table"].splitlines()[2:]
    assert f"{context['baseline_failures']}/{context['n']} (reference)" in context["readme_confirmation_table"]
    cohort = next(item for item in evidence["cohorts"] if item["id"] == "fresh_audit")
    rows = [item for item in cohort["conditions"] if item["kind"] not in ("random", "shuffled")
            and (item["kind"] == "baseline" or item["alpha"] < 0)]
    table = context["readme_audit_table"]
    assert len(table.splitlines()) == len(rows) + 2
    for row in rows:
        assert f"| {row['label']} |" in table
        assert f"{row['metrics']['failures']}/{row['n_pairs']}" in table
        if row["pairedEffect"]["ci95"]:
            endpoints = ", ".join(f"{value:.3f}" for value in row["pairedEffect"]["ci95"])
            assert f"[{endpoints}]" in table


def test_real_templates_render_complete_pdfs_and_are_idempotent(publication_sources, tmp_path):
    """Exercise the full renderer with real compact inputs, then the same check comparison used by CLI."""
    root, evidence = publication_sources
    first, second = tmp_path / "first", tmp_path / "second"
    original_readme = (root / "README.md").read_bytes()
    context = source_context(root, evidence)
    outputs = render_documents(root, evidence, output_root=first)
    assert (root / "README.md").read_bytes() == original_readme
    paths = {path.relative_to(first).as_posix() for path in outputs}
    assert {"README.md", "docs/huggingface-model-card.md", "reports/paper.md", "reports/paper.pdf",
            "reports/ant-classic-005-independent-audit.md", "reports/ant-classic-005-independent-audit.pdf"} <= paths
    for path in outputs:
        if path.suffix == ".pdf":
            data = path.read_bytes()
            assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
        else:
            text = path.read_text(encoding="utf-8")
            assert "{{" not in text and "}}" not in text
    readme = (first / "README.md").read_text(encoding="utf-8")
    assert context["effect"] in readme
    assert context["readme_confirmation_table"] in readme and context["readme_audit_table"] in readme
    assert "confirmation_failed" in readme and "world-Y" in readme
    assert "clip(R clip(f))" in readme and "clip(R f)" in readme
    assert "application" in readme and "body-relative" in readme
    before, after = "<!-- BEGIN GENERATED README RESULTS -->", "<!-- END GENERATED README RESULTS -->"
    original_text = original_readme.decode("utf-8")
    assert readme.split(before)[0] == original_text.split(before)[0]
    assert readme.split(after)[1] == original_text.split(after)[1]
    for relative in ("docs/huggingface-model-card.md", "reports/paper.md", "reports/ant-classic-005-independent-audit.md"):
        assert (first / relative).read_text(encoding="utf-8") == (root / relative).read_text(encoding="utf-8")
    assert context["fresh_effect"] in (first / "reports/ant-classic-005-independent-audit.md").read_text(encoding="utf-8")
    # The next invocation reads already-generated prose, catching duplicate sections and unstable PDF IDs.
    for output in outputs:
        target = root / output.relative_to(first)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, target)
    repeated = render_documents(root, evidence, output_root=second)
    assert compare_outputs(first, second, repeated) == []
    (first / "README.md").write_text("stale", encoding="utf-8")
    assert "README.md" in compare_outputs(first, second, repeated)


@pytest.mark.parametrize("template_name", ["featured.md.in", "readme-results.md.in"])
def test_document_generation_rejects_missing_required_data_and_template(publication_sources, tmp_path, template_name):
    """Neither absent clipping evidence nor an absent template may yield a plausible partial publication."""
    root, evidence = publication_sources
    missing = copy.deepcopy(evidence)
    del missing["fitting"]["contact_wrench_clipping"]
    with pytest.raises(KeyError, match="contact_wrench_clipping"):
        source_context(root, missing)
    (root / "docs/publication" / template_name).unlink()
    with pytest.raises(FileNotFoundError):
        render_documents(root, evidence, output_root=tmp_path / "output")


@pytest.mark.parametrize("episode_label", ["fresh-seed", "audit"])
def test_reference_counts_regenerate_after_initial_publication(publication_sources, tmp_path, episode_label):
    """Restore corrupted reference counts from evidence for historical and already-generated prose."""
    root, evidence = publication_sources
    source = root / "reports/paper.md"
    stale = f"Integrity rebuild, 1 {episode_label} episodes, a rank test against 2 random directions"
    text, replacements = re.subn(
        r"Integrity rebuild, [\d,]+ (?:fresh-seed|audit) episodes, a rank test against [\d,]+ random directions",
        stale, source.read_text(encoding="utf-8"),
    )
    assert replacements == 1  # The reference fixture must exercise the actual published sentence.
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "regenerated"
    render_documents(root, evidence, output_root=output)
    context = source_context(root, evidence)
    expected = f"Integrity rebuild, {context['audit_episodes']:,} audit episodes, a rank test against {context['random_n']} random directions"
    rendered = (output / "reports/paper.md").read_text(encoding="utf-8")
    assert expected in rendered
    assert stale not in rendered

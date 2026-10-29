"""Local: test evidence-to-prose boundaries. Global: prevent hand-maintained result drift."""
import copy
import json
import re
import shutil
from pathlib import Path

import pytest

from rl_locomotion_steering_vectors.publication.documents import interpolate, measurement_table, render_documents, source_context
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


@pytest.fixture
def publication_sources(tmp_path):
    """Copy compact real evidence/templates into a disposable tree, never touching original inputs."""
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path / "sources"
    compact = [
        "README.md", "docs/huggingface-model-card.md", "reports/paper.md",
        "reports/experiments.md", "reports/build_paper.py",
        "reports/ant-classic-005/results.json", "reports/ant-classic-005/manifest.json",
        "reports/ant-classic-005/vector_diagnostics.json",
        "reports/ant-classic-005-independent-audit/audit.json",
        "reports/ant-publication/inputs.json",
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
    assert context["effect"] in (first / "README.md").read_text(encoding="utf-8")
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


def test_document_generation_rejects_missing_required_data_and_template(publication_sources, tmp_path):
    """Neither absent clipping evidence nor an absent template may yield a plausible partial publication."""
    root, evidence = publication_sources
    missing = copy.deepcopy(evidence)
    del missing["fitting"]["contact_wrench_clipping"]
    with pytest.raises(KeyError, match="contact_wrench_clipping"):
        source_context(root, missing)
    (root / "docs/publication/featured.md.in").unlink()
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

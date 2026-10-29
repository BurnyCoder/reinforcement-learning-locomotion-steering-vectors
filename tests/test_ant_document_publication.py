"""Publication link and allowlist boundaries, independent of Hub credentials."""
from pathlib import Path

import pytest

from scripts.publish_ant_documents import document_files, rebase_markdown


def test_rebase_links_images_and_fragments_without_touching_code():
    """HF layout changes must not break links, or rewrite documented example code."""
    checked = []
    source = "reports/ant-classic-005-findings.md"
    revision = "a" * 40
    text = ('[method](../docs/methodology.md#analysis)\n'
            '![plot](ant-publication/figures/reward.png)\n'
            '[anchor](#results) [external](https://example.org/a)\n'
            '`[example](relative.md)`\n```markdown\n[example](relative.md)\n```\n'
            '[`data`](ant-classic-005/results.json)\n')
    actual = rebase_markdown(text, source, revision, checked.append)
    assert f"github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/{revision}/docs/methodology.md#analysis" in actual
    assert f"raw.githubusercontent.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/{revision}/reports/ant-publication/figures/reward.png" in actual
    assert "[anchor](#results) [external](https://example.org/a)" in actual
    assert "`[example](relative.md)`" in actual and "```markdown\n[example](relative.md)\n```" in actual
    assert "reports/ant-classic-005/results.json" in checked and "relative.md" not in checked


def test_reference_links_and_space_paths_are_rebased():
    """Reference-style images use raw bytes; angle destinations preserve valid Markdown."""
    checked = []
    actual = rebase_markdown('![plot][figure]\n[figure]: <plots/one two.png> "Title"\n[ref]: ../docs/sources.md\n', "reports/paper.md", "b" * 40, checked.append)
    assert "raw.githubusercontent.com/" in actual and "one%20two.png> \"Title\"" in actual
    assert "/blob/" in actual and checked == ["reports/plots/one two.png", "docs/sources.md"]


def test_rebasing_rejects_repo_escape():
    """A local link may not publish a path outside the authorized repository."""
    with pytest.raises(ValueError, match="escapes"):
        rebase_markdown("[bad](../../secret)", "docs/sources.md", "c" * 40, lambda path: None)


def test_directory_links_use_the_git_tree_view():
    """A report's logs-directory link must not point at GitHub's file-only blob view."""
    result = rebase_markdown("[logs](logs/)", "reports/report.md", "d" * 40, lambda path: "tree")
    assert f"/tree/{'d' * 40}/reports/logs" in result


def test_only_final_operational_receipt_uses_main():
    """Avoid a circular publication receipt while retaining immutable scientific evidence links."""
    checked = []
    result = rebase_markdown("[receipt](ant-publication/publication-verification.json) [data](ant-publication/evidence.json)", "reports/ant-publication-completion.md", "e" * 40, checked.append)
    assert "/blob/main/reports/ant-publication/publication-verification.json" in result
    assert f"/blob/{'e' * 40}/reports/ant-publication/evidence.json" in result
    assert checked == ["reports/ant-publication/publication-verification.json", "reports/ant-publication/evidence.json"]


def test_allowlist_never_discovers_incidental_files(tmp_path):
    """An unrelated matching-looking report, credentials or model is never swept in."""
    for name in (".env", "policy.pth", "reports/ant-classic-005-unreviewed.md", "reports/paper_assets/unreviewed.png"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not for publication", encoding="utf-8")
    files = document_files(tmp_path)
    sources = {path.relative_to(tmp_path).as_posix() for path, _ in files}
    destinations = {remote for _, remote in files}
    assert ".env" not in sources and "policy.pth" not in sources
    assert "reports/ant-classic-005-unreviewed.md" not in sources
    assert "reports/paper_assets/unreviewed.png" not in sources
    assert (tmp_path / "docs/huggingface-model-card.md", "README.md") in files
    assert "publication/evidence.json" in destinations and len(destinations) == len(files)

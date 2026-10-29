"""Stage current Ant documents, then verify an explicitly uploaded immutable Hub revision.

This helper never uploads. Scientific Markdown links use an existing Git commit;
only the final operational receipt uses main. Source and publication hashes remain
distinct. Reuses the pinned-download helper:
https://huggingface.co/docs/huggingface_hub/guides/download
Markdown destination syntax: https://spec.commonmark.org/0.31.2/#links
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import quote, unquote, urlsplit

from rl_locomotion_steering_vectors.publication.inputs import download_input, local_path, sha256, write_json
from rl_locomotion_steering_vectors.storage import start_logging

ROOT = Path(__file__).resolve().parents[1]
REPO = "BurnyCoder/rl-locomotion-steering-vectors"
GITHUB_REPO = "BurnyCoder/reinforcement-learning-locomotion-steering-vectors"
STAGE = ROOT / "tmp/ant-document-stage"
OUTPUT = ROOT / "reports/ant-publication"
MANIFEST = OUTPUT / "document-upload-allowlist.json"
# This operational receipt acquires the document-upload revision after publication.
# Keeping only this link on main avoids a circular final-revision reference; all
# scientific data, reports, source files and images remain pinned to the source commit.
OPERATIONAL_RECEIPT = "reports/ant-publication/publication-verification.json"


def document_files(root: Path = ROOT) -> list[tuple[Path, str]]:
    """Use named sources only; matching-looking drafts and weights never enter staging."""
    files = [("docs/huggingface-model-card.md", "README.md")]
    files.extend((f"reports/{name}", f"paper/{name}") for name in ("paper.md", "paper.pdf", "build_paper.py", "paper_build.md"))
    reports = (
        "ant-classic-005-causal-audit.md", "ant-classic-005-diagnostics.md",
        "ant-classic-005-findings.md", "ant-classic-005-independent-audit-preregistration.md",
        "ant-classic-005-independent-audit.md", "ant-classic-005-independent-audit.pdf",
        "ant-classic-005-mechanism-audit.md", "ant-classic-005-preregistration.md",
        "ant-classic-005-y-shift-audit.md", "experiments.md", "ant-publication-completion.md",
        "fact-check-2026-10-22.md",
        "scientific-rigor-audit.md", "documentation-audit.md", "hc-classic-001-diagnostics.md",
        "hc-running-002-findings.md", "hc-running-002-validation-inspection.md",
        "hc-height-speed-003-findings.md", "hc-height-speed-004-findings.md",
    )
    files.extend((f"reports/{name}", f"documentation/experiments/{name}") for name in reports)
    files.extend((f"docs/{name}", f"documentation/{name}") for name in ("methodology.md", "sources.md", "ant-publication.md", "ant-videos.md", "pgs-adaptation.md", "fact-check-scope.json"))
    for name in ("inputs.json", "verification.json", "provenance.json", "claims.json", "report.md", "report.pdf", "checks.json", "media.json", "kinematics-follow-up.md", "kinematics-follow-up.json"):
        files.append((f"reports/fact-check-2026-10-29/{name}", f"documentation/fact-check-2026-10-29/{name}"))
    for run, plots in (("hc-classic-001",0),("hc-running-002",3),("hc-height-speed-003",1),("hc-height-speed-004",1),("ant-classic-005",2)):
        for name in ("report.md", "report.pdf", *(f"strength_response_{i}.png" for i in range(1,plots+1))):
            files.append((f"reports/{run}/{name}",f"experiments/{run}/{name}"))
    files.extend((f"reports/ant-publication/{name}.json", f"publication/{name}.json") for name in ("inputs", "evidence"))
    figures = ("confirmation", "ablations", "strength-response", "specificity", "reward", "trajectories", "architecture")
    files.extend((f"reports/ant-publication/figures/{name}.png", f"paper/ant-publication/figures/{name}.png") for name in figures)
    files.extend((f"reports/paper_assets/{name}.png", f"paper/paper_assets/{name}.png") for name in ("ant-lateral-validation", "ant-topdown-validation"))
    return [(root / source, remote) for source, remote in files]


def git_revision(root: Path, requested: str) -> str:
    """Resolve an actual local commit; a plausible-looking string is insufficient provenance."""
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", f"{requested}^{{commit}}"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    shown = subprocess.check_output(["git", "show", "--no-patch", "--format=%H", commit], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or shown != commit:
        raise ValueError("--git-revision must resolve to an existing Git commit")
    return commit


def rebase_markdown(text: str, source: str, revision: str, require_target) -> str:
    """Rebase actual links, preserving external URLs, fragments, code and optional titles."""
    image_references = set(re.findall(r"!\[[^\]\n]*\]\[([^\]\n]+)\]", text))
    inline_link = re.compile(r"(?P<open>!?\[(?:\\.|[^\]\\\n])*\]\()(?P<url><[^>\n]+>|[^\s)]+)(?P<close>(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\))")
    reference_link = re.compile(r"^(?P<open> {0,3}\[(?P<label>[^\]\n]+)\]:\s*)(?P<url><[^>\n]+>|\S+)(?P<close>.*)$")

    def destination(value: str, image: bool) -> str:
        wrapped = value.startswith("<") and value.endswith(">")
        value = value[1:-1] if wrapped else value
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return f"<{value}>" if wrapped else value
        decoded = unquote(parsed.path)
        relative = posixpath.normpath(posixpath.join(posixpath.dirname(source), decoded))
        if relative == ".." or relative.startswith("../") or relative.startswith("/") or "\\" in relative:
            raise ValueError(f"Markdown target escapes repository: {source}")
        target_type = require_target(relative)
        if image and target_type == "tree":
            raise ValueError(f"Markdown image points to a directory: {source}")
        kind = "tree" if target_type == "tree" else "blob"
        link_revision = "main" if relative == OPERATIONAL_RECEIPT and not image else revision
        prefix = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{revision}/" if image else f"https://github.com/{GITHUB_REPO}/{kind}/{link_revision}/"
        result = prefix + quote(relative, safe="/")
        if parsed.query:
            result += "?" + parsed.query
        if parsed.fragment:
            result += "#" + parsed.fragment
        return f"<{result}>" if wrapped else result

    output, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            output.append(line)
            continue
        if fence:
            output.append(line)
            continue
        code_ranges = [(match.start(), match.end()) for match in re.finditer(r"(`+).*?\1", line)]

        def replace_inline(match):
            if any(start <= match.start() < end for start, end in code_ranges):
                return match.group(0)
            return match["open"] + destination(match["url"], match["open"].startswith("!")) + match["close"]

        reference = reference_link.match(line)
        if reference:
            newline = "\n" if line.endswith("\n") else ""
            output.append(reference["open"] + destination(reference["url"], reference["label"] in image_references) + reference["close"] + newline)
        else:
            output.append(inline_link.sub(replace_inline, line))
    return "".join(output)


def stage(requested_revision: str) -> dict:
    """Write only the allowlisted current bytes, with auditable Markdown link transformations."""
    revision = git_revision(ROOT, requested_revision)
    files = document_files()
    stage_root = local_path(ROOT, STAGE.relative_to(ROOT).as_posix())
    allowed = {local_path(stage_root, remote) for _, remote in files}
    if stage_root.exists():
        for path in stage_root.rglob("*"):
            if path.is_symlink() or path.is_file() and path.resolve() not in allowed:
                raise ValueError("Staging contains an unlisted file; inspect tmp/ant-document-stage before proceeding")
    checked_targets = {}

    def require_target(relative: str) -> str:
        local_path(ROOT, relative)
        if relative not in checked_targets:
            result = subprocess.run(["git", "cat-file", "-t", f"{revision}:{relative}"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if result.returncode:
                raise ValueError(f"Link target is absent from Git revision {revision}: {relative}")
            checked_targets[relative] = result.stdout.strip()
        return checked_targets[relative]

    rows, payloads = [], []
    for path, remote in files:
        relative = path.relative_to(ROOT).as_posix()
        source = local_path(ROOT, relative)
        original = source.read_bytes()
        published = rebase_markdown(original.decode("utf-8").replace("\r\n", "\n"), relative, revision, require_target).encode("utf-8") if source.suffix == ".md" else original
        target = local_path(stage_root, remote)
        payloads.append((target, published))
        rows.append({"local_path": relative, "hub_path": remote,
                     "source_sha256": hashlib.sha256(original).hexdigest(), "source_size": len(original),
                     "publication_sha256": hashlib.sha256(published).hexdigest(), "publication_size": len(published),
                     "transformation": "relative Markdown URLs rebased to GitHub; final operational receipt on main; LF newlines" if source.suffix == ".md" else "exact bytes"})
    # Validate every source and link before altering an earlier staging directory.
    for (target, published), row in zip(payloads, rows):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(published)
        logging.info("Staged document %s -> %s", row["local_path"], row["hub_path"])
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "repo_id": REPO,
                "github_repo": GITHUB_REPO, "git_revision": revision, "stage_path": STAGE.relative_to(ROOT).as_posix(),
                "files": rows, "rebased_git_targets": sorted(checked_targets),
                "mutable_operational_links": [OPERATIONAL_RECEIPT] if OPERATIONAL_RECEIPT in checked_targets else []}
    write_json(MANIFEST, manifest)
    logging.info("Staged %d allowed documents/assets; no upload performed", len(rows))
    return manifest


def verify(revision: str) -> dict:
    """Hash-check the existing staged bytes against one public immutable Hub revision."""
    from huggingface_hub import HfApi

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--verify requires a full immutable Hub commit")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {(path.relative_to(ROOT).as_posix(), remote) for path, remote in document_files()}
    actual = {(row["local_path"], row["hub_path"]) for row in manifest["files"]}
    if manifest.get("repo_id") != REPO or actual != expected or len(manifest["files"]) != len(expected):
        raise ValueError("Saved document allowlist does not match the explicit source mapping")
    info = HfApi(token=False).model_info(REPO, revision=revision)
    if info.sha != revision or info.private:
        raise ValueError("Verification requires the named public immutable Hub revision")

    def check(row: dict) -> dict:
        try:
            staged = local_path(STAGE, row["hub_path"])
            if sha256(staged) != row["publication_sha256"] or staged.stat().st_size != row["publication_size"]:
                raise ValueError("Staged bytes changed after allowlisting")
            record = {"hub_path": row["hub_path"], "sha256": row["publication_sha256"], "size": row["publication_size"]}
            # Each remote path gets a short distinct cache, so duplicate file bytes cannot mask a missing remote path.
            cache_key = revision[:12] + "-" + hashlib.sha256(row["hub_path"].encode()).hexdigest()[:12]
            downloaded = download_input(ROOT, record, REPO, revision, cache_dir=ROOT / ".cache/ant-doc-verify" / cache_key)
            logging.info("Verified published document %s", row["hub_path"])
            return {"hub_path": row["hub_path"], "status": "ok", "sha256": sha256(downloaded)}
        except Exception as error:
            logging.error("Document verification failed %s (%s)", row["hub_path"], type(error).__name__)
            return {"hub_path": row["hub_path"], "status": "error", "error_type": type(error).__name__}

    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(check, manifest["files"]))
    receipt = {"checked_utc": datetime.now(timezone.utc).isoformat(), "repo_id": REPO, "revision": revision,
               "git_revision": manifest["git_revision"], "allowlist_sha256": sha256(MANIFEST),
               "files": len(checks), "all_hashes_match": all(row["status"] == "ok" for row in checks), "checks": checks}
    write_json(OUTPUT / "document-upload-verification.json", receipt)
    if not receipt["all_hashes_match"]:
        raise ValueError("Some published document hashes failed; see the verification receipt")
    return receipt


def main() -> int:
    """Expose staging and verification; credentials and uploads remain outside this helper."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage", action="store_true")
    mode.add_argument("--verify", metavar="HUBREV")
    parser.add_argument("--git-revision", help="existing Git commit used for Markdown links during staging")
    args = parser.parse_args()
    if args.stage and not args.git_revision:
        parser.error("--stage requires --git-revision")
    if args.verify and args.git_revision:
        parser.error("--git-revision applies only to --stage")
    start_logging(OUTPUT / "logs")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        stage(args.git_revision) if args.stage else verify(args.verify)
    except (ValueError, FileNotFoundError) as error:
        logging.error("Document publication check failed: %s", error)
        return 1
    except Exception as error:
        logging.error("Document publication check failed (%s)", type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

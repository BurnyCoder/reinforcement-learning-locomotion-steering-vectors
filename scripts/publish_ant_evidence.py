"""Stage and verify an explicit Ant evidence allowlist; never upload an entire workspace.

Local: map immutable arrays and audit evidence to public Hub paths and record SHA256.
Global: make publication reproducible without collecting episodes or redistributing policy weights.
Sources: https://huggingface.co/docs/huggingface_hub/guides/upload and
https://huggingface.co/docs/huggingface_hub/guides/download (revision-pinned retrieval).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import shutil

from huggingface_hub import HfApi
from rl_locomotion_steering_vectors.publication.inputs import download_input
from rl_locomotion_steering_vectors.storage import start_logging

ROOT = Path(__file__).resolve().parents[1]
REPO = "BurnyCoder/rl-locomotion-steering-vectors"
OUTPUT = ROOT / "reports/ant-publication"
STAGE = ROOT / "tmp/ant-evidence-stage"


def digest(path: Path) -> str:
    """Hash complete bytes; the file identity must not depend on its modification time."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evidence_files() -> list[tuple[Path, str]]:
    """Enumerate only simulation evidence and its audit provenance, excluding arbitrary files."""
    result = []
    for run in ("ant-classic-005", "ant-005-independent-audit"):
        base = ROOT / "runs" / run
        for path in sorted((base / "episodes").rglob("*")):
            if path.is_file() and path.suffix in (".npz", ".json"):
                result.append((path, f"experiments/{run}/{path.relative_to(base).as_posix()}"))
    original = ROOT / "runs/ant-classic-005"
    for name in ("manifest.json", "results.json", "selection.json", "vectors.npz", "action_biases.npz", "vector_diagnostics.json"):
        result.append((original / name, f"experiments/ant-classic-005/{name}"))
    for directory in sorted((ROOT / "reports").glob("ant-classic-005*")):
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in (".json", ".npz", ".log"):
                    result.append((path, "publication-inputs/" + path.relative_to(ROOT).as_posix()))
    for name in ("scientific-rigor-audit/rigor-audit.json", "documentation-audit/ant-sensitivity-verification.json"):
        path = ROOT / "reports" / name
        result.append((path, "publication-inputs/" + path.relative_to(ROOT).as_posix()))
    for path in sorted(original.glob("*.log")):
        result.append((path, f"experiments/ant-classic-005/logs/{path.name}"))
    return result


def stage() -> None:
    """Create a complete allowlist and a workspace-local upload directory of the listed files."""
    rows = []
    for path, remote in evidence_files():
        target = STAGE / remote
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(path)
        if not target.exists() or digest(target) != checksum:
            shutil.copyfile(path, target)
        rows.append(dict(local_path=path.relative_to(ROOT).as_posix(), hub_path=remote,
                         sha256=checksum, size=path.stat().st_size))
    # The manifest stays outside staging: a final immutable Hub revision is added after upload.
    manifest = dict(schema_version=1, repo_id=REPO, revision=None, files=rows)
    (OUTPUT / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logging.info("Staged %s explicit files (%s bytes)", len(rows), sum(row["size"] for row in rows))


def verify(revision: str) -> None:
    """Download every allowlisted input at one immutable revision and compare complete bytes."""
    manifest_path = OUTPUT / "inputs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = HfApi().model_info(REPO, revision=revision).sha

    def check(row: dict) -> dict:
        # A separate cache forces verification against Hub bytes, not the staging directory.
        downloaded = download_input(ROOT, row, REPO, resolved,
                                    cache_dir=ROOT / ".cache/ant-verification")
        actual = digest(downloaded)
        if actual != row["sha256"]:
            raise ValueError(f"Published byte mismatch: {row['hub_path']}")
        logging.info("Verified %s %s", row["hub_path"], actual)
        return dict(hub_path=row["hub_path"], sha256=actual)

    with ThreadPoolExecutor(max_workers=6) as pool:
        checks = list(pool.map(check, manifest["files"]))
    manifest["revision"] = resolved
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    receipt = dict(repo_id=REPO, revision=resolved, files=len(checks), all_hashes_match=True,
                   manifest_sha256=digest(manifest_path), checks=checks)
    (OUTPUT / "input-upload-verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    logging.info("Verified every input at immutable revision %s", resolved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--verify", metavar="REVISION")
    args = parser.parse_args()
    start_logging(OUTPUT / "logs")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.stage:
        stage()
    elif args.verify:
        verify(args.verify)
    else:
        parser.error("Select --stage or --verify REVISION")

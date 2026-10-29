"""Explicit project-only provenance inventory; excludes environments and credentials.

Git supplies the original publication inventory; pathlib and hashlib cover saved
experiment artifacts. https://git-scm.com/docs/git-ls-tree
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from .audit import digest

BASE_REVISION = "3fb62bc"


def snapshot(root: Path) -> dict:
    """Record original bytes once; future checks never refresh away a mismatch."""
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE_REVISION], cwd=root, text=True).splitlines()
    evidence = {p.relative_to(root).as_posix() for p in (root/"runs").rglob("*") if p.is_file() and p.suffix in {".npz", ".json", ".log"}}
    evidence |= {p for p in tracked if p.endswith(".npz") or p.endswith("/results.json") or p.endswith("/audit.json")}
    def record(relative):
        p = root/relative
        return {"path": relative, "size": p.stat().st_size, "sha256": digest(p)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        raw = list(pool.map(record, sorted(evidence)))
    return {"source_revision": subprocess.check_output(["git", "rev-parse", BASE_REVISION], cwd=root, text=True).strip(),
            "immutable_evidence": raw, "original_publications": tracked,
            "boundary": "Project runs and tracked research artifacts only; no environments, credentials or unrelated projects."}


def verify(root: Path, manifest: dict) -> dict:
    """Rehash all immutable inputs, including files not consumed by a particular numerical check."""
    def check(row):
        p = root/row["path"]
        return row["path"] if not p.is_file() or p.stat().st_size != row["size"] or digest(p) != row["sha256"] else None
    with ThreadPoolExecutor(max_workers=8) as pool:
        changed = [p for p in pool.map(check, manifest["immutable_evidence"]) if p]
    return {"files": len(manifest["immutable_evidence"]), "changed": changed}


def encode(value) -> bytes:
    """Deterministic readable JSON; nonfinite evidence is an error, never JavaScript NaN."""
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")

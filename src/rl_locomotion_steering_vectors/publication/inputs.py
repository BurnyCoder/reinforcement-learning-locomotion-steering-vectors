"""Pinned input restoration and exact hashes for the Ant publication build.

Uses the Hub's revision-pinned download API and standard-library SHA256:
https://huggingface.co/docs/huggingface_hub/guides/download
https://docs.python.org/3.12/library/hashlib.html#hashlib.file_digest
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def canonical_json(value: object) -> bytes:
    """Stable keys and finite JSON make stale-output comparisons meaningful."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    """Stream a selected file; never include incidental caches in scientific identities."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(root: Path, relative: str) -> Path:
    """Resolve only a project-relative file, including checking symlink destinations."""
    path = Path(relative)
    target = (root / path).resolve()
    if path.is_absolute() or not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError(f"Input path must remain relative and inside the project: {relative}")
    return target


def download_input(root: Path, record: dict, repo_id: str, revision: str, *, cache_dir: Path | None = None) -> Path:
    """Download public pinned bytes into a short Windows-safe content-addressed cache.

The Hub helper constructs the public URL; urllib follows signed redirects in memory.
Only public relative paths are logged. Hash verification precedes the atomic replace.
Sources: https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.hf_hub_url
https://docs.python.org/3.12/library/urllib.request.html#urllib.request.urlopen
"""
    from huggingface_hub import hf_hub_url

    if not re.fullmatch(r"[0-9a-f]{40}", revision or "") or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError("Downloads require an immutable revision and SHA256")
    cache = (cache_dir or root / ".cache/ant-inputs").resolve()
    if not cache.is_relative_to(root.resolve()):
        raise ValueError("Download cache must stay inside the project")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / record["sha256"]
    valid = lambda path: path.stat().st_size == record["size"] and sha256(path) == record["sha256"]
    if target.is_symlink():
        raise ValueError("Evidence cache entries must be ordinary files")
    if target.exists():
        if not target.is_file() or not valid(target):
            raise ValueError(f"Cached evidence hash/size mismatch: {record['hub_path']}")
        return target
    public_url = hf_hub_url(repo_id=repo_id, filename=record["hub_path"], revision=revision)
    for attempt in range(3):
        temporary = cache / (uuid.uuid4().hex + ".part")
        try:
            logging.info("Downloading pinned evidence %s at %s (attempt %d)", record["hub_path"], revision, attempt + 1)
            with urlopen(public_url, timeout=55) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if not valid(temporary):
                raise ValueError(f"Downloaded evidence hash/size mismatch: {record['hub_path']}")
            os.replace(temporary, target)
            return target
        except (HTTPError, URLError, HTTPException, TimeoutError, ConnectionError):
            if attempt == 2:
                raise RuntimeError(f"Unable to download public evidence after 3 attempts: {record['hub_path']}") from None
            time.sleep(attempt + 1)
        finally:
            if temporary.exists():
                temporary.unlink()  # A single owned partial file, within the already-validated cache.
    raise AssertionError("Unreachable download retry state")


def validate_inputs(root: Path, manifest_path: Path, *, restore: bool = False) -> dict:
    """Verify every registered input; restore only missing files, never replace mismatches."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Input manifest needs schema_version=1 and a nonempty files list")
    seen, verified = set(), []
    for record in manifest["files"]:
        relative = record["local_path"]
        path = local_path(root, relative)
        if path in seen:
            raise ValueError(f"Duplicate publication input: {relative}")
        seen.add(path)
        revision = record.get("revision", manifest.get("revision", ""))
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Input revision must be an immutable commit: {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError(f"Invalid SHA256 for {relative}")
        if not path.is_file():
            if not restore:
                raise FileNotFoundError(f"Missing registered input: {relative}; run the publication build to restore it")
            if revision is None:
                raise ValueError(f"Cannot restore evidence before its publication revision is pinned: {relative}")
            logging.info("Restoring pinned evidence %s at %s", record["hub_path"], revision)
            downloaded = download_input(root, record, manifest["repo_id"], revision)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(uuid.uuid4().hex + ".restore")
            try:
                shutil.copyfile(downloaded, temporary)
                os.replace(temporary, path)  # An interrupted copy must not leave a partial scientific input.
            finally:
                if temporary.exists():
                    temporary.unlink()
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Registered evidence hash/size mismatch: {relative}")
        verified.append({"local_path": relative, "sha256": record["sha256"], "size": record["size"]})
    logging.info("Verified %d exact input files", len(verified))
    return {"repo_id": manifest["repo_id"], "revision": manifest["revision"], "files_verified": len(verified), "manifest_sha256": sha256(manifest_path), "inputs_sha256": hashlib.sha256(canonical_json(verified)).hexdigest(), "files": verified}


def write_json(path: Path, value: object) -> Path:
    """Write one deterministic derived artifact; callers select a publication output root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))
    return path

"""Local: atomic JSON/NPZ and logging; global: preserve auditable resumable runs.

Sources: https://numpy.org/doc/1.26/reference/generated/numpy.load.html
https://docs.python.org/3.12/library/logging.html
https://docs.python.org/3.12/library/pathlib.html#pathlib.Path.replace
https://docs.python.org/3.12/library/subprocess.html#subprocess.run
https://git-scm.com/docs/git-rev-parse
"""
import hashlib  # Content hashes distinguish actual source from its nominal Git revision.
import json  # JSON stores readable protocol decisions without executable objects.
import logging  # Standard handlers share timestamped terminal and file records.
import subprocess  # Query local Git with argument arrays rather than shell interpolation.
import sys  # The terminal handler uses the current output stream.
from datetime import datetime, timezone  # UTC makes records comparable across machines.
from pathlib import Path  # Paths remain platform independent.

import numpy as np  # NPZ preserves simulator arrays with explicit dtypes.


def utc_now() -> str:
    """Local: produce an aware timestamp; global: establish event chronology."""
    return datetime.now(timezone.utc).isoformat()  # ISO 8601 includes the UTC offset.


def code_identity(root: Path) -> dict:
    """Local: identify owned source and lockfile; global: record reproducible code without reading credentials."""
    paths = sorted((root / "src" / "rl_locomotion_steering_vectors").rglob("*.py")) + [root / name for name in ("pyproject.toml", "uv.lock")]  # Explicit source/dependency paths exclude configuration secrets and experiment data.
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest() for path in paths if path.is_file()}  # Normalize checkout line endings while retaining every implementation edit.
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False)  # Read one public commit identifier without invoking a shell or displaying Git configuration.
        commit = revision.stdout.strip() if revision.returncode == 0 else None  # Source archives may legitimately lack Git metadata.
    except (OSError, subprocess.TimeoutExpired):  # Missing or stalled Git must not erase the available source fingerprints.
        commit = None  # Make unavailable provenance explicit rather than inventing a revision.
    return {"git_revision": commit, "source_sha256": hashes, "hash_convention": "UTF-8 text with universal newlines"}  # A commit is a reference; actual file hashes also describe local edits.


def save_json(path: Path, value: dict | list) -> None:
    """Local: atomically replace a record; global: interrupted writes stay resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the artifact's parent.
    temporary = path.with_suffix(path.suffix + ".tmp")  # Keep replacement on the same filesystem.
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Reject invalid scientific numbers.
    temporary.replace(path)  # Publish only the completed JSON document.


def load_json(path: Path) -> dict:
    """Local: parse a saved record; global: reuse the exact previous decisions."""
    return json.loads(path.read_text(encoding="utf-8"))  # JSON performs no code execution.


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Local: write typed arrays atomically; global: forbid pickled vector payloads."""
    clean = {key: np.asarray(value) for key, value in arrays.items()}  # Normalize scalar metadata too.
    if any(value.dtype.hasobject for value in clean.values()):  # Object dtypes require unsafe pickle.
        raise ValueError("Artifact arrays must not have object dtype")  # Fail before writing a partial artifact.
    path.parent.mkdir(parents=True, exist_ok=True)  # Keep phase directories self contained.
    temporary = path.with_suffix(".tmp")  # An open file prevents NumPy adding a second extension.
    with temporary.open("wb") as stream:  # Closing flushes all compressed members before replacement.
        np.savez_compressed(stream, **clean)  # Numeric and Unicode arrays require no pickle.
    temporary.replace(path)  # A completed episode becomes the unit of resumption.


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Local: copy NPZ members; global: close files and reject executable arrays."""
    with np.load(path, allow_pickle=False) as archive:  # Explicitly disable Python object loading.
        return {key: archive[key] for key in archive.files}  # Materialize before the archive closes.


def check_manifest(saved: dict, expected: dict) -> None:
    """Local: compare protocol keys; global: reject scientifically invalid resumption."""
    for key in ("model_key", "config", "seed_splits", "software"):  # Time and machine paths may legitimately differ.
        if key not in saved and key not in expected:  # Small pure-test manifests need no installed package inventory.
            continue  # Production manifests always include their measured versions.
        if saved[key] != expected[key]:  # Reusing episodes is valid only for the same experiment.
            raise ValueError(f"Run manifest {key} differs; use a new run directory")  # Preserve prior evidence.


def ensure_identity(path: Path, identity: dict) -> None:
    """Local: freeze cache identity; global: never combine different interventions under one name."""
    if path.exists():  # Completed and partial conditions use the same provenance contract.
        if load_json(path) != identity:  # Array contents and runtime settings must remain identical.
            raise ValueError(f"Condition identity changed: {path}; use a new run directory")  # Preserve all previous scientific evidence.
    else:
        if path.parent.exists() and any(path.parent.glob("*.npz")):  # Unidentified legacy arrays need an explicit external audit.
            raise ValueError(f"Episode cache has no identity: {path}")  # Do not silently associate old trials with current code.
        save_json(path, identity)  # Establish provenance before collecting the first episode.


def start_logging(run_dir: Path) -> None:
    """Local: attach realtime handlers; global: record progress without truncation."""
    run_dir.mkdir(parents=True, exist_ok=True)  # The log belongs beside its artifacts.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # Every invocation gets its own log.
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / f"{stamp}.log", encoding="utf-8")], force=True)  # Both handlers receive the same complete message.
    logging.Formatter.converter = __import__("time").gmtime  # Align formatter clock with the explicit UTC suffix.

"""Local: atomic JSON/NPZ and logging; global: preserve auditable resumable runs.

Sources: https://numpy.org/doc/1.26/reference/generated/numpy.load.html
https://docs.python.org/3.12/library/logging.html
https://docs.python.org/3.12/library/pathlib.html#pathlib.Path.replace
"""
import json  # JSON stores readable protocol decisions without executable objects.
import logging  # Standard handlers share timestamped terminal and file records.
import sys  # The terminal handler uses the current output stream.
from datetime import datetime, timezone  # UTC makes records comparable across machines.
from pathlib import Path  # Paths remain platform independent.

import numpy as np  # NPZ preserves simulator arrays with explicit dtypes.


def utc_now() -> str:
    """Local: produce an aware timestamp; global: establish event chronology."""
    return datetime.now(timezone.utc).isoformat()  # ISO 8601 includes the UTC offset.


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
    for key in ("model_key", "config", "seed_splits"):  # Time and machine paths may legitimately differ.
        if saved[key] != expected[key]:  # Reusing episodes is valid only for the same experiment.
            raise ValueError(f"Run manifest {key} differs; use a new run directory")  # Preserve prior evidence.


def start_logging(run_dir: Path) -> None:
    """Local: attach realtime handlers; global: record progress without truncation."""
    run_dir.mkdir(parents=True, exist_ok=True)  # The log belongs beside its artifacts.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # Every invocation gets its own log.
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / f"{stamp}.log", encoding="utf-8")], force=True)  # Both handlers receive the same complete message.
    logging.Formatter.converter = __import__("time").gmtime  # Align formatter clock with the explicit UTC suffix.

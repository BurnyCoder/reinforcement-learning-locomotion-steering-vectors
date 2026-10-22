"""Local: shared paths, protocol constants and lightweight episode readers. Global: one definition for every audit phase.

Sources: NumPy NpzFile lazy member access https://numpy.org/doc/1.26/reference/generated/numpy.load.html ;
hashlib file_digest https://docs.python.org/3.12/library/hashlib.html#hashlib.file_digest
"""
from __future__ import annotations  # Keep annotations lightweight.

import hashlib  # Content identities of inputs and vectors.
from pathlib import Path  # Repository-relative locations.

import numpy as np  # Episode arrays.

ROOT = Path(__file__).resolve().parents[2]  # Repository root (scripts/ant005_independent/common.py → root).
SOURCE_RUN = ROOT / "runs" / "ant-classic-005"  # Original experiment bundle with all raw episodes.
AUDIT_RUN = ROOT / "runs" / "ant-005-independent-audit"  # Fresh audit episodes (large, gitignored).
REPORT_DIR = ROOT / "reports" / "ant-classic-005-independent-audit"  # Committed machine-readable outputs and logs.
FRESH_SEEDS = tuple(range(540000, 540020))  # Fresh resets, disjoint from every recorded split; 540000–540009 is also Ant 005's convention-reserved application cohort (demonstration.fresh_application_seeds), which this audit consumed.
ALPHA, WARMUP, MAX_STEPS = 0.1, 100, 1000  # Matched strength magnitude, onset and horizon of the original protocol.
RANDOM_COUNT, RANDOM_SEED = 99, 540099  # 99 isotropic directions give rank-p resolution 0.01.
SHUFFLE_SEEDS = tuple(range(1, 20))  # 19 label-shuffle redraws (seed 0 is the original control).
PRIMARY, OTHER_LABEL = "lateral", "turning"  # Primary vector and the other-label contrast vector.
BOOTSTRAP_SEED = 20261016  # One recorded seed for every audit resampling procedure.


def condition_dir(run: Path, phase: str, condition: str) -> Path:
    """Local: locate one condition's episodes. Global: share the production cache layout."""
    return run / "episodes" / phase / condition  # Same layout as execution.collect_episodes.


def load_members(path: Path, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Local: read selected NPZ members only. Global: avoid decompressing observations when only metrics are needed."""
    with np.load(path, allow_pickle=False) as archive:  # No executable payloads.
        return {key: archive[key] for key in (keys or archive.files) if key in archive.files}  # Materialize before close.


def load_condition(run: Path, phase: str, condition: str, keys: tuple[str, ...] | None = None) -> dict[int, dict]:
    """Local: read every episode of one condition. Global: key episodes by reset seed for pairing."""
    episodes = {}  # seed → arrays.
    for path in sorted(condition_dir(run, phase, condition).glob("*.npz")):  # identity.json is not an episode.
        item = load_members(path, None if keys is None else (*keys, "seed", "length", "terminated", "onset_hash", "dt"))  # Always keep pairing metadata.
        episodes[int(item["seed"])] = item  # Seed is the independent unit.
    return episodes  # Possibly empty if the condition was not collected.


def post_onset_mean(episode: dict, key: str, warmup: int = WARMUP) -> float:
    """Local: mean of one per-step metric after onset. Global: the recorded estimand, with zero for prefix-only episodes."""
    length = int(episode["length"])  # Recorded duration.
    return float(np.mean(np.asarray(episode[key])[warmup:length])) if length > warmup else 0.0  # Original zero-summary convention.


def failed(episode: dict, warmup: int = WARMUP) -> bool:
    """Local: classify one episode. Global: same pilot failure rule as analysis._summary (termination, prefix-only, >5% inversion/contact)."""
    length = int(episode["length"])  # Recorded duration.
    physical = length > warmup and max(post_onset_mean(episode, "inversion"), post_onset_mean(episode, "bad_contact")) > 0.05  # Physical failure fractions.
    return bool(episode["terminated"]) or length <= warmup or physical  # Any failure source counts.


def sha256_file(path: Path) -> str:
    """Local: hash one file. Global: record exact audit inputs."""
    with path.open("rb") as stream:  # Binary read.
        return hashlib.file_digest(stream, "sha256").hexdigest()  # Streamed digest.


def sha256_vector(vector) -> str:
    """Local: hash a vector exactly as the runtime receives it. Global: match condition identity files."""
    return hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()  # Same rule as execution.condition_identity.

"""Local: collect the pre-registered fresh-seed episodes. Global: reuse the production rollout cache for every activation arm.

Activation arms go through execution.collect_episodes with a RolloutPool, so they share
the original identity, pairing and logging rules. Perception arms need a per-episode
wrapper around the model, so they run serially with their own identity record.
"""
from __future__ import annotations  # Keep annotations lightweight.

import hashlib  # Source fingerprint of the perception module.
import logging  # Timestamped progress.
import shutil  # Free-space check before a large collection.

from rl_locomotion_steering_vectors.config import Config  # Same warmup and horizon as the original protocol.
from rl_locomotion_steering_vectors.execution import collect_episodes, use_pool  # Production collector and pool scope.
from rl_locomotion_steering_vectors.experiment import condition_name  # Production cache labels.
from rl_locomotion_steering_vectors.parallel import RolloutPool  # Isolated worker processes.
from rl_locomotion_steering_vectors.perception import PerceivedHeadingModel  # Observation-rotation intervention.
from rl_locomotion_steering_vectors.runtime import parameter_hash, run_episode, summarise_episode  # Serial rollout and summary.
from rl_locomotion_steering_vectors.storage import ensure_identity, save_arrays  # Identity guard and atomic NPZ writes.

from .common import ALPHA, AUDIT_RUN, FRESH_SEEDS, MAX_STEPS, ROOT, WARMUP, condition_dir  # Protocol.

PHASE = "audit"  # Episode folder phase name.
MINIMUM_FREE_BYTES = 3 * 1024 ** 3  # Leave headroom on a nearly full disk.


def perception_condition(delta: float) -> str:
    """Local: name a perception arm. Global: keep its folder distinct from activation arms."""
    return f"perceived_heading__{delta:+.6f}rad"  # Radians, six decimals.


def _collect_perception(model, delta: float) -> None:
    """Local: run one perception arm serially. Global: same seeds, onset and horizon as activation arms."""
    folder = condition_dir(AUDIT_RUN, PHASE, perception_condition(delta))  # Arm folder.
    source = (ROOT / "src" / "rl_locomotion_steering_vectors" / "perception.py").read_text(encoding="utf-8")  # Intervention code.
    ensure_identity(folder / "identity.json", {"model_key": "ant", "policy_sha256": parameter_hash(model), "intervention": "perceived_heading_rotation",
                                               "perception_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(), "delta_rad": delta,
                                               "warmup": WARMUP, "max_steps": MAX_STEPS, "stochastic": False})  # Cache provenance.
    for seed in FRESH_SEEDS:  # Same resets as every other arm.
        path = folder / f"{seed}.npz"  # Episode file.
        if path.exists():  # Resume.
            continue  # Already collected.
        episode = run_episode(PerceivedHeadingModel(model, delta, WARMUP), "ant", seed, warmup=WARMUP, max_steps=MAX_STEPS)  # Fresh wrapper per episode.
        save_arrays(path, episode)  # Atomic write.
        logging.info("episode phase=%s condition=%s seed=%s metrics=%s", PHASE, perception_condition(delta), seed, summarise_episode(episode, WARMUP))  # Same log shape as collect_episodes.


def collect_fresh(model, directions: dict, workers: int) -> dict:
    """Local: collect baseline, ±alpha activation arms and ±delta perception arms. Global: complete paired data for every hypothesis."""
    free = shutil.disk_usage(ROOT).free  # Bytes available.
    if free < MINIMUM_FREE_BYTES:
        raise RuntimeError(f"only {free / 1024 ** 3:.1f} GiB free; need at least 3 GiB")  # Refuse to fill the disk.
    config = Config(rollout_workers=workers)  # Default warmup 100, horizon 1000.
    seeds = list(FRESH_SEEDS)  # Pre-registered resets.
    names = []  # Collected condition names.
    with RolloutPool("ant", ROOT, workers=workers) as pool, use_pool(pool):  # Parallel production collector.
        collect_episodes(model, "ant", AUDIT_RUN, PHASE, seeds, config)  # Shared baseline.
        for name, vector in directions["vectors"].items():  # Every direction.
            for alpha in (-ALPHA, ALPHA):  # Both signs for odd/even parts.
                condition = condition_name(name, alpha)  # Production label.
                collect_episodes(model, "ant", AUDIT_RUN, PHASE, seeds, config, condition=condition, vector=vector, alpha=alpha)  # Episodes saved; return value dropped to bound memory.
                names.append(condition)  # Record.
            logging.info("collected direction %s (%d/%d)", name, len(names) // 2, len(directions["vectors"]))  # Progress.
    for delta in (-directions["delta_eq"], directions["delta_eq"]):  # Perception arms.
        _collect_perception(model, delta)  # Serial collection.
        names.append(perception_condition(delta))  # Record.
    return {"conditions": len(names) + 1, "episodes": (len(names) + 1) * len(seeds), "free_gib_before": free / 1024 ** 3}  # Bookkeeping.

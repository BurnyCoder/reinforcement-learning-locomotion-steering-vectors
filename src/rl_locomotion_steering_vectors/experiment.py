"""Local: calibration and held-out comparison; global: freeze choices before confirmation.

Sources: https://arxiv.org/abs/2312.06681 (signed activation additions)
https://numpy.org/doc/1.26/reference/generated/numpy.mean.html
"""
import logging  # Record each completed condition and its full effect estimate.
from pathlib import Path  # Each experiment keeps its own numerical artifacts.

import numpy as np  # Arrays represent interventions without executable serialization.
import torch  # Batch fitting observations using the frozen actor.
from baukit import Trace  # Reuse the audited generic output-editing implementation.

from .analysis import paired_effect  # Apply one consistent episode-level usefulness criterion.
from .config import Config  # Validation and confirmation use the same horizon and onset.
from .execution import collect_episodes, summaries  # Shared rollout code preserves causal pairing.
from .storage import save_json  # Incremental results survive an interrupted search.

STRENGTHS = (-.5, -.2, -.1, -.05, .05, .1, .2, .5)  # Zero is collected once as the shared paired baseline.


def condition_name(name: str, alpha: float) -> str:
    """Local: make stable cache keys; global: distinguish every intervention strength."""
    return f"{name}__{alpha:+.4f}"  # Signed decimal strengths are portable directory names.


def calibrate(model, model_key: str, run_dir: Path, vectors: dict, seeds: list[int], config: Config,
              *, action_biases: dict | None = None, previous: list | None = None) -> list[dict]:
    """Local: evaluate a prespecified grid; global: give candidate and controls equal selection data."""
    baseline = summaries(collect_episodes(model, model_key, run_dir, "validation", seeds, config), config)  # Identical prefix and seeds anchor each comparison.
    results = list(previous or [])  # Bias comparators may be added after vector calibration.
    for name, vector in vectors.items():  # Controls are fitted and scaled before this phase.
        behavior = name.split("_")[0]  # Vector IDs deliberately begin with the measured behavior.
        for alpha in STRENGTHS:  # Both signs receive the same validation opportunity.
            if any(row["vector"] == name and row["alpha"] == alpha for row in results):  # Resume completed comparisons only.
                continue  # Saved effects already reference the same immutable protocol.
            bias = None if action_biases is None else action_biases.get(name)  # Action controls use the same rollout engine.
            episodes = collect_episodes(model, model_key, run_dir, "validation", seeds, config,
                                        condition=condition_name(name, alpha), vector=vector if bias is None else None,
                                        alpha=alpha, action_bias=None if bias is None else alpha * bias)  # Runtime applies a constant action offset after warmup.
            effect = paired_effect(baseline, summaries(episodes, config), behavior, seed=710, warmup=config.warmup)  # Bootstrap whole paired episodes.
            row = {"vector": name, "behavior": behavior, "alpha": alpha, "effect": effect}  # Save signed measured effects, not only pass/fail.
            results.append(row)  # No unsuccessful conditions are discarded.
            save_json(run_dir / "validation.json", results)  # Persist after every grid point.
            logging.info("validation %s", row)  # Full condition results are visible immediately.
    return results  # Selection occurs only after the specified grid is complete.


def select_conditions(rows: list[dict], behaviors: list[str]) -> list[dict]:
    """Local: choose the smallest near-best useful setting; global: lock held-out hypotheses."""
    selected = []  # At most one strength per vector enters confirmation.
    for behavior in behaviors:  # Treat the registered behavior candidates explicitly.
        names = sorted({row["vector"] for row in rows if row["behavior"] == behavior}, key=lambda name: (name != behavior, name))  # Keep primary vectors ahead of controls.
        for name in names:  # Controls get the same maximization opportunity as the candidate.
            choices = [row for row in rows if row["vector"] == name and row["effect"]["gate"]]  # Usefulness includes locomotion and quality.
            if not choices:  # A failing candidate does not become a post-hoc hypothesis.
                if name == behavior:
                    continue  # Retain failures in validation.json rather than confirmation selection.
                choices = [row for row in rows if row["vector"] == name]  # Keep even unsuccessful control results for comparison.
            score = lambda row: abs(row["effect"]["effect"])  # Within one behavior all effects share physical units.
            best = max(score(row) for row in choices)  # The validation maximum defines the equivalence band.
            near = [row for row in choices if score(row) >= .95 * best]  # A prespecified 5% band favors gentler interventions.
            chosen = min(near, key=lambda row: (abs(row["alpha"]), -score(row)))  # Deterministic tie breaking prevents manual cherry picking.
            selected.append({key: chosen[key] for key in ("vector", "behavior", "alpha")} | {"direction": chosen["effect"]["direction"],
                            "role": "candidate" if name == behavior else "control"})  # Freeze the direction before new seeds.
    candidate_behaviors = {row["behavior"] for row in selected if row["role"] == "candidate"}  # Controls accompany a shortlisted hypothesis.
    return [row for row in selected if row["behavior"] in candidate_behaviors]  # Avoid spending confirmation data on rejected searches.


def derive_action_biases(model, episodes: list[dict], selected: list[dict], vectors: dict, warmup: int) -> dict:
    """Local: average action displacement; global: test whether a constant action offset explains usefulness."""
    observations = np.concatenate([episode["observations"][warmup::10] for episode in episodes])  # Fit observations only; subsampling bounds temporary memory.
    baseline, _ = model.predict(observations, deterministic=True)  # SB3 retains its actual preprocessing and action scaling.
    biases = {}  # Each shortlisted behavior gets one constant fitted action direction.
    for row in selected:
        if row["role"] != "candidate":  # Random vectors are already separate controls.
            continue  # Avoid redundant action comparators.
        direction = torch.as_tensor(vectors[row["vector"]], dtype=torch.float32)  # Match CPU actor activation dtype.
        with Trace(model.actor.latent_pi[1], retain_output=False, edit_output=lambda output: output + row["alpha"] * direction):  # Reuse the same first-ReLU addition.
            treated, _ = model.predict(observations, deterministic=True)  # Batch evaluation never changes simulator state.
        biases[f"{row['behavior']}_bias"] = np.mean(treated - baseline, axis=0).astype(np.float32) / .5  # Alpha=.5 reproduces the fitted mean displacement; the same signed grid explores smaller offsets.
    return biases  # These vectors live in action space and are clearly labeled as such.


def evaluate(model, model_key: str, run_dir: Path, phase: str, selected: list[dict], vectors: dict,
             seeds: list[int], config: Config, biases: dict | None = None) -> list[dict]:
    """Local: run frozen settings on fresh seeds; global: estimate confirmation or replication effects."""
    baseline = summaries(collect_episodes(model, model_key, run_dir, phase, seeds, config), config)  # Pair every selected condition against the exact same seeds.
    results = []  # Keep control and failed-candidate estimates alike.
    for row in selected:  # Never optimize using held-out outcomes.
        name, alpha = row["vector"], row["alpha"]  # Read the previously frozen selection.
        bias = (biases or {}).get(name)  # Action offsets have an explicit separate representation.
        episodes = collect_episodes(model, model_key, run_dir, phase, seeds, config,
                                    condition=condition_name(name, alpha), vector=vectors.get(name) if bias is None else None,
                                    alpha=alpha, action_bias=None if bias is None else alpha * bias)  # Equal onset, horizon, and simulator settings.
        effect = paired_effect(baseline, summaries(episodes, config), row["behavior"], seed=720 if phase == "confirmation" else 730, warmup=config.warmup)  # Independent deterministic bootstrap streams.
        results.append(dict(row, effect=effect))  # Preserve the original selected direction beside the measured direction.
        save_json(run_dir / f"{phase}.json", results)  # Resume episodes even if report generation later fails.
        logging.info("%s %s", phase, results[-1])  # Log the complete estimate and every gate.
    return results  # The caller decides success without changing settings.


def successful_candidates(selected: list[dict], results: list[dict]) -> list[dict]:
    """Local: enforce the frozen sign; global: do not reinterpret a reversed effect as success."""
    return [row for row in selected if row["role"] == "candidate" and any(
        result["vector"] == row["vector"] and result["alpha"] == row["alpha"] and result["effect"]["gate"]
        and result["effect"]["direction"] == row["direction"] for result in results)]  # All predeclared usefulness gates must pass.

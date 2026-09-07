"""Local: thin named-phase wrapper; global: execute the registered scientific sequence.

Sources: https://docs.python.org/3.12/library/importlib.metadata.html
Research protocol: docs/methodology.md; phase implementations own numerical details.
"""
import importlib.metadata  # Record actual installed distributions rather than assumed versions.
import logging  # Emit decisions at each scientific boundary.
from pathlib import Path  # Resolve one self-contained run directory.

import torch  # Set the measured CPU inference configuration once.

from .analysis import fit_vectors  # Extraction remains independent of orchestration.
from .config import Config  # Explicit settings become part of the immutable manifest.
from .execution import collect_episodes, diagnose  # Reuse collection for diagnostic and fitting data.
from .experiment import calibrate, derive_action_biases, evaluate, select_conditions, successful_candidates  # Separate numerical evaluation from phase ordering.
from .runtime import prepare_model  # Verify upstream weights and runtime compatibility first.
from .storage import check_manifest, load_arrays, load_json, save_arrays, save_json, utc_now  # Atomic artifacts make every completed phase resumable.


def prepare(model_key: str, run_dir: Path, config: Config) -> tuple:
    """Local: verify model and immutable protocol; global: establish provenance before evidence."""
    torch.set_num_threads(config.torch_threads)  # Avoid excessive threads for tiny MLP batches.
    model, provenance = prepare_model(model_key, Path.cwd())  # Check exact revision/hash, spaces, and frozen architecture.
    splits = {name: list(range(start, start + getattr(config, f"{name}_episodes"))) for name, start in
              (("diagnostic", 0), ("fit", 1000), ("validation", 10000), ("confirmation", 20000), ("replication", 30000))}  # Whole episodes stay in one phase.
    all_seeds = [seed for group in splits.values() for seed in group]  # Guard future configuration expansions.
    if len(all_seeds) != len(set(all_seeds)):  # Excessively large counts could overlap nominal ranges.
        raise ValueError("Configured seed splits overlap; specify a new protocol")  # Stop rather than allow leakage.
    manifest = {"schema": 1, "created_utc": utc_now(), "model_key": model_key, "config": config.as_dict(), "seed_splits": splits,
                "provenance": provenance, "decisions": [], "software": {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "torchvision", "stable-baselines3", "sb3-contrib", "gymnasium", "mujoco", "baukit")}}  # Save measured package versions alongside checkpoint identity.
    path = run_dir / "manifest.json"  # One run directory has one immutable protocol.
    if path.exists():  # Resume only when configuration and data splits match.
        saved = load_json(path)  # Keep the original creation time and provenance.
        check_manifest(saved, manifest)  # Reject incompatible cached evidence.
        if saved["provenance"]["sha256"] != provenance["sha256"]:  # A changed policy invalidates all causal comparisons.
            raise ValueError("Checkpoint differs from saved manifest")  # Never reuse episodes under different weights.
        manifest = saved  # Resume the exact existing experiment identity.
    else:
        save_json(path, manifest)  # Persist the protocol before the first diagnostic rollout.
    logging.info("prepared manifest=%s", manifest)  # Complete nonsensitive configuration is retained in logs.
    return model, manifest  # Later phases receive resolved state rather than repeating setup.


def run_pipeline(model_key: str, run_dir: Path, config: Config, stop_after: str | None = None) -> dict:
    """Local: call clearly named phases; global: preserve the fitting-to-replication boundary."""
    model, manifest = prepare(model_key, run_dir, config)  # Verify the reusable foundation.
    seeds = manifest["seed_splits"]  # Seed groups are fixed before observing results.
    result = load_json(run_dir / "results.json") if (run_dir / "results.json").exists() else {"status": "prepared", "provenance": manifest["provenance"]}  # Resume reporting state without redefining the protocol.
    diagnostic = collect_episodes(model, model_key, run_dir, "diagnostic", seeds["diagnostic"], config, stochastic=True)  # Characterize natural policy variation.
    result.update(diagnostics=diagnose(diagnostic, config), status="diagnosed")  # Save measured behavior, quality, and outliers.
    save_json(run_dir / "results.json", result)  # Diagnostic evidence precedes extraction.
    if stop_after == "diagnose":
        return result  # Permit a human-visible diagnostic checkpoint through the public CLI.
    fitting = collect_episodes(model, model_key, run_dir, "fit", seeds["fit"], config, capture=True, stochastic=True)  # Independent stochastic data supplies activation contrasts.
    behaviors = ["speed", "effort", "height"] if model_key == "halfcheetah" else ["lateral", "turning"]  # Register environment-appropriate candidates.
    vectors, extraction = fit_vectors(fitting, behaviors, seed=0)  # Fit candidates and controls using fitting episodes only.
    save_arrays(run_dir / "vectors.npz", vectors)  # Store scaled directions as non-executable arrays.
    save_json(run_dir / "vector_diagnostics.json", extraction)  # Preserve raw vectors, scale, groups, and numerical gates.
    result.update(extraction=extraction, status="fitted" if vectors else "no_eligible_vectors")  # A null extraction remains an explicit research result.
    save_json(run_dir / "results.json", result)  # Keep every phase's outcome even when escalation is required.
    if stop_after == "extract" or not vectors:
        return result  # Do not invent contrasts when the fitting criteria fail.
    previous = load_json(run_dir / "validation.json") if (run_dir / "validation.json").exists() else []  # Resume the fixed strength grid.
    validation = calibrate(model, model_key, run_dir, vectors, seeds["validation"], config, previous=previous)  # Give every direction the same validation opportunities.
    selected = select_conditions(validation, behaviors)  # Choose useful candidate signs/strengths outside confirmation data.
    if selected and not (run_dir / "selection.json").exists():  # Derive action controls before the final selection is frozen.
        biases = derive_action_biases(model, fitting, selected, vectors, config.warmup)  # Fit a constant action-displacement comparator.
        save_arrays(run_dir / "action_biases.npz", biases)  # Clearly separate action-space offsets from activation vectors.
        validation = calibrate(model, model_key, run_dir, biases, seeds["validation"], config, action_biases=biases, previous=validation)  # Calibrate action controls on the same validation seeds.
        selected = select_conditions(validation, behaviors)  # Include the calibrated comparator in the frozen selection.
        save_json(run_dir / "selection.json", {"created_utc": utc_now(), "conditions": selected, "rule": "useful gate; smallest strength within 5% of best absolute effect"})  # Timestamp choices before confirmation begins.
    elif (run_dir / "selection.json").exists():
        selected = load_json(run_dir / "selection.json")["conditions"]  # Never tune a previously confirmed selection again.
    result.update(validation=validation, selected=selected, status="validated" if selected else "no_validation_candidate")  # Retain the full grid and selected controls.
    save_json(run_dir / "results.json", result)  # Report selection without consulting held-out data.
    if stop_after == "calibrate" or not selected:
        return result  # A failed grid motivates a separately registered hypothesis.
    biases = load_arrays(run_dir / "action_biases.npz") if (run_dir / "action_biases.npz").exists() else {}  # Reuse frozen action controls.
    confirmation = evaluate(model, model_key, run_dir, "confirmation", selected, vectors, seeds["confirmation"], config, biases)  # Test frozen hypotheses on thirty new pairs.
    confirmed = successful_candidates(selected, confirmation)  # Enforce the validation-selected direction and all gates.
    result.update(confirmation=confirmation, status="confirmed" if confirmed else "confirmation_failed")  # Failed attempts remain in the experiment record.
    save_json(run_dir / "results.json", result)  # Save before independent replication.
    if stop_after == "confirm" or not confirmed:
        return result  # Do not retune on failed confirmation episodes.
    replication = evaluate(model, model_key, run_dir, "replication", selected, vectors, seeds["replication"], config, biases)  # Independently repeat every frozen condition.
    replicated = successful_candidates(confirmed, replication)  # Require the same useful effect twice.
    result.update(replication=replication, replicated=replicated, status="replicated" if replicated else "replication_failed")  # Successful rollouts are not yet a practical demonstration.
    save_json(run_dir / "results.json", result)  # Preserve the complete evidence before report generation.
    return result  # CLI reporting is separate so numerical work survives rendering failures.

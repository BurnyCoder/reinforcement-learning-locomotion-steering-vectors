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
from .phases import run_evaluation_phases, save_fixed_vectors  # Fitted and imported treatments share cache guards and the complete evidence sequence.
from .runtime import prepare_model  # Verify upstream weights and runtime compatibility first.
from .storage import check_manifest, code_identity, load_json, save_json, utc_now  # Atomic artifacts and source fingerprints make every completed phase reproducible.


def prepare(model_key: str, run_dir: Path, config: Config, source_manifest: dict | None = None) -> tuple:
    """Local: verify model and immutable protocol; global: establish provenance before evidence."""
    torch.set_num_threads(config.torch_threads)  # Avoid excessive threads for tiny MLP batches.
    model, provenance = prepare_model(model_key, Path.cwd())  # Check exact revision/hash, spaces, and frozen architecture.
    implementation = code_identity(Path.cwd())  # Capture current owned code even when resuming an earlier manifest.
    logging.info("invocation code=%s", implementation)  # Keep each invocation's actual source identity in its timestamped log.
    splits = {name: list(range(config.seed_offset + start, config.seed_offset + start + getattr(config, f"{name}_episodes"))) for name, start in
              (("diagnostic", 0), ("fit", 1000), ("validation", 10000), ("confirmation", 20000), ("replication", 30000))}  # Whole episodes stay in one phase.
    if source_manifest is not None:  # A registered follow-up can reuse existing fitting evidence without refitting a chosen vector.
        if source_manifest["provenance"]["sha256"] != provenance["sha256"] or any(source_manifest["config"][key] != config.as_dict()[key] for key in ("warmup", "max_steps")):  # Imported evidence must describe the same actor and measurement schedule.
            raise ValueError("Imported fitting policy or rollout schedule differs")  # Reject incompatible activation and action-bias fitting data.
        splits.update({phase: source_manifest["seed_splits"][phase] for phase in ("diagnostic", "fit")})  # Keep original fitting identities rather than pretend to collect new data.
    all_seeds = [seed for group in splits.values() for seed in group]  # Guard future configuration expansions.
    if len(all_seeds) != len(set(all_seeds)):  # Excessively large counts could overlap nominal ranges.
        raise ValueError("Configured seed splits overlap; specify a new protocol")  # Stop rather than allow leakage.
    manifest = {"schema": 1, "created_utc": utc_now(), "model_key": model_key, "config": config.as_dict(), "seed_splits": splits,
                "provenance": provenance, "code": implementation, "decisions": [], "software": {name: importlib.metadata.version(name) for name in
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
    vectors, extraction = fit_vectors(fitting, behaviors, seed=0, warmup=config.warmup, minimum_speed_fraction=config.fit_min_speed_fraction)  # Fit candidates and controls using fitting episodes only.
    save_fixed_vectors(run_dir, vectors)  # Preserve exact treatment identity before accepting cached comparisons.
    save_json(run_dir / "vector_diagnostics.json", extraction)  # Preserve raw vectors, scale, groups, and numerical gates.
    result.update(extraction=extraction, status="fitted" if vectors else "no_eligible_vectors")  # A null extraction remains an explicit research result.
    save_json(run_dir / "results.json", result)  # Keep every phase's outcome even when escalation is required.
    if stop_after == "extract" or not vectors:
        return result  # Do not invent contrasts when the fitting criteria fail.
    return run_evaluation_phases(model, model_key, run_dir, vectors, behaviors, seeds, config, result,
                                 lambda: fitting, stop_after=stop_after)  # Reuse the common calibration-to-replication protocol and its inspection boundaries.

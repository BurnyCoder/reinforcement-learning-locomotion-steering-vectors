"""Local: shared evidence phases; global: give fitted and imported vectors the same causal protocol.

Sources: https://refactoring.com/catalog/extractFunction.html
https://numpy.org/doc/1.26/reference/generated/numpy.array_equal.html
Numerical selection and analysis stay in experiment.py; this module only orders them.
"""
from collections.abc import Callable  # Load source fitting observations only when a comparator is needed.
from pathlib import Path  # Keep all phase artifacts inside the selected run.

import numpy as np  # Compare treatment identity before permitting cache reuse.

from .config import Config  # All phases receive the same immutable experiment settings.
from .experiment import ANALYSIS_ID, calibrate, derive_action_biases, evaluate, select_conditions, successful_candidates  # Reuse the existing numerical implementation without changing its fingerprint.
from .storage import load_arrays, load_json, save_arrays, save_json, utc_now  # Persist each completed scientific boundary atomically.


def save_fixed_vectors(run_dir: Path, vectors: dict) -> None:
    """Local: check before saving treatments; global: prohibit altered vectors beneath cached outcomes."""
    path = run_dir / "vectors.npz"  # Both fitted and imported treatments share this artifact contract.
    if path.exists():  # Resumption must preserve the original intervention exactly.
        saved = load_arrays(path)  # Read numerical contents without executable serialization.
        if saved.keys() != vectors.keys() or any(saved[name].dtype != vectors[name].dtype or not np.array_equal(saved[name], vectors[name]) for name in vectors):  # Compare names, dtype, shape, and every value.
            raise ValueError("Run vectors changed; preserve this run and start a new experiment")  # Reject before any overwrite or cached comparison.
    save_arrays(path, vectors)  # Publish only a new or exactly matching direction set.


def run_evaluation_phases(model, model_key: str, run_dir: Path, vectors: dict, behaviors: list[str],
                          seeds: dict, config: Config, result: dict, fitting_loader: Callable[[], list[dict]],
                          stop_after: str | None = None) -> dict:
    """Local: calibrate/select/confirm/replicate; global: freeze useful hypotheses before fresh evidence."""
    previous = load_json(run_dir / "validation.json") if (run_dir / "validation.json").exists() else []  # Resume the prespecified grid with its existing analysis-identity guard.
    validation = calibrate(model, model_key, run_dir, vectors, seeds["validation"], config, previous=previous)  # Candidate and direction controls receive identical opportunities.
    selected = select_conditions(validation, behaviors)  # Selection uses validation data exclusively.
    selection_path = run_dir / "selection.json"  # One frozen selection governs every held-out phase.
    if selected and not selection_path.exists():  # Fit action controls only for a newly shortlisted hypothesis.
        biases = derive_action_biases(model, fitting_loader(), selected, vectors, config.warmup,
                                     reference_strength=max(abs(value) for value in config.strengths))  # Fit only on original fitting observations and preserve the full action-displacement range when the grid is finer.
        save_arrays(run_dir / "action_biases.npz", biases)  # Keep action-space comparators separate from activation vectors.
        validation = calibrate(model, model_key, run_dir, biases, seeds["validation"], config, action_biases=biases, previous=validation)  # Give the comparator the same calibration grid and episodes.
        selected = select_conditions(validation, behaviors)  # Include calibrated controls before locking the hypothesis.
        save_json(selection_path, {"created_utc": utc_now(), "conditions": selected, "analysis_sha256": ANALYSIS_ID,
                                   "rule": "useful gate; smallest strength within 5% of best absolute effect"})  # Preserve sign, strength, rule, and metric implementation before confirmation.
    elif selection_path.exists():  # Even a saved empty selection belongs to its original analysis definition.
        selection = load_json(selection_path)  # Previously selected settings cannot be retuned on resumption.
        if selection.get("analysis_sha256") != ANALYSIS_ID:  # A changed metric implementation requires an explicitly new attempt.
            raise ValueError("Selected analysis changed; preserve this attempt and register a new experiment")  # Keep previously inspected evidence attached to its original meaning.
        selected = selection["conditions"]  # Reuse the exact original strengths and controls.
    result.update(validation=validation, selected=selected, status="validated" if selected else "no_validation_candidate")  # A null grid remains a complete recorded outcome.
    save_json(run_dir / "results.json", result)  # Persist the selection boundary before spending fresh episodes.
    if stop_after == "calibrate" or not selected:
        return result  # Respect inspection stops and do not confirm rejected hypotheses.
    biases = load_arrays(run_dir / "action_biases.npz") if (run_dir / "action_biases.npz").exists() else {}  # Reuse frozen comparator offsets rather than fitting again.
    confirmation = evaluate(model, model_key, run_dir, "confirmation", selected, vectors, seeds["confirmation"], config, biases)  # Test every frozen candidate and control on new paired episodes.
    confirmed = successful_candidates(selected, confirmation)  # Require the selected sign and every usefulness gate.
    result.update(confirmation=confirmation, status="confirmed" if confirmed else "confirmation_failed")  # All failed attempts remain visible.
    save_json(run_dir / "results.json", result)  # Save confirmation before independent replication begins.
    if stop_after == "confirm" or not confirmed:
        return result  # Failed confirmation cannot trigger tuning on the same evidence.
    replication = evaluate(model, model_key, run_dir, "replication", selected, vectors, seeds["replication"], config, biases)  # Repeat the entire frozen comparison, including its controls.
    replicated = successful_candidates(confirmed, replication)  # Success must occur independently in the same direction twice.
    result.update(replication=replication, replicated=replicated, status="replicated" if replicated else "replication_failed")  # Replication still precedes the separate practical demonstration.
    save_json(run_dir / "results.json", result)  # Preserve complete numerical evidence before optional rendering.
    return result  # Both research wrappers return the same documented result schema.

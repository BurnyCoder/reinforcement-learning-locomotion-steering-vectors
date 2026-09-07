"""Local: test a saved contrast against another outcome; global: register cross-behavior effects explicitly.

Sources: https://arxiv.org/abs/2312.06681 (contrastive activation additions)
https://numpy.org/doc/1.26/reference/generated/numpy.array_equal.html
https://numpy.org/doc/1.26/reference/generated/numpy.ndarray.tobytes.html
https://docs.python.org/3.12/library/hashlib.html
This adapter reuses the existing rollout, calibration, control, and inference phases.
"""
import logging  # Record the new hypothesis and each completed evidence stage.
import hashlib  # Pin numerical source contents independently of their mutable container file.
from pathlib import Path  # Imported evidence must remain inside the project.

from .config import Config  # Fresh evaluation splits use the same explicit protocol.
from .phases import run_evaluation_phases, save_fixed_vectors  # Imported and fitted vectors must follow identical selection and evidence rules.
from .pipeline import prepare  # Checkpoint verification and split separation remain centralized.
from .storage import ensure_identity, load_arrays, load_json, save_json, utc_now  # Preserve exact source directions and immutable selection.


def import_family(source: dict, original: str, target: str) -> dict:
    """Local: relabel a complete family; global: change the tested outcome without changing directions."""
    suffixes = ("", "_random0", "_random1", "_random2", "_shuffled")  # Require all controls to accompany the candidate.
    if any(original + suffix not in source for suffix in suffixes):  # Missing controls would weaken the registered comparison.
        raise ValueError("Source vector family lacks required random/shuffled controls")  # Do not silently substitute newly sampled controls.
    return {target + suffix: source[original + suffix].copy() for suffix in suffixes}  # Copies preserve exact values while protecting source arrays from later mutation.


def run_retarget(source_dir: Path, run_dir: Path, original: str, target: str, config: Config, stop_after: str | None = None) -> dict:
    """Local: register/import/calibrate/evaluate; global: confirm a cross-behavior hypothesis on untouched episodes."""
    if target not in ("speed", "effort", "height", "lateral", "turning"):  # Only implemented physical outcomes may be tested.
        raise ValueError("Unknown target behavior")  # Reject ambiguous outcome definitions before collecting data.
    source_manifest = load_json(source_dir / "manifest.json")  # Retain the parent policy and exact fitting split.
    source_results = load_json(source_dir / "results.json")  # Parent outcomes informed this explicitly exploratory follow-up.
    vectors = import_family(load_arrays(source_dir / "vectors.npz"), original, target)  # Keep the candidate and control directions fixed.
    model_key = source_manifest["model_key"]  # The follow-up concerns the same frozen policy, not cross-policy transfer.
    model, manifest = prepare(model_key, run_dir, config, source_manifest=source_manifest)  # Verify weights and disjoint reused-fitting/fresh-evaluation splits.
    old_evaluation = {seed for phase in ("validation", "confirmation", "replication") for seed in source_manifest["seed_splits"][phase]}  # Parent evaluation may have informed hypothesis formation.
    new_evaluation = {seed for phase in ("validation", "confirmation", "replication") for seed in manifest["seed_splits"][phase]}  # The new statistical evidence must be untouched.
    if old_evaluation & new_evaluation:  # Never treat previously examined reset outcomes as new confirmation evidence.
        raise ValueError("Retargeted evaluation requires fresh seeds; change STEERING_SEED_OFFSET")  # Force an explicit new partition.
    specification = {"source_run": str(source_dir.relative_to(Path.cwd())), "source_vector": original, "target_behavior": target,
                     "source_manifest": source_manifest, "source_status": source_results["status"],
                     "source_vectors": {original + name[len(target):]: {"shape": list(vector.shape), "dtype": str(vector.dtype), "sha256": hashlib.sha256(vector.tobytes()).hexdigest()} for name, vector in vectors.items()}}  # Bind every candidate/control payload as well as the parent protocol without copying large trajectories.
    ensure_identity(run_dir / "retarget.json", specification)  # A resumed follow-up cannot swap its source vector or outcome.
    save_fixed_vectors(run_dir, vectors)  # Reject changed local imports before reusing any cached condition results.
    source_extraction = load_json(source_dir / "vector_diagnostics.json")  # Retain raw contrasts, scaling, and contributor metadata.
    extraction = {"imported": specification, "source_extraction": source_extraction,
                  "behaviors": {target: {"status": "imported unchanged", "source_behavior": original}},
                  "vectors": {new: dict(source_extraction["vectors"][original + new[len(target):]], source_vector=original + new[len(target):], target_behavior=target) for new in vectors}}  # Keep extraction label and tested outcome distinct.
    save_json(run_dir / "vector_diagnostics.json", extraction)  # Document the full numerical provenance rather than implying a new fitted speed concept.
    if not manifest["decisions"]:  # Register the hypothesis before this run's validation observations exist.
        manifest["decisions"].append({"time_utc": utc_now(), "decision": f"Test whether the unchanged {original}-contrast vector from {specification['source_run']} provides useful {target} control. Recalibrate its complete control family on fresh validation episodes, then freeze selection before confirmation.",
                                      "evidence": "Parent validation suggested a cross-behavior effect. This is an explicitly adaptive exploratory follow-up; it does not identify an internal semantic concept."})  # State why this experiment exists and what it can establish.
        save_json(run_dir / "manifest.json", manifest)  # Timestamp the new question before any rollout.
    result = {"status": "imported", "provenance": manifest["provenance"], "extraction": extraction,
              "decision": f"The direction was fitted using {original} contrasts and is now tested for {target}; the numerical vector and source fitting data are unchanged."}  # Keep the report's claim aligned with the actual experiment.
    if stop_after == "extract":  # The imported treatment is reviewable before spending fresh validation episodes.
        save_json(run_dir / "results.json", result)  # Persist the import-only inspection boundary.
        return result  # Follow-up inspection consumes no new evaluation data.
    result = run_evaluation_phases(model, model_key, run_dir, vectors, [target], manifest["seed_splits"], config, result,
        lambda: [load_arrays(source_dir / "episodes" / "fit" / "baseline" / f"{seed}.npz") for seed in source_manifest["seed_splits"]["fit"]],
        stop_after=stop_after)  # Load original fitting observations only when the common phase derives an action comparator.
    logging.info("retarget completed status=%s", result["status"])  # Distinguish this follow-up from the rejected parent search.
    return result  # The existing report command renders this saved schema.

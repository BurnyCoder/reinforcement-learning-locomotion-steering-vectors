"""Local: verify imported identity and shared phase boundaries; global: protect fresh evidence.

Source: https://docs.pytest.org/en/stable/how-to/monkeypatch.html
"""
import numpy as np  # Synthetic directions make exact preservation independently testable.
import pytest  # Missing controls must be explicit errors.
from rl_locomotion_steering_vectors import retarget  # Replace model setup while retaining real artifact checks.
from rl_locomotion_steering_vectors.config import Config  # Exercise the public orchestration interface.
from rl_locomotion_steering_vectors.retarget import import_family  # Test the small adapter around existing experiment phases.
from rl_locomotion_steering_vectors.storage import load_arrays, load_json, save_arrays, save_json  # Test the actual safe round-trip format.


def test_import_preserves_candidate_and_three_control_vectors():
    """Local: rename only keys; global: preserve the exact previously discovered intervention."""
    source = {"height" + suffix: np.arange(3, dtype=np.float32) + i for i, suffix in enumerate(("", "_random0", "_random1", "_random2", "_shuffled"))}  # Distinct payloads detect accidental regeneration.
    result = import_family(source, "height", "speed")  # Change the tested outcome, not the numerical direction.
    assert len(result) == 5  # Retain all prescribed random and shuffled controls.
    for name, vector in source.items():
        np.testing.assert_array_equal(result[name.replace("height", "speed", 1)], vector)  # Require exact values and correspondence.
    result["speed"][0] = -100  # Simulate later in-memory manipulation of an imported array.
    assert source["height"][0] == 0  # The parent fitting artifact must remain independent.


def test_import_rejects_missing_random_control():
    """Local: enforce a complete control family; global: avoid an easier follow-up comparison."""
    with pytest.raises(ValueError, match="controls"):
        import_family({"height": np.ones(3)}, "height", "speed")  # A lone direction is insufficient for this registered workflow.


@pytest.fixture
def parent_run(tmp_path, monkeypatch):
    """Local: create a tiny complete source; global: test provenance without invoking simulation."""
    monkeypatch.chdir(tmp_path)  # Keep source-relative paths inside this test's isolated project.
    source, destination = tmp_path / "parent", tmp_path / "followup"  # Distinct experiments must own distinct artifacts.
    vectors = {"height" + suffix: np.arange(3, dtype=np.float32) + i for i, suffix in enumerate(("", "_random0", "_random1", "_random2", "_shuffled"))}  # Values identify every inherited direction.
    manifest = {"model_key": "halfcheetah", "provenance": {"sha256": "same-policy"}, "decisions": [],
                "seed_splits": {"diagnostic": [0], "fit": [1000], "validation": [10000], "confirmation": [20000], "replication": [30000]}}  # The parent evaluation has already influenced the follow-up question.
    fresh = dict(manifest, decisions=[], seed_splits=dict(manifest["seed_splits"], validation=[210000], confirmation=[220000], replication=[230000]))  # Only untouched evaluation seeds qualify as new evidence.
    save_json(source / "manifest.json", manifest)  # Retain the exact source protocol for the identity guard.
    save_json(source / "results.json", {"status": "no_validation_candidate"})  # This follow-up is explicitly adaptive to a negative parent search.
    save_json(source / "vector_diagnostics.json", {"vectors": {name: {"behavior": "height"} for name in vectors}})  # Imported metadata stays distinguishable from the new measured outcome.
    save_arrays(source / "vectors.npz", vectors)  # Exercise actual non-executable serialized directions.
    monkeypatch.setattr(retarget, "prepare", lambda *args, **kwargs: (None, fresh))  # Replace expensive model loading, not scientific identity logic.
    return source, destination, fresh  # Individual cases vary one condition at a time.


def test_retarget_rejects_reusing_parent_evaluation_seeds(parent_run):
    """Local: inject one leaked seed; global: reject a falsely fresh confirmation sample."""
    source, destination, fresh = parent_run  # Reuse the verified tiny artifact setup.
    fresh["seed_splits"]["confirmation"] = [10000]  # A parent validation seed may not become a follow-up confirmation seed.
    with pytest.raises(ValueError, match="fresh seeds"):
        retarget.run_retarget(source, destination, "height", "speed", Config())  # Reject before any evaluation function is reached.
    assert not (destination / "vectors.npz").exists()  # Failed partition checks must not publish imported treatments.


@pytest.mark.parametrize("changed_artifact", ["source", "destination"])
def test_retarget_rejects_changed_vectors_on_resume(parent_run, changed_artifact):
    """Local: alter vector bytes under identical manifests; global: forbid cached old-treatment comparisons."""
    source, destination, _ = parent_run  # Neither policy identity nor episode splits will change.
    result = retarget.run_retarget(source, destination, "height", "speed", Config(), stop_after="extract")  # Freeze numerical provenance without spending validation data.
    assert result["status"] == "imported"  # This inspection boundary must stop before calibration.
    original = load_arrays(destination / "vectors.npz")  # Keep the initial imported payload for comparison.
    path = (source if changed_artifact == "source" else destination) / "vectors.npz"  # Cover upstream changes and corrupted local imports separately.
    altered = load_arrays(path)  # Modify a numeric vector while leaving both manifests unchanged.
    altered["height" if changed_artifact == "source" else "speed"][0] += 1  # Even one altered component changes the intervention.
    save_arrays(path, altered)  # Reproduce a realistic changed file on resumption.
    with pytest.raises(ValueError, match="identity|vectors changed"):
        retarget.run_retarget(source, destination, "height", "speed", Config(), stop_after="extract")  # Refuse silent overwrite beneath cached condition names.
    if changed_artifact == "source":
        np.testing.assert_array_equal(load_arrays(destination / "vectors.npz")["speed"], original["speed"])  # Failed import must preserve the original treatment.


@pytest.mark.parametrize("stop_after, expected", [("calibrate", "validated"), ("confirm", "confirmed"), (None, "replicated")])
def test_shared_phases_preserve_frozen_controls_and_stop_boundaries(tmp_path, monkeypatch, stop_after, expected):
    """Local: replace costly rollouts; global: verify controls precede selection and held-out data stays untouched until allowed."""
    from rl_locomotion_steering_vectors import phases  # Import inside the test so missing shared infrastructure fails explicitly.
    calls, fitting = [], [{"observations": np.ones((2, 3))}]  # Keep a trace of which scientific stages actually run.
    candidate = {"vector": "speed", "behavior": "speed", "alpha": .1, "effect": {"effect": .2, "gate": True, "direction": 1}}  # A useful synthetic validation outcome advances to held-out evaluation.

    def calibrate(*args, action_biases=None, previous=None):
        """Local: synthesize comparable grid outcomes; global: assert action controls are also calibrated."""
        calls.append("action_control" if action_biases is not None else "vectors")  # Distinguish the two uses of the same reusable phase.
        return [candidate] if action_biases is None else [candidate, dict(candidate, vector="speed_bias")]  # Preserve the candidate and calibrated comparator together.

    def derive(model, episodes, selected, vectors, warmup):
        """Local: inspect the action-control inputs; global: restrict comparator fitting to the provided fitting set."""
        assert episodes is fitting  # The shared wrapper must not substitute validation or confirmation observations.
        calls.append("fit_action_control")  # Derivation must precede the final selection file.
        return {"speed_bias": np.ones(2)}  # A different dimension keeps action and activation spaces distinct.

    def evaluate(model, model_key, run_dir, phase, selected, vectors, seeds, config, biases):
        """Local: expose frozen condition use; global: prohibit held-out retuning and omitted controls."""
        calls.append(phase)  # The test checks that inspection stops do not spend the next phase.
        assert selected == load_json(run_dir / "selection.json")["conditions"]  # Every held-out evaluation must match the saved choice.
        assert {row["vector"] for row in selected} == {"speed", "speed_bias"}  # Confirmation and replication both retain the comparator.
        return [dict(row, effect={"gate": True, "direction": 1}) for row in selected]  # Both samples support the prespecified sign.

    monkeypatch.setattr(phases, "calibrate", calibrate)  # Isolate sequencing from physics without replacing selection rules.
    monkeypatch.setattr(phases, "derive_action_biases", derive)  # Verify fitting data routing at the actual helper boundary.
    monkeypatch.setattr(phases, "evaluate", evaluate)  # Keep all stop points observable and deterministic.
    result = phases.run_evaluation_phases(None, "halfcheetah", tmp_path, {"speed": np.ones(3)}, ["speed"],
        {"validation": [1], "confirmation": [2], "replication": [3]}, Config(), {}, lambda: fitting, stop_after=stop_after)  # Exercise the single common continuation used by both wrappers.
    assert result["status"] == expected  # The externally visible phase state is preserved.
    assert calls == ["vectors", "fit_action_control", "action_control"] + ([] if stop_after == "calibrate" else ["confirmation"]) + (["replication"] if stop_after is None else [])  # No later evidence is consumed across an inspection boundary.
    if stop_after is None:
        assert [row["vector"] for row in result["replicated"]] == ["speed"]  # Controls accompany discovery claims but cannot themselves satisfy the objective.


def test_shared_phases_reject_old_selection_analysis_and_skip_empty_candidates(tmp_path, monkeypatch):
    """Local: exercise two early exits; global: preserve analysis identity and avoid spent held-out data after a null grid."""
    from rl_locomotion_steering_vectors import phases  # Resolve the same phase function used by both research paths.
    monkeypatch.setattr(phases, "calibrate", lambda *args, **kwargs: [])  # A correct null result is valid research progress.
    loader = lambda: pytest.fail("Rejected or already selected hypotheses must not reload fitting data")  # Detect unnecessary work or accidental data reuse.
    args = (None, "halfcheetah", tmp_path, {}, ["speed"], {"validation": [1]}, Config(), {}, loader)  # No held-out split exists, so any attempted evaluation would also fail.
    assert phases.run_evaluation_phases(*args)["status"] == "no_validation_candidate"  # A zero-candidate grid stops normally.
    save_json(tmp_path / "selection.json", {"analysis_sha256": "prior-definition", "conditions": []})  # Even an empty selection belongs to one metric implementation.
    with pytest.raises(ValueError, match="Selected analysis changed"):
        phases.run_evaluation_phases(*args)  # Existing selection guards take precedence over silently accepting a new null grid.

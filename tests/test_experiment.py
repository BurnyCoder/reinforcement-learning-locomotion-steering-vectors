"""Local: test frozen selection; global: prevent confirmation-driven retuning.

Source: https://docs.pytest.org/en/stable/how-to/assert.html
"""
from types import SimpleNamespace  # A tiny actor isolates comparator normalization from training.

import numpy as np  # Exact fitting observations make expected action shifts independently calculable.
import pytest  # Explicit provenance failures prevent incompatible result reuse.
import torch  # Run the real Baukit hook against an ordinary PyTorch actor layer.
from rl_locomotion_steering_vectors import experiment  # Replace expensive rollout collection at the calibration boundary.
from rl_locomotion_steering_vectors.config import Config  # Supply normal protocol settings to the calibration boundary.
from rl_locomotion_steering_vectors.experiment import select_conditions, successful_candidates, calibrate


def row(name, alpha, effect, gate=True):
    """Local: synthesize validation rows; global: isolate selection logic from physics."""
    return {"vector": name, "behavior": "speed", "alpha": alpha,
            "effect": {"effect": effect, "relative_effect": effect, "gate": gate, "direction": 1}}


def test_selection_prefers_small_equivalent_intervention():
    """Local: use a fixed five-percent equivalence band; global: avoid excess perturbation."""
    selected = select_conditions([row("speed", .1, .2), row("speed", .5, .205), row("speed_random0", .2, .1)], ["speed"])
    assert selected[0]["alpha"] == .1  # Nearly equivalent change should use less steering.
    assert selected[0]["role"] == "candidate"  # Controls cannot satisfy the discovery objective.
    assert selected[1]["role"] == "control"  # Preserve a random comparator for held-out testing.


def test_reversed_confirmation_is_not_success():
    """Local: compare signs; global: a changed direction cannot count as replication."""
    selected = [{"vector": "speed", "behavior": "speed", "alpha": .1, "direction": 1, "role": "candidate"}]
    confirmed = [dict(selected[0], effect={"gate": True, "direction": -1})]  # Magnitude alone is insufficient.
    assert successful_candidates(selected, confirmed) == []  # Reject post-hoc reinterpretation.


def test_changed_analysis_cannot_reuse_saved_validation(tmp_path):
    """Local: reject stale derived results; global: keep metric corrections auditable."""
    previous = [dict(row("speed", .1, .2), analysis_sha256="old-code")]  # Simulate a different failure definition.
    with pytest.raises(ValueError, match="analysis"):
        calibrate(None, "halfcheetah", tmp_path, {}, [1, 2], Config(), previous=previous)  # Stop before loading or collecting episodes.


def test_calibration_uses_only_the_configured_grid(tmp_path, monkeypatch):
    """Local: inspect collected strengths; global: make the frozen grid govern candidate and controls."""
    collected = []  # Capture actual rollout requests rather than inspecting implementation text.

    def collect(*args, **kwargs):
        """Local: record a condition request; global: avoid consuming real validation seeds in a unit test."""
        collected.append(kwargs.get("alpha", 0.0))  # Baseline is separate from the nonzero signed grid.
        return []  # Other boundaries below supply the irrelevant statistical payload.

    monkeypatch.setattr(experiment, "collect_episodes", collect)  # Keep the calibration loop under test.
    monkeypatch.setattr(experiment, "summaries", lambda *args: [])  # Pure sequencing needs no simulated physical summaries.
    monkeypatch.setattr(experiment, "paired_effect", lambda *args, **kwargs: {"gate": False})  # A null outcome must not suppress a grid condition.
    config = Config(strength_grid="-0.02,-0.01,0.01,0.02")  # Use a grid with no values shared by the old hard-coded tuple.
    rows = calibrate(None, "halfcheetah", tmp_path, {"speed": np.ones(2)}, [5], config)  # Real persistence writes the selected protocol's outcomes.
    assert collected == [0.0, *config.strengths]  # Zero appears once, followed by every prespecified nonzero condition.
    assert tuple(result["alpha"] for result in rows) == config.strengths  # Saved numerical results must describe the same search.


@pytest.mark.parametrize("reference_strength", [.5, .025])
def test_action_bias_reference_reproduces_fitted_displacement(reference_strength):
    """Local: verify comparator units with a real hook; global: fine grids retain comparable action offsets."""
    actor = SimpleNamespace(latent_pi=torch.nn.Sequential(torch.nn.Identity(), torch.nn.ReLU()))  # Positive fixture observations make the baseline transparent.

    def predict(observations, deterministic=True):
        """Local: run a small frozen actor; global: exercise the same predict-and-hook boundary as SB3."""
        with torch.no_grad():
            return actor.latent_pi(torch.as_tensor(observations, dtype=torch.float32)).numpy(), None  # Activation addition directly equals action displacement in this fixture.

    model = SimpleNamespace(actor=actor, predict=predict)  # Only the documented inference interface is needed for normalization.
    episodes = [{"observations": np.full((31, 2), 2.0, dtype=np.float32)}]  # Several fitting subsamples should yield the same expected shift.
    selected = [{"vector": "speed", "behavior": "speed", "alpha": -.125, "role": "candidate"}]  # Negative candidate strengths must retain their induced action direction.
    vector = np.array([1.0, -2.0], dtype=np.float32)  # Mixed coordinates detect accidental norm rather than componentwise scaling.
    bias = experiment.derive_action_biases(model, episodes, selected, {"speed": vector}, warmup=1, reference_strength=reference_strength)["speed_bias"]  # Normalize only the comparator's parameterization.
    np.testing.assert_allclose(reference_strength * bias, -.125 * vector, rtol=1e-6)  # The maximum configured magnitude reproduces the fitted mean displacement.
    np.testing.assert_array_equal(predict(episodes[0]["observations"])[0], episodes[0]["observations"])  # The fitting hook must be removed after derivation.


@pytest.mark.parametrize("reference_strength", [0.0, -.1, float("nan"), float("inf")])
def test_action_bias_rejects_invalid_reference_before_prediction(reference_strength):
    """Local: reject undefined normalization; global: prevent nonfinite comparator artifacts."""
    with pytest.raises(ValueError, match="reference_strength"):
        experiment.derive_action_biases(None, [], [], {}, warmup=1, reference_strength=reference_strength)  # Invalid input must fail before any model or data access.

"""Local: test frozen selection; global: prevent confirmation-driven retuning.

Source: https://docs.pytest.org/en/stable/how-to/assert.html
"""
from rl_locomotion_steering_vectors.experiment import select_conditions, successful_candidates


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

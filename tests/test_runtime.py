"""Local: test rollout mechanics. Global: protect causal comparisons from runtime artifacts.

Sources: https://docs.pytest.org/en/stable/how-to/fixtures.html and
https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/sac.html.
"""

from types import SimpleNamespace  # Local: construct minimal contact fixtures; global: isolate measurement logic.

import gymnasium as gym  # Local: create the real simulator; global: test the same environment used in research.
import numpy as np  # Local: express expected physics; global: retain exact array comparisons.
import pytest  # Local: define reusable fixtures and failures; global: drive test-first implementation.
from stable_baselines3 import SAC  # Local: build an untrained actor; global: verify real SB3 inference integration.

from rl_locomotion_steering_vectors.runtime import (  # Local: import the runtime contract; global: test public behavior.
    METRIC_NAMES, ActivationIntervention, bad_ground_contact, quaternion_yaw,
    run_episode, state_hash, summarise_episode, wrapped_angle_difference,
)


def test_quaternion_yaw_uses_all_components_and_wraps():
    """Local: verify scalar-first quaternion math. Global: prevent interpreting Ant quaternion entries as angles."""
    angle = 0.7  # Local: choose a known rotation; global: provide an analytic oracle.
    quaternion = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])  # Local: encode yaw; global: match MuJoCo ordering.
    assert quaternion_yaw(quaternion) == pytest.approx(angle)  # Local: recover yaw; global: establish correct turning measurements.
    assert quaternion_yaw(-quaternion) == pytest.approx(angle)  # Local: negate equivalent quaternion; global: prevent sign discontinuities.
    assert wrapped_angle_difference(-np.pi + 0.1, np.pi - 0.1) == pytest.approx(0.2)  # Local: cross the branch cut; global: avoid spurious yaw rates.
    with pytest.raises(ValueError):  # Local: exercise invalid input; global: do not silently emit invented orientations.
        quaternion_yaw(np.zeros(4))  # Local: pass a nonrotation; global: require explicit failure.


def test_bad_contact_requires_floor_and_actual_penetration():
    """Local: distinguish torso-floor contact from allowed feet. Global: measure locomotion failure correctly."""
    pairs = [(0, 1, -0.01), (0, 2, 0.005), (1, 2, -0.02)]  # Local: vary geometry and gap; global: include false positives.
    data = SimpleNamespace(ncon=3, contact=[SimpleNamespace(geom1=a, geom2=b, dist=d) for a, b, d in pairs])  # Local: mimic MuJoCo contacts; global: isolate classifications.
    assert bad_ground_contact(data, 0, {1}) == 1.0  # Local: detect torso-floor penetration; global: flag inappropriate support.
    assert bad_ground_contact(data, 0, {2}) == 0.0  # Local: ignore positive-gap contact; global: avoid treating contact margins as impact.


def test_state_hash_covers_integration_state():
    """Local: compare complete simulator states. Global: prove paired intervention prefixes coincide."""
    with gym.make("HalfCheetah-v5") as env:  # Local: open real MuJoCo data; global: test its documented integration-state API.
        env.reset(seed=3)  # Local: establish reproducible reset; global: remove uncontrolled initial variation.
        before = state_hash(env.unwrapped)  # Local: hash the initial state; global: retain a comparison reference.
        assert state_hash(env.unwrapped) == before  # Local: repeat without mutation; global: require deterministic hashing.
        env.unwrapped.data.qacc_warmstart[0] += 0.01  # Local: alter solver memory; global: ensure hashing exceeds qpos/qvel.
        assert state_hash(env.unwrapped) != before  # Local: detect warmstart mutation; global: prevent incomplete equality claims.


@pytest.fixture(scope="module")  # Local: reuse one expensive actor; global: keep correctness checks bounded.
def actor():
    """Local: create real SB3 prediction machinery. Global: verify hooks without downloading research weights."""
    with gym.make("HalfCheetah-v5") as env:  # Local: obtain authentic spaces; global: avoid toy-shape compatibility claims.
        model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, seed=0, device="cpu")  # Local: initialize a correctly shaped actor; global: test supported architecture.
    model.policy.set_training_mode(False)  # Local: select deterministic layer behavior; global: protect repeated predictions.
    model.policy.requires_grad_(False)  # Local: freeze all parameters; global: enforce intervention-only changes.
    return model  # Local: expose the shared fixture; global: test through normal model.predict.


def test_activation_hook_handles_batch_identity_capture_and_cleanup(actor):
    """Local: test Baukit around real predictions. Global: ensure additive steering leaves the base policy intact."""
    obs = np.zeros((3, 17), dtype=np.float32)  # Local: create three observations; global: verify batched intervention broadcasting.
    base = actor.predict(obs, deterministic=True)[0]  # Local: save normal actions; global: establish zero-intervention identity.
    layer = actor.actor.latent_pi[1]  # Local: select the first ReLU; global: match the experimental intervention site.
    count = len(layer._forward_hooks)  # Local: record existing hooks; global: detect resource leaks without assuming none exist.
    with ActivationIntervention(layer, np.ones(256), capture=True) as intervention:  # Local: attach one managed edit; global: exercise library integration.
        intervention.alpha = 0.0  # Local: disable the offset; global: require exact baseline behavior.
        assert np.array_equal(actor.predict(obs, deterministic=True)[0], base)  # Local: compare exact actions; global: prohibit hidden baseline perturbations.
        captured = intervention.activation.copy()  # Local: save original layer values; global: verify capture occurs before editing.
        intervention.alpha = 0.2  # Local: activate a nonzero offset; global: exercise real downstream policy effects.
        altered = actor.predict(obs, deterministic=True)[0]  # Local: run the same batch; global: isolate the activation intervention.
        assert np.array_equal(intervention.activation, captured)  # Local: compare captured inputs; global: prevent measuring the edited values as baseline activations.
        assert not np.array_equal(altered, base)  # Local: observe a causal action change; global: detect silently ineffective hooks.
    assert len(layer._forward_hooks) == count  # Local: verify cleanup; global: prevent cross-condition contamination.
    assert np.array_equal(actor.predict(obs, deterministic=True)[0], base)  # Local: restore baseline prediction; global: show policy function recovery.


def test_hook_cleanup_on_exception(actor):
    """Local: fail inside an intervention context. Global: prevent failed experiments from contaminating later runs."""
    layer = actor.actor.latent_pi[1]  # Local: reuse the actual target; global: test operational cleanup.
    count = len(layer._forward_hooks)  # Local: preserve preexisting hooks; global: check only this context's effects.
    with pytest.raises(RuntimeError):  # Local: expect deliberate failure; global: verify exceptional control flow.
        with ActivationIntervention(layer, np.ones(256)):  # Local: install a real hook; global: exercise cleanup ownership.
            raise RuntimeError("intentional test failure")  # Local: interrupt the body; global: ensure exception cleanup is reliable.
    assert len(layer._forward_hooks) == count  # Local: count remaining hooks; global: reject leaked edits.


def test_real_rollouts_keep_identical_prefixes_and_frozen_weights(actor):
    """Local: run a short complete pipeline segment. Global: verify pairing, artifact shapes, and frozen parameters."""
    baseline = run_episode(actor, "halfcheetah", 7, max_steps=9, warmup=3, capture=True)  # Local: collect an unsteered reference; global: use the actual rollout interface.
    zero = run_episode(actor, "halfcheetah", 7, vector=np.ones(256), alpha=0, max_steps=9, warmup=3)  # Local: execute a disabled edit; global: test runtime zero identity.
    steered = run_episode(actor, "halfcheetah", 7, vector=np.ones(256), alpha=0.2, max_steps=9, warmup=3)  # Local: enable steering after startup; global: isolate the treatment window.
    assert np.array_equal(baseline["actions"], zero["actions"])  # Local: compare all actions; global: establish zero-control fidelity.
    assert baseline["onset_hash"] == steered["onset_hash"]  # Local: compare full simulator states; global: prove a matched intervention onset.
    assert np.array_equal(baseline["actions"][:3], steered["actions"][:3])  # Local: compare startup actions; global: ensure no premature steering.
    assert baseline["activations"].shape == (9, 256)  # Local: check stored activations; global: guarantee extraction-ready dimensions.
    assert not any(parameter.requires_grad for parameter in actor.policy.parameters())  # Local: inspect parameters; global: enforce frozen-policy research.


def test_summary_retains_early_failures_instead_of_dropping_them():
    """Local: summarize an episode ending before onset. Global: keep failures in aggregate denominators."""
    episode = {name: np.ones(2) for name in METRIC_NAMES}  # Local: represent two recorded transitions; global: test a realistic partial artifact.
    episode.update(length=np.array(2), terminated=np.array(True), truncated=np.array(False), dt=np.array(0.05))  # Local: attach failure metadata; global: preserve early termination semantics.
    summary = summarise_episode(episode, warmup=3)  # Local: request an unavailable post-startup segment; global: exercise explicit failure handling.
    assert summary["failure"] == 1.0  # Local: retain the failed episode; global: prevent survivor bias.
    assert summary["x_velocity"] == 0.0  # Local: use the documented empty-window value; global: avoid fabricated successful movement.
    assert summary["return"] == 2.0  # Local: retain original rewards; global: preserve the actual complete episode return.

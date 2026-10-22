"""Local: verify continuous-action score estimation; global: prevent incorrect PGS likelihoods and hidden learning.

Sources: https://stable-baselines3.readthedocs.io/en/v2.4.1/_modules/stable_baselines3/common/distributions.html
https://docs.pytest.org/en/stable/how-to/fixtures.html
"""
import json  # Verify that fitted provenance is directly serializable.
from pathlib import Path  # Resolve the cached pinned checkpoint inside the current project.

import gymnasium as gym  # Use real continuous-action simulator interfaces.
import numpy as np  # Keep independent numerical assertions explicit.
import pytest  # Parameterized failures must be rejected rather than silently filtered.
import torch  # Compare score gradients against fixed-action finite differences.
from stable_baselines3 import SAC  # Exercise the actual SB3 squashed Gaussian implementation.
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution  # Supply a source-library likelihood oracle.

from rl_locomotion_steering_vectors.pgs import collect_score_episode, episode_credits, fit_pgs_vectors, intervention_log_prob  # Test the standalone adapter before pipeline integration.
from rl_locomotion_steering_vectors.runtime import parameter_hash, prepare_model, run_episode  # Verify the unchanged physics engine and frozen policy.


@pytest.fixture(scope="module")
def model():
    """Local: initialize one real actor; global: test mechanics without claiming trained-policy performance."""
    torch.set_num_threads(1)  # Small test batches do not benefit from many CPU threads.
    with gym.make("HalfCheetah-v5") as env:  # Match the production observation/action spaces.
        policy = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, buffer_size=1, seed=0, device="cpu")  # Use one initialization and no training transitions.
    policy.policy.set_training_mode(False)  # Match frozen evaluation behavior.
    policy.policy.requires_grad_(False)  # Only the temporary activation offset may receive gradients.
    return policy  # Reuse the same initialized actor across bounded correctness tests.


def test_collection_preserves_prefix_and_actual_sample_likelihood(model):
    """Local: reconstruct every sampled action; global: keep score samples tied to the actual executed trajectory."""
    baseline = run_episode(model, "halfcheetah", 11, warmup=3, max_steps=9)  # The normal runtime defines the deterministic comparison prefix.
    sampled = collect_score_episode(model, "halfcheetah", 11, warmup=3, max_steps=9)  # The proxy changes only the post-prefix sampling mode.
    np.testing.assert_array_equal(sampled["actions"][:3], baseline["actions"][:3])  # PGS collection cannot change intervention onset.
    assert sampled["onset_hash"] == baseline["onset_hash"]  # Check complete simulator state, not merely visible position.
    np.testing.assert_array_equal(sampled["score_mask"], np.arange(9) >= 3)  # Deterministic prefix actions must never enter the likelihood score estimate.
    reconstructed = model.policy.unscale_action(torch.tanh(torch.as_tensor(sampled["gaussian_actions"])).numpy())  # Replay the exact SB3 Gaussian-to-environment action transform.
    np.testing.assert_array_equal(reconstructed, sampled["actions"])  # Clipped or resampled substitutes cannot pass this check.
    logp = intervention_log_prob(model, sampled["observations"], sampled["gaussian_actions"], sampled["squashed_actions"], torch.zeros(256))  # Reevaluate fixed recorded actions at the same zero offset.
    np.testing.assert_allclose(logp.detach().numpy(), sampled["sampled_log_prob"], rtol=2e-5, atol=2e-4)  # Batched and single-observation floating-point paths agree within explicit tolerance.
    assert not np.array_equal(sampled["actions"][3:], baseline["actions"][3:])  # Later transitions must actually use stochastic policy sampling.


def test_score_gradient_matches_finite_difference_and_detaches_samples(model):
    """Local: differentiate a fixed sampled action; global: exclude the rsample path from the policy score."""
    episode = collect_score_episode(model, "halfcheetah", 12, warmup=1, max_steps=5)  # Keep valid observations and exact on-policy samples together.
    gaussian = torch.tensor(episode["gaussian_actions"][1:], requires_grad=True)  # Deliberately provide a differentiable input to detect accidental sample gradients.
    squashed = torch.tanh(gaussian)  # This path must be detached along with the recorded Gaussian.
    offset = torch.zeros(256, requires_grad=True)  # Only this activation addition is an optimization variable.
    direction = torch.as_tensor(np.random.default_rng(2).normal(size=256), dtype=torch.float32)  # An independent direction checks the full hidden offset rather than one chosen component.
    direction /= torch.linalg.vector_norm(direction)  # Keep the finite-difference step well scaled.
    observations = episode["observations"][1:]  # Prefix likelihoods never participate in PGS fitting.
    value = intervention_log_prob(model, observations, gaussian, squashed, offset).sum()  # Score fixed likelihoods under a shared additive offset.
    gradient, sample_gradient = torch.autograd.grad(value, (offset, gaussian), allow_unused=True)  # Ask explicitly whether an unwanted sample path survives.
    epsilon = .002  # A small float32 step balances truncation and rounding error.
    plus = intervention_log_prob(model, observations, gaussian, squashed, epsilon * direction).sum()  # Perturb only the hidden intervention.
    minus = intervention_log_prob(model, observations, gaussian, squashed, -epsilon * direction).sum()  # Hold observations and sampled actions exactly fixed.
    assert float(torch.dot(gradient, direction)) == pytest.approx(float((plus - minus) / (2 * epsilon)), rel=.03, abs=.003)  # An incorrect likelihood gradient fails this external numerical check.
    assert sample_gradient is None  # REINFORCE requires score-function differentiation, not differentiating through rsample.
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.policy.parameters())  # No parameter receives or accumulates a policy update.


def test_saturated_actions_use_saved_gaussian_and_hooks_cleanup_on_error(model):
    """Local: use exact tanh saturation; global: prevent inverse-tanh clipping from corrupting scores."""
    observations = np.zeros((2, 17), dtype=np.float32)  # A fixed physical-input shape isolates likelihood arithmetic.
    gaussian = torch.tensor([[12.] * 6, [-12.] * 6])  # These values saturate float32 tanh to exactly +/-1.
    squashed = torch.tanh(gaussian)  # The executed action alone cannot recover the original Gaussian sample.
    count = len(model.actor.latent_pi[1]._forward_hooks)  # Existing hooks must survive exceptional cleanup unchanged.
    before = parameter_hash(model)  # Protect actor, critic, and policy buffers alike.
    offset = torch.zeros(256, requires_grad=True)  # Score the baseline distribution at the required zero intervention.
    actual = intervention_log_prob(model, observations, gaussian, squashed, offset)  # Explicit Gaussian samples avoid inverse-tanh reconstruction.
    obs_tensor, _ = model.policy.obs_to_tensor(observations)  # Reuse the actor's normal preprocessing.
    mean, log_std, _ = model.actor.get_action_dist_params(obs_tensor)  # Obtain the same unsteered Gaussian parameters.
    distribution = SquashedDiagGaussianDistribution(6).proba_distribution(mean, log_std)  # Build an independent SB3 oracle.
    torch.testing.assert_close(actual, distribution.log_prob(squashed, gaussian_actions=gaussian))  # Match the library's explicitly supplied Gaussian likelihood.
    assert not torch.allclose(actual, distribution.log_prob(squashed))  # Omitting the actual Gaussian must demonstrably change the saturated-sample likelihood.
    assert torch.isfinite(torch.autograd.grad(actual.sum(), offset)[0]).all()  # Saturation must not cause nonfinite intervention gradients.
    with pytest.raises((ValueError, RuntimeError)):
        intervention_log_prob(model, observations, torch.zeros((2, 7)), squashed, offset)  # Trigger an incompatible action dimension inside SB3 log_prob while the Baukit hook is installed.
    assert len(model.actor.latent_pi[1]._forward_hooks) == count  # Exceptions cannot leave steering attached to the base actor.
    assert parameter_hash(model) == before  # Likelihood evaluation performs no training or buffer updates.


def test_episode_credits_keep_boundaries_and_early_failures():
    """Local: compare hand-computed returns; global: keep fixed-horizon rewards and independent episode baselines."""
    episodes = [{"x_velocity": np.array(values)} for values in ([1000., 3., 6.], [1000., 9.], [1000.])]  # The third episode fails before any stochastic action and still belongs to the sample.
    returns, advantages = episode_credits(episodes, warmup=1, max_steps=4)  # Reward is velocity divided by the three-step post-prefix horizon.
    np.testing.assert_allclose(returns, [[3., 2., 0.], [3., 0., 0.], [0., 0., 0.]])  # Reverse accumulation must neither include startup nor spill into the next episode.
    np.testing.assert_allclose(advantages, [[1.5, 2., 0.], [1.5, -1., 0.], [-3., -1., 0.]])  # Each baseline uses the other two complete episodes, including padded zeros after termination.


def test_fitting_retains_unhealthy_episodes_and_freezes_numerical_provenance(model):
    """Local: fit a tiny complete family; global: check score extraction, controls, normalization, and resumption identity."""
    episodes = [collect_score_episode(model, "halfcheetah", seed, warmup=1, max_steps=6) for seed in range(20, 28)]  # Distinct stochastic episodes supply independent return credits.
    for episode in episodes:
        episode["inversion"][:] = 1  # Simulate unfavorable outcomes without changing the policy likelihood evidence.
        episode["bad_contact"][:] = 1  # PGS must retain these observations instead of selecting survivors.
    before = parameter_hash(model)  # Both fitted candidates and controls must leave the base policy frozen.
    vectors, diagnostics = fit_pgs_vectors(model, episodes, warmup=1, max_steps=6, batch_size=3)  # Small chunks exercise credit accumulation across boundaries.
    assert set(vectors) == {"speed", "speed_random0", "speed_random1", "speed_random2", "speed_shuffled"}  # Preserve the registered control family for the existing evaluator.
    assert diagnostics["episodes"] == diagnostics["unhealthy_episodes_retained"] == 8  # Failures and contact outcomes are counted, never filtered from the score estimate.
    for vector in vectors.values():
        assert np.linalg.norm(vector) == pytest.approx(diagnostics["activation_rms"], rel=1e-6)  # Candidates and controls share the measured centered activation norm.
    json.dumps(diagnostics, allow_nan=False)  # Complete provenance must be directly publishable as strict JSON.
    assert parameter_hash(model) == before  # Extraction changes no policy parameters or buffers.
    assert all(parameter.grad is None for parameter in model.policy.parameters())  # Autograd only differentiates the temporary intervention.
    episodes[0]["actions"][2, 0] += .02  # Corrupt the executed-action record while preserving the Gaussian sample and source manifest.
    with pytest.raises(ValueError, match="action"):
        fit_pgs_vectors(model, episodes, warmup=1, max_steps=6)  # Reject inconsistent likelihood/action evidence before computing a new vector.


def test_pinned_tqc_actor_preserves_recorded_likelihood_and_frozen_score():
    """Local: exercise the actual research checkpoint; global: verify TQC uses the same supported score path as SAC."""
    policy, metadata = prepare_model("halfcheetah", Path.cwd())  # Use the already cached and SHA256-verified upstream RL checkpoint.
    episode = collect_score_episode(policy, "halfcheetah", 29, warmup=3, max_steps=9)  # This short software check is separate from research fitting/evaluation seeds.
    offset = torch.zeros(256, requires_grad=True)  # The checkpoint remains frozen while a local intervention graph is differentiated.
    likelihood = intervention_log_prob(policy, episode["observations"][3:], episode["gaussian_actions"][3:], episode["squashed_actions"][3:], offset)  # Score the actual stochastic post-prefix draws.
    np.testing.assert_allclose(likelihood.detach().numpy(), episode["sampled_log_prob"][3:], rtol=2e-5, atol=2e-4)  # Production TQC likelihood replay must match collection as closely as the SAC mechanism tests.
    gradient, = torch.autograd.grad(likelihood.sum(), offset)  # Request only the first-ReLU additive gradient.
    assert torch.isfinite(gradient).all() and torch.linalg.vector_norm(gradient) > 0  # The checkpoint exposes a usable finite score, without making a behavioral-effect claim.
    assert parameter_hash(policy) == metadata["parameter_sha256"]  # Verify complete base-state identity after collection and gradient evaluation.


@pytest.mark.parametrize("key", ["pgs_policy_sha256", "pgs_source_sha256", "pgs_warmup", "pgs_max_steps"])
def test_fitting_rejects_incompatible_collection_identity(model, key):
    """Local: alter one provenance field; global: reject stale or off-policy evidence before likelihood fitting."""
    episodes = [collect_score_episode(model, "halfcheetah", seed, warmup=1, max_steps=3) for seed in (31, 32)]  # Distinct short episodes provide a valid surrounding sampling contract.
    episodes[0][key] = np.array("different" if "sha256" in key else 9)  # Change one identity component while preserving all sampled action bytes.
    with pytest.raises(ValueError, match="identity"):
        fit_pgs_vectors(model, episodes, warmup=1, max_steps=3)  # A new policy, collector, or schedule cannot inherit existing score evidence.

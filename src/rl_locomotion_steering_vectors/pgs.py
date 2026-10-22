"""Local: continuous-action Policy Gradient Steering; global: estimate one frozen-policy activation direction.

Adapted from the authors' MIT-licensed pgs_direction and reward_matrix at commit
8b733320735751e21ed0f95a39ad2b62706d247f; attribution and license: docs/pgs-adaptation.md.
Sources: https://github.com/Xmaster6y/policy-gradient-steering/tree/8b733320735751e21ed0f95a39ad2b62706d247f
https://stable-baselines3.readthedocs.io/en/v2.4.1/_modules/stable_baselines3/common/distributions.html
https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/rl_intro3.rst
https://numpy.org/doc/1.26/reference/generated/numpy.cumsum.html
"""
import hashlib  # Bind saved likelihood evidence to its policy and collection implementation.
import logging  # Send complete extraction diagnostics to the existing timestamped handlers.
from pathlib import Path  # Read only this implementation and the reused runtime for provenance.

import numpy as np  # Keep sampled actions and vectors in non-executable array formats.
import torch  # Differentiate an activation offset while keeping sampled actions and parameters fixed.
from baukit import Trace  # Reuse automatic hook ownership for differentiable output additions.
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution  # Reuse SB3's exact likelihood and squash correction.

from .runtime import actor_layer, parameter_hash, run_episode  # Reuse all simulator mechanics and frozen-policy checks unchanged.

PGS_ID = hashlib.sha256((Path(__file__).read_text(encoding="utf-8") + Path(__file__).with_name("runtime.py").read_text(encoding="utf-8")).encode("utf-8")).hexdigest()  # Freeze normalized collection/extraction source identity when the process imports this module.


def _check_actor(model) -> None:
    """Local: verify supported stochastic inference; global: reject silently different likelihood models."""
    actor_layer(model)  # Require the existing two-256-ReLU architecture and first-ReLU intervention site.
    if model.actor.use_sde or not isinstance(model.actor.action_dist, SquashedDiagGaussianDistribution):  # This adaptation covers independent diagonal Gaussian noise, not gSDE.
        raise ValueError("PGS requires SB3 SquashedDiagGaussianDistribution with use_sde=False")  # A different sampler needs its own likelihood adaptation.
    if any(parameter.requires_grad for parameter in model.policy.parameters()):  # Parameter gradients are forbidden even when no optimizer is constructed.
        raise ValueError("PGS requires all base policy parameters frozen")  # Fail before collecting or scoring policy-learning data.
    if model.policy.training:  # Dropout or mutable training buffers would invalidate collection/scoring agreement.
        raise ValueError("PGS requires policy evaluation mode")  # Preserve the already-frozen inference function.


class _SampleRecorder:
    """Local: proxy SB3 predictions and save actual samples; global: reuse the existing physics loop intact."""

    def __init__(self, model, warmup: int):
        """Local: configure a single episode; global: prevent sample counters leaking across resets."""
        self.model, self.warmup = model, warmup  # Forward to the original SB3 model and fix the deterministic prefix length.
        self.gaussian, self.squashed, self.log_prob = [], [], []  # Capture each distribution result before the next prediction overwrites it.

    def __getattr__(self, name):
        """Local: forward the model interface; global: retain runtime architecture, spaces, and hashing checks."""
        return getattr(self.model, name)  # The proxy changes prediction scheduling only.

    def predict(self, observation, *args, deterministic=False, **kwargs):
        """Local: preserve the actual SB3 action draw; global: never substitute clipped actions or deterministic means for scores."""
        prefix = len(self.gaussian) < self.warmup  # Runtime calls predict exactly once per transition.
        action, state = self.model.predict(observation, *args, deterministic=prefix, **kwargs)  # Ordinary SB3 inference draws stochastic actions after the identical prefix.
        distribution = self.model.actor.action_dist  # Read the same distribution that produced this action immediately.
        gaussian = distribution.gaussian_actions.detach().clone()  # Break any rsample path and preserve the exact unsquashed sample.
        squashed = torch.tanh(gaussian)  # Reconstruct the bounded policy action before environment scaling.
        if gaussian.shape != (1, *self.model.action_space.shape):  # The reused runtime owns one simulator per episode.
            raise ValueError("PGS recorder expects a single environment action")  # Reject unsupported vectorized sampling instead of misaligning timesteps.
        reconstructed = self.model.policy.unscale_action(squashed.cpu().numpy())[0]  # Use SB3's actual affine mapping to environment bounds.
        if not np.array_equal(reconstructed, action):  # Correct likelihoods require the recorded sample to explain the executed action.
            raise ValueError("Recorded Gaussian sample does not reconstruct the executed action")  # Stop before accepting an invalid score/action pair.
        self.gaussian.append(gaussian[0].cpu().numpy().copy())  # Keep one row per runtime transition, including explicitly masked prefix rows.
        self.squashed.append(squashed[0].cpu().numpy().copy())  # Save exact tanh outputs even when float32 saturation loses invertibility.
        self.log_prob.append(float(distribution.log_prob(squashed, gaussian_actions=gaussian)[0]))  # Store the contemporaneous normalized-action likelihood without inverse tanh.
        return action, state  # The unchanged runtime applies this exact action to the simulator.


def collect_score_episode(model, model_key: str, seed: int, *, warmup: int = 100, max_steps: int = 1000) -> dict:
    """Local: record one unsteered on-policy episode; global: collect fresh evidence for a zero-offset gradient."""
    _check_actor(model)  # Verify supported likelihoods and frozen parameters before any simulation.
    if warmup < 0 or max_steps <= warmup:  # The finite-horizon objective needs a positive post-prefix duration.
        raise ValueError("PGS requires max_steps > warmup >= 0")  # Reject meaningless schedules before recording artifacts.
    before = parameter_hash(model)  # Bind every sample to the complete frozen policy state.
    recorder = _SampleRecorder(model, warmup)  # One proxy owns one reset and sampling schedule.
    episode = run_episode(recorder, model_key, seed, warmup=warmup, max_steps=max_steps, capture=True)  # Reuse integration, metrics, state hashes, deterministic seeding, and exception cleanup without copying the physics loop.
    episode.update(gaussian_actions=np.asarray(recorder.gaussian), squashed_actions=np.asarray(recorder.squashed),
                   sampled_log_prob=np.asarray(recorder.log_prob), score_mask=np.arange(int(episode["length"])) >= warmup)  # Prefix mode actions are retained for auditing but excluded from stochastic score estimation.
    episode.update(pgs_policy_sha256=np.array(before), pgs_source_sha256=np.array(PGS_ID), pgs_warmup=np.array(warmup), pgs_max_steps=np.array(max_steps))  # Non-executable scalar provenance prevents incompatible cache reuse.
    logging.info("PGS collected seed=%s stochastic_steps=%s policy_sha256=%s source_sha256=%s", seed, int(episode["score_mask"].sum()), before, PGS_ID)  # Make the proxy's mixed deterministic/stochastic schedule explicit in logs.
    return episode  # Orchestration may atomically save this complete numeric/Unicode dictionary with the existing storage module.


def intervention_log_prob(model, observations, gaussian_actions, squashed_actions, offset: torch.Tensor) -> torch.Tensor:
    """Local: score detached recorded actions under an offset; global: implement a likelihood gradient, never a mean-action or rsample gradient."""
    _check_actor(model)  # The likelihood must describe the same supported frozen actor used at collection.
    if offset.shape != (256,) or not torch.isfinite(offset).all():  # Reject unintended broadcasting and undefined interventions.
        raise ValueError("PGS offset must be a finite 256-vector")  # Make the first-ReLU contract explicit.
    observation_tensor, _ = model.policy.obs_to_tensor(observations)  # Preserve SB3 observation preprocessing and device placement.
    gaussian = torch.as_tensor(gaussian_actions, device=observation_tensor.device, dtype=torch.float32).detach()  # Treat actual unsquashed samples as fixed score-function inputs.
    squashed = torch.as_tensor(squashed_actions, device=observation_tensor.device, dtype=torch.float32).detach()  # Detach the tanh Jacobian correction as well as the Gaussian sample.
    with Trace(actor_layer(model), retain_output=False, edit_output=lambda output: output + offset):  # Baukit removes the differentiable addition even if actor evaluation raises.
        mean, log_std, _ = model.actor.get_action_dist_params(observation_tensor)  # Differentiate the actual downstream policy mean and scale through the first-ReLU addition.
        distribution = SquashedDiagGaussianDistribution(model.action_space.shape[0], epsilon=model.actor.action_dist.epsilon).proba_distribution(mean, log_std)  # A separate SB3 object avoids replacing the recorder's most recent sample.
        likelihood = distribution.log_prob(squashed, gaussian_actions=gaussian)  # Passing the actual Gaussian avoids inverse-tanh clipping, including at exact action saturation.
    return likelihood  # The graph remains differentiable after hook removal; base parameters remain frozen.


def episode_credits(episodes: list[dict], *, warmup: int = 100, max_steps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Local: compute fixed-horizon return-to-go and independent episode baselines; global: retain failed trajectories without survivor selection."""
    horizon, count = max_steps - warmup, len(episodes)  # Reward normalization uses the registered horizon, never the realized surviving length.
    if warmup < 0 or horizon <= 0 or count < 2:  # A leave-one-episode-out baseline needs another independent rollout.
        raise ValueError("PGS credits need at least two episodes and a positive post-prefix horizon")  # Reject undefined estimates explicitly.
    rewards = np.zeros((count, horizon), dtype=np.float64)  # Absorbed timesteps after termination have zero future reward.
    for index, episode in enumerate(episodes):
        velocity = np.asarray(episode["x_velocity"], dtype=np.float64)[warmup:]  # Use measured post-prefix speed, excluding upstream reward and startup motion.
        if len(velocity) > horizon or not np.isfinite(velocity).all():  # Do not silently crop or remove invalid physical outcomes.
            raise ValueError("PGS episode speed exceeds its horizon or is nonfinite")  # Invalid records need correction rather than selective omission.
        rewards[index, :len(velocity)] = velocity / horizon  # Default reward is vx/900, so episode return is fixed-horizon average forward speed.
    returns = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1].copy()  # Adapt the authors' reverse reward recurrence at gamma=1 without crossing episode boundaries.
    baseline = (returns.sum(axis=0, keepdims=True) - returns) / (count - 1)  # Other episodes provide a per-time baseline independent of this episode's sampled action.
    return returns, returns - baseline  # Full padded matrices retain early failures in the episode denominator.


def _validate_episodes(model, episodes: list[dict], warmup: int, max_steps: int) -> dict:
    """Local: verify immutable sampling evidence; global: prohibit off-policy or inconsistent likelihood reconstruction."""
    identity = parameter_hash(model)  # Compare every episode against one current frozen policy identity.
    seeds = [int(episode["seed"]) for episode in episodes]  # Distinct seeds define the independent episode units.
    if len(seeds) != len(set(seeds)):
        raise ValueError("PGS fitting requires distinct episode seeds")  # Duplicating trajectories would distort both the baseline and uncertainty diagnostics.
    hashes = {}  # Bind exact trajectory inputs used in this fitted gradient.
    for episode, seed in zip(episodes, seeds, strict=True):
        expected = {"pgs_policy_sha256": identity, "pgs_source_sha256": PGS_ID, "pgs_warmup": warmup, "pgs_max_steps": max_steps}  # Cached samples belong to one policy, collector, and schedule.
        if any(episode[key].item() != value for key, value in expected.items()):  # Numeric and Unicode arrays become comparable Python scalars.
            raise ValueError("PGS collection identity differs from this policy, source, or schedule")  # Never reinterpret stale likelihood data as current on-policy samples.
        length = int(episode["length"])  # Real episode length includes startup and any early failure.
        if length > max_steps or not np.array_equal(episode["score_mask"], np.arange(length) >= warmup):  # Prefix means must not enter the stochastic score estimator.
            raise ValueError("PGS score mask or episode length is inconsistent")  # Preserve the intended causal onset boundary.
        for key in ("observations", "actions", "activations", "gaussian_actions", "squashed_actions", "sampled_log_prob", "x_velocity"):
            if len(episode[key]) != length or not np.isfinite(episode[key]).all():  # Validate every used row instead of selecting only favorable observations.
                raise ValueError(f"PGS {key} is nonfinite or misaligned")  # Fail loudly on malformed trajectory evidence.
        squashed = torch.tanh(torch.as_tensor(episode["gaussian_actions"])).numpy()  # Reconstruct using the exact SB3 Torch tanh transform.
        if not np.array_equal(squashed, episode["squashed_actions"]) or not np.array_equal(model.policy.unscale_action(squashed), episode["actions"]):  # Detect clipped, replaced, or misaligned action records.
            raise ValueError("PGS recorded action does not match its sampled Gaussian")  # Correct action likelihoods require the action that actually drove these transitions.
        digest = hashlib.sha256()  # Construct a stable content identity independently of NPZ compression metadata.
        for key in ("observations", "actions", "activations", "gaussian_actions", "squashed_actions", "sampled_log_prob", "x_velocity", "score_mask"):
            value = np.asarray(episode[key])  # Preserve full numerical precision and shape.
            digest.update(f"{key}:{value.dtype}:{value.shape}".encode())  # Bind the interpretation of each byte sequence as well as its contents.
            digest.update(value.tobytes())  # Hash all score-relevant trajectory values in C order.
        hashes[str(seed)] = digest.hexdigest()  # Keep individual episode identities available for reproducibility audits.
    return {"policy_sha256": identity, "source_sha256": PGS_ID, "episode_sha256": hashes}  # The caller includes these exact identities in its fitted-vector manifest.


def _score_direction(model, episodes: list[dict], credits: np.ndarray, warmup: int, batch_size: int) -> tuple[np.ndarray, float]:
    """Local: average reward-weighted per-episode likelihood gradients; global: adapt the authors' PGS estimator to continuous actions."""
    direction, maximum_error = np.zeros(256, dtype=np.float64), 0.0  # Accumulate numerically stable sums and likelihood replay diagnostics.
    for index, episode in enumerate(episodes):
        for start in range(warmup, int(episode["length"]), batch_size):  # Prefix-only failures contribute zero gradient while remaining in the denominator.
            end = min(start + batch_size, int(episode["length"]))  # Chunking bounds graph memory without crossing episode credit boundaries.
            offset = torch.zeros(256, device=model.device, requires_grad=True)  # Every estimate is evaluated at the same unsteered policy, with no optimizer or iterative offset update.
            likelihood = intervention_log_prob(model, episode["observations"][start:end], episode["gaussian_actions"][start:end], episode["squashed_actions"][start:end], offset)  # Differentiate the fixed actual sampled actions.
            saved = np.asarray(episode["sampled_log_prob"])[start:end]  # Compare against likelihoods recorded immediately after sampling.
            if not np.allclose(likelihood.detach().cpu().numpy(), saved, rtol=2e-5, atol=2e-4):  # Batched inference permits only explicit small numerical differences.
                raise ValueError("PGS replay likelihood differs from the sampled policy likelihood")  # Reject an incompatible distribution before accepting its gradient.
            maximum_error = max(maximum_error, float(np.max(np.abs(likelihood.detach().cpu().numpy() - saved))))  # Report actual reconstruction error rather than assuming exact batched arithmetic.
            weights = torch.as_tensor(credits[index, start - warmup:end - warmup], device=model.device, dtype=torch.float64)  # Align each action with its own future-return credit.
            gradient, = torch.autograd.grad((likelihood.double() * weights).sum(), offset)  # Reuse the authors' return-times-grad-logp estimator; only the offset is differentiated.
            direction += gradient.detach().cpu().numpy() / len(episodes)  # Sum timesteps within episodes, then give every complete episode equal weight.
    return direction, maximum_error  # This is one local ascent direction, not an updated policy or a fitted action predictor.


def fit_pgs_vectors(model, episodes: list[dict], *, warmup: int = 100, max_steps: int = 1000, seed: int = 0, batch_size: int = 256) -> tuple[dict, dict]:
    """Local: fit speed-gradient and matched controls; global: create a standalone candidate for fresh causal validation."""
    _check_actor(model)  # Freeze the policy/distribution contract before inspecting any outcomes.
    if batch_size <= 0:
        raise ValueError("PGS batch_size must be positive")  # A chunk loop must make progress.
    provenance = _validate_episodes(model, episodes, warmup, max_steps)  # Cached sampling identities and executed actions must match exactly.
    returns, credits = episode_credits(episodes, warmup=warmup, max_steps=max_steps)  # Keep all episodes, including early terminations and sustained unhealthy movement.
    observed = [torch.as_tensor(episode["activations"][warmup:], dtype=torch.float64) for episode in episodes if int(episode["length"]) > warmup]  # Only nonexistent post-prefix activations are absent from the normalization scale.
    center = torch.stack([values.mean(dim=0) for values in observed]).mean(dim=0) if observed else torch.zeros(256, dtype=torch.float64)  # Balance episodes rather than their different realized lengths.
    scale = float(torch.sqrt(torch.stack([(values - center).square().sum(dim=1).mean() for values in observed]).mean())) if observed else 0.0  # Use centered RMS over every observed post-prefix activation without a health or speed filter.
    diagnostics = {"provenance": provenance, "seed": seed, "warmup": warmup, "max_steps": max_steps, "batch_size": batch_size, "gamma": 1.0,
                   "authors_revision": "8b733320735751e21ed0f95a39ad2b62706d247f", "episodes": len(episodes), "stochastic_steps": sum(max(0, int(episode["length"]) - warmup) for episode in episodes),
                   "unhealthy_episodes_retained": sum(bool(episode.get("terminated", False)) or int(episode["length"]) <= warmup or any(np.mean(episode[key][warmup:]) > .05 for key in ("inversion", "bad_contact")) for episode in episodes),
                   "activation_rms": scale, "rms_convention": "episode-balanced all observed post-prefix timestep RMS about episode-balanced mean; no outcome filtering",
                   "objective": f"sum of x_velocity/{max_steps - warmup} after {warmup} deterministic steps; zero rewards after termination; gamma=1",
                   "baseline": "leave-one-episode-out mean return-to-go at each time; failures remain in denominator", "mean_objective": float(returns[:, 0].mean()),
                   "behaviors": {"speed": {"status": "fitted", "method": "policy_gradient_score"}}, "vectors": {}}  # Preserve a JSON-compatible extraction schema for the existing reporting/evaluation phases.
    raw, likelihood_error = _score_direction(model, episodes, credits, warmup, batch_size)  # Estimate the single zero-offset speed-ascent direction.
    rng = np.random.default_rng(seed)  # Controls are fixed before validation or confirmation data exist.
    permutation = rng.permutation(len(episodes))  # Uniformly shuffle whole credit trajectories; record possible fixed points instead of selectively rejecting them.
    shuffled, shuffled_error = _score_direction(model, episodes, credits[permutation], warmup, batch_size)  # Break credit-to-episode assignment while preserving time structure and the same likelihood scores.
    directions = {"speed": (raw, "policy_gradient_score"), "speed_shuffled": (shuffled, "shuffled_episode_credits")}  # Label the method distinctly from classic high-minus-low activation contrasts.
    directions.update({f"speed_random{index}": (rng.normal(size=256), "random") for index in range(3)})  # Give three independent random directions exactly the same activation norm.
    diagnostics.update(shuffle_permutation=permutation.tolist(), maximum_likelihood_replay_error=max(likelihood_error, shuffled_error))  # Retain control randomization and measured numerical agreement.
    vectors = {}  # Reject incomplete or numerically meaningless families explicitly.
    for name, (direction, method) in directions.items():
        norm = float(np.linalg.norm(direction))  # Raw gradient magnitude remains visible even though all saved interventions share an activation scale.
        if not np.isfinite(norm) or norm <= 1e-12 or not np.isfinite(scale) or scale <= 1e-12:
            diagnostics["behaviors"]["speed"] = {"status": "skipped", "reason": f"nonfinite or negligible {name} norm or activation RMS"}  # Scaling cannot create information absent from the estimated score.
            vectors = {}  # A missing shuffled control must not silently weaken the registered experiment.
            break  # Preserve the explicit null result for a separately motivated next hypothesis.
        vectors[name] = (direction * (scale / norm)).astype(np.float32)  # Match runtime activation dtype while retaining full-precision raw vectors below.
        diagnostics["vectors"][name] = {"behavior": "speed", "method": method, "raw_vector": direction.tolist(), "raw_norm": norm, "norm": scale, "scale_factor": scale / norm}  # Save exact normalization and source estimator metadata alongside every direction.
    if parameter_hash(model) != provenance["policy_sha256"]:  # Cover policy buffers as well as trainable tensors after both gradient estimates.
        raise RuntimeError("PGS changed the frozen base policy")  # Never publish a steering result produced by hidden parameter learning.
    logging.info("PGS extraction %s", diagnostics)  # Retain complete numerical diagnostics and raw directions in the existing realtime log streams.
    return vectors, diagnostics  # Fresh validation and confirmation remain the caller's responsibility.

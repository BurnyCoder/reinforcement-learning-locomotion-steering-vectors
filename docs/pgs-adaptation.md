# Continuous-action Policy Gradient Steering adapter

This is a prepared fallback method, implemented in `src/rl_locomotion_steering_vectors/pgs.py`. Its software checks are separate from research experiments. The recorded tests support likelihood and derivative correctness on their tested cases; they do not prove correctness for all inputs or establish a useful locomotion vector. Those model-dependent tests were not rerun during the [saved-data fact-check](../reports/fact-check-2026-10-29/report.md). The existing classic-vector and cross-behavior experiments remain separate attempts.

The public entry points are `collect_score_episode(model, model_key, seed, *, warmup=100, max_steps=1000)` and `fit_pgs_vectors(model, episodes, *, warmup=100, max_steps=1000, seed=0, batch_size=256)`; arguments after `*` are keyword-only. The first returns the existing runtime episode dictionary plus sampling evidence. The second returns the existing `(vectors, diagnostics)` interface, with `speed`, `speed_random0`, `speed_random1`, `speed_random2`, and `speed_shuffled` directions when all norms are usable. Pipeline integration and a separately registered fresh-seed experiment remain the orchestrator's responsibility. No training or behavioral cloning is performed.

## Reused sources and the necessary adaptation

The authors' repository is pinned to commit [`8b733320735751e21ed0f95a39ad2b62706d247f`](https://github.com/Xmaster6y/policy-gradient-steering/tree/8b733320735751e21ed0f95a39ad2b62706d247f). Actual inspected source functions were:

| Source | Reused idea or implementation |
|---|---|
| [`gridworld/core.py::pgs_direction`](https://github.com/Xmaster6y/policy-gradient-steering/blob/8b733320735751e21ed0f95a39ad2b62706d247f/src/policy_gradient_steering/gridworld/core.py#L784) | Sum reward-weighted gradients of sampled-action log likelihoods at an activation site; normalize the resulting direction. |
| [`football/core.py::reward_matrix`](https://github.com/Xmaster6y/policy-gradient-steering/blob/8b733320735751e21ed0f95a39ad2b62706d247f/src/policy_gradient_steering/football/core.py#L611) | Reverse-time reward-to-go accumulation, adapted to the explicit undiscounted velocity objective. |
| [`football/core.py::_torch_intervention`, `_torch_hooks`, and `fit_intervention`](https://github.com/Xmaster6y/policy-gradient-steering/blob/8b733320735751e21ed0f95a39ad2b62706d247f/src/policy_gradient_steering/football/core.py#L1146) | Treat additive hidden-state interventions as the differentiated variables while retaining the underlying policy. Existing project Baukit hooks provide this mechanism. |
| [SB3 2.4.1 `SquashedDiagGaussianDistribution`](https://stable-baselines3.readthedocs.io/en/v2.4.1/_modules/stable_baselines3/common/distributions.html#SquashedDiagGaussianDistribution) | Actual Gaussian sampling, stored `gaussian_actions`, tanh squashing, and `log_prob(actions, gaussian_actions=...)`. This code is imported directly rather than reimplemented. |
| [SB3-Contrib 2.4.0 TQC `Actor.get_action_dist_params`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.4.0/sb3_contrib/tqc/policies.py#L146) | Original actor feature preprocessing, downstream mean/log-standard-deviation computation, and log-standard-deviation clamping. |
| [OpenAI Spinning Up policy-gradient derivation and implementation](https://github.com/openai/spinningup/blob/master/docs/spinningup/rl_intro3.rst) | Reward-to-go, action-independent baselines, equal episode averages, and the requirement that score-estimation data come from the policy at the differentiated parameters. |
| [Baukit `Trace`, pinned commit](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py) | Temporary differentiable first-ReLU output addition with automatic cleanup. |

The authors' released tasks use discrete-action policies and task-specific runners. Installing their entire package would introduce unrelated football/chess/Gridworld dependencies and a different dependency stack. The small adapter therefore reuses their score estimator and reward-credit method, imports SB3's continuous-action distribution, and delegates simulation to this project's unchanged `runtime.run_episode`. The custom parts are the sample-recording proxy, the continuous-action offset likelihood, the explicit reward/baseline adapter, and evidence checks. They do not replace Gymnasium physics, SB3 prediction, action scaling, or distribution arithmetic.

## Collection and the exact score

`_SampleRecorder` forwards attribute access to the original SB3 model and overrides only `predict`. The first `warmup` actions are deterministic SB3 predictions. Later actions are ordinary stochastic predictions at **zero activation offset**. Immediately after each prediction, the proxy copies the distribution's actual unsquashed Gaussian sample `u`, its `tanh(u)` action, and the corresponding SB3 log probability. Reapplying SB3's `unscale_action` must reproduce the action returned to the simulator exactly.

All transitions are retained. The saved Boolean `score_mask` excludes the deterministic prefix from score estimation. Prefix Gaussian records are distribution modes, not random draws, and their log probabilities are audit information only. The runtime continues to own seeds, observations, action application, physical metrics, full onset-state hashes, termination, truncation, and frozen-weight verification.

The differentiated quantity for a fixed recorded observation/action is

\[
s_t=\left.\nabla_z\log\pi_z(a_t\mid o_t)\right|_{z=0},
\qquad h_1\leftarrow h_1+z,\quad z\in\mathbb R^{256}.
\]

Both `u` and `tanh(u)` are detached before likelihood evaluation. `get_action_dist_params` recomputes the actor's distribution parameters through a temporary first-ReLU addition, and SB3 evaluates `log_prob(tanh(u), gaussian_actions=u)`. The base actor remains frozen. Calling `model.predict` inside the gradient calculation would disable autograd; drawing a new action or retaining an `rsample` graph would produce a different derivative. Neither occurs here.

Float32 tanh can equal exactly +/-1 for large Gaussian samples. Inverting the bounded action then loses the original `u` and invokes SB3's clipping. Saving `u` avoids this failure. The tanh correction uses SB3's existing epsilon; because the recorded action is detached, that correction is constant with respect to the intervention and does not corrupt the Gaussian score gradient. Stored probabilities use normalized policy-action coordinates; the affine environment-action transform contributes only an intervention-independent density constant.

## Objective, credit, failures, and normalization

For post-prefix horizon \(H=\texttt{max_steps}-\texttt{warmup}\), currently 900 by default, the reward is \(r_{it}=v^x_{it}/H\). Future rewards are zero after a rollout ends. Upstream environment reward, startup speed, and physical health labels do not enter the gradient reward. With gamma equal to one,

\[
G_{it}=\sum_{k=t}^{H-1}r_{ik},\qquad
A_{it}=G_{it}-\frac{1}{N-1}\sum_{j\ne i}G_{jt},\qquad
d_{\rm raw}=\frac1N\sum_i\sum_{t<T_i}A_{it}s_{it}.
\]

Only observed stochastic timesteps have score terms. Episodes ending before onset contribute zero score and remain in \(N\). Other episodes' return-to-go values provide a time-dependent baseline independent of the current episode's action. The estimate sums timesteps within each episode and averages episodes; it does **not** divide by each episode's surviving length. This corresponds to expected fixed-horizon speed with zero progress after termination. It differs from the evaluation summary's average over observed post-prefix steps when episodes terminate early; held-out usefulness and failure gates remain mandatory.

No inversion, contact, speed-floor, or survivor filter is applied. In particular, outcome-based filtering would change the score estimator. Health labels are reported only to establish that unfavorable episodes were retained. If the speed objective produces unhealthy steering, the later prespecified usefulness gates must reject it.

The saved vector is \(d=S d_{\rm raw}/\|d_{\rm raw}\|\). The scale \(S\) is centered activation RMS over all observed post-prefix activations, with each contributing episode weighted equally. A prefix-only episode has no post-prefix activation values from which to compute an RMS, but is still retained in the gradient denominator. This differs explicitly from the classic extractor's eligible complete-window scale. Both raw vectors and the scale convention are recorded. Nonfinite or norms at most \(10^{-12}\) produce an explicit skipped result; a failed shuffled-control norm rejects the whole incomplete family.

The method computes one local zero-offset gradient. It does not optimize the offset for multiple steps using stale actions and does not update policy weights. Another nonzero-offset extraction would require newly sampled on-policy data and a separately registered experiment.

## Controls and immutable evidence

Three independent Gaussian random directions receive the same norm \(S\). The shuffled control uniformly permutes entire episode credit trajectories, preserving their time structure, and recomputes the score-weighted direction with the same actual actions. The full permutation is recorded. It can contain fixed points; rejecting those selectively would change the specified randomization. A shuffled control can align with the fitted direction by chance and is not presumed to have zero effect in any individual finite sample. All controls require the same validation opportunities as the candidate.

Each collected episode stores `gaussian_actions`, `squashed_actions`, `sampled_log_prob`, `score_mask`, `pgs_policy_sha256`, `pgs_source_sha256`, `pgs_warmup`, and `pgs_max_steps`, alongside existing runtime fields. `pgs_source_sha256` fingerprints normalized source for both this adapter and the reused runtime at import. Fitting rejects mismatched policy/source/schedule identities, duplicate episode seeds, malformed masks, action/sample mismatches, nonfinite records, and likelihood replay differences beyond `rtol=2e-5, atol=2e-4`.

Fitted diagnostics additionally hash every score-relevant array with its name, dtype, and shape. They record the maximum observed likelihood replay error, complete raw vectors, scale factors, sampling seeds, retained unfavorable episodes, random permutation, and objective/baseline definitions. The parent orchestrator must still record installed software versions, the pinned checkpoint revision/hash, disjoint data splits, and an immutable run manifest before collecting research data. Existing safe NPZ/JSON storage is used; no new pickle or executable-array format is introduced.

For 64 complete default episodes, the gradient uses 57,600 stochastic transitions after the deterministic prefixes. Gaussian actions, tanh actions, probabilities, and masks add about 4 MB in uncompressed arrays for six-action HalfCheetah. Captured 256-dimensional float32 activations add roughly 65.5 MB across all 64,000 transitions. Graph memory is bounded by the default batch size of 256; raw numerical artifacts remain resumable units for the parent runner. These are layout estimates, not measured throughput or compressed file-size claims.

## Verification

Run `.venv/Scripts/python.exe -m pytest tests/test_pgs.py -q` on the project environment, or `uv run pytest tests/test_pgs.py -q`. The tests exercise real SB3 SAC inference and the pinned TQC checkpoint, exact executed-action reconstruction, baseline likelihood replay, finite-difference agreement, detached samples, exact tanh saturation, exceptional hook cleanup, frozen parameters/buffers, episode credit boundaries, unfavorable-episode retention, control norms, and modified-action rejection. Short software-test rollouts are not research training, validation, confirmation, replication, or practical-use evidence.

## Upstream license notice

The adapted PGS estimator and reward-credit source are provided under this notice from the pinned upstream repository:

```text
MIT License

Copyright (c) 2026 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

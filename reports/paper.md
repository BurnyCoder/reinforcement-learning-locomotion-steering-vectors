# Causal activation steering in frozen RL locomotion policies

**Featured result: Ant lateral steering on fresh episodes, alongside four earlier HalfCheetah experiments**

Evidence cutoff: **2026-09-08, after experiments 001–005 completed their recorded numerical stages**. This is a research snapshot, not a peer-reviewed publication. Numerical results and protocol decisions are preserved in the accompanying experiment reports. Preliminary media depict existing validation trajectories; rendering does not create new statistical evidence.

The [documentation audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/documentation-audit.md) records later corrections to terminology and provenance without changing the scientific artifacts or recorded outcomes.

## Abstract

We investigate whether a constant additive intervention inside a reinforcement-learning locomotion policy can produce useful physical behavior without behavioral cloning or changing policy weights. We evaluate one pretrained checkpoint per policy, TQC on HalfCheetah and SAC on Ant, each recording seed 0. Contrastive directions are extracted from naturally occurring, episode-weighted behavioral variation and injected after the actor's first ReLU. An adaptive five-experiment sequence tests speed, action effort, torso height, lateral velocity, and yaw rate, with random, shuffled-label, and constant-action-bias controls. The initial HalfCheetah extraction was rejected because a few near-stationary recovery windows dominated activation means. Later experiments produced causal posture and speed changes, but the selected smaller-strength speed intervention failed physical quality on thirty fresh episodes. Ant provided the clearest lateral effect: the locked direction changed lateral velocity by **+0.40238 m/s**, paired 95% interval **[+0.30803, +0.49127]**, while retaining **96.83%** of mean forward speed on thirty fresh episodes. Its formal acceptance still failed because an additional strict rule rejects any episode ending before intervention; one identical baseline/treatment episode ended at step 34. Overall failure frequency was 4/30 in both arms, with differing failed-seed identities. These findings support causal locomotion effects in the tested networks, while exposing distinctions among observable effects, practical acceptance, baseline competence, and semantic specificity. No intervention completed the separate replication or application stages.

**Explore Ant 005:** [tracking video](https://github.com/user-attachments/assets/676ef6c8-ade4-4f85-a0b7-01d7fcdb86c9), [top-down video](https://github.com/user-attachments/assets/17286e19-c5a2-4ab9-96ab-f50444d494ca), [fixed far-camera video with trails](https://github.com/user-attachments/assets/9e0719a6-c15b-474e-b74e-3c5e70954e2a), and [published vector bundle](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/blob/1cd00cfcce13bcb86f1c370a71889ce6d34c278e/experiments/ant-classic-005/vectors.npz). All three movies show the same predetermined validation seed, 510000; they illustrate existing trajectories rather than fresh confirmation data.

## 1. Research question and scope

The objective is to find an interesting, useful causal change in RL locomotion, not to require a particular speed effect or an internal concept with a predetermined name. Contrastive activation addition (CAA) offers an existing construction based on differences of hidden means [R1]; Baukit supplies reusable PyTorch intervention hooks [R2]. We adapt those components to continuous control and physical measurements. We do not claim novelty of activation steering, a complete literature comparison, or that the original language-model findings validate locomotion control.

| Policy used | Verified actor and interface | Upstream training record |
| --- | --- | --- |
| HalfCheetah-v5 TQC medium | Two 256-unit ReLU hidden layers; 17 observations, 6 actions | Saved seed 0; 5,920,000 recorded environment timesteps |
| Ant-v5 SAC medium | Two 256-unit ReLU hidden layers; 105 observations, 8 actions | Saved seed 0; 8,770,000 recorded environment timesteps |

Both checkpoints are public Farama-Minari releases [R3, R4]. Training seed and step counts were read from their ZIP metadata without executing serialized objects. Saved optimizer settings include learning rate 0.0003, discount 0.99, batch size 256, and replay capacity 1,000,000; these are provenance, not hyperparameters tuned here. Curriculum, full learning curves, and upstream training history were not independently reproduced. There was **no local policy training, fine-tuning, action-demonstration fitting, or behavioral cloning**. Each environment is represented by one evaluated checkpoint; many evaluation resets do not establish generalization across separately trained policies.

<!-- pagebreak -->

## 2. Methods: data and interventions

### 2.1 Frozen inference and behavioral measurements

The actor maps observations to bounded continuous motor actions. We intervene at `actor.latent_pi[1]`, the first ReLU of the verified architecture, leaving all learned parameters fixed. The rule is `h_new = h + alpha * d_scaled`, applied after a common 100-step unsteered prefix. A later nonlinear layer makes state-dependent action effects possible. Every episode runs for at most 1,000 steps; the simulator control timestep is 0.05 seconds. Model revision, file hash, parameter hash, software versions, environment interface and selected layer are recorded per run.

Measured targets are forward velocity (m/s), lateral velocity on the ground (m/s), mean torso height (m), yaw rate (rad/s), and an effort proxy: mean squared normalized actions. Lateral is the world's ground-plane y velocity; it does not establish body-frame sidestepping or jumping. Forward-speed retention is a ratio of episode-averaged speeds, not retained distance traveled or full-horizon completion. The effort proxy is not motor energy, work, or battery consumption. Turning uses quaternion-derived orientation with wrapped angle differences; no quaternion coordinate is treated as an angle. Original reward, action saturation, inversion, inappropriate ground contact and episode failure are recorded alongside target effects [R5, R6].

### 2.2 Contrastive extraction

Each new fitting study starts with 16 stochastic diagnostic episodes and 64 stochastic fitting episodes. Full, non-overlapping 100-step windows after startup provide local variation even when whole-episode behavior is nearly constant. Nonfinite windows and windows exceeding 5% inversion or forbidden head/torso-ground contact are excluded from fitting with recorded counts. Short tails do not supply incomplete windows. Failed evaluation episodes are never removed to improve results.

Upper and lower behavior quartiles define the contrast groups. Each group must have at least eight independent contributing episodes. Window contributions are averaged within episode before averaging episodes, so long or frequent contributions do not become extra independent observations. The raw direction is `d_raw = mean(h_high) - mean(h_low)`. Effort contrasts are formed within up to five forward-speed bins, with equal bin weighting; residual confounding remains possible. Quartile gaps and vector norms must exceed small numerical tolerances before normalization.

The direction's norm is scaled using `S = sqrt(E_fit[||h - mean(h)||_2^2])` and `d_scaled = S * d_raw / ||d_raw||_2`. This is the centered vector-norm RMS, with no division by hidden dimension. The expectation uses eligible timesteps, including within-window variation, with equal episode weighting. Alpha is relative to this source activation scale, not a physical-unit command. Split-half cosine and fitting sensitivity diagnose extraction stability; they do not prove causal effectiveness.

Experiment 001 used health-only eligibility. Following fitting-only sensitivity analysis, 002 added a forward-speed floor: retain finite healthy fitting windows strictly above half their median forward speed. This 0.5 fraction is a project pilot heuristic. It excluded no additional windows in the fresh 002 or 005 samples, so their results cannot be attributed empirically to the floor removing observations in those samples. Experiments 003 and 004 reuse the exact 002 height family and fitting evidence without refitting; only the evaluated target and fresh evaluation partitions change.

### 2.3 Controls and treatment identity

Each fitted primary family in these studies has three random directions and a shuffled-label direction, all matched in norm and given the same validation strength grid. The general extraction procedure can skip a shuffled contrast that fails eligibility checks; no missing control is silently counted as evaluated. When a primary candidate qualifies, a constant action-bias comparator is fitted using every tenth post-startup source observation. It averages steered-minus-baseline deterministic actions at the selected hidden strength and divides by the largest absolute comparator-grid strength. Multiplication by that positive magnitude reproduces the fitted mean displacement before runtime clipping. This is a fixed action offset, not a learned imitation policy. Exact vectors, control provenance, action biases and source mappings remain saved. In retargeted runs, an alias such as `speed` can denote a height-derived array; the original extraction label remains explicit. The [bias construction](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/src/rl_locomotion_steering_vectors/experiment.py#L72) records its normalization.

<!-- pagebreak -->

## 3. Evaluation, acceptance and adaptive chronology

Evaluation uses deterministic actions and paired reset seeds. Each pair reaching intervention must have matching complete simulator-state hashes at onset. Per-episode summaries average available post-onset steps; all planned pairs remain in the estimator. Prefix-only episodes receive zero behavioral summaries but still count as failures. Effects are equal-weight means of paired episode differences, not pooled timestep averages or fixed-horizon travel. Confidence intervals use 2,000 percentile-bootstrap resamples of paired episodes; seeds are 710 for validation and 720 for confirmation. Intervals are nominal, with no correction for the many exploratory conditions. The [summary implementation](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/src/rl_locomotion_steering_vectors/analysis.py#L194) defines these conventions.

The [implemented target gates](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/src/rl_locomotion_steering_vectors/analysis.py) define the following project thresholds; library defaults do not supply these usefulness criteria.

| Target | Practical change threshold | Movement preservation |
| --- | --- | --- |
| Forward speed | At least 5% absolute change | At least 50% baseline forward speed |
| Effort proxy | At least 10% reduction | Forward speed within 5% of baseline |
| Torso height | At least 0.03 m absolute change | Forward speed within 10% of baseline |
| Lateral velocity | At least 0.2 m/s absolute change | At least 80% baseline forward speed |
| Yaw rate | At least 0.1 rad/s absolute change | At least 80% baseline planar speed |

All targets also require an interval excluding zero and quality acceptance. The quality-increase rule permits at most five percentage points additional failure, inversion, or forbidden contact. After the 002 measurement review, an individual episode is physically failed when its post-onset inversion or forbidden-contact fraction exceeds 5%, combined with explicit termination/runtime failure. Episode-averaged time fractions remain separately reported. A further implementation criterion requires zero **prefix-only** episodes in both arms: an episode with no post-onset steps disqualifies the full condition. Such pairs remain in the denominator with the project's zero-value summary convention. This is not a general missing-data method. It determines Ant 005's formal outcome and must be distinguished from intervention-induced failure.

Primary candidates are selected on ten validation seeds using the largest absolute eligible effect, preferring a smaller absolute strength only if within 5% of that best effect. Controls retain their strongest informative comparison even if they fail validation. The exact primary and control choices are saved before thirty fresh confirmation episodes. A separate thirty-episode replication is attempted only after successful primary confirmation. A later fixed-target application uses another fresh set and its own locked specification. None of these studies reached replication or application.

| Run | Fitting provenance | Validation seeds | Confirmation / replication seeds |
| --- | --- | --- | --- |
| 001 | Fresh 1000–1063 | 10000–10009, unused | 20000–20029 / 30000–30029, unused |
| 002 | Fresh 101000–101063 | 110000–110009 | 120000–120029 / 130000–130029, unused |
| 003 | Reuse 002 unchanged | 210000–210009 | 220000–220029 / 230000–230029, unused |
| 004 | Reuse 002 unchanged | 310000–310009 | 320000–320029 used / 330000–330029 unused |
| 005 | Fresh 501000–501063 | 510000–510009 | 520000–520029 used / 530000–530029 unused |

This is an adaptive research sequence. Each subsequent hypothesis may use earlier fitting/validation findings, with fresh outcome partitions for its test. The original signed grid is ±0.05, ±0.1, ±0.2, ±0.5. Run 004 replaces it with ±0.01, ±0.02, ±0.025, ±0.03, ±0.035, ±0.04. The 002 quality measurement was revised during validation, before selection or held-out inspection, and old estimates remain archived. Later reports do not choose different strengths, drop failed episodes, or rewrite those historical decisions.

<!-- pagebreak -->

## 4. HalfCheetah: causal changes and failed practical acceptance

### 4.1 Rejected extraction and the episode-quality measurement problem

Run 001 produced 576 fitting windows, 517 initially eligible. Ten near-stationary recovery windows, just 1.9% of the eligible set, strongly influenced the mean activations despite passing the current health checks. Removing them changed direction cosine to 0.298 for speed, 0.113 for height, and −0.555 for effort. The extraction was rejected before validation. The measured issue was sensitivity to rare observation histories, including accumulated torso rotation, rather than insufficient natural variation [E1].

Run 002 used fresh fitting data: 564 of 576 windows from all 64 episodes were eligible, with no additional floor exclusions. Direction split-half cosines were 0.982 for speed, 0.895 for effort and 0.980 for height. None of its 24 primary vector/strength conditions passed all target gates. Height +0.05 nevertheless raised the torso **4.505 cm**, interval **[4.367, 4.645] cm**, with zero measured physical failures/contact/inversion. Forward speed fell **12.85%**, exceeding the height target's 10% allowance. Eight random/shuffled settings passed exploratory validation. This supports a controlled posture effect, without a confirmed useful height intervention or extraction specificity [E2].

After 74 validation conditions, inspection of speed −0.05 found that **4.47% episode-averaged contact concealed one episode stalled on its head for 400 consecutive steps**, or 20 simulated seconds. Named geometry contacts and identical replayed arrays ruled out an indexing/replay mismatch. Quality analysis was amended to classify failure per episode before averaging, while retaining the existing five-point allowance. The same condition then had one failure in ten trials and was rejected. All 120 final conditions were analyzed consistently; the [old/new analysis audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/hc-running-002/quality-analysis-audit.json) preserves identical target effects, intervals and seed metadata for the original 74. Two gate decisions changed. This documented amendment used validation observations, not held-out results.

### 4.2 Retargeted speed control and fresh-reset instability

The overall goal allowed testing the height-derived direction as speed control. Run 003 imported the exact family and tested 40 conditions on fresh seeds. Every primary strength failed physical quality. At +0.05, the direction slowed speed 17.01%, but two of ten episodes physically failed; two random-control strengths passed validation. The earlier clean sample did not establish reliable control across resets [E3].

Run 004 used the finer signed grid. Its 72 validation conditions comprise 60 activation settings and 12 fitted action-bias settings. Six primary strengths passed, including positive strengths +0.02 to +0.04. The selected −0.02 had the largest eligible slowdown: **16.43%**, with zero failures in ten validation episodes. Its locked confirmation on thirty new episodes gave **−5.359 m/s**, interval **[−6.986, −3.999]**, a **31.89%** slowdown, but **ten physical failures** and 18.24% episode-averaged forbidden contact. The primary failed confirmation; positive validation alternatives were not substituted afterward [E4].

The following locked-control values are in the [004 confirmation results](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/hc-height-speed-004/results.json), with choices preserved in [selection.json](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/hc-height-speed-004/selection.json).

| Run 004 confirmation control | Speed change | Failures / 30 | Pilot gate |
| --- | --- | --- | --- |
| Constant action bias | −0.67% | 0 | Fails size |
| Random 0 | −4.84% | 0 | Fails size |
| Random 1 | −5.09% | 0 | Passes |
| Random 2 | −3.29% | 1 | Fails size |
| Shuffled label | −13.92% | 1 | Passes |

The shuffled condition's one failure is within the five-percentage-point allowance at n=30; a pilot pass therefore does not imply zero failures. These controls retain meaningful causal evidence and prevent a claim that contrastive extraction uniquely supplies effective directions. Neither control was replicated or turned into an application.

<!-- pagebreak -->

## 5. Ant: a directional lateral effect

### 5.1 Fitting feasibility

Ant 005 tests lateral velocity and turning in a policy with a free three-dimensional root. The 16 stochastic diagnostic episodes show mean forward speed 6.351 m/s and lateral means from −0.090 to +0.443 m/s. Two diagnostic and eight fitting episodes ended early, with final torso height above 1 m, consistent with the default Ant upper healthy-height bound [R6]. They remain in the recorded data.

Fitting yielded **531 complete eligible windows from 62 of 64 episodes**, RMS **5.15135**, with zero observed fitting-window inversion/contact and no speed-floor exclusions. The lateral low/high contrast was **0.980 m/s**, split-half cosine **0.9657**. It remained substantial at 200- and 300-step windows. Low/high forward means differed by only 0.059 m/s, but the lateral window label correlated with heading at 0.792. Thus heading or diagonal progress is a plausible association; the evidence does not identify a pure sideways-motion concept [E5].

Turning was less sustained: its 100-step contrast was 0.0908 rad/s, falling to 0.0520 and 0.0303 at longer windows, with adjacent-window correlation −0.505. Recomputed yaw rates agreed with saved quaternion-derived values to 1.79e−14 rad/s where successor observations existed. This supports alternating correction as an interpretation, without proving a specific gait mechanism. No turning primary setting passed all validation gates.

### 5.2 Validation response and selection

At **lateral −0.1**, mean lateral velocity changed **0.12536 to 0.67037 m/s**, paired effect **+0.54501**, interval **[+0.44638, +0.63939]**. Mean forward speed retained 97.65% of baseline; treated failures were zero versus one at baseline. The opposite strength **+0.1** gave **−0.26147 m/s**, interval **[−0.37796, −0.13919]**, also with zero treated failures. These results demonstrate bidirectional lateral changes over the tested range. The sign of alpha is opposite the fitting label's apparent direction, which further cautions against a simple semantic reading.

![Figure 1. Ant lateral validation strength response. Every primary and random/shuffled activation setting is shown; error bars are paired-episode percentile intervals. The separate action-bias comparator was calibrated after a primary qualified.](paper_assets/ant-lateral-validation.png)

The full [validation results](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/ant-classic-005/results.json) contain 88 conditions: 80 activation settings across lateral/turning families plus eight action-bias settings. Eleven conditions pass pilot gates, including two primary lateral settings and nine generic controls. The stronger eligible primary response at −0.1 was [locked with five controls](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/ant-classic-005/selection.json) before confirmation at 00:09:03 UTC. Selecting a direction from this exploratory grid requires the separate fresh test; the figure alone cannot establish reliable usefulness.

<!-- pagebreak -->

## 6. Ant held-out effect and the strict prefix criterion

On thirty fresh paired reset episodes, the locked −0.1 direction preserves the positive lateral effect: **+0.40238 m/s**, interval **[+0.30803, +0.49127]**. Lateral mean changes **0.17472 to 0.57710 m/s**. Forward speed changes **6.26183 to 6.06317 m/s**, retaining **96.83%** of baseline. Baseline and treatment each have four failed episodes; contact remains zero, and inversion increases only 0.71685 percentage points. The direction retains the same effect sign, exceeds the 0.2 m/s threshold, and satisfies all movement and five-point quality-increase thresholds.

The formal result is nevertheless **`confirmation_failed`**. Seed **520025 terminates at step 34** under baseline, primary intervention and every selected control, before the prescribed step-100 intervention begins. The complete episode arrays are consumed confirmation data, not an additional experiment. The strict criterion that neither arm may contain any prefix-only episode alone rejects the primary. That failure cannot have been caused by an intervention that had not started. It also means these data cannot demonstrate post-onset control for every enrolled reset.

All thirty pairs remain in the reported estimator. The pre-onset pair contributes zero to both behavioral summaries, hence a paired difference of zero; no artificial trajectory is imputed after termination. No threshold was changed, no episode was dropped and no positive result was substituted after confirmation. We report the held-out causal effect and the protocol's formal rejection separately. The evidence supports a promising practical lateral intervention in this simulator, but not a fully accepted, replicated application under the current rules.

The [saved confirmation results](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/ant-classic-005/results.json) supply this table; the [prefix-failure audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/ant-classic-005/prefix-failure-audit.json) verifies the shared early episode.

| Locked Ant condition | Lateral effect, m/s | 95% paired interval | Failure count / 30 |
| --- | --- | --- | --- |
| Primary −0.1 | +0.40238 | [+0.30803, +0.49127] | 4 versus 4 baseline |
| Constant bias +0.5 | +0.13439 | [+0.07586, +0.19724] | 1 |
| Random 0 −0.05 | +0.15550 | [+0.09492, +0.21724] | 3 |
| Random 1 −0.1 | +0.25282 | [+0.19463, +0.31103] | 7 |
| Random 2 −0.05 | +0.15472 | [+0.08920, +0.21874] | 9 |
| Shuffled −0.2 | +0.89039 | [+0.77057, +0.97974] | 9 |

No selected control passes the full formal gate, because every arm includes the same prefix-only episode; some also fail effect size or increase failure beyond the allowance. The shuffled vector causes a larger lateral shift but raises failure from four to nine episodes. The primary's smaller effect is therefore not dominated on every measured criterion by that comparator. We make no claim of global optimality, superiority over all random directions, or semantic uniqueness.

A separate within-project [causal and implementation audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/ant-classic-005-causal-audit.md) reproduced the lateral candidate/control vectors and held-out effect and interval. Checks of 28 saved observations reproduced actions through both the hook and the explicit SAC actor computation, with the policy hash unchanged and hooks removed afterward. A complete baseline/treatment replay of seed 510000 reproduced all 2,000 saved actions and associated observations, rewards and physical measurements exactly. This verifies implementation and replay correspondence; it does not add independent outcome data or establish a body-frame steering concept.

The inherited physical-quality rules are deliberately operational. For Ant, early termination may reflect an excessive-height event rather than inversion or contact. Equal total failure rates do not imply identical failure mechanisms or identical failed seeds after onset. The additional prefix rule provides a clear evidence boundary but conflates baseline failure before treatment with acceptance of the treatment's later behavior. Revising that criterion would require a separately specified future analysis or experiment; the present outcome remains unchanged.

The audit also quantifies tradeoffs: confirmation mean squared action rises from 0.24702 to 0.29108, and original environment reward falls from 6.23937 to 5.86443 per analyzed step. A post hoc rotation into yaw-aligned coordinates gives only +0.05943 m/s side-velocity change, versus +0.40238 m/s in world Y. This is consistent with changed direction/heading and does not prove a sole causal mediator. These descriptive checks were not used for selection or acceptance. Episode means use available post-onset durations, so an early ending does not represent sustained motion over the full horizon.

<!-- pagebreak -->

## 7. Visual evidence: sideways motion on the ground

![Figure 2. Top-down displacement for the first predetermined validation seed, 510000, comparing zero steering with lateral alpha −0.1. Both axes lie on the ground and use equal physical scale. Positions are reconstructed from saved velocities and timestep relative to reset. This is one existing validation example, not a fresh application test.](paper_assets/ant-topdown-validation.png)

In this example, post-startup mean lateral velocity is **+0.073 m/s without steering and +0.547 m/s with steering**. Forward progress continues while the steered path accumulates more displacement sideways. The vertical plotting axis is the ground's sideways coordinate; it is not body height and does not show jumping. Torso height is measured separately in the numerical records.

The first three predetermined validation seeds, 510000–510002, have paired movies showing baseline on the left and the saved −0.1 intervention on the right. All six complete 1,000 steps with zero measured inversion or torso-ground contact. Steering begins after five simulation seconds. Media audits compare every replayed NPZ array's shape, dtype and bytes against its existing validation episode and check identical onset states and action prefixes. The 1,001 frames at 20 fps encode 50.05 seconds, including the initial reset frame; they depict 50 simulation seconds.

The ordinary tracking camera follows the actor, and the checkerboard can disappear beyond its finite rendered extent. Completed top-down and fixed-camera replays use scene-only ground visualization while preserving the trajectory. The fixed view uses identical camera framing in both panels, a 20 m grid, and labeled position rings and past-path trails. These are visual annotations, not robot geometry or additional evaluation evidence.

Watch the public seed-510000 [tracking](https://github.com/user-attachments/assets/676ef6c8-ade4-4f85-a0b7-01d7fcdb86c9), [top-down](https://github.com/user-attachments/assets/17286e19-c5a2-4ab9-96ab-f50444d494ca), and [fixed far-camera](https://github.com/user-attachments/assets/9e0719a6-c15b-474e-b74e-3c5e70954e2a) movies. Full media and audits are in the [pinned Hugging Face release](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/865e68e9bf9eadc8b4efc20873e98f2d70e5e45a/experiments/ant-classic-005/videos) and locally under `runs/ant-classic-005/videos/`. These examples were chosen by seed order, not by selecting the largest effect, and remain validation illustrations after the held-out outcome is known.

<!-- pagebreak -->

## 8. Discussion and limitations

Three distinctions organize the findings. First, an extraction association is not its causal effect: stable height contrasts also change speed, and the Ant intervention sign does not follow a simple high/low label interpretation. Second, a causal effect is not a complete usefulness claim: the HalfCheetah selected slowdown includes physical failures, while Ant's held-out lateral effect satisfies the quantitative movement and quality-increase thresholds but fails a separate enrollment-wide prefix criterion. Third, a control effect is evidence, not a nuisance to suppress: random and shuffled directions frequently influence behavior, limiting specificity claims about contrastive extraction.

The adaptive sequence contains 001's rejected construction, 002's disclosed measurement amendment, 003's failed coarse retargeting, 004's failed locked speed confirmation and 005's nuanced lateral result. All outcomes belong in the research record. Failure rates based on ten or thirty resets are coarse estimates, not precise population guarantees. The five-percentage-point allowance corresponds to zero additional failures at n=10 and potentially one at n=30. A gate can consequently pass with a nonzero failure count. A zero observed count likewise does not establish zero underlying failure probability.

Only two pretrained checkpoints were studied, one per policy, each with saved seed 0; upstream training history was not independently reproduced. Reset replication would still use the same learned parameters, and no run reached that stage. All claims concern the tested simulator, policy, intervention site, horizon, startup and observed states. We did not test real robots, command-following feedback, obstacle avoidance, terrain changes, long-horizon stability, environmental domain shift or transfer across independently trained policies. Mean action effort is not energy, nominal intervals are not multiplicity-adjusted, and physical-health proxies can miss unmeasured gait defects.

The strongest current result is **a held-out Ant lateral effect of about +0.40 m/s while retaining about 97% of forward speed**, with unchanged aggregate failure rate and an explicitly rejected formal gate caused by a shared pre-intervention endpoint. This is a concrete causal result worth documenting with its numerical and visual context. It is not a claim that all requirements for a reliable locomotion controller have been met. No behavioral cloning or further policy training is needed to describe the evidence already obtained.

## 9. Reproducibility and artifact provenance

The implementation uses native Windows/Python 3.12, Torch 2.5.1 CPU, Torchvision 0.20.1 CPU, Stable-Baselines3 2.4.1, sb3-contrib 2.4.0, Gymnasium 1.0.0 and MuJoCo 3.1.5. Baukit is pinned to commit `9d51abd51ebf29769aecc38c4cbef459b731a36e`; its generic hook API avoids language-model token assumptions. Four isolated rollout workers and one Torch thread per worker were used in later studies. The source repository's `uv.lock`, `.env` configuration, timestamped logs and complete manifests are the execution record [R7].

HalfCheetah checkpoint revision: `b4ce04da6f246f06ae4c4258b8ab39624e6600b4`; Ant: `de5978d34b0118341df2b0a30232c5d2bfcb148e`. Archive and parameter hashes remain in each manifest. Run 001's recorded `6fb6381` is retrospective: its commit occurred during the run, so it is not an independently established invocation-time source pin. Run 002 lacks that field. Runs 003/004 record `6600176`/`2377452`. Ant 005 diagnosis/fitting used `8ad6ce3`; its [phase audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/ant-classic-005/phase-provenance-audit.json) records validation/confirmation invocation `1b835f98446b86aa30c8ba3e50a6826b12ef4068` and source hashes. The [documentation audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/documentation-audit.md) gives exact chronology and provenance limits.

During 002, an inference-only loader override reduced an unused replay-buffer allocation from 308,000,000 to 1,540 bytes per loaded HalfCheetah model. The [memory audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/reports/hc-running-002/inference-memory-audit.json) records matching policy hashes and arrays in three checked episodes before cached runtime identities were migrated. This was not a training-resumption equivalence test. Raw scientific outcomes were unchanged. NPZ arrays are loaded with `allow_pickle=False`; full trajectories, source vectors, old analyses and immutable selections remain available for audit.

Run commands use `uv run locomotion-steering run`, `retarget`, `replay` and `report`, with source paths, saved configuration and fresh seed offsets. The application command supports replicated speed or lateral candidates but has not been exercised as a successful research application. Rebuild with `uv run python reports/build_paper.py`; `paper.md` is its only prose source, and both figures retain their measured source images. The PDF was visually checked page by page. Paper rendering does not select new treatments or launch research runs.

<!-- pagebreak -->

## References and detailed experiment records

[R1] Panickssery et al. **Steering Llama 2 via Contrastive Activation Addition** (2024 revision). [Primary paper](https://arxiv.org/abs/2312.06681). Source of the additive contrast construction in language models; locomotion methods here are adaptations.

[R2] Bau. **Baukit Trace implementation**, pinned source. [nethook.py](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py). Reused arbitrary-PyTorch activation capture/edit and cleanup mechanics.

[R3] Farama-Minari. **HalfCheetah-v5 TQC medium checkpoint**. [Pinned model files](https://huggingface.co/farama-minari/HalfCheetah-v5-TQC-medium/tree/b4ce04da6f246f06ae4c4258b8ab39624e6600b4). Architecture and training metadata were additionally verified in the downloaded archive.

[R4] Farama-Minari. **Ant-v5 SAC medium checkpoint**. [Pinned model files](https://huggingface.co/farama-minari/Ant-v5-SAC-medium/tree/de5978d34b0118341df2b0a30232c5d2bfcb148e). The evaluated checkpoint records seed 0; no new policy training occurred in this study.

[R5] Farama Foundation. **HalfCheetah environment**, [official documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) and [Gymnasium 1.0.0 source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/half_cheetah_v5.py). Physical interfaces and episode semantics; not the source of our usefulness thresholds.

[R6] Farama Foundation. **Ant environment**, [official documentation](https://gymnasium.farama.org/environments/mujoco/ant/) and [Gymnasium 1.0.0 source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py). Orientation, reward, health and termination context.

[R7] **RL locomotion steering vectors: implementation and experiment records**. [Source repository](https://github.com/BurnyCoder/rl-locomotion-steering-vectors). [Detailed methods](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/docs/methodology.md), [source/reuse audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/docs/sources.md), and [cumulative experiment register](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/experiments.md). This is the project evidence, not independent validation.

[R8] Stable-Baselines3. [SAC inference documentation](https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/sac.html), [TQC implementation documentation](https://sb3-contrib.readthedocs.io/en/v2.4.0/modules/tqc.html), and [checkpoint loading source](https://stable-baselines3.readthedocs.io/en/v2.4.1/_modules/stable_baselines3/common/base_class.html#BaseAlgorithm.load). Used for authentic frozen actor inference and the audited inference-buffer override.

[E1] **hc-classic-001**. [Fitting rejection and sensitivity analysis](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-classic-001-diagnostics.md); [complete compact bundle](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-classic-001/README.md).

[E2] **hc-running-002**. [Results and measurement amendment](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-running-002-findings.md); [physical contact inspection](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-running-002-validation-inspection.md); [complete compact bundle](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-running-002/README.md), including the archived 74-condition analysis and comparison audit.

[E3] **hc-height-speed-003**. [Failed coarse-grid retargeting](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-height-speed-003-findings.md); [complete compact bundle](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-height-speed-003/README.md), including unchanged-vector audit.

[E4] **hc-height-speed-004**. [Locally dated protocol record](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-height-speed-004-preregistration.md); [failed speed confirmation](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-height-speed-004-findings.md); [complete compact bundle](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/hc-height-speed-004/README.md).

[E5] **ant-classic-005**. [Locally dated protocol record](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/ant-classic-005-preregistration.md); [fitting inspection](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/ant-classic-005-diagnostics.md); [lateral results and prefix-only criterion](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/ant-classic-005-findings.md); [complete compact bundle](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/ant-classic-005/README.md). Paired movies use the first three fixed validation seeds; Figure 2 and the three attached views use seed 510000. The protocol records are project documents, not independent registrations; 005's first Git record follows the start of diagnostics.

Published evidence includes [002 commit 9fa4bb9](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/commit/9fa4bb97ef2ad7cc50d2888146c68407bc850d7c), the [003 completed-report release d4d4e76](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/d4d4e76cc3e9f5408bc348c9c85ba714b818b55b/experiments/hc-height-speed-003), and [004 release dc0748d](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/dc0748d926dd9e9b038d14741ccada3d2f123673/experiments/hc-height-speed-004). Later narrative/media publication is separate from scientific outcomes; bundle indexes and the documentation audit record contents and verification.

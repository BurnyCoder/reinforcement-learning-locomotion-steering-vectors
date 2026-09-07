# Experimental methodology

## Question and hypothesis

Can a constant offset at an internal layer of a frozen locomotion policy reliably change a useful behavior while preserving movement? The initial hypothesis is that contrasting naturally occurring behavior produces a direction with a stronger useful effect than generic perturbations. This is an adaptation of contrastive activation addition to continuous control, not a claim established by its original language-model paper. Sources are catalogued in [sources.md](sources.md).

There is no overall local-compute budget. Individual experiments remain finite, specified, checkpointed, and reviewed. A failed contrast or failed confirmation motivates a new hypothesis; it does not justify repeatedly selecting favorable test episodes. The first implementation does not train a new policy.

## Policy and intervention

The starting checkpoints are Farama-Minari's HalfCheetah-v5 TQC medium and Ant-v5 SAC medium policies. Their upstream algorithms learn through reinforcement learning. Policy parameters remain frozen during collection, extraction, calibration, and evaluation. Pretrained model revision, file hash, software versions, and observation/action spaces must be retained with each run.

The intervention site is the output of `actor.latent_pi[1]`, the first ReLU in the expected two-hidden-layer actor. Runtime architecture checks guard this assumption. The next nonlinear layer gives the direction an opportunity to change state-dependent action computation. The activation rule is `h_new = h + alpha * d_scaled`, where `alpha` is constant over the intervention segment. Baukit owns hook registration and removal.

## Data collection and vector extraction

Default reset partitions are diagnostic `0-15`, fitting `1000-1063`, validation `10000-10009`, confirmation `20000-20029`, and replication `30000-30029`. The manifest records the actual configuration and full seed lists; `.env` can change episode counts. All data derived from an episode stay in its partition.

Episodes run for at most 1,000 steps, with a 100-step unsteered prefix. Fitting uses non-overlapping 100-step windows after startup. This allows local behavioral variation even when whole-episode speed is nearly constant. Incomplete or unhealthy fitting windows are excluded with recorded reasons; failed evaluation episodes remain in the analysis.

For each behavior, upper and lower quartiles define contrasting activation means. Each group needs at least eight independent contributing episodes. Contributing episodes receive equal weight within each group, so many windows from a single episode do not manufacture independent evidence. The raw direction is `mean(h_high) - mean(h_low)`. Effort contrasts are formed within up to five speed bins, with equal bin weighting, so simply moving slowly does not define reduced action effort.

Vector magnitude is normalized to the fitting set's centered activation RMS, including variation within eligible windows and equal episode weighting. Quartile gaps must exceed `1e-8 * max(1, mean absolute label)`, and raw vector norms must exceed `max(1e-12, RMS * 1e-8)`. These small numerical tolerances prevent normalizing accidental floating-point differences; they are not usefulness thresholds. Nonfinite vectors and insufficient independent contrast are rejected. The diagnostics record the raw direction, scale, contributing episodes, and exclusion reasons. A lack of natural contrast is a diagnosed failure mode rather than a null intervention experiment.

## Calibration, controls, and fresh evaluation

The initial strength grid is `0, +/-0.05, +/-0.1, +/-0.2, +/-0.5`. Evaluation uses deterministic actions, paired reset seeds, and matching complete simulator states immediately before intervention. This supports a causal comparison of the activation change within the tested simulator and policy.

Every candidate has three random directions and a shuffled-label direction with the same norm. Controls receive the same validation opportunity. Selection is recorded before confirmation. Later report rendering cannot choose a different strength.

Fresh confirmation and replication episodes evaluate locked candidates. Confidence intervals are percentile bootstraps of paired episode differences, not independent resampling of correlated timesteps. Replication here means new reset episodes using the same policy checkpoint. One trained policy cannot establish robustness to the training seed.

| Behavior | Initial practical effect threshold | Movement preservation |
| --- | --- | --- |
| Speed | At least 5% sustained absolute change | Retain at least 50% of baseline forward speed |
| Effort | At least 10% lower mean squared action | Speed within 5% of baseline |
| Height | At least 0.03 m sustained torso-height change | Speed within 10% of baseline |
| Lateral movement | At least 0.2 m/s directed change | Retain 80% of baseline forward progress |
| Turning | At least 0.1 rad/s directed yaw-rate change | Retain 80% of baseline planar speed |

A useful condition must also have a confidence interval excluding zero and avoid increasing failure, inversion, or inappropriate torso-ground-contact fractions by more than five percentage points. These are explicit pilot heuristics. Gates are evaluated from measured outcomes and are not independent proof of meaningful internal concepts.

Mean squared normalized action is an effort proxy. It is not motor energy, battery consumption, or mechanical work. Changes in effort must be interpreted alongside movement, original reward, action saturation, and locomotion-quality metrics. Turning uses yaw-rate geometry rather than reading a quaternion component as an angle.

## Interpretation and escalation

Report a direction as useful only when its locked effect and quality conditions survive fresh evaluation. Check random and shuffled controls before claiming specificity. Report one-sided effects, disruptive effects, insufficient contrast, and null results explicitly.

A constant action-bias comparator and a fresh application demonstration remain follow-up experiments for promising candidates; neither is silently implied by a passing gate. Possible escalation includes a continuous-action adaptation of policy-gradient steering or an online-RL policy trained to respond to varied commands. These additions require their own implementation, validation, source review, and experiment specification. No behavioral cloning is permitted.

The research loop is: define a question, inspect prior work and diagnostics, state a hypothesis, run an experiment, interpret all results, and revise the hypothesis. Keep the reasoning for each change in its experiment report, including what failed, why the next intervention is plausible, and which fresh data will test it.

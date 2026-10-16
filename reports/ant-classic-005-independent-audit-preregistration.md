# Independent Ant 005 audit: pre-registered fresh-seed experiment

Written 2026-10-16, before any episode on seeds 540000–540019 existed. The commit that adds this file is pushed before collection starts, and the audit log records the collection start time. The design fixes the treatments, estimands, decision rules and analysis code ([`scripts/ant005_independent/`](../scripts/ant005_independent/)) in advance.

## Claim under audit

Adding the saved `lateral` vector at α = −0.1 after the frozen Ant SAC actor's first ReLU (`actor.latent_pi[1]`) raises mean post-onset world-Y velocity by +0.402 m/s [0.308, 0.491] on 30 confirmation pairs ([experiment findings](ant-classic-005-findings.md)). This audit asks three questions. Does the effect replicate on fresh resets? Is it specific to this direction, or would any perturbation of this size do the same? And by what mechanism does the vector act?

## What informed these hypotheses (not fresh evidence)

Before this protocol was written, the auditor inspected saved data only.
- **Validation (n = 10), sign-dependent ("odd") world-Y parts at |α| = 0.1:** lateral −0.40; three lateral random directions −0.14 to −0.21; turning −0.60; lateral_shuffled −0.55.
- **Confirmation kinematics:** about +4.9° torso yaw and +3.7° travel direction.
- **Offline first-layer gradient:** rotating the observed world about the vertical axis moves first-ReLU activations along a direction g with cos(lateral, g) = 0.915.

These observations shaped hypotheses M1 and M2, so they count as exploratory. The fresh seeds below test them. The saved-data integrity checks and reanalysis ran before this protocol; they use no fresh episodes.

## Design

- **Policy and environment:** the pinned `farama-minari/Ant-v5-SAC-medium` checkpoint (revision `de5978d…`), deterministic actions, unmodified `Ant-v5`.
- **Episodes:** steering from step 100 (onset) to step 1,000 or termination. The production rollout code (`runtime.run_episode`, `execution.collect_episodes`) and the condition-identity guards are reused.
- **Seeds:** 540000–540019 (n = 20), disjoint from every recorded split (500000–500015, 501000–501063, 510000–510009, 520000–520029, 530000–530029). The replication split stays unused.
- **Arms:** one shared baseline (α = 0). Every direction runs at α = −0.1 and α = +0.1; the perception arms run at ±δ_eq.
- **Directions:** all at norm 5.1513, the fitting activation RMS shared by the saved vectors, unless marked "raw".
  - `lateral` (primary) and `turning` (other-label contrast vector): the saved arrays.
  - `random000`–`random098`: 99 isotropic Gaussian directions, `SeedSequence(540099)`.
  - `shuffled01`–`shuffled19`: the `lateral_shuffled` output of the production estimator `analysis.fit_vectors(fit, ["lateral"], seed=k)` for k = 1…19. These reuse the saved fitting episodes and the original rules.
  - `heading_parallel` (raw), `heading_orthogonal` (raw), `heading_orthogonal_renorm`, `heading_axis_renorm`: the decomposition of `lateral` along ĝ.
    - g is the mean central difference (±2°) of first-ReLU activations when the observed world is rotated about z, over every tenth post-warmup fitting state. The rotation acts on the quaternion, world linear velocity and world-oriented contact wrenches ([`perception.py`](../src/rl_locomotion_steering_vectors/perception.py)).
    - `heading_parallel` + `heading_orthogonal` = `lateral` exactly.
- **Perception arms (no activation edit):**
  - The observed world is rotated by ∓δ_eq after onset; the physics is unchanged.
  - δ_eq = 0.1·(lateral·ĝ)/‖g‖ = **0.160876 rad (9.217°)**. Under linearization, this is the perceived rotation that moves first-layer activations along ĝ as much as α = 0.1 of `lateral` does.
  - A test checks the rotation against physically rotated MuJoCo states.
- **Treatment hashes:** every array and its float32 SHA-256 are in [`directions.npz`](ant-classic-005-independent-audit/directions.npz) and [`directions.json`](ant-classic-005-independent-audit/directions.json).
- **Size:** 124 directions. That is 1 + 2 × 124 + 2 = 251 arms × 20 seeds = 5,020 episodes.

**Unit and estimand.** One reset is one unit. f = treated minus baseline mean post-onset world-Y velocity, with zero for an episode that ends before onset (the original convention). Failed episodes are kept. Odd part = (f(+) − f(−))/2; even part = (f(+) + f(−))/2.

**Power.** The confirmation paired-difference SD is ≈ 0.26 m/s, so SE ≈ 0.058 at n = 20. That gives power ≈ 0.94 to detect a 0.2 m/s effect at two-sided 0.05. The 99 random directions give rank-p resolution 0.01.

## Hypotheses and decision rules

| ID | Question | Statistic | Decision |
|---|---|---|---|
| **P1** (primary) | Does the locked arm replicate? | Mean f at lateral −0.1; BCa 95% CI (10,000 resamples); sign-flip randomization p (10,000 Monte Carlo sign patterns) | **Falsified** if the CI includes 0 or the mean is below 0.2 m/s |
| **P2** (primary) | Does `lateral` beat a generic perturbation of the same size? | Rank p = (1 + #{random \|odd\| ≥ lateral \|odd\|})/100 | **Falsified** if the Holm-adjusted p (over P1's sign-flip p and P2) exceeds 0.05 |
| S1 | Is it label-specific? | Joint-seed bootstrap CI of \|odd(turning)\| − \|odd(lateral)\|; rank of lateral \|odd\| among 19 shuffle redraws | **Label-specific** only if the CI is entirely below 0 **and** the shuffle rank p ≤ 0.05 |
| S2 | Does any perturbation push toward +Y? | Mean even part over random directions; two-way bootstrap (directions × seeds) | Estimate only. Sanity check: the mean odd part over random directions has a CI covering 0 |
| M1 | Does the perceived-heading component carry the effect? | Odd parts of `heading_parallel`, `heading_orthogonal` (additivity gap), and rank p of `heading_orthogonal_renorm` among random directions | **Falsified** if the orthogonal remainder ranks at p ≤ 0.05, **or** the heading part's odd sign differs from lateral's, **or** \|odd(heading_parallel)\| < 0.5 \|odd(lateral)\| |
| M2 | Does rotating perception alone reproduce it? | Odd part of the ±δ_eq perception arms (BCa CI) | **Supported** if the CI is entirely below 0 (−δ_eq pushes toward +Y). **Falsified** if it is entirely above 0. Otherwise inconclusive |
| K1 | Rotated path, sidestep, or continuing turn? | Δ travel angle, Δ torso yaw, Δ gait-axis lateral velocity, per-window Δ yaw and its slope | Descriptive |
| E1 (exploratory) | Do perceptual gradients predict any direction's response? | OLS of random directions' odd parts on projections onto ĝ_yaw and ĝ_ωz; leave-one-out R²; predictions for held-out labelled directions | Exploratory, not a test |

Other effects are reported with intervals but no new decision rules: failures, forward-speed retention, the production gates, and each arm separately. Each control's cosine with the primary is reported. Any deviation from this protocol will be listed in the audit report.

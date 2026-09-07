# RL locomotion steering: hc-classic-001

Report generated: 2026-09-07T23:08:33+00:00

## Outcome

Recorded run status: extraction_rejected_before_validation.
Fitting-only sensitivity analysis found ten near-stationary recovery windows dominate raw vectors. No validation or confirmation was run. Follow-up uses a prespecified half-median speed eligibility floor and entirely fresh seed partitions.
Validation selection: No selected intervention was recorded.
A gate pass is a pilot usefulness decision for one condition; it does not establish generalization across training seeds.

## Question and method

Can adding a constant vector after the first actor ReLU change locomotion while preserving movement? The policy stays frozen. Vectors are episode-weighted differences of contrasting unsteered activations, normalized to centered activation RMS. The protocol includes random and shuffled-label controls on the same evaluation split when that stage is reached. No behavioral cloning is used.

## Interpretation

Delta is treated minus baseline in the behavior's units. Speed/lateral use m/s, height uses m, turning uses rad/s, and effort is mean squared normalized action, not physical energy. Confidence intervals resample paired episodes. Validation selects strengths; confirmation and replication provide fresh evidence when recorded. Useful effects require the target threshold, a confidence interval excluding zero, and locomotion/quality checks. Control effects must be considered before attributing specificity to the extracted direction.

## Fitting diagnostics

Fitting windows: episodes=64; total windows=576; eligible windows=517; eligible episodes=62; excluded nonfinite=0; excluded unhealthy=59; discarded tail steps=0; warmup=100; window=100.
Centered activation RMS: 34.204. Convention: episode-balanced eligible timestep RMS about episode-balanced mean; full 100-step windows after 100-step warmup.
effort (mean squared action): status=fitted; high episodes=54; low episodes=55; high windows=130; low windows=130; contrast=0.043926; raw norm=7.4358; split-half cosine=0.99627.
height (m): status=fitted; high episodes=57; low episodes=56; high windows=130; low windows=130; contrast=0.0463; raw norm=4.0843; split-half cosine=0.72829.
speed (m/s): status=fitted; high episodes=51; low episodes=52; high windows=130; low windows=130; contrast=3.2697; raw norm=8.1466; split-half cosine=0.99642.
Split-half cosine measures extraction agreement, not causal effectiveness. Full bin statistics and vector coordinates are retained in vector_diagnostics.json.

## Research decisions

2026-09-07T23:03:30.425196+00:00: Outcome decision recorded above. Evidence: reports/hc-classic-001-diagnostics.md.

## Unmeasured stages

No recorded measurements: validation, confirmation, replication. No causal steering outcome is inferred for these stages.

## Reproduction and provenance

Created: 2026-09-07T22:55:57.345507+00:00; code commit: 6fb6381.
Policy: farama-minari/HalfCheetah-v5-TQC-medium; algorithm: TQC; environment: HalfCheetah-v5.
Checkpoint revision: b4ce04da6f246f06ae4c4258b8ab39624e6600b4. SHA256: 8231919095d5a35a2b799e865e968feca535d5af3f40a4b092947c70a976b7e1.
Layer: actor.latent_pi.1; hidden dimension: 256; observation/action shapes: [17] / [6].
Seed partitions below are registered assignments, not completion counts.
Diagnostic: 16 seeds; range 0-15.
Fit: 64 seeds; range 1000-1063.
Validation: 10 seeds; range 10000-10009.
Confirmation: 30 seeds; range 20000-20029.
Replication: 30 seeds; range 30000-30029.
Configuration: torch_threads=1; max_steps=1000; warmup=100; diagnostic_episodes=16; fit_episodes=64; validation_episodes=10; confirmation_episodes=30; replication_episodes=30.
Software: numpy 1.26.4; torch 2.5.1+cpu; torchvision 0.20.1+cpu; stable-baselines3 2.4.1; sb3-contrib 2.4.0; gymnasium 1.0.0; mujoco 3.1.5; baukit 0.0.1.
Full exact seed lists, checkpoint metadata and the original specification remain in adjacent manifest.json. Full measured analysis is in results.json; extraction details are in vector_diagnostics.json. The report does not change these artifacts or rerun inference.

## Sources

CAA method: https://arxiv.org/abs/2312.06681
Baukit hooks: https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py
Environment and measurement references: docs/sources.md in the repository.
Reporting: https://docs.reportlab.com/reportlab/userguide/ and https://matplotlib.org/stable/users/index.html

Full artifacts: [manifest](manifest.json), [results](results.json), [extraction diagnostics](vector_diagnostics.json).

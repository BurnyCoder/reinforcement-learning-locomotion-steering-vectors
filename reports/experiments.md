# Experiment register and learnings

## Current evidence

The first experiment, **hc-classic-001**, was rejected at fitting review before validation. Ten near-stationary windows, only 1.9% of the initially eligible fitting windows, substantially changed all three extracted directions when removed. No causal steering outcome has been measured for this experiment. See its [artifact bundle](hc-classic-001/README.md) and the separate [diagnostic analysis](hc-classic-001-diagnostics.md).

Follow-up **hc-running-002** completed 120 validation conditions after correcting an episode-failure measurement problem. None of the 24 primary settings passed every gate for its registered target. The height vector at +0.05 raised the torso 4.5048 cm without measured physical failure, contact, or inversion, but its 12.85% slowdown exceeded the height target's 10% movement allowance. Eight control settings passed exploratory gates. No selection, confirmation, or replication was performed. See the [findings](hc-running-002-findings.md) and [complete bundle](hc-running-002/README.md).

The next registered hypothesis, **hc-height-speed-003**, tests whether the frozen height-derived direction is useful for speed control. It reuses 002 fitting evidence openly and uses fresh validation and held-out episodes for the new target. No confirmed useful vector is claimed yet.

Each research run retains its numerical evidence; the separate `report` command writes its Markdown/PDF report. The table below records decisions, including rejected extractions, and links the detailed reasoning instead of duplicating complete results.

| Experiment | Question / hypothesis | Observed outcome | Decision and learning |
| --- | --- | --- | --- |
| [hc-classic-001](hc-classic-001/README.md) | Do health-filtered speed, effort, and height contrasts describe sustained running? | 16 diagnostic and 64 fitting episodes; 517/576 fitting windows initially eligible. Removing 10 near-stationary windows rotated the speed and height directions strongly and changed the effort direction to negative cosine similarity. No validation was performed. | Reject the initial construction. Upright/contact checks and broad episode contributor counts do not prevent unusual observation magnitudes from influencing activation means. See [sensitivity evidence](hc-classic-001-diagnostics.md). |
| [hc-running-002](hc-running-002-findings.md) | Do sustained-running contrasts yield useful control of their speed, effort, or height target? | 64 fresh fitting episodes; 564/576 eligible windows, with zero additional speed-floor exclusions. All 24 primary settings fail at least one gate; height +0.05 gives a clean +4.5048 cm change but slows 12.85%. Eight control settings pass validation only. | Record a causal posture effect, with no confirmed useful intervention. A mid-validation physical-failure measurement correction catches a collapse hidden by the pooled mean; all 120 conditions were reanalyzed and the old 74 estimates preserved. See the separate [inspection chronology](hc-running-002-validation-inspection.md). |
| hc-height-speed-003 (registered follow-up) | Can the exact 002 height-derived direction provide useful forward-speed control, regardless of its extraction label? | Not yet reported. Reuse 002 fitting/diagnostic evidence and exact vectors; original eight nonzero strengths; fresh validation 210000–210009, confirmation 220000–220029, replication 230000–230029. | Recalibrate the height-derived family and controls against the speed gate, then lock before fresh evaluation. Report posture and effort as collateral effects. Smaller-strength posture calibration remains optional. |

## Policies considered

| Policy | Actor architecture | Training algorithm and objective | Training data | Curriculum | Steering accuracy / useful effect | Episode completion / failure | Domain | Evidence status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Farama-Minari HalfCheetah-v5 TQC medium | Two 256-unit ReLU hidden layers, verified on load | TQC online RL; environment return | Upstream policy-environment interactions; no action demonstrations used here | Not established by this audit | 001 rejected before validation. 002: clean +4.5048 cm posture effect, but no primary target meets every usefulness gate; eight control passes remain exploratory | 002 height +0.05: zero measured physical failures in ten episodes. Speed −0.05: one of ten collapses, now rejected by episode-level quality analysis | Simulated planar locomotion | No confirmation or replication recorded; 003 tests the height-derived direction for speed control |
| Farama-Minari Ant-v5 SAC medium | Two 256-unit ReLU hidden layers, verified on load | SAC online RL; environment return | Upstream policy-environment interactions; no action demonstrations used here | Not established by this audit | No research result recorded | Runtime smoke rollout and video checked; no statistical locomotion study recorded | Simulated quadruped locomotion | Runtime feasibility verified |

Checkpoint links and training provenance are in [the source audit](../docs/sources.md). Classification accuracy and token fill/completion rates do not apply to these continuous-control policies. Reports instead measure behavioral change, episode completion/failure, quality, and uncertainty.

## Initial design learning

The closest directly reusable intervention component found was Baukit `Trace`: it edits arbitrary PyTorch layer outputs and works with ordinary actor prediction calls. NNsight and pyvene support generic models but add integration machinery; the `steering-vectors` package assumes language-model token dimensions. This motivated reusing Baukit for hook mechanics and writing only locomotion measurements, contrasts, and evaluation. A plain-function adapter resolved Baukit's handling of bound-method signatures during integration; the prediction and cleanup behavior subsequently passed runtime tests. The exact source audit and fix rationale are in [sources.md](../docs/sources.md).

HalfCheetah's initial eligible fitting windows had a median speed of 14.593 m/s and an interquartile range of 13.899-15.192 m/s. Variation was present. The important obstacle was contrast composition: low-speed recovery histories could have unusually large hidden activations even when the current health proxies passed. The revised protocol preserves the original trajectories and separates fitting exclusions from evaluation, where failed episodes must remain counted.

The second review showed why averaging contact fractions across episodes can hide a collapse: one trial's long failure is diluted by nine clean trials. The new analysis first labels physically failed episodes, then averages those labels while retaining the pooled contact/inversion metrics. The inspection report records the original rule and observations; the subsequent registered measurement decision is recorded in the run manifest. This chronology preserves the distinction between diagnosing a problem during validation and testing a locked hypothesis on held-out episodes. See [methodology](../docs/methodology.md) for the exact criterion rather than treating a low pooled mean as evidence that every rollout remained competent.

Experiment 002 also separates an extraction label from its causal effects. Its height contrast was associated with lower speed in fitting, and its positive intervention changed both posture and speed. Strong split-half extraction agreement did not ensure a useful intervention for the original target. A random direction also changed height within the original pilot gates, so semantic specificity is unsupported. The next hypothesis targets the observed speed effect explicitly and tests it on new episodes; it does not relabel the completed experiment as a success.

Before the scientific run, the core implementation passed 23 tests; both checkpoints were exercised on native Windows/Python 3.12. A 1,000-step HalfCheetah smoke rollout took about 1.10 seconds under the tested CPU setup, and an Ant video was checked. These are development checks, not steering-effect evidence or performance guarantees for another machine.

## Terms in plain language

| Term | Meaning here |
| --- | --- |
| Policy / actor | The neural network that maps the robot's observations to motor commands |
| Reinforcement learning | Learning actions from interaction and rewards, rather than copying demonstration actions |
| SAC | Soft Actor-Critic, a reinforcement-learning algorithm for continuous actions |
| TQC | Truncated Quantile Critics, a reinforcement-learning variant using a distributional value estimate |
| Hidden activation | An intermediate numerical representation inside the actor |
| Steering vector | A fixed list of numbers added to a hidden activation to change the actor's behavior |
| Contrastive activation addition | Subtracting mean hidden activations of contrasting groups and injecting the resulting direction |
| Strength / alpha | The multiplier controlling how much of a direction is added |
| Frozen policy | The actor's learned weights do not change during this experiment |
| Paired episode | Baseline and intervention runs sharing a reset seed and the same state before steering starts |
| Validation | Data used to choose a candidate and its strength |
| Confirmation / replication | Fresh episode sets checking a choice after it has been locked |
| Confidence interval | A bootstrap estimate of uncertainty in the mean paired behavioral change |
| Median / interquartile range | The middle observed value / the interval covering the middle half of observed values |
| Cosine similarity | How similarly two vectors point; 1 means aligned, 0 means perpendicular, and a negative value means opposing components dominate |
| Fitting speed floor | A minimum forward-window speed used only to choose extraction data, not to discard evaluation failures |
| Pooled contact fraction | The mean share of time with forbidden contact across episodes; a low average can hide a long failure in one trial |
| Episode physical-failure rate | The fraction of episodes classified as failed before averaging, retaining individual collapses in the quality assessment |
| Effort proxy | Mean squared normalized actions, which is not a measurement of physical energy |
| Useful gate | A predeclared threshold combining behavioral effect, uncertainty, and locomotion preservation |
| Causal effect | A behavioral difference caused by the controlled activation intervention in the tested system |

## Entry requirements

For each experiment, record its question and hypothesis, reason for choosing the policy/descriptor/parameters, exact Git revision and artifact path, outcome including controls, interpretation, problems and fixes, supporting sources, and the next hypothesis. For a future training experiment also record architecture, objective, data generation, curriculum, hyperparameters, training progression, checkpoint selection, and evaluation splits. Distinguish an untested explanation from an observed cause.

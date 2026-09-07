# Experiment register and learnings

## Current evidence

The first experiment, **hc-classic-001**, was rejected at fitting review before validation. Ten near-stationary windows, only 1.9% of the initially eligible fitting windows, substantially changed all three extracted directions when removed. No causal steering outcome has been measured for this experiment. See its [artifact bundle](hc-classic-001/README.md) and the separate [diagnostic analysis](hc-classic-001-diagnostics.md).

Each research run retains its numerical evidence; the separate `report` command writes its Markdown/PDF report. The table below records decisions, including rejected extractions, and links the detailed reasoning instead of duplicating complete results.

| Experiment | Question / hypothesis | Observed outcome | Decision and learning |
| --- | --- | --- | --- |
| [hc-classic-001](hc-classic-001/README.md) | Do health-filtered speed, effort, and height contrasts describe sustained running? | 16 diagnostic and 64 fitting episodes; 517/576 fitting windows initially eligible. Removing 10 near-stationary windows rotated the speed and height directions strongly and changed the effort direction to negative cosine similarity. No validation was performed. | Reject the initial construction. Upright/contact checks and broad episode contributor counts do not prevent unusual observation magnitudes from influencing activation means. See [sensitivity evidence](hc-classic-001-diagnostics.md). |
| hc-classic-002 | Does a fitting-only speed floor remove the influential near-stationary regime while retaining useful contrasts? | Registered follow-up; no outcome recorded here yet. | Use a floor of half the healthy fitting-window median speed, fresh seed offset 100000, and unchanged held-out quality gates. This is a project pilot heuristic, not an established literature result. |

## Policies considered

| Policy | Actor architecture | Training algorithm and objective | Training data | Curriculum | Steering accuracy / useful effect | Episode completion / failure | Domain | Evidence status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Farama-Minari HalfCheetah-v5 TQC medium | Two 256-unit ReLU hidden layers, verified on load | TQC online RL; environment return | Upstream policy-environment interactions; no action demonstrations used here | Not established by this audit | Not established: experiment 001 stopped before causal evaluation | 16 diagnostic + 64 fitting episodes collected; 7/64 fitting episodes exceed a 5% whole-episode inversion/contact proxy | Simulated planar locomotion | Initial vectors rejected after fitting-only review |
| Farama-Minari Ant-v5 SAC medium | Two 256-unit ReLU hidden layers, verified on load | SAC online RL; environment return | Upstream policy-environment interactions; no action demonstrations used here | Not established by this audit | No research result recorded | Runtime smoke rollout and video checked; no statistical locomotion study recorded | Simulated quadruped locomotion | Runtime feasibility verified |

Checkpoint links and training provenance are in [the source audit](../docs/sources.md). Classification accuracy and token fill/completion rates do not apply to these continuous-control policies. Reports instead measure behavioral change, episode completion/failure, quality, and uncertainty.

## Initial design learning

The closest directly reusable intervention component found was Baukit `Trace`: it edits arbitrary PyTorch layer outputs and works with ordinary actor prediction calls. NNsight and pyvene support generic models but add integration machinery; the `steering-vectors` package assumes language-model token dimensions. This motivated reusing Baukit for hook mechanics and writing only locomotion measurements, contrasts, and evaluation. A plain-function adapter resolved Baukit's handling of bound-method signatures during integration; the prediction and cleanup behavior subsequently passed runtime tests. The exact source audit and fix rationale are in [sources.md](../docs/sources.md).

HalfCheetah's initial eligible fitting windows had a median speed of 14.593 m/s and an interquartile range of 13.899-15.192 m/s. Variation was present. The important obstacle was contrast composition: low-speed recovery histories could have unusually large hidden activations even when the current health proxies passed. The revised protocol preserves the original trajectories and separates fitting exclusions from evaluation, where failed episodes must remain counted.

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
| Effort proxy | Mean squared normalized actions, which is not a measurement of physical energy |
| Useful gate | A predeclared threshold combining behavioral effect, uncertainty, and locomotion preservation |
| Causal effect | A behavioral difference caused by the controlled activation intervention in the tested system |

## Entry requirements

For each experiment, record its question and hypothesis, reason for choosing the policy/descriptor/parameters, exact Git revision and artifact path, outcome including controls, interpretation, problems and fixes, supporting sources, and the next hypothesis. For a future training experiment also record architecture, objective, data generation, curriculum, hyperparameters, training progression, checkpoint selection, and evaluation splits. Distinguish an untested explanation from an observed cause.

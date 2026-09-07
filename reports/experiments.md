# Experiment register and learnings

## Current evidence

No scientific locomotion experiment is recorded in this register yet. The initial code and protocol are being prepared. Library source inspection and synthetic/runtime tests establish implementation feasibility; they do not demonstrate a useful steering vector.

Each executed research run writes its own `report.md` and matching `report.pdf` alongside the manifest and full results. Add a concise entry here when its artifacts have been inspected. A negative or interrupted experiment also belongs in the register. Avoid copying full per-strength tables here; link the individual report instead.

## Policies considered

| Policy | Actor architecture expected by implementation | Training algorithm and objective | Training data | Curriculum | Steering accuracy / useful effect | Episode completion / failure | Domain | Evidence status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Farama-Minari HalfCheetah-v5 TQC medium | Two 256-unit ReLU hidden layers; runtime assertion required | TQC online RL; environment return | Upstream policy-environment interactions; no action demonstrations used here | Not established by this audit | Not measured locally | Not measured locally | Simulated planar locomotion | Pinned pretrained checkpoint selected |
| Farama-Minari Ant-v5 SAC medium | Two 256-unit ReLU hidden layers; runtime assertion required | SAC online RL; environment return | Upstream policy-environment interactions; no action demonstrations used here | Not established by this audit | Not measured locally | Not measured locally | Simulated quadruped locomotion | Pinned pretrained checkpoint selected |

Checkpoint links and training provenance are in [the source audit](../docs/sources.md). Classification accuracy and token fill/completion rates do not apply to these continuous-control policies. Reports instead measure behavioral change, episode completion/failure, quality, and uncertainty.

## Initial design learning

The closest directly reusable intervention component found was Baukit `Trace`: it edits arbitrary PyTorch layer outputs and works with ordinary actor prediction calls. NNsight and pyvene support generic models but add integration machinery; the `steering-vectors` package assumes language-model token dimensions. This motivated reusing Baukit for hook mechanics and writing only locomotion measurements, contrasts, and evaluation. Runtime compatibility remains a claim to verify through the real user workflow.

Nearly constant episode-average speed may hide variation within a gait or across time windows. The first protocol therefore inspects post-startup windows, retains episode-level independence, and supports descriptors beyond speed. Whether this yields useful vectors is an experimental question, not an established result.

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
| Effort proxy | Mean squared normalized actions, which is not a measurement of physical energy |
| Useful gate | A predeclared threshold combining behavioral effect, uncertainty, and locomotion preservation |
| Causal effect | A behavioral difference caused by the controlled activation intervention in the tested system |

## Entry requirements

For each experiment, record its question and hypothesis, reason for choosing the policy/descriptor/parameters, exact Git revision and artifact path, outcome including controls, interpretation, problems and fixes, supporting sources, and the next hypothesis. For a future training experiment also record architecture, objective, data generation, curriculum, hyperparameters, training progression, checkpoint selection, and evaluation splits. Distinguish an untested explanation from an observed cause.

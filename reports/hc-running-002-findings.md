# hc-running-002: posture changes, but no primary target passes

The frozen reinforcement-learning actor responded causally to a height-derived hidden vector: at strength **+0.05**, mean torso height increased **4.5048 cm** (paired 95% interval **4.3673–4.6452 cm**) with zero measured physical failures, forbidden contact, or inversion in ten validation episodes. Forward speed fell **12.85%**, exceeding the predeclared 10% allowance for a useful height intervention. All **24 primary vector/strength settings** failed at least one gate for their registered target. Eight control settings passed exploratory validation gates. No intervention was selected, confirmed, or replicated in this experiment.

The [complete numerical report and PDF](hc-running-002/README.md) retain all 120 tested conditions, curves, vectors, provenance, and the superseded quality analysis. This narrative explains the decisions and limitations. The earlier [physical inspection](hc-running-002-validation-inspection.md) is a historical record of the measurement problem before its correction.

## Hypothesis and experimental choice

Experiment 001 showed that a few near-stationary recovery windows could dominate activation contrasts despite passing window-level health checks. The new hypothesis was that contrasts drawn from sustained running would produce useful speed, effort, or height control. The [saved configuration](hc-running-002/manifest.json) specified retaining healthy fitting windows strictly above half their median forward speed before new calibration. The fraction is a pilot heuristic motivated by the [001 sensitivity analysis](hc-classic-001-diagnostics.md), not a threshold supplied by an upstream paper or an independent registration.

We reused the [Farama-Minari HalfCheetah-v5 TQC medium checkpoint](https://huggingface.co/farama-minari/HalfCheetah-v5-TQC-medium), pinned to revision `b4ce04da6f246f06ae4c4258b8ab39624e6600b4`. The loaded actor has two 256-unit ReLU hidden layers. Its weights remained frozen; no policy training or behavioral cloning took place. A single checkpoint provides a practical initial test of the intervention, with no claim about robustness across RL training seeds.

The intervention adds a fixed vector after the actor's first ReLU, using [Baukit's existing PyTorch hook implementation](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py). The contrast-of-means construction adapts [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681), which studied language models; its success in that domain does not establish locomotion effectiveness. Episode weighting, physical measurements, speed-conditioned effort contrasts, and the usefulness criteria are project adaptations described in [methodology](../docs/methodology.md).

## Protocol and parameter rationale

| Setting | Recorded value and reason |
| --- | --- |
| Fresh data | 16 diagnostic seeds 100000–100015; 64 fitting seeds 101000–101063; ten validation seeds 110000–110009. All differ from 001. |
| Planned held-out data | Confirmation 120000–120029 and replication 130000–130029. Neither partition was used. |
| Rollout and onset | At most 1,000 steps; first 100 unsteered; paired deterministic evaluation from matching onset states. The prefix separates initial settling from sustained intervention. |
| Extraction windows | Full, non-overlapping 100-step windows after startup; finite and healthy windows only. Each high/low quartile needs at least eight contributing episodes, with equal episode weighting. |
| Speed floor | Retain fitting windows strictly above 0.5 times the healthy median. This excludes recovery data only during fitting; evaluation failures remain counted. |
| Descriptors | Forward speed; mean squared normalized action as an effort proxy; torso height. Effort is contrasted within up to five speed bins to reduce speed confounding. |
| Strengths | −0.5, −0.2, −0.1, −0.05, +0.05, +0.1, +0.2, +0.5; one shared zero baseline. This initial coarse grid covers both signs and a tenfold magnitude range after RMS normalization. |
| Controls | Three random directions and one shuffled-label direction for each descriptor, with equal vector norm and the same strength grid. Fifteen vectors produce 120 conditions: 24 primary and 96 controls. |
| Uncertainty | Ten paired reset episodes per condition; 2,000 paired-episode percentile bootstrap resamples, seed 710. Timesteps are not treated as independent replicates. |
| Compute | Native Windows, Python 3.12, CPU Torch, four rollout workers and one Torch thread per worker. Exact package versions and seed lists are in the manifest. |

The target gates were unchanged: at least 5% absolute speed change while retaining half the baseline speed; at least 10% reduced action effort with speed within 5%; or at least 3 cm height change with speed within 10%. Each condition also needs an interval excluding zero and the quality criteria. These thresholds operationalize a pilot notion of usefulness; they are not an optimization objective for the frozen RL actor.

## Fitting evidence and software corrections

All 64 fitting episodes contributed eligible data. Of 576 windows, 564 passed and 12 were excluded as unhealthy; none were nonfinite. Healthy median speed was **14.6540 m/s**, giving a floor of **7.3270 m/s**. **The speed floor excluded zero additional windows in this fresh sample.** Differences from 001 therefore cannot be attributed empirically to removing slow windows in 002. Fresh sampling and the floor changed together at the protocol level, but the floor had no realized exclusion effect here.

| Contrast | High / low contributing episodes | Raw vector norm | Split-half cosine |
| --- | ---: | ---: | ---: |
| Speed | 61 / 59 | 1.02096 | 0.98237 |
| Effort | 60 / 57 | 0.43919 | 0.89514 |
| Height | 61 / 61 | 1.20069 | 0.97976 |

The centered activation RMS was **8.70504**. Split-half agreement shows reproducibility of extraction within this sample, not causal usefulness. The height contrast remained associated with speed: high-height fitting windows averaged **13.6862 m/s**, compared with **15.1210 m/s** for the low-height group. Its label therefore does not identify a speed-independent internal height concept. Effort binning also leaves small residual speed differences; full bin statistics are preserved.

Before calibration, the SB3 loader was found to allocate an unused training-sized replay buffer in each inference worker. The documented [`load` keyword override](https://stable-baselines3.readthedocs.io/en/v2.4.1/_modules/stable_baselines3/common/base_class.html#BaseAlgorithm.load) allowed an inference-only `buffer_size=1` setting. Replay-array allocation fell from **308,000,000 to 1,540 bytes** per loaded model. The policy hash stayed identical and all saved arrays matched bitwise in audited diagnostic seed 100000 and fitting seeds 101000 and 101010. Only the corresponding cached runtime identities were migrated. The [memory audit](hc-running-002/inference-memory-audit.json) records exact evidence; no training-resumption equivalence is claimed.

After 74 validation conditions, physical inspection found a separate analysis problem. The speed vector at −0.05 passed the original rule based on episode-averaged contact/inversion despite seed 110001 spending 400 consecutive steps stalled on its head. Nine clean trials diluted the mean contact fraction to **4.47%**. Named simulator contacts and identical replayed arrays ruled out a geometry-indexing or replay mismatch. HalfCheetah's own episode termination does not diagnose this failure: the [official environment documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#episode-end) specifies time-limit truncation and no task termination.

Calibration paused before selection or any confirmation. The recorded correction first labels an episode physically failed if its post-onset inversion or forbidden-contact fraction exceeds 5%, combining this with explicit failure or termination. It then averages episode failure labels. Episode-averaged contact/inversion fractions and the existing maximum five-percentage-point increases remain in the gate. The failed speed condition now has a **10% physical-failure rate**, so it fails quality. The 5% episode cutoff is a project heuristic, not a Gymnasium rule. The [documentation audit](documentation-audit.md) distinguishes this estimator from pooled timestep averaging and preserves the original archive filename.

All 120 conditions were analyzed under the revised criterion. Original trajectories and vectors were preserved, and the earlier 74-condition estimates remain in [the superseded analysis](hc-running-002/validation_pooled-quality-v1.json). The [quality audit](hc-running-002/quality-analysis-audit.json) checks that their target effects, intervals, paired seeds, and bootstrap metadata stayed identical. This was a validation-informed measurement revision, openly recorded before held-out evaluation; it was not an untouched initial analysis.

## Results and interpretation

| Primary target, eight strengths each | Target-size passes | Interval excludes zero | Movement passes | Quality passes | Complete useful passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Speed | 8 | 8 | 5 | 0 | 0 |
| Effort | 5 | 5 | 0 | 0 | 0 |
| Height | 6 | 2 | 0 | 2 | 0 |

Pass counts refer to separate components; a useful condition must satisfy all components simultaneously. Every speed-vector strength slowed the policy, including both vector signs. The extracted speed label did not produce a demonstrated bidirectional speed control.

The cleanest primary posture result was height **+0.05**. Mean torso height changed from **0.592742 to 0.637790 m**, while speed changed from **16.7955 to 14.6371 m/s**. Its height effect clears the size and interval criteria, with zero measured failure/contact/inversion; its **12.85% slowdown** fails the registered height movement criterion. Height **+0.2** also remained physically clean and increased height by **8.0281 cm**, but slowed forward motion by **38.51%**. A lower action-effort measurement accompanying slower movement is not evidence of energy savings at equal locomotion performance.

Eight of the 96 control settings passed their registered validation gates: seven speed controls and height_random0 at **+0.05**. That random height control raised the torso **3.2571 cm** (interval **3.2116–3.3033 cm**), with a **6.00%** slowdown and zero measured failure/contact/inversion. It is a promising exploratory intervention, but its selection from many conditions needs fresh evaluation. These controls prevent a claim that contrastive extraction uniquely identifies useful directions. The full report retains every favorable and unfavorable control result.

No primary target passed, so the pipeline recorded `no_validation_candidate`. It did not construct action-bias comparators, select interventions, or collect confirmation/replication episodes. This result establishes a controlled posture change within validation and reveals collateral speed control. It does not establish a confirmed useful height vector or an independent semantic representation.

## Limitations and next hypothesis

The same ten validation seeds support many comparisons and the analysis-method revision; nominal bootstrap intervals are exploratory, without a family-wise multiplicity correction. One checkpoint and one simulator do not show robustness across training seeds, policies, tasks, or hardware. Zero failures in ten episodes does not establish zero failure probability. The current health proxies and time-averaged outcomes can also miss unmeasured gait defects. No application demonstration or downstream benefit has been tested.

The next hypothesis after 002 was that the frozen **height-derived direction could control forward speed** while retaining competent locomotion, even though its fitting label was height. Experiment **[hc-height-speed-003](hc-height-speed-003-findings.md)** subsequently tested the exact 002 vector and random/shuffled family on fresh validation seeds **210000–210009**, retaining the original eight nonzero strengths. It failed to select a primary candidate; confirmation **220000–220029** and replication **230000–230029** remained unused. This chronology preserves the new target hypothesis without retroactively scoring 002 as successful. Finer posture calibration remained an optional hypothesis, not an observed result.

## Reproduction and evidence

See the repository [setup instructions](../README.md) for installation and the [bundle index](hc-running-002/README.md) for portable evidence. Reproducing the original 002 run requires its saved manifest configuration and original strength grid, rather than assuming future defaults remain identical. `uv run locomotion-steering report --run-dir runs/hc-running-002` renders saved results without rerunning selection. Full trajectories and timestamped execution logs remain in the local run directory; the compact report bundle preserves exact JSON values, vectors, and both analysis versions.

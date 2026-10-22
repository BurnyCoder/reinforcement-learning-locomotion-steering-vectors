# Experiment register and learnings

## Current evidence

The [documentation audit](documentation-audit.md) records the later fact-check corrections and their evidence. Locally dated protocol documents and saved phase specifications are distinguished from independent registration. Generated numerical reports remain historical snapshots; narrative terminology is corrected here without changing results.

The [independent fact-check](fact-check-2026-10-22.md) (2026-10-22) recomputed the documented numbers from raw episodes without using the earlier audits. It found no wrong recorded number or status. It corrected signs, stale "no replication" statements, overclaims, provenance notes, sources and the Hugging Face card, and found that the independent audit's fresh seeds consumed Ant 005's convention-reserved application cohort.

The [scientific-rigor audit](scientific-rigor-audit.md) (2026-10-16) checks whether the interpretations follow from the design. It recomputes all 332 saved conditions with production code and finds no error that invalidates a recorded number or status. It corrects three readings:
- **The Ant world-Y effect is robust but only partly specific.**
  - It survives a sign-flip randomization test (p ≈ 5×10⁻⁶, the floor of 200,000 Monte Carlo flips) and a displacement estimand: +17.4 m, interval [12.8, 21.7] m.
  - It exceeds each norm-matched random direction and the action bias by 0.15–0.27 m/s on the same seeds. Only random 1 ran at the primary's strength; it trails by 0.150 m/s [0.079, 0.215].
  - The three lateral-family random directions still show 35–112% of its sign-dependent lateral response at |α| ≤ 0.1. Across all six isotropic draws the range is −13% to 112%, with mixed signs.
- **Shuffled-label controls are not label-free.** They share the primary's dominant activation subspace: cosine +0.60 (Ant lateral), +0.58 (HalfCheetah height) and −0.83 (HalfCheetah effort).
- **Every HalfCheetah "speed effect" is degradation.** No HalfCheetah condition raised speed by more than about 0.4% (maximum +0.40%). For every primary the slowdown dominates at both signs; the 004 random controls slow at one sign only. The 004 calibration passes are degradation, not controllable speed.

Verdict on the methodology's initial hypothesis that contrast directions beat generic perturbations:

| Comparison | Verdict |
| --- | --- |
| Ant lateral vs isotropic random directions | Borderline: on fresh seeds, rank p = 0.05 among 99 norm-matched directions at \|α\| = 0.1 (validation: six draws, p = 1/7) |
| Ant lateral vs label-shuffle redraws | Not supported: 7 of 19 fresh redraws match or exceed it (rank p = 0.40) |
| Ant lateral vs the other-label (turning) contrast vector | Not supported: turning moves world Y more, also on fresh seeds (+0.19 m/s larger odd part, 95% CI +0.12 to +0.26) |
| HalfCheetah speed | Unsupported |
| HalfCheetah 002 height | Specific in validation only |

A post hoc reward-optimality hypothesis explains why posture and heading respond with a sign-dependent (odd) component while rewarded speed responds with a sign-independent (even) one. It awaits a fresh-seed test.

The [world-Y specificity audit](ant-classic-005-y-shift-audit.md) (2026-10-16, saved data, post hoc) checks the featured Ant claim. No error invalidates it as a causal, held-out, world-frame effect. It sharpens three readings:
- **Not specific to the lateral label.** The turning contrast vector (cosine +0.114) has a larger sign-dependent world-Y response at every |α| ≤ 0.2. At −0.1 it would pass the lateral gate.
- **About a quarter of the selected shift is sign-independent.** The even part is positive in 17 of 18 non-shuffled entries at |α| ≤ 0.1.
- **The shift is a rotated path, not a continuing turn.** The direction of travel rotates +3.84° [+3.01, +4.58] (28 of 29 pairs), with a bounded 4–7° heading offset and no sustained yaw rate. A rigid rotation predicts 18.0 m of extra world Y against 17.4 m measured.

Thirty pairs were enough to detect the threshold (about 13 needed), but a specificity rank test needs at least 19 random directions, and only six exist.

The [path-mechanism audit](ant-classic-005-mechanism-audit.md) (2026-10-16, saved data, post hoc) finds no error that invalidates the claim. Over the steps both arms recorded the effect is +0.392 m/s [+0.299, +0.477]. It corrects the two bullets above:
- **The 17-of-18 count shares one baseline.** Resampled jointly, the generic drift is +0.074 [−0.027, +0.164] m/s at |α| = 0.05 and +0.108 [+0.020, +0.187] at 0.1. The quarter (26%) has an interval of about [8%, 39%].
- **The 18.0-vs-17.4 m agreement is arithmetic.** A pure sidestep matches it equally well.
- **The rotation is not rigid.** The torso turns +5.23°, more than the path, so travel direction relative to the torso shifts −1.39° [−2.03, −0.85].
- **Pairing gives no variance reduction for world Y** (arm r = −0.40). The intervals stay valid, and a studentized sign-flip test agrees (p at its 200,000-flip floor, ≈ 5×10⁻⁶).

The [independent audit](ant-classic-005-independent-audit.md) (2026-10-16) did not use the audits above. It rebuilt the evidence chain from raw episodes and found no implementation, pairing, leakage or reporting error. It then ran a [pre-registered](ant-classic-005-independent-audit-preregistration.md) experiment on 20 fresh resets (seeds 540000–540019, 5,020 episodes):
- **Replicated.** Lateral −0.1 gives +0.462 m/s [0.343, 0.587], with 19/20 pairs positive. On these seeds it fails the failure gate (2/20 vs 0/20), as equal-size random perturbations typically do. Seeds 540000–540009 are also the application cohort `demonstrate` reserves by convention for 005, so a future 005 application needs other seeds.
- **Borderline against generic perturbations.** Rank p = 0.05 among 99 norm-matched random directions.
- **No detectable generic drift.** The mean sign-independent part of the random directions is +0.034 [−0.032, +0.104] m/s. The validation-based +0.108 estimate does not replicate.
- **Mechanism: perceived heading.** The vector has cosine 0.91 with the first-layer response to a rotated observation. Its heading component alone reproduces the effect. The remainder's sign-dependent response ranks like a random direction (p = 0.21), but rescaled to the full norm it fails 8/20 and 14/20 episodes. Rotating only the observation by −9.2° shifts world Y by +0.90 m/s (20/20 pairs) without failures. The policy corrects a heading it believes is rotated toward −Y, which explains why the useful strength is negative.

**Featured: Ant experiment 005.** The locked vector at −0.1 changes world-ground lateral velocity by **+0.40238 m/s**, paired 95% interval **[+0.30803, +0.49127]**, on thirty fresh confirmation episodes while retaining **96.83%** of forward speed. Failure counts are **4/30 in both arms**. This is a measured causal effect in the tested frozen policy; it is not a claim of body-frame sidestepping or a newly discovered internal concept.

Its recorded status is nevertheless `confirmation_failed`, solely because an additional strict rule forbids any episode ending before steering: seed 520025 ends identically at step 34 in both arms. The effect, movement preservation, and allowed quality increases pass. Equal aggregate failure counts do not mean identical failed-seed identities. No formal replication or application was run; the independent audit's fresh-seed reproduction is described above. See [005 findings and three attached videos](ant-classic-005-findings.md), its [complete bundle](ant-classic-005/README.md), the [separate within-project causal/code audit](ant-classic-005-causal-audit.md), and the standalone [research paper](paper.md) / [PDF](paper.pdf).

Watch the [tracking](https://github.com/user-attachments/assets/676ef6c8-ade4-4f85-a0b7-01d7fcdb86c9), [top-down](https://github.com/user-attachments/assets/17286e19-c5a2-4ab9-96ab-f50444d494ca), and [fixed far-camera](https://github.com/user-attachments/assets/9e0719a6-c15b-474e-b74e-3c5e70954e2a) views of the same predetermined validation pair. The [Ant vectors and controls are published on Hugging Face](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/blob/1cd00cfcce13bcb86f1c370a71889ce6d34c278e/experiments/ant-classic-005/vectors.npz), with pinned provenance and numerical results. Videos illustrate existing data and do not add independent evaluation trials.

The first experiment, **hc-classic-001**, was rejected at fitting review before validation. Ten near-stationary windows, only 1.9% of the initially eligible fitting windows, substantially changed all three extracted directions when removed. No causal steering outcome has been measured for this experiment. See its [artifact bundle](hc-classic-001/README.md) and the separate [diagnostic analysis](hc-classic-001-diagnostics.md).

Follow-up **hc-running-002** completed 120 validation conditions after correcting an episode-failure measurement problem. None of the 24 primary settings passed every gate for its registered target. The height vector at +0.05 raised the torso 4.5048 cm without measured physical failure, contact, or inversion, but its 12.85% slowdown exceeded the height target's 10% movement allowance. Eight control settings passed exploratory gates. No selection, confirmation, or replication was performed. See the [findings](hc-running-002-findings.md) and [complete bundle](hc-running-002/README.md).

**hc-height-speed-003** tested that frozen height-derived direction for speed control on 40 fresh validation conditions. All eight primary strengths failed physical quality; +0.05 produced two physical failures in ten trials. Two settings of the reused random control passed exploratory gates. No primary candidate was selected and no held-out data were consumed. No confirmed useful vector is claimed from this experiment.

**hc-height-speed-004** completed with failed confirmation. Six primary strengths passed calibration; the locked −0.02 setting slowed the policy by 31.89% on thirty fresh episodes but physically failed in ten. Two controls passed confirmation pilot gates; none received replication. See [004 findings](hc-height-speed-004-findings.md) and its [complete bundle](hc-height-speed-004/README.md).

The 002 numerical bundle was published at [Hugging Face commit 9fa4bb9](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/commit/9fa4bb97ef2ad7cc50d2888146c68407bc850d7c). The 003 raw run is at [revision 238bd2d](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/238bd2d9eba45fc1e69588f2beb2bbf15450f7b9/experiments/hc-height-speed-003), with numerical reports at [commit d4d4e76](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/d4d4e76cc3e9f5408bc348c9c85ba714b818b55b/experiments/hc-height-speed-003). The 004 run data and numerical reports are at [commit dc0748d](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/dc0748d926dd9e9b038d14741ccada3d2f123673/experiments/hc-height-speed-004). Those pins lack the local 002 quality-analysis and 003/004 vector-import audit files; the [documentation audit](documentation-audit.md) tracks their publication refresh. Publication status is separate from experimental success; later narrative corrections have their own version history.

Each research run retains its numerical evidence; the separate `report` command writes its Markdown/PDF report. The table below records decisions, including rejected extractions, and links the detailed reasoning instead of duplicating complete results.

| Experiment | Question / hypothesis | Observed outcome | Decision and learning |
| --- | --- | --- | --- |
| [hc-classic-001](hc-classic-001/README.md) | Do health-filtered speed, effort, and height contrasts describe sustained running? | 16 diagnostic and 64 fitting episodes; 517/576 fitting windows initially eligible. Removing 10 near-stationary windows rotated the speed and height directions strongly and changed the effort direction to negative cosine similarity. No validation was performed. | Reject the initial construction. Upright/contact checks and broad episode contributor counts do not prevent unusual observation magnitudes from influencing activation means. See [sensitivity evidence](hc-classic-001-diagnostics.md). |
| [hc-running-002](hc-running-002-findings.md) | Do sustained-running contrasts yield useful control of their speed, effort, or height target? | 64 fresh fitting episodes; 564/576 eligible windows, with zero additional speed-floor exclusions. All 24 primary settings fail at least one gate; height +0.05 gives a clean +4.5048 cm change but slows 12.85%. Eight control settings pass validation only. | Record a causal posture effect, with no confirmed useful intervention. A mid-validation physical-failure measurement correction catches a collapse hidden by the across-episode mean; all 120 conditions were reanalyzed and the old 74 estimates preserved. See the separate [inspection chronology](hc-running-002-validation-inspection.md). |
| [hc-height-speed-003](hc-height-speed-003-findings.md) | Can the exact 002 height-derived direction provide useful forward-speed control, regardless of its extraction label? | 40 fresh validation conditions on seeds 210000–210009. All eight primary strengths fail quality; +0.05 slows 17.01% but has two physical failures among ten trials. Two source height_random0 strengths pass validation. | No primary candidate, selection, or held-out evaluation. Unchanged vectors can fail physically on fresh resets even after a clean earlier sample. Preserve positive control evidence and distinguish a causal slowdown from reliable useful control. |
| [hc-height-speed-004](hc-height-speed-004-findings.md) | Can smaller signed strengths of the unchanged 002 height vector retain a useful speed effect while avoiding physical failures? | 72 validation conditions, including action bias; six primary strengths pass. Locked −0.02 confirmation has −31.89% speed change but 10/30 physical failures. Two confirmation controls pass pilot gates; no replication. | Record `confirmation_failed` and preserve selection. The largest eligible validation effect did not retain physical quality on fresh resets; other positive calibration settings were not substituted after the failure. The [locally dated protocol](hc-height-speed-004-preregistration.md) retains the original grid and decision rules with timestamp qualifications. |
| [ant-classic-005](ant-classic-005-findings.md) | Does a frozen Ant policy support useful lateral or turning steering from sustained natural contrasts? | 88 validation and six confirmation conditions. Locked lateral −0.1 has +0.40238 m/s held-out effect, 96.83% forward-speed retention, and 4/30 failures in each arm. No turning primary passes validation. | Record the positive causal lateral effect and `confirmation_failed` separately. One shared pre-intervention failure alone violates an additional strict criterion; no thresholds or exclusions changed. No formal replication or application; a later pre-registered audit reproduced the effect on fresh seeds. The [locally dated protocol](ant-classic-005-preregistration.md) and [fitting diagnostics](ant-classic-005-diagnostics.md) retain the prior reasoning. |

## Recorded partitions for the latest experiments

These seed sets were assigned before the corresponding phase outcomes. Assignment does not mean a phase has completed. The full per-run manifests remain the machine-readable record.

| Experiment | Diagnostic and fitting provenance | Fresh validation | Fresh confirmation | Fresh replication | Conditional application |
| --- | --- | --- | --- | --- | --- |
| hc-height-speed-004 | Reuse 002: diagnostic 100000–100015 and fitting 101000–101063; no refitting | 310000–310009, completed | 320000–320029, completed | 330000–330029, unused | 340000–340009, unused |
| ant-classic-005 | Diagnostic 500000–500015; fitting 501000–501063 | 510000–510009, completed | 520000–520029, completed | 530000–530029, unused | No application run; a lateral application requires successful replication and its own locked specification |
| ant-classic-005 independent audit | Reuses 005's fitting episodes and vectors | — | 540000–540019, completed (pre-registered audit experiment, not the formal replication split) | 530000–530029 still unused | None; its seeds consumed 005's convention-reserved application cohort 540000–540009 |

## Policies considered

| Policy | Actor architecture | Training algorithm and objective | Training data | Curriculum | Steering accuracy / useful effect | Episode completion / failure | Domain | Evidence status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Farama-Minari HalfCheetah-v5 TQC medium | Two 256-unit ReLU hidden layers, verified on load | TQC online RL; environment return | Upstream policy-environment interactions; no action demonstrations used here | Not established by this audit | 001 rejected before validation; 002/003 have no passing primary selection. 004's smaller additions pass calibration, but the locked primary fails confirmation. Some controls pass pilot gates | 004 selected −0.02 has zero failures in ten validation episodes, then ten failures in thirty confirmation episodes | Simulated planar locomotion | No replicated primary intervention or application result; complete negative evidence retained |
| Farama-Minari Ant-v5 SAC medium | Two 256-unit ReLU hidden layers, verified on load | SAC online RL; environment return | Upstream policy-environment interactions; no action demonstrations used here | Not established by this audit | 005 lateral −0.1 gives +0.40238 m/s on fresh held-out episodes and retains 96.83% forward speed. The independent audit reproduces +0.462 m/s on 20 further resets; the vector acts through perceived heading | Confirmation failure counts 4/30 in both arms; one identical pair ends before intervention. Fitting completion 56/64 | Simulated quadruped locomotion | Causal lateral effect measured; strict pre-onset criterion blocks formal acceptance, replication and application |

Checkpoint links and training provenance are in [the source audit](../docs/sources.md). Classification accuracy and token fill/completion rates do not apply to these continuous-control policies. Reports instead measure behavioral change, episode completion/failure, quality, and uncertainty.

## Initial design learning

The closest directly reusable intervention component found was Baukit `Trace`: it edits arbitrary PyTorch layer outputs and works with ordinary actor prediction calls. NNsight and pyvene support generic models but add integration machinery; the `steering-vectors` package assumes language-model token dimensions. This motivated reusing Baukit for hook mechanics and writing only locomotion measurements, contrasts, and evaluation. A plain-function adapter resolved Baukit's handling of bound-method signatures during integration; the prediction and cleanup behavior subsequently passed runtime tests. The exact source audit and fix rationale are in [sources.md](../docs/sources.md).

HalfCheetah's initial eligible fitting windows had a median speed of 14.593 m/s and an interquartile range of 13.899-15.192 m/s. Variation was present. The important obstacle was contrast composition: low-speed recovery histories could have unusually large hidden activations even when the current health proxies passed. The revised protocol preserves the original trajectories and separates fitting exclusions from evaluation, where failed episodes must remain counted.

The second review showed why averaging contact fractions across episodes can hide a collapse: one trial's long failure is diluted by nine clean trials. The corrected analysis first labels physically failed episodes, then averages those labels while retaining episode-averaged contact/inversion fractions. It averages each episode's available timesteps first, then weights episodes equally; this is not general pooled timestep averaging. The [inspection](hc-running-002-validation-inspection.md) and [old/new analysis audit](hc-running-002/quality-analysis-audit.json) preserve the validation-informed correction before held-out evaluation. A low across-episode mean does not establish competence in every rollout.

Experiment 002 also separates an extraction label from its causal effects. Its height contrast was associated with lower speed in fitting, and its positive intervention changed both posture and speed. Strong split-half extraction agreement did not ensure a useful intervention for the original target. A random direction also changed height within the original pilot gates, so semantic specificity is unsupported. Experiment 003 targeted the observed speed effect explicitly on new episodes, then rejected the coarse strengths for physical failures. A clean ten-episode sample was not sufficient evidence of reliable control across resets. Both studies retain random-direction successes as exploratory evidence, without relabeling a failed primary experiment as confirmed success.

Experiment 004 made this failure of generalization explicit after a locked selection. Several smaller positive strengths passed calibration, but the larger eligible slowdown at −0.02 won the specified selection rule and then failed physical quality on confirmation. The favorable alternatives remain calibration evidence, not replacement confirmations. One confirmed shuffled-control gate also tolerated a single failed episode because 1/30 is within the five-percentage-point allowance; a pilot pass must not be described as zero failures.

Experiment 005 exposes a different issue: a pre-existing failure before intervention can block a strict acceptance procedure even when a held-out causal effect and every allowed quality increase pass. All thirty pairs remain counted; the early pair receives zero behavioral/time-fraction summaries under the project convention but still counts as failed, while other episode summaries use their available post-onset steps. This is not a general missing-data correction. The rule remains unchanged after confirmation. A future protocol could distinguish pre-intervention baseline competence from intervention-induced failure, but it would not retroactively pass this experiment. Random and shuffled directions also cause lateral effects; some increase failure, and no isolated internal concept is established. The first three fixed validation seeds supply paired movies; the first seed supplies the paper's ground-plane plot. These existing trajectories add explanatory visuals, not new evaluation data.

Before the scientific run, the core implementation passed 23 tests; both checkpoints were exercised on native Windows/Python 3.12. A 1,000-step HalfCheetah smoke rollout was noted at about 1.10 seconds, but that timing was not logged; a logged re-measurement on 2026-10-22 gave 1.20–1.25 seconds on the same machine with one Torch thread ([log](fact-check-2026-10-22/)). An Ant video was also checked. These are development checks, not steering-effect evidence or performance guarantees for another machine.

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
| Episode-averaged contact fraction | Average the available post-onset contact indicator within each episode, then weight episode summaries equally; a low average can hide a long failure in one trial |
| Episode physical-failure rate | The fraction of episodes classified as failed before averaging, retaining individual collapses in the quality assessment |
| Prefix-only episode | An episode with no post-onset steps; its behavioral/time-fraction summaries are assigned zero but it still counts as failed, the pair remains counted, and the original strict rule blocks the full quality gate |
| Forward-speed retention | The ratio of steered to baseline episode-averaged forward speeds; it is not distance retention, completion rate, or evidence of full-horizon persistence |
| Lateral velocity | Speed along the world's ground-plane y axis; it is distinct from torso height and does not establish sideways motion relative to the body's heading |
| Effort proxy | Mean squared normalized actions, which is not a measurement of physical energy |
| Useful gate | A predeclared threshold combining behavioral effect, uncertainty, locomotion preservation, and physical-quality limits (failure, inversion and contact rise at most five percentage points; no prefix-only episode) |
| Causal effect | A behavioral difference caused by the controlled activation intervention in the tested system |
| Odd / even response | Halves of the difference and of the sum of the effects at +alpha and −alpha. A steerable direction has a large odd part; generic degradation appears as an even part with the same sign at both alpha signs. |
| Shuffled-control alignment | Cosine between a label-shuffled control and its primary. A large value means the control re-tests part of the primary direction instead of being a label-free null. |
| Travel angle | Direction of an episode's net ground displacement after steering starts. Its change measures how much the whole path is rotated, without relying on torso orientation. |
| Rank test | Compares one direction's response with k random directions: p = (1 + number at least as large) / (k + 1). With k = 6 the smallest possible p is 1/7, so at least 19 are needed to reach 0.05. |
| Body-relative travel change | Change of the travel direction measured from the torso: travel-angle change minus torso-heading change. It is zero if the whole gait turns rigidly and equals the path change for pure sidestepping. |
| Matched horizon | Comparing two arms only over the steps both recorded, so an earlier termination in one arm cannot change the comparison. |
| Arm correlation | Correlation, across resets, between a reset's baseline and treated outcomes. Pairing narrows intervals only when it is clearly positive. |
| Perceived-heading direction | The change in first-layer activations when the policy's observation is rotated about the vertical axis, as if the robot faced a different way. The Ant "lateral" vector has cosine 0.91 with it, so adding the vector changes where the policy thinks it is heading, not a sideways motor command. |
| Perception intervention | Rotating only what the policy observes (orientation, velocity and contact-force directions) while the simulator stays unchanged. It tests whether a steering effect works through the policy's beliefs about its state. |

## Entry requirements

For each experiment, record its question and hypothesis, reason for choosing the policy/descriptor/parameters, exact Git revision and artifact path, outcome including controls, interpretation, problems and fixes, supporting sources, and the next hypothesis. For a future training experiment also record architecture, objective, data generation, curriculum, hyperparameters, training progression, checkpoint selection, and evaluation splits. Distinguish an untested explanation from an observed cause.

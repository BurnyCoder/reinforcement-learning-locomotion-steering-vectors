# hc-height-speed-004: smaller steering passes calibration, fails confirmation

Six primary strengths passed validation, but the locked choice **−0.02 failed confirmation**: it reduced mean speed **31.89%** while producing physical failures in **10 of 30 episodes**. The run stopped at `confirmation_failed`; replication and application episodes were not collected. Two control conditions passed the confirmation pilot gates, and are retained in the complete evidence. This experiment does not establish a replicated useful primary intervention.

The [artifact bundle](hc-height-speed-004/README.md) contains every one of the 72 validation and six confirmation conditions, the PDF, vectors, action-bias comparator, locked selection, and provenance. The [pre-registration](hc-height-speed-004-preregistration.md) preserves the hypothesis and thresholds recorded before this attempt.

## Hypothesis and fixed protocol

Experiment 003 found physical failures at every primary setting of the coarse height-derived speed-control grid. Experiment 004 tested a narrower dose-response hypothesis: smaller additions might slow the frozen policy meaningfully before physical instability appears. The numerical grid was an adaptive project choice motivated by that prior result, not a value recommended by an upstream paper.

The exact height contrast and its three random directions and shuffled-label direction were imported from 002, without refitting. The run aliases them to `speed` and `speed_*` because speed is the evaluated outcome. The 16 diagnostic and 64 fitting episodes, activation scale, and source checkpoint are inherited from 002. Array hashes and source names are retained in [retarget.json](hc-height-speed-004/retarget.json). No policy training or behavioral cloning was performed.

The experiment reuses [CAA's additive contrast idea](https://arxiv.org/abs/2312.06681), [Baukit's layer hooks](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py), and the pinned [HalfCheetah TQC policy](https://huggingface.co/farama-minari/HalfCheetah-v5-TQC-medium/tree/b4ce04da6f246f06ae4c4258b8ab39624e6600b4). The new grid and usefulness rules are locomotion-specific project adaptations. Physical failures require explicit measurement because [HalfCheetah's ordinary termination](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#episode-end) does not diagnose unsuccessful locomotion.

| Component | Registered or recorded value |
| --- | --- |
| Signed strengths | ±0.01, ±0.02, ±0.025, ±0.03, ±0.035, ±0.04; one shared zero baseline |
| Source fitting | Reuse seeds 101000–101063, with unchanged first-ReLU vectors and RMS normalization |
| Fresh validation | Ten paired seeds 310000–310009 |
| Fresh confirmation | Thirty paired seeds 320000–320029 |
| Reserved replication | 330000–330029, unused |
| Conditional application | 340000–340009, unused |
| Rollout | 1,000 steps, identical 100-step unsteered prefix, deterministic evaluation |
| Speed gate | At least 5% absolute change with an interval excluding zero and at least half baseline forward speed retained |
| Quality | Existing episode-level physical failure and pooled inversion/contact criteria; no more than five percentage points increase in any quality measure |
| Selection | Largest absolute useful effect, preferring a smaller absolute strength only when within 5% of that best effect |
| Execution | Four CPU workers; one Torch thread each; code commit `2377452`; full software/configuration in the manifest |

Once a primary setting passed calibration, the pipeline derived a constant action-bias comparator from the original fitting observations. Its stored scale makes the largest positive grid strength, +0.04, reproduce the fitted mean action displacement, and the same signed grid tests it. Consequently, validation contains **60 activation conditions plus 12 action-bias conditions**, each evaluated on the same paired seed set. Confirmation follows the immutable [selection](hc-height-speed-004/selection.json), saved at **2026-09-08 00:03:15 UTC**, rather than choosing a new condition after viewing held-out outcomes.

## Validation and the selected strength

The primary passes were **−0.02, +0.02, +0.025, +0.03, +0.035, and +0.04**. All ten validation episodes for these strengths had zero physical failures. Positive strengths +0.02 through +0.04 produced slowdowns from **6.71% to 11.92%**, with no measured inversion or forbidden contact. These favorable calibration observations remain in the report; they were not independently confirmed in this run.

The selected negative strength **−0.02** produced the largest passing slowdown, **16.43%**, with mean speed **16.80834 → 14.04651 m/s**. Its paired effect was **−2.76183 m/s**, interval **[−2.97538, −2.49581]**, with zero physical failures/inversion and **0.0111%** pooled forbidden contact. It therefore won under the recorded selection rule. Four shuffled-label strengths also passed validation. None of the random-direction or action-bias grid settings passed all validation gates; their strongest measured comparisons were nevertheless retained for confirmation.

The selection rule rewards a large eligible effect, not an independent estimate of dynamical robustness. A clean sample of ten episodes can therefore promote a condition that is unreliable on new resets. That explanation concerns sampling and selection; it does not establish a particular gait-level failure mechanism.

## Confirmation and controls

The selected vector at −0.02 retained a causal slowdown on thirty new episodes, but its quality did not generalize. Speed changed **16.80155 → 11.44279 m/s**, paired effect **−5.35876 m/s**, interval **[−6.98568, −3.99911]**. Physical failure rose from zero to **33.33%**; pooled forbidden contact was **18.24%** and inversion **3.13%**. The target-size, interval, and forward-movement gates passed, while quality failed. Every failed episode remains in the effect and uncertainty calculations.

| Locked confirmation condition | Strength | Speed change | Physical failures / 30 | Pilot gate |
| --- | ---: | ---: | ---: | --- |
| Height-derived primary (`speed`) | −0.02 | −31.89% | 10 | Fail: quality |
| Constant action bias | −0.025 | −0.67% | 0 | Fail: effect size |
| Source height_random0 | +0.04 | −4.84% | 0 | Fail: effect size |
| Source height_random1 | −0.035 | −5.09% | 0 | Pass |
| Source height_random2 | −0.035 | −3.29% | 1 | Fail: effect size |
| Source height_shuffled | −0.035 | −13.92% | 1 | Pass |

The random1 control's effect was **−0.85519 m/s**, interval **[−0.96866, −0.76348]**, with zero measured physical failures, contact, or inversion. The shuffled control's effect was **−2.33813 m/s**, interval **[−3.19761, −1.86238]**. It passed the pilot gate despite one physical failure because **1/30 = 3.33 percentage points**, within the predeclared five-point allowance. A pilot pass must therefore not be described as zero failures. Neither control received replication or an application test.

The comparison does not support unique usefulness of the contrastive primary direction. It also does not show that no steering vector can be useful: random and shuffled directions retain measured effects. The action-bias comparator failed the practical speed-effect threshold for its locked setting; that result is limited to this fitted comparator and calibration procedure.

## Conclusion and evidence boundary

The finer grid produced candidate-quality calibration results, but the locked primary failed the first fresh check. The correct outcome is failed confirmation, not a selection of a more favorable positive strength from the now-observed evidence. Replication seeds remain unused. All reported intervals resample paired episodes; validation involves many comparisons, and a single checkpoint cannot establish robustness across RL training seeds or simulators.

The parallel [Ant experiment 005](ant-classic-005-preregistration.md) explores lateral and turning control in a different frozen policy. Its final usefulness and media claims require its own completed evidence. This report introduces no new method or additional training attempt.

Reproduction uses the repository's `uv.lock`, the saved manifest including its exact strength grid, and the source 002 fitting artifacts. The commands are `uv run locomotion-steering retarget --source-run runs/hc-running-002 --run-dir runs/hc-height-speed-004 --vector height --behavior speed` and the separate `report --run-dir runs/hc-height-speed-004` command. Changed protocols require a new run identity. The compact bundle preserves complete numerical results and fixed selection; full raw episodes and timestamped logs remain in the local run directory for publication.

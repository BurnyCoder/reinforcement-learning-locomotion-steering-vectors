# hc-height-speed-003: coarse speed control fails physical quality

The unchanged height-derived vector caused substantial slowdown on fresh validation episodes, but **all eight primary strengths failed the physical-quality gate**. The positive strengths +0.05, +0.1, and +0.2 retained more than half the baseline forward speed, yet produced physical failures in two, one, and two of ten episodes respectively. None qualified for selection. Two random-control settings passed validation. No confirmation, replication, or application episodes were collected.

The [complete bundle](hc-height-speed-003/README.md) contains all 40 measured conditions, a strength-response figure, PDF, exact vectors, and their source mapping. This report explains why the new target was tested and what the result changes.

## Question and hypothesis

Experiment 002's height vector at +0.05 increased torso height while slowing the actor by 12.85%, with no measured physical failures in its ten validation episodes. The original height usefulness criterion rejected that slowdown. The broader research goal allows a vector's useful effect to differ from its fitting label, so the follow-up explicitly tested **height-derived speed control**. The [retarget specification](hc-height-speed-003/retarget.json) records this hypothesis before new calibration; it does not retroactively turn 002 into a successful height experiment. This is a project specification, not an independent registration; see the [documentation audit](documentation-audit.md).

The frozen [HalfCheetah-v5 TQC medium actor](https://huggingface.co/farama-minari/HalfCheetah-v5-TQC-medium/tree/b4ce04da6f246f06ae4c4258b8ab39624e6600b4) and first-ReLU intervention were retained. No training or behavioral cloning took place. The direction uses the same contrast-of-means adaptation of [CAA](https://arxiv.org/abs/2312.06681) and [Baukit hooks](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py) as 002. The new question concerns a physical speed effect, not proof of an internal height or speed concept.

## Exact reuse and fresh evidence

The five saved source arrays were imported unchanged. The pipeline labels the family by the new evaluation target, so **`speed` in this run means 002's `height` vector**, and `speed_random0` means 002's `height_random0`. It is not the original 002 speed contrast. The [retarget specification](hc-height-speed-003/retarget.json) records every source name, shape, dtype, and array hash; a [separate audit](hc-height-speed-003/vector-import-audit.json) verifies byte equality for all five arrays.

No diagnostic or fitting episodes were recollected. The 16 diagnostic and 64 fitting episodes from 002 remain the declared source data. They supplied 564 eligible windows and centered activation RMS 8.70504; see [002's fitting evidence](hc-running-002-findings.md) for exclusions, contributor counts, and residual speed association. A label change does not remove those associations or create a new fitted direction.

| Protocol component | Recorded specification |
| --- | --- |
| New target | Forward speed, evaluated on the exact height-derived family |
| Calibration | Ten fresh paired seeds 210000–210009; deterministic actions; all 40 conditions |
| Reserved held-out sets | Confirmation 220000–220029; replication 230000–230029. Both unused. |
| Strength grid | −0.5, −0.2, −0.1, −0.05, +0.05, +0.1, +0.2, +0.5, plus one shared zero baseline |
| Controls | The original three random directions and shuffled-label direction; equal norms and equal calibration opportunity |
| Horizon and onset | 1,000 steps, with 100 unsteered steps before intervention; matched onset states |
| Useful speed threshold | At least 5% absolute change, interval excluding zero, and at least 50% of baseline forward speed retained |
| Quality | Episode physical failure plus episode-averaged inversion/contact; maximum five-percentage-point increase in each metric, using the corrected rule already established before this run |
| Uncertainty | 2,000 paired-episode percentile bootstrap resamples, seed 710 |
| Execution | Four CPU workers, one Torch thread each; code commit `6600176`; pinned software in the manifest |

Keeping the original grid isolates the change of target and tests whether 002's promising observations recur on different resets. Failure, contact, and inversion definitions were not relaxed. The project classifies physical failure because [HalfCheetah's own termination behavior](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#episode-end) does not detect unsuccessful locomotion. The exact operational thresholds are project pilot choices described in [methodology](../docs/methodology.md).

## Primary results

Baseline speed averaged **16.80795 m/s**, with zero measured physical failures. Every primary strength slowed the policy and cleared the target-size and interval criteria. Only the three moderate positive strengths retained sufficient mean forward speed; all strengths failed quality. This speed ratio is not a travel-distance or completion measure.

| Strength | Mean speed after steering, m/s | Speed change | Physical failures / 10 | Retains half baseline speed | Useful |
| --- | ---: | ---: | ---: | --- | --- |
| −0.5 | 0.77668 | −95.38% | 10 | No | No |
| −0.2 | 4.33044 | −74.24% | 3 | No | No |
| −0.1 | 7.93445 | −52.79% | 5 | No | No |
| −0.05 | 6.41489 | −61.83% | 7 | No | No |
| +0.05 | 13.94952 | −17.01% | 2 | Yes | No |
| +0.1 | 11.59412 | −31.02% | 1 | Yes | No |
| +0.2 | 9.26645 | −44.87% | 2 | Yes | No |
| +0.5 | 1.78256 | −89.39% | 7 | No | No |

At **+0.05**, the paired slowdown was **−2.85844 m/s**, with interval **[−4.02963, −2.19343] m/s**. Mean torso height increased from **0.592693 to 0.635091 m**. Episode-averaged forbidden contact was **3.556%**, but two individual episodes exceeded the physical-failure criterion. This repeats the lesson that a low across-episode mean cannot establish competent behavior in every trial. Unlike 002's sample, this fresh sample did not show uniformly clean outcomes at +0.05 or +0.2.

The study establishes sensitivity to reset episodes under a fixed intervention; it does not identify the dynamical mechanism responsible for every failure. These results are numerical trajectory measurements, without an additional video-based diagnosis in this report. No failed episodes were removed from means or bootstrap denominators.

## Controls and limitations

Two settings from the same random direction passed all speed-validation gates. At **speed_random0 +0.05**, mean speed was **15.81472 m/s**, a **5.91%** slowdown (paired effect −0.99323 m/s; interval [−1.09334, −0.90097]), with zero observed failure/contact/inversion. At **+0.1**, it was **14.97058 m/s**, a **10.93%** slowdown (effect −1.83737 m/s; interval [−2.04882, −1.66931]), with zero physical failures/inversion and **0.0111%** episode-averaged contact. This is 002's height_random0 array, not a newly drawn control.

The pipeline only shortlists a primary candidate; neither control was selected or confirmed here. Their exploratory passes are evidence against claiming unique effectiveness of contrastive extraction, and they remain possible candidates for a separately registered discovery procedure. Failure of the primary family does not establish that every hidden direction is useless.

These are 40 comparisons on ten validation seeds, with no correction of the reported confidence intervals for multiple comparisons. Hypothesis formation used 002; this was a fresh calibration study, not a locked confirmation of a previously successful candidate. One policy checkpoint and simulator limit generalization. The observed failure counts do not estimate a precise population failure probability, and the quality proxies cannot capture every gait defect.

The result is `no_validation_candidate`. No action-bias comparator was fitted because no primary candidate reached that stage. Reserved confirmation, replication, and application data remain unused, preserving their separation from this unsuccessful calibration.

## Next hypothesis and reproduction

The next hypothesis recorded after 003 was that **smaller signed strengths** of the unchanged 002 height vector could retain a meaningful speed effect with fewer physical failures. This was a dose-response hypothesis motivated by the adverse coarse-grid outcomes, not a claim that a working interval existed. Experiment 004 subsequently tested it with its [locally dated protocol and exact grid](hc-height-speed-004-preregistration.md), then [failed primary confirmation](hc-height-speed-004-findings.md). The control family and quality requirements remain relevant; a favorable random direction must not be omitted because it was called a control.

The new `retarget` command reproduces this experiment when `.env` matches its manifest, including seed offset 200000 and the original grid:

```powershell
uv run locomotion-steering retarget --source-run runs/hc-running-002 --run-dir runs/hc-height-speed-003 --vector height --behavior speed
uv run locomotion-steering report --run-dir runs/hc-height-speed-003
```

Use a new directory for a changed protocol. Full raw episodes and timestamped logs remain in the local run directory. The 003 raw run was published at [Hugging Face commit 238bd2d](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/commit/238bd2d9eba45fc1e69588f2beb2bbf15450f7b9); its numerical reports were published at [commit d4d4e76](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/d4d4e76cc3e9f5408bc348c9c85ba714b818b55b/experiments/hc-height-speed-003). The [bundle index](hc-height-speed-003/README.md) distinguishes the audit-file publication refresh. Publication does not change the recorded `no_validation_candidate` outcome. Later narrative corrections have their own publication history.

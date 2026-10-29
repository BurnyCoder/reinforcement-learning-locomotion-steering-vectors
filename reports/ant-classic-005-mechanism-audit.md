# Ant 005 path shift: estimand, mechanism wording and shared-baseline inference

**Historical saved-data audit.** The [original narrative snapshot](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/fc8cf6337e92fbbe0fba9746671e0afd55b8149c/reports/ant-classic-005-mechanism-audit.md) and numerical artifacts are retained. The [current generated findings](ant-classic-005-findings.md) include later evidence and bounded interpretation. Use [saved-evidence publication](../docs/methodology.md#saved-evidence-publication) for regeneration without replay. The later observation intervention has the [clipped-wrench limitation](../docs/methodology.md#calibration-controls-and-fresh-evaluation); neither this descriptive kinematics audit nor that comparator establishes a sole semantic mediator.

Audit date: 2026-10-16 UTC. Scope: a second rigor check of the featured claim. The claim is that the saved lateral vector at α = −0.1, added to the frozen Ant-v5 SAC actor's first ReLU, shifts the ground path toward world +Y. It is a post hoc analysis of saved data only: no policy inference, simulation, new seeds, selection changes or gate changes. The recorded status `confirmation_failed` is unchanged.

Four audits came before this one: [causal](ant-classic-005-causal-audit.md), [documentation](documentation-audit.md), [scientific-rigor](scientific-rigor-audit.md) and [world-Y specificity](ant-classic-005-y-shift-audit.md). This audit re-reads their scripts and asks which of their inferential steps were not themselves tested.

The historical saved-data audit command, run from the project root with its recorded source/dependency revisions and original `runs/` directories, is:

```powershell
uv run --no-sync python scripts/audit_ant005_path_mechanism.py
```

The [audit script](../scripts/audit_ant005_path_mechanism.py) runs five named phases over [tested helpers](../scripts/rigor_statistics.py) ([tests](../tests/test_rigor_audit.py)). It reuses:
- the y-shift audit's pairing and world-Y estimator;
- the production `_summary`;
- the hashing `EvidenceReader`.

It writes [mechanism-audit.json](ant-classic-005-mechanism-audit/mechanism-audit.json), [input hashes](ant-classic-005-mechanism-audit/input-identities.json) and a timestamped log in `reports/ant-classic-005-mechanism-audit/`. The recorded run, at commit `c20174f`, passed:
- 1,123 fields were compared with zero numerical error, including the recorded confirmation effect and the y-shift audit's travel angle, heading change and all eighteen even parts;
- every consumed input file was byte-identical afterwards.

A trial run before that commit gave identical numbers and was deleted.

**Integrity note.** I found these issues during read-only exploration, before writing the script. All comparisons are therefore descriptive. Intervals are nominal 95% paired percentile bootstraps (2,000 resamples, original seeds 710/720).

## Verdict

**No invalidating error was found in these checks.** The recorded locked intervention shifts the ground path toward world +Y on held-out resets for this policy.

A matched-horizon estimand compares the two arms only over steps both recorded. It gives **+0.392 m/s [+0.299, +0.477]**, with 28 of 29 pairs positive, against the recorded +0.402 m/s. Termination still determines the common window's length, so this is not a fixed-horizon effect after an episode ends.

**Four statements made by the [world-Y specificity audit](ant-classic-005-y-shift-audit.md) and the [scientific-rigor audit](scientific-rigor-audit.md) need correcting:**
1. **The "rigid rotation predicts 18.0 m of 17.4 m" check is arithmetic, not evidence.** A pure sidestep passes it equally well.
2. **The averaged kinematics are inconsistent with a rigid rotation.** Torso heading turns +5.23° but the path only +3.84°. Seen from the torso, travel direction shifts −1.39° [−2.03, −0.85]. This does not support increased body-relative sideways motion toward +Y on this aggregate measure.
3. **"Even part positive in 17 of 18 entries" counts one shared baseline 18 times.** Resampling seeds jointly, the evidence for a generic +Y drift is absent at |α| = 0.05 and marginal at 0.1.
4. **Pairing does not remove reset variability for this outcome.** Arms decorrelate after onset (median r = −0.04). Intervals remain valid. The sign-flip test needs its weak-null reading, under which the result still holds.

## 1. The displacement "rotation check" cannot tell rotation from sidestepping

The y-shift audit said a rigid rotation of the baseline path by the travel-angle change predicts the measured world-Y shift (18.0 m vs 17.4 m), and treated this as support for path rotation. Travel angle, however, is defined as the direction of net displacement, atan2(ΔY, ΔX). For any mechanism that leaves the forward distance roughly unchanged, ΔX · tan(Δangle) ≈ ΔY holds by construction.

The phase `rotation_check_discrimination` makes this concrete on the 23 pairs where neither arm terminated. It builds a synthetic **pure sidestep** for each pair: the baseline path plus a constant world-Y velocity that reproduces that pair's measured displacement change, with no change of heading or forward motion.

| Path | Travel-angle change | Predicted world-Y shift | Measured |
| --- | ---: | ---: | ---: |
| Real treated arms | +3.94° | 20.06 m | 19.51 m |
| Synthetic pure sidestep | +3.82° | 19.47 m | 19.51 m by construction |

The sidestep matches at least as well as the real data. A unit test ([test_rigor_audit.py](../tests/test_rigor_audit.py), `test_displacement_rotation_check_cannot_distinguish_rotation_from_sidestep`) demonstrates the same on synthetic gaits. Torso-heading measurements distinguish these two constructed explanations; the displacement identity alone does not.

## 2. The torso turns more than the path

The phase `heading_vs_travel` uses the 29 confirmation pairs with post-onset data in both arms. The new helper `body_relative_travel_change` returns the change in (travel angle − circular-mean torso yaw). Its two reference values:
- a rigidly rotated gait gives 0;
- a pure sidestep gives the whole path change.

| Per-pair change, degrees | Mean | 95% interval | Pairs > 0 | Sign-flip p |
| --- | ---: | --- | ---: | ---: |
| Travel direction | +3.84 | [+3.01, +4.58] | 28/29 | 5×10⁻⁶ (floor) |
| Torso heading (circular mean) | +5.23 | [+4.33, +6.09] | 28/29 | 5×10⁻⁶ (floor) |
| Travel relative to torso | **−1.39** | **[−2.03, −0.85]** | 6/29 | 2.5×10⁻⁵ |

"Floor" marks the smallest estimated p attainable with 200,000 Monte Carlo sign flips, 1/200,001: no sampled flip was as extreme. This finite-sampling estimate is not an upper bound on the exact p-value.

About 73% of the torso-heading change appears as path rotation. The rest is a small change in the direction of travel *relative to the body*, toward the side opposite +Y.

The descriptive check of rigid rotation used whether the interval excluded zero, and this one does. Both "the gait is rotated rather than converted into sidestepping" and "a rigid rotation" therefore overstate the case. The supported reading is narrower: the world +Y shift accompanies a heading change, with an opposite change in travel direction relative to the body. These averaged measurements do not establish a sole causal mediator or a complete gait-level explanation.

Both angles use each arm's available post-onset steps. Heading is time-weighted, while travel direction depends on displacement. The 96.8% mean-speed retention does not itself establish constant within-episode speed or equivalence of those weightings.

## 3. A shared baseline makes 18 even parts look like 18 pieces of evidence

Every validation direction is compared with the same ten unsteered episodes. Each even part, (f(+α) + f(−α))/2 with f = treated − baseline, therefore contains the same −baseline term.

The validation baseline mean, +0.125 m/s (SE 0.044, n = 10), happens to sit below the confirmation baseline, +0.181 m/s (n = 29). A baseline that is low by chance makes every even part positive at once. Agreement in sign across directions is consequently not independent evidence.

Odd parts cancel the baseline exactly. The label non-specificity finding, that the turning vector moves world Y more, is therefore unaffected.

The phase `common_mode_even_parts` averages the nine non-shuffled directions within each seed (both primaries, six isotropic draws, the action bias). It then bootstraps over seeds, which resamples every direction and the shared baseline jointly:

| \|α\| | Directions with even > 0 | Cross-direction mean even part, m/s | Seeds > 0 | Sign-flip p |
| --- | ---: | --- | ---: | ---: |
| 0.05 | 8/9 | +0.074 [−0.027, +0.164] | 7/10 | 0.19 |
| 0.1 | 9/9 | +0.108 [+0.020, +0.187] | 9/10 | 0.047 |

What this means:
- "Almost every perturbation pushes the Ant toward +Y" is not supported at |α| = 0.05 and is marginal at 0.1, on ten post hoc resets.
- For the selected condition, the sign-independent share of the +0.545 m/s validation effect is 26%, with a ratio interval of about [8%, 39%] (the recorded bootstrap seed gives [8.5%, 38.6%]; the lower bound ranges from 6.5% to 8.9% across bootstrap seeds). "About a quarter" is the point estimate of a poorly determined quantity.
- Confirmation tested only −0.1, so the even part cannot be estimated on fresh resets.

## 4. Pairing does not reduce variance for world Y, and what that means for the tests

The phase `arm_correlation` correlates each reset's baseline and treated episode means.

| Arms | World-Y r | Forward-speed r |
| --- | --- | --- |
| Validation, all 88 conditions: median (range) | −0.04 (−0.73 to +0.56) | −0.04 (−0.83 to +0.64) |
| Confirmation, locked lateral −0.1 | −0.40 | +0.26 |
| Confirmation, five locked controls | −0.08 to +0.18 | −0.34 to −0.03 |

A shared onset state does not keep the 900 post-onset steps aligned. A perturbation of the closed loop sends the trajectory down a different path; the measured correlations show that pairing need not reduce variance, without implying statistical independence. Monotonicity and continuity conditions can provide sufficient guarantees for common-random-number variance reduction ([Glasserman & Yao 1992](https://doi.org/10.1287/mnsc.38.6.884)); those conditions have not been established here.

For the locked primary, over the 29 pairs with post-onset data, the paired-difference SD is 0.249 m/s. Two independent arms would give √(0.122² + 0.173²) = 0.212. With all 30 pairs, the values are 0.256 and 0.237. Two consequences follow.

**What stays appropriate.** The paired bootstrap resamples whole reset pairs and therefore preserves observed within-pair dependence; it does not guarantee exact finite-sample coverage. The measured pairing correlation does not justify claiming that pairing narrowed this interval. The rigor audit's reason for comparing controls on shared seeds, "pairing removes reset variability", is unsupported for these outcomes. Its other reason still holds: comparing separate intervals for overlap is not a test of the difference.

**Which null the sign-flip test addresses.** Exact sign-flip calibration requires symmetry/exchangeability under the sign changes, not merely a zero mean. A perturbation can change trajectories while having zero average effect; chaotic divergence does not prove or disprove that distributional assumption. For a weak zero-mean null without symmetry, validity is asymptotic under the relevant regularity conditions (Romano 1990, *JASA* 85:686–692; reviewed by [Ritzwoller, Romano & Shaikh 2024](https://arxiv.org/abs/2406.09521)). Studentizing is a robustness device ([Chung & Romano 2013](https://arxiv.org/abs/1304.5939)), not a finite-sample guarantee for this dataset.

`sign_flip_pvalue(..., studentized=True)` gives p at its floor of 1/200,001 for the recorded effect (t = 8.61) and for the matched-horizon effect (t = 8.60). The conclusion does not change.

## 5. Additional checks that passed

- **Matched horizon.** World Y averaged over steps both arms recorded gives +0.392 m/s [+0.299, +0.477], 28/29 positive. Unequal durations do not drive the effect.
- **Independent isotropic draws.** The six isotropic directions are independent draws: the largest pairwise |cosine| is 0.126, about two SDs of an isotropic 256-dimensional cosine (1/16). This was checked during exploration and is not in the JSON. They come from separate behavior-specific random streams in `fit_vectors`.
- **Selection on the same data.** The y-shift audit's rank test (rank 1 of 7 at |α| = 0.1) uses the same validation resets that selected |α| = 0.1 for the primary. That favours the primary, so p = 1/7 is optimistic as well as underpowered.

## 6. Corrections applied

Each is a short qualifying edit that links here. No data, thresholds or statuses changed.
- [World-Y specificity audit](ant-classic-005-y-shift-audit.md): verdict items 2–3 and §3–4.
- [Scientific-rigor audit](scientific-rigor-audit.md): §7, the pairing rationale.
- [Ant findings](ant-classic-005-findings.md) and [experiment register](experiments.md).
- [README](../README.md).
- [Paper](paper.md) / [PDF](paper.pdf): §6 and §8.
- [Methodology](../docs/methodology.md): report arm correlations and jointly resampled even parts.
- [Sources](../docs/sources.md).
- [AGENTS.md](../AGENTS.md).

## 7. What was tried, and learnings

**Why these checks.** The earlier audits verified numbers and estimators. The interpretive sentences that sit on top of those numbers were not checked against an alternative mechanism. I asked of each one: what would it look like if the opposite mechanism were true?

- **Rotation check.** Writing travel angle as atan2(ΔY, ΔX) showed that the 18.0-vs-17.4 m agreement follows from the definition. The synthetic-sidestep construction turns that algebra into a falsification test the check fails to make. A heading-based estimand was then the natural replacement, because torso yaw is the one measurement that differs between the two mechanisms.
- **Heading vs travel.** The y-shift audit reported +3.84° travel and +5.23° heading side by side without differencing them. Their per-pair difference is the direct test of "rigid".
- **Even parts.** Writing the even part as ½(T₊ + T₋) − B showed the common −B term. Averaging within seed, then bootstrapping seeds, is the simplest analysis that propagates it. The alternative, a mixed model with a direction random effect, would add assumptions and code for ten seeds.
- **Pairing.** The confirmation paired SD (0.256 over 30 pairs, as in the y-shift audit's power section) exceeds the baseline SD (0.124 over 30 pairs; 0.122 over 29) by more than √2. That prompted the arm correlation, and it also changed which null the sign-flip test addresses.

**Not done in this audit.** No new rollouts. The [y-shift audit's follow-up proposal](ant-classic-005-y-shift-audit.md#limitations) called for fresh-seed specificity and even-part tests, with heading-relative travel and arm correlations. The later independent audit supplied additional evidence; the current findings describe its scope and the remaining limitations.

## Limitations

- Post hoc. The patterns were seen before the script recorded them.
- Even parts use ten validation resets.
- Heading and travel directions are time- and displacement-weighted averages, not a gait-phase analysis. No internal variable was shown to mediate the heading change.
- One pretrained policy.

# Ant 005 world-Y shift: label specificity, generic drift and mechanism

Audit date: 2026-10-16 UTC. Scope: whether the featured claim survives a scientific-rigor check. The claim is that adding the saved lateral vector at α = −0.1 to the frozen Ant-v5 SAC actor's first ReLU shifts the ground path toward world +Y. This is a post hoc analysis of saved data only: no policy inference, simulation, new seeds, selection changes or gate changes. The recorded status `confirmation_failed` is unchanged.

The [causal audit](ant-classic-005-causal-audit.md), [documentation audit](documentation-audit.md) and [scientific-rigor audit](scientific-rigor-audit.md) already reproduce every recorded number. None of them compared vector families, or looked at how the effect develops over the episode. This audit does both.

Reproduce from the project root, with the original `runs/` directories present:

```powershell
uv run --no-sync python scripts/audit_ant005_specificity.py
```

The [audit script](../scripts/audit_ant005_specificity.py) runs three named phases over [tested helpers](../scripts/rigor_statistics.py) ([tests](../tests/test_rigor_audit.py)). It reuses the production `_summary` and `paired_effect`, and the hashing `EvidenceReader` of the earlier audits. It writes [specificity-audit.json](ant-classic-005-y-shift-audit/specificity-audit.json), [input hashes](ant-classic-005-y-shift-audit/input-identities.json) and a timestamped log in `reports/ant-classic-005-y-shift-audit/`. The recorded run passed:
- 1,040 fields were compared, including all 88 recorded validation world-Y means;
- the maximum numerical error was 4.4×10⁻¹⁶;
- every consumed input file was byte-identical afterwards.

**Integrity note.** I first saw these patterns while doing read-only exploration for this audit. The script then re-derives them reproducibly. The check therefore has no prespecified hypotheses, and its comparisons are descriptive. Intervals are nominal 95% paired percentile bootstraps and are not adjusted for multiplicity.

## Verdict

**No methodological error invalidates the claim.** The claim holds as a causal, held-out, world-frame effect for this one frozen policy. The supporting checks:
- **Pairing.** Arms share onset-state hashes, use deterministic actions and have identical prefixes.
- **Order of steps.** Selection was locked before confirmation, and the seed splits are disjoint.
- **Statistics.** The sign-flip p-value is 5×10⁻⁶, and 28 of 30 pairs are positive.
- **Measurement.** World Y comes from root `qpos`, and the replays are bit-exact.
- **Figures and movies.** The top-down movie camera uses azimuth 90 and elevation −90 ([render_topdown.py:47](../scripts/render_topdown.py#L47)), so +X is right and +Y is up. The last frame of the fixed-camera movie shows the steered trail rising toward +Y while the baseline stays level. The XY plot uses equal axes.

**Three readings need correcting:**
1. **The lateral label is not what produces the +Y shift.**
   - The *turning* vector was fitted on yaw-rate labels and is nearly orthogonal to the lateral vector (cosine +0.114). It moves world Y more than the lateral vector at every |α| ≤ 0.2.
   - At −0.1 it would pass the lateral gate: +0.625 m/s, 94.4% speed retention, failures 1→1.
   - The isotropic-random comparison supports specificity only weakly: rank 1 of 7 at |α| = 0.1, p = 1/7.
2. **About a quarter of the selected shift is sign-independent.** Almost every perturbation, including random directions and the constant action bias, pushes the Ant toward +Y at either sign.
3. **The mechanism is a bounded heading offset.** On average the direction of travel rotates by +3.84°. It is not a sustained turn and not body-frame sidestepping. Individual paths still wander, as the seed-510000 movie shows.

## 1. The turning contrast moves world Y more than the lateral contrast

Every saved Ant direction has the same norm, 5.1513, the fitting activation RMS. All of them were run at ±0.05, ±0.1, ±0.2 and ±0.5 on the same ten validation resets (510000–510009). Any direction's world-Y response therefore splits seed by seed into a sign-dependent part, odd = (f(+α) − f(−α))/2, and a sign-independent part, even = (f(+α) + f(−α))/2. Here f is the treated-minus-baseline world-Y velocity (`cross_family_world_y`).

| Direction (cosine with lateral) | Odd world-Y at \|α\| = 0.1, m/s | \|odd\| minus primary's, m/s, paired 95% | Even at \|α\| = 0.1, m/s |
| --- | ---: | ---: | ---: |
| **Lateral primary** (1) | −0.403 [−0.441, −0.366] | — | +0.142 [+0.037, +0.238] |
| **Turning primary** (+0.114) | −0.596 [−0.644, −0.548] | **+0.193 [+0.143, +0.246]** | +0.028 |
| Lateral shuffled (+0.599) | −0.552 | +0.149 [+0.091, +0.203] | −0.058 |
| Turning shuffled (−0.409) | +0.338 | −0.066 [−0.168, +0.051] | +0.213 |
| Lateral random 0 / 1 / 2 (+0.055 / −0.007 / −0.067) | −0.203 / −0.209 / −0.143 | −0.200 / −0.195 / −0.261 | +0.016 / +0.071 / +0.210 |
| Turning random 0 / 1 / 2 (−0.050 / +0.003 / −0.050) | −0.138 / **+0.053** / **+0.045** | −0.265 / −0.350 / −0.358 | +0.116 / +0.143 / +0.152 |
| Constant action bias | +0.036 | −0.367 | +0.096 |

The turning vector's advantage over the primary at the other magnitudes:
- +0.149 [+0.075, +0.232] m/s at |α| = 0.05;
- +0.436 [+0.159, +0.699] m/s at |α| = 0.2.

Its yaw-rate response is negligible: the recorded turning effect at −0.1 is +0.0052 rad/s, far below the 0.1 rad/s gate. Applied to world Y, though, the lateral usefulness gate would pass the turning vector at four signed strengths: −0.1, +0.05, +0.1 and +0.2. The lateral vector itself passes at two.

What this means:
- Two nearly orthogonal contrast directions, fitted on different labels, both shift world Y. The one fitted on the *wrong* label shifts it more.
- So the effect does not identify a lateral-specific direction. At |α| = 0.1 all four contrast-derived directions (two primaries, two shuffles) have larger |odd| world-Y responses (0.338–0.596 m/s) than every isotropic draw (at most 0.209). They lie in the dominant activation subspace, whose participation ratio the rigor audit measured at 7.45.
- This sharpens the rigor audit's "partly specific" verdict. The vector beats isotropic noise, but not other directions built the same way.

**Falsifier.** The specificity reading would have been strengthened if other-label contrast directions had responded less than the primary. They responded more.

## 2. Six isotropic directions, not three: the coincidence resolves

The rigor audit noted, as unexplained, that all three lateral-family random directions share the primary's odd sign. The turning family contributes three more isotropic draws, with the same norm and the same seeds. Their odd world-Y responses at |α| = 0.1 are −0.138, +0.053 and +0.045: mixed signs and small. Across all six draws, signs agree with the primary four times out of six. That is consistent with chance; the earlier three-of-three result had probability 1/8.

Treating the six as exchangeable reference directions gives a rank test on |odd| (`isotropic_rank_tests`; p = (1 + #reference ≥ observed)/(k + 1), Barnard's Monte Carlo test):

| \|α\| | Primary \|odd\| | Largest isotropic \|odd\| | Rank p (1 of 7 is 0.143) |
| --- | ---: | ---: | ---: |
| 0.05 | 0.143 | 0.160 | 0.571 |
| 0.1 | 0.403 | 0.209 | 0.143 |
| 0.2 | 0.520 | 0.411 | 0.143 |
| 0.5 | 0.211 | 0.423 | 0.286 |

With six reference directions, p cannot fall below 1/7. Reaching p ≤ 0.05 needs at least 19 (`design_power`). The saved design therefore cannot test specificity against isotropic noise, even in principle. At |α| = 0.05 the primary is indistinguishable from random directions.

## 3. A sign-independent drift toward +Y

The even part is positive in **17 of 18** non-shuffled entries at |α| ≤ 0.1. Those entries cover both primaries, the six isotropic directions and the action bias. The single exception is the lateral primary at 0.05 (−0.005). Many individual even intervals include zero; the evidence is the consistent sign across directions, not any one interval. Perturbing this policy in almost any direction therefore tends to move it toward +Y, at either sign.

The baseline already drifts that way: +0.125 m/s in validation and +0.175 m/s in confirmation. For the locked condition, the even part is +0.142 of the +0.545 m/s validation effect, about 26%. The rest is the sign-dependent part, −(−0.403) = +0.403.

Confirmation ran only α = −0.1, so it cannot separate the two parts on fresh resets. The selected random controls' confirmation effects, +0.15 to +0.25 m/s, are consistent with this drift plus their chance-aligned odd parts.

## 4. Mechanism: a bounded heading offset that rotates the path

The phase `heading_time_course` uses the thirty confirmation pairs, with locked lateral −0.1 against baseline. Unwrapped torso yaw comes from saved pre-action quaternions. Each bin includes only pairs whose arms both cover it.

| Steps after reset | Pairs | Heading change, degrees, 95% | World-Y change, m/s | Yaw-rate change, rad/s, 95% |
| --- | ---: | ---: | ---: | ---: |
| 100–200 | 27 | +2.55 [+0.10, +4.73] | +0.197 | +0.0174 [+0.0067, +0.0278] |
| 200–400 | 27 | +7.03 [+5.36, +8.56] | +0.574 | +0.0010 [−0.0064, +0.0084] |
| 400–600 | 26 | +4.48 [+3.02, +6.04] | +0.354 | −0.0082 [−0.0187, +0.0023] |
| 600–800 | 25 | +4.20 [+2.85, +5.46] | +0.335 | +0.0093 [−0.0003, +0.0205] |
| 800–1000 | 23 | +6.24 [+4.70, +7.78] | +0.568 | +0.0015 [−0.0089, +0.0130] |

What the table shows:
- After a one-off yaw transient in the first 100 steps, the heading offset stays within roughly 4–7° and does not grow.
- If the onset yaw rate of +0.0174 rad/s had persisted, heading would have drifted about 45° by step 1000.
- The intervention therefore shifts the heading the closed-loop policy holds. It does not make the Ant turn continuously.

The direction of net post-onset travel rotates by **+3.84°** [+3.01, +4.58]: median +3.75°, 28 of 29 pairs positive, sign-flip p = 5×10⁻⁶. This is the most direct summary of the path shift, because it needs no torso-orientation convention.

A rigid rotation of the baseline path by that angle predicts the observed lateral shift. Baseline travels 267.6 m forward after onset; tan(3.84°) × 267.6 m = 18.0 m, against a measured world-Y displacement change of +17.4 m. The yaw-aligned side-velocity change, +0.059 m/s in the causal audit, is small. Together these say the *gait* is rotated rather than converted into sidestepping. Gait patterns themselves were not measured.

**Heading conventions.** The mean torso-heading change depends on how unequal episode durations are handled (`heading_change_deg_by_convention`, 29 pairs):
- +5.23°: circular mean over each arm's own steps, the rigor audit's convention;
- +5.68°: unwrapped mean over each arm's own steps;
- +5.37°: unwrapped mean over the steps both arms recorded.

Treated seed 520010 ends at step 193 after 21.5% inverted steps; while the Ant is inverted, torso yaw is ill-defined. The travel-angle estimand avoids both issues.

## 5. Design power

The threshold test was adequately powered (`design_power`, two-sided normal approximation at 80% power):
- The validation paired SD of 0.168 m/s, known before confirmation, implies about 6 pairs to detect the 0.2 m/s threshold.
- The confirmation SD of 0.256 m/s implies 13.
- Thirty pairs were used.

The approximation slightly underestimates exact paired-t sample sizes, which does not change this conclusion. Specificity was not testable, as Section 2 shows.

## 6. Corrections applied

Each is a short qualifying statement linked here. No data, thresholds or statuses changed.
- [Ant findings](ant-classic-005-findings.md), [scientific-rigor audit](scientific-rigor-audit.md) §2, [experiment register](experiments.md) and [README](../README.md): label non-specificity, the sign-independent drift and the path-rotation mechanism.
- [Paper](paper.md) / [PDF](paper.pdf): abstract, §6, §8 and reference [E7].
- [Methodology](../docs/methodology.md): future specificity nulls must include other-label contrast directions and at least 19 random directions per family at both signs.
- [Sources](../docs/sources.md): rank test and sample-size formula.
- [AGENTS.md](../AGENTS.md): compare primaries with other-label contrast vectors.

## 7. What was tried, and learnings

**Why these checks.** The claim names a vector, "lateral", and a frame, world +Y. Earlier audits tested the frame and the estimator, but the role of the name, i.e. the label specificity, was only tested against three random draws and one aligned shuffle.

Two cheap checks were possible because the saved validation grid already crosses every direction with world Y:
- outcomes the turning family never targeted;
- three additional isotropic draws.

The time course was added because a world-Y velocity effect is ambiguous on its own. It can come from a constant heading offset, which rotates the path, or from a sustained yaw rate, which gives a curving path. These two hypotheses predict different heading trajectories.

**How the turning result was found.** It came from tabulating world-Y odd/even parts for the turning family, whose recorded effects are yaw rates and had therefore never been read for world Y. It was the largest single surprise of this check.

**Why travel angle instead of torso heading.** Torso yaw is about 45° away from the direction of travel in this gait. It is undefined while inverted, and its episode mean depends on the duration convention (Section 4). The direction of net displacement has none of these problems.

**Not done.** No new rollouts were run. Validation has only ten resets, so the cross-family numbers are post hoc descriptions, not confirmatory evidence.

## Limitations

- Post hoc. The patterns were seen during exploration before the script recorded them.
- Cross-family odd/even parts use ten validation resets per sign, and only six isotropic directions exist.
- Intervals are nominal and not multiplicity-adjusted.
- One pretrained policy.
- The heading mechanism is descriptive. No internal variable was shown to mediate it.

**Next experiment.** A prespecified fresh-seed test would settle specificity. It should use new seeds, not the reserved replication seeds 530000–530029, and include:
- the lateral and turning vectors;
- at least 19 isotropic and at least 19 activation-covariance-shaped random directions, each orthogonalized to the primary;
- every direction at ±0.1;
- travel angle as the primary estimand, and odd/even decomposition.

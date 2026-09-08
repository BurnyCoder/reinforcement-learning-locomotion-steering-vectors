# HalfCheetah classic-vector fitting diagnostics

Inspected on 2026-09-08 before reviewing validation or confirmation outcomes. This report analyzes existing `runs/hc-classic-001/episodes/{diagnostic,fit}/baseline/*.npz` files; no policy rollouts were rerun. The checkpoint is the pinned HalfCheetah TQC policy recorded in this run's manifest.

The [manifest](hc-classic-001/manifest.json) records revision `6fb6381` retrospectively. Its [commit timestamp](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/commit/6fb638125edb71625c13dc4d11130bdc4c239e98), 22:57:14 UTC, falls within the logged 22:55:57–22:57:33 run. It is not an independently verified invocation-time source pin. The [documentation audit](documentation-audit.md) records this provenance limit and the terminology corrections below.

## Question and measurement

Do speed, effort, and height contrasts describe sustained running, or do a few recovery episodes determine their activation differences?

Measurements exclude the first 100 steps and use complete, non-overlapping 100-step windows. The implemented initial eligibility rule requires window inversion and inappropriate torso/head-ground contact fractions each to be at most 5%. All reported contrast-group means first average within contributing episodes, then across episodes, matching the implemented estimator. Diagnostic files contain physical measurements but no activations; activation-direction sensitivity below uses fitting files only.

The original [contrastive activation-addition method](https://arxiv.org/abs/2312.06681) motivates mean differences, but does not establish that arbitrary locomotion contrasts isolate a target. The [official HalfCheetah implementation](https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/half_cheetah_v5.py) returns raw generalized positions in observations and does not terminate merely because the agent falls. Consequently, completed episodes and a currently upright posture are insufficient evidence of uninterrupted competent running.

## Coverage and observed behavior

| Quantity | Diagnostic split | Fitting split |
|---|---:|---:|
| Episodes | 16, seeds 0–15 | 64, seeds 1000–1063 |
| Post-startup windows | 144 | 576 |
| Initially eligible windows | 124 | 517 |
| Episodes supplying eligible windows | 16 | 62 |
| Windows excluded for health proxies | 20 | 59 |
| Nonfinite windows / incomplete-tail steps | 0 / 0 | 0 / 0 |
| Eligible speed median, m/s | 14.607 | 14.593 |
| Eligible speed interquartile range, m/s | 13.983–15.163 | 13.899–15.192 |
| Eligible windows below 5 m/s | 5 | 10 |
| Eligible windows below 10 m/s | 5 | 15 |

Fitting episodes 1051 and 1053 supply no eligible windows. Seven fitting episodes exceed 5% inversion or bad contact over their complete post-startup interval: 1006, 1017, 1019, 1051, 1053, 1054, and 1062. Inversion fractions are 85.2% for 1017, 78.0% for 1019, 100% for 1053, and 75.3% for 1062. These observations demonstrate genuine failures within the fitting collection even though the environment does not mark them as task terminations.

Most eligible windows therefore represent rapid forward movement, but the health-only rule admits some nearly stationary windows following accumulated torso rotations.

## Initial contrasts

| Target | Low/high windows | Low/high contributing episodes | Low/high target means | Low/high speed, m/s |
|---|---:|---:|---:|---:|
| Speed | 130 / 130 | 52 / 51 | 12.226 / 15.496 m/s | 12.226 / 15.496 |
| Height | 130 / 130 | 56 / 57 | 0.60748 / 0.65378 m | 14.463 / 13.647 |

No episode supplies more than 5.4% of the low-speed group's windows, and none supplies more than 3.9% of either height group's windows. Episode balancing further limits unequal duration weights. However, broad contributor counts do not prevent unusual activation magnitudes from dominating a mean difference.

Fifteen of the 130 low-speed windows are below 10 m/s. Twenty-eight begin at step 100, compared with one high-speed window, showing a residual temporal difference. Height has a 4.63 cm contrast accompanied by approximately 5.6% lower speed in its high-height group; it is not a speed-matched posture contrast.

Effort is strongly associated with speed among initially eligible windows (Pearson correlation 0.848). Five speed quantile bins reduce this association but do not eliminate it. In the lowest bin, low/high effort groups have balanced mean speeds of 9.964 and 12.128 m/s, respectively, a 2.164 m/s mismatch. Other bins have absolute mean-speed differences of at most 0.061 m/s. The lowest-bin low-effort group contains 26 windows from 17 episodes, six from one episode. Calling this entire initial direction an effort-at-fixed-speed representation would overstate the evidence.

## Decisive activation sensitivity

Ten initially eligible fitting windows have near-zero mean speed. They come from seeds 1006 (start 700), 1015 (700), 1029 (700 and 900), and 1054 (100, 200, 300, 600, 800, 900). Their mean observed torso pitch is approximately 6.3–6.6 radians, or 25.2–25.6 radians for seed 1029: raw observation values preserve completed rotations even when the current body axis is upright. Several neighboring windows have substantial contact or inversion. This supports interpreting these windows as a distinct rotation-history/near-stationary regime, rather than simply the slow end of ordinary running. No videos of these exact fitting trajectories were reviewed, so a specific gait or dragging mechanism is not asserted.

To test influence without running new trajectories, recompute the same eligible-window quartile estimator after removing these ten windows. A floor of half the initial eligible fitting-window median speed, **7.2965 m/s**, removes exactly those windows while retaining five windows with speeds 8.48–9.82 m/s.

| Target | Initial raw-vector norm | Filtered raw-vector norm | Cosine, initial vs filtered direction | Filtered target contrast |
|---|---:|---:|---:|---:|
| Speed | 8.1466 | 1.1927 | 0.2975 | 2.5029 m/s |
| Height | 4.0843 | 1.2446 | 0.1134 | 0.04629 m |
| Effort | 7.4358 | 0.5072 | −0.5551 | 0.03245 mean squared action |

The ten removed windows are only 1.9% of initially eligible windows, yet removing them substantially rotates all three directions and reverses much of the effort direction. The result is not explained by the first post-startup window: removing steps 100–199 alone leaves all three direction cosines above 0.9995. A stronger 10 m/s floor yields the same qualitative sensitivity: cosines 0.225, 0.185, and −0.475 for speed, height, and effort.

After the median-based floor, 507 windows remain. The speed groups still contain 127 windows each, contributed by 51 episodes each, with means 12.998 and 15.501 m/s. The height groups contribute 55 and 56 episodes, respectively. Thus usable contrasts remain after excluding the near-stationary regime. Effort's lowest-bin speed mismatch remains 0.696 m/s, so even the revised contrast needs cautious interpretation and held-out speed-preservation checks.

## Decision supported by these observations

**Experiment `hc-classic-001` was not calibrated. Its initial fitting construction was rejected because near-stationary, accumulated-rotation observations materially influenced the activation directions.** The original vectors are not presented as clean speed, posture, or fixed-speed effort vectors.

The adopted follow-up, executed as **[`hc-running-002`](hc-running-002-findings.md)**, adds a fitting-only forward-speed rule: retain healthy windows moving **strictly above** half the health-eligible fitting median speed, then recompute extraction. The [002 manifest](hc-running-002/manifest.json) records `fit_min_speed_fraction=0.5`; the [filter implementation](../src/rl_locomotion_steering_vectors/analysis.py#L69) excludes equality as well as lower values. Every original trajectory and this diagnosis remain preserved; exclusion from fitting must not become exclusion of failed evaluation episodes.

This floor is an exploratory change motivated by fitting data, not a universally validated locomotion criterion. Its introduction did not itself change the evaluation gates. A separate quality-measurement correction was later made during 002 validation and is documented in the [002 findings](hc-running-002-findings.md). Height/speed covariance and residual effort/speed mismatch remain measured limitations rather than being assumed solved by filtering.

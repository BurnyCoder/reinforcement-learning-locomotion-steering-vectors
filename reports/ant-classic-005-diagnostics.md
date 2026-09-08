# Ant 005: fitting contrast inspection

Inspected 2026-09-08 UTC, before this inspection accessed any validation data. Scope: the 16 saved stochastic diagnostic episodes (500000–500015), 64 saved stochastic fitting episodes (501000–501063), and existing extraction metadata. Numerical detail is preserved in `artifacts/runtime-smoke/ant-classic-005-fitting-inspection.json`. No rollouts were rerun and no vectors, thresholds, or selections were changed.

**Recommendation:** proceed with the registered lateral-motion validation. Its fitting contrast reflects moving trajectories and survives the checks below. The turning vector can remain an exploratory causal test, but its labels provide substantially weaker evidence of sustained turning: they are consistent with alternating heading corrections. Neither fitted association establishes causal usefulness.

## Movement and episode endings

There are 531 complete 100-step fitting windows from 62 episodes, after the initial 100 steps. Each window lasts five seconds (`dt=.05`). All have zero measured inversion and torso-ground contact. Their mean forward speeds range from 4.621 to 6.996 m/s, above the fitting-only floor of 3.175 m/s. No window is excluded by that floor. This differs from the stationary recovery contamination identified in HalfCheetah 001.

The policy is not uniformly successful: 14/16 diagnostic episodes and 56/64 fitting episodes reach 1000 steps. All ten early endings have final torso height above 1 m, zero recorded inversion/contact, and `terminated=True`. This is consistent with Ant's upper height termination boundary, rather than evidence of falling or dragging; the pinned [Gymnasium 1.0.0 source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py) uses a healthy torso-height range of `(0.2, 1.0)` by default. Fitting seed 501030 ends at step 13 and seed 501044 at 164, so neither supplies a full post-startup window. The other six early-ending fitting episodes contribute 27 windows. Their incomplete tails total 309 steps across the fitting set and do not enter extraction. Failed episodes remain in the saved data and must remain in later evaluation denominators.

## Contrast quality

| Measurement | Lateral velocity | Yaw rate |
|---|---:|---:|
| Low / high contributing episodes | 57 / 58 | 56 / 58 |
| Low / high windows | 133 / 133 | 133 / 133 |
| Episode-balanced low / high means | −0.278 / +0.703 m/s | −0.0472 / +0.0436 rad/s |
| Behavioral contrast | 0.980 m/s | 0.0908 rad/s |
| Raw activation-vector norm | 0.4872 | 0.1886 |
| Saved split-half cosine | 0.9657 | 0.7610 |
| Most windows from one episode in either group | 6 | 4 |

Low/high forward speed differs by only 0.059 m/s for lateral and 0.043 m/s for turning. The first post-startup window supplies 13/133 low and 18/133 high lateral samples, and 18/133 low and 15/133 high turning samples. No single episode dominates direction: the worst cosine after leaving one contributing episode out is 0.9984 for lateral and 0.9920 for turning, with quartile thresholds recomputed in each sensitivity analysis.

| Sensitivity analysis, compared with the saved 100-step direction | Lateral cosine | Turning cosine |
|---|---:|---:|
| Remove the first post-startup window | 0.9964 | 0.9713 |
| Use only episodes completing 1000 steps | 0.9972 | 0.9808 |
| Trim the most extreme 1% of each behavior-label tail | 0.9971 | 0.9900 |
| Refit using 200-step windows | 0.9795 | 0.7264 |
| Refit using 300-step windows | 0.9489 | 0.7803 |

These are descriptive fitting checks, not substitute vectors or additional confirmation attempts.

## Temporal interpretation and limits

Lateral high-group windows have matching signs in their first and second 50-step halves 96.2% of the time; the low group does so 64.7% of the time. Lateral contrast remains 0.716 m/s over 200-step windows and 0.607 m/s over 300-step windows. Complete episodes span mean lateral velocities of −0.172 to +0.552 m/s. The 100-step lateral mean correlates with mean heading at 0.792, so the vector could relate to heading or diagonal movement; it is not evidence of an isolated sideways-motion representation. Adjacent window lateral means correlate only 0.106, so the evidence does not imply each episode maintains one constant lateral velocity.

Turning is more transient. Adjacent 100-step yaw-rate means correlate at **−0.505**. Only 55.6% of high and 57.9% of low windows retain the same turn sign across their two halves, and approximately 53% of individual steps share their window's sign. Its contrast falls from 0.0908 rad/s at 100 steps to 0.0520 at 200 and 0.0303 at 300. Complete episodes have post-startup mean yaw rates between −0.0113 and +0.00784 rad/s. These observations support an alternating correction/oscillation interpretation; no gait-phase identification or visual replay was performed, so a particular gait mechanism is not established.

As a metric check, yaw-rate recomputed from successive saved observation quaternions agrees with recorded yaw rate to a maximum absolute error of `1.79e-14` rad/s for every transition with a saved successor observation. This excludes treating a quaternion component as an angle as the source of the temporal pattern. The final transition of each episode lacks a successor observation and is outside that recomputation check.

# Validation inspection: speed vector at alpha −0.05

**Exploratory validation evidence, inspected 2026-09-08. No confirmation data, thresholds, vectors, calibration results, or selection rules were changed.**

The recorded aggregate slowdown is real, but the mean contact fraction conceals one severe physical failure. Nine of ten validation episodes sustain slower forward running without recorded head/torso contact. The remaining episode rotates and then stalls on its head for the final 20 simulated seconds. This condition should not be described as uniformly competent locomotion merely because it passes the existing pilot gate.

## Inputs and aggregate result

Inspected existing episode arrays from `runs/hc-running-002/episodes/validation/baseline` and `speed__-0.0500`, using paired seeds 110000–110009 and post-startup steps 100–999. The saved validation result reports mean velocity **16.7955 → 14.8188 m/s**, a **−11.769%** change. The paired episode-bootstrap effect interval is **[−3.5535, −1.1366] m/s**. Mean bad-contact fraction rises from zero to **4.4667%**, and mean inversion fraction rises from zero to **0.1%**. Its recorded pilot gate is `true`.

These aggregate quantities remain unchanged. The following inspection explains their distribution rather than substituting a different analysis.

| Seed | Baseline speed, m/s | Steered speed, m/s | Head/torso contact steps out of 900 | Contact fraction |
|---|---:|---:|---:|---:|
| 110000 | 16.776 | 15.646 | 0 | 0% |
| 110001 | 16.778 | 7.878 | 402 | 44.667% |
| 110002 | 16.780 | 15.707 | 0 | 0% |
| 110003 | 16.822 | 15.338 | 0 | 0% |
| 110004 | 16.807 | 15.511 | 0 | 0% |
| 110005 | 16.782 | 15.691 | 0 | 0% |
| 110006 | 16.794 | 15.480 | 0 | 0% |
| 110007 | 16.822 | 15.867 | 0 | 0% |
| 110008 | 16.802 | 15.610 | 0 | 0% |
| 110009 | 16.792 | 15.460 | 0 | 0% |

All recorded bad contact belongs to seed 110001. The pooled 4.47% fraction therefore describes a severe event in **one of ten episodes**, not occasional mild contact spread across ten episodes. Natural Gymnasium termination remains false; HalfCheetah does not terminate simply because locomotion becomes physically unsuccessful.

## Failure timing and geometry verification

For seed 110001, contact runs occur at zero-based steps 587, 594, and **600–999 inclusive**. The final run lasts 400 control steps, or **20 seconds** at the simulator's 0.05-second control timestep.

| Segment | Mean steered speed | Distance traveled | Mean bad-contact fraction |
|---|---:|---:|---:|
| Steps 100–499 | 15.427 m/s | 308.543 m | 0% |
| Steps 500–599 | 9.163 m/s | 45.814 m | 2% |
| Steps 600–999 | **0.00760 m/s** | **0.152 m** | **100%** |

Mean raw torso pitch changes from 0.042 radians in the first segment to 4.328 radians during the transition and 7.341 radians in the final segment, consistent with a completed rotation followed by a different support posture. The matching baseline continues at 16.931 m/s in the final segment.

Replayed the worst-contact seed 110001 and the prespecified typical seed 110000 using the saved speed vector, alpha −0.05, and deterministic prediction. Instrumentation observed contact records without modifying simulator state or the returned metric. **Every replayed array matched the cached array byte-for-byte**, including shapes, dtypes, actions, physical metrics, and intervention-onset hashes.

The actual loaded geometry names are floor=0, torso=1, and head=2, matching the [official HalfCheetah XML](https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/assets/half_cheetah.xml). The replay finds **401 head-floor contact steps and one torso-floor contact step** after onset. Contacts require the named floor/body pair and nonpositive signed distance, following [MuJoCo contact fields](https://mujoco.readthedocs.io/en/3.1.5/APIreference/APItypes.html#mjcontact). The most negative recorded distance is −0.01024 m. Foot contacts are excluded from this metric. The behavior is therefore not a geometry-indexing error or a foot-contact false positive.

Inspected frames at approximately 27.55, 29.40, 30.05, and 40.05 seconds in the worst-seed replay. They show the transition from a large rotation to a persistent nose-down posture. The typical seed's 30.05-second frame shows the ordinary horizontal running posture. Late frames have limited ground-texture cues under the default camera; named simulator contacts and displacement measurements establish head support and near-stalling more directly than appearance alone.

Replay MP4s, selected frames, and contact-step JSON records are stored locally under `artifacts/runtime-smoke/hc-running-002-validation/`, named by seed. They are inspection artifacts, not replacement evaluation episodes.

## Interpretation without changing the protocol

Among the other nine episodes, descriptive mean speed changes from **16.7974 to 15.5900 m/s (−7.188%)**, with zero recorded bad contact or inversion. This shows that a sustained slowdown exists beyond the catastrophic episode. This subgroup calculation is explanatory only: the failed episode remains in the official effect and all statistical denominators.

The selected pilot quality rule averages time fractions and can admit one long failure in a small sample. Consequently, passing that gate alone does not justify an unqualified claim of reliable or uniformly competent control. Retain this failure narrative alongside any subsequent held-out results, preserve the fixed confirmation protocol, and report per-episode failures and contact concentration when assessing usefulness. This inspection does not authorize changing a threshold after seeing validation outcomes.

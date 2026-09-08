# Validation inspection: speed vector at alpha −0.05

**Historical exploratory validation inspection, recorded 2026-09-08 local time (Europe/Berlin). This inspection itself did not change vectors, selection or analysis.** It describes the superseded 74-condition analysis. A subsequent, explicitly recorded correction during validation changed the derived episode-failure measurement before selection or confirmation; see the [old/new comparison audit](hc-running-002/quality-analysis-audit.json) and [documentation audit](documentation-audit.md).

The recorded aggregate slowdown is real, but the mean contact fraction conceals one severe physical failure. Nine of ten validation episodes sustain slower forward running without recorded head/torso contact. The remaining episode rotates and then stalls on its head for the final 20 simulated seconds. This condition should not be described as uniformly competent locomotion merely because it passes the existing pilot gate.

## Inputs and aggregate result

Inspected existing episode arrays from `runs/hc-running-002/episodes/validation/baseline` and `speed__-0.0500`, using paired seeds 110000–110009 and post-startup steps 100–999. The [superseded validation analysis](hc-running-002/validation_pooled-quality-v1.json) reports mean velocity **16.7955 → 14.8188 m/s**, a **−11.769%** change. The paired episode-bootstrap effect interval is **[−3.5535, −1.1366] m/s**. Episode-averaged bad-contact fraction rises from zero to **4.4667%**, and inversion fraction rises from zero to **0.1%**. Its historical pilot gate was `true`; the final corrected analysis records `false`.

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

All recorded bad contact belongs to seed 110001. The 4.47% mean first averages available timesteps within each episode, then weights episodes equally. It therefore describes a severe event in **one of ten episodes**, not occasional mild contact spread across ten episodes. Because these ten episodes have equal duration, the value also equals a pooled timestep fraction here; that equivalence does not hold generally. Natural Gymnasium termination remains false; HalfCheetah does not terminate simply because locomotion becomes physically unsuccessful.

## Failure timing and geometry verification

For seed 110001, contact runs occur at zero-based steps 587, 594, and **600–999 inclusive**. The final run lasts 400 control steps, or **20 seconds** at the simulator's 0.05-second control timestep.

| Segment | Mean steered speed | Forward displacement | Mean bad-contact fraction |
|---|---:|---:|---:|
| Steps 100–499 | 15.427 m/s | 308.543 m | 0% |
| Steps 500–599 | 9.163 m/s | 45.814 m | 2% |
| Steps 600–999 | **0.00760 m/s** | **0.152 m** | **100%** |

Forward displacement is the sum of saved forward velocity times the 0.05-second timestep over each listed segment. It is not total path length. Mean-speed retention elsewhere in the reports is also distinct from retained travel distance.

Mean raw torso pitch changes from 0.042 radians in the first segment to 4.328 radians during the transition and 7.341 radians in the final segment, consistent with a completed rotation followed by a different support posture. The matching baseline continues at 16.931 m/s in the final segment.

Replayed the worst-contact seed 110001 and the prespecified typical seed 110000 using the saved speed vector, alpha −0.05, and deterministic prediction. Instrumentation observed contact records without modifying simulator state or the returned metric. **Every replayed array matched the cached array byte-for-byte**, including shapes, dtypes, actions, physical metrics, and intervention-onset hashes.

The actual loaded geometry names are floor=0, torso=1, and head=2, matching the [official HalfCheetah XML](https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/assets/half_cheetah.xml). The replay finds **401 head-floor contact steps and one torso-floor contact step** after onset. Contacts require the named floor/body pair and nonpositive signed distance, following [MuJoCo contact fields](https://mujoco.readthedocs.io/en/3.1.5/APIreference/APItypes.html#mjcontact). The most negative recorded distance is −0.01024 m. Foot contacts are excluded from this metric. The behavior is therefore not a geometry-indexing error or a foot-contact false positive.

Inspected frames at approximately 27.55, 29.40, 30.05, and 40.05 seconds in the worst-seed replay. They show the transition from a large rotation to a persistent nose-down posture. The typical seed's 30.05-second frame shows the ordinary horizontal running posture. Late frames have limited ground-texture cues under the default camera; named simulator contacts and displacement measurements establish head support and near-stalling more directly than appearance alone.

Replay MP4s, selected frames, and contact-step JSON records are stored locally under `artifacts/runtime-smoke/hc-running-002-validation/`, named by seed. They are inspection artifacts, not replacement evaluation episodes.

## Interpretation without changing the protocol

Among the other nine episodes, descriptive mean speed changes from **16.7974 to 15.5900 m/s (−7.188%)**, with zero recorded bad contact or inversion. This shows that a sustained slowdown exists beyond the catastrophic episode. This subgroup calculation is explanatory only: the failed episode remains in the official effect and all statistical denominators.

The then-current pilot quality rule averaged per-episode time fractions and admitted one long failure in a small sample. Its historical pass did not justify an unqualified claim of reliable or uniformly competent control. The later [recorded measurement correction](hc-running-002/quality-analysis-audit.json) used this validation evidence before any selection or held-out evaluation; all 120 conditions were analyzed under that correction, while the original 74 estimates remained archived. This report preserves the failure evidence and the chronology, rather than presenting the superseded gate as the final outcome.

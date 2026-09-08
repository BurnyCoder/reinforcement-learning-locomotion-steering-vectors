# Featured result — ant-classic-005: lateral effect on fresh episodes

The frozen Ant policy shows a causal change in **world-ground lateral velocity** on both validation and held-out episodes. At the locked strength **−0.1**, the confirmation effect is **+0.40238 m/s** (paired 95% interval **[+0.30803, +0.49127]**), while mean forward speed retains **96.83%** of baseline. This measurement concerns the world's y axis; it does not establish body-frame sidestepping or an isolated internal concept.

The recorded status is `confirmation_failed`: a strict implementation rule rejects any episode ending before intervention. Seed **520025 ends at step 34 in both arms**, before steering begins at step 100. The failure count is **4/30 in both arms**, and all allowed increases in aggregate quality metrics are satisfied. Equal counts do not mean identical failed-seed identities or absence of intervention-induced failures. The shared pre-onset failure itself cannot have been caused by an intervention that had not started.

**Published evidence:** [downloadable vectors and controls](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/blob/1cd00cfcce13bcb86f1c370a71889ce6d34c278e/experiments/ant-classic-005/vectors.npz), [pinned manifest, selection and numerical results](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/1cd00cfcce13bcb86f1c370a71889ce6d34c278e/experiments/ant-classic-005), and the [independent causal/code audit](ant-classic-005-causal-audit.md). The downloaded vector archive was checked against its local SHA256. No extra policy training or behavioral cloning was used.

## Watch the same paired trajectory in three views

All three public attachments show baseline and lateral −0.1 for predetermined validation seed **510000**, the same existing 50-second simulated trajectory. They are explanatory media, not three independent tests or held-out confirmation episodes. The direct replay audit verifies numerical correspondence. World-ground lateral movement may reflect heading or diagonal progress; it does not demonstrate body-frame sidestepping.

**Tracking view** — useful for seeing the gait close up.

https://github.com/user-attachments/assets/676ef6c8-ade4-4f85-a0b7-01d7fcdb86c9

**Top-down view** — a world-oriented view from above.

https://github.com/user-attachments/assets/17286e19-c5a2-4ab9-96ab-f50444d494ca

**Fixed far-camera view with trails** — both panels use the same stationary camera and scale; the 20 m grid and labeled position rings/past-path trails show accumulated ground displacement.

https://github.com/user-attachments/assets/9e0719a6-c15b-474e-b74e-3c5e70954e2a

The movies and rendering audits are also preserved in the [pinned Hugging Face media release](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/865e68e9bf9eadc8b4efc20873e98f2d70e5e45a/experiments/ant-classic-005/videos).

## Acceptance status and complete records

No thresholds, selection, or data exclusions were changed after confirmation. All thirty pairs remain counted. No replication or application run was performed. The [complete bundle](ant-classic-005/README.md) preserves 88 validation and six confirmation conditions, the fixed selection, vectors, action-bias comparator, and PDF. The [locally dated protocol record](ant-classic-005-preregistration.md) and [fitting inspection](ant-classic-005-diagnostics.md) preserve earlier decisions. Their timestamp limits and later narrative corrections are recorded in the [documentation audit](documentation-audit.md); they are not independent registrations.

## Hypothesis, policy, and fitting

Ant was chosen because its free three-dimensional root supports lateral velocity and yaw rate, whereas planar HalfCheetah cannot test these outcomes. We used one frozen [Farama-Minari Ant-v5 SAC medium checkpoint](https://huggingface.co/farama-minari/Ant-v5-SAC-medium/tree/de5978d34b0118341df2b0a30232c5d2bfcb148e), with two 256-unit ReLU actor layers, 105 observations and eight actions. The first ReLU receives a fixed activation addition through [Baukit](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py). No behavioral cloning or policy training was performed.

Diagnostic seeds 500000–500015 were followed by fitting seeds 501000–501063. There were **531 eligible 100-step windows from 62 contributing episodes**, with centered activation RMS **5.15135**. No fitting speed-floor exclusion occurred. The lateral contrast was **0.980 m/s**, with split-half cosine **0.9657**; the yaw-rate contrast was **0.0908 rad/s**, cosine **0.7610**. Longer-window sensitivity supported sustained lateral variation more strongly than turning. The lateral label also correlated with heading, so the vector does not establish an isolated sideways-motion concept.

Eight fitting episodes ended early. All observed early endpoints had torso height above 1 m, consistent with Ant's [default healthy-height boundary](https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py); no observed fitting-window inversion or torso contact explained those endings. This matters because baseline competence is imperfect even without intervention.

## Protocol and validation

The original signed strengths ±0.05, ±0.1, ±0.2, ±0.5 were tested for lateral and turning candidates and each candidate's three random and one shuffled-label controls. Evaluation used ten paired seeds **510000–510009**, deterministic actions, a 100-step common prefix and 1,000-step horizon. Lateral usefulness requires at least **0.2 m/s** absolute change, an interval excluding zero, at least **80%** baseline forward speed, and the existing quality rules. Turning requires at least **0.1 rad/s** and 80% planar-speed retention. These are project pilot choices, not environment reward objectives.

Lateral **−0.1** passed validation: **0.12536 → 0.67037 m/s**, effect **+0.54501 m/s**, interval **[+0.44638, +0.63939]**. Forward speed changed **6.46976 → 6.31790 m/s**, and failure fell from one episode to zero. Lateral **+0.1** also passed, changing the outcome in the opposite direction: effect **−0.26147 m/s**, interval **[−0.37796, −0.13919]**, with zero treated failures. No turning primary setting passed all gates.

The larger passing primary lateral effect at −0.1 was locked for confirmation. A fitted constant action-bias comparator received the same eight-strength calibration grid. Total validation comprises **80 activation conditions plus eight action-bias conditions**, with eleven pilot passes: two primary lateral settings and nine random/shuffled settings. Selection and controls were fixed at **2026-09-08 00:09:03 UTC**. Confirmation used fresh seeds **520000–520029**; replication **530000–530029** remains unused.

## Held-out outcome and the exact gate failure

| Confirmation measurement, locked lateral −0.1 | Baseline | Steered |
| --- | ---: | ---: |
| Mean lateral velocity, m/s | 0.17472 | 0.57710 |
| Mean forward speed, m/s | 6.26183 | 6.06317 |
| Episode failures | 4/30 | 4/30 |
| Inversion time fraction | 0% | 0.71685% |
| Forbidden torso-contact fraction | 0% | 0% |
| Episodes ending before steering | 1/30 | 1/30 |

Effect size, interval, forward-speed retention, and the maximum five-percentage-point increases in failure/inversion/contact all pass. The additional requirement that both arms have **zero prefix-only episodes** alone makes `quality_pass` false for the primary. Seed 520025 ends at step 34 identically in baseline, primary, and every selected control, as verified in the [prefix audit](ant-classic-005/prefix-failure-audit.json). Its behavioral summary is zero in both arms under the project convention, while it still counts as a failure. Every pair remains counted; other episode summaries average available post-onset steps. For this shared early pair, retaining zero scales the observed paired effect by 29/30 relative to the other 29 pairs; this is not a generally validated missing-data correction or a fixed-horizon locomotion outcome.

The selected action bias gives **+0.13439 m/s**, below the lateral threshold. Random controls give **+0.15550**, **+0.25282**, and **+0.15472 m/s**; the shuffled control gives **+0.89039 m/s**. None passes the complete gate because every arm includes the shared prefix failure; some also fail size or increase failure. The shuffled control has nine failures versus four at baseline, so its larger effect does not establish equally competent control. The primary's positive effect should be distinguished from both a formal protocol pass and a claim of superiority to all generic directions.

## Visual evidence and interpretation

The first three predetermined validation seeds, **510000–510002**, were replayed with and without −0.1 steering. All six episodes complete 1,000 steps without measured inversion/contact. Paired videos are preliminary validation illustrations, not new independent evaluations. The media audit verifies exact replay-array equality, matching initial states and unsteered action prefixes. The tracking camera and finite ground texture can obscure world displacement late in the video; the accompanying XY plot shows **sideways motion on the ground, not torso height or jumping**. For seed 510000, mean lateral velocity changes from +0.073 to +0.547 m/s. See the media index under `runs/ant-classic-005/videos/preliminary/` and the cumulative [paper](paper.md). A completed top-down paired movie is at `runs/ant-classic-005/videos/topdown/lateral__-0.1000/seed-510000-preliminary-topdown-paired.mp4`; the scene-only ground improvement preserves the numerical trajectory.

Diagnosis and fitting record code commit `8ad6ce3`; validation/confirmation ran from `1b835f98446b86aa30c8ba3e50a6826b12ef4068`, recorded with source hashes in `20260908T000243916712Z.log`. The manifest migration retains the original coarse grid and does not imply that every phase used the initial manifest commit. Full phase logs remain part of the raw run provenance.

The requested fixed far-camera movie is also complete: `runs/ant-classic-005/videos/fixed-camera/lateral__-0.1000/seed-510000-preliminary-fixed-camera-paired.mp4`. Both panels use the same stationary camera and scale, with a 20 m ground grid and labeled position rings and past-path trails. These annotations aid visibility; they are not changed robot geometry. Its adjacent audit verifies observation, action, reward and metadata arrays against the existing validation rollouts, unchanged model geometry/material, and preserved simulation state. No fresh evaluation data were collected.

The [separate within-project arithmetic audit](ant-classic-005/math-audit.json) found that confirmation mean squared action rises from **0.24702 to 0.29108**, while episode-averaged original reward falls from **6.23937 to 5.86443** per available post-onset step. A post hoc yaw-aligned coordinate calculation gives **+0.05943 m/s** side-velocity change, compared with **+0.40238 m/s** in world Y. These measurements are consistent with a change in direction/heading, without proving a sole mediator. Gait-pattern preservation was not directly measured. The descriptive analysis did not change fitting, selection or acceptance. Speed retention is not retained travel distance or full-horizon completion; episode summaries use available post-onset durations.

The result supports a causal world-ground lateral shift on the tested frozen policy, including fresh held-out reset episodes. It does not establish a formally accepted or replicated useful intervention, a feedback-controlled target, cross-training-seed robustness, or an internal semantic representation. The strict prefix criterion and imperfect baseline competence limit the current acceptance procedure. A future protocol could distinguish baseline failures before intervention from intervention quality, but this experiment's rule and recorded outcome remain unchanged. The research priority here is documenting the existing effect, failures, controls, and media rather than starting another method.

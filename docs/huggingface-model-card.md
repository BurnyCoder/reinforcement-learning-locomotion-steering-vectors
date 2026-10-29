---
language: en
tags:
- reinforcement-learning
- activation-steering
- mujoco
- ant
- halfcheetah
- causal-intervention
---

# Steering vectors for RL locomotion

This repository publishes activation vectors, controls, numerical episodes, and reports from [the research code](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors). A fixed vector is added to a frozen RL actor's first ReLU output. **No behavioral cloning, imitation training, or new policy training was performed.** Upstream checkpoint weights are referenced, not redistributed.

This is an evidence and intervention-vector repository. It is not a fine-tune, adapter, quantization, or merge of the referenced policies, so its metadata omits `base_model`; those relationships have specific meanings in the [Hub metadata specification](https://huggingface.co/docs/hub/model-cards#specifying-a-base-model). The pinned upstream checkpoint links below identify the policies used.

The [Ant locomotion steering evidence collection](https://huggingface.co/collections/BurnyCoder/ant-locomotion-steering-evidence-6ae2900bb110a4be69bbfb0b) groups these artifacts with the upstream Ant checkpoint reference. The checkpoint does not provide a verified license grant; Baukit's package classifier and missing full license text also leave licensing unresolved. See the [source and license audit](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/docs/sources.md) before reuse. This card does not assign an upstream license.

## Featured result: Ant's ground path shifts toward world +Y

Adding the saved **256-dimensional lateral vector at strength -0.1** after the frozen Ant actor's first ReLU increases mean **world-Y ground-plane velocity by +0.402 m/s**, with a paired bootstrap 95% interval **[0.308, 0.491]**, on **30 confirmation pairs**. Mean forward-speed retention is **96.83%**. These are equal-episode summaries of available post-onset motion; they do not establish body-relative sidestepping. [Saved results](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/ant-classic-005/results.json) · [current findings](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/ant-classic-005-findings.md).

**The original outcome remains `confirmation_failed`.** Baseline and treated arms each have 4/30 failures, on partly different seeds. A shared episode ends before steering begins and triggers the strict prefix criterion. Mean squared action rises +17.84%, and mean post-onset environment reward decreases from 6.239 to 5.864 per step. Equal failure counts do not establish equal safety on every reset.

| Condition (strength) | Δ world-Y m/s | 95% interval | Forward speed retained | Squared-action change | Failures baseline → treated | Full gate |
| --- | --- | --- | --- | --- | --- | --- |
| lateral (-0.1) | +0.402 | [0.308, 0.491] | 96.83% | +17.84% | 4/30 → 4/30 | fail |
| lateral_bias (+0.5) | +0.134 | [0.076, 0.197] | 99.40% | +1.21% | 4/30 → 1/30 | fail |
| lateral_random0 (-0.05) | +0.155 | [0.095, 0.217] | 97.90% | -4.58% | 4/30 → 3/30 | fail |
| lateral_random1 (-0.1) | +0.253 | [0.195, 0.311] | 95.72% | +7.34% | 4/30 → 7/30 | fail |
| lateral_random2 (-0.05) | +0.155 | [0.089, 0.219] | 96.31% | -6.22% | 4/30 → 9/30 | fail |
| lateral_shuffled (-0.2) | +0.890 | [0.771, 0.980] | 88.31% | +14.12% | 4/30 → 9/30 | fail |

Each control uses its own validation-selected strength. The intervals above use the original paired percentile bootstrap; the baseline is shared. Larger target effects do not erase physical failures. Full per-seed trajectories and all validation settings are available in the [interactive evidence explorer](https://BurnyCoder.github.io/reinforcement-learning-locomotion-steering-vectors/).

**Later audit, separate cohort.** On 20 further resets, the locked vector produces **+0.462 m/s [0.343, 0.587]** (BCa 95% interval), with 19/20 positive pairs. Failures rise from 0/20 to 2/20. Against 99 norm-matched random directions its sign-dependent response has **comparison rank 0.05**; label specificity is unsupported. This audit consumed a previously reserved application cohort and is not the formal replication split. [Audit and ablations](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/ant-classic-005-independent-audit.md).

These ranks describe comparisons with the specified control ensembles. Treating them as calibrated p-values requires exchangeability under an explicit null; norm matching alone does not make a label-derived, selected primary exchangeable with random directions. See [statistical assumptions](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/docs/methodology.md#statistical-assumptions-and-rank-interpretation).

**Mechanism is qualified.** Projection ablations and a specified observation transformation support a heading-related account for this checkpoint. The transformation rotates already-clipped contact-force channels and clips again: `clip(R clip(f))` generally differs from physical rotation's `clip(R f)`. It does not establish exact physical-heading equivalence or a unique semantic mediator. The original activation-steering result is unchanged.

Regenerate the current Ant publication from saved data with `uv run --locked --no-sync python scripts/build_ant_publication.py`; add `--check` to verify generated outputs. No policy loading or new episode collection is needed. [Reproduction and completion ledger](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/docs/ant-publication.md) · [paper](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/paper.md) · [PDF](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/paper.pdf).

## Watch the same paired episode

All views show predetermined **validation seed 510000**, baseline left and steering right. They are illustrations of existing data, not fresh statistical evidence. The fixed camera uses a shared stationary frame, a 20 m checker grid, and labeled position-ring/path annotations; actual robot geometry and physics are unchanged.

<video controls src="https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/resolve/865e68e9bf9eadc8b4efc20873e98f2d70e5e45a/experiments/ant-classic-005/videos/fixed-camera/lateral__-0.1000/seed-510000-preliminary-fixed-camera-paired.mp4"></video>

[Fixed far-away camera](https://github.com/user-attachments/assets/9e0719a6-c15b-474e-b74e-3c5e70954e2a) · [Top-down tracking](https://github.com/user-attachments/assets/17286e19-c5a2-4ab9-96ab-f50444d494ca) · [Original tracking](https://github.com/user-attachments/assets/676ef6c8-ade4-4f85-a0b7-01d7fcdb86c9)

## Download and apply the vector

For saved-data publication, use the repository's locked installation and run `uv run --locked --no-sync python scripts/build_ant_publication.py`. Missing evidence is restored from the input manifest's immutable Hub revision `b262002001df6af2fd87dce82cc385f6b132e184`. The builder verifies complete file hashes before rendering and does not load the policy.

The original [vector archive](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/blob/1cd00cfcce13bcb86f1c370a71889ce6d34c278e/experiments/ant-classic-005/vectors.npz) and [manifest/selection](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/1cd00cfcce13bcb86f1c370a71889ce6d34c278e/experiments/ant-classic-005) remain pinned. The `lateral` array is already RMS-scaled: its norm is 5.15135, and strength -0.1 applies norm 0.51513. Load with `numpy.load(path, allow_pickle=False)` without renormalizing. The same archive includes activation controls; `action_biases.npz` is a separate action-space comparator.

The base checkpoint is [Farama-Minari Ant SAC medium](https://huggingface.co/farama-minari/Ant-v5-SAC-medium/tree/de5978d34b0118341df2b0a30232c5d2bfcb148e), SHA256 `dea0e6dbb847776ac9e91dc8b7df5ba975742924eb55831d2da08a0abc786ca2`. It has 105 observation coordinates, 8 actions, and 256 first-layer ReLU units. The intervention begins after 100 steps. Upstream weights are referenced rather than repackaged. [Video and simulator reproduction](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/docs/ant-videos.md) requires the corresponding historical source revision; current source changes must not be hidden by rewriting old cache identities.

## Research record

| Experiment | Observed result |
| --- | --- |
| `hc-classic-001` | Fitting rejected because a few recovery windows dominated the contrasts; no validation |
| `hc-running-002` | No primary setting passed all original behavioral gates; posture changes came with excessive slowdown |
| `hc-height-speed-003` | Retargeted height direction failed physical quality on fresh validation |
| `hc-height-speed-004` | Locked smaller-strength slowdown failed confirmation with 10/30 treatment failures |
| `ant-classic-005` | World-Y effect on 30 confirmation pairs and 20 audit resets; original `confirmation_failed`; no formal replication |

The [experiment register](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/experiments.md), [paper source](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/paper.md), and [PDF paper](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/resolve/main/paper/paper.pdf) cover all attempts and the later audits. Complete data reside under `experiments/`; the paper is a research snapshot, not a peer-reviewed publication or a declaration that the original discovery objective has been met. Hub files under `documentation/` and `paper/` are publication copies; the [completion ledger](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/ant-publication-completion.md) records their versions and verification receipts. The earlier [fact-check](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/fact-check-2026-10-22.md) remains a historical audit.

HalfCheetah attempts used the separately pinned [TQC medium checkpoint](https://huggingface.co/farama-minari/HalfCheetah-v5-TQC-medium/tree/b4ce04da6f246f06ae4c4258b8ab39624e6600b4), archive SHA256 `8231919095d5a35a2b799e865e968feca535d5af3f40a4b092947c70a976b7e1`. This project evaluates one checkpoint per policy; both archives record training seed 0. The upstream [training loop](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/train.py#L135-L173) and [upload script](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/upload_model.py#L23-L47) support RL provenance. The upload is the `EvalCallback` best model of the seed-0 run, not its final model. The complete upstream training history was not independently reproduced. Many reset episodes do not establish robustness across independently trained policies. Mean squared action is an effort proxy, not physical energy.

---
language: en
tags:
- reinforcement-learning
- activation-steering
- mujoco
- ant
- halfcheetah
- causal-intervention
base_model:
- farama-minari/Ant-v5-SAC-medium
- farama-minari/HalfCheetah-v5-TQC-medium
---

# Steering vectors for RL locomotion

This repository publishes activation vectors, controls, numerical episodes, and reports from [the research code](https://github.com/BurnyCoder/rl-locomotion-steering-vectors). A fixed vector is added to a frozen RL actor's first ReLU output. **No behavioral cloning, imitation training, or new policy training was performed.** Upstream checkpoint weights are referenced, not redistributed.

## Featured result: Ant ground-plane trajectory steering

The **256-component `lateral` vector at alpha −0.1** increases world-Y velocity by **+0.40238 m/s**, paired 95% interval **[+0.30803, +0.49127]**, on **30 fresh paired confirmation episodes**, while retaining **96.83%** of baseline mean forward speed. These are equal-episode averages of each episode's available post-onset mean velocities; 96.83% is a ratio of mean X velocities, not total distance retention ([recorded estimator](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/d0cd2c07e3641531c39037223eb98e309f44c4dc/src/rl_locomotion_steering_vectors/analysis.py#L194-L261)). This is motion in the world's ground plane, with heading involvement; it does not establish body-relative sidestepping or jumping.

The recorded outcome remains **`confirmation_failed`**: an additional strict rule rejects either arm containing an episode with no post-onset transitions (`length <= warmup`). Seed 520025 ends identically at step 34 in both arms, before steering begins after 100 steps. It remains in both denominators, counts as a failure, and has zero assigned to its unavailable post-onset outcomes; equal early failures do not bypass the strict veto. Failure counts are 4/30 in both arms, on partly different seeds. Mean squared action rises 17.84%, and original reward decreases. No independent replication or practical-target application was performed. [The findings](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/ant-classic-005-findings.md) and [causal audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/ant-classic-005-causal-audit.md) preserve the controls, costs, and limitations.

The audit independently reproduced the vector and controls from fitting data, checked disjoint seed partitions, matched 28 actions to a separately reconstructed SAC forward pass, and replayed 2,000 transitions through the unchanged Ant environment. It detected no scripted sideways motor rule, reward substitution, policy learning, or camera-induced physics change. This supports a causal trajectory effect for this policy, not an isolated internal semantic concept.

## Watch the same paired episode

All views show predetermined **validation seed 510000**, baseline left and steering right. They are illustrations of existing data, not fresh statistical evidence. The fixed camera uses a shared stationary frame, a 20 m checker grid, and labeled position-ring/path annotations; actual robot geometry and physics are unchanged.

<video controls src="https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/resolve/865e68e9bf9eadc8b4efc20873e98f2d70e5e45a/experiments/ant-classic-005/videos/fixed-camera/lateral__-0.1000/seed-510000-preliminary-fixed-camera-paired.mp4"></video>

[Fixed far-away camera](https://github.com/user-attachments/assets/9e0719a6-c15b-474e-b74e-3c5e70954e2a) · [Top-down tracking](https://github.com/user-attachments/assets/17286e19-c5a2-4ab9-96ab-f50444d494ca) · [Original tracking](https://github.com/user-attachments/assets/676ef6c8-ade4-4f85-a0b7-01d7fcdb86c9)

## Download and apply the vector

Use the source repository's pinned `uv.lock` and run from its root:

```powershell
uv run hf download BurnyCoder/rl-locomotion-steering-vectors experiments/ant-classic-005/vectors.npz experiments/ant-classic-005/manifest.json experiments/ant-classic-005/vector_diagnostics.json experiments/ant-classic-005/selection.json --revision 1cd00cfcce13bcb86f1c370a71889ce6d34c278e --local-dir artifacts/published
uv run locomotion-steering replay --run-dir artifacts/published/experiments/ant-classic-005 --vector lateral --alpha -0.1 --seed 510000
```

The `lateral` key in [vectors.npz](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/blob/1cd00cfcce13bcb86f1c370a71889ce6d34c278e/experiments/ant-classic-005/vectors.npz) is already RMS-scaled. Load with `numpy.load(path, allow_pickle=False)`; do not normalize again. Its norm is approximately 5.15135, so the applied offset has norm 0.515135. The non-executable bundle also preserves random/shuffled controls. `vector_diagnostics.json` contains raw directions and scaling, while `manifest.json` identifies the checkpoint, layer, software and seed splits. `action_biases.npz` is a separate comparator, not the activation vector. The published vector archive was downloaded back and verified against SHA256 `3626c79cb49539ef0c5170a6cabdbd9183d3800aeec82ab9196a8804764df36d`.

The pinned base checkpoint is [Farama-Minari Ant SAC medium](https://huggingface.co/farama-minari/Ant-v5-SAC-medium/tree/de5978d34b0118341df2b0a30232c5d2bfcb148e), archive SHA256 `dea0e6dbb847776ac9e91dc8b7df5ba975742924eb55831d2da08a0abc786ca2`. It has 105 observations, eight actions and two 256-unit ReLU hidden layers. The intervention site is `actor.latent_pi[1]`. The default `Ant-v5` environment, deterministic evaluation, and the 100-step common prefix are retained. [Reproduction instructions](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/docs/ant-videos.md) also cover all three videos and rendering audits.

## Research record

| Experiment | Observed result |
| --- | --- |
| `hc-classic-001` | Fitting rejected because a few recovery windows dominated the contrasts; no validation |
| `hc-running-002` | No primary setting passed all original behavioral gates; posture changes came with excessive slowdown |
| `hc-height-speed-003` | Retargeted height direction failed physical quality on fresh validation |
| `hc-height-speed-004` | Locked smaller-strength slowdown failed confirmation with 10/30 treatment failures |
| `ant-classic-005` | Held-out world-Y effect reproduced; strict pre-onset gate failed, no replication |

The [experiment register](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/experiments.md), [paper source](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/paper.md), and [PDF paper](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/resolve/main/paper/paper.pdf) cover all attempts. Complete data reside under `experiments/`; the paper is a research snapshot, not a peer-reviewed publication or a declaration that the original discovery objective has been met.

HalfCheetah attempts used the separately pinned [TQC medium checkpoint](https://huggingface.co/farama-minari/HalfCheetah-v5-TQC-medium/tree/b4ce04da6f246f06ae4c4258b8ab39624e6600b4), archive SHA256 `8231919095d5a35a2b799e865e968feca535d5af3f40a4b092947c70a976b7e1`. This project evaluates one checkpoint per policy; both archives record training seed 0. The upstream [training loop](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/train.py#L135-L173) and [upload script](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/upload_model.py#L23-L47) support RL provenance and seed-0 checkpoint selection; the complete upstream training history was not independently reproduced. Many reset episodes do not establish robustness across independently trained policies. Mean squared action is an effort proxy, not physical energy. The [documentation audit](https://github.com/BurnyCoder/rl-locomotion-steering-vectors/blob/main/reports/documentation-audit.md) records claim checks and their limits without changing the original experimental evidence.

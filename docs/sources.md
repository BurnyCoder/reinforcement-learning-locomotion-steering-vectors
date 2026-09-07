# Sources and reuse audit

Audited September 2026. Links below distinguish what the source establishes from this project's adaptation. Runtime compatibility claims require actual local checks; a generic API signature alone is insufficient.

## Scientific method and RL policies

| Source | Reused or established | Project-specific adaptation |
| --- | --- | --- |
| [Contrastive activation addition](https://arxiv.org/abs/2312.06681) | Difference-of-activation-means steering methodology for language models | Locomotion descriptors, episode weighting, physical effects, and causal evaluation |
| [HalfCheetah TQC medium checkpoint](https://huggingface.co/farama-minari/HalfCheetah-v5-TQC-medium/tree/b4ce04da6f246f06ae4c4258b8ab39624e6600b4) and [Minari provenance](https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/) | A pretrained online-RL checkpoint and its training context | Frozen-policy experiments, without using demonstrations to fit actions |
| [Ant SAC medium checkpoint](https://huggingface.co/farama-minari/Ant-v5-SAC-medium/tree/de5978d34b0118341df2b0a30232c5d2bfcb148e) | A pretrained SAC policy | Directional locomotion interventions |
| [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/sac.html) and [TQC](https://sb3-contrib.readthedocs.io/en/v2.4.0/modules/tqc.html) | Loading and ordinary policy inference | Choosing an actor activation as the injection site |
| [Gymnasium HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) and [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) | Environment observations, actions, reward structure, and termination semantics | Window contrasts and useful-behavior thresholds |
| [MuJoCo state API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-getstate) | Simulator state serialization | Hashing paired intervention onset states |
| [NumPy random generator](https://numpy.org/doc/1.26/reference/random/generator.html) | Reproducible pseudorandom draws | Paired episode bootstrap and norm-matched random controls |

## Why Baukit is the initial intervention dependency

[Baukit's pinned Trace implementation](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py) accepts an arbitrary PyTorch module, captures output, allows a differentiable `edit_output` callable, and removes its forward hook on context exit. That interface fits repeated ordinary SB3 predictions without tokenization or a different model-execution API. This project reuses that implementation and owns only the locomotion-specific wrapper and experiment logic.

[Package metadata](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/setup.cfg) advertises Python >=3.7, OS independence, and an MIT classifier. The repository does not supply a full root LICENSE file, and GitHub's license detector returns no recognized license. This is documented provenance, not a claim that license text was verified. Its main commit was last updated in February 2024. Package import also loads Torchvision, so Torch and Torchvision are pinned together.

Alternative libraries were inspected before selecting the integration:

| Library | Verified capability | Why not the initial backend |
| --- | --- | --- |
| [NNsight](https://nnsight.net/getting-started/quickstart/) | Explicit support for arbitrary PyTorch modules and tensor inputs; MIT | Its [persistent edits](https://github.com/ndif-team/nnsight/blob/main/docs/usage/edit.md) run during later NNsight traces, requiring an inference-dispatch adaptation; it adds broader dependencies and [patches tensor backward](https://nnsight.net/features/3_gradients/) |
| [pyvene](https://github.com/stanfordnlp/pyvene) | Generic interventions and built-in [activation addition](https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/interventions.py); Apache 2.0 | Useful for richer trainable/composed interventions, but requires more model-adapter and dependency work for the first frozen MLP experiment |
| [steering-vectors](https://github.com/steering-vectors/steering-vectors) | CAA extraction and injection for language models; MIT | [Training](https://github.com/steering-vectors/steering-vectors/blob/main/steering_vectors/train_steering_vector.py) consumes text/tokenizers; [injection](https://github.com/steering-vectors/steering-vectors/blob/main/steering_vectors/steering_vector.py) assumes token-shaped activations, so custom layer names alone do not make it an SB3 backend |

PyTorch's own [forward-hook API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) is the minimal maintained fallback if the selected dependency proves unsuitable. Avoid rebuilding a general intervention framework.

## Experiment infrastructure and reporting

- [uv projects](https://docs.astral.sh/uv/guides/projects/) supplies project initialization, local environments, dependency locking, and execution.
- [Python logging](https://docs.python.org/3.12/library/logging.html) supplies shared console/file handlers; logging format and research messages belong to this project.
- [NumPy compressed NPZ](https://numpy.org/doc/1.26/reference/generated/numpy.savez_compressed.html) supplies array artifacts.
- [Matplotlib plotting](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html) and [saved figures](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html) supply strength-response charts.
- [ReportLab Platypus](https://docs.reportlab.com/reportlab/userguide/ch5_platypus/), [paragraphs](https://docs.reportlab.com/reportlab/userguide/ch6_paragraphs/), and [tables](https://docs.reportlab.com/reportlab/userguide/ch7_tables/) supply pagination and PDF layout. Reports read recorded results; the renderer performs no new selection or statistical tests.
- [pytest](https://docs.pytest.org/en/stable/) supplies tests for numerical contracts, simulator measurements, pairing, and cleanup.

## Related work reserved for later experiments

[Policy Gradient Steering](https://arxiv.org/abs/2607.27574) is a candidate escalation when natural contrast exists but classic vectors do not control the behavior. Its relevance does not mean this repository implements it yet. Continuous-action log probabilities and any trust-region/regularization machinery must be checked against the actual SAC/TQC distribution; categorical-language-model code is not directly interchangeable.

Command-conditioned HalfCheetah velocity is a possible online-RL training experiment, informed by [existing velocity-task wrappers](https://ut-austin-rpl.github.io/amago/_modules/amago/envs/builtin/half_cheetah_v4_vel.html). Training rewards, convergence criteria, and extraction pairs must be specified before that experiment and documented as adaptations.

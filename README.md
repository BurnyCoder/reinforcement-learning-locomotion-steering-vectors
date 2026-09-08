# RL locomotion steering vectors

This project tests whether adding a constant vector to a frozen reinforcement-learning policy's hidden activations can produce useful changes in locomotion. The first implementation collects unsteered rollouts, extracts contrasting activation means, selects intervention strengths on validation episodes, and checks them on fresh paired episodes. It uses **no behavioral cloning** and does not update the pretrained policy's weights.

The scientific objective is a useful, repeatable causal effect, rather than a particular speed change. HalfCheetah candidates concern speed, torso height, and action effort; Ant also enables lateral movement and turning. A nearly constant behavior is an extraction diagnostic, not proof that hidden-state steering is impossible. See [the experiment protocol](docs/methodology.md) for the contrast rules, controls, usefulness thresholds, and limitations. [The experiment register](reports/experiments.md) records measured progress; planned capabilities are not results.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, clone this repository, and run these commands from its root. The project uses Python 3.12 and a local `.venv`; `uv` resolves the interpreter and installs the dependencies specified by `uv.lock`. Both pinned policies and the Baukit intervention have been exercised on native Windows with CPU inference, and an Ant rollout video has been checked. Other operating systems require their own runtime verification.

```powershell
git clone https://github.com/BurnyCoder/rl-locomotion-steering-vectors.git
cd rl-locomotion-steering-vectors
uv sync --locked
Copy-Item .env.example .env
uv run locomotion-steering run --model halfcheetah --run-dir runs/halfcheetah-classic-001
uv run locomotion-steering report --run-dir runs/halfcheetah-classic-001
```

The first run downloads the pinned public checkpoint, so internet access is required. Local experiment settings and episode counts are in `.env`; `.env.example` is the configuration reference. Current defaults include four rollout workers, one Torch thread per worker, and a fitting speed floor of half the healthy fitting-window median. `STEERING_SEED_OFFSET` reserves fresh partitions for a new hypothesis. Keep secrets out of both tracked configuration and logs. A run performs actual simulator rollouts, including evaluation of random and shuffled-label controls, so it takes longer than a single policy demonstration.

The `run` command writes numerical evidence under the supplied run directory. The separate `report` command renders `report.md`, `report.pdf`, and validation strength curves from that evidence. A completed run can legitimately report no eligible vectors or a failed confirmation. These outcomes are research findings, not useful steering successes.

```powershell
uv run locomotion-steering run --model ant --run-dir runs/ant-classic-001
uv run locomotion-steering replay --run-dir runs/halfcheetah-classic-001 --vector speed --alpha 0.1
uv run locomotion-steering replay --run-dir runs/halfcheetah-classic-001 --vector speed --alpha 0.1 --off-at 700
uv run locomotion-steering report --run-dir runs/halfcheetah-classic-001
uv run pytest
```

Replay requires the named activation vector to have been extracted; inspect `vectors.npz` and `vector_diagnostics.json` or the report before choosing one. The example strength `0.1` is a demonstration argument, not a recommendation or a claim that it improves locomotion. Replay writes an MP4 and matching numerical NPZ under `videos/`; `--off-at` removes steering later in the episode, and `--seed` chooses the reset (default `40000`). Action-bias comparators are evaluated by the research pipeline and are not activation-vector replay inputs.

Re-run the same `run` command to resume completed episode artifacts under the saved model, configuration, and seed splits. Use a new run directory when changing the protocol. `--stop-after diagnose`, `extract`, `calibrate`, or `confirm` provides an inspection boundary. Report regeneration reads saved evidence without changing selection or running new evaluations.

To test an existing direction against a different behavioral outcome, use `retarget` with a separate run directory and a fresh `STEERING_SEED_OFFSET` in `.env`. For example, experiment 003 uses offset `200000` and the following command:

```powershell
uv run locomotion-steering retarget --source-run runs/hc-running-002 --run-dir runs/hc-height-speed-003 --vector height --behavior speed
```

This reuses the exact source vector, its complete random/shuffled control family, and source fitting evidence. It recalibrates the new outcome on fresh episodes, preserving the source mapping in `retarget.json`. In the new bundle, vector keys use the target name: `speed` here identifies the imported height direction. Keep the source run's fitting trajectories available because shortlisted candidates require them for action-bias comparison. The [experiment register](reports/experiments.md) links the actual outcomes and the uploaded 002 evidence.

After a speed candidate has passed confirmation and replication, `uv run locomotion-steering demonstrate --run-dir runs/your-replicated-run` tests its fixed, validation-derived speed target on ten new episodes. It restores the saved configuration, records numerical targeting and physical-quality checks, and renders the first three paired videos plus an off/on/off example. The command rejects runs without a replicated speed candidate. No completed experiment through 003 qualifies yet; this is an implemented workflow, not an application result. Exact acceptance rules are in [methodology](docs/methodology.md#fixed-speed-application).

## What happens during a run

```mermaid
flowchart TD
    CLI[CLI and .env configuration] --> PLAN[Manifest and disjoint seed splits]
    PLAN --> LOAD[Pinned pretrained RL checkpoint]
    LOAD --> BASE[Unsteered diagnostic and fitting episodes]
    BASE --> FIT[Eligible behavior windows and contrastive mean vectors]
    FIT --> CONTROLS[Norm-matched random and shuffled-label vectors]
    CONTROLS --> VALIDATE[Paired validation strength sweep]
    VALIDATE --> BIAS[Fit and calibrate constant action-bias comparators]
    BIAS --> LOCK[Saved selection before confirmation]
    LOCK --> CONFIRM[Fresh paired confirmation episodes]
    CONFIRM --> REPLICATE[Fresh episode replication]
    REPLICATE --> RESULTS[Saved numerical results]
    RESULTS --> REPORT[Explicit report command: plots, Markdown and PDF]
    SOURCE[Saved family and source fitting evidence] --> RETARGET[Retarget: new outcome and fresh seed partitions]
    RETARGET --> VALIDATE
    REPLICATE --> DEMO[Demonstrate: fixed speed target on fresh episodes]
    DEMO --> MEDIA[Application measurements and paired videos]
    BASE --> STORE[Compressed episode artifacts and timestamped logs]
    VALIDATE --> STORE
    CONFIRM --> STORE
    REPLICATE --> STORE
```

The thin pipeline coordinates reusable configuration, checkpoint/rollout, analysis, storage, and reporting modules. The first actor ReLU receives the intervention through the existing [Baukit tracing utility](https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py). Ordinary Stable-Baselines3 prediction continues to handle action inference. The intervention is removed after use, and tests check that the original policy function is restored.

## Reproduce and inspect

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Model revision/hash, software, configuration, seed splits, and research decisions |
| `episodes/` | Compressed per-episode measurements and fitting activations |
| `vectors.npz`, `vector_diagnostics.json` | Actual vectors, extraction diagnostics, and control provenance |
| `retarget.json` | Imported-family source mapping and hashes, when testing a new outcome |
| `action_biases.npz` | Constant action-space comparators, when candidates reach calibration |
| `selection.json` | Validation choices recorded before confirmation |
| `results.json` | Stage results, paired effects, confidence intervals, and quality checks |
| `report.md`, `report.pdf`, `strength_response_*.png` | Human-readable evidence rendered from saved results |
| `*.log` | One timestamp-named log per invocation, with complete logged messages |
| `videos/` | Replay MP4s and matching numerical episodes |
| `application_spec.json`, `application.json`, `application-media/` | Fixed-target application contract, complete outcomes, and audited videos/plots, when a replicated speed candidate exists |

Preserve the run directory together with the exact Git commit and lockfile when sharing an experiment. New manifests automatically record the Git revision and normalized SHA256 hashes of owned source and dependency files; each resumed invocation logs its current code identity. These fingerprints exclude `.env`. Compare the saved checkpoint hash and configuration before interpreting a rerun. Episode replication reuses the same frozen checkpoint with fresh resets; it does not establish generalization across independently trained models.

The project currently implements classic activation-difference extraction on pretrained policies. Command-conditioned RL training, gradient-based extraction, and additional environments are research extensions to implement when diagnostics justify them. Local experimental compute has no overall time limit; each experiment still has a declared protocol and review point. One training seed per attempted policy is permitted.

## References and project records

[Sources and reuse decisions](docs/sources.md) distinguish upstream library behavior from this project's experimental adaptations. [AGENTS.md](AGENTS.md) contains contributor guidance. The [experiment register](reports/experiments.md) defines terms and links individual run reports, including negative outcomes.

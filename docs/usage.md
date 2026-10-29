# Advanced usage and artifact reference

Start with [installation and reproduction](../README.md#install-and-reproduce). Run the commands below from the repository root in the local environment. The [methodology](methodology.md) defines the scientific protocol; this guide explains how to operate the implemented commands.

## Runtime and configuration

Use the committed [lockfile](../uv.lock) and [project configuration](../pyproject.toml). After `uv sync --locked`, the examples use `uv run --locked --no-sync` to run the installed environment without synchronizing it again. If the environment needs repair, repeat the locked sync. These flags follow [uv's locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

The measured policy stack uses CPU PyTorch wheels. Both pinned checkpoints and the Baukit intervention have been exercised on native Windows, including Ant video rendering; other operating systems need their own runtime verification. See the [documentation audit](../reports/documentation-audit.md) for the recorded checks and [sources](sources.md) for compatibility and upstream provenance. Preserve this stack when reproducing historical results: [SB3 checkpoints contain both model parameters and serialized algorithm data](https://stable-baselines3.readthedocs.io/en/v2.4.1/guide/save_format.html), so a newer package version is not evidence of equivalent inference.

The first simulator run downloads the revision-pinned public checkpoint and requires internet access. Local settings live in `.env`; [`.env.example`](../.env.example) is the configuration reference. Keep credentials out of both configuration committed to Git and invocation logs.

| Setting | Operational meaning |
| --- | --- |
| `STEERING_ROLLOUT_WORKERS` | Concurrent rollout workers; the example configuration uses four. Each worker always uses one Torch thread. |
| `STEERING_TORCH_THREADS` | Main-process Torch threads, including `replay` and `demonstrate`; the example configuration uses one. |
| `STEERING_MAX_STEPS`, `STEERING_WARMUP` | Episode horizon and unsteered prefix. Replay restores the saved values. |
| `STEERING_*_EPISODES` | Diagnostic, fitting, validation, confirmation, and replication sample sizes. |
| `STEERING_FIT_MIN_SPEED_FRACTION` | Fitting-window speed floor relative to the healthy fitting-window median; see the registered extraction rules in [methodology](methodology.md#data-collection-and-vector-extraction). |
| `STEERING_STRENGTH_GRID` | Prespecified signed strengths for candidates and controls; changing the grid requires a new experiment. |
| `STEERING_SEED_OFFSET` | Offset for the experiment's reset partitions. A new offset must be checked against all previously used and reserved cohorts. |

## Resume, inspect, and render a run

Re-run the same `run` command to resume completed episode artifacts under the saved model, configuration, and seed splits. Use a new run directory when changing the protocol. The available inspection boundaries are `--stop-after diagnose`, `extract`, `calibrate`, and `confirm`:

```powershell
uv run --locked --no-sync locomotion-steering run --model ant --run-dir runs/my-ant-experiment --stop-after extract
uv run --locked --no-sync locomotion-steering run --model ant --run-dir runs/my-ant-experiment
uv run --locked --no-sync locomotion-steering report --run-dir runs/my-ant-experiment
```

These are simulator commands for an already specified experiment, not saved-publication reproduction. They collect paired baseline and intervention episodes, including available random and shuffled-label controls. A completed run can legitimately have no eligible vectors or fail confirmation. Inspect the saved status and full logs before interpreting it as useful steering.

`report` reads saved evidence and produces Markdown, PDF, and validation strength curves without changing selection or running new evaluations. To regenerate the current Ant publication instead, use the separate [saved-evidence build](ant-publication.md#setup-and-one-command-build).

Historical simulator reproduction requires the original code revision and configuration. Cache identities include source fingerprints; even comments or later seed guards can change those fingerprints. Do not rewrite a manifest to bypass a source mismatch. See [historical reproduction and seed safety](ant-publication.md#historical-reproduction-and-seed-safety).

## Replay a saved activation vector

Replay requires the named vector to exist in `vectors.npz`. Inspect `vector_diagnostics.json` and the run report first. In this example, replace the run path and vector with an available family, and set the PowerShell variable `$replaySeed` to an integer outside every used or reserved evaluation/application cohort before running:

```powershell
uv run --locked --no-sync locomotion-steering replay --run-dir runs/my-experiment --vector speed --alpha 0.1 --seed $replaySeed
uv run --locked --no-sync locomotion-steering replay --run-dir runs/my-experiment --vector speed --alpha 0.1 --seed $replaySeed --off-at 700
```

The example strength and off-step illustrate CLI arguments; they are not selected scientific settings or claims of improved locomotion. `--off-at` removes steering later in the episode. Each invocation writes an MP4 and matching numerical NPZ under `videos/`. Action-bias comparators belong to the research pipeline and are not activation-vector replay inputs. For the exact featured historical Ant movies, use the [pinned media reproduction instructions](ant-videos.md), whose replays must match their saved numerical episodes.

## Seed allocation and configuration

The replay default seed is `40000`. For a run with seed offset zero, that overlaps the application reservation. More generally, `demonstrate` reserves ten resets starting at `seed_offset + 40000`; these reservations are absent from historical manifests. Consult the [Ant seed-cohort ledger](../src/rl_locomotion_steering_vectors/seed_cohorts.json) and the relevant manifests/protocols before allocation. The ledger covers recorded Ant reservations, not every possible project cohort. The later Ant audit consumed the original application reservation, so those resets cannot be called fresh application data. Exact historical replays also do not supply fresh evidence.

## Test another outcome with the same vector

`retarget` imports a saved direction into a separate run directory and evaluates a newly registered behavioral outcome. Set a fresh, checked `STEERING_SEED_OFFSET` in `.env` before execution. For a source run containing the `height` family:

```powershell
uv run --locked --no-sync locomotion-steering retarget --source-run runs/source-run --run-dir runs/new-height-speed-run --vector height --behavior speed
```

The command reuses the exact source vector, its complete random/shuffled control family, and source fitting evidence. It recalibrates the new outcome on separate episodes and preserves the mapping in `retarget.json`. In the destination bundle, vector keys use the target name: `speed` identifies the imported height direction here. Keep the source fitting trajectories available because shortlisted candidates need them for action-bias comparison.

The [experiment register](../reports/experiments.md) links the historical height-to-speed follow-ups and published source evidence. Those completed reset cohorts are historical records, not default fresh seeds for another hypothesis.

## Apply a replicated candidate

After a supported speed or lateral candidate passes confirmation and replication:

```powershell
uv run --locked --no-sync locomotion-steering demonstrate --run-dir runs/your-replicated-run
```

The command restores the saved configuration and locks a validation-derived target before testing ten application episodes. It records targeting and physical-quality checks, then renders the first three paired videos and an off/on/off example. Runs without a supported replicated candidate are rejected. None of the completed experiments in the [register](../reports/experiments.md) qualifies; this is an implemented workflow, not an application result. The [fixed-speed application protocol](methodology.md#fixed-speed-application) states the exact acceptance rules and the lateral adaptation. Application allocation must also pass the seed checks described above.

## Artifacts and provenance

| Artifact in a run directory | Purpose |
| --- | --- |
| `manifest.json` | Model revision/hash, software, configuration, seed splits, and research decisions |
| `episodes/` | Compressed per-episode measurements and fitting activations |
| `vectors.npz`, `vector_diagnostics.json` | Extracted vectors, diagnostics, and control provenance |
| `retarget.json` | Source mapping and hashes for an imported family, when applicable |
| `action_biases.npz` | Constant action-space comparators, when candidates reach calibration |
| `selection.json` | Validation choices saved before confirmation |
| `results.json` | Stage outcomes, paired effects, confidence intervals, and quality checks |
| `report.md`, `report.pdf`, `strength_response_*.png` | Reports and plots rendered from saved results |
| `*.log` | Timestamp-named invocation logs with complete logged messages |
| `videos/` | Replay MP4s and matching numerical episodes |
| `application_spec.json`, `application.json`, `application-media/` | Fixed application contract, outcomes, and audited media when a supported replicated candidate exists |

Share the run directory with its exact Git commit and lockfile. New manifests record the Git revision and normalized SHA256 hashes of owned source and dependency files; resumed invocations log their current code identity. Historical bundles predate that field and disclose their later-added `code_commit` values in their bundle READMEs. Source fingerprints exclude `.env`, while manifests retain the resolved experiment configuration. Compare the saved checkpoint hash and configuration before interpreting a rerun.

Episode replication reuses the frozen checkpoint on new resets. It does not establish generalization across independently trained policies. The [source audit](sources.md) distinguishes verified checkpoint bytes and local inference from upstream training-history claims, and documents unresolved upstream redistribution terms.

Run the development suite with `uv run --locked --no-sync pytest`. Such tests verify implementation mechanics; they do not provide additional scientific steering results. Keep exact commands, failures, retries, and runtime limitations in the [documentation audit](../reports/documentation-audit.md).

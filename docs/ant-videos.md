# Reproduce the Ant steering vector and videos

The [featured experiment](../reports/ant-classic-005-findings.md) uses the frozen upstream Ant SAC checkpoint and the saved `lateral` array at **alpha = −0.1**, applied after step 100 at `actor.latent_pi[1]`. The array has 256 float32 components and already includes the documented activation-RMS scaling. Do not normalize it again. The direction's name describes its extraction label; the observed positive world-Y response occurs at the negative strength.

## Download and use the published vector

After the README's `uv sync --locked` setup, run from the project root:

```powershell
uv run hf download BurnyCoder/rl-locomotion-steering-vectors experiments/ant-classic-005/vectors.npz experiments/ant-classic-005/manifest.json experiments/ant-classic-005/vector_diagnostics.json experiments/ant-classic-005/selection.json --revision 1cd00cfcce13bcb86f1c370a71889ce6d34c278e --local-dir artifacts/published
uv run locomotion-steering replay --run-dir artifacts/published/experiments/ant-classic-005 --vector lateral --alpha -0.1 --seed 510000
```

Replay downloads and verifies the separately pinned upstream checkpoint, uses the saved configuration, and writes an MP4 plus its numerical NPZ beside the downloaded bundle. It runs the actual actor with the activation hook. It does not train the policy, change the environment reward, or apply the separate action-bias comparator.

The published `vectors.npz` was downloaded back from the pinned revision and matched SHA256 `3626c79cb49539ef0c5170a6cabdbd9183d3800aeec82ab9196a8804764df36d`. Load arrays with `numpy.load(path, allow_pickle=False)` and select the `lateral` key. Its random/shuffled controls and the separate turning family are also preserved. Raw vectors, the RMS convention, and scaling diagnostics are in `vector_diagnostics.json`; exact policy identity and layer dimensions are in `manifest.json`.

## Download the displayed movies and source trajectories

The complete 005 run, including numerical episodes, logs and media, is available at [Hugging Face revision 865e68e](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/865e68e9bf9eadc8b4efc20873e98f2d70e5e45a/experiments/ant-classic-005). To download that experiment alone:

```powershell
uv run hf download BurnyCoder/rl-locomotion-steering-vectors --revision 865e68e9bf9eadc8b4efc20873e98f2d70e5e45a --include="experiments/ant-classic-005/*" --local-dir artifacts/published
```

The published GitHub attachments are the same seed 510000 movies:

| View | Watch | File within the downloaded experiment |
| --- | --- | --- |
| Original tracking | [Tracking movie](https://github.com/user-attachments/assets/676ef6c8-ade4-4f85-a0b7-01d7fcdb86c9) | `videos/preliminary/seed-510000-preliminary-paired.mp4` |
| Top-down tracking | [Top-down movie](https://github.com/user-attachments/assets/17286e19-c5a2-4ab9-96ab-f50444d494ca) | `videos/topdown/lateral__-0.1000/seed-510000-preliminary-topdown-paired.mp4` |
| Fixed far-away | [Fixed-camera movie](https://github.com/user-attachments/assets/9e0719a6-c15b-474e-b74e-3c5e70954e2a) | `videos/fixed-camera/lateral__-0.1000/seed-510000-preliminary-fixed-camera-paired.mp4` |

## Render top-down and stationary cameras again

```powershell
uv run python scripts/render_topdown.py --run-dir artifacts/published/experiments/ant-classic-005 --vector lateral --alpha -0.1 --seed 510000
uv run python scripts/render_topdown.py --run-dir artifacts/published/experiments/ant-classic-005 --vector lateral --alpha -0.1 --seed 510000 --camera fixed
```

This helper replays every saved action in the default Ant simulator and verifies the resulting observations, actions, rewards, episode endings and onset state against the original evaluation. Rendering itself must leave the complete integration state unchanged. It changes only the visual scene, using the public [MuJoCo renderer](https://mujoco.readthedocs.io/en/3.1.5/python.html#rendering); these are existing-data illustrations, not new evaluation episodes.

The fixed mode computes one common camera from both complete paths and never moves it during either panel. Its checker squares span 20 m (a complete two-square texture period spans 40 m), following the scene-only plane sizing and [MuJoCo 3.1.5 texture mapping](https://github.com/google-deepmind/mujoco/blob/3.1.5/src/render/render_gl3.c#L105-L143). Colored past-path trails and 3 m-radius position rings make distant displacement legible and are explicitly labeled annotations. The original robot is still rendered; its geometry, forces and state are not enlarged or translated. Adjacent `seed-510000-audit.json` files record the video hashes, exact framing, helper hash and replay checks. The movie's displacement illustration is separate from the statistical forward-speed retention measure, which is a ratio of equal-episode mean X velocities, not total distance retention.

Both audit scripts read `runs/ant-classic-005`. In a fresh clone, after downloading the full experiment above, place a copy at that path and run:

```powershell
if (Test-Path runs/ant-classic-005) { throw 'Use the existing run for the audit; do not overwrite it.' }
New-Item -ItemType Directory -Force runs | Out-Null
Copy-Item -LiteralPath artifacts/published/experiments/ant-classic-005 -Destination runs/ant-classic-005 -Recurse
uv run python scripts/audit_ant005_math.py
uv run python scripts/audit_ant005_runtime.py
```

If the original run already exists, run only the two audit commands. They preserve the scientific inputs and write new audit JSON and timestamped logs in `reports/ant-classic-005`. The [causal audit](../reports/ant-classic-005-causal-audit.md) separates a causal world-Y trajectory change from a claim of body-relative sidestepping. The experiment's strict failed confirmation status remains unchanged.

## Publication mechanics

Videos were uploaded as actual GitHub PR attachments using the official [`gh --attach` workflow](https://docs.github.com/en/github-cli/github-cli/attaching-files-with-github-cli); their URLs are reused in the README and findings. The PDF has clickable movie links and measured figures, rather than an embedded video player. The official Windows CLI release was downloaded into the project cache and SHA256-checked against its release checksum file before use.

Hugging Face publication uses the existing [`hf upload` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), followed by a pinned download/hash check. In this installed CLI version, repeat `--exclude` for each pattern; a first invocation incorrectly supplied multiple patterns to one flag and failed argument parsing before uploading. Repeating the flags fixed the command. Credentials and the local environment are excluded from both repositories. The canonical Hub model card is maintained in [huggingface-model-card.md](huggingface-model-card.md).

The documented pinned-vector download and `replay` command were run from the published bundle: all 18 resulting arrays/metadata fields matched the original steered validation episode byte for byte. This saved comparison was checked again during the [documentation audit](../reports/documentation-audit.md), which distinguishes reviewed artifacts from newly executed checks. GitHub's recorded Markdown-renderer response contains three video players for the README attachments; see [renderer verification](../reports/documentation-audit/readme-render-verification.json).

The recorded core-suite check initially had [80 passes and one `WinError 5` failure](../reports/documentation-audit/tests-initial.log). A [targeted retry passed](../reports/documentation-audit/tests-targeted-retry.log), followed by [81 passes in the full retry](../reports/documentation-audit/tests-full-retry.log). These logs preserve the executed commands and outputs; the specific cause of the initial access error is not established. Paused PGS code was excluded from that check because it is not part of these experiments or publication. The audit records the applicable versions and commit so the historical count is not mistaken for a result from a different checkout.

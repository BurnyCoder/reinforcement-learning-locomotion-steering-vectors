# Reproduce the Ant publication from saved evidence

This workflow rebuilds the existing Ant world-Y result. It does not train, retune, load policy weights, or collect simulator episodes. The original failed confirmation and later audit are separate cohorts. See the [completion ledger](../reports/ant-publication-completion.md) for requirement status and the [current findings](../reports/ant-classic-005-findings.md) for scientific interpretation.

## Setup and one-command build

From this repository root, install the pinned Python environment with `uv sync --locked`. Use the committed `uv.lock` and local `.venv`; no API keys are required to download public evidence. Existing `.env` values do not select different strengths or change the saved-data publication.

```powershell
uv run --locked --no-sync python scripts/build_ant_publication.py
uv run --locked --no-sync python scripts/build_ant_publication.py --check
```

The first command validates every entry in [inputs.json](../reports/ant-publication/inputs.json), restores missing files from its immutable Hugging Face revision, and regenerates the Ant prose, comparison tables, figures, both affected PDFs, and explorer JSON. A present file with a wrong hash is rejected, never overwritten. Missing files can be restored; changed evidence requires investigation. The manifest contains explicit local and remote paths, byte sizes and SHA256 hashes, including later audit episodes that were formerly local only.

`--check` verifies existing inputs and generates temporary outputs under `tmp/ant-publication`. It compares bytes with the published files, reports stale or obsolete outputs, and does not modify publications or restore missing inputs. PDF timestamps and identifiers are fixed by ReportLab's invariant mode. Timestamped build logs are separate from deterministic scientific outputs.

The publication source templates are in `docs/publication`; edit them rather than generated Ant sections. The README uses `readme-results.md.in` inside explicit `BEGIN GENERATED README RESULTS` / `END GENERATED README RESULTS` HTML comments. Its surrounding overview and usage text remain manually maintained; the Hugging Face featured section has its own template. The README tables add a reference baseline and separate the original confirmation from the later audit's negative named arms. The full signed results remain in the audit and explorer. The thin script delegates restoration, saved-data reductions, plotting and document rendering to the `publication` package. The original JSON/NPZ files remain immutable inputs. The experiment register preserves other experiments, and historical protocols/reports retain their original evidence and linked errata.

## What the outputs mean

The full-precision [evidence JSON](../reports/ant-publication/evidence.json) records the original results, independent audit, provenance, estimators, condition summaries, source hashes and confidence-interval methods. Original intervals are reused after checking raw means. Additional display intervals are descriptive paired percentile intervals, with a fixed resampling seed and no new selection decisions. Baseline entries have no comparison interval. Unknown gates and unavailable decompositions are explicit, not replaced by zero or success.

Each episode remains in the primary analysis, including failures before steering. Reward curves average observed segments and stop at termination; aggregate plots show available-episode counts. These are evaluation rewards, not training losses or training reward curves. The fitting and audit-baseline contact-wrench clipping counts are descriptive reductions of saved observations; they are distinct from action saturation and do not isolate a causal contribution.

The explorer uses static HTML, CSS, JavaScript and SVG. It reads the same evidence and lazy-loaded episode shards as the figures. Trajectories are downsampled only for display, retain intervention onset and recorded endpoints, and use equal world-X/world-Y distance scales. Full arrays determine all statistics. Positive, null, negative and physically failed conditions remain selectable.

To preview locally:

```powershell
uv run --locked --no-sync python -m http.server 8765 --bind 127.0.0.1 --directory site
```

Open [the local explorer](http://127.0.0.1:8765). The [public GitHub Pages explorer](https://BurnyCoder.github.io/reinforcement-learning-locomotion-steering-vectors/) is deployed from the reviewed `site` directory. Raw episodes remain on Hugging Face rather than being bundled into the website.

## Historical reproduction and seed safety

Publication regeneration is separate from rerunning a historical simulator experiment. The simulator's cache identities include source fingerprints. Current comments and seed guards change those fingerprints even when comparator arithmetic is unchanged. Use the exact historical revision for exact historical replay; never rewrite a saved manifest to bypass a source mismatch. Existing [movie reproduction instructions](ant-videos.md) and the [historical audit](https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/fc8cf6337e92fbbe0fba9746671e0afd55b8149c/reports/ant-classic-005-independent-audit.md) retain the relevant chronology.

The cohort registry records both reservations and observed uses. The independent audit consumed an application reservation that was absent from the original manifest. New application/audit allocations now reject this overlap. Exact historical audit verification is explicitly distinguished from allocation of a new experiment. No replacement seeds or new experiment are authorized by this build command.

## Publication maintenance

`scripts/publish_ant_evidence.py --stage` builds an explicit evidence allowlist and staging directory. Upload that staging directory with the Hugging Face CLI, then run `--verify <immutable-revision>` to download and hash-check every input. This publishes original experiment artifacts, never upstream policy weights, credentials, `.env`, caches or the whole workspace. The [verification receipt](../reports/ant-publication/input-upload-verification.json) records the resulting revision and comparisons.

The refreshed model card and paper link to the original artifacts and the [upstream checkpoint](https://huggingface.co/farama-minari/Ant-v5-SAC-medium). Missing upstream license information is disclosed in [sources](sources.md); no license is silently assigned to upstream weights. The measured dependency stack is intentionally retained, as selected for this task. The publication workflow uses current GitHub Pages tooling without claiming the research packages are newest.

Checks include the full test suite, input-byte identity, saved-value agreement, fresh restoration, repeatable builds, PDF visual inspection, and local/live explorer interaction. Exact commands, observed failures and retries, source versions, and publication receipts are recorded in the [completion ledger](../reports/ant-publication-completion.md).

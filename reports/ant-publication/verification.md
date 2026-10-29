# Publication verification and repair history

This revision changes publication and software safeguards, using existing Ant evidence only. The authoritative completion state and exact artifact identities are in [publication-verification.json](publication-verification.json); raw download hashes, clean reproduction, PDF inspections and deployment checks have separate receipts. These are development checks, not new scientific experiments.

## Commands and interpretation

The working environment uses the committed `uv.lock`. Run `uv sync --locked`, then `uv run --locked --no-sync python scripts/build_ant_publication.py`, and finally the same command with `--check`. The build restores missing inputs, verifies complete input hashes, produces numerical summaries from saved arrays and renders the publication. Check mode generates temporary outputs and compares complete bytes. It neither restores inputs nor modifies the published outputs.

`uv run --locked --no-sync pytest -q` runs the full suite. `uv run --locked --no-sync python scripts/check_ant_site.py` validates every published shard; `node --check site/app.js` checks JavaScript syntax. The GitHub Pages workflow repeats its portable static checks on Linux. The clean reproduction receipt specifies its archive revision, separate virtual environment and use of independently downloaded, verified public cache bytes. It does not claim a second independent network download of identical bytes.

## Failures found and corrected

- The initial integrated test invocation ran while the static-checker file was still being written and failed at import. Another integrated invocation overlapped the deliberately failing reference-count regression. Those failures remain in the logs; the final completed suite ran after the fixes.
- The E9 reference-count replacement matched its historical wording but not its own generated wording. A regression first demonstrated stale counts surviving regeneration; the corrected matcher restores saved-data values after either wording.
- The first complete publication build used incorrect figure paths in templates. The image paths were corrected and both PDF outputs then built successfully.
- The initial Hub verifier hit the Windows maximum-path constraint in the default cache. Public revision-pinned retrieval now uses short, content-addressed cache names, bounded retries and complete size/SHA256 verification. Errors omit signed redirect URLs.
- Fresh-archive inspection found Git newline conversion in original logs and in the hash-addressed input manifest. Scoped Git attributes preserve their original bytes. The scientific arrays, manifests, selections and result values were not edited.
- Visual inspection found missing/incorrect Greek glyphs in PDF table headings and inline formulas. The minimal renderer now spells the mathematical variable names portably. A failing regression preceded the fix; both complete PDFs were inspected again.
- Browser inspection found that the evidence link opened JSON instead of declaring a file download. The link now uses the standard same-origin `download` attribute. The browser automation download-event API did not report an event; direct HTTP retrieval and hash verification independently check the downloadable bytes. This tool limitation is not recorded as a successful browser download event.

## Substantive review

One integrated review checked numerical reductions, split integrity, uncertainty, failures, source identity, simulator independence, renderer idempotence and explorer selection consistency. It found stale mechanism claims outside generated sections, the E9 regeneration defect and an incorrect CAA reference. All were corrected. No new numerical or statistical defect in the reviewed saved-data path was found; this is a bounded review result, not an absence-of-error guarantee.

## Visual and interactive inspection

Every page of both updated PDFs was visually inspected after the notation fix. Checks covered text/table wrapping, complete figures and axes, equations, author metadata, and the paper's final disclosure. The PDF receipts identify the exact inspected bytes.

The local and deployed explorer were exercised through the in-app browser: original confirmation, the pre-onset failed pair, validation strengths and cross-label interventions, audit component ablations, observation-rotation units, random-direction filtering, seed changes, trajectories, reward curves and source links. Mobile controls and layout were inspected at a narrow viewport with no horizontal document overflow. Browser console checks found no application errors during those flows. Screenshot receipts preserve representative views; static validation covers all shards, while browser interaction covers the described selections.

## Limits retained

Historical replay checks were reviewed rather than repeated. No simulation, training, retuning, alternative-method work or research-stack migration was performed. Training-seed generalization, formal replication, practical control usefulness and unique heading mediation remain unestablished. Historical preregistration timing and unresolved license information remain disclosed.

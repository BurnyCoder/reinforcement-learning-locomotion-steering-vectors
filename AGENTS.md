# Contributor guidance

Work only in this project directory. Preserve the user's no-behavioral-cloning constraint. Use online reinforcement learning if a new policy becomes necessary; do not train on demonstration actions.

- Search primary documentation and existing implementations before adding a solution. Reuse library APIs where they fit, and record the exact source and adaptation in `docs/sources.md`.
- Keep the CLI/pipeline thin. Put simulator mechanics in runtime, statistics in analysis, artifact I/O in storage, and report rendering in reporting. Prefer a small reusable function over duplicated condition logic.
- Use `uv`, the local `.venv`, `.env`, and the committed lockfile. Never commit credentials, local environments, or authentication caches.
- Add manually written comments explaining local mechanics and their role in the experiment. Link authoritative documentation for nontrivial library usage. Keep comments accurate when code changes.
- Write meaningful tests before changes to numerical, simulator, serialization, or intervention behavior. Run the documented user workflow after integration, inspect complete logs and outputs, and fix defects before interpreting results.
- Use timestamped logging. Do not truncate scientific measurements or future LLM prompts/responses; the current experiment does not invoke an LLM. Exclude credentials from logged inputs.
- Keep development checks separate from scientific results. Synthetic tests and random untrained-policy fixtures establish mechanics, not locomotion steering efficacy.
- Preserve held-out split integrity and immutable selection. Report unsuccessful experiments. Interpret a new hypothesis using fresh confirmation data after earlier results informed the hypothesis.
- Update the experiment register and individual reports with observed results and reasoning. Keep setup in README, methodology in `docs/methodology.md`, source audits in `docs/sources.md`, and findings in `reports` without copying entire sections between them.
- Push working core changes before experiments and training code before training. Use meaningful functional commits and feature PRs, perform one substantive review, and merge passing changes. Publish resulting steering artifacts and any newly trained policy with provenance; reference upstream weights instead of silently repackaging them.

Inspect the generated PDF visually before sharing it. Verify claims against actual artifacts and logs; passing a pilot usefulness gate does not establish semantic disentanglement or generalization across training seeds. State the measured coordinate and aggregation unit, distinguish speed retention from distance retention, and separate evaluated-checkpoint metadata from unverified upstream training history. Keep exact test commands, versions/commits, failures and retries with the [documentation audit](reports/documentation-audit.md); do not imply a recorded check was independently repeated when it was only reviewed.

For the featured Ant experiment, preserve the original `confirmation_failed` status and all failed episodes. Describe its measured coordinate as world-Y ground-plane velocity, with heading involvement, rather than asserting body-relative sidestepping. Media replays must match saved numerical episodes; fixed-camera rings and trails are visual annotations. Keep movie downloads/reproduction in `docs/ant-videos.md` and the independent audit in its experiment report. Preserve the paused alternative-method work without treating it as a completed experiment.

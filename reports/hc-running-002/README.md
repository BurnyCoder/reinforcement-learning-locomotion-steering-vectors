# hc-running-002: complete validation evidence

This bundle records 120 validation conditions for three primary contrasts and their controls. No primary setting passed all gates for its registered target. The height vector at +0.05 raised torso height 4.5048 cm with zero measured physical failure, contact, or inversion, but its 12.85% slowdown failed the height target's movement allowance. Eight control settings passed exploratory gates. No selection, confirmation, or replication was performed.

- [Experiment findings and research decisions](../hc-running-002-findings.md), with hypothesis, parameter rationale, limitations, and the next experiment.
- [Complete measured report](report.md) and [visually checked PDF](report.pdf), including all 120 conditions and three strength-response figures.
- [Original validation inspection](../hc-running-002-validation-inspection.md), documenting the collapse hidden by pooled contact before the analysis correction.
- [Exact manifest](manifest.json), [full results](results.json), [extraction diagnostics](vector_diagnostics.json), and [saved vectors](vectors.npz).
- [Superseded 74-condition analysis](validation_pooled-quality-v1.json) and [comparison audit](quality-analysis-audit.json). Target effects, intervals, seeds, and bootstrap metadata match the final analysis; two formerly passing conditions now fail the episode-level quality rule.
- [Inference-memory audit](inference-memory-audit.json), recording the loader optimization and unchanged policy/rollout checks.

JSON whitespace is compacted without dropping or changing source values. Report files, vector bytes, and all three referenced chart files are copied exactly from `runs/hc-running-002`. Audit source hashes identify the original local files, whose whitespace differs from the compact copies. The seven-page PDF was rendered with Poppler and every page inspected for readable text, complete tables, and unclipped figures.

Full trajectories and timestamped execution logs remain in the local run directory and are not included in this compact Git bundle. The manifest has no saved Git commit field; checkpoint hashes, package versions, analysis hash, and audited runtime hashes provide the recorded provenance. Its full seed lists describe registered assignments: later held-out partitions were not used. See the [experiment register](../experiments.md) for cross-experiment conclusions.

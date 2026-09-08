# hc-height-speed-003: unchanged height vector tested for speed

This run imported the exact 002 height-derived family and evaluated speed control on 40 fresh validation conditions. All eight primary settings failed physical quality, including two physical failures among ten trials at +0.05. Two random-control settings passed exploratory gates. No primary candidate was selected; confirmation, replication, and application stages were not performed.

- [Findings, protocol, limitations, and next hypothesis](../hc-height-speed-003-findings.md).
- [All measured conditions and curve](report.md), with [PDF companion](report.pdf).
- [Exact manifest](manifest.json), [complete recorded results](results.json), and [vector diagnostics](vector_diagnostics.json).
- [Retarget specification](retarget.json), [saved vectors](vectors.npz), and [byte-equality audit](vector-import-audit.json).
- [Original 002 fitting evidence](../hc-running-002-findings.md), reused rather than recollected.

Vector keys describe the new target: `speed` here is 002's `height`, and `speed_random0` is 002's `height_random0`. All five source arrays match byte-for-byte. This mapping changes the tested hypothesis, not the underlying treatment or extraction history. The declared diagnostic and fitting seeds are inherited from 002; validation seeds 210000–210009 are fresh, and held-out partitions remain unused.

JSON whitespace is compacted while preserving every recorded source value. Report, chart, and vector files are copied exactly from `runs/hc-height-speed-003`; trajectories and timestamped execution logs remain in that local run directory. The bundle is prepared locally for publication. The prior 002 evidence has already been [uploaded to Hugging Face](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/commit/9fa4bb97ef2ad7cc50d2888146c68407bc850d7c).

The final four-page PDF was rendered with Poppler and every page visually inspected. All 40 measured rows are present, inherited fitting statistics retain their source labels, and headings remain with their associated text or figure.

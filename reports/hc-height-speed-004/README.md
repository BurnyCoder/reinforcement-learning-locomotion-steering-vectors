# hc-height-speed-004: failed confirmation

The smaller-strength grid yielded six passing primary validation settings, but the locked choice −0.02 failed physical quality on confirmation: a 31.89% slowdown accompanied ten physical failures among thirty episodes. Two confirmation controls passed the pilot gates. No replication or application was performed, and no replacement primary strength was selected after confirmation.

- [Findings and interpretation](../hc-height-speed-004-findings.md), with [the pre-data specification](../hc-height-speed-004-preregistration.md).
- [Full numerical report](report.md) and [visually checked PDF](report.pdf), preserving all 72 validation and six confirmation conditions.
- [Manifest](manifest.json), [complete results](results.json), and [locked selection](selection.json).
- [Source mapping](retarget.json), [inherited fitting diagnostics](vector_diagnostics.json), [activation vectors](vectors.npz), and [import audit](vector-import-audit.json).
- [Fitted action-bias comparator](action_biases.npz), which has action dimensions and is separate from the hidden activation vectors.

The `speed` activation array remains the exact height-derived vector from 002. All five imported candidate/control arrays were checked byte-for-byte against that source. The action bias was newly derived from the original fitting observations after candidate calibration, without training the policy.

JSON whitespace is compacted with all original values preserved. Report, vector, action-bias, and chart files are exact copies of the local run artifacts. The final five-page PDF was rendered with Poppler and every page inspected; all measured rows, source labels, and the figure are readable. Full raw episodes and timestamped logs remain under `runs/hc-height-speed-004` for publication. This compact bundle is prepared locally and does not claim a completed upload.

# hc-height-speed-004: failed confirmation

The smaller-strength grid yielded six passing primary validation settings, but the locked choice −0.02 failed physical quality on confirmation: a 31.89% slowdown accompanied ten physical failures among thirty episodes. Two confirmation controls passed the pilot gates. No replication or application was performed, and no replacement primary strength was selected after confirmation.

- [Findings and interpretation](../hc-height-speed-004-findings.md), with [the locally dated protocol](../hc-height-speed-004-preregistration.md), first committed before this run's validation.
- [Full numerical report](report.md) and [visually checked PDF](report.pdf), preserving all 72 validation and six confirmation conditions.
- [Manifest](manifest.json), [complete results](results.json), and [locked selection](selection.json).
- [Source mapping](retarget.json), [inherited fitting diagnostics](vector_diagnostics.json), [activation vectors](vectors.npz), and [import audit](vector-import-audit.json).
- [Fitted action-bias comparator](action_biases.npz), which has action dimensions and is separate from the hidden activation vectors.

The `speed` activation array remains the exact height-derived vector from 002. All five imported candidate/control arrays were checked byte-for-byte against that source. The action bias was newly derived from the original fitting observations after candidate calibration, without training the policy.

Source JSON is compacted with all original values preserved. PDF, vector, action-bias, and chart files match the local run bytes; generated report Markdown matches after CRLF/LF normalization. The final five-page PDF was rendered with Poppler and every page inspected; all measured rows, source labels, and the figure are readable. Run data and numerical reports were published at [Hugging Face commit dc0748d](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/tree/dc0748d926dd9e9b038d14741ccada3d2f123673/experiments/hc-height-speed-004). That pinned release does not include the local `vector-import-audit.json`; the [documentation audit](../documentation-audit.md) tracks the audit-file refresh and later corrections. Raw episodes and logs also remain locally, and the failed-confirmation outcome is unchanged.

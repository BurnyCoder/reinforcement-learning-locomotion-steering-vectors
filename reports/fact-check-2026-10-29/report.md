# Independent saved-data fact-check — 2026-10-29

Author: Libor “Burny” Burian. This audit starts from source revision `3fb62bcea7e159be8f748028467cad6f5852ed66` and treats earlier audits as claims, not authorities. It executes saved-array statistics and document tooling only. No policies, inference, gradients, training, simulator episodes or video replays were run.

## Numerical result and scope

The independent reference calculations reproduce all **332 original validation/confirmation comparisons**. They also reconstruct **250 treated audit arms**, **124 signed activation directions**, and **20 paired audit resets**. Means, paired intervals, failure counts, pilot gates, selected strengths, original fit/control vectors and later audit direction construction match within the documented tolerances. The later locked-primary mean is **+0.462103 m/s**, BCa 95% interval **[0.343428, 0.587365]**. These are arithmetic checks on saved observations, not new behavior measurements.

| Experiment | Validation conditions | Confirmation conditions | Recorded outcome |
| --- | ---: | ---: | --- |
| hc-classic-001 | 0 | 0 | `extraction_rejected_before_validation` |
| hc-running-002 | 120 | 0 | `no_validation_candidate` |
| hc-height-speed-003 | 40 | 0 | `no_validation_candidate` |
| hc-height-speed-004 | 72 | 6 | `confirmation_failed` |
| ant-classic-005 | 88 | 6 | `confirmation_failed` |

The [machine verification](verification.json) records every comparison. [Input hashes](inputs.json) preserve **9611 files**; none changed. Scalar comparisons use absolute tolerance 1e-10 and relative tolerance 1e-09. Reconstructed float32 vectors allow absolute error at most 1e-7; action-effort and planar-speed reductions allow 1e-10. Comparison ranks retain historical field names in immutable JSON; their interpretation is corrected below. Bootstrap procedures use the installed SciPy version recorded in [provenance](provenance.json), with explicit seeds and whole reset units.

## Corrections

- Random-direction and shuffled-control ranks are descriptive ensemble comparisons. Calibrated Monte Carlo p-values additionally require exchangeability under an explicit null, which norm matching and validation selection do not establish. Holm adjustment does not repair invalid input p-values. The historical arithmetic and decisions are preserved. [Assumptions](../../docs/methodology.md#statistical-assumptions-and-rank-interpretation).
- Odd/even components mean sign-dependent/sign-independent responses. They do not identify steering versus disruption. HalfCheetah slowdowns do not prove near-optimality or a unique degradation mechanism. Second-order optimality reasoning requires a local maximum of a smooth objective along the actual intervention path.
- “Conditions were confirmed” is replaced by “evaluated in confirmation.” Failed acceptance remains failed. Fixed fitted vectors are distinguished from observation-dependent activations.
- Hugging Face metadata no longer implies fine-tuning/merging of the frozen upstream policies. Explicit pinned checkpoint references remain. [Hub specification](https://huggingface.co/docs/hub/model-cards#specifying-a-base-model).
- Heading-relative kinematics pairs pre-transition input orientation with the ensuing displacement velocity. It is a discrete-time projection. Root-position velocity is not asserted to be bitwise identical to the torso-position velocity used by Ant reward.
- The [kinematics follow-up](kinematics-follow-up.md) corrects slope units for nondefault windows, single-window trend availability, angle branch-cut subtraction, and timestep validation. It checks that published default-window means and denominators remain compatible with saved evidence. A finite-horizon heading offset does not prove bounded long-run behavior.
- Further corrections distinguish a passing mean-effect gate from temporal persistence, selection bias from an observed validation/confirmation difference, covariance participation ratio from matrix rank, and current generated report bytes from historical run-directory renderings. Aligned directions can still be compared; alignment limits semantic interpretation rather than making an effect comparison untestable.
- Historical runtime checks, training metadata and paused PGS work are attributed records. Static inspection and saved arrays cannot establish untested model behavior. Existing perceived-heading claims retain the clipped-wrench qualification: rotating an already-clipped wrench need not equal clipping a physically rotated wrench.
- Archived README screenshots in `reports/readme-restructure` retain the former “p =” rank label; interpret it as the descriptive comparison rank above. Historical “perceived-heading axis” plot labels name the saved operational response axis, not a verified unique mediator or exact physical rotation. These preserved images do not supersede the corrected current prose and figures.

## Independent calculations

The reference implementation imports neither production estimators nor earlier audit calculations. It reads numeric NPZ members without pickle, verifies paired seeds/onset hashes/prefix arrays, reconstructs fitting windows and exclusions, episode-balanced activation RMS and contrasts, random streams, shuffle controls, split-half cosines, and frozen near-best selection. It retains prefix-only failures and unequal available horizons. Tests include missing/duplicate pairs, exact failure thresholds, within-window variance, unequal lengths, corrupt measurements and the reference coordinate transformation.

For the later audit, it recomputes P1/P2/S1/S2/M1/M2 arithmetic, including BCa/percentile intervals, sign-flip and Wilcoxon statistics, joint reset and crossed direction/reset resampling, ranks and Holm arithmetic. K1 geometry and its window denominators are recomputed. Original confirmation alternatives include prefix exclusion, both-arm survival, displacement, trimming, leave-one-out sensitivity and arm correlation. These alternatives are post hoc and do not replace the primary estimator. Conditional surviving horizons remain affected by treatment.

The publication check independently reconstructs all **347 displayed conditions** and **6120 trajectory pairs**, including reward-block denominators, failure markers, paired intervals and odd/even components. Display coordinates and reward blocks are rounded to six decimals before comparison; statistical effects retain full precision. HalfCheetah numerical prose and its complete locked-control table are generated from the independently checked saved results. Original report rendering uses explicit LF newlines so Git normalization cannot change deterministic bytes.

The saved neural response axes are used to check projections and component algebra; their neural derivation is not re-executed. The historical E1 fitted regression, power analyses, studentized sign-flip calculations and runtime replay results not explicitly present in verification.json remain **unresolved as independent reproductions**. They are not needed to retain the directly recomputed effect or failed acceptance. Earlier reports carry an erratum pointing here; their raw calculations remain preserved.

## Provenance and chronology

The provenance record maps cached runtime/perception fingerprints to historical Git source: **0 unmatched identities**. Checkpoint ZIP hashes and non-executable JSON training metadata were inspected without loading tensors or policies. Metadata records what was saved, not independently reproduced training history. Original policy parameter hashes, action-bias derivation and runtime cleanup remain historical evidence.

Original Ant's protocol first enters Git after diagnostics began. The later audit protocol's Git record precedes its logged collection; Git/log dates do not independently attest push time or prove that no earlier observations existed. HC retargeting deliberately reuses the earlier fitted height family. The audit overlaps **10 previously reserved application seeds**; it is not the formal replication split. Manifest decisions and failed/retried log events are retained in [provenance.json](provenance.json).

## Claim ledger and limits

The [claim ledger](claims.json) inventories **693 original and added source files** and **5979 prose/comment blocks**, linking locations, evidence, check scope and disposition. File coverage includes derived images/PDFs, recorded JSON, logs and source. Classification is conservative: a source citation does not automatically verify a claim; a group disposition applies only to its stated checks. Any uncovered block remains explicitly unresolved.

| Claim-block disposition | Blocks |
| --- | ---: |
| checked_and_qualified | 289 |
| checked_saved_data_only | 70 |
| checked_with_runtime_boundary | 145 |
| corrected_and_qualified | 381 |
| qualified | 52 |
| qualified_correspondence | 167 |
| qualified_historical_record | 1238 |
| recomputed_with_limits | 537 |
| screened_execution_or_static_check | 1361 |
| source_checked_runtime_unverified | 201 |
| static_check_only | 1538 |

Primary-source reading and project adaptations are recorded in [the scope/source registry](../../docs/fact-check-scope.json) and [sources](../../docs/sources.md). The visual/media record and [checks actually executed](checks.json) are separate from old test totals. Movie metadata and saved correspondence receipts can bind artifacts but do not independently establish every frame's state correspondence without replay. No such replay was performed.

## Rebuild

Run `uv run --locked --no-sync python scripts/build_publication.py` to independently verify evidence and rebuild all current experiment reports, Ant figures/prose, the paper and this audit. Add `--check` to compare deterministic outputs without editing publications. The existing `scripts/build_ant_publication.py` remains available. Missing historical raw inputs must be restored using the pinned records; the checker never substitutes missing data or refreshes mismatching hashes.

Completion of these checks does not guarantee the absence of undiscovered errors. It establishes the documented dispositions and numerical agreements within the no-model boundary.

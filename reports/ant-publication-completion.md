# Ant sideways-result completion ledger

Scope: existing Ant world-Y evidence only. This is a publication/software audit, not another locomotion experiment. Original confirmation remains `confirmation_failed`; its headline is +0.402 m/s [0.308, 0.491] on 30 paired resets. The separate audit remains +0.462 m/s [0.343, 0.587] on 20 resets. Sources: [original results](ant-classic-005/results.json), [audit results](ant-classic-005-independent-audit/audit.json), and [generated evidence](ant-publication/evidence.json).

## Requirement-by-requirement disposition

| Requirement group | Disposition | Evidence or limit |
| --- | --- | --- |
| Prior work, source-grounded methods and license checking | Updated | [Sources](../docs/sources.md) adds earlier frozen-RL activation interventions, exact adaptation boundaries and license status. Baukit's full license and the upstream checkpoint license remain unresolved; no clearance is asserted. |
| Falsifiable hypotheses and analysis plans | Historical evidence retained | Original [protocol](ant-classic-005-preregistration.md) first entered Git after diagnostics/fitting began. This cannot be retroactively repaired. The [later audit protocol](ant-classic-005-independent-audit-preregistration.md) predates its fresh collection. |
| Baselines, controls, failures and uncertainty | Preserved and exposed | All original conditions and later random/shuffled/component/observation arms remain in the generated evidence and explorer. Shared-baseline and estimand distinctions are explicit. |
| Seed integrity | Application/audit guard added | A versioned reserved/consumed ledger rejects reuse of the audit's former application reservation. Original manifests are unchanged. This is scoped protection, not a claim of a general experiment scheduler. |
| Causal/mechanistic correctness | Interpretation corrected | `clip(R clip(f))` is distinguished from `clip(R f)`; unsaturated-frame and saturation-counterexample tests bound the claim. Heading involvement is supported; unique mediation and physical equivalence are unestablished. |
| Model/seed/hyperparameter generalization | Explicit limitation | Fitting sensitivity already exists; independent policy training seeds, new strength sweeps, real robots and domain shifts remain outside the selected existing-evidence scope. |
| No leakage, cherry-picking or p-value overclaiming | Scoped checks and limitations | Saved splits, locking and failures preserved; audit/application overlap disclosed; MC floors described as estimates, not exact-p upper bounds. No blanket absence-of-error guarantee. |
| Modular code, reuse and comments | Implemented for changed code | Thin publication CLI delegates I/O, analysis, figures and prose. Source-linked comments are manually written. No unnecessary policy/backend rewrite. |
| README and paper comparison tables | Generated | Original confirmation and later audit have separate summaries, intervals, physical costs and failure counts. Baseline values and control strengths remain explicit. |
| Figures, architecture, rewards, visual examples | Generated | Architecture, uncertainty, strength-response, odd/even, ablations, trajectories and evaluation reward. No training curves are fabricated. |
| Interactive UI and GitHub Pages | Implemented; see deployment receipt | Static explorer exposes cohorts, conditions, strengths, seeds, failures and source downloads. [Live explorer](https://BurnyCoder.github.io/reinforcement-learning-locomotion-steering-vectors/). |
| Paper sections, author, AI statement | Updated | Existing paper retains broader historical context, adds current Ant interpretation and technical appendix, uses Libor “Burny” Burian, and carries the requested statement at its bottom only. |
| Reports, register, learnings and lay definitions | Updated | Current Ant reports are generated; earlier experiment outcomes and the register glossary are preserved. Historical snapshots link to the exact earlier Git revision. |
| Every empirical number generated; one-command rebuilding | Implemented for current Ant publication | Templates bind current Ant claims/tables to saved fields; [one command](../docs/ant-publication.md) regenerates current outputs. Historical snapshots and unaffected HalfCheetah content retain their prior provenance. |
| Timestamped complete logs | Retained | Original experiment/audit logs plus publication/test/verification logs. No experimental LLM call occurs, so LLM prompt logging is inapplicable. |
| TDD, user workflow, one substantive review | Evidence recorded below | Numerical/serialization/seed boundaries have regression tests; full reproduction and visual checks are recorded with their actual scope. |
| Git, functional commits, PRs and merges | Use existing repository | Core changes are grouped and pushed; no duplicate repository is created. Final PR/deployment identities appear in publication receipts. |
| Hugging Face models/artifacts and collection | Published with provenance; see receipts | Original steering artifacts, later audit episodes and documentation; upstream checkpoint referenced, not repackaged. |
| uv, local environment, configuration, GPU | Existing environment retained | Locked uv environment and `.env` setup remain. This saved-data CPU build needs no new GPU training; research-stack migration was explicitly excluded. |
| Work only in project; preserve alternative method pause | Preserved | No other local project folders edited. No behavioral cloning, new training, retuning, scientific collection or alternative-method experiment. |
| Historical push-before-first-run ordering | Not retroactively repairable | [Documentation audit](documentation-audit.md) records chronology and provenance uncertainty. Current publication does not claim historical compliance. |

## What changed and why

The initial audit found extensive scientific checks already present, so the implementation concentrated on missing reproducibility and presentation. The contact-wrench discrepancy was found by following the pinned Ant observation construction and comparing it with the physical-equivalence test. That test intentionally scaled forces below clipping. The fix preserves historical numerical semantics, narrows the claim, adds a saturated-force counterexample, and counts saturation in saved observations rather than rerunning a different comparator.

The application cohort existed only as a convention and was missed by manifest-only overlap checks. A small explicit cohort ledger fixes the allocation boundary and documents the consumed reservation. It does not silently replace seeds or rewrite old source identities.

The earlier publication renderer read manually maintained numerical prose and required simulation-oriented commands to rebuild some audit outputs. The new builder separates saved-data validation, reductions, figures and rendering; hashed inputs and deterministic outputs make stale claims detectable. Named ablations and failed examples are available without starting another scientific run. Display-only trajectory reduction retains endpoints and onset while preserving full-data statistics.

Raw mechanism-audit episodes had remained local despite public narrative reports. They are now included in the explicit publication allowlist, with original directions and identity evidence. Hash verification uses downloads from one immutable revision rather than trusting upload success. No missing data, curve or generalization result is invented to fill a checklist item.

## Verification record

The machine-readable [input receipt](ant-publication/input-upload-verification.json) records immutable Hub revision `b262002001df6af2fd87dce82cc385f6b132e184` and complete downloaded-byte comparisons. The [publication verification](ant-publication/publication-verification.json) records the final GitHub/Hugging Face/Pages/collection identities. [Build and test logs](ant-publication/logs/) retain failures and retries. [Reproduction instructions](../docs/ant-publication.md) specify the exact workflow.

Development fixtures verify mechanics only. Historical replay and earlier tests are labelled as reviewed historical evidence unless explicitly repeated. All original source arrays, compact numerical results, selections and gates remain unchanged; new clipping counts and display intervals are post hoc saved-data summaries. Final acceptance requires the complete verification receipts rather than this prose alone.

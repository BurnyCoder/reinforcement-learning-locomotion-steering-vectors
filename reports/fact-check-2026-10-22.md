# Independent documentation fact-check, 2026-10-22

**Scope.** Every factual claim in README.md, AGENTS.md, `docs/`, the experiment reports and register, the earlier audits, the paper and its PDF, the result bundles, code comments and the Hugging Face model card was checked against primary evidence. The earlier audits (documentation, rigor, y-shift, mechanism, independent, `math-audit.json`) were **not** used as evidence: their claims were treated as claims to check.

**Method.**
- **Numbers.** Three parallel read-only reviews (Claude subagents) wrote their own recomputation code against the raw episodes in `runs/` (gitignored, local), the saved JSON/NPZ bundles, and the source code. Each review covered one part: docs, code and external sources; Ant 005 with the fresh-seed audit; HalfCheetah, the paper and the cross-cutting reports.
- **Bootstrap intervals.** Production intervals were reproduced with their recorded seeds and checked against an independent RNG and t-intervals.
- **External claims.** These were checked against the pinned upstream sources: Baukit, SB3/sb3-contrib, Gymnasium 1.0.0, MuJoCo 3.1.5, the Minari scripts, the Hugging Face revisions, and the cited papers.
- **Re-checks.** Every correction below was re-checked by the main session before editing. The numbers newly cited here come from [`scripts/factcheck_2026_10_22.py`](../scripts/factcheck_2026_10_22.py), which reuses the production estimators. It logs to [`fact-check-2026-10-22/`](fact-check-2026-10-22/).

## Result

**No recorded number, interval, count, seed range or status is wrong.** The errors were signs, stale statements, overclaims, unlabelled sample sizes, provenance gaps, a reserved-seed collision, broken or misdirected links, misleading code comments, a PDF glyph bug, and a Hugging Face card and mirrors frozen at 2026-09-08.

The following were reproduced from raw episodes:
- **Ant 005 confirmation:** +0.40238 m/s [0.30803, 0.49127] (seed 720); 96.83% retention; 4/30 failures in each arm on different seeds; seed 520025 ending at step 34 in all seven arms; effort +17.84%; reward falls; all control effects, failures and cosines.
- **Ant validation:** 88 conditions with 11 passes.
- **Ant path measures:** displacement +17.41 m [12.81, 21.65]; matched horizon +0.392; travel angle +3.84°; torso yaw +5.23°; body-relative −1.39°.
- **Ant fresh-seed audit:** 5,020 episodes; +0.462 [0.343, 0.587]; rank p = 0.05 (4 of 99 random directions); 7 of 19 shuffles; drift +0.034; ±δ perception arms +0.904/−0.913.
- **HalfCheetah:** all 238 conditions (002: 120; 003: 40; 004: 72 + 6), the 001 sensitivity numbers, and the 002 contact timeline.
- **Other:** checkpoint metadata (seed 0; 5.92M/8.77M timesteps); `paper.pdf` and the HC report PDFs against their Markdown.

The main session also re-simulated validation seed 510000 in a fresh process for baseline and lateral −0.1. All 18 saved fields were bit-identical ([log](fact-check-2026-10-22/20261022T162024058273Z.log)).

## Material findings

| Finding | Evidence | Correction |
| --- | --- | --- |
| **Reserved-seed collision.** The independent audit's fresh seeds 540000–540019 include Ant 005's application cohort 540000–540009. `demonstrate` reserves `seed_offset + 40000` onward by convention, and `fresh_application_seeds` checks only recorded manifest splits. | `demonstration.py` (`fresh_application_seeds`); `scripts/ant005_independent/common.py` (`FRESH_SEEDS`); manifest `seed_offset` 500000 | Disclosed in the audit report §1/§5/§7, its pre-registration errata, methodology, the register partitions table, findings, paper §8 and README. A future 005 application needs another cohort. |
| **Retrospective `code_commit`.** Manifests 001, 003, 004 and 005 carry a `code_commit` that their logged "prepared manifest" lacks. Only 001's note and 005's grid migration were disclosed. | `runs/<run>/2026*.log` versus saved manifests; the commits precede `created_utc` by 13 s (003), 50 s (004) and 104 s (005) | Notes added to the 003/004/005 bundle READMEs, README "Reproduce", and paper §9 |
| **Sign of the perception arm.** README said "9.2°"; the +0.90 m/s arm is δ = −0.160876 rad. | `perceived_heading__-0.160876rad` arm; `perception.py` sign convention | "−9.2°, so the perceived heading turns toward −Y" |
| **Stale "no replication".** Findings §3, the bundle README, the register and the HF card predate the fresh-seed audit. | Independent audit P1 | "No formal replication (530000–530029); a pre-registered audit on 540000–540019 reproduced +0.462 m/s" |
| **Failures omitted.** README's "Replicated" omitted 2/20 treated failures vs 0/20 (+10 points, above the 5-point allowance). | `fact-check-2026-10-22` log, `fresh_failures` | Added |
| **Controls at unequal strength.** The 0.15–0.27 m/s margins compare against random 0/2 at −0.05, the shuffled control at −0.2 and the bias at +0.5. Only random 1 matches −0.1, and it trails by 0.150 [0.079, 0.215]. | `selection.json`; `control_margins` | Strengths named in README, findings, register and paper |
| **"Remainder behaves like a random direction."** Its sign-dependent response ranks like random (p = 0.21), but renormalized to full norm it fails 8/20 and 14/20 episodes; random arms average 1.8–2.3. | `fresh_failures` | Qualified in findings, register and paper |
| **"No intervention ever sped HalfCheetah up."** 12 conditions sped up. The largest, +0.404%, is 004 action bias −0.035: +0.068 m/s [0.045, 0.088]. | `max_speedup_004` | "No more than about 0.4% (maximum +0.40%)" in the rigor audit, register, methodology, 004 findings and paper |
| **"Every run-004 slowdown, primary or control, is sign-independent."** True for the primary only; random 1/2 at ±0.035 slow at one sign. | Rigor audit §4 table | Paper §4 corrected |
| **PDF glyph bug.** `build_paper.py` mapped ∓ to ±, reversing "∓9.22°" in the independent-audit PDF. | `tests/test_build_paper.py` failed before the fix | Maps to "-/+"; both PDFs rebuilt |
| **Hugging Face card and mirrors.** The live card equalled the 2026-09-08 `docs/huggingface-model-card.md`. It said "no independent replication", called the within-project audit "independent", and linked a nine-page paper copy without [E6]–[E9]. The `documentation/` and bundle-README mirrors were also September snapshots. | `hf://models/BurnyCoder/rl-locomotion-steering-vectors` | Card rewritten; mirrors refreshed (see below) |
| **Repository renamed.** GitHub now serves `BurnyCoder/reinforcement-learning-locomotion-steering-vectors`; old links redirect. | `gh repo view` | Live docs, paper and card updated; pre-registrations and audit receipts left as historical records |

## Wording and precision corrections

- **Sign-flip p-values.** "p = 5×10⁻⁶" is the floor 1/200,001 of a 200,000-flip Monte Carlo test; no flip was as extreme (paired t = 8.6 over the 30 pairs). It is now labelled as the floor everywhere it appears.
- **"35–112%".** This covers only the three lateral-family random draws. Across all six isotropic draws the range is −13% to 112%, with mixed signs.
- **Even share "[9%, 39%]".** The point estimate is 26%. The interval's lower bound is Monte Carlo-sensitive: 6.5–8.9% across bootstrap seeds, 8.5% with the recorded seed. It now reads "about [8%, 39%]".
- **"60%/91%/58% aligned".** These are cosines (0.599, 0.914, 0.584), not percentages.
- **Pre-registration "cos 0.915".** The saved value is 0.91419. An errata section was appended; the original text is unchanged.
- **Mixed sample sizes.** The paired SD 0.249 (n = 29) and 0.256 (n = 30), and the rigid-rotation arithmetic, mixed n. On 29 pairs: 276.8 m baseline, 18.6 m predicted, 18.0 m measured. On 30 pairs: 267.6 m, 18.0 m, 17.4 m.
- **Audit chronology.** The audit run started at 14:25:59 UTC. The first fresh-seed episode was logged at 14:26:32, after the 14:25:40 protocol commit.
- **"2.3× overstatement".** This is the odd world-Y ratio (0.908/0.397). By heading the ratio is 1.4–1.9× (7.8°/8.1° vs 5.5°/4.2°).
- **"Turning +0.12 to +0.26 larger".** That range is the CI; the point estimate is +0.19 m/s.
- **"Neither a turn nor sidestepping".** Body-relative travel even drifts slightly toward −Y (gait-axis −0.12 m/s).
- **Naming.** "Mean sideways speed" is now "mean world-Y velocity". The within-project causal/math audit is no longer called "independent".
- **README facts.**
  - Ant candidates are lateral and turning only (`pipeline.py`).
  - Rollout workers always use one Torch thread; `STEERING_TORCH_THREADS` sets only the main process.
  - The replay default seed 40000 equals the first application seed reserved for offset-0 runs.
- **Register.**
  - The "useful gate" definition now includes the quality limits.
  - The unlogged "1.10 s" smoke timing is marked as such. A logged re-measurement gave 1.20–1.25 s.
- **Reshuffle null.** The mean |cos| range 0.24–0.52 is Monte Carlo. An independent 300-draw rerun (seed 77) gave 0.24–0.53; methodology now says "about 0.24–0.53".
- **`paper_build.md`.** Edition bullets were out of order: the 56- and 54-link counts sat together, and the E6 bullets sat under E8. It also had stale "no list support" and a wrong PDF location. It is now chronological, with E10 added.
- **`audit_ant005_independent.py`.** The default `--workers 12` exhausted memory; the default and usage are now 6.

## Sources added or corrected

- **Methodology.** The Baukit `Trace` anchor now spans hook registration to removal (`nethook.py#L50-L108`). The sb3-contrib v2.4.0 TQC actor source is added.
- **Sources.**
  - Added the Minari Ant medium page (SB3 SAC, 10⁷ steps).
  - The upstream upload is `run_0/best_model.zip`, the `EvalCallback` best model, not the final model. Consistent with this, `num_timesteps` is below the training lengths on the Minari pages (6×10⁶ and 10⁷).
  - Corrected the steering-vectors anchors (`L99-L107`, `L188-L212`).
  - Pinned the SciPy links to 1.14.1.
  - Cited the [MuJoCo 3.1.5 overview](https://mujoco.readthedocs.io/en/3.1.5/overview.html#floating-objects) for free-joint velocity frames, which the earlier audit verified only empirically.
- **Code comments.**
  - `runtime.py`: the Hub docs link now points to v1.30.0, the locked version.
  - `reporting.py`:
    - Cites `Axes.plot`/`Axes.vlines`, which the code uses, instead of `pyplot.errorbar`, which it does not.
    - The ReportLab user-guide link printed into every run report returned HTTP 403; it is replaced by `.../userguide/ch5_platypus/`.
  - `analysis.py`:
    - The bootstrap source is now NumPy `Generator.integers`.
    - The "pairing removes reset variability" comment is corrected: pairing narrows intervals only when the arms stay correlated.
  - `rigor_statistics.py`: arXiv 1307.0239 is named as Myllymäki et al. 2017, which reviews Barnard's test.
  - `documentation_evidence.py` mirrors `analysis.paired_effect`, not SciPy.
  - `verify_documentation_publication.py` allows four concurrent requests, not four requests.
  - `pgs-adaptation.md` now shows the keyword-only signatures.

## Regenerated artifacts

- **Run reports.** All five `report.md`/`report.pdf` were regenerated from saved evidence for the ReportLab link fix. Only the Sources line and the "Report generated" timestamp changed. Page counts (2, 7, 4, 5, 6) and all charts are byte-identical. The pages carrying the change were rendered and inspected.
- **Paper and audit PDFs.** `paper.pdf` (10 pages) and `ant-classic-005-independent-audit.pdf` (7 pages) were rebuilt. Every changed page was rendered and inspected. `pypdf` 5.1.0 confirmed that the PDF link annotations equal the Markdown links: 60 and 9.
- **Paper byline.** The paper now names its author, Libor Burian.

## Hugging Face

The corrected card (`docs/huggingface-model-card.md`) adds the fresh-seed result, the weak specificity, the perceived-heading mechanism, corrected audit naming and the best-model provenance. It is uploaded as the repository `README.md`. The same commit refreshes `paper/` (rebuilt paper, build notes and assets) and `documentation/` (current methodology, sources and video docs, and the experiment and audit reports, including the October audits). It also adds a correction note to each `experiments/*/README.md` mirror and uploads the regenerated run reports. Data, vectors, logs and `documentation-audit/` are unchanged. Markdown mirrors have their relative links pinned to GitHub merge commit `7c5d0a5`, following the earlier mirrors' convention; all 114 pinned links returned HTTP 200.

**Receipt.** Hugging Face commit [`31987a0`](https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/commit/31987a0337761b098f9ff41ff062d5ce58be47f7) holds all 42 staged files. Each was downloaded back at that revision and matched its staged SHA256; the [staging manifest](fact-check-2026-10-22/hf-stage-manifest.json) and [verification result](fact-check-2026-10-22/hf-verify-result.json) record this. A later commit refreshes only this report's mirror so that it includes this receipt.

## Not verified

- Actions and transitions that need policy inference beyond seed 510000; the 28-action reconstruction and offline activation differences were not re-run.
- The GitHub video attachments' bytes against the HF MP4s.
- Byte-exact re-derivation of the rigor audit's 300-reshuffle values (RNG-dependent; reproduced to about ±0.02).
- The leave-one-episode-out and 1%-trim cosines in the Ant diagnostics.
- The causal audit's heading 0.725497/0.869187 rad and side velocity −4.4339/−4.3517: its weighting is unspecified, and window-weighted means give 0.723/0.868 and −4.404/−4.353.
- Pre-registration clock times beyond their Git commit times.

## Learnings

- **Sign conventions and "aligned" percentages.** These propagated through several documents because each later audit quoted the previous one's prose. Recomputing from raw data, not from audit JSON, exposed them.
- **Monte Carlo floors.** A floor reported as an exact p-value looks more precise than it is. Report "p ≤ 1/(B+1)", or say "floor".
- **Reserved cohorts.** The application cohort is implied by a convention, not written to the manifest, so seed-disjointness checks miss it. Any future audit or application should list every reserved range explicitly.
- **External mirrors.** Hugging Face mirrors do not update with the GitHub docs; they must be refreshed whenever the docs change.
- **Line anchors.** Relative `#L` anchors break when comments add lines. Edits here kept the line counts of anchored files unchanged, and pinned permalinks were preferred where they existed.

## Reproduction

```powershell
uv run --no-sync python scripts/factcheck_2026_10_22.py
uv run --no-sync pytest -q
uv run python reports/build_paper.py
uv run python reports/build_paper.py reports/ant-classic-005-independent-audit.md
```

The test suite passed: 144 tests, including the paused PGS tests and the new renderer test ([log](fact-check-2026-10-22/tests.log)).

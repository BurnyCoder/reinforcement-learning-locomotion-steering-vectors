# Kinematics and statistical-claim follow-up — 2026-10-29

This additional review found errors in the numerical helper's boundary cases and an unsupported long-horizon interpretation. Raw episodes, historical analyses and selections remain unchanged. No models or simulator episodes ran.

## Corrections and evidence

- The heading slope was degrees per chosen window but labeled degrees per 100 steps. It now multiplies the fitted slope by `100 / window`. A single window returns an unavailable slope (`None`), rather than a fabricated zero. The [NumPy linear-fit definition](https://numpy.org/doc/1.26/reference/generated/numpy.polyfit.html) establishes the independent-variable units.
- Subtracting principal path angles could report an almost full revolution across the ±180° branch cut. Differences now wrap to the shortest signed angle in `[-180, 180)`, consistent with [NumPy's arctan2 range](https://numpy.org/doc/1.26/reference/generated/numpy.arctan2.html). This path-direction statistic does not measure accumulated turns.
- Matching step indices requires equal positive finite simulator timesteps. The helper now checks both arms instead of silently using the baseline timestep for both displacements. Invalid window/prefix lengths fail explicitly.
- A finite-horizon heading offset or small fitted trend does not establish bounded long-run behavior or cessation of turning. Current audit prose and contributor guidance now state this limit. Contact-fraction comments no longer claim statistical unbiasedness from merely counting noncontact steps.
- The original speed and height gates threshold episode means; they do not test temporal persistence. Methodology and paper prose now say so, with a transient-effect example in the [analysis tests](../../tests/test_analysis.py). No historical threshold or outcome was changed.
- A lower confirmation estimate does not diagnose a winner's-curse mechanism or prove unbiasedness. The historical rigor report now separates that observation from the assumptions in [Cawley and Talbot (2010)](https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf). It also distinguishes covariance participation ratio from matrix rank and measured alignment from a necessary consequence of shared span. Aligned interventions can still have their effects compared.
- Paper-build instructions now identify generated templates and distinguish current report renderings from preserved run-directory originals. Test documentation separates historical full-suite records from the screened no-model checks performed here.

See the [corrected helper](../../src/rl_locomotion_steering_vectors/kinematics.py) and [analytic regression cases](../../tests/test_kinematics.py). Tests cover two window resolutions away from the historical default, the angle branch cut, one-window data, and invalid timestep/span metadata. The original default-window summaries remain compatible with their saved values:

| Cohort | Pairs checked | Historical eligible pairs | Mean fields checked | Maximum absolute difference |
| --- | ---: | ---: | ---: | ---: |
| original confirmation | 30 | 27 | 24 | 0 |
| primary_minus | 20 | 20 | 24 | 0 |
| heading_parallel_minus | 20 | 20 | 24 | 0 |
| perception_minus_delta | 20 | 20 | 24 | 0 |

These are compatibility checks of published means and denominators, not new behavioral measurements or a fresh confidence-interval analysis. The separate [independent reference checker](verification.json) retains its historical estimator for reproducibility. Reproduce this table with `uv run --locked --no-sync python scripts/check_kinematics_followup.py`; the complete `scripts/build_publication.py` command also includes it. [Machine-readable comparison](kinematics-follow-up.json).

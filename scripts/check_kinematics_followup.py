"""Check corrected geometry against saved published reductions, without model execution.

This compatibility check calls the corrected numerical helper, not a simulator.
It is separate from the independent reference calculations in audit_saved_claims.
Angle and fit conventions: https://numpy.org/doc/1.26/reference/generated/numpy.arctan2.html
and https://numpy.org/doc/1.26/reference/generated/numpy.polyfit.html.
"""
import json
from pathlib import Path

import numpy as np

from rl_locomotion_steering_vectors.kinematics import pair_kinematics

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("reports/fact-check-2026-10-29")
KEYS = ("delta_y_velocity", "delta_y_displacement", "delta_travel_angle_deg",
        "delta_mean_yaw_deg", "delta_gait_lateral_velocity", "yaw_slope_deg_per_100_steps")


def generate(root=ROOT, destination=ROOT):
    """Compare every published K1 cohort and common-window mean to the original JSON."""
    audit = json.loads((root / "reports/ant-classic-005-independent-audit/audit.json").read_text())
    original = root / "runs/ant-classic-005/episodes/confirmation"
    fresh = root / "runs/ant-005-independent-audit/episodes/audit"
    perception = list(fresh.glob("perceived_heading__-*"))
    if len(perception) != 1:
        raise ValueError("Expected exactly one saved negative observation intervention")
    cases = [("original confirmation", original, "lateral__-0.1000", audit["saved"]["kinematics_confirmation"])]
    cases += [(name, fresh, condition, audit["fresh"]["K1"][name]) for name, condition in (
        ("primary_minus", "lateral__-0.1000"), ("heading_parallel_minus", "heading_parallel__-0.1000"),
        ("perception_minus_delta", perception[0].name))]

    def load(path):
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in ("observations", "x_velocity", "y_velocity", "length", "dt")}

    rows = []
    for name, folder, condition, saved in cases:
        base, treated = folder / "baseline", folder / condition
        seeds = sorted(p.name for p in base.glob("*.npz"))
        if seeds != sorted(p.name for p in treated.glob("*.npz")):
            raise ValueError(f"Missing or unmatched pair in {name}")
        pairs = [pair_kinematics(load(base / seed), load(treated / seed)) for seed in seeds]
        usable = [p for p in pairs if p["common_steps"] >= 200]
        if len(usable) != saved["pairs"]:
            raise ValueError(f"Changed historical denominator in {name}")
        errors = [abs(float(np.mean([p[key] for p in usable])) - saved[key]["mean"]) for key in KEYS]
        for window in saved["windows"]:
            index = (window["start"] - 100) // 100
            available = [p for p in usable if len(p["window_delta_yaw_deg"]) > index]
            if len(available) != window["pairs"]:
                raise ValueError(f"Changed window denominator in {name}")
            for key, target in (("window_delta_yaw_deg", "delta_yaw_deg"), ("window_delta_y_velocity", "delta_y_velocity")):
                errors.append(abs(float(np.mean([p[key][index] for p in available])) - window[target]["mean"]))
        error = max(errors)
        if error > 1e-10:
            raise ValueError(f"Changed published kinematics in {name}: {error}")
        rows.append(dict(cohort=name, pairs_checked=len(pairs), retained_pairs=len(usable),
                         mean_fields_checked=len(errors), maximum_absolute_difference=error))

    result = dict(date="2026-10-29", comparison="Current numerical helper versus saved published means and denominators",
                  model_execution=False, absolute_tolerance=1e-10, cohorts=rows)
    table = "\n".join(f"| {r['cohort']} | {r['pairs_checked']} | {r['retained_pairs']} | {r['mean_fields_checked']} | {r['maximum_absolute_difference']:.3g} |" for r in rows)
    report = f"""# Kinematics and statistical-claim follow-up — 2026-10-29

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
{table}

These are compatibility checks of published means and denominators, not new behavioral measurements or a fresh confidence-interval analysis. The separate [independent reference checker](verification.json) retains its historical estimator for reproducibility. Reproduce this table with `uv run --locked --no-sync python scripts/check_kinematics_followup.py`; the complete `scripts/build_publication.py` command also includes it. [Machine-readable comparison](kinematics-follow-up.json).
"""
    outputs = {OUT / "kinematics-follow-up.json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
               OUT / "kinematics-follow-up.md": report}
    for relative, text in outputs.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return [destination / relative for relative in outputs]


if __name__ == "__main__":
    for output in generate():
        print(output.relative_to(ROOT))

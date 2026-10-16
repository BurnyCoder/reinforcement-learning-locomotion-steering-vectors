"""Second-pass audit of the Ant 005 world-Y path shift: estimand, mechanism wording and shared-baseline inference.

Run from the project root: uv run --no-sync python scripts/audit_ant005_path_mechanism.py
Reads saved episodes, results and the recorded y-shift audit only. No policy weights, simulator,
new seeds, selection changes or gate changes. Outputs go to reports/ant-classic-005-mechanism-audit/.
Interpretation: reports/ant-classic-005-mechanism-audit.md. Source: https://docs.python.org/3.12/library/logging.html
"""
import hashlib  # Re-verify that every consumed input file is unchanged after the audit.
import logging  # Timestamped terminal and file records without truncation.
import sys  # Record the literal command and Python version.
from datetime import datetime, timezone  # Distinguish this audit's time from the original research chronology.
from pathlib import Path  # Project-relative inputs and outputs.
import numpy as np  # Array reductions over saved per-reset outcomes.
from audit_ant005_specificity import ISOTROPIC, PRIMARY, RUN, SEEDS, episodes, world_y_differences  # Same run, seeds, pairing and estimator as the y-shift audit.
from audit_scientific_rigor import json_text  # Strict NumPy-aware JSON encoding shared by the earlier audits.
from documentation_checks import Comparisons, EvidenceReader  # Hashing reader and 1e-12 comparator shared by the earlier audits.
from rigor_statistics import body_relative_travel_change, matched_horizon_difference, mean_heading, percentile_interval, post_onset_displacement, sign_flip_pvalue, travel_angle  # Tested inference helpers.
from rl_locomotion_steering_vectors.analysis import _summary  # Production per-episode reduction.
from rl_locomotion_steering_vectors.experiment import condition_name  # Production directory naming for saved conditions.
from rl_locomotion_steering_vectors.runtime import wrapped_angle_difference  # Production circular difference.
from rl_locomotion_steering_vectors.storage import code_identity, start_logging  # Provenance and the project's logging convention.

RECORDED = "reports/ant-classic-005-y-shift-audit/specificity-audit.json"  # The earlier audit whose mechanism and drift statements are re-examined.
LOCKED = condition_name(PRIMARY, -0.1)  # Locked confirmation condition, lateral -0.1.
EVEN_DIRECTIONS = (PRIMARY, "turning", "lateral_bias") + ISOTROPIC  # The nine non-shuffled directions behind the "17 of 18" even-part count.


def describe(values, seed):
    """Local: mean, paired percentile interval, positive count and both sign-flip p-values; global: one summary format for every per-pair estimand."""
    values = np.asarray(values, dtype=float)  # One value per reset pair.
    return {"mean": float(values.mean()), "ci95": percentile_interval(values, seed=seed), "positive_pairs": int((values > 0).sum()), "n_pairs": len(values),
            "sign_flip_pvalue": sign_flip_pvalue(values), "studentized_sign_flip_pvalue": sign_flip_pvalue(values, studentized=True), "t_statistic": float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))}  # Exact-under-sharp-null and weak-null-robust tests side by side.


def confirmation_arms(reader, condition=LOCKED, fitting=False):
    """Local: seed-aligned baseline and treated confirmation arms; global: the locked held-out comparison."""
    base, treated = episodes(reader, "confirmation", "baseline", fitting=fitting), episodes(reader, "confirmation", condition, fitting=fitting)  # Seed-sorted arms.
    if [int(item["seed"]) for item in base] != [int(item["seed"]) for item in treated]:
        raise ValueError("confirmation arms are not aligned by reset seed")  # Positional pairing below requires identical seed order.
    return base, treated  # Pairs share reset seed and onset state.


def matched_horizon_effect(reader, checks):
    """Local: world-Y change over steps both arms recorded; global: remove unequal termination timing from the headline estimand."""
    base, treated = confirmation_arms(reader)  # Locked condition against baseline.
    recorded = world_y_differences(base, treated)  # Production per-reset differences, including the prefix-only zero convention.
    checks.compare("recorded_confirmation_effect", reader.json(f"{RUN}/results.json")["confirmation"][0]["effect"]["effect"], float(recorded.mean()))  # Ties this audit to the recorded +0.40238 m/s.
    matched = [value for value in (matched_horizon_difference(b, t, "y_velocity") for b, t in zip(base, treated)) if value is not None]  # Prefix-only pair has no common post-onset step.
    return {"recorded_estimator": describe(recorded, SEEDS["confirmation"]), "matched_horizon": describe(matched, SEEDS["confirmation"])}  # Interpreted in the report.


def heading_vs_travel(reader, checks):
    """Local: torso-heading change against travel-direction change per pair; global: test whether the gait is rotated rigidly."""
    base, treated = confirmation_arms(reader, fitting=True)  # Observations are needed for heading.
    travel, heading, relative = [], [], []  # Per-pair changes in degrees.
    for before, after in zip(base, treated):
        if min(int(before["length"]), int(after["length"])) <= 100:
            continue  # The shared prefix-only reset has no post-onset direction.
        travel.append(np.degrees(wrapped_angle_difference(travel_angle(after), travel_angle(before))))  # Change of net ground-path direction.
        heading.append(np.degrees(wrapped_angle_difference(mean_heading(after["observations"], after["length"]), mean_heading(before["observations"], before["length"]))))  # Change of circular-mean torso yaw.
        relative.append(np.degrees(body_relative_travel_change(before, after)))  # Travel direction seen from the torso: 0 if rigid, the path change if pure sidestep.
    recorded = reader.json(RECORDED)["calculations"]["heading_time_course"]  # Values the y-shift audit reported.
    checks.compare("travel_angle_change_deg", recorded["travel_angle_change_deg"]["mean"], float(np.mean(travel)))  # Same +3.84 degrees.
    checks.compare("circular_heading_change_deg", recorded["heading_change_deg_by_convention"]["circular_mean_each_arm_available_steps"], float(np.mean(heading)))  # Same +5.23 degrees.
    seed = SEEDS["confirmation"]  # Original confirmation bootstrap seed.
    return {"travel_angle_change_deg": describe(travel, seed), "torso_heading_change_deg": describe(heading, seed), "body_relative_travel_change_deg": describe(relative, seed),
            "share_of_heading_change_seen_in_path": float(np.mean(travel) / np.mean(heading))}  # 1 for a rigid rotation, 0 for a heading change with no path change.


def rotation_check_discrimination(reader):
    """Local: re-run the y-shift audit's displacement 'rotation check' on real pairs and on synthetic pure sidesteps; global: decide whether it is evidence."""
    base, treated = confirmation_arms(reader)  # Full-horizon pairs keep forward distances comparable.
    complete = [(b, t) for b, t in zip(base, treated) if int(b["length"]) == int(t["length"]) == 1000]  # Pairs where neither arm terminated.
    forward = np.mean([post_onset_displacement(b, "x_velocity") for b, _ in complete])  # Mean baseline post-onset forward distance.
    lateral = np.array([post_onset_displacement(t, "y_velocity") - post_onset_displacement(b, "y_velocity") for b, t in complete])  # Measured world-Y displacement change per pair.
    sidesteps = [{**b, "y_velocity": b["y_velocity"] + np.r_[np.zeros(100), np.full(900, change / (900 * float(b["dt"])))]} for (b, _), change in zip(complete, lateral)]  # Baseline path plus a constant world-Y velocity: same displacement change, no torso rotation.
    real = np.degrees([wrapped_angle_difference(travel_angle(t), travel_angle(b)) for b, t in complete])  # Observed travel-angle changes.
    synthetic = np.degrees([wrapped_angle_difference(travel_angle(s), travel_angle(b)) for (b, _), s in zip(complete, sidesteps)])  # Travel-angle changes a pure sidestep would give.
    predict = lambda angles: float(forward * np.tan(np.radians(np.mean(angles))))  # The y-shift audit's 'rigid rotation' prediction.
    return {"complete_pairs": len(complete), "baseline_forward_distance_m": float(forward), "measured_world_y_displacement_change_m": float(lateral.mean()),
            "real_pairs": {"travel_angle_change_deg": float(real.mean()), "predicted_m": predict(real)}, "pure_sidestep_synthetic": {"travel_angle_change_deg": float(synthetic.mean()), "predicted_m": predict(synthetic)}}  # Both match: the check cannot discriminate.


def common_mode_even_parts(reader, checks):
    """Local: even parts of nine directions with the shared validation baseline; global: replace a sign count with one joint seed-resampled interval."""
    recorded = reader.json(RECORDED)["calculations"]["cross_family_world_y"]["directions"]  # Per-direction even parts the y-shift audit reported.
    base = episodes(reader, "validation", "baseline")  # One unsteered arm shared by every direction.
    output = {}  # One record per magnitude.
    for magnitude in (0.05, 0.1):
        signed = {name: [world_y_differences(base, episodes(reader, "validation", condition_name(name, alpha))) for alpha in (magnitude, -magnitude)] for name in EVEN_DIRECTIONS}  # Per-seed changes at both signs.
        even = np.array([(plus + minus) / 2 for plus, minus in signed.values()])  # Directions x resets; every row subtracts the same baseline.
        for name, row in zip(EVEN_DIRECTIONS, even):
            checks.compare(f"even.{name}.{magnitude}", recorded[name]["magnitudes"][str(magnitude)]["even"], float(row.mean()))  # Same even parts as the y-shift audit.
        output[str(magnitude)] = {"per_direction_mean": dict(zip(EVEN_DIRECTIONS, even.mean(axis=1).tolist())), "positive_directions": int((even.mean(axis=1) > 0).sum()),
                                  "cross_direction_mean": describe(even.mean(axis=0), SEEDS["validation"])}  # Resampling seeds of the per-seed average resamples all directions and the shared baseline jointly.
    effect, opposite = (world_y_differences(base, episodes(reader, "validation", condition_name(PRIMARY, alpha))) for alpha in (-0.1, 0.1))  # Selected condition and its opposite sign.
    even = (effect + opposite) / 2  # Per-seed sign-independent part of the selected validation effect.
    indices = np.random.default_rng(SEEDS["validation"]).integers(0, len(effect), size=(2000, len(effect)))  # Same paired-bootstrap index scheme as the production estimator.
    ratio = even[indices].mean(axis=1) / effect[indices].mean(axis=1)  # Sign-independent share of the selected validation effect, per resample.
    baseline = np.array([_summary(item)["lateral"] for item in base])  # Validation baseline world-Y means.
    confirmation = np.array([_summary(item)["lateral"] for item in episodes(reader, "confirmation", "baseline") if not _summary(item)["prefix_only"]])  # Fresh baseline for comparison.
    return {"magnitudes": output, "lateral_even_share_of_selected_effect": {"point": float(even.mean() / effect.mean()), "ci95": np.quantile(ratio, [0.025, 0.975]).tolist()},
            "validation_baseline_world_y": {"mean": float(baseline.mean()), "sd": float(baseline.std(ddof=1)), "se": float(baseline.std(ddof=1) / np.sqrt(len(baseline))), "n": len(baseline)},
            "confirmation_baseline_world_y_excluding_prefix_only": {"mean": float(confirmation.mean()), "sd": float(confirmation.std(ddof=1)), "n": len(confirmation)}}  # Interpreted in the report.


def arm_correlation(reader):
    """Local: per-reset correlation of baseline and treated outcomes; global: measure how much pairing actually reduces variance."""
    def correlation(base, treated, metric):
        """Local: Pearson r and SDs over resets with post-onset data in both arms; global: compare paired and unpaired precision."""
        rows = [(_summary(b)[metric], _summary(t)[metric]) for b, t in zip(base, treated) if not (_summary(b)["prefix_only"] or _summary(t)["prefix_only"])]  # Prefix-only resets have no outcome.
        x, y = np.array(rows).T  # Baseline and treated columns.
        return {"r": float(np.corrcoef(x, y)[0, 1]), "sd_baseline": float(x.std(ddof=1)), "sd_treated": float(y.std(ddof=1)), "sd_difference": float((y - x).std(ddof=1)), "n": len(x)}  # Pairing helps only when r > 0.
    confirmation = {item["vector"] + "/" + f"{item['alpha']:+g}": {metric: correlation(*confirmation_arms(reader, condition_name(item["vector"], item["alpha"])), metric) for metric in ("lateral", "speed")}
                    for item in reader.json(f"{RUN}/results.json")["confirmation"]}  # Locked primary and every locked control.
    base = episodes(reader, "validation", "baseline")  # Shared validation baseline.
    validation = {metric: [correlation(base, episodes(reader, "validation", condition_name(row["vector"], row["alpha"])), metric)["r"] for row in reader.json(f"{RUN}/results.json")["validation"]] for metric in ("lateral", "speed")}  # All 88 validation conditions.
    return {"confirmation": confirmation, "validation_r_summary": {metric: {"median": float(np.median(r)), "min": float(np.min(r)), "max": float(np.max(r)), "n_conditions": len(r)} for metric, r in validation.items()}}  # Interpreted in the report.


def main():
    """Local: run named phases and save evidence; global: an auditable second pass over the featured Ant interpretation."""
    root = Path.cwd(); destination = root / "reports" / "ant-classic-005-mechanism-audit"  # Fixed project-relative output location.
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run from the project root with the saved runs/ directories restored")  # Avoid a vacuous audit elsewhere.
    start_logging(destination)  # Timestamped terminal and file handlers.
    started, reader, checks, command = datetime.now(timezone.utc).isoformat(), EvidenceReader(root), Comparisons(), [sys.executable, *sys.argv]  # Audit context.
    logging.info("Ant 005 path-mechanism audit command: %s", command)  # Complete invocation in the log.
    outputs, failures = {}, []  # Phase results and explicit phase errors.
    phases = {"matched_horizon_effect": lambda: matched_horizon_effect(reader, checks), "heading_vs_travel": lambda: heading_vs_travel(reader, checks), "rotation_check_discrimination": lambda: rotation_check_discrimination(reader),
              "common_mode_even_parts": lambda: common_mode_even_parts(reader, checks), "arm_correlation": lambda: arm_correlation(reader)}  # Independent read-only phases.
    for name, phase in phases.items():
        try:
            outputs[name] = phase(); logging.info("Phase %s complete\n%s", name, json_text(outputs[name]))  # Full phase result in terminal and log.
        except Exception as error:
            failures.append({"phase": name, "error": repr(error)}); logging.exception("Phase %s failed", name)  # Keep failures visible.
    for name, digest in reader.identities.items():
        checks.compare(f"scientific_inputs_unchanged.{name}", digest, hashlib.sha256((root / name).read_bytes()).hexdigest())  # Inputs must be byte-identical afterwards.
    identity = code_identity(root); identity["audit_source_sha256"] = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in ("audit_ant005_path_mechanism.py", "audit_ant005_specificity.py", "rigor_statistics.py", "documentation_checks.py")}  # Exact audit code.
    record = {"created_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "command": command, "python": sys.version, "numpy": np.__version__, "source_identity": identity,
              "passed": not checks.mismatches and not failures, "checked_leaf_fields": checks.count, "max_numeric_error": checks.max_numeric_error, "mismatches": checks.mismatches, "phase_errors": failures,
              "calculations": outputs, "scope": "Saved-data post hoc audit of ant-classic-005. No inference, rollouts, new seeds, selection or gate changes; recorded status unchanged."}  # What this audit does and does not establish.
    for name, value in {"mechanism-audit.json": record, "input-identities.json": reader.identities}.items():
        text = json_text(value); (destination / name).write_text(text, encoding="utf-8")  # Only this audit's own outputs are written.
        logging.info("Saved %s\n%s", destination / name, text)  # Complete evidence in terminal and log.
    logging.info("Audit finished: passed=%s; compared fields=%d; mismatches=%d; failed phases=%d", record["passed"], checks.count, len(checks.mismatches), len(failures))  # Summary line.
    return 0 if record["passed"] else 1  # Nonzero exit on any mismatch or failed phase.


if __name__ == "__main__":
    raise SystemExit(main())  # Importing the module never runs the audit.

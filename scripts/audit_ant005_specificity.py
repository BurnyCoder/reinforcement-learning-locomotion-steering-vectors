"""Offline audit of the Ant 005 world-Y ground-path shift: label specificity, generic drift and mechanism.

Run from the project root: uv run --no-sync python scripts/audit_ant005_specificity.py
Reads saved vectors, results and episodes only. No policy weights, simulator, new seeds,
selection changes or gate changes. Outputs go to reports/ant-classic-005-y-shift-audit/.
Interpretation: reports/ant-classic-005-y-shift-audit.md. Source: https://docs.python.org/3.12/library/logging.html
"""
import hashlib  # Re-verify that every consumed input file is unchanged after the audit.
import logging  # Timestamped terminal and file records without truncation.
import sys  # Record the literal command and Python version.
from datetime import datetime, timezone  # Distinguish this audit's time from the original research chronology.
from pathlib import Path  # Project-relative inputs and outputs.
import numpy as np  # Array reductions over saved per-reset outcomes.
from audit_scientific_rigor import json_text  # Same strict NumPy-aware JSON encoding as the rigor audit.
from documentation_checks import Comparisons, EvidenceReader  # Hashing reader and 1e-12 comparator shared by the earlier audits.
from rigor_statistics import heading_series, mean_heading, odd_even, pairs_for_power, percentile_interval, post_onset_displacement, rank_pvalue, sign_flip_pvalue, travel_angle, window_means  # Tested inference helpers.
from rl_locomotion_steering_vectors.analysis import _summary, paired_effect  # Production per-episode reduction and paired estimator.
from rl_locomotion_steering_vectors.experiment import condition_name  # Production directory naming for saved conditions.
from rl_locomotion_steering_vectors.runtime import wrapped_angle_difference  # Production circular difference.
from rl_locomotion_steering_vectors.storage import code_identity, start_logging  # Provenance and the project's logging convention.

RUN = "runs/ant-classic-005"  # The only experiment examined here.
PRIMARY, MAGNITUDES = "lateral", (0.05, 0.1, 0.2, 0.5)  # Locked vector family and the prespecified validation grid magnitudes.
ISOTROPIC = tuple(f"{family}_random{number}" for family in ("lateral", "turning") for number in range(3))  # Six norm-matched isotropic draws, three per fitted family.
WINDOWS = ((100, 200), (200, 400), (400, 600), (600, 800), (800, 1000))  # Post-onset time bins; the first isolates the onset transient.
SEEDS = {"validation": 710, "confirmation": 720}  # Original bootstrap seeds (experiment.py), reused for the new intervals.


def episodes(reader, phase, condition, *, fitting=False):
    """Local: one saved arm in seed order; global: identical pairing to the recorded analysis."""
    return reader.episodes(f"{RUN}/episodes/{phase}/{condition}", fitting=fitting)  # Seed-sorted file names give aligned arms.


def world_y_differences(base, treated):
    """Local: per-reset treated-minus-baseline world-Y means; global: the recorded estimator before averaging."""
    if [int(item["seed"]) for item in base] != [int(item["seed"]) for item in treated]:
        raise ValueError("arms are not aligned by reset seed")  # Positional pairing below requires identical seed order.
    return np.array([_summary(after)["lateral"] - _summary(before)["lateral"] for before, after in zip(base, treated)])  # Production reduction, including the prefix-only zero convention.


def cross_family_world_y(reader, checks):
    """Local: odd/even world-Y parts for every saved direction; global: test whether the lateral label is what produces the +Y shift."""
    vectors, recorded = reader.arrays(f"{RUN}/vectors.npz"), reader.json(f"{RUN}/results.json")["validation"]  # Saved directions and recorded validation rows.
    unit = vectors[PRIMARY] / np.linalg.norm(vectors[PRIMARY])  # Locked primary direction.
    rows = {(row["vector"], row["alpha"]): row["effect"] for row in recorded}  # Recorded effects keyed by condition.
    base = episodes(reader, "validation", "baseline")  # Shared unsteered arm of the ten validation resets.
    if any(_summary(item)["prefix_only"] for item in base):
        raise ValueError("validation baseline contains a prefix-only reset")  # The zero convention would otherwise enter the odd/even parts.
    output, odd = {}, {}  # Per-direction results and per-seed odd parts for paired comparisons.
    for name in sorted(vectors) + ["lateral_bias"]:  # Ten activation directions plus the constant action-bias comparator.
        entry = {"cosine_with_lateral": float(vectors[name] @ unit / np.linalg.norm(vectors[name])) if name in vectors else None, "magnitudes": {}}  # Geometry relative to the primary.
        for magnitude in MAGNITUDES:
            signed = {}  # Per-sign treated arms and their per-seed differences.
            for alpha in (magnitude, -magnitude):
                treated = episodes(reader, "validation", condition_name(name, alpha))  # Saved arm at this signed strength.
                signed[alpha] = world_y_differences(base, treated)  # Per-reset world-Y change.
                effect = rows[(name, alpha)]  # Recorded row for the same condition.
                checks.compare(f"world_y.{name}.{alpha}", effect["treated_means"]["lateral"] - effect["baseline_means"]["lateral"], float(signed[alpha].mean()))  # Ties this audit to the recorded means.
                as_lateral = paired_effect(base, treated, "lateral", seed=SEEDS["validation"])  # Hypothetical: the lateral gate applied to this condition's world-Y outcome.
                signed[alpha] = (signed[alpha], {"effect": as_lateral["effect"], "ci95": as_lateral["ci95"], "speed_retention": as_lateral["treated_means"]["speed"] / as_lateral["baseline_means"]["speed"],
                                                 "failures_treated": int(round(as_lateral["treated_means"]["failure"] * len(base))), "failures_baseline": int(round(as_lateral["baseline_means"]["failure"] * len(base))), "lateral_gate_if_scored_on_world_y": as_lateral["gate"]})  # Post hoc, never used for selection.
            parts = odd_even(signed[magnitude][0], signed[-magnitude][0])  # Per-seed sign-dependent and sign-independent parts.
            odd[(name, magnitude)] = parts[0]  # Kept for paired comparisons with the primary.
            entry["magnitudes"][str(magnitude)] = {"odd": float(parts[0].mean()), "odd_ci95": percentile_interval(parts[0], seed=SEEDS["validation"]), "even": float(parts[1].mean()), "even_ci95": percentile_interval(parts[1], seed=SEEDS["validation"]),
                                                   "plus": signed[magnitude][1], "minus": signed[-magnitude][1]}  # Both signed conditions remain visible.
        output[name] = entry  # One record per direction.
    for name, entry in output.items():
        for magnitude in MAGNITUDES:
            mine, primary = odd[(name, magnitude)], odd[(PRIMARY, magnitude)]  # Same ten resets, so the comparison is paired.
            difference = np.sign(mine.mean()) * mine - np.sign(primary.mean()) * primary  # Magnitude of each direction's odd response along its own sign.
            entry["magnitudes"][str(magnitude)]["abs_odd_minus_primary"] = {"mean": float(difference.mean()), "ci95": percentile_interval(difference, seed=SEEDS["validation"])}  # Positive: larger sign-dependent world-Y response than the primary.
    ranks = {str(magnitude): {"primary_abs_odd": abs(float(odd[(PRIMARY, magnitude)].mean())), "isotropic_abs_odd": {name: abs(float(odd[(name, magnitude)].mean())) for name in ISOTROPIC},
                              "rank_pvalue": rank_pvalue(abs(float(odd[(PRIMARY, magnitude)].mean())), [abs(float(odd[(name, magnitude)].mean())) for name in ISOTROPIC])} for magnitude in MAGNITUDES}  # Exchangeable-direction rank test among seven.
    small = [entry["magnitudes"][str(magnitude)]["even"] for name, entry in output.items() if not name.endswith("shuffled") for magnitude in (0.05, 0.1)]  # Primaries, isotropic draws and action bias at |alpha| <= 0.1.
    return {"directions": output, "isotropic_rank_tests": ranks, "positive_even_count_non_shuffled_abs_alpha_le_0_1": [int(np.sum(np.array(small) > 0)), len(small)], "n_resets": len(base)}  # Interpreted in the report.


def heading_time_course(reader):
    """Local: binned heading, yaw-rate and world-Y changes after onset; global: distinguish a set-point shift from a sustained turn."""
    results = reader.json(f"{RUN}/results.json")["confirmation"]  # Locked conditions in recorded order.
    locked = condition_name(results[0]["vector"], results[0]["alpha"])  # The selected primary, lateral -0.1.
    base, treated = episodes(reader, "confirmation", "baseline", fitting=True), episodes(reader, "confirmation", locked, fitting=True)  # Observations are needed for heading.
    bins = {field: [] for field in ("heading_deg", "y_velocity", "x_velocity", "yaw_rate")}  # Per-pair window differences.
    angles, matched, circular, available = [], [], [], []  # Per-pair travel-angle and heading-change variants.
    for before, after in zip(base, treated):
        if int(before["seed"]) != int(after["seed"]):
            raise ValueError("confirmation arms are not aligned by reset seed")  # Pairing integrity.
        lengths = int(before["length"]), int(after["length"])  # Durations differ when either arm terminates.
        headings = [np.degrees(heading_series(item["observations"], length)) for item, length in zip((before, after), lengths)]  # Unwrapped torso yaw in degrees.
        bins["heading_deg"].append(window_means(headings[1], lengths[1], WINDOWS) - window_means(headings[0], lengths[0], WINDOWS))  # NaN unless both arms cover the window.
        for field in ("y_velocity", "x_velocity", "yaw_rate"):
            bins[field].append(window_means(after[field], lengths[1], WINDOWS) - window_means(before[field], lengths[0], WINDOWS))  # Same coverage rule for kinematics.
        if min(lengths) > 100:
            angles.append(np.degrees(wrapped_angle_difference(travel_angle(after), travel_angle(before))))  # Change in net ground-path direction.
            shared = min(lengths)  # Steps recorded in both arms.
            matched.append(float(np.mean(headings[1][100:shared] - headings[0][100:shared])))  # Heading change over a common horizon.
            available.append(float(np.mean(headings[1][100:lengths[1]]) - np.mean(headings[0][100:lengths[0]])))  # Each arm's own available steps, as in a whole-episode mean.
            circular.append(float(np.degrees(wrapped_angle_difference(mean_heading(after["observations"], lengths[1]), mean_heading(before["observations"], lengths[0])))))  # The rigor audit's per-episode circular-mean convention.
    windows = {}  # Per-bin summaries over pairs that cover the bin in both arms.
    for index, (start, stop) in enumerate(WINDOWS):
        windows[f"{start}-{stop}"] = {}  # One record per time bin.
        for field, values in bins.items():
            column = np.array(values)[:, index]; column = column[np.isfinite(column)]  # Pairs with complete coverage only.
            windows[f"{start}-{stop}"][field] = {"mean": float(column.mean()), "sd": float(column.std(ddof=1)), "ci95": percentile_interval(column, seed=SEEDS["confirmation"]), "n_pairs": int(len(column))}  # Descriptive bin statistics.
    angles = np.array(angles)  # Pairs with post-onset data in both arms.
    forward = np.mean([post_onset_displacement(item, "x_velocity") for item in base])  # Mean baseline post-onset forward distance.
    lateral = np.mean([post_onset_displacement(b, "y_velocity") - post_onset_displacement(a, "y_velocity") for a, b in zip(base, treated)])  # Measured world-Y displacement change.
    return {"locked_condition": locked, "windows": windows,
            "travel_angle_change_deg": {"mean": float(angles.mean()), "median": float(np.median(angles)), "ci95": percentile_interval(angles, seed=SEEDS["confirmation"]), "positive_pairs": int((angles > 0).sum()), "n_pairs": int(len(angles)), "sign_flip_pvalue": sign_flip_pvalue(angles, seed=0)},
            "rotation_consistency": {"baseline_forward_distance_m": float(forward), "predicted_world_y_displacement_m": float(forward * np.tan(np.radians(angles.mean()))), "measured_world_y_displacement_change_m": float(lateral)},
            "heading_change_deg_by_convention": {"circular_mean_each_arm_available_steps": float(np.mean(circular)), "unwrapped_mean_each_arm_available_steps": float(np.mean(available)), "unwrapped_mean_common_steps": float(np.mean(matched)), "n_pairs": len(matched)}}  # Interpreted in the report.


def design_power(reader):
    """Local: sample-size arithmetic from the observed paired SDs; global: separate adequate effect detection from untestable specificity."""
    validation = world_y_differences(episodes(reader, "validation", "baseline"), episodes(reader, "validation", condition_name(PRIMARY, -0.1)))  # Information available before confirmation.
    confirmation = world_y_differences(episodes(reader, "confirmation", "baseline"), episodes(reader, "confirmation", condition_name(PRIMARY, -0.1)))  # Retrospective check only.
    return {"threshold_m_s": 0.2, "validation_paired_sd": float(validation.std(ddof=1)), "confirmation_paired_sd": float(confirmation.std(ddof=1)),
            "pairs_for_80pct_power_from_validation_sd": pairs_for_power(validation.std(ddof=1), 0.2), "pairs_for_80pct_power_from_confirmation_sd": pairs_for_power(confirmation.std(ddof=1), 0.2), "confirmation_pairs": len(confirmation),
            "random_directions_needed_for_rank_p_le_0_05": next(count for count in range(1, 1000) if rank_pvalue(1., np.zeros(count)) <= 0.05), "random_directions_available": len(ISOTROPIC),
            "approximation": "two-sided normal approximation; slightly underestimates exact paired-t sample sizes"}  # Interpreted in the report.


def main():
    """Local: run named phases and save evidence; global: an auditable check of the featured Ant interpretation."""
    root = Path.cwd(); destination = root / "reports" / "ant-classic-005-y-shift-audit"  # Fixed project-relative output location.
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run from the project root with the saved runs/ directories restored")  # Avoid a vacuous audit elsewhere.
    start_logging(destination)  # Timestamped terminal and file handlers.
    started, reader, checks, command = datetime.now(timezone.utc).isoformat(), EvidenceReader(root), Comparisons(), [sys.executable, *sys.argv]  # Audit context.
    logging.info("Ant 005 world-Y specificity audit command: %s", command)  # Complete invocation in the log.
    outputs, failures = {}, []  # Phase results and explicit phase errors.
    phases = {"cross_family_world_y": lambda: cross_family_world_y(reader, checks), "heading_time_course": lambda: heading_time_course(reader), "design_power": lambda: design_power(reader)}  # Independent read-only phases.
    for name, phase in phases.items():
        try:
            outputs[name] = phase(); logging.info("Phase %s complete", name)  # Progress after each phase.
        except Exception as error:
            failures.append({"phase": name, "error": repr(error)}); logging.exception("Phase %s failed", name)  # Keep failures visible.
    for name, digest in reader.identities.items():
        checks.compare(f"scientific_inputs_unchanged.{name}", digest, hashlib.sha256((root / name).read_bytes()).hexdigest())  # Inputs must be byte-identical afterwards.
    identity = code_identity(root); identity["audit_source_sha256"] = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in ("audit_ant005_specificity.py", "rigor_statistics.py", "documentation_checks.py")}  # Exact audit code.
    record = {"created_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "command": command, "python": sys.version, "numpy": np.__version__, "source_identity": identity,
              "passed": not checks.mismatches and not failures, "checked_leaf_fields": checks.count, "max_numeric_error": checks.max_numeric_error, "mismatches": checks.mismatches, "phase_errors": failures,
              "calculations": outputs, "scope": "Saved-data post hoc audit of ant-classic-005. Cross-family odd/even parts use the ten validation resets; time courses use the thirty confirmation resets. No inference, rollouts, new seeds, selection or gate changes; recorded status unchanged."}  # What this audit does and does not establish.
    for name, value in {"specificity-audit.json": record, "input-identities.json": reader.identities}.items():
        text = json_text(value); (destination / name).write_text(text, encoding="utf-8")  # Only this audit's own outputs are written.
        logging.info("Saved %s\n%s", destination / name, text)  # Complete evidence in terminal and log.
    logging.info("Audit finished: passed=%s; compared fields=%d; mismatches=%d; failed phases=%d", record["passed"], checks.count, len(checks.mismatches), len(failures))  # Summary line.
    return 0 if record["passed"] else 1  # Nonzero exit on any mismatch or failed phase.


if __name__ == "__main__":
    raise SystemExit(main())  # Importing the module never runs the audit.

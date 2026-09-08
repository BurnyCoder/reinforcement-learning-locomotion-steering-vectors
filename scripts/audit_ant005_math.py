"""Independently recompute Ant005 evidence from saved arrays; never collect or alter research data.

Sources: https://numpy.org/doc/1.26/reference/generated/numpy.quantile.html
https://numpy.org/doc/1.26/reference/generated/numpy.mean.html
https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.apply.html
This audit deliberately does not call project extraction or effect-estimation functions.
"""
import hashlib  # Bind the independent audit to its inputs and its own implementation.
import json  # Read immutable manifests and publish portable audit results.
import logging  # Record complete audit findings through the project's timestamped handlers.
from datetime import datetime, timezone  # Timestamp the audit separately from the original experiments.
from pathlib import Path  # Read only the named project run and write only the audit destination.

import numpy as np  # Reuse primary-library array arithmetic without importing the production estimator.
from rl_locomotion_steering_vectors.storage import start_logging  # Reuse terminal/file logging only; all numerical oracle calculations remain independent.


def arrays(path):
    """Read numeric/Unicode arrays with executable deserialization disabled."""
    with np.load(path, allow_pickle=False) as archive:  # Reject object-array pickle payloads.
        return {name: archive[name] for name in archive.files}  # Close the archive after materializing its members.


def balanced(rows, field):
    """Average selected windows within each seed, then average contributing episodes equally."""
    identities = sorted({row["seed"] for row in rows})  # Episode identities are independent contributing units.
    return np.mean([np.mean([row[field] for row in rows if row["seed"] == seed], axis=0) for seed in identities], axis=0)  # This independently expresses the documented two-stage weighting.


def contrast(rows, labels=None):
    """Compute independent quartile groups and their raw high-minus-low activation difference."""
    values = np.array([row["y_velocity"] for row in rows]) if labels is None else labels  # Default labels are measured world-y velocity.
    cuts = np.quantile(values, [.25, .75], method="linear")  # State the exact NumPy quartile convention explicitly.
    low = [row for row, value in zip(rows, values) if value <= cuts[0]]  # Keep full eligible low-quartile windows.
    high = [row for row, value in zip(rows, values) if value >= cuts[1]]  # Keep full eligible high-quartile windows.
    return balanced(high, "activation") - balanced(low, "activation"), low, high, cuts  # Return groups for independently checkable descriptive contrasts.


def body_motion(episode):
    """Rotate recorded world-horizontal velocity into the pre-action yaw-aligned frame; describe rather than redefine the target."""
    quaternion = np.asarray(episode["observations"][:, 1:5], dtype=np.float64).copy()  # Copy Ant's scalar-first quaternion so diagnostic normalization cannot mutate the evidence in memory.
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)  # Normalize roundoff before calculating orientation.
    w, x, y, z = quaternion.T  # Respect MuJoCo's actual quaternion component order.
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # Compute world-z heading without interpreting a quaternion component as an angle.
    forward = np.cos(yaw) * episode["x_velocity"] + np.sin(yaw) * episode["y_velocity"]  # Apply the inverse planar rotation to measured world velocity.
    lateral = -np.sin(yaw) * episode["x_velocity"] + np.cos(yaw) * episode["y_velocity"]  # Separate yaw-aligned side motion from changing body heading; yaw uses the observation before the transition.
    return {"yaw_heading": yaw, "body_forward": forward, "body_lateral": lateral}  # These descriptive quantities are not added to eligibility, fitting, selection, or gates.


def summary(episode, warmup):
    """Independently summarize complete post-prefix physical outcomes, retaining prefix-only failures."""
    length = int(episode["length"])  # Use recorded duration rather than assuming completion.
    values = {key: float(np.mean(episode[key][warmup:])) if length > warmup else 0. for key in ("x_velocity", "y_velocity", "height", "yaw_rate", "effort", "inversion", "bad_contact")}  # Preserve the project's explicit zero value for unavailable post-prefix outcomes.
    motion = body_motion(episode)  # Inspect whether the already-recorded lateral displacement accompanies heading changes.
    values.update({key: float(np.mean(value[warmup:])) if length > warmup else 0. for key, value in motion.items() if key != "yaw_heading"})  # Report average yaw-aligned planar motion without changing world-y effect estimates.
    values["circular_mean_heading"] = float(np.arctan2(np.sin(motion["yaw_heading"][warmup:]).mean(), np.cos(motion["yaw_heading"][warmup:]).mean())) if length > warmup else 0.  # Avoid averaging raw heading angles across a wrap boundary.
    values["failure"] = bool(episode["terminated"]) or length <= warmup or values["inversion"] > .05 or values["bad_contact"] > .05  # Reproduce the registered physical episode failure rule independently.
    values["prefix_only"] = length <= warmup  # Keep the strict pre-onset condition visible.
    return values  # No failed episodes are removed from the paired denominator.


def main():
    """Audit fitting, controls, data boundaries, actual dispatch identities, and held-out estimates."""
    root = Path.cwd()  # Stay inside the current project.
    run = root / "runs" / "ant-classic-005"  # Audit only the specific user-requested intervention.
    audit_folder = root / "reports" / "ant-classic-005"  # Publish reproducible audit evidence separately from the original experiment.
    start_logging(audit_folder)  # Preserve complete timestamped realtime terminal and file output in the report directory.
    manifest = json.loads((run / "manifest.json").read_text())  # Read policy and data provenance.
    diagnostics = json.loads((run / "vector_diagnostics.json").read_text())  # Use saved numbers only as comparison targets.
    vectors = arrays(run / "vectors.npz")  # Load the actual deployed steering directions.
    splits, warmup = manifest["seed_splits"], manifest["config"]["warmup"]  # Use the original immutable schedule.
    episodes = {seed: arrays(run / "episodes" / "fit" / "baseline" / f"{seed}.npz") for seed in splits["fit"]}  # Only fitting episodes enter reconstruction.
    fields = ("x_velocity", "y_velocity", "height", "yaw_rate", "effort", "inversion", "bad_contact")  # These are the exact physical fitting/eligibility descriptors.
    windows, discarded_tail = [], 0  # Retain every complete nonoverlapping fitting window and count incomplete tails.
    for seed, episode in episodes.items():
        length = int(episode["length"])  # Preserve early terminations when determining available fitting data.
        motion = body_motion(episode)  # Compute descriptive orientation-aligned motion once per fitting episode.
        discarded_tail += max(0, length - warmup) % 100  # Incomplete windows never enter the quartiles or RMS.
        for start in range(warmup, length - 99, 100):
            row = {field: float(np.mean(episode[field][start:start + 100])) for field in fields}  # Pair each window's label with exactly its captured activations.
            row.update(seed=seed, start=start, activation=episode["activations"][start:start + 100].mean(axis=0, dtype=np.float64))  # Retain the source episode rather than treating windows as independent runs.
            row.update({key: float(np.mean(value[start:start + 100])) for key, value in motion.items() if key != "yaw_heading"})  # Retain body-aligned velocity for confounding assessment only.
            heading = motion["yaw_heading"][start:start + 100]  # Use the same full fitting window for descriptive heading.
            row["circular_mean_heading"] = float(np.arctan2(np.sin(heading).mean(), np.cos(heading).mean()))  # Preserve meaningful mean orientation within the window.
            row["healthy"] = bool(np.isfinite(episode["activations"][start:start + 100]).all() and all(np.isfinite(row[field]) for field in fields) and row["inversion"] <= .05 and row["bad_contact"] <= .05)  # Apply the published fitting eligibility rule.
            windows.append(row)  # Preserve excluded windows for explicit audit counts.
    floor = manifest["config"]["fit_min_speed_fraction"] * np.median([row["x_velocity"] for row in windows if row["healthy"]])  # Derive the locomotion floor only from fitting windows.
    eligible = [row for row in windows if row["healthy"] and row["x_velocity"] > floor]  # Retain exactly the specified finite healthy running windows.
    raw, low, high, cuts = contrast(eligible)  # Independently reconstruct the lateral direction.
    activation_sets = [np.concatenate([episodes[seed]["activations"][row["start"]:row["start"] + 100] for row in eligible if row["seed"] == seed]).astype(np.float64) for seed in sorted({row["seed"] for row in eligible})]  # Use every eligible timestep while retaining episode boundaries.
    center = np.mean([values.mean(axis=0) for values in activation_sets], axis=0)  # Compute the documented episode-balanced activation mean.
    rms = float(np.sqrt(np.mean([np.mean(np.sum((values - center) ** 2, axis=1)) for values in activation_sets])))  # Independent two-pass centered RMS avoids copying the production second-moment identity.
    rng = np.random.default_rng(np.random.SeedSequence([0, 3]))  # Reproduce the registered behavior-specific lateral stream.
    halves = np.array_split(rng.permutation(sorted({row["seed"] for row in eligible})), 2)  # Split whole contributing episodes for the original stability diagnostic.
    half_raw = [contrast([row for row in eligible if row["seed"] in identities])[0] for identities in halves]  # Independently refit quartiles in each disjoint half.
    cosine = float(np.dot(*half_raw) / (np.linalg.norm(half_raw[0]) * np.linalg.norm(half_raw[1])))  # Quantify fitting-direction stability without calling it causal proof.
    family = {"lateral": raw}  # Reconstruct the learned contrast before its norm controls.
    family.update({f"lateral_random{index}": rng.standard_normal(256) for index in range(3)})  # Reproduce the actual pre-validation random directions.
    shuffled_labels = rng.permutation(np.array([row["y_velocity"] for row in eligible]))  # Preserve the saved window-label randomization protocol.
    family["lateral_shuffled"] = contrast(eligible, shuffled_labels)[0]  # Reapply the same episode-balanced high/low estimator to the shuffled labels.
    comparison = {}  # Quantify differences from the real treatment files, not from an assumed formula.
    for name, direction in family.items():
        scaled = (direction * (rms / np.linalg.norm(direction))).astype(np.float32)  # Restore the actual runtime normalization/dtype.
        comparison[name] = {"raw_max_abs_error": float(np.max(np.abs(direction - np.array(diagnostics["vectors"][name]["raw_vector"])))), "scaled_max_abs_error": float(np.max(np.abs(scaled - vectors[name]))), "scaled_bitwise_equal": scaled.tobytes() == vectors[name].tobytes(), "saved_norm": float(np.linalg.norm(vectors[name]))}  # Report exact numerical agreement and norm matching.
        assert comparison[name]["scaled_bitwise_equal"]  # Any different deployed component invalidates the claim that the saved vector came from this procedure.
    groups = {label: {"windows": len(rows), "episodes": len({row["seed"] for row in rows}), "means": {field: float(balanced(rows, field)) for field in (*fields, "body_forward", "body_lateral", "circular_mean_heading")}} for label, rows in (("low", low), ("high", high))}  # Expose correlated gait/posture/forward/heading differences rather than claiming a disentangled concept.
    all_seeds = [seed for group in splits.values() for seed in group]  # Read all original partitions together.
    assert len(all_seeds) == len(set(all_seeds))  # Any fitting/validation/confirmation overlap would invalidate this audit.
    selection = json.loads((run / "selection.json").read_text())  # Held-out settings must come from a frozen earlier choice.
    validation = json.loads((run / "validation.json").read_text())  # Inspect control opportunities without selecting new settings.
    grid = sorted(float(value) for value in manifest["config"]["strength_grid"].split(","))  # Compare against the originally configured grid.
    opportunities = {name: sorted(row["alpha"] for row in validation if row["vector"] == name) for name in (*family, "lateral_bias")}  # Include the separate action-space comparator.
    assert all(values == grid for values in opportunities.values())  # All members must receive the same eight nonzero calibration opportunities.
    biases = arrays(run / "action_biases.npz")  # Distinguish eight-dimensional torque offsets from 256-dimensional activation vectors.
    effects, height_checks = [], []  # Independently recompute paired effects and validate the z measurement alignment.
    for phase, recorded in (("validation", [row for row in validation if row["vector"] == "lateral" and row["alpha"] in (-.1, .1)]), ("confirmation", json.loads((run / "confirmation.json").read_text()))):
        baseline = {seed: arrays(run / "episodes" / phase / "baseline" / f"{seed}.npz") for seed in splits[phase]}  # Keep the complete prespecified paired cohort.
        for row in recorded:
            name, alpha = row["vector"], row["alpha"]  # Use original validation/confirmation choices without any new tuning.
            folder = run / "episodes" / phase / f"{name}__{alpha:+.4f}"  # Match the actual persisted condition name.
            applied = {seed: arrays(folder / f"{seed}.npz") for seed in splits[phase]}  # Read all successes, terminations, and pre-onset failures.
            identity = json.loads((folder / "identity.json").read_text())  # Verify what the runtime was actually instructed to apply.
            expected_vector = None if name in biases else hashlib.sha256(vectors[name].astype(np.float32).tobytes()).hexdigest()  # Candidate/random/shuffled directions act at the hidden layer.
            expected_bias = hashlib.sha256(np.asarray(alpha * biases[name], dtype=np.float32).tobytes()).hexdigest() if name in biases else None  # The torque comparator alone receives an action bias.
            assert identity["vector_sha256"] == expected_vector and identity["bias_sha256"] == expected_bias and identity["alpha"] == alpha  # Reject hidden replacement by another intervention mechanism.
            assert identity["policy_sha256"] == manifest["provenance"]["parameter_sha256"] and not identity["stochastic"]  # Evaluation must use the same frozen deterministic actor.
            before, after, prefix_only = [], [], []  # Keep one row per independent matched reset.
            for seed in splits[phase]:
                left, right = baseline[seed], applied[seed]  # Compare the exact paired stored episodes.
                assert left["onset_hash"] == right["onset_hash"] and np.array_equal(left["actions"][:warmup], right["actions"][:warmup])  # Both attained onsets and shared pre-onset failures must have identical physical prefixes.
                before.append(summary(left, warmup))  # Independently retain every baseline outcome.
                after.append(summary(right, warmup))  # Independently retain every treatment outcome.
                if int(left["length"]) <= warmup or int(right["length"]) <= warmup:
                    prefix_only.append({"seed": seed, "baseline_length": int(left["length"]), "treated_length": int(right["length"]), "complete_arrays_equal": all(np.array_equal(left[key], right[key]) for key in left)})  # Make unreachable onsets explicit rather than silently excluding them.
                height_checks.append(float(np.max(np.abs(right["height"][:-1] - right["observations"][1:, 0]))))  # Ant observation[0] is next-state root z; this is independent of horizontal displacement velocity.
            delta = np.array([b["y_velocity"] - a["y_velocity"] for a, b in zip(before, after)])  # Preserve signed world-y treatment-minus-baseline effects.
            sample = np.random.default_rng(row["effect"]["bootstrap"]["seed"]).integers(0, len(delta), size=(2000, len(delta)))  # Recreate the original whole-pair bootstrap stream without using production analysis.
            ci = np.quantile(delta[sample].mean(axis=1), [.025, .975])  # Independently recompute the published percentile interval.
            assert np.allclose(ci, row["effect"]["ci95"], rtol=0, atol=1e-12) and abs(delta.mean() - row["effect"]["effect"]) < 1e-12  # Numerical summaries must match the saved full trajectories.
            means = {label: {key: float(np.mean([item[key] for item in rows])) for key in rows[0]} for label, rows in (("baseline", before), ("treated", after))}  # Expose all competence and secondary-effect descriptors.
            effects.append({"phase": phase, "vector": name, "alpha": alpha, "n_pairs": len(delta), "effect": float(delta.mean()), "ci95": ci.tolist(), "recorded_gate": row["effect"]["gate"], "means": means, "forward_retention": means["treated"]["x_velocity"] / means["baseline"]["x_velocity"], "failed_baseline_seeds": [seed for seed, item in zip(splits[phase], before) if item["failure"]], "failed_treated_seeds": [seed for seed, item in zip(splits[phase], after) if item["failure"]], "prefix_only_pairs": prefix_only, "dispatch_identity": identity})  # Retain complete qualified evidence, including every rejected gate.
    output = {"created_utc": datetime.now(timezone.utc).isoformat(), "run": str(run.relative_to(root)), "source_files": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (run / "manifest.json", run / "vectors.npz", run / "vector_diagnostics.json", run / "selection.json", Path(__file__))},
              "seed_splits": {name: {"n": len(values), "first": min(values), "last": max(values)} for name, values in splits.items()}, "all_seed_splits_disjoint": True,
              "fitting": {"episodes": len(episodes), "total_full_windows": len(windows), "eligible_windows": len(eligible), "discarded_tail_steps": discarded_tail, "minimum_speed_floor": float(floor), "quartile_cuts": cuts.tolist(), "groups": groups, "lateral_contrast": float(balanced(high, "y_velocity") - balanced(low, "y_velocity")), "raw_norm": float(np.linalg.norm(raw)), "activation_rms": rms, "saved_rms_error": abs(rms - diagnostics["activation_rms"]), "split_half_cosine": cosine, "family_comparison": comparison,
                          "window_world_y_heading_correlation": float(np.corrcoef([row["y_velocity"] for row in eligible], [row["circular_mean_heading"] for row in eligible])[0, 1]), "body_frame_convention": "descriptive inverse world-z yaw rotation; pre-action observation quaternion and transition-average horizontal velocity; no pitch/roll rotation and no effect on selection or gates"},
              "equal_validation_strength_opportunities": opportunities, "selection_created_utc": selection["created_utc"], "paired_recomputations": effects, "max_root_z_observation_alignment_error": max(height_checks), "replication_file_exists": (run / "replication.json").exists(), "application_file_exists": (run / "application.json").exists(), "research_state": json.loads((run / "results.json").read_text())["status"]}  # Save provenance and numerical conclusions without changing the original attempt.
    destination = audit_folder / "math-audit.json"  # Keep the published independent audit beside the report without changing original run artifacts.
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Preserve complete strict-JSON evidence for the report author.
    logging.info("Independent Ant005 math audit saved to %s\n%s", destination, json.dumps(output, indent=2, allow_nan=False))  # Record complete provenance, controls, estimates, and limitations in both realtime handlers without truncation.


if __name__ == "__main__":
    main()  # Execute the audit only when explicitly invoked, never during imports.

"""Local: audit saved experiment phases; global: preserve every mismatch and input identity.

Sources: https://docs.python.org/3.12/library/hashlib.html
https://numpy.org/doc/1.26/reference/generated/numpy.load.html
Numerical calculations live in the independent documentation_evidence module.
"""
import hashlib  # Bind conclusions to actual local bytes, not merely filenames.
import json  # Read the non-executable published evidence format.
import logging  # Report progress after each bounded experiment phase.
import numpy as np  # Load arrays safely and compare explicitly documented tolerances.
from documentation_evidence import METRICS, activation_rms, contrast, cosine, episode_summary, make_windows, paired_summary, sensitivity, yaw  # Reuse the independent oracle across all audit phases.

RUNS = ("hc-classic-001", "hc-running-002", "hc-height-speed-003", "hc-height-speed-004", "ant-classic-005")  # Freeze the documented five-experiment scope.


class EvidenceReader:
    """Local: read only named project evidence; global: identify and later verify every consumed file."""
    def __init__(self, root):
        """Local: establish an explicit root; global: prevent accidental reads outside this project."""
        self.root, self.identities = root.resolve(), {}  # Hashes are keyed by portable project-relative names.

    def path(self, relative):
        """Local: record source identity on first use; global: make every numerical result traceable."""
        path = (self.root / relative).resolve()  # Resolve before validating containment.
        name = path.relative_to(self.root).as_posix()  # An escaping path raises instead of reading an unrelated folder.
        if name not in self.identities:
            self.identities[name] = hashlib.sha256(path.read_bytes()).hexdigest()  # Hash actual scientific input bytes, including NPZ metadata.
        return path  # Returning a path allows NumPy to decompress only requested members.

    def json(self, relative):
        """Local: decode one identified JSON file; global: avoid executable serialization."""
        return json.loads(self.path(relative).read_text(encoding="utf-8"))  # Credential configuration is never requested by this reader.

    def arrays(self, relative, fields=None):
        """Local: materialize selected numeric members; global: keep evaluation memory and deserialization bounded."""
        with np.load(self.path(relative), allow_pickle=False) as archive:  # Object-array pickles remain disabled explicitly.
            return {key: archive[key] for key in (fields if fields is not None else archive.files)}  # Close the ZIP after copying the required arrays.

    def episodes(self, relative, *, fitting=False):
        """Local: read complete saved reset files; global: never invoke a rollout or change experimental splits."""
        paths = sorted((self.root / relative).glob("*.npz"))  # Names are ordered deterministically for original bootstrap and sensitivity conventions.
        fields = None if fitting else list(METRICS.values()) + ["seed", "length", "terminated", "truncated", "onset_hash", "dt"]  # Evaluation does not need large observation/activation matrices.
        if not paths:
            raise FileNotFoundError(f"No saved episodes in {relative}")  # A fresh clone without raw artifacts must fail visibly, never report a vacuous pass.
        return [self.arrays(path.relative_to(self.root), fields) for path in paths]  # Every consumed raw file receives its own identity.


class Comparisons:
    """Local: accumulate leaf comparisons; global: retain all discrepancies instead of stopping at the first."""
    def __init__(self):
        """Local: initialize explicit coverage; global: distinguish a real pass from no checks."""
        self.count, self.max_numeric_error, self.mismatches = 0, 0., []  # A failed comparison always survives into the audit JSON.

    def compare(self, label, expected, actual):
        """Local: compare a reference tree at absolute tolerance 1e-12; global: identify exact artifact fields."""
        if isinstance(expected, dict):
            for key, value in expected.items():
                if key not in actual:
                    self.mismatches.append({"field": f"{label}.{key}", "expected": value, "actual": "MISSING"})  # Missing evidence cannot be silently skipped.
                else:
                    self.compare(f"{label}.{key}", value, actual[key])  # Preserve nested numerical provenance in mismatch labels.
            return  # Dictionary nodes themselves are not additional independent comparisons.
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
                self.mismatches.append({"field": label, "expected": expected, "actual": actual})  # Structured evidence must have exactly the same number of members.
                return  # Do not silently truncate unequal lists through zip.
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.compare(f"{label}[{index}]", left, right)  # Recursive comparison also supports condition lists and early-ending dictionaries.
            return  # Numerical leaves below retain the same strict absolute tolerance.
        self.count += 1  # Scalars and complete numerical arrays each count as one checked field.
        if expected is None or isinstance(expected, str):
            equal = expected == actual  # Metadata and absent floors require exact agreement.
        else:
            left, right = np.asarray(expected), np.asarray(actual)  # Uniform scalar/array handling includes seed sequences and vectors.
            equal = left.shape == right.shape and bool(np.allclose(left, right, rtol=0, atol=1e-12))  # A relative tolerance must not hide scaling mistakes.
            if left.shape == right.shape and np.issubdtype(left.dtype, np.number) and left.size:
                self.max_numeric_error = max(self.max_numeric_error, float(np.max(np.abs(left - right))))  # Report tiny arithmetic differences even when accepted.
        if not equal:
            self.mismatches.append({"field": label, "expected": expected, "actual": actual})  # JSON output contains the complete discrepancy, never a truncated preview.


def audit_conditions(reader, checks):
    """Local: reproduce all 332 saved comparisons; global: verify every reported arm mean, effect and interval."""
    output = {}  # Keep phase/condition identity with each freshly reconstructed estimate.
    for run in RUNS:
        records = reader.json(f"reports/{run}/results.json")  # Published compact results are the reference claims.
        checks.compare(f"{run}.raw_compact_results", records, reader.json(f"runs/{run}/results.json"))  # Guard against a compact report silently diverging from the actual run.
        output[run] = {}  # An unrun stage remains an explicit empty list.
        for phase in ("validation", "confirmation"):
            rows = records.get(phase, []); estimates = []  # There are no synthetic successful conditions in empty phases.
            baseline = reader.episodes(f"runs/{run}/episodes/{phase}/baseline") if rows else []  # Reuse the actual baseline arm across every condition on this split.
            for row in rows:
                name = f"{row['vector']}__{row['alpha']:+.4f}"  # Match the immutable condition-directory naming convention.
                treated = reader.episodes(f"runs/{run}/episodes/{phase}/{name}")  # Read every failed and completed treatment episode.
                estimate = paired_summary(baseline, treated, row["behavior"], row["effect"]["bootstrap"]["seed"])  # Reconstruct statistics independently of production analysis.py.
                expected = {key: row["effect"][key] for key in estimate}  # Compare all oracle outputs; gate decisions are preserved separately as historical metadata.
                checks.compare(f"{run}.{phase}.{name}", expected, estimate)  # Differences identify the exact run, condition and numerical field.
                estimates.append({"vector": row["vector"], "alpha": row["alpha"], "behavior": row["behavior"], "historical_gate": row["effect"]["gate"], **estimate})  # A historical gate is not a freshly selected intervention.
            output[run][phase] = estimates  # Retain individual unfavorable as well as favorable results.
            logging.info("Recomputed %s %s: %d conditions", run, phase, len(estimates))  # Long read-only audits remain visibly productive.
    return output  # Complete means and intervals become the published numerical evidence.


def audit_fitting(reader, checks):
    """Local: reconstruct primary vectors and RMS; global: verify extraction independently from saved metadata."""
    output = {}  # Retargeted runs are checked by array identity in a separate phase.
    for run in (RUNS[0], RUNS[1], RUNS[4]):
        episodes = reader.episodes(f"runs/{run}/episodes/fit/baseline", fitting=True)  # Use only original fitting arrays.
        saved = reader.json(f"reports/{run}/vector_diagnostics.json")  # Raw vectors, scale and group statistics are reference evidence.
        rows, stats = make_windows(episodes, fraction=saved["windows"].get("minimum_speed_fraction", 0.))  # Preserve original 001 eligibility and later fitting-only floors.
        checks.compare(f"{run}.windows", {key: value for key, value in saved["windows"].items() if key != "speed_filter_reason"}, stats)  # Compare every numeric exclusion and window count.
        rms = activation_rms(episodes, rows)  # Reconstruct within-window centered variation with equal episode weighting.
        checks.compare(f"{run}.activation_rms", saved["activation_rms"], rms)  # A correct raw direction with incorrect strength scaling is still an audit failure.
        arrays = reader.arrays(f"reports/{run}/vectors.npz"); details = {}  # Published vectors must match the independently reconstructed direction.
        for behavior in saved["behaviors"]:
            raw, detail = contrast(rows, behavior); norm = float(np.linalg.norm(raw))  # Return numerical contrast and contributor counts before normalizing.
            detail["raw_norm"] = norm  # Expose whether scale normalization hides a small original contrast.
            rng = np.random.default_rng(np.random.SeedSequence([saved["seed"], list(METRICS).index(behavior)]))  # Reproduce the original behavior-specific split RNG before random controls were drawn.
            halves = np.array_split(rng.permutation(sorted({row["seed"] for row in rows})), 2)  # Independent reset halves retain every window within its original episode.
            directions = [contrast([row for row in rows if row["seed"] in identities], behavior)[0] for identities in halves]  # Each half recomputes its own quartile/bin boundaries.
            detail["split_half_cosine"] = float(np.clip(np.dot(*directions) / (np.linalg.norm(directions[0]) * np.linalg.norm(directions[1])), -1, 1))  # Cosine diagnoses extraction stability, never causal usefulness.
            checks.compare(f"{run}.{behavior}.groups", {key: saved["behaviors"][behavior][key] for key in detail}, detail)  # Cover all group and speed-bin values, not just vector direction.
            checks.compare(f"{run}.{behavior}.raw_vector", saved["vectors"][behavior]["raw_vector"], raw.tolist())  # Compare the complete float64 numerical oracle.
            scaled = (raw * (rms / norm)).astype(np.float32)  # Match the original actor dtype only after independent float64 calculation.
            exact = scaled.dtype == arrays[behavior].dtype and scaled.shape == arrays[behavior].shape and scaled.tobytes() == arrays[behavior].tobytes()  # Exact identity is stronger than a plotting-level tolerance.
            checks.compare(f"{run}.{behavior}.scaled_bytes_equal", True, exact)  # Any changed component fails the identity check.
            details[behavior] = {**detail, "raw_vector": raw.tolist(), "scaled_vector": scaled.tolist(), "scaled_bytes_equal": exact}  # Retain sufficient fresh evidence to inspect scaling directly.
        output[run] = {"windows": stats, "activation_rms": rms, "behaviors": details}  # Original episode counts remain distinct from qualifying windows.
        if run == RUNS[0]:
            output[run]["sensitivity"] = inspect_halfcheetah_sensitivity(episodes, rows, checks)  # Preserve the decisive 001 rejection calculations too.
        logging.info("Recomputed %s fitting: %d windows, RMS %.12g", run, len(rows), rms)  # Numerical progress is recorded in both log handlers.
    return output  # No original vector, fitting data or selection is overwritten.


def audit_partitions(reader, checks):
    """Local: check reset partitions and inherited arrays; global: rule out accidental reuse of outcome data."""
    output, reuse = {}, {}  # Cross-run fitting reuse is allowed only for the explicitly retargeted pair.
    source = reader.arrays("reports/hc-running-002/vectors.npz")  # This is the fixed height family used by both retargets.
    for run in RUNS:
        splits = reader.json(f"reports/{run}/manifest.json")["seed_splits"]  # Preserve reserved as well as consumed reset IDs.
        overlaps = [(left, right) for left in splits for right in splits if left < right and set(splits[left]) & set(splits[right])]  # An episode cannot supply two within-run scientific partitions.
        checks.compare(f"{run}.partition_overlap_count", 0, len(overlaps))  # Empty overlap is necessary regardless of phase completion.
        counts = {phase: len(list((reader.root / "runs" / run / "episodes" / phase / "baseline").glob("*.npz"))) for phase in splits}  # File counts distinguish reserved seeds from executed episodes.
        for phase, seeds in splits.items():
            actual = sorted(int(path.stem) for path in (reader.root / "runs" / run / "episodes" / phase / "baseline").glob("*.npz"))  # Compare actual reset names for consumed stages.
            checks.compare(f"{run}.{phase}.saved_seeds", seeds if actual else [], actual)  # Missing individual files fail rather than quietly shrinking an executed split.
            for seed in seeds:
                reuse.setdefault(seed, []).append((run, phase))  # Detect any cross-run overlap, including reserved held-out data.
        identities = {}  # Imported treatment keys are aliases, not refitted vectors.
        if run in RUNS[2:4]:
            imported = reader.arrays(f"reports/{run}/vectors.npz")  # Compare all five family members, including negative controls.
            for name, vector in imported.items():
                original = source[name.replace("speed", "height", 1)]  # Retargeting changes the evaluation label only.
                identities[name] = vector.dtype == original.dtype and vector.shape == original.shape and vector.tobytes() == original.tobytes()  # Require exact imported array identity.
                checks.compare(f"{run}.import.{name}", True, identities[name])  # Preserve favorable/unfavorable controls equally.
        output[run] = {"seed_splits": splits, "baseline_file_counts": counts, "within_run_overlaps": overlaps, "imported_height_family_bytes_equal": identities}  # Explicit zero counts document unexecuted phases.
    repeated = sorted({tuple(values) for values in reuse.values() if len(values) > 1})  # Compress identical reuse patterns without losing participating phases.
    allowed = {tuple((run, phase) for run in (RUNS[1], RUNS[2], RUNS[3])) for phase in ("diagnostic", "fit")}  # Only declared source fitting/diagnostic reuse is acceptable.
    checks.compare("partitions.undeclared_cross_run_reuse_count", 0, sum(group not in allowed for group in repeated))  # Validation and held-out reset reuse is never silently accepted.
    return {"runs": output, "cross_run_reuse_groups": repeated}  # No generated reset is consumed by this read-only audit.


def inspect_episode_group(episodes):
    """Local: reproduce Ant completion/range descriptors; global: make variable-duration outcomes explicit."""
    keys = ("x_velocity", "y_velocity", "yaw_rate", "height", "bad_contact", "inversion")  # These physical quantities appeared in the original inspection.
    rows = [{key: float(item[key][100:].mean()) if int(item["length"]) > 100 else 0. for key in keys} for item in episodes]  # Preserve the documented prefix-only zero convention.
    early = [{"seed": int(item["seed"]), "length": int(item["length"]), "height_final": float(item["height"][-1]), "inversion_steps": int(item["inversion"].sum()), "bad_contact_steps": int(item["bad_contact"].sum()), "terminated": bool(item["terminated"]), "truncated": bool(item["truncated"])} for item in episodes if int(item["length"]) < 1000]  # All early endings remain recorded with measured endpoint properties.
    return {"n": len(episodes), "completed": len(episodes) - len(early), "early": early, "mean_postwarmup": {key: float(np.mean([row[key] for row in rows])) for key in keys}, "postwarmup_ranges": {key: [min(row[key] for row in rows), max(row[key] for row in rows)] for key in keys}}  # Ranges describe episode means rather than instantaneous extreme velocities.


def audit_ant_sensitivity(reader, checks):
    """Local: independently recreate the full original Ant inspection; global: publish previously local evidence."""
    saved = reader.json("reports/ant-classic-005/audit/fitting-inspection.json")  # The preserved original is read-only reference evidence.
    fitting = reader.episodes("runs/ant-classic-005/episodes/fit/baseline", fitting=True)  # Sensitivity never sees validation observations or selects another vector.
    diagnostic = reader.episodes("runs/ant-classic-005/episodes/diagnostic/baseline", fitting=True)  # These episodes explain baseline competence only.
    rows, _ = make_windows(fitting, fraction=.5)  # All 531 original windows are healthy and above the floor.
    fit_info = inspect_episode_group(fitting)  # Include failed and completed fitting resets.
    complete = [episode_summary(item) for item in fitting if int(item["length"]) == 1000]  # These ranges are explicitly conditional descriptive diagnostics.
    fit_info.update(completed_episode_yaw_rate_range=[min(row["turning"] for row in complete), max(row["turning"] for row in complete)], completed_episode_lateral_range=[min(row["lateral"] for row in complete), max(row["lateral"] for row in complete)])  # Do not replace the all-episode fitting summaries with these conditional ranges.
    errors = []  # Evaluate signed wrapped angular increments before taking the absolute discrepancy.
    for item in fitting:
        change = np.diff(yaw(item))  # Successor observations omit only the last transition of each episode.
        rate = np.arctan2(np.sin(change), np.cos(change)) / float(item["dt"])  # Preserve turn direction across the angle wrap boundary.
        errors.append(float(np.max(np.abs(rate - item["yaw_rate"][:-1]))))  # Compare physical rad/s quantities, not quaternion components.
    result = {"diagnostic": inspect_episode_group(diagnostic), "fit": fit_info, "windows": {"n": len(rows), "speed_quantiles": np.quantile([row["speed"] for row in rows], [0, .01, .25, .5, .75, .99, 1]).tolist(), "health_max": {key: max(row[key] for row in rows) for key in ("inversion", "bad_contact")}, "behaviors": {behavior: sensitivity(fitting, behavior) for behavior in ("lateral", "turning")}}, "yaw_metric_recomputed_max_abs_error": max(errors), "lateral_window_correlation_with_heading": float(np.corrcoef([row["lateral"] for row in rows], [row["heading"] for row in rows])[0, 1]), "lateral_turning_window_correlation": float(np.corrcoef([row["lateral"] for row in rows], [row["turning"] for row in rows])[0, 1])}  # Recreate every numerical field without copying the historical inspection's execution-scope assertions.
    checks.compare("ant_sensitivity", {key: value for key, value in saved.items() if key != "scope"}, result)  # Chronological scope is preserved in the original JSON, not invented by this later recomputation.
    logging.info("Recomputed Ant sensitivity: both behaviors, every exclusion, every leave-one-episode-out refit, 200/300-step windows")  # Completion is visible before potentially large JSON publication.
    return result  # These temporary diagnostic directions never enter a rollout or modify the source vectors.


def inspect_halfcheetah_sensitivity(episodes, rows, checks):
    """Local: reconstruct 001's decisive exclusions; global: make the rejected extraction auditable too."""
    filtered, stats = make_windows(episodes, fraction=.5)  # This removes the ten near-stationary windows using fitting data alone.
    by_seed = {int(item["seed"]): item for item in episodes}  # Original observations supply the accumulated torso-pitch evidence.
    removed = [{"seed": row["seed"], "start": row["start"], "speed": row["speed"], "mean_observed_pitch": float(by_seed[row["seed"]]["observations"][row["start"]:row["start"] + 100, 1].mean())} for row in rows if row["speed"] <= stats["minimum_speed_floor"]]  # HalfCheetah observations begin with height and pitch, excluding root X.
    checks.compare("hc001.sensitivity.removed_windows", 10, len(removed))  # This explicit count is the key published rejection claim.
    checks.compare("hc001.sensitivity.retained_windows", 507, len(filtered))  # Reproduce the altered fitting population exactly.
    rounded = {"speed": [8.1466, 1.1927, .2975, 2.5029], "height": [4.0843, 1.2446, .1134, .04629], "effort": [7.4358, .5072, -.5551, .03245]}  # Quoted values are the actual 001 Markdown table, not a new acceptance rule.
    stronger = {"speed": .225, "height": .185, "effort": -.475}  # The same narrative reports these three-decimal 10 m/s-floor cosines.
    results = {}  # Retain all sensitivity estimates rather than only rounded match flags.
    for behavior in ("speed", "height", "effort"):
        original, _ = contrast(rows, behavior); revised, detail = contrast(filtered, behavior)  # Every subset recomputes quartiles and, for effort, speed bins.
        first_removed, _ = contrast([row for row in rows if row["start"] != 100], behavior)  # This diagnoses startup influence without deleting arbitrary phases.
        floor_ten, _ = contrast([row for row in rows if row["speed"] > 10], behavior)  # Reproduce the separate stronger-floor sensitivity.
        values = [float(np.linalg.norm(original)), float(np.linalg.norm(revised)), cosine(original, revised), detail["contrast"]]  # Preserve full precision in the fresh output.
        reported = [round(value, 5 if index == 3 and behavior != "speed" else 4) for index, value in enumerate(values)]  # Compare at the exact precision used in the historical table.
        checks.compare(f"hc001.sensitivity.{behavior}.published_table", rounded[behavior], reported)  # A rounded prose claim is not incorrectly judged against an unrounded 1e-12 threshold.
        checks.compare(f"hc001.sensitivity.{behavior}.floor10_cosine", stronger[behavior], round(cosine(original, floor_ten), 3))  # Keep the separate three-decimal narrative claim traceable.
        checks.compare(f"hc001.sensitivity.{behavior}.first_window_cosine_gt_09995", True, cosine(original, first_removed) > .9995)  # Verify the reported startup-insensitivity inequality.
        results[behavior] = {"initial_raw_norm": values[0], "filtered_raw_norm": values[1], "initial_filtered_cosine": values[2], "filtered_contrast": values[3], "filtered_groups": detail, "exclude_first_window_cosine": cosine(original, first_removed), "floor_10_cosine": cosine(original, floor_ten)}  # Expose underlying group sizes and residual effort/speed differences.
    return {"floor": stats["minimum_speed_floor"], "removed_windows": removed, "retained_windows": len(filtered), "retained_sub10_speeds": [row["speed"] for row in filtered if row["speed"] < 10], "effort_speed_window_correlation": float(np.corrcoef([row["speed"] for row in rows], [row["effort"] for row in rows])[0, 1]), "behaviors": results, "scope": "Reproduces the decisive sensitivity table, specified exclusions and effort/speed correlation; does not re-establish visual gait diagnoses or original invocation chronology."}  # Explicit scope prevents overclaiming a complete historical diagnostic recreation.


def audit_denominators(reader, checks):
    """Local: show available-duration and equal-episode estimands; global: support corrected quality/travel wording."""
    output = {}  # These are descriptive recalculations of existing confirmation data, not replacement effect estimates.
    for arm, condition in (("baseline", "baseline"), ("treated", "lateral__-0.1000")):
        episodes = reader.episodes(f"runs/ant-classic-005/episodes/confirmation/{condition}")  # Reuse all thirty saved pairs without dropping the shared prefix-only episode.
        steps = sum(max(int(item["length"]) - 100, 0) for item in episodes)  # Count only actually observed post-onset transitions.
        inverted = sum(float(item["inversion"][100:].sum()) for item in episodes)  # A timestep-pooled numerator is distinct from the episode-average fraction.
        summaries = [episode_summary(item) for item in episodes]  # Preserve the historical equal-reset estimator alongside the denominator comparison.
        output[arm] = {"post_onset_steps": steps, "inverted_steps": inverted, "timestep_pooled_inversion": inverted / steps, "episode_mean_inversion": float(np.mean([row["inversion"] for row in summaries])), "mean_forward_speed": float(np.mean([row["speed"] for row in summaries])), "mean_saved_post_onset_forward_distance": float(np.mean([item["x_velocity"][100:].sum() * float(item["dt"]) for item in episodes])), "early_endings": [{"seed": int(item["seed"]), "length": int(item["length"])} for item in episodes if int(item["length"]) < 1000]}  # Distances integrate saved velocity over observed time rather than inventing motion after termination.
    output["mean_forward_speed_retention"] = output["treated"]["mean_forward_speed"] / output["baseline"]["mean_forward_speed"]  # This ratio is the one used in the original gate.
    output["saved_forward_distance_retention"] = output["treated"]["mean_saved_post_onset_forward_distance"] / output["baseline"]["mean_saved_post_onset_forward_distance"]  # Travel can differ even when mean available-duration velocities are similar.
    checks.compare("ant_denominators.treated_inverted_steps", 20, output["treated"]["inverted_steps"])  # Check the exact source of the published terminology correction.
    checks.compare("ant_denominators.treated_post_onset_steps", 24725, output["treated"]["post_onset_steps"])  # Keep its denominator reproducible.
    return output  # These values qualify existing claims without changing protocol acceptance.

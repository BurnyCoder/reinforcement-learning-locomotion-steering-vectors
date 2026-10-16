"""Offline scientific-rigor audit: do the recorded experiments support their interpretations?

Run from the project root: uv run --no-sync python scripts/audit_scientific_rigor.py
Reads saved results, vectors and episodes only. No policy weights, simulator, new seeds,
selection changes or gate changes. Outputs go to reports/scientific-rigor-audit/.
Interpretation: reports/scientific-rigor-audit.md. Source: https://docs.python.org/3.12/library/logging.html
"""
import hashlib  # Re-verify that every consumed input file is unchanged after the audit.
import json  # Write strict, non-executable evidence.
import logging  # Timestamped terminal and file records without truncation.
import sys  # Record the literal command and Python version.
from datetime import datetime, timezone  # Distinguish this audit's time from the original research chronology.
from pathlib import Path  # Project-relative inputs and outputs.
import numpy as np  # Array reductions and NumPy-scalar JSON encoding.
from documentation_checks import Comparisons, EvidenceReader  # Reuse the hashing reader and tolerance comparator of the documentation audit.
from rigor_statistics import mean_heading, odd_even, percentile_interval, post_onset_displacement, shuffled_alignment_null, sign_flip_pvalue  # Tested inference helpers.
from rl_locomotion_steering_vectors.analysis import _summary, paired_effect, window_table  # Production estimators, reused rather than copied.
from rl_locomotion_steering_vectors.experiment import condition_name  # Production directory naming for saved conditions.
from rl_locomotion_steering_vectors.runtime import summarise_episode, wrapped_angle_difference  # Production episode summaries and circular differences.
from rl_locomotion_steering_vectors.storage import code_identity, start_logging  # Provenance and the project's logging convention.

EVALUATED = ("hc-running-002", "hc-height-speed-003", "hc-height-speed-004", "ant-classic-005")  # 001 was rejected before any evaluation.
BOOTSTRAP_SEEDS = {"validation": 710, "confirmation": 720}  # Seeds used by the original pipeline (experiment.py).
FAMILIES = {"ant-classic-005": ("lateral", "turning"), "hc-running-002": ("speed", "effort", "height")}  # Runs that fitted their own vectors; 003/004 import the 002 height family.
ANT, PRIMARY = "ant-classic-005", "lateral__-0.1000"  # Featured locked condition.


def json_text(value):
    """Local: strict JSON with NumPy support; global: reject NaN evidence."""
    return json.dumps(value, indent=2, allow_nan=False, default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()) + "\n"  # Same encoding convention as the documentation audit.


def summaries(reader, run, phase, condition):
    """Local: production summaries of one saved arm; global: identical reductions to the recorded results."""
    return [summarise_episode(episode, 100) for episode in reader.episodes(f"runs/{run}/episodes/{phase}/{condition}")]  # All evaluated runs used warmup 100 (manifest config).


def production_recomputation(reader, checks):
    """Local: recompute every saved condition with production code; global: confirm this audit reads the recorded evidence."""
    counts = {}  # Conditions per run and phase.
    for run in EVALUATED:
        results = reader.json(f"runs/{run}/results.json")  # Recorded effects, intervals and gates.
        for phase, seed in BOOTSTRAP_SEEDS.items():
            rows = results.get(phase) or []  # Unrun phases are empty.
            if not rows:
                continue  # Nothing recorded, nothing to recompute.
            baseline = summaries(reader, run, phase, "baseline")  # Shared paired baseline arm.
            for row in rows:
                effect = paired_effect(baseline, summaries(reader, run, phase, condition_name(row["vector"], row["alpha"])), row["behavior"], seed=seed)  # Exact original call.
                checks.compare(f"{run}.{phase}.{row['vector']}.{row['alpha']}", {key: row["effect"][key] for key in ("effect", "ci95", "gate", "quality_pass")}, effect)  # Leaf comparison at 1e-12.
            counts[f"{run}.{phase}"] = len(rows)  # Coverage record.
    return counts  # 332 conditions in total.


def load_ant_confirmation(reader):
    """Local: complete Ant confirmation arms with observations; global: share one read across robustness phases."""
    results = reader.json(f"runs/{ANT}/results.json")  # Locked conditions in recorded order.
    names = ["baseline"] + [condition_name(row["vector"], row["alpha"]) for row in results["confirmation"]]  # Baseline plus six locked conditions.
    return {name: reader.episodes(f"runs/{ANT}/episodes/confirmation/{name}", fitting=True) for name in names}  # fitting=True keeps observations for heading.


def ant_robustness(arms, reader):
    """Local: alternative estimators and estimands; global: test whether the headline depends on analysis conventions."""
    base = arms["baseline"]; base_rows = [_summary(item) for item in base]  # Production per-episode summaries, including failures.
    validation = {row["vector"]: row["effect"]["effect"] for row in reader.json(f"runs/{ANT}/results.json")["validation"] if row["vector"] == "lateral" and row["alpha"] == -0.1}  # Selected condition's validation estimate.
    output = {"validation_effect_of_selected_condition": validation["lateral"]}  # Allows a winner's-curse shrinkage comparison.
    for name, treated in arms.items():
        if name == "baseline":
            continue  # Baseline is the comparison arm.
        if [int(item["seed"]) for item in base] != [int(item["seed"]) for item in treated]:
            raise ValueError(f"{name}: arms are not aligned by reset seed")  # Positional pairing below requires identical seed order.
        rows = [_summary(item) for item in treated]  # Same reductions for the treated arm.
        effect = paired_effect(base, treated, "lateral", seed=720)  # Recorded estimator, reproduced exactly.
        differences = np.array([after["lateral"] - before["lateral"] for before, after in zip(base_rows, rows)])  # Per-reset paired differences (seed-sorted files).
        usable = np.array([not (before["prefix_only"] or after["prefix_only"]) for before, after in zip(base_rows, rows)])  # Pairs with post-onset data in both arms.
        clean = np.array([before["failure"] == 0 and after["failure"] == 0 for before, after in zip(base_rows, rows)])  # Neither arm failed: conditions on a post-treatment outcome.
        displacement = np.array([post_onset_displacement(b, "y_velocity") - post_onset_displacement(a, "y_velocity") for a, b in zip(base, treated)])  # Treated minus baseline world-Y travel.
        headings = [(mean_heading(b["observations"], b["length"]), mean_heading(a["observations"], a["length"])) for a, b in zip(base, treated)]  # (treated, baseline) circular means.
        heading = [wrapped_angle_difference(t, s) for t, s in headings if t is not None and s is not None]  # Wrapped treated-minus-baseline heading.
        output[name] = {
            "recorded_effect": effect["effect"], "recorded_ci95": effect["ci95"], "recorded_gate": effect["gate"],
            "sign_flip_pvalue": sign_flip_pvalue(differences, seed=0), "sign_flip_resamples": 200000,
            "positive_pairs": int((differences > 0).sum()), "zero_pairs": int((differences == 0).sum()), "n_pairs": len(differences), "median_difference": float(np.median(differences)),
            "effect_excluding_prefix_only_pairs": float(differences[usable].mean()), "n_pairs_with_post_onset_data": int(usable.sum()),
            "effect_on_pairs_without_failure_post_treatment_conditioned": float(differences[clean].mean()), "n_pairs_without_failure": int(clean.sum()),
            "world_y_displacement_difference_m": float(displacement.mean()), "world_y_displacement_ci95_m": percentile_interval(displacement, seed=720),
            "x_distance_retention": float(sum(post_onset_displacement(b, "x_velocity") for b in treated) / sum(post_onset_displacement(a, "x_velocity") for a in base)),
            "forward_speed_retention": effect["treated_means"]["speed"] / effect["baseline_means"]["speed"],
            "mean_heading_change_rad": float(np.mean(heading)), "mean_heading_change_deg": float(np.degrees(np.mean(heading))), "n_heading_pairs": len(heading),
            "effort_ratio": effect["treated_means"]["effort"] / effect["baseline_means"]["effort"],
            "failures": {"baseline": [int(item["seed"]) for item, row in zip(base, base_rows) if row["failure"]], "treated": [int(item["seed"]) for item, row in zip(treated, rows) if row["failure"]]},
        }  # Every number is descriptive and post hoc; none changes selection or the recorded status.
    return output  # Keyed by saved condition directory name.


def paired_control_contrasts(arms):
    """Local: primary minus each control on shared seeds; global: quantify specificity against the tested controls."""
    return {name: {key: value for key, value in paired_effect(arm, arms[PRIMARY], "lateral", seed=720).items() if key in ("effect", "ci95", "n_pairs")}
            for name, arm in arms.items() if name not in ("baseline", PRIMARY)}  # Treated arms share each seed's unsteered onset state, so paired_effect pairs them validly.


def control_alignment(reader):
    """Local: geometry of saved controls; global: test whether the shuffled control is a label-free null."""
    output = {}  # Per run and behavior.
    for run, behaviors in FAMILIES.items():
        config = reader.json(f"runs/{run}/manifest.json")["config"]  # Actual fitting settings.
        vectors = reader.arrays(f"runs/{run}/vectors.npz")  # Saved primary and control directions.
        table = window_table(reader.episodes(f"runs/{run}/episodes/fit/baseline", fitting=True), warmup=config["warmup"], minimum_speed_fraction=config["fit_min_speed_fraction"])  # Production fitting table.
        eligible = {key: value[table["eligible"]] for key, value in table.items() if isinstance(value, np.ndarray)}  # Same eligible rows as fit_vectors.
        eigenvalues = np.linalg.eigvalsh(np.cov(eligible["activations"].T))[::-1]  # Spectrum of eligible window-mean activations.
        output[run] = {"window_mean_participation_ratio": float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()), "eligible_windows": int(len(eligible["episode"])), "behaviors": {}}  # Effective dimension of the population shuffled contrasts sample.
        for behavior in behaviors:
            unit = vectors[behavior] / np.linalg.norm(vectors[behavior])  # Primary direction.
            null = shuffled_alignment_null(eligible, behavior, unit, draws=300, seed=2026)  # Fresh reshuffles under the production rules.
            saved = {name.split("_", 1)[1]: float(vectors[name] @ unit / np.linalg.norm(vectors[name])) for name in vectors if name.startswith(behavior + "_")}  # Saved controls' cosines.
            output[run]["behaviors"][behavior] = {"saved_control_cosines": saved, "reshuffle_null": {"draws": len(null), "mean_abs_cosine": float(np.abs(null).mean()), "sd_cosine": float(null.std()),
                                                  "fraction_abs_at_least_saved": float(np.mean(np.abs(null) >= abs(saved["shuffled"])))},
                                                  "isotropic_cosine_sd_256d": 1 / np.sqrt(256)}  # Random 256-d directions have cosine SD ~ 1/sqrt(d).
    for run in ("hc-height-speed-003", "hc-height-speed-004"):
        imported, source = reader.arrays(f"runs/{run}/vectors.npz"), reader.arrays("runs/hc-running-002/vectors.npz")  # 003/004 relabel the 002 height family as speed.
        output[run] = {"imports_002_height_family_bytes_equal": all(np.array_equal(imported[name], source[name.replace("speed", "height", 1)]) for name in imported)}  # So 002 height cosines apply.
    return output  # Interpreted in the report.


def odd_even_responses(reader):
    """Local: decompose +/-alpha validation effects; global: separate steerable responses from generic degradation."""
    output = {}  # Per run, vector and |alpha|.
    for run in EVALUATED:
        results = reader.json(f"runs/{run}/results.json")  # Effects verified by production_recomputation.
        rows = {(row["vector"], row["alpha"]): row["effect"] for row in results["validation"]}  # Signed effects by condition.
        relative = lambda effect: effect["treated_means"]["speed"] / effect["baseline_means"]["speed"] - 1  # Relative forward-speed change.
        run_output = {}  # Vector-level decomposition.
        for vector in sorted({name for name, _ in rows}):
            entries = []  # One entry per tested magnitude with both signs.
            for magnitude in sorted({abs(alpha) for name, alpha in rows if name == vector}):
                plus, minus = rows.get((vector, magnitude)), rows.get((vector, -magnitude))  # The two signed conditions.
                if plus and minus:
                    target, speed = odd_even(plus["effect"], minus["effect"]), odd_even(relative(plus), relative(minus))  # Target behavior units and relative speed.
                    entries.append({"abs_alpha": magnitude, "target_odd": target[0], "target_even": target[1], "relative_speed_odd": speed[0], "relative_speed_even": speed[1]})  # Exact identification of both parts.
            run_output[vector] = entries  # Empty when a vector lacks symmetric strengths.
        changes = [(relative(row["effect"]), phase, row["vector"], row["alpha"]) for phase in ("validation", "confirmation") for row in results.get(phase) or []]  # Every evaluated condition.
        output[run] = {"vectors": run_output, "largest_relative_speed_change": max(changes)[0], "largest_relative_speed_change_condition": max(changes)[1:]}  # Did anything ever speed the policy up?
    return output  # Interpreted in the report.


def main():
    """Local: run named phases and save evidence; global: an auditable inference check of experiments 002-005."""
    root = Path.cwd(); destination = root / "reports" / "scientific-rigor-audit"  # Fixed project-relative output location.
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run from the project root with the saved runs/ directories restored")  # Avoid a vacuous audit elsewhere.
    start_logging(destination)  # Timestamped terminal and file handlers.
    started, reader, checks, command = datetime.now(timezone.utc).isoformat(), EvidenceReader(root), Comparisons(), [sys.executable, *sys.argv]  # Audit context.
    logging.info("Scientific-rigor audit command: %s", command)  # Complete invocation in the log.
    outputs, failures = {}, []  # Phase results and explicit phase errors.
    phases = {"production_recomputation": lambda: production_recomputation(reader, checks), "control_alignment": lambda: control_alignment(reader), "odd_even_responses": lambda: odd_even_responses(reader)}  # Independent read-only phases.
    for name, phase in phases.items():
        try:
            outputs[name] = phase(); logging.info("Phase %s complete", name)  # Progress after each phase.
        except Exception as error:
            failures.append({"phase": name, "error": repr(error)}); logging.exception("Phase %s failed", name)  # Keep failures visible.
    try:
        arms = load_ant_confirmation(reader)  # One shared read of the Ant confirmation arms.
        outputs["ant_robustness"], outputs["paired_control_contrasts"] = ant_robustness(arms, reader), paired_control_contrasts(arms)  # Robustness and specificity phases.
    except Exception as error:
        failures.append({"phase": "ant_robustness", "error": repr(error)}); logging.exception("Ant phases failed")  # Keep failures visible.
    for name, digest in reader.identities.items():
        checks.compare(f"scientific_inputs_unchanged.{name}", digest, hashlib.sha256((root / name).read_bytes()).hexdigest())  # Inputs must be byte-identical afterwards.
    count = sum(outputs.get("production_recomputation", {}).values())  # Coverage of recomputed conditions.
    checks.compare("coverage.condition_count", 332, count)  # All saved validation/confirmation conditions of 002-005.
    identity = code_identity(root); identity["audit_source_sha256"] = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in ("audit_scientific_rigor.py", "rigor_statistics.py", "documentation_checks.py")}  # Exact audit code.
    record = {"created_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "command": command, "python": sys.version, "numpy": np.__version__, "source_identity": identity,
              "passed": not checks.mismatches and not failures, "condition_count": count, "checked_leaf_fields": checks.count, "max_numeric_error": checks.max_numeric_error, "mismatches": checks.mismatches, "phase_errors": failures,
              "calculations": outputs, "scope": "Saved-data inference audit. Post hoc descriptive analyses; no policy inference, rollouts, new seeds, selection or gate changes. Recorded statuses are unchanged."}  # What this audit does and does not establish.
    for name, value in {"rigor-audit.json": record, "input-identities.json": reader.identities}.items():
        text = json_text(value); (destination / name).write_text(text, encoding="utf-8")  # Only this audit's own outputs are written.
        logging.info("Saved %s\n%s", destination / name, text)  # Complete evidence in terminal and log.
    logging.info("Audit finished: passed=%s; conditions=%d; compared fields=%d; mismatches=%d; failed phases=%d", record["passed"], count, checks.count, len(checks.mismatches), len(failures))  # Summary line.
    return 0 if record["passed"] else 1  # Nonzero exit on any mismatch or failed phase.


if __name__ == "__main__":
    raise SystemExit(main())  # Importing the module never runs the audit.

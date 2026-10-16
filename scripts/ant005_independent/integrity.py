"""Local: verify the saved Ant 005 evidence chain. Global: rule out implementation, leakage and pairing errors before interpreting effects.

Sources: Baukit Trace https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py ;
PyTorch forward hooks https://pytorch.org/docs/2.5/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook ;
SB3 predict https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/sac.html
"""
from __future__ import annotations  # Keep annotations lightweight.

import hashlib  # Runtime source fingerprint comparison.
import logging  # Timestamped progress.
import re  # Parse timestamps from the original logs.
from datetime import datetime, timezone  # Compare file and log chronology in UTC.

import numpy as np  # Array comparisons.

from rl_locomotion_steering_vectors.analysis import fit_vectors, window_table  # Production extraction, rerun on saved fitting data.
from rl_locomotion_steering_vectors.perception import first_relu  # Offline first-ReLU evaluation.
from rl_locomotion_steering_vectors.runtime import ActivationIntervention, actor_layer, parameter_hash, run_episode  # Production runtime as the oracle.
from rl_locomotion_steering_vectors.storage import load_arrays, load_json  # Safe artifact readers.

from .common import FRESH_SEEDS, ROOT, SOURCE_RUN, WARMUP, condition_dir, load_condition, sha256_vector  # Shared protocol.


def _seed_splits() -> dict:
    """Local: check that all recorded and audit splits are disjoint. Global: exclude seed leakage between phases."""
    splits = dict(load_json(SOURCE_RUN / "manifest.json")["seed_splits"], audit=list(FRESH_SEEDS))  # Recorded splits plus the audit's fresh seeds.
    seeds = [seed for group in splits.values() for seed in group]  # Flatten.
    return {"disjoint": len(seeds) == len(set(seeds)), "ranges": {name: [min(group), max(group), len(group)] for name, group in splits.items()}}  # Evidence.


def _independent_lateral(fit: list[dict], minimum_speed_fraction: float = 0.5) -> np.ndarray:
    """Local: re-derive the lateral contrast from the documented rule. Global: check the extraction without trusting analysis.py internals.

    Rule (docs/methodology.md): non-overlapping 100-step windows after a 100-step warmup; drop windows with
    >5% inversion or torso contact and windows at or below half the median healthy forward speed; label each
    window by mean world-Y velocity; high/low = top/bottom quartile; average windows within episode, then
    episodes; scale the difference to the centered first-ReLU timestep RMS of eligible windows.
    """
    rows = []  # (seed, start, speed, label, healthy, mean activation)
    for episode in fit:  # One reset at a time.
        length = int(episode["length"])  # Recorded steps.
        for start in range(WARMUP, length - 99, 100):  # Full windows only.
            part = slice(start, start + 100)  # Window span.
            healthy = max(np.mean(episode["inversion"][part]), np.mean(episode["bad_contact"][part])) <= 0.05  # Health rule.
            rows.append((int(episode["seed"]), start, float(np.mean(episode["x_velocity"][part])), float(np.mean(episode["y_velocity"][part])), healthy,
                         episode["activations"][part].astype(np.float64).mean(axis=0)))  # Window record.
    speeds = np.array([row[2] for row in rows])  # Window forward speeds.
    healthy = np.array([row[4] for row in rows])  # Health mask.
    eligible = healthy & (speeds > minimum_speed_fraction * np.median(speeds[healthy]))  # Speed floor.
    labels = np.array([row[3] for row in rows])[eligible]  # Eligible world-Y labels.
    seeds = np.array([row[0] for row in rows])[eligible]  # Eligible window seeds.
    means = np.stack([row[5] for row in rows])[eligible]  # Eligible window activation means.
    low_cut, high_cut = np.quantile(labels, [0.25, 0.75])  # Quartile cut points.
    balanced = lambda mask: np.mean([means[mask & (seeds == s)].mean(axis=0) for s in np.unique(seeds[mask])], axis=0)  # Episode-balanced mean.
    direction = balanced(labels >= high_cut) - balanced(labels <= low_cut)  # High minus low.
    moments, energies = [], []  # Per-episode first and second moments of eligible timesteps.
    for episode in fit:  # Same eligible windows.
        starts = [row[1] for row, keep in zip(rows, eligible) if keep and row[0] == int(episode["seed"])]  # Eligible starts for this reset.
        if starts:  # Episodes without eligible windows do not define the scale.
            hidden = np.concatenate([episode["activations"][s:s + 100] for s in starts]).astype(np.float64)  # Timestep activations.
            moments.append(hidden.mean(axis=0))  # Episode mean.
            energies.append(float(np.square(hidden).sum(axis=1).mean()))  # Episode E||h||².
    scale = np.sqrt(np.mean(energies) - np.square(np.mean(moments, axis=0)).sum())  # Centered RMS about the episode-balanced mean.
    return direction * scale / np.linalg.norm(direction)  # Scaled steering vector.


def _vectors(fit: list[dict]) -> dict:
    """Local: refit vectors from saved fitting episodes. Global: confirm the evaluated arrays came only from the fitting split."""
    saved = load_arrays(SOURCE_RUN / "vectors.npz")  # Evaluated vectors.
    refit, _ = fit_vectors(fit, ["lateral", "turning"], seed=0, warmup=WARMUP, minimum_speed_fraction=0.5)  # Production rule, fitting data only.
    independent = _independent_lateral(fit)  # Reimplementation from the written rule.
    stats = window_table(fit, warmup=WARMUP, minimum_speed_fraction=0.5)["stats"]  # Window bookkeeping for the report.
    lateral = saved["lateral"].astype(np.float64)  # Primary vector.
    return {"bitwise_equal": {name: bool(name in refit and refit[name].dtype == saved[name].dtype and np.array_equal(refit[name], saved[name])) for name in saved},
            "independent_lateral_max_abs_diff": float(np.abs(independent - lateral).max()),
            "independent_lateral_cosine": float(independent @ lateral / np.linalg.norm(independent) / np.linalg.norm(lateral)),
            "lateral_norm": float(np.linalg.norm(lateral)), "window_stats": stats}  # Evidence.


def _activations(model, fit: list[dict]) -> dict:
    """Local: compare saved activations with offline evaluation. Global: confirm vectors live in actor.latent_pi[1] space."""
    differences = [float(np.abs(first_relu(model, episode["observations"]) - episode["activations"]).max()) for episode in fit[:4]]  # Four resets.
    return {"episodes": 4, "max_abs_diff": max(differences)}  # Float32 agreement.


def _identities(model) -> dict:
    """Local: check confirmation condition identity files. Global: the evaluated arrays and strengths are the selected ones."""
    selection = load_json(SOURCE_RUN / "selection.json")["conditions"]  # Locked conditions.
    vectors, biases = load_arrays(SOURCE_RUN / "vectors.npz"), load_arrays(SOURCE_RUN / "action_biases.npz")  # Treatments.
    runtime_hash = hashlib.sha256((ROOT / "src" / "rl_locomotion_steering_vectors" / "runtime.py").read_text(encoding="utf-8").encode("utf-8")).hexdigest()  # Current runtime source.
    policy = parameter_hash(model)  # Current pinned checkpoint state.
    rows = []  # One row per locked condition.
    for item in selection:  # Each confirmation arm.
        identity = load_json(condition_dir(SOURCE_RUN, "confirmation", f"{item['vector']}__{item['alpha']:+.4f}") / "identity.json")  # Saved identity.
        bias = biases.get(item["vector"])  # Action-space comparator, if any.
        rows.append({"vector": item["vector"], "alpha": item["alpha"], "alpha_matches": identity["alpha"] == item["alpha"],
                     "treatment_hash_matches": identity["bias_sha256"] == sha256_vector(item["alpha"] * bias) if bias is not None else identity["vector_sha256"] == sha256_vector(vectors[item["vector"]]),  # Bias arms store alpha × bias (experiment.evaluate).
                     "policy_matches": identity["policy_sha256"] == policy, "runtime_matches_current": identity["runtime_sha256"] == runtime_hash,
                     "deterministic": not identity["stochastic"]})  # Evidence per arm.
    return {"conditions": rows, "all_match": all(row["alpha_matches"] and row["treatment_hash_matches"] and row["policy_matches"] for row in rows)}  # Summary.


def _prefixes() -> dict:
    """Local: compare every treated arm with its baseline. Global: verify matched onset states and steering starting at step 100."""
    report = {}  # phase → counts.
    for phase in ("validation", "confirmation"):  # Both evaluation phases.
        baseline = load_condition(SOURCE_RUN, phase, "baseline", ("actions",))  # Reference arm.
        counts = {"pairs": 0, "onset_equal": 0, "prefix_actions_equal": 0, "first_divergence_at_onset": 0, "prefix_only": 0}  # Tallies.
        for folder in sorted(p for p in (SOURCE_RUN / "episodes" / phase).iterdir() if p.name != "baseline"):  # Every treated condition.
            for seed, episode in load_condition(SOURCE_RUN, phase, folder.name, ("actions",)).items():  # Every reset.
                base = baseline[seed]  # Matched baseline.
                counts["pairs"] += 1  # Count pair.
                counts["onset_equal"] += str(episode["onset_hash"]) == str(base["onset_hash"])  # Same simulator state at onset.
                end = min(WARMUP, len(episode["actions"]), len(base["actions"]))  # Shared prefix length.
                counts["prefix_actions_equal"] += bool(np.array_equal(episode["actions"][:end], base["actions"][:end]))  # Identical unsteered controls.
                if min(len(episode["actions"]), len(base["actions"])) <= WARMUP:  # Episode ended before onset.
                    counts["prefix_only"] += 1  # No steered step exists.
                    continue  # Divergence undefined.
                counts["first_divergence_at_onset"] += not np.array_equal(episode["actions"][WARMUP], base["actions"][WARMUP])  # Steering acts at step 100.
        report[phase] = counts  # Phase summary.
    return report  # Evidence.


def _actions(model) -> dict:
    """Local: rebuild treated actions from saved observations. Global: the saved arm is exactly first-ReLU addition plus ordinary SB3 inference."""
    vector = load_arrays(SOURCE_RUN / "vectors.npz")["lateral"]  # Primary treatment.
    layers = model.actor.latent_pi  # Actor trunk.
    calls = []  # Forward-hook call log.
    handle = layers[1].register_forward_hook(lambda module, inputs, output: calls.append(1))  # Count edits of the steered layer.
    with ActivationIntervention(actor_layer(model), vector) as hook:  # Production editing path.
        hook.alpha = -0.1  # Locked strength.
        model.predict(np.zeros((1, 105), dtype=np.float32), deterministic=True)  # One forward pass.
    handle.remove()  # Restore the module.
    worst = 0.0  # Largest action discrepancy.
    episodes = load_condition(SOURCE_RUN, "confirmation", "lateral__-0.1000", ("observations", "actions"))  # Saved treated arm.
    for seed, episode in episodes.items():  # Every confirmation reset.
        if int(episode["length"]) <= WARMUP:  # No steered steps.
            continue  # Nothing to reconstruct.
        with ActivationIntervention(actor_layer(model), vector) as hook:  # Fresh hook per batch.
            prefix, _ = model.predict(episode["observations"][:WARMUP], deterministic=True)  # Unsteered steps.
            hook.alpha = -0.1  # Steered steps.
            active, _ = model.predict(episode["observations"][WARMUP:], deterministic=True)  # Steered steps.
        worst = max(worst, float(np.abs(np.concatenate([prefix, active]) - episode["actions"]).max()))  # Batch vs per-step float32 agreement.
    return {"hook_calls_per_forward": len(calls), "relu_modules_distinct": layers[1] is not layers[3], "episodes": len(episodes), "max_abs_action_diff": worst}  # Evidence.


def _resimulate(model) -> dict:
    """Local: rerun three confirmation pairs in this process. Global: the saved trajectories are what the code produces."""
    vector = load_arrays(SOURCE_RUN / "vectors.npz")["lateral"]  # Primary treatment.
    rows = []  # One row per rerun episode.
    for condition, alpha in (("baseline", 0.0), ("lateral__-0.1000", -0.1)):  # Both arms.
        for seed in (520000, 520010, 520025):  # Normal, treated-failure and prefix-only resets.
            saved = load_arrays(condition_dir(SOURCE_RUN, "confirmation", condition) / f"{seed}.npz")  # Original arrays.
            fresh = run_episode(model, "ant", seed, vector=None if alpha == 0 else vector, alpha=alpha, warmup=WARMUP, max_steps=1000)  # Rerun.
            numeric = [key for key in saved if saved[key].dtype.kind in "fiub" and saved[key].shape == fresh[key].shape]  # Comparable numeric arrays.
            rows.append({"condition": condition, "seed": seed, "keys_equal": sorted(saved) == sorted(fresh),
                         "shapes_equal": all(saved[key].shape == fresh[key].shape for key in saved),
                         "max_abs_diff": max(float(np.abs(saved[key].astype(float) - fresh[key].astype(float)).max(initial=0.0)) for key in numeric),
                         "onset_hash_equal": str(saved["onset_hash"]) == str(fresh["onset_hash"])})  # Comparison.
    return {"episodes": rows, "all_identical": all(row["keys_equal"] and row["shapes_equal"] and row["max_abs_diff"] == 0.0 and row["onset_hash_equal"] for row in rows)}  # Summary.


def _chronology() -> dict:
    """Local: order fitting, validation, selection and confirmation. Global: confirmation data could not inform selection."""
    stamps = {}  # phase → (first, last) log timestamps.
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})Z INFO episode phase=(\w+)")  # Log line prefix.
    for log in sorted(SOURCE_RUN.glob("*.log")):  # Every original invocation.
        for line in log.read_text(encoding="utf-8").splitlines():  # Line by line.
            match = pattern.match(line)  # Episode completion record.
            if match:
                when = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)  # UTC timestamp.
                first, last = stamps.get(match.group(2), (when, when))  # Running range.
                stamps[match.group(2)] = (min(first, when), max(last, when))  # Update.
    selected = datetime.fromisoformat(load_json(SOURCE_RUN / "selection.json")["created_utc"])  # Selection lock time.
    mtimes = lambda phase: [datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) for p in (SOURCE_RUN / "episodes" / phase).rglob("*.npz")]  # File times.
    order_ok = stamps["fit"][1] < stamps["validation"][0] and stamps["validation"][1] < selected < stamps["confirmation"][0]  # Required order.
    return {"log_ranges": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in stamps.items()}, "selection_created": selected.isoformat(),
            "first_confirmation_file": min(mtimes("confirmation")).isoformat(), "last_validation_file": max(mtimes("validation")).isoformat(),
            "order_ok": bool(order_ok and max(mtimes("validation")) < min(mtimes("confirmation")))}  # Evidence.


def verify_integrity(model) -> dict:
    """Local: run every saved-data integrity check. Global: establish that the evidence chain is sound before interpreting it."""
    fit = list(load_condition(SOURCE_RUN, "fit", "baseline").values())  # Fitting episodes with activations and observations.
    checks = {"seed_splits": _seed_splits(), "vectors": _vectors(fit), "activations": _activations(model, fit), "identities": _identities(model),
              "prefixes": _prefixes(), "actions": _actions(model), "resimulation": _resimulate(model), "chronology": _chronology()}  # All checks.
    logging.info("integrity %s", checks)  # Full result in the log.
    return checks  # For audit.json.

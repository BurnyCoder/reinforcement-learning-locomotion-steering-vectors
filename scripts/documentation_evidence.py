"""Independent saved-array arithmetic for the documentation audit; never simulate or select.

Local: NumPy reductions reconstruct the published estimators and fitting diagnostics.
Global: a separate oracle checks production results without importing their calculations.
Sources: https://numpy.org/doc/1.26/reference/generated/numpy.mean.html
https://numpy.org/doc/1.26/reference/generated/numpy.quantile.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
"""
import numpy as np  # Reuse documented array operations instead of implementing numerical primitives.

METRICS = {"speed": "x_velocity", "effort": "effort", "height": "height", "lateral": "y_velocity", "turning": "yaw_rate", "planar_speed": "planar_speed", "inversion": "inversion", "bad_contact": "bad_contact", "reward": "reward", "saturation": "saturation"}  # Map report labels to actual saved simulator fields.


def balanced(rows, field):
    """Local: average within reset, then across resets; global: reconstruct equal episode weights."""
    return np.mean([np.mean([row[field] for row in rows if row["seed"] == seed], axis=0) for seed in sorted({row["seed"] for row in rows})], axis=0)  # Each contributing reset supplies exactly one mean.


def quartiles(rows, behavior):
    """Local: explicitly use NumPy linear quartiles; global: keep selection boundaries reproducible."""
    cuts = np.quantile([row[behavior] for row in rows], [.25, .75], method="linear")  # The original extraction used this interpolation convention.
    return [row for row in rows if row[behavior] <= cuts[0]], [row for row in rows if row[behavior] >= cuts[1]], cuts  # Return the actual groups for contributor and temporal diagnostics.


def contrast(rows, behavior):
    """Local: reconstruct classic differences, matching speed for effort; global: audit saved directions."""
    speeds = np.array([row["speed"] for row in rows])  # Strata depend only on fitting-window speed.
    unique = np.unique(speeds)  # Repeated exact speeds must not be split by redundant boundaries.
    bins = np.searchsorted(unique, speeds) if len(unique) <= 5 else np.searchsorted(np.unique(np.quantile(speeds, [.2, .4, .6, .8])), speeds, side="right")  # Match the documented exact-speed/quantile-bin alternatives.
    bins = bins if behavior == "effort" else np.zeros(len(rows), int)  # Other behaviors retain their original unconditioned contrasts.
    differences, gaps, details, all_low, all_high = [], [], [], [], []  # Keep both numerical estimates and contributor evidence.
    for index in np.unique(bins):
        subset = [row for row, group in zip(rows, bins) if group == index]  # No labels cross a speed stratum.
        low, high, cuts = quartiles(subset, behavior)  # Recompute thresholds for this particular subset.
        if cuts[1] - cuts[0] <= 1e-8 * max(1., np.mean([abs(row[behavior]) for row in subset])):
            continue  # Tied labels do not define the original estimator's direction.
        differences.append(balanced(high, "h") - balanced(low, "h"))  # This is an activation difference, never a fitted action predictor.
        gaps.append(float(balanced(high, behavior) - balanced(low, behavior)))  # Preserve the measured behavioral contrast before scaling.
        details.append({"bin": int(index), "low_cut": float(cuts[0]), "high_cut": float(cuts[1]), "low_windows": len(low), "high_windows": len(high), "low_speed": float(balanced(low, "speed")), "high_speed": float(balanced(high, "speed"))})  # Expose residual speed matching error.
        all_low.extend(low); all_high.extend(high)  # Count the union of contributing episodes across strata below.
    if not differences:
        raise ValueError("No nonnegligible behavior contrast in audit subset")  # Undefined sensitivity estimates must remain visible failures.
    detail = {"low_episodes": len({row["seed"] for row in all_low}), "high_episodes": len({row["seed"] for row in all_high}), "low_windows": len(all_low), "high_windows": len(all_high), "bins": details, "contrast": float(np.mean(gaps))}  # Counts match the original equal-stratum estimator.
    return np.mean(differences, axis=0), detail  # Equal bin weighting avoids composition-weighting a speed stratum.


def yaw(episode):
    """Local: decode Ant's scalar-first quaternion; global: verify heading association and yaw-rate units.

    Source: https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py
    """
    q = np.asarray(episode["observations"][:, 1:5], dtype=float).copy()  # Ant excludes root X/Y and stores height before the quaternion.
    q /= np.linalg.norm(q, axis=1, keepdims=True)  # Normalize a copy so the scientific input cannot be changed in memory.
    w, x, y, z = q.T  # MuJoCo quaternion order is w,x,y,z.
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # Return an angle, not an individual quaternion coordinate.


def make_windows(episodes, size=100, fraction=0.):
    """Local: reconstruct complete eligible windows; global: keep fitting health/exclusions separate from evaluation."""
    rows, tail = [], 0  # Incomplete tails never cross reset boundaries.
    for item in episodes:
        n, hidden = int(item["length"]), item["activations"]  # The recorded trajectory defines available samples.
        tail += max(n - 100, 0) % size  # Count only incomplete post-startup observations.
        heading = yaw(item) if "observations" in item and item["observations"].shape[1] == 105 else None  # Heading diagnostics apply only to the verified Ant interface.
        for start in range(100, n - size + 1, size):
            segment = slice(start, start + size)  # Every window contains exactly the requested duration.
            row = {key: float(np.mean(item[field][segment])) for key, field in METRICS.items() if key not in ("planar_speed", "reward", "saturation")}  # Fitting eligibility depends on these original descriptors.
            row.update(seed=int(item["seed"]), start=start, h=hidden[segment].mean(axis=0, dtype=np.float64), size=size)  # Float64 accumulation matches the original stable means.
            row["finite"] = all(np.isfinite(row[key]) for key in METRICS if key in row) and np.isfinite(hidden[segment]).all()  # Nonfinite hidden samples cannot hide inside means.
            row["healthy"] = row["finite"] and row["inversion"] <= .05 and row["bad_contact"] <= .05  # Apply the project pilot health allowance only to fitting.
            if heading is not None:
                row["heading"] = float(heading[segment].mean())  # Reproduce the original inspection's arithmetic heading mean, not the later circular-mean audit.
            rows.append(row)  # Retain excluded rows until the reason counts have been formed.
    healthy = [row for row in rows if row["healthy"]]  # Only finite healthy fitting windows define the speed floor.
    median = float(np.median([row["speed"] for row in healthy])) if healthy else None  # There is no validation-data access in this calculation.
    floor = fraction * median if fraction and median is not None else None  # Zero disables the later heuristic for original experiment 001.
    good = [row for row in healthy if floor is None or row["speed"] > floor]  # The actual floor uses a strict greater-than comparison.
    stats = {"episodes": len(episodes), "total_windows": len(rows), "eligible_windows": len(good), "eligible_episodes": len({row["seed"] for row in good}), "excluded_nonfinite": sum(not row["finite"] for row in rows), "excluded_unhealthy": sum(row["finite"] and not row["healthy"] for row in rows), "discarded_tail_steps": tail, "warmup": 100, "window": size, "minimum_speed_fraction": fraction, "minimum_speed_floor": floor, "healthy_median_speed": median, "excluded_slow_windows": len(healthy) - len(good)}  # Explicit exclusions make the independently reconstructed scale inspectable.
    return good, stats  # Evaluation uses a separate path which never applies fitting exclusions.


def activation_rms(episodes, rows):
    """Local: compute centered eligible-timestep vector RMS; global: preserve the meaning of intervention strength."""
    means, energies = [], []  # One contribution per eligible episode, regardless of its number of windows.
    for item in episodes:
        chunks = [item["activations"][row["start"]:row["start"] + row["size"]] for row in rows if row["seed"] == int(item["seed"])]  # Recover every timestep inside eligible complete windows.
        if chunks:
            hidden = np.concatenate(chunks).astype(float)  # Within-window activation variation must not disappear into window means.
            means.append(hidden.mean(axis=0)); energies.append(np.square(hidden).sum(axis=1).mean())  # Sufficient statistics retain all hidden dimensions without dividing by width.
    return float(np.sqrt(np.mean(energies) - np.square(np.mean(means, axis=0)).sum()))  # E||h-hbar||² = E||h||² - ||hbar||² under the same episode weights.


def episode_summary(item):
    """Local: independently summarize available post-onset duration; global: retain every failed reset."""
    prefix = int(item["length"]) <= 100  # Prefix-only episodes have no measured intervention interval.
    means = {key: 0. if prefix else float(np.mean(item[field][100:])) for key, field in METRICS.items()}  # Preserve zero summaries for prefix-only cases and available-duration means otherwise.
    means["failure"] = float(prefix or bool(item["terminated"]) or means["inversion"] > .05 or means["bad_contact"] > .05)  # Termination and physical episode failure remain separate from completion at a time limit.
    means["prefix_only"] = float(prefix)  # The strict original gate tracks these resets explicitly.
    return means  # This is an episode-weighted estimand, not timestep pooling or imputed full-horizon motion.


def paired_summary(before, after, behavior, bootstrap_seed):
    """Local: match resets and bootstrap complete pairs; global: reproduce estimates without production statistics."""
    left, right = ({int(item["seed"]): item for item in group} for group in (before, after))  # Input order cannot change pairing.
    if set(left) != set(right) or len(left) != len(before) or len(right) != len(after) or len(left) < 2:
        raise ValueError("paired seed identities must be equal and unique")  # Missing or duplicated resets cannot silently improve a result.
    seeds = sorted(left)  # Stable order reproduces the historical RNG index stream.
    if any(str(left[seed]["onset_hash"]) != str(right[seed]["onset_hash"]) for seed in seeds):
        raise ValueError("paired onset states differ")  # Empty equal hashes occur only in the explicitly retained prefix-only examples.
    summaries = [[episode_summary(group[seed]) for seed in seeds] for group in (left, right)]  # Keep physical failures in both arms.
    means = [{key: float(np.mean([item[key] for item in group])) for key in group[0]} for group in summaries]  # Equal episode weights define the published means.
    delta = np.array([treated[behavior] - baseline[behavior] for baseline, treated in zip(*summaries)])  # Bootstrap the differences rather than unpaired groups.
    indices = np.random.default_rng(bootstrap_seed).integers(0, len(seeds), size=(2000, len(seeds)))  # Same paired-index construction as SciPy's documented paired bootstrap.
    return {"effect": float(delta.mean()), "ci95": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(), "baseline_mean": means[0][behavior], "treated_mean": means[1][behavior], "baseline_means": means[0], "treated_means": means[1], "n_pairs": len(seeds), "seeds": seeds, "bootstrap": {"method": "paired episode percentile", "resamples": 2000, "seed": bootstrap_seed}}  # All stored numerical estimates can be compared field by field.


def cosine(left, right):
    """Local: compare raw direction orientation; global: label sensitivity as a descriptive fitting diagnostic."""
    return float(np.clip(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)), -1, 1))  # Clamp only floating-point overshoot of the cosine range.


def sensitivity(episodes, behavior):
    """Local: reproduce Ant's original fitting-only table; global: publish diagnostics without replacing a vector."""
    rows, _ = make_windows(episodes, fraction=.5)  # This sample had no realized floor exclusions, as independently checked elsewhere.
    raw, _ = contrast(rows, behavior)  # Every sensitivity cosine references the original 100-step direction.
    low, high, cuts = quartiles(rows, behavior)  # The displayed high/low groups use the original thresholds.
    by_seed = {int(item["seed"]): item for item in episodes}  # Preserve actual episode boundaries for halves and temporal correlations.
    groups = {}  # Explain contributor sizes and the stability of sign within each window.
    for name, group in (("low", low), ("high", high)):
        halves, agreement = [], []  # Temporal descriptors use observed behavior, not the activation direction.
        for row in group:
            samples = by_seed[row["seed"]][METRICS[behavior]][row["start"]:row["start"] + 100]  # First/second halves share the same five-second window.
            halves.append([samples[:50].mean(), samples[50:].mean()]); agreement.append(np.mean(np.sign(samples) == np.sign(row[behavior])))  # Keep timestep sign agreement distinct from half-window agreement.
        halves = np.asarray(halves)  # Vectorized comparison preserves one contribution per window.
        groups[name] = {"episodes": len({row["seed"] for row in group}), "windows": len(group), "max_windows_one_episode": max(sum(row["seed"] == seed for row in group) for seed in {row["seed"] for row in group}), "episode_balanced_behavior": float(balanced(group, behavior)), "episode_balanced_speed": float(balanced(group, "speed")), "start100_windows": sum(row["start"] == 100 for row in group), "half_window_same_sign_fraction": float(np.mean(np.sign(halves[:, 0]) == np.sign(halves[:, 1]))), "step_same_sign_mean": float(np.mean(agreement)), "half_means": halves.mean(axis=0).tolist()}  # These are the fields in the preserved original inspection.
        if "heading" in group[0]:
            groups[name]["mean_heading"] = float(np.mean([row["heading"] for row in group]))  # The original descriptive heading statistic weights windows equally; behavior/activation contrasts above weight episodes equally.
    bounds = np.quantile([row[behavior] for row in rows], [.01, .99])  # Trim labels using fitting-only tails.
    subsets = {"exclude_first_window": [row for row in rows if row["start"] != 100], "complete_episodes_only": [row for row in rows if int(by_seed[row["seed"]]["length"]) == 1000], "trim_label_extreme_1pct": [row for row in rows if bounds[0] <= row[behavior] <= bounds[1]]}  # Each analysis recomputes quartile boundaries after exclusion.
    checks = {}  # Preserve every sensitivity rather than only the most favorable case.
    for name, subset in subsets.items():
        direction, detail = contrast(subset, behavior)  # Refitting is diagnostic only; no scientific vector is written.
        checks[name] = {"windows": len(subset), "cosine": cosine(raw, direction), "raw_norm": float(np.linalg.norm(direction)), "contrast": detail["contrast"]}  # Report magnitude as well as orientation.
    leave_one = []  # Removing an entire episode measures reset influence without treating windows as independent resets.
    for seed in sorted({row["seed"] for row in rows}):
        direction, _ = contrast([row for row in rows if row["seed"] != seed], behavior)  # Recompute all quartile thresholds after removing the episode.
        leave_one.append([cosine(raw, direction), seed, float(np.linalg.norm(direction))])  # Record the worst contributing reset by cosine.
    adjacent = [(left[behavior], right[behavior]) for seed in by_seed for left, right in zip([row for row in rows if row["seed"] == seed][:-1], [row for row in rows if row["seed"] == seed][1:]) if right["start"] - left["start"] == 100]  # Never correlate windows across episode or missing-window boundaries.
    longer = {}  # Longer windows test whether the contrast persists beyond one five-second segment.
    for size in (200, 300):
        subset, _ = make_windows(episodes, size=size, fraction=.5)  # Discard incomplete tails separately for each size.
        direction, detail = contrast(subset, behavior)  # Each duration gets its own eligible-window quartiles.
        longer[str(size)] = {"windows": len(subset), "cosine": cosine(raw, direction), "raw_norm": float(np.linalg.norm(direction)), "contrast": detail["contrast"]}  # These are descriptive estimates, not new candidates.
    return {"quartile_cuts": cuts.tolist(), "range": np.quantile([row[behavior] for row in rows], [0, .01, .5, .99, 1]).tolist(), "groups": groups, "sensitivity": checks, "leave_one_episode_out_worst": min(leave_one), "adjacent_window_correlation": float(np.corrcoef(np.asarray(adjacent).T)[0, 1]), "longwindow": longer}  # Publish all original table calculations in non-executable JSON.

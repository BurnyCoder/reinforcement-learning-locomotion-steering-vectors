"""Local: contrast vectors and paired statistics. Global: auditable locomotion evidence.

The high-minus-low estimator adapts https://arxiv.org/abs/2312.06681 to
episode-balanced locomotion windows; it is not a causal estimator until intervened.
Numerical reductions reuse https://numpy.org/doc/1.26/reference/generated/numpy.mean.html.
Bootstrap index pairing follows https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html.
"""

from __future__ import annotations  # Keep annotations independent of runtime evaluation.

import numpy as np  # Reuse array reductions, norms, quantiles, and reproducible random generators.

METRICS = {"speed": "x_velocity", "effort": "effort", "height": "height", "lateral": "y_velocity", "turning": "yaw_rate"}  # Public behaviors map to measured simulator descriptors.
QUALITY = ("inversion", "bad_contact")  # Fractions detect physically disruptive interventions.
EXTRAS = ("planar_speed", "saturation", "reward")  # Retain useful non-target outcomes for interpretation.


def window_table(episodes: list[dict], warmup: int = 100, window: int = 100, *, minimum_speed_fraction: float = 0.0) -> dict:
    """Local: summarize full windows and mark exclusions. Global: preserve episode identity.

The returned arrays include ineligible windows for auditing; fitting must use
``eligible``. Only complete post-warmup windows enter the table or its RMS scale.
An optional speed floor is a project pilot heuristic, not a standard CAA rule.
It uses only fitting-window data and NumPy's median implementation:
https://numpy.org/doc/1.26/reference/generated/numpy.median.html.
"""
    if warmup < 0 or window < 1:
        raise ValueError("warmup must be nonnegative and window must be positive")  # Reject ill-defined temporal partitions.
    if not np.isfinite(minimum_speed_fraction) or minimum_speed_fraction < 0:
        raise ValueError("minimum_speed_fraction must be finite and nonnegative")  # Zero preserves the original protocol; malformed floors cannot change eligibility silently.
    names = {**METRICS, **{name: name for name in QUALITY}}  # Use one measurement map for every fitting window.
    rows = []  # Accumulate a small table rather than copying complete trajectories.
    seen = set()  # Reset identity is the independent sampling unit.
    width = 0  # Preserve a predictable empty-table shape if no episodes are supplied.
    tail_steps = 0  # Explicitly count data omitted because a window is incomplete.
    for item in episodes:
        identity = int(item["seed"])  # Normalize NumPy scalar seed metadata to a Python integer.
        if identity in seen:
            raise ValueError("episode seeds must be unique within a fitting split")  # Duplicate resets cannot inflate contributor counts.
        seen.add(identity)  # Retain every independent episode identity once.
        hidden = np.asarray(item["activations"])  # Preserve the rollout's efficient float32 storage.
        if hidden.ndim != 2 or hidden.shape[1] == 0:
            raise ValueError("activations must have shape (timesteps, hidden_units)")  # Require a well-defined vector space.
        if width and width != hidden.shape[1]:
            raise ValueError("activation width differs between episodes")  # Different policy layers cannot share a fitted vector.
        width = hidden.shape[1]  # Track the common actor-layer dimensionality.
        length = len(hidden)  # Actual stored samples are the source of temporal truth.
        if int(item.get("length", length)) != length:
            raise ValueError("episode length disagrees with activation length")  # A partial artifact is an implementation error.
        arrays = {name: np.asarray(item[source], dtype=float) for name, source in names.items()}  # Convert descriptors once per episode.
        if any(value.shape != (length,) for value in arrays.values()):
            raise ValueError("measurement length disagrees with activation length")  # Align every descriptor with the action's timestep.
        tail_steps += max(0, length - warmup) % window  # Count incomplete post-startup tails without treating them as full windows.
        for start in range(warmup, length - window + 1, window):
            segment = slice(start, start + window)  # Non-overlap prevents duplicated samples in the fitting table.
            activation = hidden[segment].mean(axis=0, dtype=np.float64)  # Float64 accumulation avoids float32 reduction error.
            row = {name: float(value[segment].mean()) for name, value in arrays.items()}  # All behavior scores summarize exactly this window.
            finite = bool(np.isfinite(hidden[segment]).all() and all(np.isfinite(value) for value in row.values()))  # Nonfinite samples never silently enter fitting.
            healthy = all(row[name] <= 0.05 for name in QUALITY)  # The planned health allowance is five percent per window.
            row.update(activations=activation, episode=identity, start=start, eligible=finite and healthy, finite=finite)  # Keep the exclusion and its original identity.
            rows.append(row)  # Retain excluded rows so diagnostics can explain discarded data.
    keys = list(names) + ["episode", "start", "eligible", "finite"]  # Scalar columns remain directly indexable arrays.
    table = {key: np.asarray([row[key] for row in rows], dtype=bool if key in {"eligible", "finite"} else int if key in {"episode", "start"} else float) for key in keys}  # Explicit dtypes make empty tables reliable.
    table["activations"] = np.asarray([row["activations"] for row in rows]).reshape(-1, width) if width else np.empty((0, 0))  # Keep two-dimensional activation shape.
    healthy = table["eligible"].copy()  # Preserve the original finite-and-healthy mask so exclusion reasons remain separate.
    median_speed = float(np.median(table["speed"][healthy])) if healthy.any() else None  # Only supplied finite healthy fitting windows establish typical speed.
    floor = float(minimum_speed_fraction * median_speed) if minimum_speed_fraction > 0 and median_speed is not None else None  # Explicit zero disables the new heuristic exactly.
    table["excluded_slow"] = healthy & (table["speed"] <= floor) if floor is not None else np.zeros(len(rows), dtype=bool)  # Strictly greater speed is required only when the optional floor exists.
    table["eligible"] = healthy & ~table["excluded_slow"]  # Exclude stationary outliers from both contrast extraction and RMS normalization.
    eligible = table["eligible"]  # Reuse the final mask when reporting retained windows and episodes.
    reason = "disabled: minimum_speed_fraction=0" if minimum_speed_fraction == 0 else "no finite healthy fitting windows to set a floor" if median_speed is None else "project pilot heuristic: exclude finite healthy fitting windows at or below fraction times their median speed"  # State the actual decision without attributing this heuristic to a paper.
    table["stats"] = {"episodes": len(episodes), "total_windows": len(rows), "eligible_windows": int(eligible.sum()), "eligible_episodes": len(np.unique(table["episode"][eligible])), "excluded_nonfinite": int((~table["finite"]).sum()), "excluded_unhealthy": int((table["finite"] & ~healthy).sum()), "discarded_tail_steps": tail_steps, "warmup": warmup, "window": window}  # Existing exclusion counts retain their original meanings.
    table["stats"].update(minimum_speed_fraction=float(minimum_speed_fraction), minimum_speed_floor=floor, healthy_median_speed=median_speed, excluded_slow_windows=int(table["excluded_slow"].sum()), speed_filter_reason=reason)  # Record the fitted absolute floor, chosen fraction, and additional exclusions for reproducibility.
    return table  # The caller can plot all windows while fitting only eligible data.


def _episode_mean(values: np.ndarray, identities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Local: average within episode, then across episodes. Global: prevent duration bias."""
    return np.mean([values[mask & (identities == identity)].mean(axis=0) for identity in np.unique(identities[mask])], axis=0)  # Equal episode weights follow the experiment specification.


def _speed_bins(speed: np.ndarray) -> np.ndarray:
    """Local: form up to five speed strata. Global: reduce speed confounding of effort."""
    unique = np.unique(speed)  # Exact repeated commands should remain in separate homogeneous strata.
    if len(unique) <= 5:
        return np.searchsorted(unique, speed)  # Matching exact speeds removes between-command confounding entirely.
    edges = np.unique(np.quantile(speed, [0.2, 0.4, 0.6, 0.8]))  # Fixed quantile boundaries adapt to available fitting coverage.
    return np.searchsorted(edges, speed, side="right")  # Every window belongs to exactly one ordered stratum.


def _contrast(table: dict, behavior: str, labels: np.ndarray, minimum: int = 8) -> tuple[np.ndarray | None, dict]:
    """Local: fit equal-bin, equal-episode contrasts. Global: expose unsupported groups."""
    identities = table["episode"]  # Windows from a seed remain one independent contributor.
    bins = _speed_bins(table["speed"]) if behavior == "effort" else np.zeros(len(labels), dtype=int)  # Only effort is explicitly conditioned on speed.
    differences, contrasts, details = [], [], []  # Retain each valid stratum's estimate and provenance.
    high_all, low_all = np.zeros(len(labels), dtype=bool), np.zeros(len(labels), dtype=bool)  # Track distinct contributors across strata.
    for group in np.unique(bins):
        member = bins == group  # Conditioning uses only this speed stratum's behavior distribution.
        low_cut, high_cut = np.quantile(labels[member], [0.25, 0.75])  # NumPy's default linear quantile defines the documented quartiles.
        if high_cut - low_cut <= 1e-8 * max(1.0, float(np.abs(labels[member]).mean())):
            continue  # Tied or negligible labels do not establish a behavioral contrast.
        low, high = member & (labels <= low_cut), member & (labels >= high_cut)  # Distinct thresholds make the two groups disjoint.
        differences.append(_episode_mean(table["activations"], identities, high) - _episode_mean(table["activations"], identities, low))  # Classic activation difference with episode balancing.
        contrasts.append(float(_episode_mean(labels, identities, high) - _episode_mean(labels, identities, low)))  # Report the actual descriptor separation before scaling.
        high_all |= high  # Union independent episodes rather than summing per-bin counts.
        low_all |= low  # The same episode may legitimately contribute to both contrast groups.
        details.append({"bin": int(group), "low_cut": float(low_cut), "high_cut": float(high_cut), "low_windows": int(low.sum()), "high_windows": int(high.sum()), "low_speed": float(_episode_mean(table["speed"], identities, low)), "high_speed": float(_episode_mean(table["speed"], identities, high))})  # Residual within-bin speed differences remain visible.
    diagnostics = {"high_episodes": len(np.unique(identities[high_all])), "low_episodes": len(np.unique(identities[low_all])), "high_windows": int(high_all.sum()), "low_windows": int(low_all.sum()), "bins": details}  # Make all group-size decisions auditable.
    if not differences:
        return None, {**diagnostics, "reason": "insufficient behavioral contrast within eligible windows"}  # Do not fabricate ranks when labels tie.
    if min(diagnostics["high_episodes"], diagnostics["low_episodes"]) < minimum:
        return None, {**diagnostics, "reason": f"fewer than {minimum} independent episodes in a contrast group"}  # Timesteps do not substitute for independent contributors.
    return np.mean(differences, axis=0), {**diagnostics, "contrast": float(np.mean(contrasts))}  # Equal stratum weighting removes between-bin composition differences.


def _activation_rms(episodes: list[dict], table: dict) -> float:
    """Local: center eligible timestep activations. Global: give alpha a stated RMS scale."""
    moments, energies = [], []  # One sufficient-statistic pair per eligible episode keeps memory bounded.
    size = table["stats"]["window"]  # Match the exact full windows used for fitting.
    for item in episodes:
        starts = table["start"][table["eligible"] & (table["episode"] == int(item["seed"]))]  # Select only eligible windows belonging to this episode.
        if not len(starts):
            continue  # Episodes with no eligible data cannot define the fitting scale.
        hidden = np.concatenate([item["activations"][start:start + size] for start in starts]).astype(np.float64)  # Preserve within-window gait variation in float64.
        moments.append(hidden.mean(axis=0))  # Each episode receives equal weight regardless of duration.
        energies.append(float(np.square(hidden).sum(axis=1).mean()))  # Compute E||h||² from all eligible timestep activations.
    if not moments:
        return 0.0  # The caller reports insufficient fitting data instead of dividing by zero.
    center = np.mean(moments, axis=0)  # Define the same episode-balanced global activation mean.
    return float(np.sqrt(max(0.0, float(np.mean(energies) - np.square(center).sum()))))  # E||h-hbar||² equals the second moment minus squared mean.


def _split_half_cosine(table: dict, behavior: str, rng: np.random.Generator) -> float | None:
    """Local: compare disjoint-episode contrasts. Global: diagnose direction stability only."""
    halves = np.array_split(rng.permutation(np.unique(table["episode"])), 2)  # Split independent episodes without separating their windows.
    directions = []  # Estimate each direction independently within its episode subset.
    for identities in halves:
        mask = np.isin(table["episode"], identities)  # Preserve all windows for each selected episode.
        if not mask.any():
            return None  # An empty half cannot provide a stability estimate.
        subset = {name: value[mask] for name, value in table.items()}  # This helper receives array-only eligible tables.
        direction, _ = _contrast(subset, behavior, subset[behavior], minimum=1)  # Diagnostic estimates need not pass the full fitting contributor gate.
        if direction is None or np.linalg.norm(direction) <= 1e-12:
            return None  # Undefined cosine is recorded explicitly rather than filled with zero.
        directions.append(direction)  # Keep the raw direction so normalization does not affect cosine.
    return float(np.clip(np.dot(*directions) / (np.linalg.norm(directions[0]) * np.linalg.norm(directions[1])), -1, 1))  # Bound floating-point roundoff to the cosine domain.


def fit_vectors(episodes: list[dict], behaviors: list[str], seed: int = 0, *, warmup: int = 100, minimum_speed_fraction: float = 0.0) -> tuple[dict[str, np.ndarray], dict]:
    """Local: fit contrasts and matched controls. Global: persist interpretable steering artifacts.

RMS uses every timestep in eligible full windows, balancing episodes equally.
Shuffled controls permute window labels within effort speed strata; they are
negative controls, not a formal exchangeability test for correlated windows.
"""
    if any(behavior not in METRICS for behavior in behaviors):
        raise ValueError(f"unknown behavior; expected one of {list(METRICS)}")  # Fail visibly on an experiment specification typo.
    table = window_table(episodes, warmup=warmup, minimum_speed_fraction=minimum_speed_fraction)  # Apply the configured onset and fitting-only locomotion eligibility rule once.
    scale = _activation_rms(episodes, table)  # Use timestep variability, not only the variance of window means.
    diagnostics = {"windows": table["stats"], "activation_rms": scale, "rms_convention": f"episode-balanced eligible timestep RMS about episode-balanced mean; full 100-step windows after {warmup}-step warmup", "seed": seed, "behaviors": {}, "vectors": {}}  # Metadata contains all scaling and exclusion decisions.
    vectors = {}  # Only supported finite directions are returned for rollout intervention.
    eligible = {name: value[table["eligible"]] for name, value in table.items() if isinstance(value, np.ndarray)}  # Excluded windows remain in diagnostics but cannot affect extraction.
    for behavior in dict.fromkeys(behaviors):
        if not len(eligible["episode"]) or scale <= 1e-12 or not np.isfinite(scale):
            diagnostics["behaviors"][behavior] = {"status": "skipped", "reason": "no finite nonzero eligible activation scale"}  # A scale-free vector cannot be safely normalized.
            continue  # Other behaviors share the same data limitation, but receive explicit reports.
        raw, detail = _contrast(eligible, behavior, eligible[behavior])  # Compute the planned high-minus-low behavior direction.
        norm = float(np.linalg.norm(raw)) if raw is not None else 0.0  # Inspect magnitude before normalization hides weak contrasts.
        if raw is None or not np.isfinite(norm) or norm <= max(1e-12, scale * 1e-8):
            diagnostics["behaviors"][behavior] = {**detail, "status": "skipped", "reason": detail.get("reason", "negligible or nonfinite activation contrast"), "raw_norm": norm}  # Record the precise unsupported-candidate reason.
            continue  # Never normalize numerical noise into a large intervention.
        rng = np.random.default_rng(np.random.SeedSequence([seed, list(METRICS).index(behavior)]))  # Behavior-specific streams make results independent of behavior ordering.
        diagnostics["behaviors"][behavior] = {**detail, "status": "fitted", "raw_norm": norm, "split_half_cosine": _split_half_cosine(eligible, behavior, rng)}  # Stability is a diagnostic rather than causal proof.
        candidates = {behavior: (raw, "activation_difference")}  # The behavioral direction defines the primary candidate.
        for number in range(3):
            candidates[f"{behavior}_random{number}"] = (rng.standard_normal(raw.shape), "random_direction")  # Isotropic directions provide three fixed perturbation controls.
        shuffled = eligible[behavior].copy()  # Preserve original labels for diagnostic reporting and fitting reproducibility.
        bins = _speed_bins(eligible["speed"]) if behavior == "effort" else np.zeros(len(shuffled), dtype=int)  # Effort shuffling retains its speed-stratum structure.
        for group in np.unique(bins):
            mask = bins == group  # Labels cannot cross the effort matching boundary.
            shuffled[mask] = rng.permutation(shuffled[mask])  # Break activation-label association without changing the label distribution.
        shuffled_raw, shuffled_detail = _contrast(eligible, behavior, shuffled)  # Apply the same contributor and quartile rules to the negative control.
        if shuffled_raw is not None and np.linalg.norm(shuffled_raw) > max(1e-12, scale * 1e-8):
            candidates[f"{behavior}_shuffled"] = (shuffled_raw, "shuffled_labels")  # Keep an actual shuffled-label estimator rather than relabeling random noise.
        else:
            diagnostics["behaviors"][behavior]["shuffled_control_skipped"] = shuffled_detail.get("reason", "negligible shuffled contrast")  # A degenerate control is reported without silent replacement.
        for name, (direction, method) in candidates.items():
            direction_norm = float(np.linalg.norm(direction))  # Normalize every candidate to the identical fitting-derived magnitude.
            vectors[name] = (direction * (scale / direction_norm)).astype(np.float32)  # Match the actor activation dtype while retaining raw float64 metadata.
            diagnostics["vectors"][name] = {"behavior": behavior, "method": method, "raw_vector": direction.tolist(), "raw_norm": direction_norm, "norm": scale, "scale_factor": scale / direction_norm}  # JSON-safe provenance accompanies every saved vector.
    return vectors, diagnostics  # The pipeline owns NPZ/JSON serialization and timestamped reporting.


def _summary(item: dict, warmup: int = 100) -> dict:
    """Local: reduce a rollout to one outcome. Global: keep failed episodes in inference.

Physical failure means post-onset inversion or forbidden contact exceeds five
percent within this episode. This is an explicit project pilot definition;
HalfCheetah itself never terminates for falling, as documented at
https://gymnasium.farama.org/environments/mujoco/half_cheetah/#episode-end.
"""
    result = {}  # One observation per reset avoids timestep pseudoreplication.
    prefix_only = int(item.get("length", warmup + 1)) <= warmup  # Match runtime's convention for absent post-onset evidence.
    mapping = {**METRICS, **{name: name for name in QUALITY + EXTRAS}}  # Alias raw simulator measurements to stable report names.
    for name, source in mapping.items():
        value = np.asarray(item[source] if source in item else item[name], dtype=float)  # Accept complete rollouts or runtime scalar summaries.
        selected = value[warmup:] if value.ndim and len(value) > warmup else value  # Reduce raw arrays while leaving runtime scalar summaries intact.
        if not selected.size or not np.isfinite(selected).all():
            raise ValueError(f"nonfinite or empty evaluation metric: {name}")  # Invalid artifacts cannot become an apparent effect.
        result[name] = 0.0 if prefix_only else float(selected.mean())  # Absent post-onset outcomes are zero, matching runtime.summarise_episode without dropping the reset.
    failure = float(item.get("failure", bool(item.get("terminated", False))))  # Preserve the runtime summary's explicit failure flag when termination metadata has been reduced away.
    if not np.isfinite(failure) or not 0 <= failure <= 1:
        raise ValueError("evaluation failure must be a finite fraction between zero and one")  # Invalid failure metadata cannot silently pass the quality gate.
    physical_failure = float(any(result[name] > 0.05 for name in QUALITY))  # Classify each episode before averaging so isolated collapses cannot hide in pooled contact fractions.
    result["failure"] = max(failure, float(prefix_only), float(bool(item.get("terminated", False))), physical_failure)  # Combine explicit termination and the declared physical pilot criterion without erasing either source of failure.
    result["prefix_only"] = float(prefix_only)  # Such episodes cannot demonstrate sustained steering.
    return result  # Every supplied reset contributes exactly once to the paired analysis.


def paired_effect(baseline: list[dict], treated: list[dict], behavior: str, seed: int = 0, *, warmup: int = 100) -> dict:
    """Local: paired bootstrap and pilot gates. Global: quantify useful causal effects.

Confidence intervals are descriptive percentile intervals from 2,000 resampled
episode pairs. The caller must additionally lock the validation-selected sign,
account for adaptive searches, and obtain independent replication.
"""
    if behavior not in METRICS:
        raise ValueError(f"unknown behavior: {behavior}")  # Do not infer an unintended target from vector naming.
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")  # Evaluation cannot start before the recorded rollout.
    left = {int(item["seed"]): item for item in baseline}  # Canonical reset maps permit harmless input reordering.
    right = {int(item["seed"]): item for item in treated}  # Pairing is keyed by reset identity rather than list position.
    if len(left) != len(baseline) or len(right) != len(treated) or set(left) != set(right) or len(left) < 2:
        raise ValueError("paired evaluation requires at least two identical, unique seed sets")  # Missing or duplicate resets invalidate a paired analysis.
    seeds = sorted(left)  # Stable ordering also makes bootstrap results reproducible.
    if any("onset_hash" not in left[key] or "onset_hash" not in right[key] or str(left[key]["onset_hash"]) != str(right[key]["onset_hash"]) for key in seeds):
        raise ValueError("paired episodes have missing or unequal intervention-onset states")  # Causal comparisons must begin at the same physical state.
    baseline_rows = [_summary(left[key], warmup) for key in seeds]  # Summarize every baseline episode at the configured onset, including failures.
    treated_rows = [_summary(right[key], warmup) for key in seeds]  # Summarize matching treatment episodes with the same onset and no exclusions.
    before = {name: float(np.mean([row[name] for row in baseline_rows])) for name in baseline_rows[0]}  # Equal reset weights define reported population means.
    after = {name: float(np.mean([row[name] for row in treated_rows])) for name in treated_rows[0]}  # Use the exact same metric set for treatment.
    differences = np.asarray([changed[behavior] - original[behavior] for original, changed in zip(baseline_rows, treated_rows)])  # Pairing removes shared reset variability.
    effect = float(differences.mean())  # Report the signed physical-unit treatment-minus-baseline effect.
    indices = np.random.default_rng(seed).integers(0, len(seeds), size=(2000, len(seeds)))  # Reuse episode indices jointly as prescribed by paired bootstrap.
    interval = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]).tolist()  # A percentile interval keeps the estimator explicit and reproducible.
    quality = {name: after[name] - before[name] for name in QUALITY + ("failure",)}  # These changes are in fraction units, not relative percentages.
    quality_pass = all(value <= 0.05 + 1e-12 for value in quality.values()) and after["prefix_only"] == 0 and before["prefix_only"] == 0  # Failed pre-onset episodes remain counted and disqualify sustained claims.
    magnitude = abs(effect)  # Either direction can be useful except for the explicitly minimizing effort target.
    reference = abs(before[behavior])  # Relative effects use a stable positive denominator.
    if behavior == "speed":
        size_pass, movement = magnitude >= 0.05 * reference, before["speed"] > 0 and after["speed"] >= 0.5 * before["speed"]  # Preserve forward locomotion while allowing useful acceleration or deceleration.
    elif behavior == "effort":
        size_pass, movement = -effect >= 0.10 * reference, before["speed"] > 0 and abs(after["speed"] - before["speed"]) <= 0.05 * abs(before["speed"]) + 1e-12  # Reduced action use must retain comparable travel speed.
    elif behavior == "height":
        size_pass, movement = magnitude >= 0.03, before["speed"] > 0 and abs(after["speed"] - before["speed"]) <= 0.10 * abs(before["speed"]) + 1e-12  # Posture changes must preserve forward speed within ten percent.
    elif behavior == "lateral":
        size_pass, movement = magnitude >= 0.2, before["speed"] > 0 and after["speed"] >= 0.8 * before["speed"]  # Sideways steering must retain most forward progress.
    else:
        size_pass, movement = magnitude >= 0.1, before["planar_speed"] > 0 and after["planar_speed"] >= 0.8 * before["planar_speed"]  # Turning can change heading while preserving planar locomotion.
    evidence = interval[1] < 0 if behavior == "effort" else interval[0] > 0 or interval[1] < 0  # A desired effect must have an interval excluding zero in its useful direction.
    return {"behavior": behavior, "effect": effect, "ci95": interval, "relative_effect": effect / reference if reference > 1e-12 else None, "baseline_mean": before[behavior], "treated_mean": after[behavior], "baseline_means": before, "treated_means": after, "n_pairs": len(seeds), "seeds": seeds, "gate": bool(size_pass and movement and evidence and quality_pass), "effect_size_pass": bool(size_pass), "ci_excludes_zero": bool(evidence), "locomotion_pass": bool(movement), "quality_pass": bool(quality_pass), "quality_deltas": quality, "direction": int(np.sign(effect)), "bootstrap": {"method": "paired episode percentile", "resamples": 2000, "seed": seed}}  # Compact JSON-safe evidence supports selection, plotting, and confirmation.

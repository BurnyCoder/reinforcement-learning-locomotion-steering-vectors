"""Project Ant arrays into an auditable publication schema without simulation.

The existing analysis._summary owns the historical estimator. New intervals for
previously untabulated arms are explicitly descriptive paired percentile intervals.
Sources: https://numpy.org/doc/1.26/reference/generated/numpy.load.html and
https://numpy.org/doc/1.26/reference/random/generated/numpy.random.Generator.integers.html
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ..analysis import _summary
from .inputs import canonical_json, write_json

METRICS = ("x_velocity", "y_velocity", "planar_speed", "height", "yaw_rate", "effort", "inversion", "bad_contact", "reward", "saturation")
ALIASES = {"speed": "x_velocity", "lateral": "y_velocity", "turning": "yaw_rate"}
PUBLICATION_SEED = 20261029  # Fixed saved-data presentation resampling; not a new experiment seed.
REPO_URL = "https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors"


def summarize_episode(item: dict, warmup: int) -> dict:
    """Validate aligned arrays, then reuse the original equal-episode failure convention."""
    length = int(item["length"])
    if length <= 0 or not np.isfinite(float(item["dt"])) or float(item["dt"]) <= 0:
        raise ValueError("Episode length and simulator timestep must be positive")
    for key in METRICS:
        values = np.asarray(item[key])
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid episode metric {key}: expected {length} finite values")
    result = {ALIASES.get(key, key): value for key, value in _summary(item, warmup).items()}
    result["total_return"] = float(np.sum(item["reward"]))
    return result


def paired_summary(baseline: dict, treated: dict, *, warmup: int) -> dict:
    """Resample complete reset pairs, retaining every failed or prefix-only episode."""
    if len(baseline) < 2 or set(baseline) != set(treated):
        raise ValueError("Paired publication requires identical seed sets with at least two resets")
    seeds = sorted(baseline)
    for seed in seeds:
        if str(baseline[seed]["onset_hash"]) != str(treated[seed]["onset_hash"]):
            raise ValueError("Paired onset states differ")
        if float(baseline[seed]["dt"]) != float(treated[seed]["dt"]):
            raise ValueError("Paired simulator timesteps differ")
    left = [summarize_episode(baseline[seed], warmup) for seed in seeds]
    right = [summarize_episode(treated[seed], warmup) for seed in seeds]
    differences = np.array([b["y_velocity"] - a["y_velocity"] for a, b in zip(left, right)])
    draws = np.random.default_rng(PUBLICATION_SEED).integers(0, len(seeds), (2000, len(seeds)))
    return {"mean": float(differences.mean()), "ci95": np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist(), "ci_method": "descriptive paired percentile, 2000 resamples; unadjusted", "source": "saved arrays, publication reanalysis", "gate": None}


def trajectory(item: dict, *, warmup: int, stride: int = 10) -> dict:
    """Retain onset/end positions; rewards are block means, never fabricated post-failure data."""
    summary = summarize_episode(item, warmup)
    length, dt = int(item["length"]), float(item["dt"])
    steps = sorted(set([0, length, *range(stride, length, stride), *([warmup] if warmup < length else [])]))
    x, y = (np.r_[0.0, np.cumsum(item[key]) * dt] for key in ("x_velocity", "y_velocity"))
    rewards = [None, *[float(np.mean(item["reward"][a:b])) for a, b in zip(steps[:-1], steps[1:])]]
    rounded = lambda values: [round(float(value), 6) for value in values]
    return {"length": length, "failed": bool(summary["failure"]), "prefix_only": bool(summary["prefix_only"]), "steps": steps, "time": rounded(np.array(steps) * dt), "x": rounded(x[steps]), "y": rounded(y[steps]), "reward": [None if value is None else round(value, 6) for value in rewards], "summary": summary}


def _load_condition(root: Path, directory: Path, seeds: list[int], registered: set[str], *, observations: bool = False) -> dict:
    """Read only manifest-verified NPZ members; extra or missing resets are a hard error."""
    paths = sorted(directory.glob("*.npz"))
    episodes = {}
    for path in paths:
        if path.relative_to(root).as_posix() not in registered:
            raise ValueError(f"Unregistered episode input: {path}")
        with np.load(path, allow_pickle=False) as archive:
            episode = {key: archive[key] for key in (*METRICS, "seed", "length", "dt", "terminated", "truncated", "onset_hash")}
            if observations:
                episode["observations"] = archive["observations"]
        seed = int(episode["seed"])
        if seed in episodes or path.stem != str(seed):
            raise ValueError(f"Duplicate or misnamed episode seed: {path}")
        episodes[seed] = episode
    if sorted(episodes) != sorted(seeds):
        raise ValueError(f"Incomplete condition {directory}: expected {len(seeds)} exact seeds, found {len(episodes)}")
    return episodes


def _means(episodes: dict, warmup: int, baseline_speed: float | None = None) -> dict:
    """Equal episode means and explicit counts keep quantities and denominators distinct."""
    rows = [summarize_episode(item, warmup) for item in episodes.values()]
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in ("failure", "prefix_only")}
    result.update(failures=sum(int(row["failure"]) for row in rows), prefix_only=sum(int(row["prefix_only"]) for row in rows))
    result["forward_speed_retention"] = result["x_velocity"] / baseline_speed if baseline_speed and baseline_speed > 0 else None
    return result


def _kind(vector: str) -> str:
    """Classify saved treatment names without interpreting a name as a semantic finding."""
    if vector == "baseline":
        return "baseline"
    if vector == "lateral":
        return "primary"
    if vector == "turning":
        return "other_label"
    if "random" in vector:
        return "random"
    if "shuffled" in vector:
        return "shuffled"
    if "bias" in vector:
        return "action_bias"
    return "perception" if vector == "perceived_heading" else "ablation"


def _reward_windows(episodes: dict, warmup: int, window: int = 100) -> list[dict]:
    """Available observations only: a shrinking count is displayed beside every reward mean."""
    output = []
    for start in range(0, max(int(item["length"]) for item in episodes.values()), window):
        values = [float(np.mean(item["reward"][start:start + window])) for item in episodes.values() if int(item["length"]) > start]
        output.append({"start": start, "end": start + window, "n": len(values), "mean": float(np.mean(values))})
    return output


def _saved_effect(row: dict | None, audit: dict, cohort: str, condition: str, observed: dict) -> dict:
    """Reuse historical intervals exactly where their estimand is world-Y velocity."""
    stored = None
    if row and row["behavior"] == "lateral":
        effect = row["effect"]
        stored = {"mean": effect["effect"], "ci95": effect["ci95"], "ci_method": "saved paired percentile, 2000 resamples; unadjusted", "source": "original results.json", "gate": effect["gate"]}
    if cohort == "fresh_audit":
        primary = condition == "lateral__-0.1000"
        perception = condition.startswith("perceived_heading__")
        if primary or perception:
            effect = audit["fresh"]["P1"] if primary else audit["fresh"]["M2"]["minus_delta_arm" if "__-" in condition else "plus_delta_arm"]
            stored = {"mean": effect["mean"], "ci95": effect["bca_ci"], "ci_method": "saved paired BCa, 10000 resamples", "source": "independent audit.json", "gate": effect["production_gate"]["gate"] if primary else None}
    if stored and not np.isclose(stored["mean"], observed["mean"], atol=1e-12, rtol=1e-10):
        raise ValueError(f"Saved world-Y effect disagrees with raw episodes: {cohort}/{condition}")
    return stored or observed


def _saturation(episodes: dict, warmup: int) -> dict:
    """Describe saved action-limit prevalence; this is not a training or novelty metric."""
    available = [np.asarray(item["saturation"])[warmup:] for item in episodes.values() if int(item["length"]) > warmup]
    return {"episodes": len(episodes), "episodes_with_post_onset_data": len(available), "mean_episode_action_fraction": float(np.mean([values.mean() for values in available])), "pooled_action_fraction": float(np.concatenate(available).mean()), "mean_episode_step_fraction_any_saturated": float(np.mean([(values > 0).mean() for values in available])), "definition": "Saved fraction of action dimensions within 0.01 of either bound at each transition; available post-onset data. Equal-episode and pooled summaries are separately named."}


def _contact_clipping(episodes: dict, warmup: int) -> dict:
    """Count post-onset states with clipped planar wrench channels in saved observations.

Ant-v5 appends 13 six-component cfrc_ext rows after 27 position/velocity fields
and clips them to the default [-1,1] range. Planar yaw rotates x/y forces and
x/y moments. Clipped values cannot reconstruct the unclipped rotated wrench.
Source: https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py
"""
    states, saturated, prefix_only = 0, 0, 0
    for item in episodes.values():
        length = int(item["length"])
        observed = np.asarray(item["observations"])
        if observed.shape != (length, 105) or not np.isfinite(observed).all():
            raise ValueError("Contact-clipping audit requires finite saved Ant observations of shape (length,105)")
        prefix_only += int(length <= warmup)
        wrenches = observed[warmup:, 27:].reshape(-1, 13, 6)
        clipped = np.any(np.abs(wrenches[:, :, [0, 1, 3, 4]]) >= 1.0, axis=(1, 2))
        states += len(clipped)
        saturated += int(clipped.sum())
    return {"episodes": len(episodes), "prefix_only_episodes": prefix_only, "states": states, "states_saturated": saturated, "fraction": saturated / states if states else None, "clip_bounds": [-1, 1], "channels": [0, 1, 3, 4], "definition": "Fraction of recorded post-onset states with at least one planar contact-force or torque observation at a clipping bound; pooled states, not equal-episode means. This is a limitation of observation rotation, not a causal effect test."}


def build_evidence(root: Path, verified: dict, *, output_root: Path | None = None) -> tuple[dict, list[Path]]:
    """Assemble original and audit evidence, with all conditions and per-reset visual shards."""
    root, output_root = root.resolve(), (output_root or root).resolve()
    registered = {Path(row["local_path"]).as_posix() for row in verified["files"]}
    def read(relative):
        if relative not in registered:
            raise ValueError(f"Unregistered JSON input: {relative}")
        return json.loads((root / relative).read_text(encoding="utf-8"))
    original = read("reports/ant-classic-005/results.json")
    manifest = read("reports/ant-classic-005/manifest.json")
    selection = read("reports/ant-classic-005/selection.json")
    audit = read("reports/ant-classic-005-independent-audit/audit.json")
    directions = read("reports/ant-classic-005-independent-audit/directions.json")
    warmup = int(manifest["config"]["warmup"])
    outputs, cohorts = [], []
    fresh = audit["fresh"]
    fresh_seeds = list(range(fresh["seeds"][0], fresh["seeds"][1] + 1))
    alpha = directions["alpha"]
    fresh_names = [f"{name}__{sign * alpha:+.4f}" for name in fresh["directions"] for sign in (-1, 1)]
    fresh_names += [f"perceived_heading__{sign * directions['delta_eq_rad']:+.6f}rad" for sign in (-1, 1)]
    cases = [(phase, f"Original {phase}", root / "runs/ant-classic-005/episodes" / phase, manifest["seed_splits"][phase], original[phase]) for phase in ("validation", "confirmation")]
    cases.append(("fresh_audit", "Fresh audit resets (not formal replication)", root / "runs/ant-005-independent-audit/episodes/audit", fresh_seeds, None))
    for cohort_id, label, folder, seeds, rows in cases:
        baseline = _load_condition(root, folder / "baseline", seeds, registered, observations=cohort_id == "fresh_audit")
        base_metrics = _means(baseline, warmup)
        names = [f"{row['vector']}__{row['alpha']:+.4f}" for row in rows] if rows is not None else fresh_names
        row_map = dict(zip(names, rows)) if rows is not None else {}
        conditions = []
        for name in ["baseline", *names]:
            episodes = baseline if name == "baseline" else _load_condition(root, folder / name, seeds, registered)
            vector, strength = name.rsplit("__", 1) if "__" in name else ("baseline", "0")
            observed = paired_summary(baseline, episodes, warmup=warmup)
            effect = _saved_effect(row_map.get(name), audit, cohort_id, name, observed)
            if name == "baseline":
                effect = {"mean": 0.0, "ci95": None, "ci_method": "reference arm; no comparison interval", "source": "saved baseline episodes", "gate": None}
            metrics = _means(episodes, warmup, base_metrics["x_velocity"])
            if name in row_map:
                for key, value in row_map[name]["effect"]["treated_means"].items():
                    target = "failures" if key == "failure" else key
                    actual = metrics[target] / len(seeds) if target in ("failures", "prefix_only") else metrics[ALIASES.get(target, target)]
                    if not np.isclose(actual, value, atol=1e-12, rtol=1e-10):
                        raise ValueError(f"Saved metric disagrees with raw evidence: {cohort_id}/{name}/{key}")
            parts = fresh["directions"].get(vector) if cohort_id == "fresh_audit" else None
            if parts:
                arm = 0 if float(strength) < 0 else 1
                stored_arm = parts["minus_arm" if arm == 0 else "plus_arm"]
                if not np.isclose(effect["mean"], stored_arm, atol=1e-12, rtol=1e-10) or metrics["failures"] != parts["failures"][arm]:
                    raise ValueError(f"Saved fresh-audit arm disagrees with raw evidence: {name}")
            if cohort_id == "validation":
                parts = audit["saved"]["validation_world_y_decomposition"].get(vector, {}).get(str(abs(float(strength))))
            relative = f"site/data/episodes/{cohort_id}--{name}.json"
            shard = {
                "cohort": cohort_id, "condition": name, "warmup": warmup,
                "dt": float(next(iter(baseline.values()))["dt"]),
                "episodes": [{"seed": seed, "baseline": trajectory(baseline[seed], warmup=warmup),
                              "treated": trajectory(episodes[seed], warmup=warmup)} for seed in seeds],
            }
            outputs.append(write_json(output_root / relative, shard))
            detail = directions["directions"].get(vector, {}) if cohort_id == "fresh_audit" else audit["saved"]["validation_world_y_decomposition"].get(vector, {})
            conditions.append({
                "id": name, "label": name.replace("__", " "), "vector": vector,
                "alpha": float(strength.removesuffix("rad")),
                "alpha_unit": "radians" if vector == "perceived_heading" else "activation strength",
                "kind": _kind(vector), "metrics": metrics, "pairedEffect": effect,
                "oddEven": {key: parts[key] for key in ("odd", "even")} if parts else None,
                "cosine_with_primary": detail.get("cosine_with_primary", detail.get("cosine_with_lateral")),
                "seeds": seeds, "n_pairs": len(seeds), "episodesUrl": relative.removeprefix("site/"),
                "reward_windows": _reward_windows(episodes, warmup), "saved_target": row_map.get(name),
            })
        cohorts.append({"id": cohort_id, "label": label, "n": len(seeds), "seeds": seeds, "conditions": conditions})
        logging.info("Assembled %s: %d conditions, %d seeds", cohort_id, len(conditions), len(seeds))
        if cohort_id == "fresh_audit":
            fresh_saturation = _saturation(baseline, warmup)
            fresh_clipping = _contact_clipping(baseline, warmup)
    fitting = _load_condition(root, root / "runs/ant-classic-005/episodes/fit/baseline", manifest["seed_splits"]["fit"], registered, observations=True)
    selected = next(item for item in selection["conditions"] if item["vector"] == "lateral" and item["role"] == "candidate")
    metadata = {
        "title": "Ant world-Y velocity steering", "status": original["status"],
        "coordinate": "world-Y ground-plane velocity (m/s)",
        "estimand": "Equal-episode mean of available post-onset velocities; prefix-only episodes receive zero and remain failed pairs",
        "warmup": warmup, "dt": float(next(iter(fitting.values()))["dt"]),
        "featured": {"cohort": "confirmation", "condition": f"lateral__{selected['alpha']:+.4f}"},
        "reward_curve_definition": (
            "At endpoint t each displayed reward is the mean of the preceding block of up to 10 transitions; t=0 has no reward. "
            "Positions integrate original saved velocities at block endpoints. Curves stop at the recorded episode end; total_return sums all original rewards."
        ),
        "limitations": [
            "One frozen checkpoint; no evidence of training-seed or real-robot generalization",
            "Heading-related evidence; exact equivalence/unique mediation unestablished; lateral-label specificity is unsupported",
            "Original confirmation_failed is retained; later audit resets do not complete formal replication",
            "Fresh audit consumed the convention-reserved application cohort",
            "Additional publication intervals are descriptive and unadjusted; no new selection or simulation",
        ],
        "source_links": [
            {"label": "Original numerical results", "url": REPO_URL + "/blob/main/reports/ant-classic-005/results.json"},
            {"label": "Independent audit and preregistration", "url": REPO_URL + "/blob/main/reports/ant-classic-005-independent-audit.md"},
            {"label": "Pinned steering vectors", "url": "https://huggingface.co/BurnyCoder/rl-locomotion-steering-vectors/blob/1cd00cfcce13bcb86f1c370a71889ce6d34c278e/experiments/ant-classic-005/vectors.npz"},
            {"label": "Fixed-camera paired video", "url": "https://github.com/user-attachments/assets/9e0719a6-c15b-474e-b74e-3c5e70954e2a"},
        ],
    }
    evidence = {
        "schema_version": 1, "metadata": metadata,
        "provenance": {key: value for key, value in verified.items() if key != "files"},
        "cohorts": cohorts,
        "fitting": {"saturation": _saturation(fitting, warmup), "fresh_baseline_saturation": fresh_saturation},
        "audit": audit, "fresh_audit": fresh, "directions": directions,
        "original_results": original, "original_manifest": manifest, "selection": selection,
    }
    evidence["fitting"].update(contact_wrench_clipping=_contact_clipping(fitting, warmup), fresh_baseline_contact_wrench_clipping=fresh_clipping)
    evidence["metadata"]["source_links"].append({"label": "Pinned upstream Ant checkpoint", "url": "https://huggingface.co/farama-minari/Ant-v5-SAC-medium/tree/" + manifest["provenance"]["revision"]})
    if verified["revision"]:
        evidence["metadata"]["source_links"].append({"label": "Immutable published episode evidence", "url": "https://huggingface.co/" + verified["repo_id"] + "/tree/" + verified["revision"]})
    evidence["provenance"]["derived_evidence_sha256"] = hashlib.sha256(canonical_json(evidence)).hexdigest()
    return evidence, outputs

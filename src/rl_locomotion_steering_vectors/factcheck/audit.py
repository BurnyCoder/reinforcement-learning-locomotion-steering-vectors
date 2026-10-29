"""Compare independent arithmetic with every saved production condition.

Reads numeric NPZ members only (never unpickles checkpoint objects):
https://numpy.org/doc/1.26/reference/generated/numpy.load.html .
Hashes identify bytes, not the truth of the measurements inside them.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .numerical import METRICS, contrast, episode_summary, fitting, paired
from .secondary import ensemble_statistics, geometry, robustness

EXPERIMENTS = ("hc-classic-001", "hc-running-002", "hc-height-speed-003", "hc-height-speed-004", "ant-classic-005")


def read_json(path: Path):
    """Read an artifact; the separate output serializer rejects nonfinite public JSON values."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Stream large artifacts; do not hold episode archives or videos in memory to hash them."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arrays(path: Path, *, activation: bool = False) -> dict:
    """Load observations/actions for consistency checks and activations only for fitting."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(METRICS.values()) | {"seed", "length", "dt", "terminated", "truncated", "onset_hash", "actions", "observations"}
        if activation:
            keys.add("activations")
        return {k: z[k] for k in keys if k in z.files}


def condition(folder: Path, *, activation: bool = False) -> list[dict]:
    """Reject duplicate or misnamed resets instead of silently replacing one in a mapping."""
    output, identities = [], set()
    for path in sorted(folder.glob("*.npz")):
        item = arrays(path, activation=activation)
        identity = int(item["seed"])
        if identity in identities or path.stem != str(identity):
            raise ValueError(f"Duplicate/misnamed seed: {path}")
        identities.add(identity)
        output.append(item)
    if not output:
        raise ValueError(f"No saved episodes: {folder}")
    return output


def differences(actual, recorded, prefix="", *, atol=1e-10, rtol=1e-9, require_fields=False) -> list[dict]:
    """Compare shared fields recursively; absent historical fields are not fabricated checks."""
    if isinstance(actual, dict) and isinstance(recorded, dict):
        missing = [{"field": f"{prefix}/{key}", "error": "missing recorded field"} for key in sorted(actual.keys()-recorded.keys())] if require_fields else []
        return missing + [d for key in sorted(actual.keys() & recorded.keys()) for d in differences(actual[key], recorded[key], f"{prefix}/{key}", atol=atol, rtol=rtol, require_fields=require_fields)]
    if isinstance(actual, (list, tuple)) and isinstance(recorded, (list, tuple)):
        if len(actual) != len(recorded):
            return [{"field": prefix, "actual": actual, "recorded": recorded}]
        # Check trajectory columns in one NumPy reduction; report every mismatching index.
        if all(x is None or isinstance(x, (int, float, np.number)) for x in (*actual, *recorded)):
            a = np.array([0 if x is None else x for x in actual], dtype=float)
            b = np.array([0 if x is None else x for x in recorded], dtype=float)
            same = np.isclose(a, b, atol=atol, rtol=rtol)
            same &= np.array([x is None for x in actual]) == np.array([x is None for x in recorded])
            return [{"field": f"{prefix}/{i}", "actual": actual[i], "recorded": recorded[i]} for i in np.flatnonzero(~same)]
        return [d for i, (a, b) in enumerate(zip(actual, recorded)) for d in differences(a, b, f"{prefix}/{i}", atol=atol, rtol=rtol, require_fields=require_fields)]
    if isinstance(actual, (int, float, np.number)) and isinstance(recorded, (int, float)):
        equal = bool(np.isclose(actual, recorded, atol=atol, rtol=rtol))
    else:
        equal = actual == recorded
    return [] if equal else [{"field": prefix, "actual": actual, "recorded": recorded}]


def check_measurements(episodes: list[dict]) -> dict:
    """Check recorded effort and planar norm without running the policy or recomputing physics."""
    errors = {"effort_max_abs_error": 0., "planar_speed_max_abs_error": 0.}
    for item in episodes:
        if item['actions'].ndim != 2 or len(item['actions']) != int(item['length']) or not np.isfinite(item['actions']).all():
            raise ValueError('Invalid saved actions')
        effort = np.square(item["actions"]).mean(axis=1)
        planar = np.hypot(item["x_velocity"], item["y_velocity"])
        errors["effort_max_abs_error"] = max(errors["effort_max_abs_error"], float(np.max(np.abs(effort-item["effort"]))))
        errors["planar_speed_max_abs_error"] = max(errors["planar_speed_max_abs_error"], float(np.max(np.abs(planar-item["planar_speed"]))))
    for key, error in errors.items():
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f"Inconsistent saved measurement {key}: {error}")
    return errors


def prefix_equal(left: list[dict], right: list[dict], warmup: int) -> bool:
    """Check every available paired prefix, including episodes ending before onset."""
    baseline = {int(x["seed"]): x for x in left}
    for b in right:
        a = baseline[int(b["seed"])]
        n = min(warmup, int(a["length"]), int(b["length"]))
        if min(int(a["length"]), int(b["length"])) < warmup and int(a["length"]) != int(b["length"]):
            return False
        if any(not np.array_equal(a[k][:n], b[k][:n]) for k in (*METRICS.values(), "actions", "observations")):
            return False
    return True


def fitted(run: Path, manifest: dict) -> dict:
    """Check primary extraction, raw diagnostics and all saved vector/control norms and cosines."""
    if (run/"retarget.json").exists():
        source = run.parent / "hc-running-002"
        with np.load(source/"vectors.npz", allow_pickle=False) as z, np.load(run/"vectors.npz", allow_pickle=False) as target:
            matches = {k: bool(np.array_equal(target[k], z[k.replace("speed", "height", 1)])) for k in target.files}
        if not all(matches.values()):
            raise ValueError(f"Retargeted vector changed: {run.name}")
        return {"imported_from": "hc-running-002", "source_family": "height", "bitwise_import_matches": matches}
    episodes = condition(run/"episodes/fit/baseline", activation=True)
    config, saved = manifest["config"], read_json(run/"vector_diagnostics.json")
    table, windows, scale = fitting(episodes, warmup=config["warmup"], fraction=config.get("fit_min_speed_fraction", 0))
    output = {"windows": windows, "activation_rms": scale, "differences": differences(windows, saved["windows"], "windows"), "vectors": {}}
    output["differences"] += differences(scale, saved["activation_rms"], "activation_rms")
    with np.load(run/"vectors.npz", allow_pickle=False) as z:
        for name, detail in saved["behaviors"].items():
            raw, computed = contrast(table, name)
            if raw is None:
                output["vectors"][name] = {"supported": False, "contributors": computed}
                continue
            raw_norm = float(np.linalg.norm(raw))
            computed["raw_norm"] = raw_norm
            output["differences"] += differences(computed, detail, name)
            scaled = (raw*(scale/raw_norm)).astype(np.float32)
            item = dict(contributors=computed, reference_vector_max_abs_error=float(np.max(np.abs(z[name]-scaled))),
                        norm=float(np.linalg.norm(z[name].astype(float))), controls={})
            # Match the recorded random stream while independently rebuilding each estimator.
            order = ["speed", "effort", "height", "lateral", "turning"].index(name)
            rng = np.random.default_rng(np.random.SeedSequence([saved["seed"], order]))
            halves = np.array_split(rng.permutation(np.unique(table["episode"])), 2)
            partial = [contrast({k: v[np.isin(table["episode"], ids)] for k,v in table.items()}, name, minimum=1)[0] for ids in halves]
            if all(v is not None and np.linalg.norm(v) > 1e-12 for v in partial):
                cosine = float(partial[0]@partial[1]/(np.linalg.norm(partial[0])*np.linalg.norm(partial[1])))
                item["split_half_cosine"] = cosine
                output["differences"] += differences(cosine, detail.get("split_half_cosine"), name+"/split_half_cosine")
            candidates = {name: raw}
            for number in range(3):
                candidates[f"{name}_random{number}"] = rng.standard_normal(raw.shape)
            labels = table[name].copy()
            speed = table["speed"]
            groups = np.zeros(len(speed), int)
            if name == "effort":
                distinct = np.unique(speed)
                groups = np.searchsorted(distinct, speed) if len(distinct) <= 5 else np.searchsorted(np.unique(np.quantile(speed,[.2,.4,.6,.8])),speed,side="right")
            for group in np.unique(groups):
                mask = groups == group
                labels[mask] = rng.permutation(labels[mask])
            shuffled, _ = contrast(table, name, labels=labels)
            if shuffled is not None:
                candidates[name+"_shuffled"] = shuffled
            for key in z.files:
                if key == name or key.startswith(name+"_"):
                    v, primary = z[key].astype(float), z[name].astype(float)
                    item["controls"][key] = {"norm": float(np.linalg.norm(v)), "cosine": float(v@primary/(np.linalg.norm(v)*np.linalg.norm(primary)))}
                    if key in candidates:
                        reference = candidates[key] * (scale/np.linalg.norm(candidates[key]))
                        item["controls"][key]["reconstruction_max_abs_error"] = float(np.max(np.abs(v-reference.astype(np.float32))))
                        if item["controls"][key]["reconstruction_max_abs_error"] > 1e-7:
                            output["differences"].append({"field": key, "error": "vector reconstruction differs by more than 1e-7"})
            output["vectors"][name] = item
    if run.name == "hc-classic-001":
        filtered, stats_filtered, _ = fitting(episodes, warmup=config["warmup"], fraction=.5)
        sensitivity = {}
        for name in saved["behaviors"]:
            a, _ = contrast(table, name)
            b, details = contrast(filtered, name)
            sensitivity[name] = {"original_norm": float(np.linalg.norm(a)), "filtered_norm": float(np.linalg.norm(b)),
                                 "cosine": float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))), "filtered_contrast": details["contrast"]}
        output["half_median_sensitivity"] = {"windows": stats_filtered, "vectors": sensitivity}
    return output


def select(rows: list[dict]) -> list[dict]:
    """Recompute the frozen near-best selection from validation, never from held-out outcomes."""
    chosen = []
    for name in sorted({r["vector"] for r in rows}):
        family = [r for r in rows if r["vector"] == name]
        behavior = family[0]["behavior"]
        eligible = [r for r in family if r["effect"]["gate"]]
        if not eligible:
            if name == behavior:
                continue
            eligible = family
        maximum = max(abs(r["effect"]["effect"]) for r in eligible)
        best = min((r for r in eligible if abs(r["effect"]["effect"]) >= .95*maximum), key=lambda r: (abs(r["alpha"]), -abs(r["effect"]["effect"])))
        chosen.append({k: best[k] for k in ("vector", "behavior", "alpha")} | {"direction": best["effect"]["direction"], "role": "candidate" if name == behavior else "control"})
    families = {r["behavior"] for r in chosen if r["role"] == "candidate"}
    return sorted((r for r in chosen if r["behavior"] in families), key=lambda r: r["vector"])


def original_experiment(root: Path, name: str) -> dict:
    """Verify all saved validation/confirmation rows, raw measurements and selection."""
    run = root/"runs"/name
    manifest, result = read_json(run/"manifest.json"), read_json(run/"results.json")
    warmup = manifest["config"]["warmup"]
    output = {"status": result["status"], "fitting": fitted(run, manifest), "phases": {}, "differences": []}
    report_result = read_json(root/"reports"/name/"results.json")
    output["differences"] += differences(result, report_result, "run_vs_report", require_fields=True)
    for phase in ("validation", "confirmation", "replication"):
        if not result.get(phase):
            if list((run/'episodes'/phase).glob('*/*.npz')):
                raise ValueError(f'Unreported {phase} observations in {name}')
            continue
        baseline = condition(run/"episodes"/phase/"baseline")
        if sorted(int(x['seed']) for x in baseline) != sorted(manifest['seed_splits'][phase]):
            raise ValueError(f"Recorded {phase} differs from manifest seeds: {name}")
        check_measurements(baseline)
        expected={f"{r['vector']}__{r['alpha']:+.4f}" for r in result[phase]}|{'baseline'}
        actual={p.name for p in (run/'episodes'/phase).iterdir() if p.is_dir() and any(p.glob('*.npz'))}
        if actual!=expected:
            raise ValueError(f'Unreported or missing {phase} conditions in {name}: {actual^expected}')
        rows = []
        for row in result[phase]:
            folder = f"{row['vector']}__{row['alpha']:+.4f}"
            treated = condition(run/"episodes"/phase/folder)
            observed = paired(baseline, treated, row["behavior"], warmup=warmup, seed=row["effect"].get("bootstrap", {}).get("seed", 710 if phase == "validation" else 720))
            observed_row = {"vector": row["vector"], "behavior": row["behavior"], "alpha": row["alpha"], "effect": observed}
            rows.append(observed_row)
            output["differences"] += differences(observed, row["effect"], f"{phase}/{folder}")
            observed_row["prefixes_equal"] = prefix_equal(baseline, treated, warmup)
            observed_row["measurements"] = check_measurements(treated)
            if not observed_row["prefixes_equal"]:
                output["differences"].append({"field": f"{phase}/{folder}/prefix", "actual": False, "recorded": True})
        output["phases"][phase] = rows
        logging.info("Checked %s %s: %d conditions", name, phase, len(rows))
    selected = select(output["phases"].get("validation", []))
    output["selection"] = selected
    output["differences"] += differences(selected, sorted(result.get("selected", []), key=lambda x: x["vector"]), "selection")
    return output


def fresh_experiment(root: Path) -> dict:
    """Independently reconstruct every signed audit arm and random-direction comparison."""
    folder = root/"runs/ant-005-independent-audit/episodes/audit"
    artifact = read_json(root/"reports/ant-classic-005-independent-audit/audit.json")
    recorded = artifact["fresh"]
    base = condition(folder/"baseline")
    seed_order = [int(x["seed"]) for x in base]
    summary = lambda items: [episode_summary(e, 100) for e in items]
    b = summary(base)
    arms, delta, mismatches, geometry_results = {}, {}, [], {}
    geometry_arms = {'lateral__-0.1000':'primary_minus','heading_parallel__-0.1000':'heading_parallel_minus','perceived_heading__-0.160876rad':'perception_minus_delta'}
    for sub in sorted(p for p in folder.iterdir() if p.is_dir() and p.name != "baseline"):
        items = condition(sub)
        if [int(x["seed"]) for x in items] != seed_order:
            raise ValueError(f"Unexpected audit seeds in {sub}")
        rows = summary(items)
        if sub.name in geometry_arms:
            geometry_results[geometry_arms[sub.name]] = geometry(base, items)
        if sub.name == 'lateral__-0.1000':
            primary_gate = paired(base, items, 'lateral', warmup=100, seed=740)
        delta[sub.name] = np.array([r["lateral"]-a["lateral"] for r, a in zip(rows, b)])
        arms[sub.name] = {"effect": float(delta[sub.name].mean()), "failures": int(sum(r["failure"] for r in rows)),
                          "speed_retention": float(np.mean([r["speed"] for r in rows])/np.mean([r["speed"] for r in b])),
                          "prefixes_equal": prefix_equal(base, items, 100), "measurements": check_measurements(items)}
        if not arms[sub.name]["prefixes_equal"]:
            mismatches.append({"field": sub.name+"/prefix", "actual": False, "recorded": True})
    directions, odd_arrays = {}, {}
    for name in recorded["directions"]:
        lo, hi = (delta[f"{name}__{a:+.4f}"] for a in (-.1,.1))
        odd_arrays[name] = (hi-lo)/2
        directions[name] = {"odd": float(((hi-lo)/2).mean()), "even": float(((hi+lo)/2).mean()),
                            "minus_arm": float(lo.mean()), "plus_arm": float(hi.mean()),
                            "failures": [arms[f"{name}__{a:+.4f}"]["failures"] for a in (-.1,.1)]}
    mismatches += differences(directions, recorded["directions"], "directions")
    recomputed = ensemble_statistics(delta, arms, directions)
    recomputed['P1']['production_gate'] = {k:primary_gate[k] for k in recorded['P1']['production_gate']}
    recomputed['P1']['speed_retention'] = primary_gate['treated_means']['speed']/primary_gate['baseline_means']['speed']
    recomputed['P1']['failures'] = [int(sum(r['failure'] for r in b)),arms['lateral__-0.1000']['failures']]
    p1 = recomputed['P1']
    p1['falsified'] = bool(p1['bca_ci'][0]<=0<=p1['bca_ci'][1] or p1['mean']<.2)
    recomputed['K1'] = geometry_results
    mismatches += differences(recomputed, recorded, "fresh", require_fields=True)
    original = root/'runs/ant-classic-005/episodes/confirmation'
    before, after = condition(original/'baseline'), condition(original/'lateral__-0.1000')
    saved = dict(robustness=robustness(before,after),kinematics_confirmation=geometry(before,after))
    mismatches += differences(saved, artifact['saved'], 'saved', require_fields=True)
    logging.info("Checked audit: %d arms, %d directions, %d resets", len(arms), len(directions), len(base))
    return {"seeds": seed_order, "arms": arms, "directions": directions, "recomputed": recomputed, "saved": saved, "differences": mismatches,
            "rank_interpretation": "Monte Carlo comparison with the specified random-direction ensemble; exchangeability of the primary is an additional assumption for a calibrated null-hypothesis p-value."}

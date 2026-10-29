"""Reference arithmetic independent of production estimators and earlier auditors.

These deliberately separate calculations test the recorded experiment, not a new
policy. NumPy reductions and quantiles follow the pinned 1.26 API:
https://numpy.org/doc/1.26/reference/generated/numpy.quantile.html . Resampling
whole reset pairs follows https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats.bootstrap.html .
Pilot thresholds below are project definitions, not recommendations from those sources.
"""
from __future__ import annotations

import numpy as np

METRICS = {"speed": "x_velocity", "lateral": "y_velocity", "height": "height",
           "turning": "yaw_rate", "effort": "effort", "planar_speed": "planar_speed",
           "reward": "reward", "saturation": "saturation", "inversion": "inversion",
           "bad_contact": "bad_contact"}


def episode_summary(item: dict, warmup: int) -> dict:
    """Reduce available post-onset transitions; an absent segment has the historical zero convention."""
    length = int(item["length"])
    if length < 1 or warmup < 0 or not np.isfinite(float(item["dt"])) or float(item["dt"]) <= 0:
        raise ValueError("Invalid episode length, warmup or dt")
    means = {}
    for label, key in METRICS.items():
        values = np.asarray(item[key], dtype=np.float64)
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved {key} array")
        means[label] = float(np.mean(values[warmup:])) if length > warmup else 0.0
    means["prefix_only"] = float(length <= warmup)
    explicit = float(item.get("failure", 0))
    if not np.isfinite(explicit) or not 0 <= explicit <= 1:
        raise ValueError("Invalid explicit failure")
    means["failure"] = max(explicit, float(bool(item["terminated"]) or length <= warmup or
                                         means["inversion"] > .05 or means["bad_contact"] > .05))
    return means


def paired(baseline: list[dict], treated: list[dict], behavior: str, *, warmup: int, seed: int) -> dict:
    """Reconstruct the published paired percentile estimator and its separate pilot predicates."""
    maps = [{int(e["seed"]): e for e in arm} for arm in (baseline, treated)]
    if len(maps[0]) != len(baseline) or len(maps[1]) != len(treated) or set(maps[0]) != set(maps[1]) or len(maps[0]) < 2:
        raise ValueError("Expected identical unique seed sets")
    seeds = sorted(maps[0])
    for identity in seeds:
        a, b = (arm[identity] for arm in maps)
        if str(a["onset_hash"]) != str(b["onset_hash"]):
            raise ValueError("Unequal onset states")
    summaries = [[episode_summary(arm[s], warmup) for s in seeds] for arm in maps]
    means = [{k: float(np.mean([row[k] for row in arm])) for k in arm[0]} for arm in summaries]
    before, after = means
    delta = np.array([b[behavior] - a[behavior] for a, b in zip(*summaries)])
    draw = np.random.default_rng(seed).integers(len(seeds), size=(2000, len(seeds)))
    ci = np.quantile(np.mean(delta[draw], axis=1), [.025, .975])
    effect = float(np.mean(delta))
    quality = {k: after[k] - before[k] for k in ("inversion", "bad_contact", "failure")}
    quality_pass = max(quality.values()) <= .05 + 1e-12 and before["prefix_only"] == after["prefix_only"] == 0
    magnitude, reference = abs(effect), abs(before[behavior])
    if behavior == "speed":
        size, movement = magnitude >= .05 * reference, before["speed"] > 0 and after["speed"] >= .5 * before["speed"]
    elif behavior in ("height", "effort"):
        size = magnitude >= .03 if behavior == "height" else -effect >= .1 * reference
        tolerance = .1 if behavior == "height" else .05
        movement = before["speed"] > 0 and abs(after["speed"] - before["speed"]) <= tolerance * abs(before["speed"]) + 1e-12
    else:
        size = magnitude >= (.2 if behavior == "lateral" else .1)
        key = "speed" if behavior == "lateral" else "planar_speed"
        movement = before[key] > 0 and after[key] >= .8 * before[key]
    excludes = ci[1] < 0 if behavior == "effort" else ci[0] > 0 or ci[1] < 0
    return dict(effect=effect, ci95=ci.tolist(), n_pairs=len(seeds), seeds=seeds,
                baseline_means=before, treated_means=after, baseline_mean=before[behavior],
                treated_mean=after[behavior], relative_effect=effect/reference if reference > 1e-12 else None,
                quality_deltas=quality, quality_pass=bool(quality_pass), effect_size_pass=bool(size),
                locomotion_pass=bool(movement), ci_excludes_zero=bool(excludes),
                gate=bool(quality_pass and size and movement and excludes), direction=int(np.sign(effect)))


def fitting(episodes: list[dict], *, warmup: int, window: int = 100, fraction: float = 0) -> tuple:
    """Rebuild windows and the episode-balanced RMS directly from saved activations."""
    rows, tail = [], 0
    for item in episodes:
        n, hidden = int(item["length"]), np.asarray(item["activations"], dtype=float)
        if hidden.ndim != 2 or len(hidden) != n:
            raise ValueError("Activation shape differs from recorded length")
        tail += max(0, n-warmup) % window
        for start in range(warmup, n-window+1, window):
            h = hidden[start:start+window]
            row = {key: float(np.mean(item[value][start:start+window])) for key, value in METRICS.items()}
            row.update(episode=int(item["seed"]), start=start, activations=h.mean(0), energy=float(np.square(h).sum(1).mean()))
            row["finite"] = bool(np.isfinite(h).all() and all(np.isfinite(row[k]) for k in METRICS))
            row["healthy"] = row["finite"] and max(row["inversion"], row["bad_contact"]) <= .05
            rows.append(row)
    healthy = [r for r in rows if r["healthy"]]
    median = float(np.median([r["speed"] for r in healthy])) if healthy else None
    floor = fraction*median if fraction > 0 and median is not None else None
    kept = [r for r in healthy if floor is None or r["speed"] > floor]
    stats = dict(episodes=len(episodes), total_windows=len(rows), eligible_windows=len(kept),
                 eligible_episodes=len({r["episode"] for r in kept}),
                 excluded_nonfinite=sum(not r["finite"] for r in rows),
                 excluded_unhealthy=sum(r["finite"] and not r["healthy"] for r in rows),
                 excluded_slow_windows=len(healthy)-len(kept), discarded_tail_steps=tail,
                 minimum_speed_floor=floor, healthy_median_speed=median)
    groups = [[r for r in kept if r["episode"] == s] for s in sorted({r["episode"] for r in kept})]
    center = np.mean([np.mean([r["activations"] for r in g], 0) for g in groups], 0) if groups else 0
    energy = np.mean([np.mean([r["energy"] for r in g]) for g in groups]) if groups else 0
    scale = float(np.sqrt(max(0., energy - np.sum(np.square(center)))))
    table = {k: np.array([r[k] for r in kept]) for k in ("episode", "activations", *METRICS)}
    return table, stats, scale


def balanced(values: np.ndarray, episodes: np.ndarray, mask: np.ndarray):
    """One equal contribution per episode in the selected group."""
    return np.mean([values[mask & (episodes == s)].mean(0) for s in np.unique(episodes[mask])], 0)


def contrast(table: dict, behavior: str, *, labels=None, minimum: int = 8) -> tuple:
    """Implement the mathematical contrast with explicit grouping, independent of analysis.py."""
    y = table[behavior] if labels is None else labels
    ids, speed = table["episode"], table["speed"]
    bins = np.zeros(len(y), dtype=int)
    if behavior == "effort":
        distinct = np.unique(speed)
        bins = np.searchsorted(distinct, speed) if len(distinct) <= 5 else np.searchsorted(np.unique(np.quantile(speed, [.2,.4,.6,.8])), speed, side="right")
    hi_all, lo_all = np.zeros(len(y), bool), np.zeros(len(y), bool)
    differences, gaps, details = [], [], []
    for group in np.unique(bins):
        members = bins == group
        low, high = np.quantile(y[members], [.25, .75])
        if high-low <= 1e-8*max(1., float(np.abs(y[members]).mean())):
            continue
        hi, lo = members & (y >= high), members & (y <= low)
        hi_all |= hi
        lo_all |= lo
        differences.append(balanced(table["activations"], ids, hi) - balanced(table["activations"], ids, lo))
        gaps.append(float(balanced(y, ids, hi)-balanced(y, ids, lo)))
        details.append(dict(bin=int(group), low_cut=float(low), high_cut=float(high), low_windows=int(lo.sum()), high_windows=int(hi.sum()),
                            low_speed=float(balanced(speed,ids,lo)), high_speed=float(balanced(speed,ids,hi))))
    detail = dict(high_episodes=len(np.unique(ids[hi_all])), low_episodes=len(np.unique(ids[lo_all])),
                  high_windows=int(hi_all.sum()), low_windows=int(lo_all.sum()))
    if not differences or min(detail["high_episodes"], detail["low_episodes"]) < minimum:
        return None, detail
    return np.mean(differences, axis=0), dict(detail, contrast=float(np.mean(gaps)), bins=details)

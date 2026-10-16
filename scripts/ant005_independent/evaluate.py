"""Local: analyze the fresh-seed experiment exactly as pre-registered. Global: decide P1, P2, S1, S2, M1, M2, K1 and the exploratory E1.

Every statistic uses whole resets as units: treated-minus-baseline world-Y velocity per seed,
the original zero-summary convention, and the ±alpha odd/even decomposition.
"""
from __future__ import annotations  # Keep annotations lightweight.

import logging  # Timestamped progress.

import numpy as np  # Array statistics.
from scipy import stats  # Rank correlation.

from rl_locomotion_steering_vectors.analysis import paired_effect  # Production usefulness gates for the replication arm.
from rl_locomotion_steering_vectors.experiment import condition_name  # Production cache labels.
from rl_locomotion_steering_vectors.runtime import METRIC_NAMES, summarise_episode  # Production episode summaries.
from rl_locomotion_steering_vectors.specificity import holm, odd_even, paired_tests, rank_p_value, two_way_bootstrap  # Shared statistics.

from .common import ALPHA, AUDIT_RUN, BOOTSTRAP_SEED, FRESH_SEEDS, OTHER_LABEL, PRIMARY, WARMUP, load_condition  # Protocol.
from .fresh import PHASE, perception_condition  # Fresh folder names.
from .saved import METRIC_KEYS, kinematics, summarize  # Shared reductions.


class Fresh:
    """Local: cache per-seed summaries of fresh conditions. Global: one consistent source for every hypothesis."""

    def __init__(self):
        """Local: load the baseline. Global: every difference is paired against it."""
        self.rows = {"baseline": self._load("baseline")}  # condition → seed → summary.

    def _load(self, condition: str) -> dict[int, dict]:
        """Local: summarize one condition. Global: fail loudly if any pre-registered seed is missing."""
        episodes = load_condition(AUDIT_RUN, PHASE, condition, METRIC_KEYS)  # Metrics only.
        if sorted(episodes) != list(FRESH_SEEDS):
            raise RuntimeError(f"{condition}: expected seeds {FRESH_SEEDS[0]}–{FRESH_SEEDS[-1]}, found {len(episodes)}")  # Incomplete collection.
        return {seed: summarize(episode) for seed, episode in episodes.items()}  # Independent summaries.

    def diff(self, condition: str, key: str = "y_velocity") -> np.ndarray:
        """Local: treated-minus-baseline per seed. Global: the paired unit."""
        if condition not in self.rows:  # Lazy load.
            self.rows[condition] = self._load(condition)  # Cache.
        return np.array([self.rows[condition][s][key] - self.rows["baseline"][s][key] for s in FRESH_SEEDS])  # Seed order fixed.

    def parts(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Local: per-seed odd and even parts of one direction. Global: separate steering from generic disruption."""
        return odd_even(self.diff(condition_name(name, -ALPHA)), self.diff(condition_name(name, ALPHA)))  # (odd, even) per seed.

    def failures(self, condition: str) -> int:
        """Local: count failed resets. Global: physical cost of each arm."""
        self.diff(condition)  # Ensure loaded.
        return int(sum(self.rows[condition][s]["failure"] for s in FRESH_SEEDS))  # Failure count.


def _seed_bootstrap(function, arrays: list[np.ndarray], resamples: int = 10_000) -> list[float]:
    """Local: resample seeds jointly across arrays. Global: CIs for differences of |means| that share resets."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)  # Recorded seed.
    draws = rng.integers(0, len(arrays[0]), size=(resamples, len(arrays[0])))  # Joint seed indices.
    values = [function(*[a[idx] for a in arrays]) for idx in draws]  # Statistic per resample.
    return np.quantile(values, [0.025, 0.975]).tolist()  # Percentile interval.


def _primary(fresh: Fresh) -> dict:
    """Local: P1 replication. Global: does the locked condition move the path toward +Y on fresh resets?"""
    d = fresh.diff(condition_name(PRIMARY, -ALPHA))  # Locked arm.
    tests = paired_tests(d, BOOTSTRAP_SEED)  # Interval and randomization evidence.
    full = lambda condition: [summarise_episode(e, WARMUP) for e in load_condition(AUDIT_RUN, PHASE, condition, METRIC_NAMES).values()]  # Production summaries.
    gate = paired_effect(full("baseline"), full(condition_name(PRIMARY, -ALPHA)), "lateral", seed=740, warmup=WARMUP)  # Production gates.
    falsified = tests["bca_ci"][0] <= 0 <= tests["bca_ci"][1] or tests["mean"] < 0.2  # Pre-registered rule.
    return {**tests, "production_gate": {k: gate[k] for k in ("effect", "ci95", "gate", "effect_size_pass", "locomotion_pass", "quality_pass", "quality_deltas")},
            "speed_retention": gate["treated_means"]["speed"] / gate["baseline_means"]["speed"], "failures": [fresh.failures("baseline"), fresh.failures(condition_name(PRIMARY, -ALPHA))],
            "falsified": bool(falsified)}  # P1 result.


def _directions(fresh: Fresh, names: list[str]) -> dict:
    """Local: odd/even means and arm effects for many directions. Global: rank tests and figures."""
    out = {}  # name → summary.
    for name in names:  # Every direction.
        odd, even = fresh.parts(name)  # Per-seed parts.
        out[name] = {"odd": float(odd.mean()), "even": float(even.mean()), "minus_arm": float(fresh.diff(condition_name(name, -ALPHA)).mean()),
                     "plus_arm": float(fresh.diff(condition_name(name, ALPHA)).mean()),
                     "failures": [fresh.failures(condition_name(name, -ALPHA)), fresh.failures(condition_name(name, ALPHA))]}  # Summary.
    return out  # Table.


def analyze_fresh(directions: dict) -> dict:
    """Local: run every pre-registered analysis. Global: the fresh-seed verdicts."""
    fresh = Fresh()  # Baseline loaded.
    names = list(directions["vectors"])  # Every activation direction.
    table = _directions(fresh, names)  # Odd/even per direction.
    randoms = [n for n in names if n.startswith("random")]  # Isotropic null.
    shuffled = [n for n in names if n.startswith("shuffled")]  # Label-shuffle redraws.
    abs_odd = lambda group: np.array([abs(table[n]["odd"]) for n in group])  # |odd| per direction.
    primary_odd, _ = fresh.parts(PRIMARY)  # Per-seed odd parts of the primary.

    p1 = _primary(fresh)  # Replication.
    p2_p = rank_p_value(abs(table[PRIMARY]["odd"]), abs_odd(randoms))  # Specificity against isotropic directions.
    best_arm_p = rank_p_value(table[PRIMARY]["minus_arm"], np.array([max(table[n]["minus_arm"], table[n]["plus_arm"]) for n in randoms]))  # Locked arm vs each random's better arm.
    adjusted = holm([p1["sign_flip_p"], p2_p])  # Family-wise control over the two primary tests.
    p2 = {"primary_abs_odd": abs(table[PRIMARY]["odd"]), "primary_odd_tests": paired_tests(primary_odd, BOOTSTRAP_SEED),
          "random_abs_odd_quantiles": np.quantile(abs_odd(randoms), [0.5, 0.9, 0.95, 0.99]).tolist(), "rank_p": p2_p,
          "best_arm_rank_p": best_arm_p, "holm_adjusted": {"P1": float(adjusted[0]), "P2": float(adjusted[1])}, "falsified": bool(adjusted[1] > 0.05)}  # P2 result.

    turn_odd, _ = fresh.parts(OTHER_LABEL)  # Other-label vector.
    s1 = {"turning_abs_odd": abs(table[OTHER_LABEL]["odd"]),
          "turning_minus_primary_abs_odd_ci": _seed_bootstrap(lambda a, b: abs(a.mean()) - abs(b.mean()), [turn_odd, primary_odd]),
          "shuffled_abs_odd": abs_odd(shuffled).tolist(), "shuffled_rank_p": rank_p_value(abs(table[PRIMARY]["odd"]), abs_odd(shuffled))}  # Label specificity.
    s1["label_specific"] = bool(s1["turning_minus_primary_abs_odd_ci"][1] < 0 and s1["shuffled_rank_p"] <= 0.05)  # Pre-registered rule.

    odd_matrix = np.stack([fresh.parts(n)[0] for n in randoms])  # Directions × seeds.
    even_matrix = np.stack([fresh.parts(n)[1] for n in randoms])  # Directions × seeds.
    s2 = {"random_even": two_way_bootstrap(even_matrix, BOOTSTRAP_SEED), "random_odd_sanity": two_way_bootstrap(odd_matrix, BOOTSTRAP_SEED),
          "random_minus_arm_mean": float(np.mean([table[n]["minus_arm"] for n in randoms])), "random_plus_arm_mean": float(np.mean([table[n]["plus_arm"] for n in randoms]))}  # Generic drift.

    par, perp = fresh.parts("heading_parallel")[0], fresh.parts("heading_orthogonal")[0]  # Additive components' odd parts.
    m1 = {"parallel_odd": table["heading_parallel"]["odd"], "orthogonal_odd": table["heading_orthogonal"]["odd"],
          "additivity_gap_ci": _seed_bootstrap(lambda a, b, c: (a + b - c).mean(), [par, perp, primary_odd]),
          "orthogonal_renorm_abs_odd": abs(table["heading_orthogonal_renorm"]["odd"]),
          "orthogonal_renorm_rank_p": rank_p_value(abs(table["heading_orthogonal_renorm"]["odd"]), abs_odd(randoms)),
          "axis_renorm_odd": table["heading_axis_renorm"]["odd"], "axis_renorm_rank_p": rank_p_value(abs(table["heading_axis_renorm"]["odd"]), abs_odd(randoms))}  # Ablation.
    m1["falsified"] = bool(m1["orthogonal_renorm_rank_p"] <= 0.05 or np.sign(m1["parallel_odd"]) != np.sign(table[PRIMARY]["odd"]) or abs(m1["parallel_odd"]) < 0.5 * abs(table[PRIMARY]["odd"]))  # Pre-registered rule.

    delta = directions["delta_eq"]  # Matched perceived rotation.
    minus, plus = fresh.diff(perception_condition(-delta)), fresh.diff(perception_condition(delta))  # Perception arms.
    rotation_odd, rotation_even = odd_even(minus, plus)  # Odd part: (+δ − (−δ))/2.
    m2 = {"delta_eq_deg": float(np.degrees(delta)), "minus_delta_arm": paired_tests(minus, BOOTSTRAP_SEED), "plus_delta_arm": paired_tests(plus, BOOTSTRAP_SEED),
          "odd": paired_tests(rotation_odd, BOOTSTRAP_SEED), "even_mean": float(rotation_even.mean()), "odd_ratio_to_primary": float(rotation_odd.mean() / primary_odd.mean()),
          "failures": [fresh.failures(perception_condition(-delta)), fresh.failures(perception_condition(delta))]}  # Perception test.
    m2["verdict"] = "supported" if m2["odd"]["bca_ci"][1] < 0 else "falsified" if m2["odd"]["bca_ci"][0] > 0 else "inconclusive"  # Pre-registered rule.

    observed = ("observations", "x_velocity", "y_velocity")  # Arrays for kinematics.
    base_obs = load_condition(AUDIT_RUN, PHASE, "baseline", observed)  # Fresh baseline with observations.
    k1 = {label: kinematics(base_obs, load_condition(AUDIT_RUN, PHASE, condition, observed)) for label, condition in
          (("primary_minus", condition_name(PRIMARY, -ALPHA)), ("perception_minus_delta", perception_condition(-delta)), ("heading_parallel_minus", condition_name("heading_parallel", -ALPHA)))}  # Path decomposition.

    info = directions["info"]["directions"]  # Projections from build_directions.
    design = lambda group: np.column_stack([np.ones(len(group)), [info[n]["projection_g_yaw"] for n in group], [info[n]["projection_g_omega"] for n in group]])  # Intercept + two perceptual projections.
    y = np.array([table[n]["odd"] for n in randoms])  # Training targets.
    coef = np.linalg.lstsq(design(randoms), y, rcond=None)[0]  # OLS fit on random directions only.
    loo = np.array([design([n]) @ np.linalg.lstsq(np.delete(design(randoms), i, 0), np.delete(y, i), rcond=None)[0] for i, n in enumerate(randoms)]).ravel()  # Leave-one-out predictions.
    held_out = [PRIMARY, OTHER_LABEL, "heading_parallel", "heading_orthogonal", "heading_orthogonal_renorm", "heading_axis_renorm", *shuffled]  # Out-of-sample directions.
    e1 = {"coefficients": coef.tolist(), "loo_r2": float(1 - np.sum((y - loo) ** 2) / np.sum((y - y.mean()) ** 2)),
          "spearman_odd_vs_g_yaw_projection": float(stats.spearmanr(y, [info[n]["projection_g_yaw"] for n in randoms]).statistic),
          "held_out": {n: {"observed_odd": table[n]["odd"], "predicted_odd": float(design([n]) @ coef)} for n in held_out}}  # Exploratory.

    output = {"seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]], "P1": p1, "P2": p2, "S1": s1, "S2": s2, "M1": m1, "M2": m2, "K1": k1, "E1": e1, "directions": table}  # All fresh evidence.
    logging.info("fresh analysis %s", output)  # Full result in the log.
    return output  # For audit.json.

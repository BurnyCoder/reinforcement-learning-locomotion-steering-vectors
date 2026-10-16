"""Local: reanalyze the saved Ant 005 validation and confirmation episodes. Global: check every published number and its robustness.

Sources: SciPy trim_mean and pearsonr https://docs.scipy.org/doc/scipy-1.14.1/reference/stats.html ;
the recorded estimand and failure rule follow runtime.summarise_episode and analysis._summary.
"""
from __future__ import annotations  # Keep annotations lightweight.

import logging  # Timestamped progress.

import numpy as np  # Array statistics.
from scipy import stats  # Robust location estimates and correlation.

from rl_locomotion_steering_vectors.kinematics import pair_kinematics  # Ground-plane decomposition.
from rl_locomotion_steering_vectors.specificity import odd_even, paired_tests  # Shared audit statistics.
from rl_locomotion_steering_vectors.storage import load_arrays, load_json  # Safe readers.

from .common import BOOTSTRAP_SEED, SOURCE_RUN, WARMUP, failed, load_condition, post_onset_mean  # Shared protocol.

METRIC_KEYS = ("x_velocity", "y_velocity", "yaw_rate", "effort", "reward", "inversion", "bad_contact")  # Per-step arrays needed for summaries.
BEHAVIOR_METRIC = {"lateral": "y_velocity", "turning": "yaw_rate"}  # Registered behavior → measured descriptor.


def summarize(episode: dict) -> dict:
    """Local: reduce one saved episode. Global: independent re-derivation of the recorded episode summary."""
    row = {key: post_onset_mean(episode, key) for key in ("x_velocity", "y_velocity", "yaw_rate", "effort", "reward")}  # Recorded estimand.
    dt = float(episode.get("dt", 0.05))  # Physical step.
    length = int(episode["length"])  # Duration.
    row.update(failure=failed(episode), terminated=bool(episode["terminated"]), length=length, prefix_only=length <= WARMUP,
               episode_return=float(np.sum(episode["reward"])), y_displacement=float(np.sum(episode["y_velocity"][WARMUP:length]) * dt))  # Extra outcomes.
    return row  # One reset's outcome.


def phase_table(phase: str) -> dict[str, dict[int, dict]]:
    """Local: summarize every condition of one phase. Global: one table feeds effect checks and decompositions."""
    folders = sorted(p.name for p in (SOURCE_RUN / "episodes" / phase).iterdir() if p.is_dir())  # Baseline and treated conditions.
    return {name: {seed: summarize(ep) for seed, ep in load_condition(SOURCE_RUN, phase, name, METRIC_KEYS).items()} for name in folders}  # Nested table.


def differences(table: dict, condition: str, key: str) -> np.ndarray:
    """Local: treated-minus-baseline per seed. Global: the paired unit of every comparison."""
    seeds = sorted(table["baseline"])  # Stable seed order (matches paired_effect).
    return np.array([table[condition][s][key] - table["baseline"][s][key] for s in seeds])  # One difference per reset.


def _recorded_ci(diff: np.ndarray, seed: int) -> list[float]:
    """Local: replicate analysis.paired_effect's percentile bootstrap. Global: verify the reported intervals exactly."""
    indices = np.random.default_rng(seed).integers(0, len(diff), size=(2000, len(diff)))  # Same draws as the original.
    return np.quantile(diff[indices].mean(axis=1), [0.025, 0.975]).tolist()  # Same quantiles.


def _check_effects(table: dict, rows: list[dict], seed: int) -> dict:
    """Local: compare recomputed and saved effects. Global: catch any arithmetic or pairing error in results.json."""
    worst_effect = worst_ci = 0.0  # Largest discrepancies.
    for row in rows:  # Every saved condition.
        diff = differences(table, f"{row['vector']}__{row['alpha']:+.4f}", BEHAVIOR_METRIC[row["behavior"]])  # Independent recomputation.
        worst_effect = max(worst_effect, abs(float(diff.mean()) - row["effect"]["effect"]))  # Effect agreement.
        worst_ci = max(worst_ci, float(np.abs(np.array(_recorded_ci(diff, seed)) - np.array(row["effect"]["ci95"])).max()))  # Interval agreement.
    return {"conditions": len(rows), "max_abs_effect_diff": worst_effect, "max_abs_ci_diff": worst_ci}  # Evidence.


def _published_numbers(table: dict, confirmation: list[dict]) -> list[dict]:
    """Local: recompute each README number. Global: accuracy of the public claim."""
    effect = {row["vector"]: float(differences(table, f"{row['vector']}__{row['alpha']:+.4f}", "y_velocity").mean()) for row in confirmation}  # Selected arms.
    seeds = sorted(table["baseline"])  # Seed order.
    mean = lambda condition, key: float(np.mean([table[condition][s][key] for s in seeds]))  # Arm mean.
    count = lambda condition: int(sum(table[condition][s]["failure"] for s in seeds))  # Failure count.
    primary = "lateral__-0.1000"  # Locked primary arm.
    lateral_ci = _recorded_ci(differences(table, primary, "y_velocity"), 720)  # Recorded interval.
    randoms = [effect[f"lateral_random{i}"] for i in range(3)]  # Random controls.
    margins = [effect["lateral"] - value for value in (*randoms, effect["lateral_bias"])]  # Primary minus each control.
    vectors = load_arrays(SOURCE_RUN / "vectors.npz")  # For the alignment claim.
    cosine = float(vectors["lateral"] @ vectors["lateral_shuffled"] / np.linalg.norm(vectors["lateral"]) / np.linalg.norm(vectors["lateral_shuffled"]))  # Shuffled alignment.
    rows = [("world-Y effect (m/s)", "0.402", round(effect["lateral"], 3) == 0.402, effect["lateral"]),
            ("95% CI", "[0.308, 0.491]", [round(v, 3) for v in lateral_ci] == [0.308, 0.491], lateral_ci),
            ("forward-speed retention (%)", "96.83", round(100 * mean(primary, "x_velocity") / mean("baseline", "x_velocity"), 2) == 96.83, 100 * mean(primary, "x_velocity") / mean("baseline", "x_velocity")),
            ("failures baseline / treated", "4/30 and 4/30", (count("baseline"), count(primary)) == (4, 4), [count("baseline"), count(primary)]),
            ("mean squared action change (%)", "+17.84", round(100 * (mean(primary, "effort") / mean("baseline", "effort") - 1), 2) == 17.84, 100 * (mean(primary, "effort") / mean("baseline", "effort") - 1)),
            ("environment reward decreases (episode return)", "decreases", mean(primary, "episode_return") < mean("baseline", "episode_return"), [mean("baseline", "episode_return"), mean(primary, "episode_return")]),
            ("random-direction effects (m/s)", "+0.15 to +0.25", [round(min(randoms), 2), round(max(randoms), 2)] == [0.15, 0.25], randoms),
            ("primary minus randoms and bias (m/s)", "0.15–0.27", round(min(margins), 2) >= 0.15 and round(max(margins), 2) <= 0.27, margins),
            ("shuffled effect (m/s)", "+0.89", round(effect["lateral_shuffled"], 2) == 0.89, effect["lateral_shuffled"]),
            ("shuffled failures", "9/30", count("lateral_shuffled__-0.2000") == 9, count("lateral_shuffled__-0.2000")),
            ("shuffled–primary alignment", "60%", round(cosine, 2) == 0.60, cosine)]  # (claim, stated, matches, recomputed).
    return [{"claim": c, "stated": s, "matches": bool(m), "recomputed": r} for c, s, m, r in rows]  # JSON rows.


def _robustness(table: dict) -> dict:
    """Local: vary the estimand and test. Global: the effect must not hinge on zero-filling, failures or one pair."""
    seeds = sorted(table["baseline"])  # Seed order.
    arm, base = table["lateral__-0.1000"], table["baseline"]  # Primary pair of arms.
    d = differences(table, "lateral__-0.1000", "y_velocity")  # Recorded estimand.
    keep_prefix = np.array([not base[s]["prefix_only"] for s in seeds])  # Drop the shared step-34 pair.
    survive = np.array([not (base[s]["failure"] or arm[s]["failure"]) for s in seeds])  # Pairs with no failure in either arm.
    loo = [float(np.delete(d, i).mean()) for i in range(len(d))]  # Leave-one-out means.
    yb, yt = np.array([base[s]["y_velocity"] for s in seeds]), np.array([arm[s]["y_velocity"] for s in seeds])  # Arm values.
    return {"recorded": paired_tests(d, BOOTSTRAP_SEED), "excluding_prefix_only": paired_tests(d[keep_prefix], BOOTSTRAP_SEED),
            "both_arms_survive": paired_tests(d[survive], BOOTSTRAP_SEED), "y_displacement_m": paired_tests(differences(table, "lateral__-0.1000", "y_displacement"), BOOTSTRAP_SEED),
            "trimmed_mean_10pct": float(stats.trim_mean(d, 0.1)), "leave_one_out_range": [min(loo), max(loo)],
            "arm_correlation": {"all_pairs": float(stats.pearsonr(yb, yt).statistic), "excluding_prefix_only": float(stats.pearsonr(yb[keep_prefix], yt[keep_prefix]).statistic)},
            "variance_ratio_paired_vs_unpaired": float(d.var(ddof=1) / (yb.var(ddof=1) + yt.var(ddof=1)))}  # <1 means pairing helps.


def kinematics(baseline: dict, treated: dict) -> dict:
    """Local: aggregate pair kinematics. Global: rotated path versus sidestep versus continuing turn."""
    pairs = [pair_kinematics(baseline[s], treated[s], warmup=WARMUP) for s in sorted(baseline)]  # One decomposition per reset.
    usable = [p for p in pairs if p["common_steps"] >= 200]  # At least two full windows in both arms.
    keys = ("delta_y_velocity", "delta_travel_angle_deg", "delta_mean_yaw_deg", "delta_gait_lateral_velocity", "yaw_slope_deg_per_100_steps", "delta_y_displacement")  # Reported quantities.
    windows = []  # Per-window torso-yaw and world-Y changes.
    for index in range(9):  # Windows 100–199 … 900–999.
        yaw = [p["window_delta_yaw_deg"][index] for p in usable if len(p["window_delta_yaw_deg"]) > index]  # Pairs reaching this window.
        vy = [p["window_delta_y_velocity"][index] for p in usable if len(p["window_delta_y_velocity"]) > index]  # Same pairs.
        if len(yaw) >= 5:  # Enough pairs for an interval.
            windows.append({"start": WARMUP + 100 * index, "pairs": len(yaw), "delta_yaw_deg": paired_tests(yaw, BOOTSTRAP_SEED, 2000), "delta_y_velocity": paired_tests(vy, BOOTSTRAP_SEED, 2000)})  # Time course.
    return {"pairs": len(usable), **{key: paired_tests([p[key] for p in usable], BOOTSTRAP_SEED) for key in keys}, "windows": windows}  # Aggregates.


def _validation_decomposition(table: dict) -> dict:
    """Local: odd/even world-Y parts of every saved vector (n = 10, exploratory). Global: compare primary, controls and other label."""
    vectors = load_arrays(SOURCE_RUN / "vectors.npz")  # All ten saved directions.
    unit = lambda v: v / np.linalg.norm(v)  # Unit direction.
    out = {}  # vector → |alpha| → parts.
    for name in vectors:  # Every saved direction.
        out[name] = {"cosine_with_lateral": float(unit(vectors[name]) @ unit(vectors["lateral"]))}  # AGENTS.md: report each control's cosine.
        for alpha in (0.05, 0.1, 0.2, 0.5):  # Grid magnitudes.
            odd, even = odd_even(differences(table, f"{name}__{-alpha:+.4f}", "y_velocity"), differences(table, f"{name}__{alpha:+.4f}", "y_velocity"))  # Per-seed parts.
            out[name][str(alpha)] = {"odd": float(odd.mean()), "even": float(even.mean())}  # Means over ten seeds.
    return out  # Exploratory table.


def reanalyze_saved() -> dict:
    """Local: run all saved-data reanalyses. Global: decide whether the published numbers are accurate and robust."""
    results = load_json(SOURCE_RUN / "results.json")  # Saved analysis.
    validation, confirmation = phase_table("validation"), phase_table("confirmation")  # Independent summaries.
    observed = ("observations", "x_velocity", "y_velocity")  # Arrays needed for kinematics.
    output = {"validation_effects": _check_effects(validation, results["validation"], 710),
              "confirmation_effects": _check_effects(confirmation, results["confirmation"], 720),
              "published_numbers": _published_numbers(confirmation, results["confirmation"]), "robustness": _robustness(confirmation),
              "kinematics_confirmation": kinematics(load_condition(SOURCE_RUN, "confirmation", "baseline", observed), load_condition(SOURCE_RUN, "confirmation", "lateral__-0.1000", observed)),
              "validation_world_y_decomposition": _validation_decomposition(validation), "status": results["status"]}  # All saved-data evidence.
    logging.info("saved reanalysis %s", output)  # Full result in the log.
    return output  # For audit.json.

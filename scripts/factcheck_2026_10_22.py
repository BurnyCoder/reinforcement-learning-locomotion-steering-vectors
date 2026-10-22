"""Local: recompute the numbers newly cited by the 2026-10-22 fact-check. Global: make each documentation correction reproducible.

Usage (from the repository root, with the original runs/ directories present):
    uv run --no-sync python scripts/factcheck_2026_10_22.py

Reuses the production estimators (analysis.paired_effect, _summary, window_table, _contrast);
reads saved episodes only; never simulates, selects or changes artifacts.
Sources: https://numpy.org/doc/1.26/reference/random/generated/numpy.random.Generator.permutation.html (reshuffles),
https://numpy.org/doc/1.26/reference/generated/numpy.corrcoef.html (arm correlation).
"""
import json  # Read the saved 004 validation rows.
import logging  # Timestamped terminal and file output.
from pathlib import Path  # Repository-relative inputs.

import numpy as np  # Means, SDs and permutations.

from rl_locomotion_steering_vectors.analysis import _contrast, _speed_bins, _summary, paired_effect, window_table  # Production estimators.
from rl_locomotion_steering_vectors.storage import load_arrays, load_json, start_logging  # Safe NPZ/JSON readers and shared logging.

ANT = Path("runs/ant-classic-005/episodes/confirmation")  # Original Ant confirmation arms.
FRESH = Path("runs/ant-005-independent-audit/episodes/audit")  # Fresh-seed audit arms.


def arm(root: Path, name: str) -> list[dict]:
    """Local: load one arm in seed order. Global: keep pairs aligned by filename."""
    return [load_arrays(path) for path in sorted((root / name).glob("*.npz"))]  # Filenames are seeds, so sorting pairs the arms.


def control_margins() -> None:
    """Local: primary minus each locked control on shared seeds. Global: name the strength behind each margin."""
    primary = arm(ANT, "lateral__-0.1000")  # Locked primary.
    for path in sorted(ANT.iterdir()):
        if path.name not in ("baseline", "lateral__-0.1000"):
            effect = paired_effect(arm(ANT, path.name), primary, "lateral", seed=720)  # Same bootstrap seed as confirmation.
            logging.info("margin primary-minus %s = %.4f ci95=%s", path.name, effect["effect"], np.round(effect["ci95"], 3).tolist())


def pairing_and_rotation() -> None:
    """Local: SDs, arm correlation and rigid-rotation arithmetic on 30 and 29 pairs. Global: label every n."""
    base, steered = arm(ANT, "baseline"), arm(ANT, "lateral__-0.1000")  # Paired confirmation episodes.
    y0, y1 = (np.array([_summary(e)["lateral"] for e in group]) for group in (base, steered))  # Recorded world-Y summaries.
    displacement = lambda e, key: float(np.sum(e[key][100:int(e["length"])]) * float(e["dt"])) if int(e["length"]) > 100 else 0.0  # Post-onset metres.
    bx, by, tx, ty = (np.array([displacement(e, k) for e in g]) for g, k in ((base, "x_velocity"), (base, "y_velocity"), (steered, "x_velocity"), (steered, "y_velocity")))
    kept = np.array([int(e["length"]) > 100 for e in base])  # Drop the shared prefix-only pair for the 29-pair figures.
    angle = np.degrees(np.arctan2(ty, tx) - np.arctan2(by, bx))[kept].mean()  # Mean travel-angle change, 29 pairs.
    for label, mask in (("30", np.ones(len(base), bool)), ("29", kept)):
        d = (y1 - y0)[mask]  # Paired world-Y differences.
        logging.info("n=%s sd_base=%.3f sd_steered=%.3f sd_diff=%.4f r=%.3f base_x=%.1fm dY=%.2fm rigid_pred=%.2fm", label, y0[mask].std(ddof=1), y1[mask].std(ddof=1), d.std(ddof=1),
                     np.corrcoef(y0[mask], y1[mask])[0, 1], bx[mask].mean(), (ty - by)[mask].mean(), np.tan(np.radians(angle)) * bx[mask].mean())


def fresh_failures() -> None:
    """Local: failure counts and world-Y means of selected fresh-seed arms. Global: check audit statements about failures."""
    for name in ("baseline", "lateral__-0.1000", "heading_orthogonal__-0.1000", "heading_orthogonal__+0.1000", "heading_orthogonal_renorm__-0.1000", "heading_orthogonal_renorm__+0.1000", "perceived_heading__-0.160876rad", "perceived_heading__+0.160876rad"):
        rows = [_summary(e) for e in arm(FRESH, name)]  # Production per-episode summaries.
        logging.info("fresh %s failures=%d/%d mean_world_y=%.3f", name, int(sum(r["failure"] for r in rows)), len(rows), np.mean([r["lateral"] for r in rows]))


def max_speedup_004() -> None:
    """Local: largest positive relative speed change in 004 validation. Global: bound the 'about 0.4%' statement."""
    rows = sorted(((r["effect"]["relative_effect"], r["vector"], r["alpha"], r["effect"]["effect"], r["effect"]["ci95"]) for r in json.loads(Path("runs/hc-height-speed-004/validation.json").read_text()) if (r["effect"].get("relative_effect") or 0) > 0), reverse=True)
    logging.info("004 positive speed changes=%d largest=%s", len(rows), [(round(100 * a, 3), b, c, round(d, 4), np.round(e, 4).tolist()) for a, b, c, d, e in rows[:3]])


def reshuffle_null(draws: int = 300, seed: int = 77) -> None:
    """Local: mean |cos| of fresh label reshuffles with each primary. Global: check the 0.24-0.52 range with a different RNG."""
    for run, behaviors in {"ant-classic-005": ["lateral", "turning"], "hc-running-002": ["speed", "effort", "height"]}.items():
        config = load_json(Path(f"runs/{run}/manifest.json"))["config"]  # Actual fitting settings.
        table = window_table([load_arrays(p) for p in sorted(Path(f"runs/{run}/episodes/fit/baseline").glob("*.npz"))], warmup=config["warmup"], minimum_speed_fraction=config["fit_min_speed_fraction"])
        eligible = {k: v[table["eligible"]] for k, v in table.items() if isinstance(v, np.ndarray)}  # Same rows as fit_vectors.
        vectors = load_arrays(Path(f"runs/{run}/vectors.npz"))  # Saved primaries.
        for behavior in behaviors:
            unit, rng, cosines = vectors[behavior] / np.linalg.norm(vectors[behavior]), np.random.default_rng(seed), []
            labels = np.asarray(eligible[behavior], float)  # Original labels.
            bins = _speed_bins(eligible["speed"]) if behavior == "effort" else np.zeros(len(labels), int)  # Effort shuffles stay within speed strata.
            for _ in range(draws):
                shuffled = labels.copy()
                for group in np.unique(bins):
                    shuffled[bins == group] = rng.permutation(shuffled[bins == group])  # Within-stratum permutation.
                raw, _ = _contrast(eligible, behavior, shuffled)  # Production contrast rules.
                if raw is not None and np.linalg.norm(raw) > 1e-12:
                    cosines.append(abs(raw @ unit) / np.linalg.norm(raw))
            logging.info("reshuffle %s %s mean_abs_cos=%.3f draws=%d", run, behavior, float(np.mean(cosines)), len(cosines))


if __name__ == "__main__":
    start_logging(Path("reports/fact-check-2026-10-22"))  # Log beside the fact-check report.
    for phase in (control_margins, pairing_and_rotation, fresh_failures, max_speedup_004, reshuffle_null):
        logging.info("phase=%s", phase.__name__)  # Name each recomputation in the log.
        phase()

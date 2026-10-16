"""Local: build every audit direction from fitting data only. Global: fix the fresh experiment's treatments before any fresh episode.

Directions: the saved primary and other-label vectors; 99 isotropic random directions;
19 label-shuffle redraws from the production estimator; and the perceived-heading
decomposition of the primary. The observation-rotation arm uses delta_eq, the perceived
rotation whose linearized first-ReLU change matches the primary's projection on g.
"""
from __future__ import annotations  # Keep annotations lightweight.

import logging  # Timestamped progress.

import numpy as np  # Vector algebra.

from rl_locomotion_steering_vectors.analysis import fit_vectors  # Production estimator for shuffled redraws.
from rl_locomotion_steering_vectors.perception import feature_gradient, perceived_heading_gradient  # Perception axes in first-ReLU space.
from rl_locomotion_steering_vectors.specificity import decompose, random_directions, renormalize  # Direction builders.
from rl_locomotion_steering_vectors.storage import load_arrays, save_arrays, save_json  # Safe artifacts.

from .common import ALPHA, OTHER_LABEL, PRIMARY, RANDOM_COUNT, RANDOM_SEED, REPORT_DIR, SHUFFLE_SEEDS, SOURCE_RUN, WARMUP, load_condition, sha256_vector  # Protocol.

OMEGA_Z_INDEX, OMEGA_EPSILON = 18, 0.05  # obs[18] = torso angular velocity about its local z axis (qvel[5]); ±0.05 rad/s difference.


def build_directions(model) -> dict:
    """Local: construct and save all treatments. Global: deterministic, fitting-only inputs for the pre-registered experiment."""
    fit = list(load_condition(SOURCE_RUN, "fit", "baseline").values())  # Fitting split with observations and activations.
    states = np.concatenate([episode["observations"][WARMUP:int(episode["length"]):10] for episode in fit])  # Post-warmup fitting states, every tenth step.
    saved = load_arrays(SOURCE_RUN / "vectors.npz")  # Evaluated vectors.
    primary = saved[PRIMARY].astype(np.float64)  # Primary direction.
    norm = float(np.linalg.norm(primary))  # Fitting activation RMS (all saved vectors share it).
    g_yaw = perceived_heading_gradient(model, states)  # Activation change per radian of perceived +y rotation.
    g_omega = feature_gradient(model, states, OMEGA_Z_INDEX, OMEGA_EPSILON)  # Activation change per rad/s of perceived yaw rate.
    parallel, orthogonal = decompose(primary, g_yaw)  # Primary = heading component + remainder.
    directions = {PRIMARY: primary, OTHER_LABEL: saved[OTHER_LABEL].astype(np.float64),
                  "heading_parallel": parallel, "heading_orthogonal": orthogonal,
                  "heading_orthogonal_renorm": renormalize(orthogonal, norm), "heading_axis_renorm": renormalize(g_yaw, norm)}  # Labeled and mechanism directions.
    for index, vector in enumerate(random_directions(RANDOM_COUNT, primary.size, norm, RANDOM_SEED)):  # Isotropic null.
        directions[f"random{index:03d}"] = vector.astype(np.float64)  # Stored at full precision; runtime casts to float32.
    for seed in SHUFFLE_SEEDS:  # Label-shuffle redraws with the production estimator.
        refit, _ = fit_vectors(fit, [PRIMARY], seed=seed, warmup=WARMUP, minimum_speed_fraction=0.5)  # Same rules, different shuffle.
        if not np.array_equal(refit[PRIMARY], saved[PRIMARY]):
            raise RuntimeError("primary vector is not reproduced by the production estimator")  # Extraction must be deterministic.
        directions[f"shuffled{seed:02d}"] = refit[f"{PRIMARY}_shuffled"].astype(np.float64)  # Redraw.
    unit = lambda v: v / np.linalg.norm(v)  # Unit direction.
    delta_eq = float(ALPHA * (primary @ unit(g_yaw)) / np.linalg.norm(g_yaw))  # Perceived rotation matching α·(v·ĝ) under linearization.
    family = lambda name: "primary" if name == PRIMARY else "other_label" if name == OTHER_LABEL else "random" if name.startswith("random") else "shuffled" if name.startswith("shuffled") else "mechanism"  # Group label.
    summary = {name: {"family": family(name), "norm": float(np.linalg.norm(v)), "sha256_float32": sha256_vector(v),
                      "cosine_with_primary": float(unit(v) @ unit(primary)), "cosine_with_g_yaw": float(unit(v) @ unit(g_yaw)),
                      "cosine_with_g_omega": float(unit(v) @ unit(g_omega)), "projection_g_yaw": float(v @ unit(g_yaw)),
                      "projection_g_omega": float(v @ unit(g_omega))} for name, v in directions.items()}  # Per-direction provenance.
    info = {"activation_norm": norm, "fit_states": int(len(states)), "g_yaw_norm_per_rad": float(np.linalg.norm(g_yaw)),
            "g_omega_norm_per_rad_s": float(np.linalg.norm(g_omega)), "delta_eq_rad": delta_eq, "delta_eq_deg": float(np.degrees(delta_eq)),
            "random_seed": RANDOM_SEED, "shuffle_seeds": list(SHUFFLE_SEEDS), "alpha": ALPHA, "directions": summary}  # Pre-registration facts.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)  # Output folder.
    save_arrays(REPORT_DIR / "directions.npz", {**{name: v.astype(np.float32) for name, v in directions.items()}, "g_yaw": g_yaw, "g_omega": g_omega})  # Exact runtime arrays plus axes.
    save_json(REPORT_DIR / "directions.json", info)  # Human-readable provenance.
    logging.info("directions delta_eq=%.6f rad (%.3f deg) |g_yaw|=%.4f cos(primary,g_yaw)=%.4f", delta_eq, np.degrees(delta_eq), np.linalg.norm(g_yaw), summary[PRIMARY]["cosine_with_g_yaw"])  # Key facts.
    return {"vectors": {name: v.astype(np.float32) for name, v in directions.items()}, "delta_eq": delta_eq, "info": info}  # For collection and analysis.

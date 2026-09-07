"""Local: reusable rollout collection; global: pair and resume every intervention.

Sources: https://stable-baselines3.readthedocs.io/en/v2.4.1/guide/examples.html
https://numpy.org/doc/1.26/reference/generated/numpy.savez_compressed.html
"""
import logging  # Emit each complete episode result to both configured handlers.
from pathlib import Path  # Keep phase artifacts under the selected run directory.

import numpy as np  # Summaries use ordinary numerical reductions.

from .config import Config  # One immutable protocol controls every condition.
from .runtime import run_episode, summarise_episode  # Reuse the same simulator path for all phases.
from .storage import load_arrays, save_arrays  # Completed episodes are safe resumable units.


def collect_episodes(model, model_key: str, run_dir: Path, phase: str, seeds: list[int], config: Config,
                     *, condition: str = "baseline", vector=None, alpha: float = 0.0,
                     capture: bool = False, stochastic: bool = False, action_bias=None) -> list[dict]:
    """Local: collect or reload episodes; global: never omit failed or completed trials."""
    episodes = []  # Keep one record for every prespecified seed.
    for seed in seeds:  # Identical seed order makes progress and logs easy to audit.
        path = run_dir / "episodes" / phase / condition / f"{seed}.npz"  # Condition names are generated internally.
        if path.exists():  # Only completely written NPZ files count as finished.
            episode = load_arrays(path)  # Reuse the exact trajectory rather than resampling.
        else:
            episode = run_episode(model, model_key, seed, vector=vector, alpha=alpha, warmup=config.warmup,
                                  max_steps=config.max_steps, deterministic=not stochastic,
                                  capture=capture, action_bias=action_bias)  # All interventions share simulator semantics.
            save_arrays(path, episode)  # Persist before another seed can start.
            logging.info("episode phase=%s condition=%s seed=%s metrics=%s", phase, condition, seed,
                         summarise_episode(episode, config.warmup))  # Preserve measurements and failures in realtime.
        episodes.append(episode)  # Failure episodes remain in statistical denominators.
    return episodes  # Fitting uses arrays; evaluation immediately reduces to episode summaries.


def summaries(episodes: list[dict], config: Config) -> list[dict]:
    """Local: reduce trajectories once; global: use whole episodes as analysis units."""
    return [summarise_episode(episode, config.warmup) for episode in episodes]  # Respect the configured onset.


def diagnose(episodes: list[dict], config: Config) -> dict:
    """Local: report measured ranges; global: reveal missing variation before extraction."""
    rows = summaries(episodes, config)  # Startup samples do not define sustained behavior.
    fields = ("x_velocity", "y_velocity", "planar_speed", "height", "yaw_rate", "effort", "failure", "inversion", "bad_contact")  # Report physical and quality metrics together.
    metrics = {key: {"mean": float(np.mean([row[key] for row in rows])),
                     "min": float(np.min([row[key] for row in rows])),
                     "max": float(np.max([row[key] for row in rows])),
                     "std": float(np.std([row[key] for row in rows]))} for key in fields}  # Dispersion is descriptive, not causal evidence.
    return {"n_episodes": len(rows), "metrics": metrics, "episodes": rows}  # Keep outliers visible for diagnosis.

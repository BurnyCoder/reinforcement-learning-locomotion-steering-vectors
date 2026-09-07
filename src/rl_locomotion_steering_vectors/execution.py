"""Local: reusable rollout collection; global: pair and resume every intervention.

Sources: https://stable-baselines3.readthedocs.io/en/v2.4.1/guide/examples.html
https://numpy.org/doc/1.26/reference/generated/numpy.savez_compressed.html
"""
import logging  # Emit each complete episode result to both configured handlers.
import hashlib  # Stable content hashes identify interventions independently of their names.
from contextlib import contextmanager  # Scope the reusable process pool to a single CLI invocation.
from pathlib import Path  # Keep phase artifacts under the selected run directory.

import numpy as np  # Summaries use ordinary numerical reductions.

from .config import Config  # One immutable protocol controls every condition.
from .runtime import run_episode, summarise_episode, parameter_hash  # Reuse the same simulator path for all phases.
from .storage import load_arrays, save_arrays, ensure_identity  # Completed episodes are safe resumable units.

_pool = None  # Local: serial operation remains the default for direct library calls.


@contextmanager
def use_pool(pool):
    """Local: scope an inference backend; global: share persistent isolated workers across phases."""
    global _pool  # The CLI runs one experiment per parent process.
    previous, _pool = _pool, pool  # Preserve any enclosing program's execution context.
    try:
        yield  # Named scientific phases remain independent of process-management details.
    finally:
        _pool = previous  # Exceptions cannot leak the selected backend into later library calls.


def condition_identity(model, model_key: str, config: Config, vector, alpha: float, capture: bool, stochastic: bool, action_bias) -> dict:
    """Local: fingerprint condition inputs; global: reject stale trajectory reuse after a code/vector change."""
    digest = lambda value: None if value is None else hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()  # Hash the exact runtime float32 direction.
    return {"model_key": model_key, "policy_sha256": parameter_hash(model),
            "runtime_sha256": hashlib.sha256(Path(__file__).with_name("runtime.py").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
            "vector_sha256": digest(vector), "bias_sha256": digest(action_bias), "alpha": float(alpha),
            "warmup": config.warmup, "max_steps": config.max_steps, "capture": bool(capture), "stochastic": bool(stochastic)}  # Include all factors that determine a cached rollout.


def collect_episodes(model, model_key: str, run_dir: Path, phase: str, seeds: list[int], config: Config,
                     *, condition: str = "baseline", vector=None, alpha: float = 0.0,
                     capture: bool = False, stochastic: bool = False, action_bias=None) -> list[dict]:
    """Local: collect or reload episodes; global: never omit failed or completed trials."""
    folder = run_dir / "episodes" / phase / condition  # One folder corresponds to one intervention specification.
    ensure_identity(folder / "identity.json", condition_identity(model, model_key, config, vector, alpha, capture, stochastic, action_bias))  # Validate content before loading cached episodes.
    episodes = {}  # Keep one record for every prespecified seed independent of completion order.
    jobs = []  # Submit only missing episodes to preserve exact resumption.
    for seed in seeds:  # Identical seed order makes progress and logs easy to audit.
        path = run_dir / "episodes" / phase / condition / f"{seed}.npz"  # Condition names are generated internally.
        if path.exists():  # Only completely written NPZ files count as finished.
            episodes[seed] = load_arrays(path)  # Reuse the exact trajectory rather than resampling.
        else:
            jobs.append(dict(seed=seed, vector=vector, alpha=alpha, warmup=config.warmup, max_steps=config.max_steps,
                             deterministic=not stochastic, capture=capture, action_bias=action_bias))  # Each worker receives a complete isolated episode specification.
    completed = _pool.run(jobs) if _pool is not None else ((job["seed"], run_episode(model, model_key, **job)) for job in jobs)  # Serial and process paths call the exact same simulator function.
    for seed, episode in completed:  # Persist results as soon as a worker finishes.
        save_arrays(folder / f"{seed}.npz", episode)  # A later worker failure cannot erase completed evidence.
        logging.info("episode phase=%s condition=%s seed=%s metrics=%s", phase, condition, seed,
                     summarise_episode(episode, config.warmup))  # Parent logging records complete worker outputs in one file.
        episodes[seed] = episode  # Failed locomotion episodes remain in statistical denominators.
    return [episodes[seed] for seed in seeds]  # Restore the declared seed order after parallel execution.


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

"""Local: test process-isolated collection. Global: require parallel execution to preserve exact causal evidence.

Sources: https://docs.python.org/3.12/library/concurrent.futures.html and
https://docs.pytest.org/en/stable/how-to/tmp_path.html.
"""

from pathlib import Path  # Local: locate project-owned weights; global: avoid looking outside the workspace.

import numpy as np  # Local: compare complete trajectory arrays; global: catch subtle process-dependent changes.
import pytest  # Local: assert lifecycle/input failures; global: test process handling before implementation.
import torch  # Local: match single-thread worker arithmetic; global: isolate scheduling from numerical configuration.

from rl_locomotion_steering_vectors.parallel import RolloutPool  # Local: import the new context contract; global: exercise the production process wrapper.
from rl_locomotion_steering_vectors.runtime import prepare_model, run_episode  # Local: reuse the serial oracle; global: compare the actual checkpoint and simulator.


def test_pool_rejects_invalid_workers_and_unentered_execution():
    """Local: verify pool ownership rules. Global: prevent silently spawned or unbounded worker resources."""
    with pytest.raises(ValueError):  # Local: require explicit input validation; global: avoid obscure subprocess failures.
        RolloutPool("halfcheetah", Path.cwd(), workers=0)  # Local: pass an invalid count; global: reject no-worker execution.
    pool = RolloutPool("halfcheetah", Path.cwd(), workers=1)  # Local: configure without entering; global: keep resource acquisition explicit.
    with pytest.raises(RuntimeError):  # Local: enforce the context boundary; global: prevent leaked executors.
        list(pool.run([{"seed": 5}]))  # Local: request premature work; global: exercise lifecycle misuse.


def test_parallel_rollouts_match_serial_checkpoint_bitwise_and_reuse_pool():
    """Local: run deterministic and stochastic real trajectories in two workers. Global: verify pairing and hook isolation."""
    torch.set_num_threads(1)  # Local: match worker inference configuration; global: remove thread-count numerical differences.
    model, _ = prepare_model("halfcheetah", Path.cwd())  # Local: load the pinned real checkpoint; global: avoid toy-network compatibility claims.
    jobs = [  # Local: include different policy sampling modes; global: test deterministic evaluation and stochastic fitting.
        {"seed": 787, "max_steps": 14, "warmup": 4, "capture": True, "deterministic": True},  # Local: test a baseline prefix; global: preserve deterministic artifacts.
        {"seed": 788, "max_steps": 14, "warmup": 4, "capture": True, "deterministic": False, "vector": np.ones(256), "alpha": 0.2},  # Local: combine randomness and hooks; global: detect shared-state contamination.
    ]
    serial = {job["seed"]: run_episode(model, "halfcheetah", **job) for job in jobs}  # Local: collect the reference paths; global: establish behavior before parallelization.
    with RolloutPool("halfcheetah", Path.cwd(), workers=2) as pool:  # Local: launch isolated persistent workers; global: test the same lifetime used by the research pipeline.
        parallel = dict(pool.run(jobs))  # Local: accept out-of-order completions; global: pair results by episode identity.
        with pytest.raises(ValueError, match="max_steps"):  # Local: require worker failures to reach the parent; global: prevent silently dropped failed trials.
            list(pool.run([{"seed": 789, "max_steps": 0}]))  # Local: submit an invalid child request; global: exercise exception serialization.
        repeated = dict(pool.run(jobs[:1]))  # Local: reuse already loaded workers; global: verify cleanup between intervention conditions.
    for seed, expected in serial.items():  # Local: check both rollout modes; global: avoid validating only deterministic actions.
        assert parallel[seed].keys() == expected.keys()  # Local: compare artifact schemas; global: ensure no lost metadata.
        for name, value in expected.items():  # Local: include states, metrics, captures, and onset hash; global: require full reproducibility.
            assert np.array_equal(parallel[seed][name], value), (seed, name)  # Local: require bitwise equality; global: reject process-induced scientific changes.
    assert np.array_equal(repeated[787]["actions"], serial[787]["actions"])  # Local: compare the reused-worker result; global: show treatment hooks did not leak.
    with pytest.raises(RuntimeError):  # Local: verify shutdown; global: prevent using an executor after its owning run ends.
        list(pool.run(jobs))  # Local: attempt reuse after exit; global: enforce explicit process lifetimes.

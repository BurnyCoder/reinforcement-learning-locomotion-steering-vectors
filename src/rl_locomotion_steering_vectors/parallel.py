"""Local: reuse Python process pools for rollouts. Global: accelerate experiments without sharing mutable policy hooks.

Sources: https://docs.python.org/3.12/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
and https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods.
Each spawned worker owns one frozen checkpoint for its whole pool lifetime.
"""

from __future__ import annotations  # Local: defer annotation resolution; global: keep the wrapper's imports uncomplicated.

from concurrent.futures import ProcessPoolExecutor, as_completed  # Local: reuse standard scheduling/lifecycle handling; global: avoid bespoke worker infrastructure.
from contextlib import AbstractContextManager  # Local: expose explicit resource ownership; global: close workers when the research run finishes.
from multiprocessing import get_context  # Local: request isolated spawn semantics; global: support Windows without inherited mutable models.
from pathlib import Path  # Local: pass one project root to workers; global: keep checkpoint artifacts within the workspace.
from typing import Any, Iterator  # Local: describe complete numerical result mappings; global: preserve the runtime API without extra wrappers.

import torch  # Local: bound each worker's tensor threads; global: prevent CPU oversubscription.

from .runtime import MODEL_SPECS, prepare_model, run_episode  # Local: reuse the validated serial path; global: keep parallel and serial scientific semantics identical.

_MODEL = None  # Local: initialize a process-local checkpoint reference; global: avoid pickling or sharing the parent's hooked actor.
_MODEL_KEY = None  # Local: retain the worker's task identity; global: prevent accidental cross-environment execution.


def _initialize_worker(model_key: str, root: str) -> None:
    """Local: load one frozen policy per process. Global: amortize preparation across all conditions and phases."""
    global _MODEL, _MODEL_KEY  # Local: populate this process's private state; global: keep model ownership isolated.
    torch.set_num_threads(1)  # Local: use one intra-operation CPU thread; global: prevent each worker monopolizing all cores.
    torch.set_num_interop_threads(1)  # Local: use one inter-operation thread; global: bound total compute parallelism to worker count.
    _MODEL, _ = prepare_model(model_key, Path(root))  # Local: verify/load the pinned checkpoint once; global: retain identical provenance in every worker.
    _MODEL_KEY = model_key  # Local: bind subsequent jobs to this environment; global: protect policy/task compatibility.


def _run_job(job: dict[str, Any]) -> tuple[int, dict]:
    """Local: execute a picklable rollout request. Global: return complete evidence without sharing simulator or hook state."""
    if _MODEL is None or _MODEL_KEY is None:  # Local: verify process initialization; global: reject execution without a frozen policy.
        raise RuntimeError("Rollout worker has not been initialized")  # Local: expose a lifecycle defect; global: do not fabricate missing episodes.
    episode = run_episode(_MODEL, _MODEL_KEY, **job)  # Local: delegate every numerical detail to serial runtime; global: preserve seeds, interventions, metrics, and checks.
    return int(episode["seed"]), episode  # Local: attach stable completion identity; global: allow immediate persistence despite out-of-order finishing.


class RolloutPool(AbstractContextManager):
    """Local: own persistent isolated rollout workers. Global: accelerate a whole experiment while preserving exact episode pairing.

Call ``run`` repeatedly inside one ``with`` block. Jobs contain ``seed`` and
the keyword arguments accepted by ``run_episode``. Results arrive in completion
order; callers should save each immediately and restore their prescribed order
by seed. Parent collection code remains responsible for timestamped result logs.
"""

    def __init__(self, model_key: str, root: Path, workers: int = 4):
        """Local: validate configuration without spawning. Global: keep expensive process ownership explicit."""
        if not isinstance(workers, int) or not 1 <= workers <= 61:  # Local: respect the documented Windows worker bound; global: reject invalid scheduling configurations early.
            raise ValueError("workers must be an integer between 1 and 61")  # Local: explain portable limits; global: prevent obscure executor failures.
        if model_key not in MODEL_SPECS:  # Local: validate the requested checkpoint; global: fail in the parent before starting processes.
            raise ValueError(f"Unknown model key: {model_key}")  # Local: identify the bad input; global: prevent mismatched worker tasks.
        self.model_key, self.root, self.workers = model_key, Path(root).resolve(), workers  # Local: retain serializable configuration; global: bind all workers to one project/checkpoint.
        self.executor = None  # Local: defer spawning until context entry; global: make construction side-effect free.

    def __enter__(self):
        """Local: create a persistent executor. Global: isolate process state and amortize checkpoint setup."""
        if self.executor is not None:  # Local: detect accidental nested ownership; global: prevent orphaning an existing pool.
            raise RuntimeError("Rollout pool is already active")  # Local: explain the lifecycle error; global: enforce one owner per executor.
        self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"), initializer=_initialize_worker, initargs=(self.model_key, str(self.root)))  # Local: reuse standard spawned workers; global: prohibit shared policy hooks and Torch RNG state.
        return self  # Local: expose repeated batch submission; global: preserve one pool across the entire research run.

    def run(self, jobs: list[dict[str, Any]]) -> Iterator[tuple[int, dict]]:
        """Local: stream completed requested episodes. Global: retain all results/errors while allowing resumable persistence."""
        if self.executor is None:  # Local: require an active context; global: prevent implicit unowned process creation.
            raise RuntimeError("Use RolloutPool.run inside an active with block")  # Local: explain correct usage; global: make cleanup obligations explicit.
        seeds = [int(job["seed"]) for job in jobs]  # Local: require reset identity on every request; global: maintain pairing metadata.
        if len(seeds) != len(set(seeds)):  # Local: detect ambiguous same-batch identities; global: prevent caller dictionaries from silently losing an episode.
            raise ValueError("Each rollout batch must contain unique episode seeds")  # Local: expose duplicate requests; global: protect statistical denominators.
        futures = [self.executor.submit(_run_job, dict(job)) for job in jobs]  # Local: send immutable-by-serialization numerical requests; global: keep tensors/models owned by workers.
        try:  # Local: own all pending batch futures; global: cancel unneeded work if collection fails or is abandoned.
            for future in as_completed(futures):  # Local: yield the next finished episode; global: permit immediate durable storage.
                yield future.result()  # Local: return full arrays or propagate the worker exception; global: never silently omit failed jobs.
        finally:  # Local: clean up remaining requests on every exit; global: avoid continuing unused conditions after an error.
            for future in futures:  # Local: visit each submitted request; global: cover partial-consumption scenarios.
                future.cancel()  # Local: cancel only tasks that have not begun; global: let active runtime contexts close their hooks safely.

    def __exit__(self, exc_type, exc_value, traceback):
        """Local: finish active episodes and release workers. Global: prevent background compute from outliving its owning run."""
        if self.executor is not None:  # Local: release only acquired resources; global: support exception-safe cleanup.
            self.executor.shutdown(wait=True, cancel_futures=True)  # Local: finish in-flight cleanup and cancel queued jobs; global: avoid leaked processes or truncated artifacts.
            self.executor = None  # Local: mark the pool inactive; global: reject accidental submissions after shutdown.
        return False  # Local: propagate parent/child errors; global: leave failed research runs explicit.


def map_rollouts(model_key: str, root: Path, jobs: list[dict[str, Any]], workers: int = 4) -> Iterator[tuple[int, dict]]:
    """Local: provide a one-batch convenience interface. Global: reuse the same safe executor lifecycle for standalone collections."""
    with RolloutPool(model_key, root, workers=workers) as pool:  # Local: own workers for the complete iteration; global: guarantee cleanup on exhaustion or generator close.
        yield from pool.run(jobs)  # Local: forward streamed full results; global: preserve exactly the persistent-pool semantics.

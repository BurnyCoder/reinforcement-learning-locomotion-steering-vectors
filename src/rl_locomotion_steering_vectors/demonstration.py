"""Local: demonstrate a frozen speed target on fresh episodes. Global: turn replicated causal evidence into an auditable application.

Sources: https://imageio.readthedocs.io/en/stable/examples.html (streamed video),
https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html (frame labels), and
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html (scientific plots).
Target calibration and physical failure classification reuse the project's existing analysis.
"""

from __future__ import annotations  # Local: defer type evaluation; global: keep annotations independent of imported library versions.

import hashlib  # Local: identify the exact saved intervention; global: prevent application retuning beneath an existing specification.
import logging  # Local: report decisions and media progress; global: use the CLI's timestamped terminal/file handlers.
from contextlib import ExitStack  # Local: close streamed readers/writers reliably; global: avoid orphaned encoders after failures.
from itertools import zip_longest  # Local: retain the longer video when an episode ends early; global: do not hide failed short trajectories.
from pathlib import Path  # Local: place application artifacts in the run directory; global: preserve project containment.

import imageio.v2 as imageio  # Local: read/write one frame at a time; global: keep video memory bounded.
import matplotlib  # Local: select a headless plotting backend; global: support unattended local experiments.
import numpy as np  # Local: compare arrays and compute target errors; global: retain transparent numerical operations.
import torch  # Local: apply the recorded inference thread setting; global: keep application execution reproducible.
from PIL import Image, ImageDraw, ImageFont  # Local: label paired frames using bundled fonts; global: avoid external font files or video ambiguity.

matplotlib.use("Agg")  # Local: render plots without an interactive window; global: keep unattended reporting reliable.
from matplotlib import pyplot as plt  # Local: reuse standard scientific plotting; global: produce standalone shareable figures.

from .analysis import _summary, paired_effect  # Local: reuse episode-level physical failure and paired inference; global: avoid a competing health definition.
from .config import Config  # Local: restore the immutable run protocol; global: avoid current .env settings changing a completed experiment.
from .execution import collect_episodes, summaries  # Local: reuse identity-checked resumable collection; global: preserve every application episode.
from .experiment import condition_name  # Local: reuse condition directory naming; global: keep artifacts consistent across phases.
from .runtime import prepare_model, run_episode  # Local: use the verified frozen actor and simulator path; global: retain intervention semantics.
from .storage import load_arrays, load_json, save_arrays, save_json, utc_now  # Local: reuse atomic non-executable storage; global: preserve partial progress and provenance.


def fresh_application_seeds(manifest: dict, source_manifests=()) -> list[int]:
    """Local: reserve ten application resets outside all inherited phases. Global: prevent leakage after retargeting a source experiment."""
    start = int(manifest["config"]["seed_offset"]) + 40000  # Local: follow the fixed application seed convention; global: make the fresh cohort reproducible.
    seeds = list(range(start, start + 10))  # Local: reserve exactly ten episodes; global: fix the denominator before observing outcomes.
    used = {int(seed) for item in (manifest, *source_manifests) for group in item.get("seed_splits", {}).values() for seed in group}  # Local: include every current/source phase; global: protect all fitting and evaluation boundaries.
    overlap = sorted(set(seeds) & used)  # Local: identify reused reset identities; global: reject observed episodes as application evidence.
    if overlap:  # Local: handle a collision before collection; global: prevent accidental leakage.
        raise ValueError(f"Application seeds overlap earlier phases: {overlap}")  # Local: report exact offending resets; global: make the protocol defect actionable.
    return seeds  # Local: expose the locked cohort; global: let specification persistence precede simulation.


def assess_target(rows: list[dict], target: float, tolerance: float, *, warmup: int = 100) -> dict:
    """Local: score every episode against a fixed speed band. Global: require coverage and zero episode-level physical failures."""
    if not rows or not np.isfinite(target) or not np.isfinite(tolerance) or target <= 0 or tolerance <= 0:  # Local: reject empty or invalid calibration; global: prevent vacuous success claims.
        raise ValueError("Require episodes and finite positive speed target/tolerance")  # Local: describe the acceptance inputs; global: fail before statistical interpretation.
    episodes = []  # Local: retain every target error and failure; global: prevent selective reporting.
    for row in rows:  # Local: score each independent reset; global: keep the analysis unit as an episode.
        measured = _summary(row, warmup)  # Local: reuse the current analysis failure criterion; global: detect isolated collapses before pooling.
        error = measured["speed"] - target  # Local: compute signed physical-unit error; global: compare to the validation-derived command.
        episodes.append(dict(seed=int(row["seed"]), speed=measured["speed"], error=error, absolute_error=abs(error),  # Local: preserve individual measurements; global: expose variation and misses.
                             within_tolerance=bool(abs(error) <= tolerance), physical_failure=bool(measured["failure"])))  # Local: separate targeting from competence; global: avoid one successful mean hiding a failure.
    fraction = float(np.mean([row["within_tolerance"] for row in episodes]))  # Local: divide by all episodes; global: retain misses in the coverage denominator.
    failed = [row["seed"] for row in episodes if row["physical_failure"]]  # Local: identify failed resets; global: make zero-failure acceptance explicit.
    return dict(target=float(target), tolerance=float(tolerance), episodes=episodes,  # Local: publish the fixed band and per-seed errors; global: support independent interpretation.
                mean_absolute_error=float(np.mean([row["absolute_error"] for row in episodes])),  # Local: average all errors; global: report misses even if 80% coverage passes.
                success_fraction=fraction, failed_episode_seeds=failed, gate=bool(fraction >= 0.8 and not failed))  # Local: implement the prespecified coverage/quality conjunction; global: prohibit post-hoc relaxation.


def _assert_episode_equal(expected: dict, actual: dict) -> None:
    """Local: verify replayed video data against numeric evidence. Global: prevent showing a different trajectory from the reported application."""
    if expected.keys() != actual.keys():  # Local: require identical artifact fields; global: catch incomplete or incompatible replays.
        raise ValueError("Video replay artifact fields differ from numeric application episode")  # Local: explain the failed audit; global: reject misleading media.
    for name in expected:  # Local: inspect all arrays and metadata; global: include onset hashes and hidden numerical changes.
        left, right = expected[name], actual[name]  # Local: align each recorded field; global: compare the same causal evidence.
        if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():  # Local: require byte-for-byte identity; global: avoid tolerance-masked replay drift.
            raise ValueError(f"Video replay differs from numeric episode in {name}")  # Local: name the mismatching field; global: stop before publishing a false visual correspondence.


def _paired_video(left_path: Path, right_path: Path, output_path: Path, seed: int, alpha: float) -> None:
    """Local: stream labeled baseline/steered frames side by side. Global: make the same-seed comparison visually inspectable without retaining a movie in RAM."""
    font = ImageFont.load_default(size=18)  # Local: use Pillow's bundled scalable font; global: avoid machine-specific font dependencies.
    with ExitStack() as stack:  # Local: own decoder/encoder cleanup; global: preserve resource safety on media errors.
        left = stack.enter_context(imageio.get_reader(str(left_path)))  # Local: open the baseline stream; global: read one frame at a time.
        right = stack.enter_context(imageio.get_reader(str(right_path)))  # Local: open the steered stream; global: keep the comparison temporally aligned.
        fps = float(left.get_meta_data()["fps"])  # Local: use the recorded simulation video rate; global: preserve physical elapsed time.
        if fps != float(right.get_meta_data()["fps"]):  # Local: reject mismatched stream timing; global: avoid false visual synchronization.
            raise ValueError("Paired videos have different frame rates")  # Local: expose the media incompatibility; global: protect comparison accuracy.
        writer = stack.enter_context(imageio.get_writer(str(output_path), fps=fps, codec="libx264", macro_block_size=1))  # Local: encode each composed frame immediately; global: bound memory independently of episode length.
        previous = [None, None]  # Local: retain only each side's most recent frame; global: show early-ended episodes rather than truncating the longer side.
        for index, pair in enumerate(zip_longest(left, right)):  # Local: include every frame from either condition; global: preserve failed short episodes in visual evidence.
            ended = [frame is None for frame in pair]  # Local: flag missing post-termination frames; global: distinguish frozen final frames from ongoing simulation.
            previous = [old if frame is None else frame for old, frame in zip(previous, pair)]  # Local: retain the last valid frame; global: avoid loading additional frame history.
            if any(frame is None for frame in previous):  # Local: detect an empty input movie; global: do not invent absent evidence.
                raise ValueError("Cannot compose an empty episode video")  # Local: explain failed rendering; global: leave the numerical result available for diagnosis.
            width = sum(frame.shape[1] for frame in previous)  # Local: preserve each native frame width; global: avoid distorting the agent's proportions.
            canvas = Image.new("RGB", (width, max(frame.shape[0] for frame in previous) + 36), "black")  # Local: allocate one labeled output frame; global: keep memory constant.
            drawing, offset = ImageDraw.Draw(canvas), 0  # Local: create the caption surface; global: associate each panel with its condition.
            for side, frame in enumerate(previous):  # Local: compose baseline then treatment; global: use a stable comparison layout.
                canvas.paste(Image.fromarray(frame), (offset, 36))  # Local: copy the current panel beneath its caption; global: preserve raw simulation imagery.
                label = "Baseline" if side == 0 else f"Steered a={alpha:+.3g}"  # Local: identify the experimental condition; global: prevent unlabeled intervention videos.
                label += f" | seed {seed} | {index / fps:.2f}s" + (" | ENDED" if ended[side] else "")  # Local: show reset/time/end status; global: make pairing and premature termination visible.
                drawing.text((offset + 8, 7), label, fill="white", font=font)  # Local: render a readable caption; global: carry context into exported media.
                offset += frame.shape[1]  # Local: advance to the next panel; global: preserve side-by-side alignment.
            writer.append_data(np.asarray(canvas))  # Local: stream the composed frame; global: avoid accumulating videos in memory.


def _series_plot(path: Path, baseline: dict, steered: dict, switched: dict, target: float, tolerance: float, config: Config) -> None:
    """Local: plot measured speed and height through switching. Global: show target control and distinguish policy restoration from trajectory reversal."""
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # Local: give speed/posture separate physical scales; global: create a readable standalone research figure.
    for label, episode in (("Baseline", baseline), ("Constant steering", steered), ("Off / on / off", switched)):  # Local: compare three matched-prefix conditions; global: expose the intervention's temporal behavior.
        dt = float(episode["dt"])  # Local: use actual simulation timing; global: avoid assuming a frame rate in numerical plots.
        width = max(1, round(1 / dt))  # Local: smooth over one second; global: reveal sustained behavior beyond gait oscillation.
        for axis, field in zip(axes, ("x_velocity", "height")):  # Local: plot only the requested physical outcomes; global: make speed/posture interactions visible.
            values = episode[field]  # Local: use preserved raw measurements; global: keep the plot traceable to NPZ evidence.
            window = min(width, len(values))  # Local: permit shorter failed episodes; global: do not drop early termination from figures.
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")  # Local: compute an explicit trailing mean; global: avoid hidden model-based smoothing.
            axis.plot(np.arange(window, len(values) + 1) * dt, smoothed, label=label, linewidth=1.5)  # Local: align each average at its final timestamp; global: preserve causal timing.
    axes[0].axhspan(target - tolerance, target + tolerance, color="gray", alpha=0.15, label="Locked target band")  # Local: show the predeclared acceptance region; global: make application errors inspectable.
    axes[0].axhline(target, color="gray", linestyle=":", linewidth=1)  # Local: mark the calibrated target; global: separate the command from measured responses.
    for axis in axes:  # Local: share intervention timing annotations; global: connect both physical outcomes to the same schedule.
        axis.axvline(config.warmup * float(baseline["dt"]), color="black", linestyle="--", linewidth=0.8)  # Local: mark the common treatment onset; global: display the matched-prefix boundary.
        axis.axvline(600 * float(baseline["dt"]), color="black", linestyle="--", linewidth=0.8)  # Local: mark off/on/off removal; global: distinguish constant and switched trajectories.
        axis.grid(alpha=0.2)  # Local: aid numerical reading; global: preserve a simple scientific presentation.
    axes[0].set_ylabel("Forward speed (m/s)")  # Local: state the target's physical unit; global: avoid an abstract activation-only presentation.
    axes[1].set_ylabel("Torso height (m)")  # Local: state the posture unit; global: reveal secondary behavior changes.
    axes[1].set_xlabel("Simulation time (s)")  # Local: label elapsed physical time; global: make onset and removal interpretable.
    axes[0].legend(loc="best", fontsize=8)  # Local: identify all traces/band; global: keep the exported figure self-contained.
    figure.suptitle(f"Fresh application seed {int(baseline['seed'])} | one-second trailing means")  # Local: disclose seed and smoothing; global: prevent overinterpreting a selected trace.
    figure.tight_layout()  # Local: fit annotations cleanly; global: avoid clipped output labels.
    figure.savefig(path, dpi=160)  # Local: export a portable raster chart; global: support direct report embedding.
    figure.savefig(path.with_suffix(".pdf"))  # Local: retain a vector-format plot; global: support the final research paper.
    plt.close(figure)  # Local: release plotting resources; global: keep repeated reporting memory bounded.


def _media(model, model_key: str, run_dir: Path, baseline: list[dict], steered: list[dict], vector, spec: dict, config: Config) -> dict:
    """Local: render prespecified paired replays and switching evidence. Global: audit that exported media depicts the numeric application episodes."""
    folder = run_dir / "application-media"  # Local: group this application evidence; global: keep artifacts attached to their immutable specification.
    folder.mkdir(parents=True, exist_ok=True)  # Local: create the media destination; global: avoid scattering generated outputs.
    videos, equality = [], []  # Local: collect media provenance; global: retain explicit numeric/video audit results.
    for original, treated in zip(baseline[:3], steered[:3]):  # Local: use the first three prespecified seeds; global: prohibit cherry-picking attractive examples.
        seed = int(original["seed"])  # Local: retain the paired reset identity; global: label exported comparisons accurately.
        left, right = folder / f"seed-{seed}-baseline.mp4", folder / f"seed-{seed}-steered.mp4"  # Local: preserve individual replays too; global: permit independent inspection.
        for path, expected, direction, alpha in ((left, original, None, 0.0), (right, treated, vector, spec["alpha"])):  # Local: replay both exact conditions; global: preserve matched experimental semantics.
            replay = run_episode(model, model_key, seed, vector=direction, alpha=alpha, warmup=config.warmup, max_steps=config.max_steps, deterministic=True, video_path=path)  # Local: reuse the verified streaming renderer; global: show normal runtime dynamics.
            _assert_episode_equal(expected, replay)  # Local: require the video's numeric trajectory to match the stored episode; global: prohibit misleading media substitutions.
        combined = folder / f"seed-{seed}-paired.mp4"  # Local: name the shareable comparison; global: keep a clear mapping to its seed.
        _paired_video(left, right, combined, seed, spec["alpha"])  # Local: label and stream both conditions together; global: support direct human comparison.
        videos.append({"seed": seed, "baseline": str(left.relative_to(run_dir)), "steered": str(right.relative_to(run_dir)), "paired": str(combined.relative_to(run_dir))})  # Local: record portable paths; global: retain provenance when publishing the run.
        equality.append({"seed": seed, "all_replay_arrays_bitwise_equal": True})  # Local: persist successful audits; global: make visual correspondence reviewable.
        logging.info("application paired video completed seed=%s path=%s", seed, combined)  # Local: report rendering progress; global: keep long-running media work visible.
    seed = int(baseline[0]["seed"])  # Local: choose the first application seed for switching; global: avoid selection using switching outcomes.
    probe = baseline[0]["observations"][:4]  # Local: retain fixed physical observation inputs; global: distinguish policy-function restoration from physical trajectory recovery.
    original_actions = model.predict(probe, deterministic=True)[0].copy()  # Local: snapshot the unhooked policy function; global: provide a restoration oracle.
    off_path = folder / f"seed-{seed}-off-on-off.mp4"  # Local: name the switching demonstration; global: associate it with the same frozen intervention.
    switched = run_episode(model, model_key, seed, vector=vector, alpha=spec["alpha"], warmup=config.warmup, max_steps=config.max_steps, deterministic=True, off_at=600, video_path=off_path)  # Local: remove steering at the fixed step; global: demonstrate reversible hook control.
    if not np.array_equal(original_actions, model.predict(probe, deterministic=True)[0]):  # Local: compare the original function at identical inputs; global: detect hook leakage after removal.
        raise RuntimeError("Policy function was not restored after the switching demonstration")  # Local: fail the restoration claim; global: preserve trustworthy causal evidence.
    if str(switched["onset_hash"]) != str(baseline[0]["onset_hash"]) or not np.array_equal(switched["actions"][:600], steered[0]["actions"][:600]):  # Local: verify onset and pre-removal identity; global: isolate the scheduled removal as the trajectory difference.
        raise ValueError("Switching demonstration differs before its prescribed intervention change")  # Local: expose a causal-pairing defect; global: reject confounded switching evidence.
    arrays_path = run_dir / "episodes" / "application" / "off-on-off" / f"{seed}.npz"  # Local: retain switching measurements; global: make the plot and replay independently reproducible.
    if arrays_path.exists():  # Local: preserve any previously completed switching episode; global: enforce deterministic resumption.
        _assert_episode_equal(load_arrays(arrays_path), switched)  # Local: verify the rerendered result; global: avoid silently replacing earlier switching data.
    else:  # Local: store the first verified switching episode; global: support later reporting without rerunning.
        save_arrays(arrays_path, switched)  # Local: write atomic non-executable arrays; global: retain full numerical evidence.
    plot = folder / "speed-height-timeseries.png"  # Local: give the figure a stable path; global: simplify report embedding.
    _series_plot(plot, baseline[0], steered[0], switched, spec["target"], spec["tolerance"], config)  # Local: compare measured trajectories; global: show both the requested outcome and posture side effects.
    return dict(video_paths=videos, video_replay_audits=equality, off_on_off_video=str(off_path.relative_to(run_dir)),  # Local: expose all media outputs; global: support publishing without guessed filenames.
                plot_path=str(plot.relative_to(run_dir)), policy_function_restored=True, videos_complete=True)  # Local: record the verified restoration result; global: distinguish function identity from a physical reset.


def demonstrate(run_dir: Path) -> dict:
    """Local: execute a locked speed-target application. Global: require fresh practical evidence after independent causal replication."""
    run_dir = Path(run_dir).resolve()  # Local: canonicalize the chosen experiment directory; global: anchor every output to one run.
    manifest, results = load_json(run_dir / "manifest.json"), load_json(run_dir / "results.json")  # Local: load completed experimental evidence; global: avoid relying on conversational success claims.
    candidates = [row for row in results.get("replicated", []) if row.get("role") == "candidate" and row["behavior"] == "speed"]  # Local: require an actually replicated speed candidate; global: exclude unconfirmed or control directions.
    if not candidates:  # Local: make unsupported behavior/success states explicit; global: do not invent an application from incomplete research.
        raise ValueError("Demonstration currently requires a replicated speed candidate")  # Local: explain the supported application; global: prevent silent fallback to another target.
    chosen = min(candidates, key=lambda row: (abs(row["alpha"]), row["vector"]))  # Local: choose deterministically before new data; global: avoid optimizing on application outcomes.
    validation = load_json(run_dir / "validation.json") if (run_dir / "validation.json").exists() else results["validation"]  # Local: reuse the saved calibration evidence; global: forbid target estimation from fresh episodes.
    matches = [row for row in validation if row["vector"] == chosen["vector"] and row["alpha"] == chosen["alpha"] and row["behavior"] == "speed"]  # Local: identify the exact validated treatment; global: prevent mixing signs or extraction aliases.
    if len(matches) != 1:  # Local: require unambiguous calibration; global: avoid guessing which target belongs to the replicated condition.
        raise ValueError("Replicated candidate must have exactly one matching validation result")  # Local: expose incomplete provenance; global: fail before collecting application data.
    effect = matches[0]["effect"]  # Local: access the matched validation outcome; global: lock the intended behavioral target.
    config = Config(**manifest["config"])  # Local: restore the saved protocol; global: ignore later .env changes.
    if not 0 <= config.warmup < 600 < config.max_steps:  # Local: require a real off/on/off schedule; global: avoid misleading removal demonstrations on too-short episodes.
        raise ValueError("Application media requires warmup < 600 < max_steps")  # Local: explain the prescribed schedule; global: reject unsupported shortened demonstrations.
    sources = [load_json(run_dir / "retarget.json")["source_manifest"]] if (run_dir / "retarget.json").exists() else []  # Local: include retargeting provenance snapshots; global: protect inherited evaluation seed boundaries.
    seeds = fresh_application_seeds(manifest, sources)  # Local: verify the ten reserved resets; global: keep all application outcomes unseen during selection.
    vector = load_arrays(run_dir / "vectors.npz")[chosen["vector"]]  # Local: load the exact saved direction; global: perform no application-time extraction or retuning.
    target, tolerance = float(effect["treated_mean"]), 0.05 * abs(float(effect["baseline_mean"]))  # Local: use the calibrated response and fixed baseline-relative band; global: lock the target before fresh collection.
    if not np.isfinite(target) or not np.isfinite(tolerance) or target <= 0 or tolerance <= 0:  # Local: validate the calibration before persistence; global: reject nonsensical application specifications.
        raise ValueError("Validation must provide finite positive speed target and tolerance")  # Local: identify invalid calibration; global: avoid unusable downstream evidence.
    torch.set_num_threads(config.torch_threads)  # Local: restore recorded inference parallelism; global: preserve deterministic computation.
    model, provenance = prepare_model(manifest["model_key"], Path.cwd())  # Local: verify and freeze the original checkpoint; global: keep intervention provenance intact.
    if provenance["parameter_sha256"] != manifest["provenance"]["parameter_sha256"]:  # Local: compare actual policy values; global: prevent a changed checkpoint from inheriting prior success.
        raise ValueError("Application policy differs from the replicated policy")  # Local: explain invalid provenance; global: stop a scientifically incompatible run.
    spec = dict(schema=1, behavior="speed", vector=chosen["vector"], alpha=float(chosen["alpha"]), direction=int(chosen["direction"]),  # Local: freeze the replicated condition; global: prohibit application-time strength selection.
                target=target, tolerance=tolerance, tolerance_rule="5% of validation baseline speed", validation_baseline=float(effect["baseline_mean"]),  # Local: record calibration arithmetic; global: make the acceptance band reproducible.
                seeds=seeds, warmup=config.warmup, max_steps=config.max_steps, off_at=600, minimum_success_fraction=0.8, maximum_physical_failures=0,  # Local: freeze cohort and acceptance rules; global: prevent post-hoc threshold changes.
                vector_sha256=hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest(), policy_sha256=provenance["parameter_sha256"])  # Local: fingerprint treatment and policy; global: make resumption identity explicit.
    spec_path = run_dir / "application_spec.json"  # Local: give the pre-data contract a stable artifact; global: establish chronological evidence.
    if spec_path.exists():  # Local: resume only the original application; global: prevent overwriting observed data's target specification.
        saved = load_json(spec_path)  # Local: read the first registered contract; global: retain its original timestamp.
        if any(saved.get(key) != value for key, value in spec.items()):  # Local: compare every frozen field; global: reject retuning beneath existing artifacts.
            raise ValueError("Application specification changed; preserve this attempt and register a new experiment")  # Local: explain the conflict; global: protect failed attempts from erasure.
        spec = saved  # Local: retain original registration time; global: keep the before-data record intact.
    else:  # Local: register this new application before rollout collection; global: separate calibration from evaluation.
        if (run_dir / "episodes" / "application").exists():  # Local: reject unidentified previous application data; global: prevent retroactive registration.
            raise ValueError("Application episode directory exists without a registered specification")  # Local: expose the provenance gap; global: preserve earlier evidence.
        spec["created_utc"] = utc_now()  # Local: timestamp the pre-data decision; global: make protocol chronology auditable.
        save_json(spec_path, spec)  # Local: persist the contract atomically; global: fix target and tolerance before any fresh outcomes.
    logging.info("application locked specification=%s", spec)  # Local: report the complete nonsensitive contract; global: retain the decision in realtime logs.
    baseline = collect_episodes(model, manifest["model_key"], run_dir, "application", seeds, config)  # Local: collect matched unsteered episodes; global: retain the reference for causal and physical-quality checks.
    steered = collect_episodes(model, manifest["model_key"], run_dir, "application", seeds, config, condition=condition_name(spec["vector"], spec["alpha"]), vector=vector, alpha=spec["alpha"])  # Local: apply only the frozen intervention; global: retain all successes and failures.
    baseline_rows, steered_rows = summaries(baseline, config), summaries(steered, config)  # Local: reduce each complete episode once; global: preserve equal episode weighting.
    result = assess_target(steered_rows, target, tolerance, warmup=config.warmup)  # Local: evaluate the predeclared band; global: prohibit application-time adjustment.
    result["paired_effect"] = paired_effect(baseline_rows, steered_rows, "speed", seed=740, warmup=config.warmup)  # Local: reuse paired causal evaluation on fresh data; global: retain the existing locomotion-preservation gates.
    result["baseline_failed_episode_seeds"] = [int(row["seed"]) for row in baseline_rows if _summary(row, config.warmup)["failure"]]  # Local: check both application conditions; global: require zero physical failures rather than only a favorable difference.
    result["gate"] = bool(result["gate"] and not result["baseline_failed_episode_seeds"] and result["paired_effect"]["gate"] and result["paired_effect"]["direction"] == spec["direction"])  # Local: combine targeting, competence, and frozen-direction evidence; global: avoid claiming usefulness from an unrelated or reversed effect.
    result.update(specification="application_spec.json", created_utc=utc_now(), videos_complete=False,  # Local: preserve numeric progress before rendering; global: survive later media failures without losing evidence.
                  limitations="Fixed open-loop calibrated strength on one pretrained policy; no application retuning. Removing the hook restores the original policy function, not the earlier physical state or trajectory.")  # Local: state practical limits; global: avoid claiming feedback control or trajectory rewind.
    save_json(run_dir / "application.json", result)  # Local: save all numeric outcomes before expensive media work; global: retain failed applications too.
    result.update(_media(model, manifest["model_key"], run_dir, baseline, steered, vector, spec, config))  # Local: render prespecified examples and audit numerical identity; global: complete a concrete practical demonstration.
    save_json(run_dir / "application.json", result)  # Local: finalize media provenance with numeric evidence; global: keep the application result self-contained.
    logging.info("application completed result=%s", result)  # Local: publish complete errors/gates/artifact paths to logs; global: make success and limitations reviewable.
    return result  # Local: expose the completed application artifact; global: allow the CLI/reporting layer to present measured results.

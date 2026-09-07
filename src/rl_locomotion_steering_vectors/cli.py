"""Local: documented command entry points; global: make the research reproducible by a user.

Source: https://docs.python.org/3.12/library/argparse.html
"""
import argparse  # Reuse the standard library's command parsing and help generation.
import logging  # All progress and exceptions reach timestamped logs.
from pathlib import Path  # Keep run paths explicit and portable.

import torch  # Replay uses the same controlled CPU thread count.

from .config import Config, read_config  # Replays use saved settings; new runs resolve .env.
from .pipeline import run_pipeline  # The thin phase wrapper owns experiment ordering.
from .execution import use_pool  # Share isolated workers across all scientific phases.
from .runtime import prepare_model, run_episode, summarise_episode  # Reuse authentic inference for videos.
from .storage import load_arrays, load_json, save_arrays, start_logging  # Safe numerical artifacts and realtime logs.


def main() -> None:
    """Local: dispatch run/replay/report; global: expose the complete reproducible workflow."""
    parser = argparse.ArgumentParser(description="Causal activation steering of frozen RL locomotion policies")  # Help describes the actual research scope.
    commands = parser.add_subparsers(dest="command", required=True)  # Require an explicit user operation.
    run = commands.add_parser("run", help="Run or resume diagnosis, extraction, calibration, and held-out evaluation")  # One command executes numerical phases.
    run.add_argument("--model", choices=("halfcheetah", "ant"), default="halfcheetah")  # Restrict models to pinned verified specifications.
    run.add_argument("--run-dir", type=Path, required=True)  # Explicit run identities prevent accidental evidence mixing.
    run.add_argument("--stop-after", choices=("diagnose", "extract", "calibrate", "confirm"))  # Optional inspection boundaries do not change the protocol.
    replay = commands.add_parser("replay", help="Replay a saved activation intervention to MP4 and NPZ")  # Produce visual and numerical evidence together.
    replay.add_argument("--run-dir", type=Path, required=True)  # Read checkpoint identity from the existing manifest.
    replay.add_argument("--vector", default="speed")  # Vector keys match vectors.npz.
    replay.add_argument("--alpha", type=float, required=True)  # Always make strength explicit for visual evidence.
    replay.add_argument("--seed", type=int, default=40000)  # Demonstration seeds are outside statistical selection/evaluation splits.
    replay.add_argument("--off-at", type=int)  # Turn steering off later to demonstrate policy-function recovery.
    report = commands.add_parser("report", help="Regenerate Markdown, plots, and PDF from saved results")  # No simulation is needed to rebuild reports.
    report.add_argument("--run-dir", type=Path, required=True)  # Reporting reads one experiment bundle.
    args = parser.parse_args()  # The standard parser handles validation and usage errors.
    run_dir = args.run_dir.resolve()  # Logs and files refer to a stable absolute path.
    if not run_dir.is_relative_to(Path.cwd().resolve()):  # User requested project-local artifacts only.
        parser.error("--run-dir must be within the current project")  # Reject accidental output elsewhere.
    start_logging(run_dir)  # Install handlers before model loading or scientific work.
    logging.info("command=%s", vars(args))  # Record only supported nonsensitive command arguments.
    try:
        if args.command == "run":
            from .parallel import RolloutPool  # Load process orchestration only for numerical runs.
            config = read_config()  # Resolve settings once before workers and manifests are initialized.
            with RolloutPool(args.model, Path.cwd(), workers=config.rollout_workers) as pool, use_pool(pool):  # Reuse frozen worker policies while guaranteeing shutdown on failure.
                result = run_pipeline(args.model, run_dir, config, args.stop_after)  # Carry the requested phases to completion.
            logging.info("run completed status=%s", result["status"])  # Distinguish valid null results from a discovery.
        elif args.command == "replay":
            manifest = load_json(run_dir / "manifest.json")  # Use the original experiment's identity.
            config = Config(**manifest["config"])  # Preserve original onset and simulator horizon.
            torch.set_num_threads(config.torch_threads)  # Match the measured CPU execution setup.
            model, _ = prepare_model(manifest["model_key"], Path.cwd())  # Reverify pinned upstream weights before replay.
            vector = load_arrays(run_dir / "vectors.npz")[args.vector]  # Pick only a saved numerical direction.
            name = f"{args.vector}_{args.alpha:+.4f}_seed{args.seed}_off{args.off_at}"  # Avoid overwriting distinct demonstrations.
            episode = run_episode(model, manifest["model_key"], args.seed, vector=vector, alpha=args.alpha,
                                  warmup=config.warmup, max_steps=config.max_steps, off_at=args.off_at,
                                  video_path=run_dir / "videos" / f"{name}.mp4")  # Stream frames while recording actual physics.
            save_arrays(run_dir / "videos" / f"{name}.npz", episode)  # Video has an auditable numerical companion.
            logging.info("replay metrics=%s", summarise_episode(episode, config.warmup))  # Make demonstration quality visible.
        else:
            from .reporting import write_report  # Load plotting/PDF dependencies only for reporting.
            logging.info("report=%s", write_report(run_dir))  # Render from saved evidence without refitting.
    except Exception:
        logging.exception("Command failed; completed artifacts remain resumable")  # Preserve the full traceback in terminal and file.
        raise  # Return a failing process status rather than silently hide defects.

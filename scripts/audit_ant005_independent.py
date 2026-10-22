"""Local: thin wrapper for the independent Ant 005 audit. Global: re-derive the world-+Y steering claim from raw data and fresh seeds.

Usage (from the repository root):
    uv run --no-sync python scripts/audit_ant005_independent.py --stop-after directions   # saved-data checks + treatments
    uv run --no-sync python scripts/audit_ant005_independent.py --workers 6               # full audit incl. fresh seeds (12 workers exhausted memory)

Phases: verify_integrity → reanalyze_saved → build_directions → collect_fresh → analyze_fresh → write_outputs.
Protocol: reports/ant-classic-005-independent-audit-preregistration.md
"""
from __future__ import annotations  # Keep annotations lightweight.

import argparse  # Command-line options.
import logging  # Timestamped progress.

import torch  # Inference thread setting.

from ant005_independent.common import REPORT_DIR, ROOT  # Protocol paths.
from ant005_independent.directions import build_directions  # Phase 3.
from ant005_independent.evaluate import analyze_fresh  # Phase 5.
from ant005_independent.fresh import collect_fresh  # Phase 4.
from ant005_independent.integrity import verify_integrity  # Phase 1.
from ant005_independent.outputs import write_outputs  # Phase 6.
from ant005_independent.saved import reanalyze_saved  # Phase 2.
from rl_locomotion_steering_vectors.runtime import prepare_model  # Pinned, verified checkpoint.
from rl_locomotion_steering_vectors.storage import start_logging  # Terminal + timestamped file logs.


def main() -> None:
    """Local: parse options and call the phases. Global: one reproducible audit entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # One-line summary.
    parser.add_argument("--stop-after", choices=("directions",), help="stop before fresh collection")  # Pre-registration boundary.
    parser.add_argument("--workers", type=int, default=6, help="rollout worker processes")  # CPU parallelism; 12 MuJoCo workers exhausted memory on the audit machine, 6 completed.
    args = parser.parse_args()  # Parsed options.
    start_logging(REPORT_DIR)  # Log file in the committed report folder.
    torch.set_num_threads(1)  # Match the original per-worker inference setting.
    logging.info("independent audit command=%s", vars(args))  # Record the invocation.
    model, provenance = prepare_model("ant", ROOT)  # Verified checkpoint.
    results = {"provenance": {k: provenance[k] for k in ("repo_id", "revision", "sha256", "parameter_sha256")}}  # Checkpoint identity.
    results["integrity"] = verify_integrity(model)  # Phase 1.
    results["saved"] = reanalyze_saved()  # Phase 2.
    directions = build_directions(model)  # Phase 3.
    results["directions"] = {k: v for k, v in directions["info"].items() if k != "directions"}  # Summary; per-direction details live in directions.json.
    if args.stop_after != "directions":  # Full run.
        results["collection"] = collect_fresh(model, directions, args.workers)  # Phase 4.
        results["fresh"] = analyze_fresh(directions)  # Phase 5.
    logging.info("outputs %s", write_outputs(results))  # Phase 6.


if __name__ == "__main__":
    main()  # Script entry point.

"""Audit saved research evidence independently, without models or simulator execution.

Usage: uv run --locked --no-sync python scripts/audit_saved_claims.py [--check]
The first build snapshots raw artifacts. Later builds/checks reject altered inputs.
"""
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import tempfile

from rl_locomotion_steering_vectors.factcheck.audit import EXPERIMENTS, fresh_experiment, original_experiment, read_json
from rl_locomotion_steering_vectors.factcheck.inventory import encode, snapshot, verify
from rl_locomotion_steering_vectors.factcheck.provenance import direction_algebra, provenance
from rl_locomotion_steering_vectors.factcheck.record import write_record
from rl_locomotion_steering_vectors.factcheck.display import publication

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"reports/fact-check-2026-10-29"


def generate(root: Path, *, check: bool = False, include_publication: bool = True) -> dict:
    """Preserve raw artifacts, then compute independent comparisons for every original arm."""
    output = root/"reports/fact-check-2026-10-29"
    path = output/"inputs.json"
    if not path.exists():
        if check:
            raise FileNotFoundError("Missing original evidence inventory; --check does not create one")
        output.mkdir(parents=True, exist_ok=True)
        logging.info("Snapshotting original saved evidence")
        path.write_bytes(encode(snapshot(root)))
    inventory = read_json(path)
    integrity = verify(root, inventory)
    if integrity["changed"]:
        raise ValueError(f"Immutable inputs changed: {integrity['changed']}")
    logging.info("Verified %d immutable files", integrity["files"])
    experiments = {name: original_experiment(root, name) for name in EXPERIMENTS}
    audit = fresh_experiment(root)
    directions = direction_algebra(root)
    displayed=publication(root) if include_publication else None
    forbidden = ("torch", "mujoco", "gymnasium", "stable_baselines3", "rl_locomotion_steering_vectors.analysis")
    if any(name in sys.modules for name in forbidden):
        raise RuntimeError("Independent audit imported model/simulator or production analysis modules")
    return {"schema_version": 1, "date": "2026-10-29", "source_revision": inventory["source_revision"],
            "integrity": integrity, "experiments": experiments, "ant_audit": audit, "audit_direction_algebra": directions, "publication":displayed,
            "numerical_tolerance": {"atol": 1e-10, "rtol": 1e-9},
            "execution_boundary": "Saved arrays and NumPy/SciPy statistics only; no model loading, inference, gradients, training or simulator execution."}


def main() -> int:
    """Write reproducible evidence, or compare it without changing publications."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    logs = ROOT/"tmp/saved-claims"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log, encoding="utf-8")], force=True)
    logging.Formatter.converter = time.gmtime
    value = generate(ROOT, check=args.check)
    source_record = provenance(ROOT)
    path = OUTPUT/"verification.json"
    if args.check:
        with tempfile.TemporaryDirectory(prefix='audit-check-',dir=ROOT/'tmp') as temporary:
            stage=Path(temporary)
            generated=write_record(ROOT,value,source_record,stage)
            stale=[p.relative_to(stage).as_posix() for p in generated if not (ROOT/p.relative_to(stage)).exists() or p.read_bytes()!=(ROOT/p.relative_to(stage)).read_bytes()]
        if stale:
            logging.error("Saved-claims outputs are stale: %s",stale)
            return 1
    else:
        write_record(ROOT,value,source_record,ROOT)
    mismatches = [d for e in value["experiments"].values() for d in e["differences"] + e["fitting"].get("differences", [])] + value["ant_audit"]["differences"]+value['publication']['differences']
    logging.info("Independent comparisons complete: %d discrepancies; details in %s", len(mismatches), path)
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline documentation audit: read saved evidence, independently recompute, then report.

Run from the project root: uv run --no-sync python scripts/audit_documentation.py
No checkpoint loading, rollout, fitting selection, or external service is invoked.
Source: https://docs.python.org/3.12/library/logging.html
"""
import hashlib  # Check that the audit never changed its scientific inputs.
import json  # Publish complete non-executable numerical verification records.
import logging  # Preserve timestamped terminal and file output without truncation.
import sys  # Record the literal invoked command and Python version.
from datetime import datetime, timezone  # Distinguish this later audit from original research chronology.
from pathlib import Path  # Keep inputs and outputs within the current project.
import numpy as np  # Record the numerical runtime and encode NumPy scalars explicitly.
from documentation_checks import Comparisons, EvidenceReader, audit_ant_sensitivity, audit_conditions, audit_denominators, audit_fitting, audit_partitions  # Clearly named independent phases keep this entry point thin.
from rl_locomotion_steering_vectors.storage import code_identity, start_logging  # Reuse infrastructure only, never production scientific estimators.


def json_text(value):
    """Local: encode strict JSON; global: make NumPy results portable and reject nonfinite evidence."""
    return json.dumps(value, indent=2, allow_nan=False, default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()) + "\n"  # No opaque or executable numerical serialization is accepted.


def main():
    """Local: orchestrate independent evidence phases; global: publish success only after complete coverage."""
    root = Path.cwd(); destination = root / "reports" / "documentation-audit"  # The documented command requires the project root and a fixed owned output directory.
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run this audit from the project root after restoring the saved run directories")  # A mistaken working directory cannot create a misleading empty audit.
    start_logging(destination)  # Attach the existing UTC terminal/file handlers before any expensive reads.
    started = datetime.now(timezone.utc).isoformat(); reader = EvidenceReader(root); checks = Comparisons()  # New calculations receive a separate timestamp and mismatch registry.
    command = [sys.executable, *sys.argv]  # Only public command arguments are logged; no environment or authentication values are read.
    logging.info("Offline documentation audit command: %s", command)  # The complete invocation remains in the timestamped log.
    outputs, failures = {}, []  # Missing or malformed evidence must be an explicit audit failure.
    phases = {"conditions": audit_conditions, "fitting": audit_fitting, "partitions": audit_partitions, "ant_sensitivity": audit_ant_sensitivity, "denominators": audit_denominators}  # These phases only read already collected arrays.
    for name, phase in phases.items():
        try:
            outputs[name] = phase(reader, checks)  # Complete each independent calculation without calling a simulator or changing a selection.
        except Exception as error:
            failures.append({"phase": name, "error": repr(error)})  # Retain unavailable phases instead of silently reducing the claimed scope.
            logging.exception("Audit phase %s failed", name)  # Preserve full error traceback in both handlers.
    for name, digest in reader.identities.items():
        checks.compare(f"scientific_inputs_unchanged.{name}", digest, hashlib.sha256((root / name).read_bytes()).hexdigest())  # Rehash every consumed input after all calculations.
    identities = code_identity(root)  # Reuse the project's public source/dependency provenance without reading .env secrets.
    identities["audit_source_sha256"] = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in ("audit_documentation.py", "documentation_checks.py", "documentation_evidence.py")}  # Include uncommitted audit source bytes as well as the current Git reference.
    count = sum(len(rows) for run in outputs.get("conditions", {}).values() for rows in run.values())  # A complete pass requires all 332 historical condition comparisons.
    checks.compare("coverage.condition_count", 332, count)  # Missing data must not yield a false zero-mismatch success.
    verification = {"created_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "command": command, "source_identity": identities, "python": sys.version, "numpy": np.__version__, "absolute_tolerance": 1e-12, "passed": not checks.mismatches and not failures, "condition_count": count, "checked_leaf_fields": checks.count, "max_numeric_error": checks.max_numeric_error, "mismatches": checks.mismatches, "phase_errors": failures, "calculations": outputs, "scope": "Saved-data reproduction only. No policy inference, rollouts, new research seeds, training, treatment selection or changed gates. Historical gates are metadata; the oracle independently reproduces means, effects, intervals, failure fractions, extraction and diagnostics."}  # State exactly what this audit proves and what it does not recalculate.
    artifacts = {"numerical-audit.json": verification, "ant-sensitivity-verification.json": {"reference": "reports/ant-classic-005/audit/fitting-inspection.json", "reference_sha256": reader.identities.get("reports/ant-classic-005/audit/fitting-inspection.json"), "calculated": outputs.get("ant_sensitivity"), "mismatches": [item for item in checks.mismatches if item["field"].startswith("ant_sensitivity")], "phase_errors": [item for item in failures if item["phase"] == "ant_sensitivity"], "conventions": {"heading": "Arithmetic pre-action quaternion-derived heading, averaged equally across windows in group descriptors; behavior/activation group means weight contributing episodes equally.", "scope": "Original chronological scope metadata is preserved but not numerically reconstructed."}}, "input-identities.json": reader.identities}  # Preserve both readable phase results and every raw input fingerprint.
    for name, value in artifacts.items():
        text = json_text(value); path = destination / name  # Strict encoding fails rather than publishing NaN as scientific evidence.
        path.write_text(text, encoding="utf-8")  # Only new audit outputs are replaced; original research artifacts remain untouched.
        logging.info("Saved %s\n%s", path, text)  # Full numerical findings and provenance are written to terminal/file logs without truncation.
    logging.info("Audit finished: passed=%s; conditions=%d; compared fields=%d; mismatches=%d; failed phases=%d", verification["passed"], count, checks.count, len(checks.mismatches), len(failures))  # A concise terminal summary follows the complete evidence.
    return 0 if verification["passed"] else 1  # Shell callers can reliably detect missing evidence or mismatches.


if __name__ == "__main__":
    raise SystemExit(main())  # Importing helper code cannot unexpectedly run an audit.

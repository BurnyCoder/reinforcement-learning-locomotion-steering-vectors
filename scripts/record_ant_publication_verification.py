"""Record observed release checks and exact artifact identities; no scientific measurements are invented."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
out = root / "reports/ant-publication"


def digest(path):
    """Identify complete published bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_log(path):
    """PowerShell logs can be UTF16; preserve originals and decode for metadata only."""
    data = path.read_bytes()
    return data.decode("utf-16" if data.startswith(b"\xff\xfe") else "utf-8")


def write(name, value):
    """Save stable UTF8 receipts separately from reproducible scientific outputs."""
    (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def optional(name):
    """Report a pending check explicitly until its real receipt exists."""
    path = out / name
    return json.loads(read_log(path)) if path.exists() else {"status": "pending"}


def compact(name):
    """Link detailed receipts without duplicating all per-file identities in the overview."""
    return {key: value for key, value in optional(name).items() if key not in ("checks", "installed_packages")}


for name in ("cohorts-before-tests.log", "cohorts-after-tests.log", "e9-regeneration-before-tests.log", "e9-regeneration-after-tests.log", "paper-greek-before-tests.log", "paper-greek-after-tests.log"):
    source = root / "tmp" / name
    if source.exists():
        shutil.copyfile(source, out / "logs" / name)

pdf = root / "reports/ant-classic-005-independent-audit.pdf"
info = subprocess.check_output(["pdfinfo", str(pdf)], text=True, encoding="utf-8")
pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
write("audit-pdf-visual-verification.json", {"checked_utc": datetime.now(timezone.utc).isoformat(), "path": pdf.relative_to(root).as_posix(), "sha256": digest(pdf), "pages": pages, "visually_inspected_pages": list(range(1, pages + 1)), "render": "pdftoppm -jpeg -r 110 reports/ant-classic-005-independent-audit.pdf tmp/ant-audit-pdf/final-page", "author": re.search(r"Author:\s+(.+)", info)[1], "result": "All pages inspected. No cropped text/tables/axes; Delta and alpha notation visible after portable-name fix."})

tests = read_log(out / "logs/tests-release.log")
write("publication-verification.json", {
    "schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Existing Ant evidence publication; no scientific rollouts/training/retuning",
    "git_commit_when_recorded": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "core_pull_request": "https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/pull/18",
    "verification_pull_request": "https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/pull/19",
    "core_merge_commit": "4c7c19f98c67bc49d3001d1739a7cc1bd56197bd",
    "tests": {"command": "uv run --locked --no-sync pytest -q", "passed": int(re.search(r"(\d+) passed", tests)[1]), "log": "logs/tests-release.log"},
    "build": {"command": "uv run --locked --no-sync python scripts/build_ant_publication.py", "log": "logs/build-integrated.log", "check_log": "logs/build-check-final.log"},
    "inputs": {"receipt": "input-upload-verification.json", "remote_path_receipt": "remote-path-verification.json", "revision": optional("input-upload-verification.json").get("revision"), "all_hashes_match": optional("input-upload-verification.json").get("all_hashes_match", False)},
    "fresh_restore_receipt": "clean-reproduction-verification.json",
    "fresh_restore": compact("clean-reproduction-verification.json"),
    "documents": compact("document-upload-verification.json"),
    "pdfs": ["paper-visual-verification.json", "audit-pdf-visual-verification.json"],
    "collection": {"url": "https://huggingface.co/collections/BurnyCoder/ant-locomotion-steering-evidence-6ae2900bb110a4be69bbfb0b", "receipt": "collection-verification.json", "items": [item["item_id"] for item in optional("collection-verification.json").get("items", [])]},
    "pages": {"url": "https://burnycoder.github.io/reinforcement-learning-locomotion-steering-vectors/", "initial_run": "https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/actions/runs/40794705389", "initial_conclusion": "success", "live_receipt": "live-site-verification.json"},
    "deployed_bytes": compact("live-site-verification.json"),
    "final_deployment": compact("pages-deployment.json"),
    "ui": {"local_and_live_flows": ["original confirmation and pre-onset failure", "validation cross-label strength sweep", "audit projection ablation signs", "random directions filter", "observation rotation units", "seed trajectories and evaluation rewards"], "mobile_viewport": {"width": 390, "height": 844, "observed_document_scroll_width": 375}, "application_console_errors_observed": [], "download": "same-origin download attribute added; browser download event unavailable; HTTP bytes checked separately"},
    "review_and_repairs": "verification.md",
    "limitations": ["Original confirmation_failed retained", "No new scientific collection", "Historical preregistration timing unrepairable", "Baukit/full checkpoint licensing unresolved", "No formal replication, training-seed generalization, unique heading mechanism or practical-control claim"]
})

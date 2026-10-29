"""Verify complete deployed static files against the reviewed local publication.

No browser interaction is simulated here; browser checks are recorded separately.
Source: https://docs.python.org/3.12/library/urllib.request.html
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parents[3]
base = "https://burnycoder.github.io/reinforcement-learning-locomotion-steering-vectors/"
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
files = sorted(path for path in (root / "site").rglob("*") if path.is_file())


def verify(path):
    """Fetch the same file a visitor receives, with a cache-busting public commit identity."""
    relative = path.relative_to(root / "site").as_posix()
    url = base + quote(relative, safe="/") + "?verify=" + revision
    with urlopen(Request(url, headers={"Cache-Control": "no-cache"}), timeout=55) as response:
        body, status = response.read(), response.status
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = hashlib.sha256(body).hexdigest()
    if status != 200 or expected != actual:
        raise ValueError(f"Deployment mismatch: {relative}")
    return {"path": relative, "sha256": actual, "bytes": len(body), "status": status}


with ThreadPoolExecutor(max_workers=6) as pool:
    checks = list(pool.map(verify, files))
receipt = {"checked_utc": datetime.now(timezone.utc).isoformat(), "url": base, "local_revision": revision, "files": len(checks), "all_hashes_match": True, "bytes": sum(row["bytes"] for row in checks), "checks": checks}
(root / "reports/ant-publication/live-site-verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"All {len(checks)} deployed files match the reviewed local bytes")

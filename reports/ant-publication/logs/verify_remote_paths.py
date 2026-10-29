"""Confirm every pinned Hub path exists, including paths sharing cached content bytes.

This complements downloaded SHA256 verification with per-path Git/LFS identities.
Source: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.get_paths_info
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

root = Path(__file__).resolve().parents[3]
manifest = json.loads((root / "reports/ant-publication/inputs.json").read_text(encoding="utf-8"))
records = manifest["files"]
api = HfApi(token=False)


def verify_batch(batch):
    """Bind each public path to its verified content, not merely to an upload response."""
    remote = {item.path: item for item in api.get_paths_info(manifest["repo_id"], [item["hub_path"] for item in batch], revision=manifest["revision"])}
    checked = []
    for row in batch:
        item = remote[row["hub_path"]]
        assert item.size == row["size"], row["hub_path"]
        if item.lfs:
            assert item.lfs.sha256 == row["sha256"], row["hub_path"]
            identity = item.lfs.sha256
        else:
            content = (root / ".cache/ant-verification" / row["sha256"]).read_bytes()
            identity = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            assert item.blob_id == identity, row["hub_path"]
        checked.append({"hub_path": row["hub_path"], "identity": identity, "kind": "lfs_sha256" if item.lfs else "git_blob_sha1"})
    return checked


with ThreadPoolExecutor(max_workers=4) as pool:
    checks = [row for batch in pool.map(verify_batch, [records[i:i + 256] for i in range(0, len(records), 256)]) for row in batch]
receipt = {"checked_utc": datetime.now(timezone.utc).isoformat(), "repo_id": manifest["repo_id"], "revision": manifest["revision"], "all_paths_match": True, "files": len(checks), "distinct_downloaded_sha256": len({row["sha256"] for row in records}), "checks": checks}
(root / "reports/ant-publication/remote-path-verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"Verified all {len(checks)} remote paths against downloaded content identities")

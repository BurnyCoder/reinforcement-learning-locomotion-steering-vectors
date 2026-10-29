"""Validate the static Ant publication using Python's standard library only.

This checks serialization, source references and display completeness; it does not
rerun statistics or load a policy. JSON and path contracts follow
https://docs.python.org/3/library/json.html and
https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import math
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit


def require(condition: bool, message: str) -> None:
    """Reject invalid artifacts even when Python runs with optimizations enabled."""
    if not condition:
        raise ValueError(message)


def numeric(value: object) -> bool:
    """Booleans are flags, not measured numerical values."""
    return type(value) in (int, float) and math.isfinite(value)


def load_json(path: Path) -> dict:
    """Reject nonfinite JSON throughout the saved evidence, including float overflow."""
    data = json.loads(path.read_text(encoding="utf-8"))

    def check(value: object) -> None:
        if isinstance(value, float):
            require(math.isfinite(value), f"{path}: nonfinite numerical value")
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(data)
    require(isinstance(data, dict), f"{path}: expected a JSON object")
    return data


def local_file(root: Path, reference: str) -> Path:
    """Resolve a relative asset under its declared root; reject escapes and missing files."""
    parsed = urlsplit(reference)
    decoded = unquote(parsed.path)
    parts = PurePosixPath(decoded).parts
    require(bool(decoded) and not parsed.scheme and not parsed.netloc and not decoded.startswith("/"), f"nonrelative asset: {reference}")
    require("\\" not in decoded and ":" not in decoded and ".." not in parts, f"unsafe asset path: {reference}")
    resolved_root = root.resolve()
    target = resolved_root.joinpath(*parts).resolve()
    require(target.is_relative_to(resolved_root), f"asset escapes its root: {reference}")
    require(target.is_file(), f"missing asset: {reference}")
    return target


class AssetLinks(HTMLParser):
    """Collect HTML asset/link references without executing browser content."""

    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.references.extend(value for key, value in attrs if key in {"src", "href", "poster"} and value)


def check_reference(root: Path, reference: str) -> None:
    """Check local assets; permit explicit HTTPS/HTTP source links without network checks."""
    if reference.startswith("#"):
        return
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc:
        require(parsed.scheme in {"http", "https"} and bool(parsed.netloc), f"unsupported link: {reference}")
    else:
        local_file(root, reference)


def check_arm(arm: dict, *, warmup: int, context: str) -> None:
    """Check chart alignment and retain onset/endpoints, including prefix-only failures."""
    steps = arm.get("steps", [])
    length = arm.get("length")
    require(type(length) is int and length >= 0, f"{context}: invalid episode length")
    require(isinstance(steps, list) and bool(steps) and all(type(step) is int for step in steps), f"{context}: invalid steps")
    require(steps[0] == 0 and steps[-1] == length, f"{context}: missing initial or recorded endpoint")
    require(all(a < b for a, b in zip(steps, steps[1:])), f"{context}: unordered steps")
    require(length <= warmup or warmup in steps, f"{context}: intervention onset omitted")
    require(type(arm.get("failed")) is bool and type(arm.get("prefix_only")) is bool, f"{context}: missing episode flags")
    for key in ("time", "x", "y", "reward"):
        values = arm.get(key, [])
        require(isinstance(values, list) and len(values) == len(steps), f"{context}: inconsistent {key} length")
        require(all(numeric(value) or key == "reward" and index == 0 and value is None for index, value in enumerate(values)), f"{context}: invalid {key} samples")


def validate_site(root: Path) -> dict[str, int]:
    """Check all published conditions and every display shard without ML dependencies."""
    root = root.resolve()
    for entry in root.rglob("*"):
        require(not entry.is_symlink(), f"site contains a symbolic link: {entry}")
    parser = AssetLinks()
    parser.feed(local_file(root, "index.html").read_text(encoding="utf-8"))
    for reference in parser.references:
        check_reference(root, reference)
    for stylesheet in root.glob("*.css"):
        for reference in re.findall(r"url\(\s*['\"]?([^)'\"\s]+)", stylesheet.read_text(encoding="utf-8")):
            check_reference(stylesheet.parent, reference)
    evidence = load_json(local_file(root, "data/ant005.json"))
    require(evidence.get("schema_version") == 1, "unsupported evidence schema")
    metadata = evidence["metadata"]
    require(metadata["status"] == "confirmation_failed", "historical experiment outcome was changed")
    require(type(metadata["warmup"]) is int and metadata["warmup"] >= 0 and numeric(metadata["dt"]) and metadata["dt"] > 0, "invalid sampling metadata")
    for link in metadata["source_links"]:
        require(isinstance(link["label"], str) and bool(link["label"]), "source link needs a label")
        check_reference(root, link["url"])
    # Figure paths are repository evidence references, not browser asset URLs.
    for figure in evidence.get("figures", []):
        local_file(root.parent, figure["path"])
    counts = {"cohorts": 0, "conditions": 0, "episode_pairs": 0}
    cohort_ids, identities = set(), set()
    for cohort in evidence["cohorts"]:
        cohort_id, seeds = cohort["id"], cohort["seeds"]
        require(cohort_id not in cohort_ids, f"duplicate cohort: {cohort_id}")
        cohort_ids.add(cohort_id)
        require(len(seeds) == cohort["n"] == len(set(seeds)) and all(type(seed) is int for seed in seeds), f"{cohort_id}: invalid seeds")
        require(sum(row["kind"] == "baseline" for row in cohort["conditions"]) == 1, f"{cohort_id}: expected one baseline")
        for condition in cohort["conditions"]:
            context = f"{cohort_id}/{condition['id']}"
            identity = (cohort_id, condition["id"])
            require(identity not in identities, f"duplicate condition: {context}")
            identities.add(identity)
            require(all(isinstance(condition.get(key), str) and bool(condition[key]) for key in ("id", "label", "vector", "kind")), f"{context}: missing condition labels")
            require(condition.get("alpha") is None or numeric(condition["alpha"]), f"{context}: invalid strength")
            require(condition["seeds"] == seeds, f"{context}: condition seed order differs")
            require(all(numeric(condition["metrics"][key]) for key in ("x_velocity", "y_velocity", "effort", "reward", "total_return", "failures", "prefix_only")), f"{context}: invalid metrics")
            for key in ("failures", "prefix_only"):
                require(type(condition["metrics"][key]) is int and 0 <= condition["metrics"][key] <= len(seeds), f"{context}: invalid {key} count")
            if condition.get("oddEven") is not None:
                require(all(numeric(condition["oddEven"][key]) for key in ("odd", "even")), f"{context}: invalid signed-response summary")
            effect = condition.get("pairedEffect")
            if condition["kind"] != "baseline":
                require(isinstance(effect, dict) and numeric(effect["mean"]), f"{context}: missing paired effect")
                require(isinstance(effect.get("source"), str) and bool(effect["source"]), f"{context}: missing estimator source")
                interval = effect.get("ci95")
                require(interval is None or len(interval) == 2 and all(numeric(value) for value in interval) and interval[0] <= interval[1], f"{context}: invalid interval")
            shard_path = condition["episodesUrl"]
            require(shard_path.startswith("data/episodes/") and shard_path.endswith(".json"), f"{context}: unexpected shard location")
            shard = load_json(local_file(root, shard_path))
            require(shard["cohort"] == cohort_id and shard["condition"] == condition["id"], f"{context}: wrong shard identity")
            require(shard["warmup"] == metadata["warmup"] and shard["dt"] == metadata["dt"], f"{context}: different sampling metadata")
            require([episode["seed"] for episode in shard["episodes"]] == seeds, f"{context}: missing, repeated or reordered episode pairs")
            for episode in shard["episodes"]:
                for arm in ("baseline", "treated"):
                    check_arm(episode[arm], warmup=metadata["warmup"], context=f"{context}/{episode['seed']}/{arm}")
            counts["conditions"] += 1
            counts["episode_pairs"] += len(seeds)
        counts["cohorts"] += 1
    featured = metadata["featured"]
    require((featured["cohort"], featured["condition"]) in identities, "featured treatment is missing")
    require(counts["cohorts"] > 0, "no evidence cohorts")
    return counts


def main() -> None:
    """Provide one read-only CI/local check with a timestamped result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path(__file__).resolve().parents[1] / "site")
    args = parser.parse_args()
    try:
        counts = validate_site(args.site)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Ant site validation failed: {error}\n")
    print(json.dumps({"checked_utc": datetime.now(timezone.utc).isoformat(), "status": "ok", **counts}))


if __name__ == "__main__":
    main()

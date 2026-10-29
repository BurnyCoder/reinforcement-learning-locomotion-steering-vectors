"""Check this README restructure's local references and README HTTP links.

Run from the project root with uv run --locked --no-sync python
reports/readme-restructure/check_links.py. This scoped evidence checker uses
stdlib path/URL handling and performs HEAD requests only; it never reads remote
response bodies or records redirect targets. It does not evaluate source claims.
Sources: https://docs.python.org/3.12/library/urllib.request.html and
https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/section-links
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import struct
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("README.md", "docs/usage.md", "docs/ant-publication.md", "docs/ant-videos.md")
INLINE_LINK = re.compile(r'(!?)\[[^\]\n]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)')


def prose_lines(text: str) -> list[tuple[int, str]]:
    """Exclude fenced code examples from Markdown-link and heading interpretation."""
    visible, fence = [], None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            mark = match[1]
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = None
            continue
        if fence is None:
            visible.append((number, line))
    return visible


def markdown_anchors(path: Path) -> set[str]:
    """Resolve the ATX headings and explicit HTML anchors used in these documents."""
    anchors, occurrences = set(), {}
    for _, line in prose_lines(path.read_text(encoding="utf-8")):
        anchors.update(re.findall(r'(?:id|name)=[\"\']([^\"\']+)[\"\']', line))
        match = re.match(r"^#{1,6}\s+(.+?)(?:\s+#+)?\s*$", line)
        if not match:
            continue
        heading = re.sub(r"<[^>]+>", "", html.unescape(match[1]))
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        suffix = occurrences.get(slug, 0)
        occurrences[slug] = suffix + 1
        anchors.add(slug if suffix == 0 else f"{slug}-{suffix}")
    return anchors


def document_links(text: str) -> list[tuple[int, str, bool]]:
    """Read inline Markdown links/images and standalone URLs in this fixed scope."""
    links = []
    for line_number, line in prose_lines(text):
        links.extend((line_number, target.strip("<>"), bool(image))
                     for image, target in INLINE_LINK.findall(line))
        if re.fullmatch(r"https?://\S+", line.strip()):
            links.append((line_number, line.strip(), False))
    return links


def local_check(source: str, line: int, target: str, image: bool) -> dict:
    """Check bounded workspace targets, Markdown fragments, and PNG file headers."""
    row = {"source": source, "line": line, "target": target, "image": image}
    parsed = urlsplit(target)
    path = unquote(parsed.path)
    resolved = ((ROOT / path.lstrip("/")) if path.startswith("/") else
                (ROOT / source).parent / path if path else ROOT / source).resolve()
    if not resolved.is_relative_to(ROOT):
        return dict(row, status="outside_workspace")
    if not resolved.exists():
        return dict(row, status="missing_target")
    if parsed.fragment and resolved.suffix.lower() == ".md":
        if unquote(parsed.fragment) not in markdown_anchors(resolved):
            return dict(row, status="missing_markdown_anchor")
    if image and resolved.suffix.lower() == ".png":
        with resolved.open("rb") as handle:
            header = handle.read(24)
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            return dict(row, status="invalid_png_header")
        width, height = struct.unpack(">II", header[16:24])
        if not width or not height:
            return dict(row, status="invalid_png_dimensions")
        row["png_dimensions"] = [width, height]
    return dict(row, status="ok")


def http_check(url: str) -> dict:
    """Use one bounded HEAD request; save neither bodies nor signed redirect URLs."""
    row = {"url": url, "method": "HEAD", "timeout_seconds": 15}
    try:
        request = Request(url, method="HEAD", headers={"User-Agent": "README-link-check/1.0"})
        with urlopen(request, timeout=15) as response:
            row.update(status="reachable", http_status=response.status,
                       content_type=response.headers.get_content_type())
    except HTTPError as error:
        row.update(status="http_error", http_status=error.code, error_type=type(error).__name__)
    except (URLError, TimeoutError, OSError) as error:
        # Exception strings can contain temporary redirect URLs; retain only the exception class.
        row.update(status="not_verified", error_type=type(error).__name__)
    return row


def main() -> int:
    """Write a timestamped verification record without changing source publications."""
    started = datetime.now(timezone.utc)
    local, external, source_hashes = [], set(), {}
    for source in SOURCES:
        data = (ROOT / source).read_bytes()
        source_hashes[source] = hashlib.sha256(data).hexdigest()
        for line, target, image in document_links(data.decode("utf-8")):
            if urlsplit(target).scheme in ("http", "https"):
                if source == "README.md":
                    external.add(target)
            else:
                local.append(local_check(source, line, target, image))
    with ThreadPoolExecutor(max_workers=6) as workers:
        remote = list(workers.map(http_check, sorted(external)))
    failed_local = [row for row in local if row["status"] != "ok"]
    unresolved_remote = [row for row in remote if row["status"] != "reachable"]
    result = {
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "uv run --locked --no-sync python reports/readme-restructure/check_links.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "summary": {"local_references": len(local), "local_failures": len(failed_local),
                    "readme_external_urls": len(remote), "external_unresolved": len(unresolved_remote)},
        "local": local, "external": remote,
        "limitations": [
            "Scoped to four named documents; HTTP checks cover README URLs only.",
            "Parses inline links/images, standalone URLs, ATX headings and explicit HTML anchors; no full Markdown renderer.",
            "HEAD success establishes HTTP reachability only, not claim accuracy, remote fragment validity, or media playback.",
            "HEAD errors can reflect server policy, rate limits, or transient network conditions; no GET fallback is attempted.",
            "No remote response bodies or redirect destinations are downloaded or recorded.",
            "PNG checks cover file signatures and dimensions, not full image decoding or visual quality.",
        ],
    }
    output = Path(__file__).parent / (started.strftime("%Y%m%dT%H%M%SZ") + "-links.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), **result["summary"]}, indent=2))
    for row in [*failed_local, *unresolved_remote]:
        print(json.dumps(row, ensure_ascii=False))
    return int(bool(failed_local or unresolved_remote))


if __name__ == "__main__":
    sys.exit(main())

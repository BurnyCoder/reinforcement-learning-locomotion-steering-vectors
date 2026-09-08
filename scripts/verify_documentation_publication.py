"""Local: check public evidence copies. Global: source publication claims without running experiments.

Sources: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.model_info
https://docs.python.org/3.12/library/urllib.request.html and
https://docs.python.org/3.12/library/hashlib.html describe public reads and content identities.
https://docs.github.com/en/rest/commits/commits#get-a-commit identifies the rendered README revision.
Only original compact scientific files are compared; revised narratives, new audits and the paper are outside this preservation check.
"""

import argparse  # Local: accept an explicit revision; global: make a publication check repeatable.
from concurrent.futures import ThreadPoolExecutor  # Local: overlap independent downloads; global: keep the audit bounded to four requests.
import hashlib  # Local: identify exact bytes; global: distinguish preservation from numerical similarity.
import json  # Local: compare scientific values; global: allow harmless JSON whitespace differences.
import logging  # Local: record progress; global: preserve complete timestamped audit output.
from pathlib import Path  # Local: resolve project files; global: avoid inspecting unrelated folders.
import re  # Local: count rendered video elements; global: verify embedding rather than just URL presence.
import subprocess  # Local: read tracked file names; global: exclude local credentials and paused drafts.
import sys  # Local: record the actual Python invocation; global: distinguish this check from its launch wrapper.
from urllib.parse import quote  # Local: encode branch names as URL components; global: keep requests on the intended public endpoint.
from urllib.request import Request, urlopen  # Local: perform public HTTP reads; global: verify accessible publication artifacts.

from huggingface_hub import HfApi  # Local: use the supported repository inventory; global: pin the exact public revision.
from rl_locomotion_steering_vectors.storage import save_json, start_logging, utc_now  # Local: reuse serialization and logging; global: avoid a second artifact framework.

REPO = "BurnyCoder/rl-locomotion-steering-vectors"  # Local: identify the authorized public repository; global: keep the check scoped.
BASELINE = "d0cd2c07e3641531c39037223eb98e309f44c4dc"  # Local: identify pre-correction artifacts; global: keep original evidence immutable.
BUNDLES = ("hc-classic-001", "hc-running-002", "hc-height-speed-003", "hc-height-speed-004", "ant-classic-005")  # Local: enumerate the five scientific bundles; global: never misclassify paper_assets as an experiment.
MEDIA = ("676ef6c8-ade4-4f85-a0b7-01d7fcdb86c9", "17286e19-c5a2-4ab9-96ab-f50444d494ca", "9e0719a6-c15b-474e-b74e-3c5e70954e2a")  # Local: retain the three published attachments; global: verify the actual featured videos.


def original_files() -> list[str]:
    """Local: list pre-correction bundle evidence. Global: keep later audits outside the preservation comparison."""
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASELINE, "reports"], text=True).splitlines()  # Local: query one explicit tree; global: make the baseline independent of working edits.
    return [name for name in names if len(Path(name).parts) == 3 and Path(name).parts[1] in BUNDLES and (Path(name).suffix in (".json", ".npz", ".png", ".pdf") or name.endswith("/report.md"))]  # Local: restrict compact bundle files; global: exclude revised paper/narrative prose.


def safe_error(error: Exception) -> dict:
    """Local: retain failure facts. Global: exclude exception text, response bodies and signed redirect URLs."""
    code = getattr(error, "code", getattr(getattr(error, "response", None), "status_code", None))  # Local: support urllib and Hub HTTP errors; global: avoid serializing response objects.
    return {"status": "error", "error_type": type(error).__name__, "http_status": code if isinstance(code, int) else None}  # Local: keep only a class name and integer; global: publish reviewable failures without credentials.


def verify_copy(name: str, revision: str) -> dict:
    """Local: compare local, original Git, and public bytes. Global: catch changed outcomes and incomplete uploads."""
    remote_path = "experiments/" + name.removeprefix("reports/")  # Local: use the established Hub layout; global: compare like-for-like evidence.
    url = f"https://huggingface.co/{REPO}/resolve/{revision}/{remote_path}"  # Local: pin the download; global: avoid races against a moving branch.
    result = {"path": name, "remote_path": remote_path, "url": url}  # Local: retain only the stable requested URL; global: make missing public files identifiable.
    try:
        local = Path(name).read_bytes()  # Local: read a named bundle artifact; global: avoid loading executable model checkpoints.
        original = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"])  # Local: recover original bytes; global: establish preservation independently of an earlier cache.
        with urlopen(url, timeout=60) as response:  # Local: read public content without credentials; global: compare actual available bytes.
            remote = response.read()  # Local: fetch only compact evidence; global: exclude large episode archives and videos from this check.
        mode = "json_values" if name.endswith(".json") else "normalized_text" if name.endswith(".md") else "bytes"  # Local: select the documented equivalence relation; global: keep copy claims precise.
        normalize = (lambda data: json.loads(data)) if mode == "json_values" else (lambda data: data.decode("utf-8").replace("\r\n", "\n")) if mode == "normalized_text" else (lambda data: data)  # Local: compare the appropriate representation; global: tolerate only documented formatting differences.
        result.update(status="ok", comparison=mode, local_sha256=hashlib.sha256(local).hexdigest(), original_sha256=hashlib.sha256(original).hexdigest(), remote_sha256=hashlib.sha256(remote).hexdigest(), unchanged_from_git=normalize(local) == normalize(original), public_copy_matches=normalize(local) == normalize(remote))  # Local: retain three content identities; global: make each preservation comparison reviewable.
    except Exception as error:  # Local: isolate one file's failed check; global: continue collecting other evidence before failing the command.
        result.update(safe_error(error))  # Local: record safe failure facts; global: never print raw redirect exceptions.
    logging.info("Copy check %s status=%s unchanged=%s public_match=%s error_type=%s http_status=%s", name, result["status"], result.get("unchanged_from_git"), result.get("public_copy_matches"), result.get("error_type"), result.get("http_status"))  # Local: expose each completed check; global: retain both mismatches and unavailable files.
    return result  # Local: return plain metadata; global: avoid retaining downloaded content in logs.


def verify_media(github_revision: str = "main") -> dict:
    """Local: inspect public headers and rendered README. Global: verify media accessibility without downloading movies."""
    videos = []  # Local: collect header results; global: report each featured attachment separately.
    for asset in MEDIA:  # Local: check the original three movies; global: avoid treating unrelated media as evidence.
        url = f"https://github.com/user-attachments/assets/{asset}"  # Local: retain only the stable public URL; global: never publish temporary signed redirect URLs.
        row = {"url": url}  # Local: preserve the stable attachment address; global: identify failures without redirected tokens.
        try:
            with urlopen(Request(url, method="HEAD"), timeout=60) as response:  # Local: request headers only; global: verify availability without fetching video bytes.
                row.update(status="ok", http_status=response.status, content_type=response.headers.get_content_type())  # Local: keep nonsensitive response facts; global: source playable-format claims.
        except Exception as error:  # Local: isolate unavailable media; global: still check remaining attachments and README.
            row.update(safe_error(error))  # Local: omit error text and headers; global: never persist temporary access tokens.
        videos.append(row)  # Local: retain every promised attachment; global: do not drop failed checks.
    readme = {"url": f"https://api.github.com/repos/{REPO}/commits/{quote(github_revision, safe='')}"}  # Local: resolve the requested GitHub ref; global: separate README identity from the Hub's revision.
    players = None  # Local: distinguish an unavailable page from zero players; global: avoid invented verification results.
    try:
        with urlopen(readme["url"], timeout=60) as response:  # Local: read public commit metadata; global: resolve a moving ref once without credentials.
            revision = json.load(response)["sha"]  # Local: keep only the immutable commit; global: exclude unrelated metadata from the evidence.
        readme.update(git_revision=revision, url=f"https://github.com/{REPO}/tree/{revision}")  # Local: pin the rendered tree; global: make the README check reproducible after later pushes.
        with urlopen(readme["url"], timeout=60) as response:  # Local: inspect this public repository revision; global: count actual rendered players.
            html = response.read().decode("utf-8")  # Local: inspect HTML in memory only; global: never save signed tokens embedded by GitHub.
        players = len(re.findall(r"<video\b", html))  # Local: count actual video tags; global: keep this rendering check separate from attachment availability.
        readme["status"] = "ok"  # Local: mark a complete page inspection; global: permit acceptance only after successful access.
    except Exception as error:  # Local: preserve metadata or rendering failure; global: save all attachment results before command failure.
        readme.update(safe_error(error))  # Local: retain safe error facts; global: exclude full HTML and redirect URLs.
    return {"videos": videos, "readme_video_players": players, "readme": readme}  # Local: expose checks and pinned stable links; global: avoid storing browser or authentication state.


def main() -> None:
    """Local: orchestrate inventory, comparisons and media. Global: provide one repeatable publication verification command."""
    parser = argparse.ArgumentParser(description=__doc__)  # Local: describe this read-only verification; global: distinguish it from publishing.
    parser.add_argument("--revision", default="main")  # Local: allow an explicit Hub pin; global: resolve moving main once before downloads.
    parser.add_argument("--github-revision", default="main")  # Local: identify the README separately; global: never confuse GitHub and Hub commit histories.
    parser.add_argument("--output", type=Path, default=Path("reports/documentation-audit/publication-verification.json"))  # Local: use a project evidence destination; global: keep the result alongside the claim ledger.
    args = parser.parse_args()  # Local: parse the requested check; global: reject misspelled options.
    if not args.output.resolve().is_relative_to(Path.cwd().resolve()):  # Local: enforce project-local writes; global: respect the user's workspace boundary.
        parser.error("Output must remain in the current project")  # Local: fail before writing; global: avoid accidental external artifacts.
    start_logging(args.output.parent)  # Local: reuse full terminal/file logs; global: preserve successful and failed attempts.
    result = {"created_utc": utc_now(), "command": [sys.executable, *sys.argv], "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "script_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode("utf-8")).hexdigest(), "script_hash_convention": "UTF-8 text with universal newlines", "baseline_git": BASELINE, "hf_requested_revision": args.revision, "hf_revision": None, "public": False, "copies": []}  # Local: record invocation and actual source identity; global: preserve provenance even if external metadata fails.
    try:
        info = HfApi(token=False).model_info(REPO, revision=args.revision)  # Local: disable authentication explicitly; global: verify a public immutable revision.
        result.update(hf_revision=info.sha, public=not info.private, repository_file_count=len(info.siblings), hf_inventory={"status": "ok"})  # Local: retain compact public inventory facts; global: distinguish inaccessible metadata from missing files.
        with ThreadPoolExecutor(max_workers=4) as pool:  # Local: limit independent HTTP requests; global: keep the audit bounded.
            result["copies"] = list(pool.map(lambda name: verify_copy(name, info.sha), original_files()))  # Local: collect every independent result; global: retain successful and failed artifact checks together.
    except Exception as error:  # Local: preserve metadata/inventory failures; global: still inspect the independent GitHub publication.
        result["hf_inventory"] = safe_error(error)  # Local: exclude response bodies and exception text; global: avoid signed URLs in failed audit records.
    media = verify_media(args.github_revision)  # Local: check a separately pinned README; global: verify delivery independently of physics and Hub content.
    result.update(media=media, readme_git_revision=media["readme"].get("git_revision"))  # Local: expose both publication revisions explicitly; global: distinguish them from the local audit checkout.
    save_json(args.output, result)  # Local: write the complete evidence before acceptance; global: retain mismatches for inspection.
    copies_pass = result["public"] and bool(result["copies"]) and all(row.get("unchanged_from_git") and row.get("public_copy_matches") and row["status"] == "ok" for row in result["copies"])  # Local: require nonempty completed comparisons; global: never accept skipped or failed evidence.
    media_pass = media["readme"].get("status") == "ok" and media["readme_video_players"] == 3 and len(media["videos"]) == 3 and all(row["status"] == "ok" and row.get("http_status") == 200 and row.get("content_type") == "video/mp4" for row in media["videos"])  # Local: require each recorded media check; global: do not infer availability from absent data.
    if not copies_pass or not media_pass:  # Local: use an unconditional runtime guard; global: preserve failure behavior under Python optimization.
        logging.error("Publication verification failed: copies_pass=%s media_pass=%s; evidence=%s", copies_pass, media_pass, args.output)  # Local: log safe acceptance facts; global: keep failed invocations reviewable.
        raise RuntimeError("Publication verification failed; inspect the saved evidence JSON")  # Local: return a failing command status; global: avoid claiming a partial check succeeded.
    logging.info("Publication verification passed: %s compact files; revision=%s", len(result["copies"]), result["hf_revision"])  # Local: summarize actual checks; global: expose completion clearly.


if __name__ == "__main__":  # Local: require explicit execution; global: keep imports free of network and filesystem side effects.
    main()  # Local: run the readable phases; global: preserve one command for audit reproduction.

"""Local: check publication evidence without network. Global: prevent false preservation claims.

Sources: https://docs.pytest.org/en/stable/how-to/monkeypatch.html and
https://docs.python.org/3.12/library/urllib.error.html describe the test seams and HTTP failures.
"""

from io import BytesIO  # Local: emulate downloaded bytes; global: keep tests independent of service availability.
import json  # Local: inspect the written audit; global: verify that failure evidence survives.
from pathlib import Path  # Local: build an isolated project; global: leave scientific inputs untouched.
from types import SimpleNamespace  # Local: supply only used metadata fields; global: avoid a network client mock framework.
from urllib.error import HTTPError  # Local: reproduce a missing public file; global: test safe error serialization.

import pytest  # Local: parameterize formats and failures; global: cover distinct comparison contracts.
from scripts import verify_documentation_publication as audit  # Local: exercise the actual verifier; global: avoid duplicating its logic.


def test_original_files_only_selects_the_five_experiment_bundles(monkeypatch):
    """Local: exclude similarly shaped paths. Global: never map paper figures to experiments/."""
    names = "\n".join(["reports/hc-classic-001/report.pdf", "reports/ant-classic-005/results.json", "reports/paper_assets/ant-topdown-validation.png", "reports/new-experiment/results.json", "reports/ant-classic-005/audit/nested.json", "reports/hc-classic-001/README.md"])  # One source tree contains both valid evidence and unrelated documents.
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *args, **kwargs: names)  # Supply a Git listing without depending on checkout contents.
    assert audit.original_files() == ["reports/hc-classic-001/report.pdf", "reports/ant-classic-005/results.json"]  # Only the explicit experiment roots belong in the public-copy check.


@pytest.mark.parametrize("suffix,local,original,remote,expected", [
    (".json", b'{"n": 2}', b'{\n  "n": 2\n}', b'{"n":2}', True),  # JSON whitespace is outside the scientific-value identity.
    (".md", b"one\r\ntwo\r\n", b"one\ntwo\n", b"one\ntwo\n", True),  # Git checkout newlines must not create a false report change.
    (".npz", b"original", b"original", b"changed", False),  # Binary evidence must match exactly, not approximately.
])
def test_copy_comparison_uses_the_documented_format_contract(tmp_path, monkeypatch, suffix, local, original, remote, expected):
    """Local: compare three representations. Global: separate byte identity from documented normalization."""
    monkeypatch.chdir(tmp_path)  # Restrict fixture writes to pytest's project directory.
    name = f"reports/ant-classic-005/evidence{suffix}"  # Exercise the ordinary bundle mapping.
    path = Path(name)  # The verifier deliberately reads the working copy.
    path.parent.mkdir(parents=True)  # Create only this fixture's artifact directory.
    path.write_bytes(local)  # Preserve the test's deliberate byte representation.
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *args, **kwargs: original)  # Supply the independent Git bytes.
    monkeypatch.setattr(audit, "urlopen", lambda *args, **kwargs: BytesIO(remote))  # A context-managed in-memory response follows the download interface.
    result = audit.verify_copy(name, "a" * 40)  # Use an immutable-looking public revision in the recorded URL.
    assert result["status"] == "ok" and result["unchanged_from_git"]  # All local fixtures preserve the original according to their declared format.
    assert result["public_copy_matches"] is expected  # A changed binary must remain a failed comparison.


def test_missing_public_file_is_a_safe_error_row(tmp_path, monkeypatch):
    """Local: retain an HTTP failure. Global: never serialize signed redirect URLs or server text."""
    monkeypatch.chdir(tmp_path)  # Avoid reading or writing any real experiment artifacts.
    name = "reports/ant-classic-005/results.json"  # The requested URL itself is stable and public.
    Path(name).parent.mkdir(parents=True)  # Create one valid local source.
    Path(name).write_bytes(b"{}")  # Download failure is independent of the local content.
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *args, **kwargs: b"{}")  # The original Git content is available.
    def unavailable(*args, **kwargs):
        """Local: emulate a redirected failure. Global: expose accidental URL leakage in diagnostics."""
        raise HTTPError("https://cdn.example/file?signature=PRIVATE", 404, "PRIVATE server text", {}, None)  # Neither the redirected URL nor message belongs in published evidence.
    monkeypatch.setattr(audit, "urlopen", unavailable)  # Fail at the external-read boundary.
    result = audit.verify_copy(name, "a" * 40)  # A missing file must be represented rather than aborting the inventory.
    assert result["status"] == "error" and result["error_type"] == "HTTPError" and result["http_status"] == 404  # Keep actionable, nonsensitive failure facts.
    assert "PRIVATE" not in json.dumps(result) and "signature=" not in json.dumps(result)  # The output must exclude raw exception content.


def test_media_checks_continue_after_missing_attachment_and_pin_the_readme(monkeypatch):
    """Local: retain one missing video. Global: still identify the exact public README that was inspected."""
    class Response(BytesIO):
        """Local: emulate public response metadata. Global: avoid a live GitHub dependency in the test."""
        status = 200  # The two available attachments return success.
        headers = SimpleNamespace(get_content_type=lambda: "video/mp4")  # Only the safe format header is consumed.
    requested = []  # Record stable requested addresses without any redirected URLs.
    def fetch(request, **kwargs):
        """Local: serve a fixed commit and page. Global: make moving-main mistakes observable."""
        url = request.full_url if hasattr(request, "full_url") else request  # Accept both HEAD Request objects and GET strings.
        requested.append(url)  # Preserve request order for the exact-revision assertion.
        if audit.MEDIA[0] in url:
            raise HTTPError("https://cdn.example?signature=PRIVATE", 404, "PRIVATE", {}, None)  # One failed attachment must not abort the remaining checks.
        if "/commits/" in url:
            return Response(json.dumps({"sha": "c" * 40}).encode())  # Resolve the requested branch once.
        return Response(b"<video></video>" * 3)  # The pinned tree page contains three rendered players.
    monkeypatch.setattr(audit, "urlopen", fetch)  # All external reads remain simulated.
    result = audit.verify_media("main")  # The verifier must inspect a resolved revision rather than the moving repository root.
    assert len(result["videos"]) == 3 and result["videos"][0]["status"] == "error"  # Retain the missing attachment and both independent later checks.
    assert result["readme"]["git_revision"] == "c" * 40 and result["readme_video_players"] == 3  # Keep successful rendering evidence despite the earlier failure.
    assert requested[-1].endswith("/tree/" + "c" * 40) and "PRIVATE" not in json.dumps(result)  # The rendered page is pinned and signed URLs remain absent.


@pytest.mark.parametrize("copies", [[], [{"status": "ok", "unchanged_from_git": True, "public_copy_matches": True}, {"status": "error", "error_type": "HTTPError", "http_status": 404}]])
def test_failed_or_empty_inventory_is_saved_before_command_fails(tmp_path, monkeypatch, copies):
    """Local: reject incomplete checks. Global: preserve all collected evidence even on command failure."""
    monkeypatch.chdir(tmp_path)  # Keep the output beneath a temporary project root.
    output = tmp_path / "audit.json"  # The same JSON remains inspectable after RuntimeError.
    monkeypatch.setattr(audit.sys, "argv", ["verify_documentation_publication.py", "--output", str(output)])  # Exercise the public command's ordinary arguments.
    monkeypatch.setattr(audit, "start_logging", lambda *args: None)  # Avoid replacing pytest's logging handlers.
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *args, **kwargs: "b" * 40)  # Supply the current local Git revision.
    monkeypatch.setattr(audit, "HfApi", lambda **kwargs: SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40, private=False, siblings=[])))  # Metadata says public but cannot make an empty inventory pass.
    monkeypatch.setattr(audit, "original_files", lambda: list(range(len(copies))))  # Each row represents a completed independent file check.
    monkeypatch.setattr(audit, "verify_copy", lambda number, revision: copies[number])  # Include a successful check before the missing file.
    monkeypatch.setattr(audit, "verify_media", lambda *args: {"videos": [{"status": "ok", "http_status": 200, "content_type": "video/mp4"}] * 3, "readme_video_players": 3, "readme": {"status": "ok", "git_revision": "c" * 40}})  # Isolate inventory acceptance from unrelated media availability.
    with pytest.raises(RuntimeError):
        audit.main()  # Explicit exceptions must guard acceptance even when Python assertions are disabled.
    saved = json.loads(output.read_text(encoding="utf-8"))  # Read the result after command failure.
    assert saved["copies"] == copies and saved["git_revision"] == "b" * 40  # Neither failures nor successful rows may disappear.
    assert saved["readme_git_revision"] == "c" * 40 and saved["hf_revision"] == "a" * 40  # Different publication systems need separate immutable identities.
    assert saved["command"] and len(saved["script_sha256"]) == 64  # The verifier invocation and actual script must remain identifiable.

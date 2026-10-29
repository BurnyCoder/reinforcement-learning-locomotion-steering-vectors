"""Saved-data publication contracts; synthetic arrays test mechanics, not steering efficacy."""

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest

from rl_locomotion_steering_vectors.publication.evidence import _contact_clipping, _saved_effect, paired_summary, trajectory, summarize_episode
from rl_locomotion_steering_vectors.publication.inputs import canonical_json, download_input, validate_inputs
from scripts.build_ant_publication import compare_outputs


def episode(seed=1, length=120, y=2.0):
    """Finite controlled trajectories make retained failures and displacement checkable."""
    item = {key: np.zeros(length) for key in ("x_velocity", "y_velocity", "planar_speed", "height", "yaw_rate", "effort", "inversion", "bad_contact", "reward", "saturation")}
    item.update(seed=np.array(seed), length=np.array(length), dt=np.array(.05), terminated=np.array(False), truncated=np.array(True), onset_hash=np.array("same"))
    item["x_velocity"][:] = 3
    item["y_velocity"][:] = y
    item["reward"][:] = np.arange(length)
    return item


def test_prefix_failure_remains_in_equal_episode_pair():
    baseline = {1: episode(1), 2: episode(2, length=34)}
    treated = {1: episode(1, y=4), 2: episode(2, length=34)}
    result = paired_summary(baseline, treated, warmup=100)
    assert result["mean"] == 1.0
    assert summarize_episode(treated[2], 100)["failure"] == 1
    assert summarize_episode(treated[2], 100)["y_velocity"] == 0


@pytest.mark.parametrize("fault", ["missing", "onset", "nonfinite"])
def test_pairing_rejects_incomplete_or_corrupt_evidence(fault):
    baseline = {1: episode(), 2: episode(2)}
    treated = {1: episode(), 2: episode(2)}
    if fault == "missing":
        treated.pop(2)
    elif fault == "onset":
        treated[2]["onset_hash"] = np.array("different")
    else:
        treated[2]["reward"][110] = np.nan
    with pytest.raises(ValueError):
        paired_summary(baseline, treated, warmup=100)


def test_downsample_preserves_onset_endpoint_and_block_reward():
    item = episode(length=123)
    reduced = trajectory(item, warmup=100, stride=10)
    assert reduced["steps"][0] == 0 and reduced["steps"][-1] == 123
    assert 100 in reduced["steps"]
    assert reduced["x"][-1] == pytest.approx(123 * 3 * .05)
    assert reduced["y"][-1] == pytest.approx(123 * 2 * .05)
    assert reduced["reward"][0] is None
    assert reduced["reward"][1] == 4.5
    assert reduced["reward"][-1] == 121


def test_manifest_hashes_exact_inputs_and_rejects_escape(tmp_path):
    target = tmp_path / "evidence.json"
    target.write_text("{}", encoding="utf-8")
    manifest = {"schema_version": 1, "repo_id": "owner/repo", "revision": "a" * 40, "files": [{"local_path": "evidence.json", "hub_path": "evidence.json", "sha256": hashlib.sha256(b"{}").hexdigest(), "size": 2}]}
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_inputs(tmp_path, path)["files_verified"] == 1
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_inputs(tmp_path, path)
    manifest["files"][0]["local_path"] = "../outside.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside|relative"):
        validate_inputs(tmp_path, path)


def test_missing_input_is_not_silently_replaced(tmp_path):
    manifest = {"schema_version": 1, "repo_id": "owner/repo", "revision": "a" * 40, "files": [{"local_path": "missing.json", "hub_path": "missing.json", "sha256": "a" * 64, "size": 2}]}
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        validate_inputs(tmp_path, path)
    assert not (tmp_path / "missing.json").exists()


def test_serialization_is_deterministic_and_rejects_nan():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    with pytest.raises(ValueError):
        canonical_json({"unknown": float("nan")})


def test_historical_interval_is_reused_and_raw_disagreement_is_rejected():
    saved = {"behavior": "lateral", "effect": {"effect": .4, "ci95": [.3, .5], "gate": False}}
    observed = {"mean": .4, "ci95": [.2, .6]}
    result = _saved_effect(saved, {}, "confirmation", "lateral__-0.1000", observed)
    assert result["ci95"] == [.3, .5] and result["gate"] is False
    with pytest.raises(ValueError, match="disagrees"):
        _saved_effect(saved, {}, "confirmation", "lateral__-0.1000", {"mean": .7})


def test_contact_clipping_counts_states_not_channels_and_excludes_prefix():
    item = episode(length=104)
    item["observations"] = np.zeros((104, 105))
    item["observations"][99, 27] = 1  # Before the intervention window.
    item["observations"][100, [27, 28]] = [1, -1]  # Two clipped channels are one state.
    item["observations"][101, 29] = 1  # Vertical force is not changed by planar rotation.
    item["observations"][102, 30] = -1  # Planar torque is included.
    prefix = episode(2, length=34)
    prefix["observations"] = np.ones((34, 105))
    result = _contact_clipping({1: item, 2: prefix}, 100)
    assert result["states"] == 4 and result["states_saturated"] == 2
    assert result["fraction"] == .5 and result["prefix_only_episodes"] == 1


def test_publication_imports_no_policy_or_simulator():
    code = "import sys; from rl_locomotion_steering_vectors.publication import evidence, figures, inputs; assert not any(x in sys.modules for x in ('torch', 'mujoco', 'gymnasium', 'stable_baselines3', 'rl_locomotion_steering_vectors.runtime'))"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_check_reports_stale_and_obsolete_outputs_without_changing_them(tmp_path):
    root, stage = tmp_path / "project", tmp_path / "stage"
    relative = Path("site/data/episodes/test.json")
    for base in (root, stage):
        (base / relative).parent.mkdir(parents=True)
    (root / relative).write_bytes(b"old")
    (stage / relative).write_bytes(b"new")
    (root / relative.parent / "obsolete.json").write_bytes(b"keep")
    assert compare_outputs(root, stage, [stage / relative]) == [relative.as_posix(), "obsolete generated shard: site/data/episodes/obsolete.json"]
    assert (root / relative).read_bytes() == b"old"
    assert (root / relative.parent / "obsolete.json").read_bytes() == b"keep"


def test_public_download_uses_short_verified_content_address(tmp_path, monkeypatch):
    payload = b"saved scientific evidence"
    calls = []
    def response(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)
    monkeypatch.setattr("rl_locomotion_steering_vectors.publication.inputs.urlopen", response)
    record = {"hub_path": "long/" * 60 + "episode.npz", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    path = download_input(tmp_path, record, "owner/repo", "a" * 40)
    assert path == tmp_path / ".cache/ant-inputs" / record["sha256"]
    assert path.read_bytes() == payload and len(calls) == 1
    assert "/resolve/" + "a" * 40 + "/" in calls[0]
    assert download_input(tmp_path, record, "owner/repo", "a" * 40) == path
    assert len(calls) == 1  # A verified content-addressed cache avoids duplicate HTTP downloads.


def test_invalid_download_never_becomes_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("rl_locomotion_steering_vectors.publication.inputs.urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt"))
    record = {"hub_path": "episode.npz", "sha256": hashlib.sha256(b"expected").hexdigest(), "size": len(b"expected")}
    with pytest.raises(ValueError, match="hash/size"):
        download_input(tmp_path, record, "owner/repo", "a" * 40)
    assert list((tmp_path / ".cache/ant-inputs").iterdir()) == []


def test_download_retries_are_bounded_and_redirect_details_are_not_exposed(tmp_path, monkeypatch):
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(1)
        raise URLError("https://signed.example/?private-signature=secret")
    monkeypatch.setattr("rl_locomotion_steering_vectors.publication.inputs.urlopen", unavailable)
    monkeypatch.setattr("rl_locomotion_steering_vectors.publication.inputs.time.sleep", lambda seconds: None)
    record = {"hub_path": "public/episode.npz", "sha256": "a" * 64, "size": 1}
    with pytest.raises(RuntimeError) as caught:
        download_input(tmp_path, record, "owner/repo", "a" * 40)
    assert len(calls) == 3 and "public/episode.npz" in str(caught.value)
    assert "secret" not in str(caught.value) and "signed.example" not in str(caught.value)
    assert list((tmp_path / ".cache/ant-inputs").iterdir()) == []

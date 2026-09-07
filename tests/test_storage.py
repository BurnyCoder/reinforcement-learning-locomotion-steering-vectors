"""Local: test resumable artifacts; global: prevent silent experiment mixing.

Source: https://docs.pytest.org/en/stable/how-to/tmp_path.html
"""
import numpy as np  # Local arrays stand in for simulator data.
import pytest  # Explicit failures protect the experiment boundary.

from rl_locomotion_steering_vectors.storage import load_arrays, save_arrays, save_json, load_json, check_manifest, ensure_identity


def test_array_round_trip(tmp_path):
    """Local: round-trip numeric/string data; global: avoid executable pickle."""
    path = tmp_path / "episode.npz"  # Isolate the test from measured artifacts.
    save_arrays(path, {"values": np.arange(4), "hash": np.asarray("abc")})  # Save supported types.
    result = load_arrays(path)  # Load through the exact public artifact API.
    np.testing.assert_array_equal(result["values"], np.arange(4))  # Require lossless numbers.
    assert result["hash"].item() == "abc"  # Preserve simulator state identifiers.


def test_object_arrays_are_rejected(tmp_path):
    """Local: reject Python objects; global: keep vector bundles non-executable."""
    with pytest.raises(ValueError, match="object"):
        save_arrays(tmp_path / "bad.npz", {"bad": np.array([{}], dtype=object)})


def test_manifest_prevents_changed_resume(tmp_path):
    """Local: compare immutable fields; global: do not reuse incompatible rollouts."""
    path = tmp_path / "manifest.json"  # One manifest represents one protocol.
    expected = {"model_key": "halfcheetah", "config": {"warmup": 100}, "seed_splits": {"fit": [1]}}
    save_json(path, expected)  # Establish an existing experiment.
    check_manifest(load_json(path), expected)  # Exact resumption is legitimate.
    changed = dict(expected, config={"warmup": 10})  # A changed intervention onset is another experiment.
    with pytest.raises(ValueError, match="config"):
        check_manifest(load_json(path), changed)


def test_condition_identity_is_immutable(tmp_path):
    """Local: reject changed interventions; global: prevent stale rollout cache reuse."""
    path = tmp_path / "condition" / "identity.json"  # One cache directory owns one scientific condition.
    ensure_identity(path, {"alpha": .1, "vector_hash": "first"})  # Persist before collecting its episodes.
    ensure_identity(path, {"alpha": .1, "vector_hash": "first"})  # The same condition may resume.
    with pytest.raises(ValueError, match="identity"):
        ensure_identity(path, {"alpha": .1, "vector_hash": "changed"})  # Names alone cannot identify a vector.


def test_unidentified_old_episode_cache_is_rejected(tmp_path):
    """Local: reject unverifiable files; global: require audited migration of old evidence."""
    save_arrays(tmp_path / "1.npz", {"value": np.zeros(1)})  # Simulate an older cache missing provenance.
    with pytest.raises(ValueError, match="identity"):
        ensure_identity(tmp_path / "identity.json", {"alpha": .1})  # Never silently bless unknown cached data.

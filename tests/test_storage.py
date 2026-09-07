"""Local: test resumable artifacts; global: prevent silent experiment mixing.

Source: https://docs.pytest.org/en/stable/how-to/tmp_path.html
"""
import numpy as np  # Local arrays stand in for simulator data.
import pytest  # Explicit failures protect the experiment boundary.

from rl_locomotion_steering_vectors.storage import load_arrays, save_arrays, save_json, load_json, check_manifest


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

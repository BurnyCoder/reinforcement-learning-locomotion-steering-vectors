"""Local: test application target and seed rules. Global: prevent retuning or hidden failed episodes in the practical demonstration.

Sources: https://docs.pytest.org/en/stable/how-to/assert.html and the existing
project episode-level quality criterion in analysis.py.
"""

import pytest  # Local: assert application gate failures; global: drive implementation from explicit acceptance criteria.
import imageio.v2 as imageio  # Local: exercise real streamed video I/O; global: verify the exported comparison path.
import numpy as np  # Local: create tiny deterministic frame fixtures; global: avoid consuming fresh application trajectories in tests.

from rl_locomotion_steering_vectors.demonstration import assess_target, fresh_application_seeds, _paired_video  # Local: import decision and media helpers; global: test acceptance and faithful failure visualization.


def _row(seed, speed, bad_contact=0.0):
    """Local: construct a complete scalar episode summary. Global: exercise the same analysis interface used by actual application data."""
    return dict(seed=seed, x_velocity=speed, y_velocity=0.0, planar_speed=abs(speed), height=0.6,  # Local: supply physical behavior; global: preserve the runtime summary schema.
                yaw_rate=0.0, effort=0.5, inversion=0.0, bad_contact=bad_contact, reward=10.0, saturation=0.0,  # Local: include quality and ancillary metrics; global: reuse analysis rather than a duplicate failure rule.
                failure=0.0, length=1000, onset_hash=str(seed))  # Local: represent a complete episode; global: ensure target scoring includes all resets.


def test_target_requires_eighty_percent_within_locked_tolerance():
    """Local: check the predeclared target coverage threshold. Global: prohibit post-hoc tolerance expansion."""
    rows = [_row(seed, 12.5 if seed < 8 else 15.0) for seed in range(10)]  # Local: create eight in-band and two out-of-band episodes; global: test the exact acceptance boundary.
    result = assess_target(rows, target=12.0, tolerance=0.5)  # Local: lock target and band before scoring; global: use one calibration-derived objective.
    assert result["success_fraction"] == 0.8 and result["gate"]  # Local: accept the inclusive boundary; global: implement the prespecified 80% criterion.
    assert len(result["episodes"]) == 10  # Local: preserve all errors; global: avoid excluding misses from the denominator.
    assert result["mean_absolute_error"] == pytest.approx(1.0)  # Local: average all ten absolute errors; global: report misses even when the gate passes.
    rows[7]["x_velocity"] = 15.0  # Local: move one episode out of band; global: test failure below the coverage threshold.
    assert not assess_target(rows, target=12.0, tolerance=0.5)["gate"]  # Local: reject seven successful episodes; global: enforce the fixed criterion.


def test_single_physical_failure_disqualifies_application_even_at_target():
    """Local: classify failures per episode. Global: prevent a collapse from hiding in a low pooled contact fraction."""
    rows = [_row(seed, 12.0, bad_contact=0.4 if seed == 4 else 0.0) for seed in range(10)]  # Local: keep all speeds at target but one episode unhealthy; global: separate targeting from competence.
    result = assess_target(rows, target=12.0, tolerance=0.5)  # Local: reuse the registered quality rule; global: test the zero-failure application requirement.
    assert result["success_fraction"] == 1.0  # Local: acknowledge target matching; global: avoid conflating independent acceptance conditions.
    assert result["failed_episode_seeds"] == [4] and not result["gate"]  # Local: retain the individual failure; global: reject an apparently successful aggregate.


def test_seed_check_includes_source_manifest_evaluation_splits():
    """Local: check current and inherited seed partitions. Global: keep application data fresh after a retargeted experiment."""
    manifest = {"config": {"seed_offset": 200000}, "seed_splits": {"fit": [101000], "replication": [230000]}}  # Local: represent a retargeted run; global: preserve the source fitting split.
    assert fresh_application_seeds(manifest) == list(range(240000, 240010))  # Local: reserve the fixed ten application seeds; global: make reproduction deterministic.
    source = {"seed_splits": {"confirmation": [240007]}}  # Local: introduce an upstream collision; global: test more than current-run disjointness.
    with pytest.raises(ValueError, match="240007"):  # Local: require a precise overlap error; global: prevent accidental reuse of observed outcomes.
        fresh_application_seeds(manifest, [source])  # Local: inspect the source snapshot too; global: preserve the scientific holdout boundary.


def test_target_rejects_empty_or_invalid_specifications():
    """Local: validate target scoring inputs. Global: do not turn missing or nonfinite data into a successful demonstration."""
    with pytest.raises(ValueError):  # Local: reject no evidence; global: avoid a vacuous coverage claim.
        assess_target([], target=12.0, tolerance=0.5)  # Local: pass an empty cohort; global: test explicit failure.
    with pytest.raises(ValueError):  # Local: reject invalid calibration; global: keep acceptance bands meaningful.
        assess_target([_row(0, 12.0)], target=12.0, tolerance=float("nan"))  # Local: pass a nonfinite band; global: prevent undefined comparisons.


def test_streamed_comparison_keeps_frames_after_one_episode_ends(tmp_path):
    """Local: compose unequal-length tiny movies. Global: prevent a short failed episode from truncating the longer comparison."""
    paths = [tmp_path / "baseline.mp4", tmp_path / "steered.mp4"]  # Local: isolate media fixtures; global: keep tests out of scientific evidence folders.
    for side, path in enumerate(paths):  # Local: create one stream per condition; global: exercise the production decoder/encoder interface.
        with imageio.get_writer(str(path), fps=20, codec="libx264", macro_block_size=1) as writer:  # Local: use the same codec as real output; global: test actual media compatibility.
            for frame in range(3 - side):  # Local: terminate one stream a frame earlier; global: model failed short episodes.
                writer.append_data(np.full((32, 48, 3), 40 + frame * 40, dtype=np.uint8))  # Local: make deterministic distinguishable frames; global: avoid any policy/simulator dependency.
    output = tmp_path / "paired.mp4"  # Local: choose a fresh output artifact; global: verify the composed movie independently.
    _paired_video(*paths, output, seed=0, alpha=0.1)  # Local: call the actual streamed compositor; global: test its failure-preserving behavior.
    with imageio.get_reader(str(output)) as reader:  # Local: decode the completed movie; global: validate produced rather than merely requested output.
        assert reader.count_frames() == 3  # Local: retain the longer sequence; global: ensure the short episode is visibly held at its final frame.
        assert reader.get_data(2).shape == (68, 96, 3)  # Local: check both panels and caption bar; global: verify the shareable video layout.

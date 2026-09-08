"""Local: test application target and seed rules. Global: prevent retuning or hidden failed episodes in the practical demonstration.

Sources: https://docs.pytest.org/en/stable/how-to/assert.html and the existing
project episode-level quality criterion in analysis.py.
"""

import pytest  # Local: assert application gate failures; global: drive implementation from explicit acceptance criteria.
import imageio.v2 as imageio  # Local: exercise real streamed video I/O; global: verify the exported comparison path.
import numpy as np  # Local: create tiny deterministic frame fixtures; global: avoid consuming fresh application trajectories in tests.

from rl_locomotion_steering_vectors.demonstration import assess_target, fresh_application_seeds, _paired_video  # Local: import decision and media helpers; global: test acceptance and faithful failure visualization.
from rl_locomotion_steering_vectors import demonstration  # Local: isolate expensive rollout/media boundaries; global: test pre-data specification and the unchanged acceptance workflow.
from rl_locomotion_steering_vectors.config import Config  # Local: create an explicit test protocol; global: keep application seeds and switch timing realistic.
from rl_locomotion_steering_vectors.storage import load_json, save_arrays, save_json  # Local: exercise actual immutable JSON/NPZ inputs; global: protect the saved candidate identity.


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


@pytest.mark.parametrize("target", [-.4, 0., .4])
def test_lateral_targets_accept_signed_velocity_and_retain_zero_failure_gate(target):
    """Local: score signed lateral commands; global: preserve the exact targeting and physical-quality conjunction."""
    rows = [dict(_row(seed, 2.), y_velocity=target + (.1 if seed < 8 else .3)) for seed in range(10)]  # Local: retain positive forward progress with eight lateral target matches; global: isolate signed target semantics.
    result = assess_target(rows, target=target, tolerance=.2, behavior="lateral")  # Local: allow a negative or zero world-y target; global: avoid importing a positive-speed assumption.
    assert result["gate"] and result["success_fraction"] == .8  # Local: accept the same coverage threshold; global: preserve the registered 80% rule.
    assert result["behavior"] == "lateral" and result["units"] == "m/s"  # Local: identify the measured coordinate; global: support unambiguous reporting.
    assert result["episodes"][0]["value"] == pytest.approx(target + .1)  # Local: expose the signed measurement; global: prevent confusing forward and lateral speeds.
    rows[0]["bad_contact"] = .4  # Local: make one well-targeted episode physically fail; global: test the existing per-episode quality criterion.
    failed = assess_target(rows, target=target, tolerance=.2, behavior="lateral")  # Local: retain target coverage despite the collapse; global: evaluate the independent competence condition.
    assert failed["failed_episode_seeds"] == [0] and not failed["gate"]  # Local: reject any treated physical failure; global: do not hide a collapse in pooled contact fractions.


@pytest.mark.parametrize("behavior, expected_gate", [("lateral", True), ("speed", False)])
def test_application_locks_behavior_rules_and_retains_baseline_failures(tmp_path, monkeypatch, behavior, expected_gate):
    """Local: mock only simulator/media work; global: preserve the speed rule and register lateral baseline handling before new outcomes."""
    config = Config(seed_offset=500000)  # Local: reserve application seeds 540000 through 540009; global: match the new Ant cohort convention.
    vector = np.arange(256, dtype=np.float32)  # Local: create a fixed imported treatment; global: detect any application-time vector modification.
    chosen = dict(vector=behavior, behavior=behavior, alpha=-.1, direction=1 if behavior == "lateral" else -1, role="candidate")  # Local: supply an already replicated condition; global: avoid application-time selection from unconfirmed candidates.
    target, reference = (-.1, -.6) if behavior == "lateral" else (10., 12.)  # Local: use a signed lateral target or the existing positive-speed case; global: test both acceptance specifications.
    manifest = dict(model_key="ant" if behavior == "lateral" else "halfcheetah", config=config.as_dict(), provenance={"parameter_sha256": "frozen"}, seed_splits={"replication": [530000]})  # Local: preserve policy and seed identities; global: retain the normal application contract.
    save_json(tmp_path / "manifest.json", manifest)  # Local: establish the completed run identity; global: exercise real artifact loading.
    save_json(tmp_path / "results.json", {"replicated": [chosen], "validation": [dict(chosen, effect={"treated_mean": target, "baseline_mean": reference})]})  # Local: provide validation-only calibration; global: prohibit target fitting on fresh episodes.
    save_arrays(tmp_path / "vectors.npz", {behavior: vector})  # Local: retain exact numerical directions; global: make this an application of an existing vector.
    model = object()  # Local: identify a fixed supplied policy without allocating another actor; global: detect accidental replacement at the delegated boundaries.
    monkeypatch.setattr(demonstration, "prepare_model", lambda *args: (model, {"parameter_sha256": "frozen"}))  # Local: replace only checkpoint I/O; global: retain the production identity comparison.
    monkeypatch.setattr(demonstration, "summaries", lambda rows, config: rows)  # Local: use already-complete scalar summaries; global: leave the actual analysis failure and paired-effect functions intact.

    def collect(actual_model, model_key, run_dir, phase, seeds, config, **kwargs):
        """Local: supply prespecified paired measurements; global: assert registration precedes the first application outcome."""
        assert actual_model is model and seeds == list(range(540000, 540010))  # Local: preserve policy/cohort identity; global: prohibit application retuning or reset reuse.
        spec = load_json(run_dir / "application_spec.json")  # Local: require a real saved contract before collection; global: make the chronological boundary testable.
        assert spec["behavior"] == behavior and spec["baseline_failure_rule"]  # Local: freeze the behavior-specific rule; global: prevent after-the-fact baseline exclusions.
        assert spec["tolerance_rule"] == ("fixed 0.2 m/s" if behavior == "lateral" else "5% of validation baseline speed")  # Local: compare the registered band definition; global: retain the unchanged speed application.
        treated = "condition" in kwargs  # Local: distinguish the existing paired conditions; global: avoid replacing the numerical engine's role.
        if treated:
            np.testing.assert_array_equal(kwargs["vector"], vector)  # Local: require the original vector values; global: forbid fitting a new direction during application.
        return [dict(_row(seed, 2. if behavior == "lateral" else (10. if treated else 12.)),
                     y_velocity=(target if treated else reference) if behavior == "lateral" else 0., failure=float(not treated and seed == seeds[0])) for seed in seeds]  # Local: retain one baseline failure and no treated failures; global: isolate the intended behavior-dependent acceptance rule.

    monkeypatch.setattr(demonstration, "collect_episodes", collect)  # Local: prevent any real fresh application simulation in software tests; global: keep scientific holdouts untouched.
    monkeypatch.setattr(demonstration, "_media", lambda *args: {"videos_complete": True, "policy_function_restored": True})  # Local: avoid rendering fake measurements; global: retain the existing media boundary unchanged.
    result = demonstration.demonstrate(tmp_path)  # Local: run the normal application wrapper; global: test the actual selection/specification/acceptance data flow.
    assert result["baseline_failed_episode_seeds"] == [540000]  # Local: report the baseline failure even when lateral application qualifies; global: prohibit survivor filtering.
    assert result["paired_effect"]["gate"] and result["gate"] == expected_gate  # Local: retain the common paired usefulness gate while preserving the stricter historical speed baseline rule.
    assert result["failed_episode_seeds"] == []  # Local: require zero treated physical failures; global: enforce the same practical competence criterion for both behaviors.
    spec = load_json(tmp_path / "application_spec.json")  # Local: inspect the original locked rule; global: make resumption identity concrete.
    spec["baseline_failure_rule"] = "changed after outcomes"  # Local: simulate a post-hoc acceptance revision; global: ensure the attempt cannot be silently reinterpreted.
    save_json(tmp_path / "application_spec.json", spec)  # Local: alter only the saved rule; global: leave vector and calibration fixed.
    with pytest.raises(ValueError, match="specification changed"):
        demonstration.demonstrate(tmp_path)  # Local: reject before collecting another outcome; global: protect application evidence from rule drift.


def test_lateral_plot_uses_world_xy_displacement_and_four_physical_panels(tmp_path, monkeypatch):
    """Local: inspect a rendered synthetic figure; global: ensure signed world-axis motion is integrated with the actual simulator timestep."""
    episode = dict(seed=np.array(1), dt=np.array(.5), x_velocity=np.array([1., 2., 3.]), y_velocity=np.array([-1., -2., -3.]), height=np.array([.5, .6, .7]))  # Local: provide analytically integrable velocities; global: distinguish displacement from velocity or position guesses.
    closed = []  # Local: retain the completed figure briefly; global: inspect actual plotting outputs rather than duplicate formulas.
    original_close = demonstration.plt.close  # Local: preserve the real cleanup function; global: release the test figure after inspecting it.
    monkeypatch.setattr(demonstration.plt, "close", lambda figure: closed.append(figure))  # Local: intercept final figure cleanup only; global: leave PNG/PDF rendering real.
    path = tmp_path / "lateral.png"  # Local: isolate synthetic visualization from research artifacts; global: do not spend application seeds for testing.
    demonstration._series_plot(path, episode, episode, episode, -.2, .2, Config(), behavior="lateral")  # Local: exercise the actual four-panel output; global: verify the Ant extension without changing the speed path.
    figure = closed[-1]  # Local: inspect the completed plotted object; global: validate meaningful axes and data.
    assert len(figure.axes) == 4 and path.exists() and path.with_suffix(".pdf").exists()  # Local: require all panels and both shareable formats; global: complete the visualization artifact contract.
    np.testing.assert_allclose(figure.axes[-1].lines[0].get_xdata(), [0., .5, 1.5, 3.])  # Local: integrate forward velocity from the origin; global: show actual world-x displacement in metres.
    np.testing.assert_allclose(figure.axes[-1].lines[0].get_ydata(), [0., -.5, -1.5, -3.])  # Local: preserve lateral sign and timestep scaling; global: prevent mirrored or dimensionally wrong planar paths.
    original_close(figure)  # Local: free Matplotlib resources; global: keep repeated test runs bounded.

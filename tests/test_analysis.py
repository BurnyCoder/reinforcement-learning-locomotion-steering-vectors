"""Local: synthetic numerical contracts. Global: prevent causal-analysis bookkeeping errors.

Sources: https://numpy.org/doc/1.26/reference/routines.testing.html and
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html.
"""

import numpy as np  # Array fixtures exercise the same numerical representation as rollouts.
import pytest  # Explicit failure assertions protect invalid-data boundaries.

from rl_locomotion_steering_vectors.analysis import fit_vectors, paired_effect, window_table  # Test public analysis contracts.


def episode(seed, speeds, hidden=None, effort=None):
    """Local: expand window specifications. Global: make independent synthetic episodes."""
    values = np.repeat(np.asarray(speeds, dtype=float), 100)  # One supplied value represents one full window.
    length = len(values) + 100  # The first 100 steps are the common unsteered prefix.
    activation = np.asarray(hidden if hidden is not None else [[value, 1] for value in speeds], dtype=float)  # Permit known directions.
    result = {"seed": seed, "length": length, "terminated": False, "truncated": True, "onset_hash": str(seed)}  # Match runtime metadata.
    result["activations"] = np.vstack([np.zeros((100, activation.shape[1])), np.repeat(activation, 100, axis=0)])  # Expand hidden windows.
    result["x_velocity"] = np.r_[np.zeros(100), values]  # Startup has intentionally different behavior.
    for name in ("y_velocity", "yaw_rate", "inversion", "bad_contact", "saturation"):
        result[name] = np.zeros(length)  # Supply every measurement needed by fitting and gates.
    result["height"] = np.full(length, 0.5)  # A healthy constant torso baseline isolates other descriptors.
    result["planar_speed"] = np.abs(result["x_velocity"])  # Keep planar speed consistent with straight motion.
    result["effort"] = np.r_[np.zeros(100), np.repeat(effort if effort is not None else np.ones(len(speeds)), 100)]  # Optional matched effort labels.
    result["reward"] = result["x_velocity"].copy()  # Returns remain separate from behavior descriptors.
    return result  # Each call owns its arrays so treatment edits cannot mutate baseline.


def summary(seed, speed=2.0, **changes):
    """Local: construct compact evaluation data. Global: test episode-level inference."""
    row = dict(seed=seed, x_velocity=speed, y_velocity=0.0, planar_speed=abs(speed), effort=0.5, height=0.5, yaw_rate=0.0, inversion=0.0, bad_contact=0.0, saturation=0.0, reward=2.0, terminated=False, truncated=True, length=1000, onset_hash=str(seed))  # Full scalar schema.
    row.update(changes)  # Apply only the requested treatment or failure.
    return row  # Caller controls independent reset identities.


def test_window_table_preserves_exclusions_and_ignores_partial_windows():
    """Local: reject unhealthy fitting windows. Global: make exclusions auditable."""
    item = episode(8, [1, 2, 3])  # Three full fitting windows follow startup.
    item["inversion"][200:206] = 1  # Six percent inversion makes only the second window ineligible.
    item["bad_contact"][300:306] = 1  # The third window independently fails contact health.
    table = window_table([item], warmup=100, window=100)  # Exclusions must remain visible in the table.
    np.testing.assert_array_equal(table["speed"], [1, 2, 3])  # Window means exclude startup.
    np.testing.assert_array_equal(table["eligible"], [True, False, False])  # The threshold is five percent.
    assert table["stats"]["eligible_windows"] == 1  # Counts support transparent fitting reports.
    short = episode(9, [1])  # Start with one complete post-startup window.
    short["length"] = 199  # Metadata truncates the episode before that window completes.
    with pytest.raises(ValueError, match="length"):
        window_table([short])  # Array/metadata disagreement is a defect, not an exclusion.
    short = episode(10, [1, 2])  # A real partial tail has consistently shortened arrays.
    for name, value in list(short.items()):
        if isinstance(value, np.ndarray):
            short[name] = value[:-1]  # Preserve alignment while losing the last timestep.
    short["length"] = 299  # One full and one incomplete fitting window now remain.
    partial = window_table([short])  # The final 99 steps must not be promoted to a full observation.
    assert len(partial["speed"]) == 1 and partial["stats"]["discarded_tail_steps"] == 99  # Both selection and exclusion count remain explicit.


def test_fit_balances_episodes_and_controls_have_identical_norm():
    """Local: prevent long episodes dominating. Global: retain independent-example weighting."""
    episodes = [episode(i, [0, 2], [[0, 0], [i, 2]]) for i in range(16)]  # Every episode contributes both contrast groups.
    episodes[-1] = episode(15, [0] + [2] * 9, [[0, 0]] + [[15, 2]] * 9)  # Duplicate one episode's high windows.
    vectors, diagnostics = fit_vectors(episodes, ["speed"])  # Fit the requested behavior and its controls.
    np.testing.assert_allclose(diagnostics["vectors"]["speed"]["raw_vector"], [7.5, 2])  # Equal episode weighting preserves the expected contrast.
    assert diagnostics["behaviors"]["speed"]["high_episodes"] == 16  # Independent contributors are counted explicitly.
    assert set(vectors) == {"speed", "speed_random0", "speed_random1", "speed_random2", "speed_shuffled"}  # Each candidate gets the planned controls.
    norms = [np.linalg.norm(value) for value in vectors.values()]  # Compare actual injected magnitudes.
    np.testing.assert_allclose(norms, norms[0], rtol=1e-6)  # Controls cannot win through larger norm alone.


def test_fit_reports_insufficient_independent_contributors_and_constant_labels():
    """Local: reject unsupported contrasts. Global: avoid normalizing accidental differences."""
    vectors, diagnostics = fit_vectors([episode(i, [i]) for i in range(16)], ["speed"])  # Quartiles contain only four episodes.
    assert not vectors  # Repeated timesteps cannot replace independent episodes.
    assert "independent" in diagnostics["behaviors"]["speed"]["reason"]  # Failure is actionable in the report.
    vectors, diagnostics = fit_vectors([episode(i, [2], [[i, 1]]) for i in range(32)], ["speed"])  # Activations vary while speed has no contrast.
    assert not vectors  # Do not invent behavioral groups from rank tie breaking.
    assert "contrast" in diagnostics["behaviors"]["speed"]["reason"]  # Explain the constant-label failure.


def test_effort_contrasts_within_speed_bins():
    """Local: balance speed strata. Global: effort direction must not merely encode speed."""
    episodes = [episode(i, [1, 1, 3, 3], [[1, 0], [1, 2], [3, 0], [3, 2]], [0, 1, 2, 3]) for i in range(16)]  # Effort and speed are deliberately correlated.
    vectors, diagnostics = fit_vectors(episodes, ["effort"])  # Conditional contrasts should isolate the second activation.
    raw = diagnostics["vectors"]["effort"]["raw_vector"]  # Inspect the unscaled result as well as injection norm.
    np.testing.assert_allclose(raw, [0, 2], atol=1e-12)  # Equal speed bins cancel the confounding first component.
    assert "effort" in vectors  # A real within-speed effort contrast remains available.


def test_centered_scale_uses_timestep_activations_not_only_window_means():
    """Local: preserve within-window variance. Global: strength has the documented RMS units."""
    episodes = [episode(i, [0, 2], [[0, 0], [2, 0]]) for i in range(16)]  # Identical lengths make the expected RMS direct.
    for item in episodes:
        item["activations"][100:, 1] = np.tile([-3, 3], 100)  # Gait variation cancels in each window mean.
    _, diagnostics = fit_vectors(episodes, ["speed"])  # Scaling must still retain this variation.
    assert diagnostics["activation_rms"] == pytest.approx(np.sqrt(10))  # Variance is one in x and nine in y.


def test_paired_effect_matches_seeds_and_bootstraps_whole_episodes():
    """Local: align paired resets. Global: uncertainty counts episodes rather than timesteps."""
    baseline = [summary(i, speed=2 + i / 100) for i in range(30)]  # Each seed starts from a different baseline.
    treated = [summary(i, speed=2.3 + i / 100) for i in reversed(range(30))]  # Reordering must not break pairing.
    result = paired_effect(baseline, treated, "speed")  # All paired deltas are exactly 0.3 m/s.
    assert result["gate"]  # The useful increase preserves healthy locomotion.
    assert result["n_pairs"] == 30  # No timestep is counted as an independent sample.
    np.testing.assert_allclose(result["ci95"], [0.3, 0.3], atol=1e-12)  # Shared-seed resampling preserves the constant effect.
    assert result["effect"] == pytest.approx(0.3)  # Effect units remain physical m/s.


@pytest.mark.parametrize("change", [{"seed": 99}, {"onset_hash": "different-state"}])
def test_paired_effect_rejects_unpaired_inputs(change):
    """Local: reject identity mismatches. Global: paired causal claims require equal starts."""
    baseline = [summary(i) for i in range(3)]  # Three independent paired starts are expected.
    treated = [summary(i, speed=2.3) for i in range(3)]  # Otherwise valid useful treatments.
    treated[-1].update(change)  # Break either seed membership or physical state equality.
    with pytest.raises(ValueError):
        paired_effect(baseline, treated, "speed")  # Refuse to silently discard mismatched episodes.


def test_quality_gate_rejects_contact_increase_and_counts_failures():
    """Local: retain failure episodes. Global: disruptive speed changes are not useful control."""
    baseline = [summary(i) for i in range(30)]  # Clean locomotion is the quality reference.
    treated = [summary(i, speed=2.3, bad_contact=0.06) for i in range(30)]  # The target improves but contact exceeds tolerance.
    assert not paired_effect(baseline, treated, "speed")["gate"]  # Useful-effect gating includes locomotion preservation.
    treated = [summary(i, speed=2.3, terminated=i < 2) for i in range(30)]  # Two failures exceed five percentage points.
    result = paired_effect(baseline, treated, "speed")  # Failed episodes stay in the paired analysis.
    assert result["n_pairs"] == 30 and not result["gate"]  # Failure exclusion must never manufacture a positive result.
    assert result["quality_deltas"]["failure"] == pytest.approx(2 / 30)  # Reports quantify the actual failure increase.


def test_effort_gate_rejects_slowing_and_accepts_comparable_speed():
    """Local: check effort-specific constraints. Global: slowing alone is not improved economy."""
    baseline = [summary(i) for i in range(30)]  # Baseline effort is 0.5 at 2 m/s.
    slow = [summary(i, speed=1.8, effort=0.4) for i in range(30)]  # Less effort comes with an unacceptable speed loss.
    assert not paired_effect(baseline, slow, "effort")["gate"]  # Enforce the five-percent speed tolerance.
    efficient = [summary(i, speed=1.98, effort=0.4) for i in range(30)]  # The same effort reduction preserves speed.
    assert paired_effect(baseline, efficient, "effort")["gate"]  # A useful effort-proxy reduction passes.


def test_raw_evaluation_uses_post_onset_measurements_and_preserves_prefix_failures():
    """Local: match intervention timing. Global: startup cannot manufacture an effect."""
    baseline = [episode(i, [2] * 9) for i in range(3)]  # Most of each episode occurs after the shared prefix.
    treated = [episode(i, [2.3] * 9) for i in range(3)]  # The intended sustained difference is exactly 0.3 m/s.
    result = paired_effect(baseline, treated, "speed")  # Array inputs should agree with equivalent scalar summaries.
    assert result["effect"] == pytest.approx(0.3) and result["gate"]  # Averaging in the zero-speed prefix would dilute this value.
    for item in treated:
        for name, value in list(item.items()):
            if isinstance(value, np.ndarray):
                item[name] = value[:100]  # Model an episode that fails before steering can begin.
        item.update(length=100, terminated=True, truncated=False)  # Preserve genuine failure metadata.
    failed = paired_effect(baseline, treated, "speed")  # Do not silently remove unsteerable reset episodes.
    assert failed["n_pairs"] == 3 and not failed["gate"]  # Prefix-only failures remain counted and invalidate useful-control claims.


def test_scalar_runtime_failure_flag_survives_paired_analysis():
    """Local: preserve summarized failure metadata. Global: scalar reduction cannot erase failed trials."""
    baseline = [summary(i) for i in range(30)]  # Healthy baseline summaries define the comparison denominator.
    treated = [summary(i, speed=2.3, failure=float(i < 2)) for i in range(30)]  # Runtime summaries expose failure instead of terminated.
    for item in baseline + treated:
        item.pop("terminated")  # Match the actual runtime summary interface exactly.
    result = paired_effect(baseline, treated, "speed")  # Two failures exceed the five-percentage-point allowance.
    assert result["quality_deltas"]["failure"] == pytest.approx(2 / 30)  # Failure information survives the second reduction.
    assert not result["gate"]  # An apparent speed increase must not hide failed episodes.


def test_custom_warmup_reaches_fitting_and_raw_evaluation():
    """Local: honor the configured onset. Global: fitting and evaluation share one temporal protocol."""
    episodes = [episode(i, [0, 2], [[0, 0], [i, 2]]) for i in range(16)]  # Only the final window remains after a 200-step warmup.
    vectors, diagnostic = fit_vectors(episodes, ["speed"], warmup=200)  # A changed onset removes the former low-speed contrast.
    assert not vectors and diagnostic["windows"]["warmup"] == 200  # Hardcoded startup would incorrectly return a vector.
    assert diagnostic["windows"]["total_windows"] == 16  # Exactly one eligible full window remains per episode.
    baseline = [episode(i, [2, 4]) for i in range(3)]  # Custom evaluation onset also changes the baseline mean.
    treated = [episode(i, [2.2, 4.2]) for i in range(3)]  # Keep a known additive treatment effect.
    result = paired_effect(baseline, treated, "speed", warmup=150)  # Both raw outcomes must use the explicit intervention onset.
    assert result["baseline_mean"] == pytest.approx(10 / 3)  # Last 150 steps contain fifty steps at2 and one hundred at4.


def test_raw_prefix_failure_uses_zero_outcomes_like_runtime_summary():
    """Local: standardize unavailable post-onset outcomes. Global: raw and summarized paths agree."""
    baseline = [summary(i) for i in range(3)]  # Scalar healthy baselines have a full post-onset period.
    treated = [episode(i, [2]) for i in range(3)]  # Start with complete array-shaped episodes.
    for item in treated:
        for name, value in list(item.items()):
            if isinstance(value, np.ndarray):
                item[name] = np.ones_like(value[:50])  # Nonzero startup measurements must not stand in for unavailable intervention outcomes.
        item.update(length=50, terminated=True, truncated=False)  # The run fails before the onset.
    result = paired_effect(baseline, treated, "speed")  # Keep the episodes while applying the documented zero-outcome convention.
    assert result["treated_mean"] == 0 and result["treated_means"]["failure"] == 1  # Match runtime.summarise_episode exactly.


def test_speed_floor_excludes_stationary_outliers_without_rewriting_original_fit():
    """Local: remove stationary fitting artifacts. Global: test the explicitly revised locomotion hypothesis."""
    episodes = [episode(i, [0.01, 2, 3], [[100, 0], [0, 0], [0, 2]]) for i in range(16)]  # Stationary post-flip activations dominate one coordinate despite passing existing health flags.
    original, original_diagnostics = fit_vectors(episodes, ["speed"])  # The default must preserve the original experiment's exact extraction.
    disabled, _ = fit_vectors(episodes, ["speed"], minimum_speed_fraction=0)  # Explicit zero is the compatibility setting.
    for name in original:
        np.testing.assert_array_equal(original[name], disabled[name])  # Default and zero produce identical vectors and controls.
    filtered, diagnostics = fit_vectors(episodes, ["speed"], minimum_speed_fraction=0.5)  # New experiments may require at least half the typical fitting speed.
    np.testing.assert_allclose(diagnostics["vectors"]["speed"]["raw_vector"], [0, 2])  # The retained contrast now describes two moving behaviors.
    assert abs(original_diagnostics["vectors"]["speed"]["raw_vector"][0]) > 50  # The fixture genuinely exposes the diagnosed domination problem.
    assert "speed" in filtered and diagnostics["windows"]["excluded_slow_windows"] == 16  # Every stationary window is excluded and accounted for.
    assert diagnostics["windows"]["healthy_median_speed"] == 2  # Only the supplied fitting distribution defines typical movement.
    assert diagnostics["windows"]["minimum_speed_floor"] == 1  # The absolute floor is reported in physical m/s.


def test_speed_floor_reference_uses_only_finite_healthy_windows_and_strict_boundary():
    """Local: separate exclusion reasons. Global: invalid or unhealthy data cannot set the locomotion floor."""
    moving = episode(1, [1, 2, 3])  # These are the only finite healthy reference windows.
    unhealthy = episode(2, [1000] * 5)  # Extreme failed behavior must not raise the reference median.
    unhealthy["bad_contact"][100:] = 1  # Mark every extreme window unhealthy through an existing criterion.
    nonfinite = episode(3, [1000] * 5)  # Nonfinite activations also exclude an otherwise valid speed descriptor.
    nonfinite["activations"][100:] = np.nan  # Preserve these rows for explicit nonfinite accounting.
    table = window_table([moving, unhealthy, nonfinite], minimum_speed_fraction=0.5)  # Fit the additional eligibility threshold only from credible movement data.
    assert table["stats"]["healthy_median_speed"] == 2 and table["stats"]["minimum_speed_floor"] == 1  # Excluded extremes cannot contaminate the threshold.
    np.testing.assert_array_equal(table["eligible"][:3], [False, True, True])  # The planned rule is speed strictly greater than the floor.
    assert table["stats"]["excluded_slow_windows"] == 1  # Count only newly excluded finite healthy windows as slow.
    assert table["stats"]["excluded_unhealthy"] == 5 and table["stats"]["excluded_nonfinite"] == 5  # Existing exclusion reasons remain mutually interpretable.


@pytest.mark.parametrize("fraction", [-0.1, float("nan"), float("inf")])
def test_speed_floor_rejects_invalid_fraction(fraction):
    """Local: validate the additional fitting parameter. Global: prevent malformed eligibility specifications."""
    with pytest.raises(ValueError, match="minimum_speed_fraction"):
        window_table([episode(1, [2])], minimum_speed_fraction=fraction)  # Only finite nonnegative fractions define an auditable speed floor.


@pytest.mark.parametrize("metric", ["bad_contact", "inversion"])
def test_one_collapsed_episode_cannot_hide_behind_pooled_health_fraction(metric):
    """Local: classify physical failure per episode. Global: one collapse in ten must not count as competent control."""
    baseline = [summary(i) for i in range(10)]  # All baseline resets retain healthy locomotion.
    treated = [summary(i, speed=2.3, **{metric: 0.45 if i == 0 else 0.0}) for i in range(10)]  # One episode spends 45 percent of its post-onset trajectory collapsed.
    result = paired_effect(baseline, treated, "speed")  # Pooling fractions alone would incorrectly permit this condition.
    assert result["quality_deltas"][metric] == pytest.approx(0.045)  # Preserve the genuine pooled timestep-fraction measurement.
    assert result["quality_deltas"]["failure"] == pytest.approx(0.1)  # Episode incidence separately captures the one-in-ten collapse.
    assert result["effect_size_pass"] and result["ci_excludes_zero"]  # The fixture isolates failure handling from the apparent useful speed effect.
    assert not result["quality_pass"] and not result["gate"]  # Ten-percent physical failure exceeds the five-percentage-point allowance.


def test_physical_episode_failure_uses_post_onset_strict_five_percent_threshold():
    """Local: apply the declared pilot boundary. Global: count physical failures consistently without changing raw trajectories."""
    baseline = [episode(i, [2] * 9) for i in range(10)]  # Each rollout provides exactly 900 post-onset samples.
    treated = [episode(i, [2.3] * 9) for i in range(10)]  # Sustained speed effects otherwise satisfy the usefulness criterion.
    for item in treated:
        item["bad_contact"][:100] = 1  # Startup contacts must not become post-intervention physical failures.
        item["bad_contact"][100:145] = 1  # Exactly45 of900 post-onset contacts equal the allowed five percent.
    boundary = paired_effect(baseline, treated, "speed")  # The operational definition uses strictly greater than five percent.
    assert boundary["treated_means"]["failure"] == 0 and boundary["gate"]  # Neither prefix contacts nor exact-boundary fractions count as failures.
    treated[0]["bad_contact"][145] = 1  # One additional contact crosses the per-episode boundary.
    crossed = paired_effect(baseline, treated, "speed")  # Recompute analysis without changing simulator termination metadata.
    assert crossed["treated_means"]["failure"] == pytest.approx(0.1) and not crossed["gate"]  # Raw array evaluation shares the scalar-summary failure definition.

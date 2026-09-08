"""Local: test independent audit arithmetic; global: catch misleading reproduction claims.

Source: https://numpy.org/doc/1.26/reference/routines.testing.html
These fixtures never call simulation or production extraction/statistics code.
"""
import numpy as np  # Construct small results whose intended values can be checked by hand.
import pytest  # Invalid paired identities must fail before calculating confidence intervals.
from scripts.documentation_evidence import activation_rms, balanced, contrast, episode_summary, make_windows, paired_summary, sensitivity  # Test the independent numerical oracle directly.


def episode(seed, values, *, length=500):
    """Local: construct aligned physical arrays; global: keep synthetic cases independent of MuJoCo."""
    data = {key: np.zeros(length) for key in ("x_velocity", "y_velocity", "planar_speed", "height", "yaw_rate", "effort", "inversion", "bad_contact", "reward", "saturation")}  # Unused descriptors are physically finite.
    data.update(seed=seed, length=length, terminated=False, truncated=True, onset_hash="same", activations=np.zeros((length, 2)))  # Two hidden dimensions suffice to expose weighting errors.
    data["y_velocity"][:] = values  # A scalar or aligned sequence supplies a measurable target.
    data["x_velocity"][:] = 2  # The synthetic actor retains forward locomotion.
    return data  # Each test changes only the property relevant to its assertion.


def test_balanced_contrast_does_not_overweight_more_windows():
    """Local: uneven episode contributions; global: distinguish episode weighting from pooled windows."""
    rows = [{"seed": seed, "lateral": value, "speed": 2., "h": np.array([value, 0.])} for seed, value in [(0, 0.), (0, 0.), (0, 0.), (1, 2.), (2, 8.), (3, 10.)]]  # Repeated low windows must not become independent episodes.
    np.testing.assert_allclose(balanced(rows[:4], "h"), [1., 0.])  # Episode means 0 and 2 receive equal weight.
    raw, detail = contrast(rows, "lateral")  # Linear quartiles put the high two values into the upper group.
    np.testing.assert_allclose(raw, [9., 0.])  # Low group is zero and high group averages 8 and 10.
    assert detail["low_episodes"] == 1 and detail["high_episodes"] == 2  # Contributor counts differ from window counts.


def test_effort_contrast_conditions_on_speed_and_weights_bins_equally():
    """Local: remove a speed-only hidden coordinate; global: guard the matched-effort audit."""
    rows = [{"seed": 10 * speed + rank, "speed": float(speed), "effort": float(rank), "h": np.array([speed, rank], float)} for speed in (1, 2) for rank in range(4)]  # Hidden dimension zero varies only with speed.
    raw, detail = contrast(rows, "effort")  # Exact repeated speeds form separate strata.
    np.testing.assert_allclose(raw, [0., 3.])  # Between-speed differences must cancel within both strata.
    assert len(detail["bins"]) == 2  # Both speed strata remain in the numerical evidence.


def test_rms_keeps_timestep_variation_inside_constant_window_means():
    """Local: alternating activations; global: prevent confusing window-mean variance with activation RMS."""
    first, second = episode(0, 0., length=200), episode(1, 0., length=300)  # Unequal lengths test equal episode weighting too.
    first["activations"][100:, 0] = np.tile([-1., 1.], 50)  # This window has zero mean and unit second moment.
    second["activations"][100:, 0] = np.tile([-3., 3.], 100)  # Twice as many timesteps must still receive one episode's weight.
    rows, _ = make_windows([first, second])  # All three complete windows are eligible.
    assert activation_rms([first, second], rows) == pytest.approx(np.sqrt(5.))  # Mean episode second moment is (1+9)/2, not 19/3.


def test_short_episode_quality_is_episode_averaged_and_prefix_is_retained():
    """Local: one brief collapse; global: expose denominator semantics in published quality fractions."""
    short, prefix = episode(0, 2., length=200), episode(1, 99., length=34)  # The prefix episode has no intervention outcome.
    short["inversion"][100:120] = 1  # Twenty of 100 available post-onset steps are inverted.
    assert episode_summary(short)["inversion"] == pytest.approx(.2)  # Do not divide by a nonexistent 900-step trajectory.
    assert episode_summary(short)["failure"] == 1  # Physical failure survives a low aggregate time-pooled rate.
    assert episode_summary(prefix)["lateral"] == 0 and episode_summary(prefix)["prefix_only"] == 1  # Preserve the historical zero-summary convention explicitly.


def test_paired_bootstrap_matches_seeds_not_input_order_and_rejects_leakage():
    """Local: permuted paired input; global: maintain complete-episode pairing in uncertainty estimates."""
    before = [episode(0, 1.), episode(1, 3.)]  # Different baselines make position-wise pairing unreliable.
    after = [episode(1, 5.), episode(0, 3.)]  # Both true paired differences equal two.
    estimate = paired_summary(before, after, "lateral", 710)  # Pair using recorded reset identities.
    assert estimate["effect"] == 2 and estimate["ci95"] == [2., 2.]  # Constant paired differences have a degenerate interval.
    with pytest.raises(ValueError, match="seed"):
        paired_summary(before, [episode(0, 3.), episode(2, 5.)], "lateral", 710)  # Missing/replaced resets cannot count as matched evidence.
    after[0]["onset_hash"] = "different"  # Equal reset seeds alone do not establish equal physical onset.
    with pytest.raises(ValueError, match="onset"):
        paired_summary(before, after, "lateral", 710)  # Reject physically unmatched comparisons.


def test_sensitivity_recomputes_quartiles_and_keeps_episode_boundaries():
    """Local: affine hidden/behavior relation; global: check exclusion and longer-window comparisons."""
    episodes = [episode(seed, np.repeat(np.arange(10) + seed / 10, 100), length=1000) for seed in range(8)]  # Behavior rises within each episode but has distinct reset offsets.
    for item in episodes:
        item["activations"][:, 0] = item["y_velocity"]  # Every legitimate contrast points along hidden coordinate zero.
    result = sensitivity(episodes, "lateral")  # Refit each diagnostic subset without changing any saved intervention.
    assert result["sensitivity"]["exclude_first_window"]["windows"] == 64  # Eight episodes lose one of nine post-onset windows each.
    assert result["longwindow"]["200"]["windows"] == 32 and result["longwindow"]["300"]["windows"] == 24  # Partial tails never cross episode boundaries.
    assert result["leave_one_episode_out_worst"][0] == pytest.approx(1.)  # Refit directions remain collinear after removing any whole episode.

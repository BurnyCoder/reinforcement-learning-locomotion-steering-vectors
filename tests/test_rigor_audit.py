"""Local: test scientific-rigor audit statistics; global: keep the inference audit trustworthy.

Synthetic fixtures with hand-checkable answers exercise each helper. Tests never
load policy weights or run a simulator.
Source: https://numpy.org/doc/1.26/reference/routines.testing.html
"""
import numpy as np  # Build small arrays whose expected statistics are known exactly.
import pytest  # Compare floating-point results at explicit tolerances.
from rl_locomotion_steering_vectors.analysis import paired_effect  # The audit reuses the production paired estimator rather than a copy.
from scripts.rigor_statistics import body_relative_travel_change, heading_series, matched_horizon_difference, mean_heading, odd_even, pairs_for_power, percentile_interval, post_onset_displacement, rank_pvalue, shuffled_alignment_null, sign_flip_pvalue, travel_angle, window_means  # Helpers under test.


def test_odd_even_separates_sign_dependent_from_sign_independent_response():
    """Local: split two signed effects; global: tell steering apart from generic degradation."""
    assert odd_even(-0.2, -0.6) == pytest.approx((0.2, -0.4))  # Both signs slow down: the even part dominates.
    assert odd_even(0.5, -0.5) == pytest.approx((0.5, 0.0))  # A purely linear response has no even part.


def test_sign_flip_pvalue_reaches_floor_for_consistent_and_one_for_symmetric_differences():
    """Local: extreme and null data; global: verify the randomization test's direction and bounds."""
    assert sign_flip_pvalue(np.ones(20), resamples=1000, seed=0) <= 2 / 1001  # Only the all-positive/all-negative patterns reach |mean| = 1; with 2^20 patterns, at most one hit is plausible.
    assert sign_flip_pvalue(np.array([-1., 1., -1., 1.]), resamples=500, seed=0) == 1.0  # A zero observed mean is matched by every relabeling.
    assert sign_flip_pvalue(np.zeros(5), resamples=100, seed=0) == 1.0  # No difference is never reported as evidence.


def test_percentile_interval_is_degenerate_for_constant_differences():
    """Local: constant paired differences; global: confirm the interval resamples whole pairs."""
    assert percentile_interval(np.full(7, 2.5), seed=3) == pytest.approx([2.5, 2.5])  # Every resampled mean equals the constant difference.
    low, high = percentile_interval(np.arange(10.), seed=3)  # Varying differences produce a nondegenerate interval.
    assert low < 4.5 < high  # The sample mean lies inside its own percentile interval.


def test_post_onset_displacement_counts_no_motion_after_termination():
    """Local: integrate available steps; global: define a fixed-horizon outcome without zero-filled velocity means."""
    early = {"y_velocity": np.full(150, 2.), "length": 150, "dt": 0.05}  # Fifty post-onset steps survive before termination.
    prefix = {"y_velocity": np.full(34, 9.), "length": 34, "dt": 0.05}  # Termination before onset leaves no post-onset motion.
    assert post_onset_displacement(early, "y_velocity", warmup=100) == pytest.approx(5.)  # 50 steps * 2 m/s * 0.05 s.
    assert post_onset_displacement(prefix, "y_velocity", warmup=100) == 0.  # Prefix-only episodes contribute zero displacement.


def test_mean_heading_recovers_constant_yaw_from_scalar_first_quaternions():
    """Local: synthetic Ant observations; global: measure torso heading change from saved observations."""
    yaw = 0.3  # A constant rotation about world Z.
    observations = np.zeros((200, 105))  # Ant observations store z, then the scalar-first quaternion.
    observations[:, 1], observations[:, 4] = np.cos(yaw / 2), np.sin(yaw / 2)  # Quaternion (w, 0, 0, z) encodes pure yaw.
    assert mean_heading(observations, length=200, warmup=100) == pytest.approx(yaw)  # Circular mean of a constant angle.
    assert mean_heading(observations, length=50, warmup=100) is None  # No post-onset observations means heading is undefined.


def test_shuffled_contrasts_align_with_dominant_between_episode_axis():
    """Local: episode offsets dominate activations; global: show why shuffled labels are not an orthogonal null."""
    rng = np.random.default_rng(0)  # Deterministic synthetic fitting table.
    episodes, windows, width = 40, 5, 20  # Forty independent episodes with five windows each.
    offsets = np.zeros((episodes, width)); offsets[:, 0] = rng.normal(0, 3, episodes)  # Only dimension zero varies between episodes.
    identity = np.repeat(np.arange(episodes), windows)  # Windows keep their episode identity.
    table = {"episode": identity, "activations": offsets[identity] + rng.normal(0, .1, (episodes * windows, width)),
             "lateral": rng.normal(size=episodes * windows), "speed": np.full(episodes * windows, 5.)}  # Labels are unrelated to activations.
    axis, other = np.eye(width)[0], np.eye(width)[1]  # Dominant and minor activation directions.
    along = np.abs(shuffled_alignment_null(table, "lateral", axis, draws=50, seed=1))  # Shuffled contrasts versus the dominant axis.
    across = np.abs(shuffled_alignment_null(table, "lateral", other, draws=50, seed=1))  # Same draws versus a low-variance axis.
    assert along.shape == (50,) and np.isfinite(along).all()  # One finite cosine per supported shuffle.
    assert along.mean() > 0.6 and across.mean() < 0.2  # Label shuffling inherits the dominant activation subspace; a chance-small offset difference occasionally lowers one draw.


def test_paired_effect_contrasts_two_treated_arms_on_shared_seeds():
    """Local: compare primary against control; global: reuse the production estimator for between-condition contrasts."""
    def arm(values):
        """Local: scalar episode summaries; global: mimic runtime summaries without a simulator."""
        return [{"seed": seed, "onset_hash": "shared", "length": 1000, "failure": 0., "y_velocity": value, "x_velocity": 6., "planar_speed": 6.,
                 "height": .5, "yaw_rate": 0., "effort": .2, "inversion": 0., "bad_contact": 0., "reward": 6., "saturation": 0.}
                for seed, value in enumerate(values)]  # Treated arms of one seed share their unsteered onset state.
    result = paired_effect(arm([0.2, 0.1, 0.3]), arm([0.6, 0.5, 0.8]), "lateral", seed=0)  # Control first, primary second.
    assert result["effect"] == pytest.approx(np.mean([0.4, 0.4, 0.5]))  # Primary minus control, paired by seed.


def test_odd_even_decomposes_per_seed_arrays_elementwise():
    """Local: split paired per-seed responses; global: bootstrap odd parts across resets without a new helper."""
    odd, even = odd_even(np.array([1., -1.]), np.array([-1., -3.]))  # Two resets measured at +alpha and -alpha.
    assert odd == pytest.approx([1., 1.]) and even == pytest.approx([0., -2.])  # The same exact identity holds seed by seed.


def test_travel_angle_recovers_a_rotated_constant_ground_velocity():
    """Local: synthetic post-onset velocities; global: measure ground-path direction without torso-orientation conventions."""
    angle = np.deg2rad(4.)  # A small leftward rotation of the direction of travel.
    episode = {"x_velocity": np.full(300, 6. * np.cos(angle)), "y_velocity": np.full(300, 6. * np.sin(angle)), "length": 300}  # Constant 6 m/s along that direction.
    assert travel_angle(episode, warmup=100) == pytest.approx(angle)  # Direction of the mean velocity equals the net displacement direction.
    assert travel_angle({**episode, "length": 34}, warmup=100) is None  # A prefix-only episode has no post-onset path.


def test_heading_series_unwraps_across_the_branch_cut_and_window_means_need_full_coverage():
    """Local: yaw crossing pi and a short episode; global: compare heading time courses without false 2*pi jumps."""
    yaws = np.linspace(3.0, 3.3, 250)  # Steady rotation through the -pi/pi boundary.
    observations = np.zeros((250, 105)); observations[:, 1], observations[:, 4] = np.cos(yaws / 2), np.sin(yaws / 2)  # Pure-yaw scalar-first quaternions.
    series = heading_series(observations, length=250)  # Unwrapped torso yaw per pre-action observation.
    assert series == pytest.approx(yaws)  # No jump of -2*pi after crossing pi.
    means = window_means(series, length=250, windows=[(100, 200), (200, 300)])  # The second window extends beyond termination.
    assert means[0] == pytest.approx(yaws[100:200].mean()) and np.isnan(means[1])  # Incomplete windows are undefined, never partially averaged.


def test_rank_pvalue_counts_ties_and_includes_the_observed_direction():
    """Local: Monte Carlo rank test; global: compare one steering vector with a small set of random directions."""
    assert rank_pvalue(5., [1., 2., 3.]) == pytest.approx(1 / 4)  # Largest of four exchangeable values: the smallest attainable p.
    assert rank_pvalue(2., [1., 2., 3.]) == pytest.approx(3 / 4)  # A tie counts as at least as extreme, which is conservative.
    assert rank_pvalue(9., np.zeros(19)) == pytest.approx(0.05)  # Nineteen reference directions are needed to reach 0.05.


def walk(heading: float, travel: float, steps: int = 300) -> dict:
    """Local: synthetic Ant episode with constant torso yaw and constant 6 m/s ground velocity; global: separate body turning from path turning."""
    observations = np.zeros((steps, 105)); observations[:, 1], observations[:, 4] = np.cos(heading / 2), np.sin(heading / 2)  # Pure-yaw scalar-first quaternion at observation indices 1..4.
    return {"observations": observations, "x_velocity": np.full(steps, 6. * np.cos(travel)), "y_velocity": np.full(steps, 6. * np.sin(travel)), "length": steps, "dt": 0.05}  # World-frame velocity along the travel direction.


def test_body_relative_travel_change_is_zero_for_rigid_rotation_and_full_for_sidestep():
    """Local: three synthetic gaits; global: only torso heading can tell a rotated gait from sidestepping."""
    crab, turn = np.deg2rad(45.), np.deg2rad(4.)  # Baseline torso yaw sits 45 degrees from the direction of travel, as in the Ant gait.
    base = walk(crab, 0.)  # Unsteered reference arm.
    assert body_relative_travel_change(base, walk(crab + turn, turn)) == pytest.approx(0., abs=1e-12)  # Body and path turn together: no change relative to the body.
    assert body_relative_travel_change(base, walk(crab, turn)) == pytest.approx(turn)  # Path turns, body does not: pure sidestep.
    assert body_relative_travel_change(base, walk(crab + np.deg2rad(5.23), np.deg2rad(3.84))) == pytest.approx(np.deg2rad(-1.39))  # Body turning more than the path shows up as a negative change.
    assert body_relative_travel_change(base, {**walk(crab, turn), "length": 34}) is None  # A prefix-only arm has no post-onset direction.


def test_displacement_rotation_check_cannot_distinguish_rotation_from_sidestep():
    """Local: the old 'forward distance x tan(travel change)' prediction; global: show that matching it is arithmetic, not mechanism evidence."""
    turn, base = np.deg2rad(4.), walk(0., 0.)  # Small path change relative to straight +X travel.
    sidestep = {**base, "y_velocity": np.full(300, 6. * np.tan(turn))}  # Same forward velocity, extra world-Y velocity, torso unchanged.
    for treated in (walk(turn, turn), sidestep):  # A rigid rotation and a pure sidestep.
        predicted = post_onset_displacement(base, "x_velocity") * np.tan(travel_angle(treated) - travel_angle(base))  # The prediction used in the y-shift audit.
        measured = post_onset_displacement(treated, "y_velocity") - post_onset_displacement(base, "y_velocity")  # World-Y displacement change.
        assert predicted == pytest.approx(measured, rel=0.01)  # Both mechanisms "pass" the check, so it carries no evidential weight.
    assert body_relative_travel_change(base, sidestep) == pytest.approx(turn)  # The heading-based estimand does separate them.


def test_matched_horizon_difference_uses_only_steps_both_arms_recorded():
    """Local: arms with different durations; global: remove termination timing from a paired velocity outcome."""
    before = {"y_velocity": np.r_[np.zeros(100), np.full(900, 1.)], "length": 1000}  # Baseline survives the full horizon at 1 m/s.
    after = {"y_velocity": np.r_[np.zeros(100), np.full(100, 3.)], "length": 200}  # Treated arm terminates at step 200.
    assert matched_horizon_difference(before, after, "y_velocity") == pytest.approx(2.)  # Only steps 100..199 are compared.
    assert matched_horizon_difference(before, {**after, "length": 34}, "y_velocity") is None  # No common post-onset step.


def test_studentized_sign_flip_matches_plain_test_at_the_extremes():
    """Local: studentized statistic on extreme and null data; global: a weak-null robustness check of the headline p-value."""
    assert sign_flip_pvalue(np.ones(20), resamples=1000, seed=0, studentized=True) <= 2 / 1001  # Zero variance gives an infinite t that only all-same-sign patterns match.
    assert sign_flip_pvalue(np.array([-1., 1., -1., 1.]), resamples=500, seed=0, studentized=True) == 1.0  # A zero observed mean is matched by every relabeling.
    assert sign_flip_pvalue(np.zeros(5), resamples=100, seed=0, studentized=True) == 1.0  # Undefined 0/0 statistics count as no evidence.
    shifted = np.random.default_rng(1).normal(1., 1., 30)  # Clear mean shift.
    assert sign_flip_pvalue(shifted, resamples=2000, seed=0, studentized=True) < 0.01  # The studentized test also detects it.


def test_pairs_for_power_uses_the_paired_normal_approximation():
    """Local: textbook sample-size arithmetic; global: check whether thirty confirmation pairs were adequate."""
    assert pairs_for_power(sd=1., delta=1.) == 8  # ((1.95996 + 0.84162) * 1 / 1)^2 = 7.85, rounded up.
    assert pairs_for_power(sd=0.25, delta=0.2) == 13  # Ant-like paired SD against the 0.2 m/s lateral threshold: 12.26, rounded up.

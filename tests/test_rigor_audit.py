"""Local: test scientific-rigor audit statistics; global: keep the inference audit trustworthy.

Synthetic fixtures with hand-checkable answers exercise each helper. Tests never
load policy weights or run a simulator.
Source: https://numpy.org/doc/1.26/reference/routines.testing.html
"""
import numpy as np  # Build small arrays whose expected statistics are known exactly.
import pytest  # Compare floating-point results at explicit tolerances.
from rl_locomotion_steering_vectors.analysis import paired_effect  # The audit reuses the production paired estimator rather than a copy.
from scripts.rigor_statistics import mean_heading, odd_even, percentile_interval, post_onset_displacement, shuffled_alignment_null, sign_flip_pvalue  # Helpers under test.


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

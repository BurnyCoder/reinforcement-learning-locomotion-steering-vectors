"""Local: test Ant ground-plane kinematics. Global: distinguish a rotated path, a sidestep and a continuing turn.

Sources: NumPy unwrap https://numpy.org/doc/1.26/reference/generated/numpy.unwrap.html ;
arctan2 https://numpy.org/doc/1.26/reference/generated/numpy.arctan2.html
"""
import numpy as np  # Synthetic trajectories with analytic answers.
import pytest  # Approximate comparisons.

from rl_locomotion_steering_vectors.kinematics import pair_kinematics, torso_yaw, travel_angle  # Public contract of the new module.


def _episode(heading: np.ndarray, travel: np.ndarray, speed: float = 6.0) -> dict:
    """Local: synthesize one rollout. Global: control torso heading and velocity direction independently."""
    observations = np.zeros((len(heading), 105))  # Only the quaternion block is read.
    observations[:, 1] = np.cos(heading / 2)  # Scalar part of a pure yaw quaternion.
    observations[:, 4] = np.sin(heading / 2)  # z part of a pure yaw quaternion.
    return {"observations": observations, "x_velocity": speed * np.cos(travel), "y_velocity": speed * np.sin(travel),
            "length": np.array(len(heading)), "dt": np.array(0.05), "terminated": np.array(False)}  # Minimal saved-episode schema.


def test_torso_yaw_unwraps_across_branch_cut():
    """Local: read heading from quaternions. Global: avoid fake ±2π jumps in heading time courses."""
    heading = np.linspace(3.0, 3.4, 5)  # Crosses +π.
    np.testing.assert_allclose(torso_yaw(_episode(heading, heading)["observations"]), heading, atol=1e-12)  # Continuous recovery.


def test_travel_angle_uses_displacement_direction():
    """Local: average velocity components first. Global: measure the path direction, not mean per-step angles."""
    assert travel_angle(np.full(10, np.cos(0.1)), np.full(10, np.sin(0.1))) == pytest.approx(np.degrees(0.1))  # Constant direction.


def test_rotated_path_has_zero_gait_lateral_change():
    """Local: rotate the whole gait after onset. Global: a pure path rotation is not a sidestep or a turn."""
    steps = 1000  # Full horizon.
    base_heading = np.full(steps, np.radians(45.0))  # Torso sits 45° off the travel direction, as in the Ant gait.
    base_travel = np.zeros(steps)  # Baseline travels along +x.
    shift = np.where(np.arange(steps) >= 100, np.radians(4.0), 0.0)  # Rigid 4° rotation from onset.
    result = pair_kinematics(_episode(base_heading, base_travel), _episode(base_heading + shift, base_travel + shift), warmup=100)  # Compare arms.
    assert result["delta_travel_angle_deg"] == pytest.approx(4.0, abs=1e-9)  # Path rotates 4°.
    assert result["delta_mean_yaw_deg"] == pytest.approx(4.0, abs=1e-9)  # Torso rotates 4°.
    assert result["delta_gait_lateral_velocity"] == pytest.approx(0.0, abs=1e-9)  # No motion across the gait axis.
    assert result["delta_y_velocity"] == pytest.approx(6.0 * np.sin(np.radians(4.0)), abs=1e-9)  # World-Y gain of a rigid rotation.
    assert result["delta_y_displacement"] == pytest.approx(6.0 * np.sin(np.radians(4.0)) * 900 * 0.05, abs=1e-6)  # Extra metres over 900 steps.
    assert result["yaw_slope_deg_per_100_steps"] == pytest.approx(0.0, abs=1e-9)  # A bounded offset has no trend.
    assert len(result["window_delta_yaw_deg"]) == 9  # Nine full 100-step windows after onset.


def test_sidestep_and_common_horizon():
    """Local: translate without rotating and truncate one arm. Global: detect crabbing and compare equal spans."""
    heading = np.zeros(1000)  # Torso faces +x in both arms.
    treated_travel = np.where(np.arange(600) >= 100, np.radians(3.0), 0.0)  # Velocity turns but the torso does not.
    result = pair_kinematics(_episode(heading, np.zeros(1000)), _episode(heading[:600], treated_travel), warmup=100)  # Treated arm ends at 600.
    assert result["common_steps"] == 500  # Steps 100–599 exist in both arms.
    assert result["delta_mean_yaw_deg"] == pytest.approx(0.0, abs=1e-9)  # No torso rotation.
    assert result["delta_gait_lateral_velocity"] == pytest.approx(6.0 * np.sin(np.radians(3.0)), abs=1e-9)  # Pure sidestep relative to the gait axis.


@pytest.mark.parametrize("window", [50, 100, 200])
def test_heading_trend_units_do_not_depend_on_window_size(window):
    """A known heading ramp has the same degrees-per-100-steps slope at every resolution."""
    heading = np.radians(np.arange(1000) * 0.01)
    result = pair_kinematics(_episode(np.zeros(1000), np.zeros(1000)), _episode(heading, heading), window=window)
    assert result["yaw_slope_deg_per_100_steps"] == pytest.approx(1.0)


def test_travel_direction_difference_wraps_the_branch_cut():
    """Two almost identical backward paths differ by two degrees, not a full revolution."""
    left, right = np.full(300, np.radians(179)), np.full(300, np.radians(-179))
    result = pair_kinematics(_episode(left, left), _episode(right, right))
    assert result["delta_travel_angle_deg"] == pytest.approx(2.0)


def test_single_window_does_not_claim_a_zero_heading_trend():
    """One observed window cannot identify a slope."""
    heading = np.radians(np.arange(200) * 0.01)
    result = pair_kinematics(_episode(np.zeros(200), np.zeros(200)), _episode(heading, heading))
    assert result["yaw_slope_deg_per_100_steps"] is None


@pytest.mark.parametrize("dt", [0.04, 0, float("nan")])
def test_paired_step_horizons_require_equal_positive_timesteps(dt):
    """Step alignment is not time alignment when recorded simulator timesteps differ."""
    episode = _episode(np.zeros(300), np.zeros(300))
    with pytest.raises(ValueError, match="timestep"):
        pair_kinematics(episode, {**episode, "dt": dt})


@pytest.mark.parametrize("warmup, window", [(-1, 100), (100, 0), (100, -1)])
def test_kinematic_windows_require_valid_step_counts(warmup, window):
    """Invalid spans must fail explicitly rather than become fabricated summaries."""
    episode = _episode(np.zeros(300), np.zeros(300))
    with pytest.raises(ValueError, match="warmup.*window"):
        pair_kinematics(episode, episode, warmup=warmup, window=window)

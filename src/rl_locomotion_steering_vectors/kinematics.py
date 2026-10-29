"""Local: Ant ground-plane kinematics from saved episodes. Global: compare recorded path and heading changes without identifying a unique mechanism.

Sources: NumPy unwrap https://numpy.org/doc/1.26/reference/generated/numpy.unwrap.html ;
arctan2 https://numpy.org/doc/1.26/reference/generated/numpy.arctan2.html ;
linear-fit units https://numpy.org/doc/1.26/reference/generated/numpy.polyfit.html ;
yaw extraction reuses runtime.quaternion_yaw (scalar-first ZYX yaw). Saved observations
are action inputs; each heading is paired with the ensuing transition's displacement
velocity. This discrete-time convention is not an instantaneous post-step body velocity.
"""
from __future__ import annotations  # Keep annotations lightweight.

import numpy as np  # Vectorized trajectory reductions.

from .runtime import quaternion_yaw  # Reuse the audited quaternion-to-yaw conversion.


def torso_yaw(observations: np.ndarray) -> np.ndarray:
    """Local: unwrapped torso heading per step (radians). Global: follow heading without ±2π jumps."""
    return np.unwrap(np.array([quaternion_yaw(q) for q in np.asarray(observations)[:, 1:5]]))  # Ant obs[1:5] is qpos[3:7].


def travel_angle(x_velocity, y_velocity) -> float:
    """Local: direction of net displacement (degrees). Global: measure the path heading, not the torso heading."""
    return float(np.degrees(np.arctan2(np.sum(y_velocity), np.sum(x_velocity))))  # Sum before the angle.


def _arm(episode: dict, span: slice, offset: float) -> dict:
    """Local: reduce one arm over a span. Global: share identical reductions between baseline and treatment."""
    yaw = torso_yaw(episode["observations"])[span]  # Torso heading per step.
    vx, vy = np.asarray(episode["x_velocity"])[span], np.asarray(episode["y_velocity"])[span]  # World velocities.
    axis = yaw - offset  # Gait axis: torso heading minus the baseline torso-to-path offset.
    return {"yaw": yaw, "vy": vy, "travel": travel_angle(vx, vy), "gait_lateral": float(np.mean(-np.sin(axis) * vx + np.cos(axis) * vy))}  # Velocity across the gait axis.


def pair_kinematics(baseline: dict, treated: dict, warmup: int = 100, window: int = 100) -> dict:
    """Compare shared recorded steps; geometric summaries do not identify causal mediation.

    Travel-angle differences use the shortest signed angle in [-180, 180).
    Heading trends need two windows and are expressed per 100 simulation steps,
    regardless of the chosen window length. Recorded timesteps must agree.
    """
    if warmup < 0 or window <= 0:
        raise ValueError("warmup must be nonnegative and window must be positive")
    timesteps = [float(episode["dt"]) for episode in (baseline, treated)]
    if not all(np.isfinite(dt) and dt > 0 for dt in timesteps) or timesteps[0] != timesteps[1]:
        raise ValueError("paired kinematics requires equal positive finite timesteps")
    end = min(int(baseline["length"]), int(treated["length"]))  # Last step present in both arms.
    span = slice(warmup, end)  # Common post-onset horizon.
    if end - warmup < window:
        return {"common_steps": max(0, end - warmup)}  # Too short for any windowed comparison.
    base_yaw = torso_yaw(baseline["observations"])[span]  # Baseline torso heading.
    offset = float(np.mean(base_yaw) - np.radians(travel_angle(np.asarray(baseline["x_velocity"])[span], np.asarray(baseline["y_velocity"])[span])))  # Torso-to-path offset of the gait.
    base, steer = _arm(baseline, span, offset), _arm(treated, span, offset)  # Same reductions, same gait offset.
    starts = range(0, end - warmup - window + 1, window)  # Full windows inside the common span.
    window_yaw = [float(np.degrees(steer["yaw"][s:s + window].mean() - base["yaw"][s:s + window].mean())) for s in starts]  # Torso heading change per window.
    window_vy = [float(steer["vy"][s:s + window].mean() - base["vy"][s:s + window].mean()) for s in starts]  # World-Y change per window.
    slope = float(np.polyfit(np.arange(len(window_yaw)), window_yaw, 1)[0]) * (100 / window) if len(window_yaw) > 1 else None  # Convert degrees/window to degrees/100 steps; one window cannot identify a trend.
    travel_change = steer["travel"] - base["travel"]
    if not -180 <= travel_change < 180:
        travel_change = (travel_change + 180) % 360 - 180  # Principal path directions must not create a spurious full revolution at the branch cut.
    dt = timesteps[0]  # Equal physical step duration checked above; no undocumented fallback.
    own = lambda episode: float(np.sum(np.asarray(episode["y_velocity"])[warmup:int(episode["length"])]) * dt)  # Post-onset world-Y displacement over the arm's own horizon.
    return {"common_steps": end - warmup, "delta_y_velocity": float(steer["vy"].mean() - base["vy"].mean()),
            "delta_y_displacement": own(treated) - own(baseline), "delta_travel_angle_deg": travel_change,
            "delta_mean_yaw_deg": float(np.degrees(steer["yaw"].mean() - base["yaw"].mean())),
            "delta_gait_lateral_velocity": steer["gait_lateral"] - base["gait_lateral"],
            "window_delta_yaw_deg": window_yaw, "window_delta_y_velocity": window_vy, "yaw_slope_deg_per_100_steps": slope}  # JSON-safe pair summary.

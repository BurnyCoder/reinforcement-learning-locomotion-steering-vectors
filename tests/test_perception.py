"""Local: test perceived-heading observation interventions. Global: make the Ant mechanism test a faithful world rotation.

Sources: Gymnasium Ant-v5 observation construction
https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py ;
MuJoCo c-frame quantities ("a global frame centered at the subtree center of mass")
https://mujoco.readthedocs.io/en/3.1.5/computation/index.html ;
mj_objectVelocity https://mujoco.readthedocs.io/en/3.1.5/APIreference/APIfunctions.html#mj-objectvelocity
"""
import gymnasium as gym  # Real Ant-v5 simulator supplies the observation layout under test.
import mujoco  # Direct API calls establish velocity frames and contact-force recomputation.
import numpy as np  # Analytic rotations and array comparisons.
import pytest  # Fixtures and approximate comparisons.
import torch  # Thread control and tensor conversion for the untrained actor.
from stable_baselines3 import SAC  # Real SB3 actor with the production two-layer architecture.

from rl_locomotion_steering_vectors.perception import (  # Public contract of the new module.
    PerceivedHeadingModel, first_relu, perceived_heading_gradient, rotate_ant_observation,
)
from rl_locomotion_steering_vectors.runtime import ActivationIntervention, actor_layer, quaternion_yaw, run_episode  # Reuse production measurement and rollout code as oracles.


def _yaw_quaternion(angle: float) -> np.ndarray:
    """Local: build a scalar-first rotation about world z. Global: give tests an analytic heading oracle."""
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])  # MuJoCo stores quaternions as (w, x, y, z).


def _tilted_observation(rng: np.random.Generator, yaw: float) -> np.ndarray:
    """Local: build a valid tilted Ant observation. Global: test rotation on more than an upright identity pose."""
    observation = rng.normal(size=105)  # Arbitrary joint, velocity and force values exercise every block.
    tilt = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])  # A 0.2 rad roll makes yaw extraction nontrivial.
    w1, x1, y1, z1 = _yaw_quaternion(yaw)  # Heading rotation applied after the body tilt.
    w2, x2, y2, z2 = tilt  # Body-frame tilt component.
    observation[1:5] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]  # Hamilton product yaw ⊗ tilt.
    observation[27:] = np.clip(observation[27:], -1, 1)  # Ant-v5 clips cfrc_ext to its contact-force range before observation.
    return observation  # One complete 105-element observation.


def test_rotation_shifts_heading_and_preserves_other_state():
    """Local: rotate one observation. Global: change only what a heading change would change."""
    rng = np.random.default_rng(0)  # Reproducible synthetic state.
    observation = _tilted_observation(rng, yaw=0.3)  # Known starting heading.
    rotated = rotate_ant_observation(observation, 0.2)  # Perceived world turned by +0.2 rad.
    assert quaternion_yaw(rotated[1:5]) == pytest.approx(0.5, abs=1e-9)  # Heading increases by exactly delta.
    assert np.linalg.norm(rotated[1:5]) == pytest.approx(np.linalg.norm(observation[1:5]))  # Rotation keeps a unit quaternion.
    up = lambda q: 1 - 2 * (q[1] ** 2 + q[2] ** 2)  # Body-z dot world-z, the runtime inversion predicate.
    assert up(rotated[1:5] / np.linalg.norm(rotated[1:5])) == pytest.approx(up(observation[1:5] / np.linalg.norm(observation[1:5])))  # Tilt relative to gravity is unchanged.
    assert np.linalg.norm(rotated[13:15]) == pytest.approx(np.linalg.norm(observation[13:15]))  # Planar speed is unchanged.
    assert np.arctan2(rotated[14], rotated[13]) == pytest.approx(np.arctan2(observation[14], observation[13]) + 0.2)  # Velocity direction turns with the world.
    np.testing.assert_array_equal(rotated[[0, 15]], observation[[0, 15]])  # Height and vertical velocity are invariant.
    np.testing.assert_array_equal(rotated[5:13], observation[5:13])  # Joint angles are body-relative.
    np.testing.assert_array_equal(rotated[16:27], observation[16:27])  # Local angular velocity and joint velocities are invariant.
    forces, original = rotated[27:].reshape(13, 6), observation[27:].reshape(13, 6)  # Per-body (torque, force) rows.
    np.testing.assert_array_equal(forces[:, [2, 5]], original[:, [2, 5]])  # Vertical torque/force components are invariant.
    assert np.all(np.abs(forces) <= 1.0)  # Rotated forces stay inside the environment's clip range.


def test_rotation_is_invertible_and_batched():
    """Local: compose rotations and batch rows. Global: let gradients use whole fitting arrays."""
    rng = np.random.default_rng(1)  # Independent synthetic states.
    batch = np.stack([_tilted_observation(rng, yaw) for yaw in (-2.0, 0.0, 3.0)])  # Include headings near the branch cut.
    batch[:, 27:] *= 0.5  # Keep forces well inside the clip range so inversion is exact.
    rotated = rotate_ant_observation(batch, 0.4)  # Rotate all rows together.
    np.testing.assert_allclose(rotated[1], rotate_ant_observation(batch[1], 0.4))  # Batch rows equal single-row rotations.
    np.testing.assert_allclose(rotate_ant_observation(rotated, -0.4), batch, atol=1e-12)  # Opposite rotation restores the state.
    assert rotate_ant_observation(batch, 0.0) is not batch  # Callers never receive an aliased input array.


def test_rotation_equals_rotated_physical_ant_state():
    """Local: compare with MuJoCo. Global: prove the layout, velocity frames and force frames are right."""
    with gym.make("Ant-v5") as env:  # Unmodified environment defines the true observation.
        env.reset(seed=7)  # Reproducible starting state.
        env.action_space.seed(7)  # Reproducible random actions (Gymnasium spaces keep their own RNG).
        for _ in range(30):  # Move into a contact-rich gait-like state.
            env.step(env.action_space.sample() * 0.5)  # Moderate random torques keep the Ant upright.
        physics = env.unwrapped  # Direct MuJoCo model/data access.
        qpos, qvel = physics.data.qpos.copy(), physics.data.qvel.copy()  # Original physical state.
        delta = 0.3  # Large enough to expose any frame error.
        c, s = np.cos(delta), np.sin(delta)  # Planar rotation coefficients.
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # Rotation about world z.
        mujoco.mj_forward(physics.model, physics.data)  # Refresh derived quantities for the original state.
        mujoco.mj_rnePostConstraint(physics.model, physics.data)  # cfrc_ext is computed here (not by mj_forward alone).
        original_obs = np.concatenate([qpos[2:], qvel, physics.data.cfrc_ext[1:].ravel()])  # Unclipped Ant-v5 observation.
        world_velocity = np.zeros(6)  # mj_objectVelocity output buffer (rot:lin).
        mujoco.mj_objectVelocity(physics.model, physics.data, mujoco.mjtObj.mjOBJ_BODY, 1, world_velocity, 0)  # Torso velocity, world orientation.
        local_velocity = np.zeros(6)  # Same velocity in the torso's local orientation.
        mujoco.mj_objectVelocity(physics.model, physics.data, mujoco.mjtObj.mjOBJ_BODY, 1, local_velocity, 1)  # Torso velocity, local orientation.
        np.testing.assert_allclose(qvel[0:3], world_velocity[3:6], atol=1e-8)  # Free-joint linear velocity is world-frame.
        np.testing.assert_allclose(qvel[3:6], local_velocity[0:3], atol=1e-8)  # Free-joint angular velocity is body-frame.
        rotated_qpos, rotated_qvel = qpos.copy(), qvel.copy()  # Physically rotated copy of the same motion.
        rotated_qpos[0:3] = rotation @ qpos[0:3]  # Rotate the torso position about the world origin.
        w, z = _yaw_quaternion(delta)[[0, 3]]  # Heading quaternion components.
        q = qpos[3:7]  # Original orientation.
        rotated_qpos[3:7] = [w * q[0] - z * q[3], w * q[1] - z * q[2], w * q[2] + z * q[1], w * q[3] + z * q[0]]  # q_z ⊗ q.
        rotated_qvel[0:3] = rotation @ qvel[0:3]  # World-frame linear velocity rotates.
        physics.set_state(rotated_qpos, rotated_qvel)  # Gymnasium helper calls mj_forward.
        mujoco.mj_rnePostConstraint(physics.model, physics.data)  # Recompute contact-based external forces.
        rotated_obs = np.concatenate([rotated_qpos[2:], rotated_qvel, physics.data.cfrc_ext[1:].ravel()])  # Physical rotated observation.
    scale = 2 * max(1.0, float(np.abs(original_obs[27:]).max()))  # Keep rotated forces inside the clip range for an exact comparison.
    original_obs[27:] /= scale  # Uniform scaling commutes with rotation.
    rotated_obs[27:] /= scale  # Same scaling on the physical rotated state.
    np.testing.assert_allclose(rotate_ant_observation(original_obs, delta), rotated_obs, atol=1e-6)  # Observation rotation equals rotated physics.


class _RecordingModel:
    """Local: stand in for an SB3 model. Global: test proxy scheduling without simulation."""

    def __init__(self):
        """Local: store inputs. Global: expose delegated attributes."""
        self.seen, self.policy = [], "frozen-policy"  # A sentinel proves attribute delegation.

    def predict(self, observation, deterministic=True):
        """Local: record the observation. Global: reveal whether the proxy rotated it."""
        self.seen.append(np.asarray(observation).copy())  # Keep the exact value passed through.
        return np.zeros(8), None  # SB3 returns (action, state).


def test_proxy_rotates_only_from_onset_and_delegates():
    """Local: count predict calls. Global: keep matched unsteered prefixes for causal pairing."""
    inner = _RecordingModel()  # Recording stand-in for the frozen policy.
    proxy = PerceivedHeadingModel(inner, delta=0.2, onset=2)  # Rotate from the third prediction onward.
    observation = _tilted_observation(np.random.default_rng(2), yaw=0.1)  # One valid observation.
    for _ in range(3):  # Two prefix calls and one active call.
        proxy.predict(observation, deterministic=True)  # Same call signature as runtime.run_episode.
    np.testing.assert_array_equal(inner.seen[0], observation)  # Prefix call 0 is untouched.
    np.testing.assert_array_equal(inner.seen[1], observation)  # Prefix call 1 is untouched.
    np.testing.assert_allclose(inner.seen[2], rotate_ant_observation(observation, 0.2))  # Onset call is rotated.
    assert proxy.policy == "frozen-policy"  # Non-predict attributes reach the wrapped model.


@pytest.fixture(scope="module")
def ant_actor():
    """Local: build an untrained production-shaped Ant actor. Global: test mechanics without claiming locomotion."""
    torch.set_num_threads(1)  # Small batches gain nothing from many threads.
    with gym.make("Ant-v5") as env:  # Authentic 105/8 spaces.
        model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, buffer_size=1, seed=0, device="cpu")  # Same actor layout as the pinned checkpoint.
    model.policy.set_training_mode(False)  # Evaluation mode like prepare_model.
    model.policy.requires_grad_(False)  # run_episode refuses trainable policies.
    return model  # Shared across tests.


def test_proxy_episode_keeps_matched_prefix(ant_actor):
    """Local: run a short real rollout. Global: show the perception arm diverges only at onset."""
    baseline = run_episode(ant_actor, "ant", seed=11, warmup=10, max_steps=15)  # Unmodified reference arm.
    treated = run_episode(PerceivedHeadingModel(ant_actor, delta=0.2, onset=10), "ant", seed=11, warmup=10, max_steps=15)  # Rotated-perception arm.
    assert str(treated["onset_hash"]) == str(baseline["onset_hash"])  # Identical simulator state at onset.
    np.testing.assert_array_equal(treated["actions"][:10], baseline["actions"][:10])  # Identical prefix actions.
    assert not np.array_equal(treated["actions"][10], baseline["actions"][10])  # Perception change acts at onset.


def test_first_relu_matches_runtime_hook_capture(ant_actor):
    """Local: compare offline and hooked activations. Global: make gradients live in the steered space."""
    observation = _tilted_observation(np.random.default_rng(3), yaw=0.0).astype(np.float32)  # One valid input.
    with ActivationIntervention(actor_layer(ant_actor), capture=True) as hook:  # Production capture path.
        ant_actor.predict(observation, deterministic=True)  # Forward pass populates the capture.
    np.testing.assert_allclose(first_relu(ant_actor, observation[None]), hook.activation, rtol=1e-6, atol=1e-6)  # Same first-ReLU output.


def test_perceived_heading_gradient_is_antisymmetric(ant_actor):
    """Local: check the central difference. Global: give the mechanism axis a sign convention."""
    observations = np.stack([_tilted_observation(np.random.default_rng(seed), yaw=0.2) for seed in range(4)])  # Small state sample.
    observations[:, 27:] *= 0.5  # Avoid clip-range edges.
    gradient = perceived_heading_gradient(ant_actor, observations, delta=0.01)  # Mean activation change per radian.
    assert gradient.shape == (256,)  # One direction in first-ReLU space.
    plus = first_relu(ant_actor, rotate_ant_observation(observations, 0.01)).mean(0)  # Forward perceived rotation.
    minus = first_relu(ant_actor, rotate_ant_observation(observations, -0.01)).mean(0)  # Backward perceived rotation.
    np.testing.assert_allclose(gradient, (plus - minus) / 0.02, rtol=1e-5, atol=1e-5)  # Central-difference definition.

"""Local: apply a yaw-like observation transform. Global: test a heading-related account of activation steering.

The first actor layer is a function of the current observation only, so a contrast
vector at the first ReLU depends on observed state. This module preserves the
historical observation-level comparator while leaving the simulator untouched.
Its quaternion and world velocity rotate exactly, but already-clipped contact
wrenches lose information: clip(R clip(f)) is not generally clip(R f). Thus this
comparator is not an exact physical-state rotation when wrench channels saturate.

Sources: Ant-v5 observation = qpos[2:], qvel, clip(cfrc_ext[1:])
https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py ;
c-frame quantities use "a global frame centered at the subtree center of mass"
https://mujoco.readthedocs.io/en/3.1.5/computation/index.html ; cfrc_ext layout is
rotation:translation https://mujoco.readthedocs.io/en/3.1.5/APIreference/APItypes.html#mjdata ;
free-joint linear velocity is global-frame and angular velocity body-frame
https://mujoco.readthedocs.io/en/3.1.5/overview.html#floating-objects , also verified with mj_objectVelocity in tests/test_perception.py.
Quaternion composition: https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
"""
from __future__ import annotations  # Keep annotations lightweight.

from typing import Any  # SB3 models are dynamic upstream objects.

import numpy as np  # Vectorized rotations over whole observation batches.
import torch  # Offline first-layer evaluation of the frozen actor.

from .runtime import actor_layer  # Reuse the validated layout check and intervention site.

QUATERNION = slice(1, 5)  # qpos[3:7] after Ant drops x and y: scalar-first torso orientation.
LINEAR_VELOCITY = slice(13, 15)  # qvel[0:2]: free-joint linear velocity in the world frame.
CONTACT_FORCES = slice(27, 105)  # cfrc_ext[1:]: 13 bodies × (torque xyz, force xyz), world-oriented.
CONTACT_LIMIT = 1.0  # Ant-v5 default contact_force_range is (-1, 1); the observation is clipped to it.


def _rotate_planar(x: np.ndarray, y: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Local: rotate planar components counterclockwise. Global: share one rotation for velocities and forces."""
    c, s = np.cos(delta), np.sin(delta)  # Standard 2-D rotation coefficients.
    return c * x - s * y, s * x + c * y  # Positive delta turns +x toward +y.


def rotate_ant_observation(observation: np.ndarray, delta: float) -> np.ndarray:
    """Local: apply the recorded yaw-like transform. Global: change policy input without changing the simulator.

    Joint angles, joint velocities, height and the body-frame angular velocity are
    invariant under a world yaw rotation; the torso quaternion, world linear
    velocity and world-oriented contact wrenches rotate. Contact wrenches are
    rotated from their already-clipped values and re-clipped. This preserves the
    evaluated comparator, but cannot recover a physical yaw rotation's clipped
    wrenches from saturated input channels; see tests/test_perception.py.
    """
    rotated = np.array(observation, dtype=np.float64, copy=True)  # Never alias the caller's array.
    if rotated.shape[-1] != 105:
        raise ValueError("expected Ant-v5 observations with 105 elements")  # The layout constants are Ant-specific.
    w, z = np.cos(delta / 2), np.sin(delta / 2)  # Yaw quaternion (w, 0, 0, z).
    q = rotated[..., QUATERNION].copy()  # Original orientation (w, x, y, z).
    rotated[..., QUATERNION] = np.stack([w * q[..., 0] - z * q[..., 3], w * q[..., 1] - z * q[..., 2],
                                         w * q[..., 2] + z * q[..., 1], w * q[..., 3] + z * q[..., 0]], axis=-1)  # Hamilton product q_yaw ⊗ q.
    velocity = rotated[..., LINEAR_VELOCITY]  # View into the planar velocity block.
    velocity[..., 0], velocity[..., 1] = _rotate_planar(velocity[..., 0].copy(), velocity[..., 1].copy(), delta)  # World-frame velocity turns with the world.
    wrench = rotated[..., CONTACT_FORCES].reshape(*rotated.shape[:-1], 13, 6)  # Copy-free view per body when contiguous.
    for start in (0, 3):  # Torque block, then force block.
        wrench[..., start], wrench[..., start + 1] = _rotate_planar(wrench[..., start].copy(), wrench[..., start + 1].copy(), delta)  # Rotate x/y; z is invariant.
    rotated[..., CONTACT_FORCES] = np.clip(wrench.reshape(*rotated.shape[:-1], 78), -CONTACT_LIMIT, CONTACT_LIMIT)  # Local: keep the evaluated transform inside the clip range; global: this does not reconstruct lost raw forces.
    return rotated  # Same shape as the input.


class PerceivedHeadingModel:
    """Local: wrap one frozen SB3 model for one episode. Global: rotate perception only after the unsteered prefix.

    ``runtime.run_episode`` calls ``predict`` exactly once per step, so the call
    count equals the step index. Create a new wrapper for every episode.
    """

    def __init__(self, model: Any, delta: float, onset: int):
        """Local: store the wrapped model and schedule. Global: keep the policy weights untouched."""
        if not np.isfinite(delta) or onset < 0:
            raise ValueError("delta must be finite and onset nonnegative")  # Reject ill-defined interventions.
        self._model, self.delta, self.onset, self.calls = model, float(delta), int(onset), 0  # Fresh per-episode counter.

    def __getattr__(self, name: str) -> Any:
        """Local: delegate everything else. Global: let runtime hash, hook and freeze checks see the real model."""
        return getattr(self._model, name)  # Only called for attributes not defined on the wrapper.

    def predict(self, observation, deterministic: bool = True, **kwargs):
        """Local: rotate after onset, then predict. Global: change what the policy sees, not the physics."""
        active = self.calls >= self.onset  # Step index of this call.
        self.calls += 1  # Advance the schedule once per environment step.
        if active and self.delta != 0.0:
            observation = rotate_ant_observation(observation, self.delta).astype(np.asarray(observation).dtype)  # Keep the simulator's dtype.
        return self._model.predict(observation, deterministic=deterministic, **kwargs)  # Unmodified SB3 inference.


def first_relu(model: Any, observations: np.ndarray) -> np.ndarray:
    """Local: evaluate actor.latent_pi[0:2] offline. Global: work in the exact space where vectors are added."""
    actor_layer(model)  # Validate the two-layer layout before indexing it.
    layers = model.actor.latent_pi  # SB3 actor trunk: Linear, ReLU, Linear, ReLU.
    with torch.no_grad():  # Inference only.
        features = model.actor.extract_features(torch.as_tensor(np.asarray(observations), dtype=torch.float32), model.actor.features_extractor)  # SB3's own preprocessing (Flatten for Box spaces).
        return layers[1](layers[0](features)).numpy()  # First ReLU output, shape (batch, 256).


def perceived_heading_gradient(model: Any, observations: np.ndarray, delta: float = np.radians(2.0)) -> np.ndarray:
    """Local: central difference of first-ReLU activations under perceived rotation. Global: define the heading axis g.

    Positive g is the response to a positive yaw-like observation transform, in
    activation units per radian; the clipped-wrench qualification above applies.
    """
    plus = first_relu(model, rotate_ant_observation(observations, delta)).mean(axis=0, dtype=np.float64)  # Perceived +delta.
    minus = first_relu(model, rotate_ant_observation(observations, -delta)).mean(axis=0, dtype=np.float64)  # Perceived −delta.
    return (plus - minus) / (2 * delta)  # Mean derivative over the supplied states.


def feature_gradient(model: Any, observations: np.ndarray, index: int, epsilon: float) -> np.ndarray:
    """Local: central difference along one observation coordinate. Global: compare vectors with other perceived quantities."""
    plus, minus = np.array(observations, dtype=np.float64), np.array(observations, dtype=np.float64)  # Independent copies.
    plus[:, index] += epsilon  # Raise one perceived quantity.
    minus[:, index] -= epsilon  # Lower it symmetrically.
    return (first_relu(model, plus).mean(axis=0, dtype=np.float64) - first_relu(model, minus).mean(axis=0, dtype=np.float64)) / (2 * epsilon)  # Mean derivative.

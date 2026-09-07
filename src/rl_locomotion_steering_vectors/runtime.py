"""Local: load frozen RL actors and measure MuJoCo rollouts. Global: support causal locomotion experiments.

Sources: Baukit https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py;
SB3 https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/sac.html;
MuJoCo https://mujoco.readthedocs.io/en/3.1.5/APIreference/APItypes.html;
Gymnasium https://github.com/Farama-Foundation/Gymnasium/tree/v1.0.0/gymnasium/envs/mujoco;
Hub https://huggingface.co/docs/huggingface_hub/v0.26.2/en/package_reference/file_download;
video https://imageio.readthedocs.io/en/stable/examples.html.
"""

from __future__ import annotations  # Local: defer type evaluation; global: keep annotations lightweight.

import hashlib  # Local: hash checkpoint/state bytes; global: verify provenance and causal pairing.
import logging  # Local: report runtime events; global: inherit the pipeline's timestamped log handlers.
from contextlib import AbstractContextManager, ExitStack  # Local: own hook/video/environment cleanup; global: prevent resource leaks.
from pathlib import Path  # Local: construct project-local paths; global: contain runtime artifacts.
from typing import Any  # Local: annotate upstream dynamic interfaces; global: avoid redundant wrapper protocols.

import gymnasium as gym  # Local: instantiate official tasks; global: preserve published simulator behavior.
import imageio.v2 as imageio  # Local: stream video frames; global: produce inspectable rollout evidence.
import mujoco  # Local: inspect integration state; global: verify complete paired simulator states.
import numpy as np  # Local: store numerical artifacts; global: avoid executable rollout formats.
import torch  # Local: freeze/instrument actors; global: apply activation-only interventions.
from baukit import Trace  # Local: reuse maintained hook management; global: avoid custom activation plumbing.
from huggingface_hub import hf_hub_download  # Local: retrieve exact revisions; global: make policy provenance reproducible.
from sb3_contrib import TQC  # Local: load the upstream HalfCheetah learner; global: use a genuinely RL-trained policy.
from stable_baselines3 import SAC  # Local: load Ant; global: reuse supported continuous-action inference.

LOGGER = logging.getLogger(__name__)  # Local: use a named logger; global: integrate all phases into one log stream.
MODEL_SPECS = {  # Local: hold immutable upstream identities; global: prevent moving-checkpoint experiments.
    "halfcheetah": {  # Local: name the first research task; global: expose a stable CLI selection.
        "env_id": "HalfCheetah-v5", "algorithm": "TQC",  # Local: select matching simulator/loader; global: preserve the training interface.
        "repo_id": "farama-minari/HalfCheetah-v5-TQC-medium",  # Local: locate upstream weights; global: retain attribution.
        "revision": "b4ce04da6f246f06ae4c4258b8ab39624e6600b4",  # Local: pin the Hub commit; global: eliminate branch drift.
        "filename": "halfcheetah-v5-TQC-medium.zip",  # Local: choose only model weights; global: avoid unnecessary replay-buffer downloads.
        "sha256": "8231919095d5a35a2b799e865e968feca535d5af3f40a4b092947c70a976b7e1",  # Local: verify HF LFS identity; global: reject corrupt/replaced bytes.
    },
    "ant": {  # Local: name the directional task; global: support the next research stage.
        "env_id": "Ant-v5", "algorithm": "SAC",  # Local: match published task/algorithm; global: keep baseline provenance accurate.
        "repo_id": "farama-minari/Ant-v5-SAC-medium",  # Local: identify the upstream repository; global: retain attribution.
        "revision": "de5978d34b0118341df2b0a30232c5d2bfcb148e",  # Local: pin the evaluated revision; global: make repeated runs comparable.
        "filename": "ant-v5-SAC-medium.zip",  # Local: distinguish the uppercase SAC artifact; global: avoid a different checkpoint in the same repo.
        "sha256": "dea0e6dbb847776ac9e91dc8b7df5ba975742924eb55831d2da08a0abc786ca2",  # Local: use the pinned HF LFS digest; global: verify downloaded weights.
    },
}
METRIC_NAMES = (  # Local: standardize recorded measurements; global: share one artifact/summary contract.
    "x_velocity", "y_velocity", "planar_speed", "height", "yaw_rate",  # Local: measure physical behavior; global: support multiple steering targets.
    "effort", "inversion", "bad_contact", "reward", "saturation",  # Local: record costs/failure proxies; global: distinguish useful control from disruption.
)


def parameter_hash(model: Any) -> str:
    """Local: hash policy state tensors. Global: detect accidental learning or buffer mutation during experiments."""
    digest = hashlib.sha256()  # Local: start a stable digest; global: cheaply compare complete policy states.
    for name, tensor in model.policy.state_dict().items():  # Local: include parameters and buffers; global: cover preprocessing-relevant state.
        digest.update(name.encode())  # Local: encode tensor identity; global: avoid collisions between different layouts.
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())  # Local: hash actual values; global: avoid pointer-based equality.
    return digest.hexdigest()  # Local: produce a portable string; global: store integrity evidence in manifests.


def actor_layer(model: Any) -> torch.nn.Module:
    """Local: validate the supported actor layout. Global: guarantee steering leaves a downstream nonlinear layer."""
    layers = model.actor.latent_pi  # Local: access SB3's actor trunk; global: avoid touching critic activations.
    valid = len(layers) == 4 and isinstance(layers[0], torch.nn.Linear) and isinstance(layers[2], torch.nn.Linear)  # Local: check linear positions; global: reject unsupported architectures.
    valid = valid and isinstance(layers[1], torch.nn.ReLU) and isinstance(layers[3], torch.nn.ReLU)  # Local: verify nonlinearities; global: make layer semantics explicit.
    valid = valid and layers[0].out_features == 256 and layers[2].in_features == 256 and layers[2].out_features == 256  # Local: check dimensions; global: prevent misapplied vectors.
    if not valid:  # Local: handle unexpected checkpoints; global: fail before collecting invalid research data.
        raise ValueError(f"Expected two 256-unit Linear/ReLU actor layers, got {layers!r}")  # Local: explain mismatch; global: make diagnosis actionable.
    return layers[1]  # Local: choose the first ReLU output; global: preserve the agreed intervention site.


def prepare_model(model_key: str, root: Path) -> tuple[Any, dict[str, Any]]:
    """Local: download, verify, and freeze one pinned policy. Global: establish reproducible experimental provenance."""
    spec = MODEL_SPECS[model_key]  # Local: reject unknown keys; global: use only specified checkpoint identities.
    destination = Path(root).resolve() / "artifacts" / "base" / model_key  # Local: keep weights inside this project; global: honor workspace containment.
    LOGGER.info("Preparing %s revision=%s", model_key, spec["revision"])  # Local: announce provenance; global: make preparation auditable.
    path = Path(hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"], local_dir=destination))  # Local: reuse Hub revision-aware download; global: retain local cache metadata.
    with path.open("rb") as stream:  # Local: open downloaded bytes; global: verify before deserializing the trusted upstream checkpoint.
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()  # Local: stream the checksum; global: verify exact LFS content identity.
    if checksum != spec["sha256"]:  # Local: compare authoritative bytes; global: prevent corrupt checkpoint execution.
        raise ValueError(f"Checkpoint SHA256 mismatch: expected {spec['sha256']}, got {checksum}")  # Local: expose the mismatch; global: stop provenance failure.
    loader = TQC if spec["algorithm"] == "TQC" else SAC  # Local: select official deserializer; global: preserve algorithm semantics.
    model = loader.load(path, device="cpu")  # Local: load without an attached training environment; global: make inference CPU-reproducible.
    model.policy.set_training_mode(False)  # Local: disable training-specific behavior; global: hold policy computation fixed.
    model.policy.requires_grad_(False)  # Local: freeze every parameter; global: exclude policy learning from steering experiments.
    actor_layer(model)  # Local: assert the intervention layout; global: fail before collecting incompatible activations.
    with gym.make(spec["env_id"]) as env:  # Local: obtain official default spaces; global: check checkpoint/environment compatibility.
        if model.observation_space != env.observation_space or model.action_space != env.action_space:  # Local: compare shape, bounds, and dtypes; global: reject hidden preprocessing mismatch.
            raise ValueError("Checkpoint spaces do not match the unmodified Gymnasium environment")  # Local: flag the precise interface problem; global: avoid silently adapted baselines.
    metadata = dict(spec, model_key=model_key, checkpoint_path=str(path), layer="actor.latent_pi.1", hidden_dimension=256)  # Local: bundle reproducible identity; global: support downstream manifests.
    metadata.update(observation_shape=list(model.observation_space.shape), action_shape=list(model.action_space.shape), parameter_sha256=parameter_hash(model))  # Local: record interfaces/state; global: expose integrity checks to reporting.
    LOGGER.info("Prepared %s sha256=%s actor=%s", model_key, checksum, model.actor.latent_pi)  # Local: report verified architecture; global: leave measured rather than assumed provenance.
    return model, metadata  # Local: expose actor and manifest; global: keep download concerns out of research orchestration.


class ActivationIntervention(AbstractContextManager):
    """Local: reuse Baukit for a managed additive edit. Global: capture untouched activations and restore hooks reliably."""

    def __init__(self, layer: torch.nn.Module, vector: np.ndarray | None = None, *, capture: bool = False):
        """Local: configure one layer/direction. Global: keep intervention state separate from model parameters."""
        self.layer = layer  # Local: retain the actual module; global: avoid repeated string-based lookup.
        self.vector = None if vector is None else torch.as_tensor(np.asarray(vector).copy(), dtype=torch.float32)  # Local: own an immutable direction copy; global: prevent external mutation.
        if self.vector is not None and (self.vector.ndim != 1 or not torch.isfinite(self.vector).all()):  # Local: validate the direction; global: reject malformed interventions early.
            raise ValueError("Steering vector must be a finite one-dimensional array")  # Local: explain the contract; global: avoid broadcasting mistakes.
        self.capture, self.alpha, self.activation = capture, 0.0, None  # Local: start disabled with no capture; global: preserve the unsteered prefix.
        self.trace = None  # Local: defer hook creation; global: let context entry own resources.

    def _edit(self, output: torch.Tensor) -> torch.Tensor:
        """Local: copy pre-edit values and optionally add a vector. Global: separate extraction evidence from intervention artifacts."""
        if not torch.isfinite(output).all():  # Local: inspect layer values; global: fail before propagating invalid activations.
            raise FloatingPointError("Nonfinite actor activation")  # Local: report numerical failure; global: preserve an explicit experiment error.
        if self.capture:  # Local: retain only requested activations; global: reduce memory during evaluation.
            self.activation = output.detach().cpu().numpy().copy()  # Local: capture before editing; global: keep fitting data unsteered.
        if self.vector is None or self.alpha == 0.0:  # Local: bypass disabled offsets; global: provide exact zero-strength identity.
            return output  # Local: return the same tensor object; global: avoid unnecessary arithmetic changes.
        if output.shape[-1] != self.vector.numel() or not np.isfinite(self.alpha):  # Local: check dimensions/strength; global: reject invalid treatment parameters.
            raise ValueError("Steering dimension or strength is invalid")  # Local: explain the failure; global: stop erroneous conditions.
        return output + self.alpha * self.vector.to(device=output.device, dtype=output.dtype)  # Local: broadcast across batches; global: edit activations without changing weights.

    def __enter__(self):
        """Local: attach Baukit only during the context. Global: own exactly one removable hook."""
        self.trace = Trace(self.layer, retain_output=False, edit_output=lambda output: self._edit(output))  # Local: expose a plain-function signature to Baukit's getfullargspec; global: avoid its bound-method self-argument mismatch while reusing hook management.
        return self  # Local: expose mutable strength/captured values; global: allow per-step intervention scheduling.

    def __exit__(self, exc_type, exc_value, traceback):
        """Local: remove hooks on success or failure. Global: prevent one condition from contaminating another."""
        if self.trace is not None:  # Local: handle only acquired resources; global: support safe exception cleanup.
            self.trace.close()  # Local: use Baukit's documented removal; global: restore the original policy function.
        return False  # Local: propagate underlying errors; global: avoid hiding failed experiments.


def quaternion_yaw(quaternion: np.ndarray) -> float:
    """Local: convert MuJoCo's scalar-first quaternion to yaw. Global: measure Ant turning in radians, not quaternion coordinates."""
    value = np.asarray(quaternion, dtype=np.float64)  # Local: normalize numerical representation; global: use stable orientation arithmetic.
    if value.shape != (4,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-12:  # Local: reject invalid rotations; global: do not fabricate turning metrics.
        raise ValueError("Expected a finite nonzero scalar-first quaternion")  # Local: identify malformed state; global: make simulator failure explicit.
    w, x, y, z = value / np.linalg.norm(value)  # Local: restore unit norm; global: tolerate harmless simulator roundoff.
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))  # Local: compute ZYX yaw; global: respect quaternion geometry.


def wrapped_angle_difference(after: float, before: float) -> float:
    """Local: subtract angles on the circle. Global: prevent yaw branch crossings from appearing as rapid turns."""
    return float(np.arctan2(np.sin(after - before), np.cos(after - before)))  # Local: wrap to [-pi, pi]; global: produce continuous small-step angular displacement.


def state_hash(env: Any) -> str:
    """Local: hash MuJoCo's complete integration state. Global: verify more than qpos/qvel at intervention onset."""
    specification = mujoco.mjtState.mjSTATE_INTEGRATION  # Local: include controls, solver warmstart, and physics; global: cover variables affecting future integration.
    state = np.empty(mujoco.mj_stateSize(env.model, specification), dtype=np.float64)  # Local: allocate the documented state buffer; global: avoid incomplete handcrafted state lists.
    mujoco.mj_getState(env.model, env.data, state, specification)  # Local: copy without mutating the simulator; global: retain identical dynamics.
    return hashlib.sha256(state.tobytes()).hexdigest()  # Local: compact state bytes; global: enable portable paired-state assertions.


def bad_ground_contact(data: Any, floor_id: int, forbidden_ids: set[int]) -> float:
    """Local: detect nonpositive-distance floor contact by named torso/head geoms. Global: exclude legitimate foot contacts."""
    for contact in data.contact[:data.ncon]:  # Local: inspect active contact records only; global: avoid unused MuJoCo buffer entries.
        other = int(contact.geom2) if contact.geom1 == floor_id else int(contact.geom1) if contact.geom2 == floor_id else -1  # Local: identify the body touching the floor; global: ignore self-collisions.
        if other in forbidden_ids and contact.dist <= 0.0:  # Local: require geometric contact rather than only collision margin; global: define the failure proxy transparently.
            return 1.0  # Local: flag this timestep once; global: let summary means represent contact fractions.
    return 0.0  # Local: indicate no inappropriate contact; global: retain an unbiased per-step denominator.


def _pose(env: Any, model_key: str) -> tuple[np.ndarray, float, float]:
    """Local: read post-integration root pose. Global: avoid stale derived-body positions and incorrect HalfCheetah height offsets."""
    qpos = env.data.qpos  # Local: read final generalized positions; global: measure the state after the last MuJoCo substep.
    if model_key == "halfcheetah":  # Local: handle the planar slide/hinge root; global: respect the environment's actual XML layout.
        position = np.array([qpos[0], 0.0, qpos[1] + env.model.body("torso").pos[2]])  # Local: add the XML torso-height offset; global: report metres above ground.
        return position, 0.0, float(np.cos(qpos[2]) < 0.0)  # Local: planar yaw is fixed and pitch determines inversion; global: avoid nonsensical 3D rotation claims.
    quaternion = qpos[3:7]  # Local: read the Ant free-joint orientation; global: preserve scalar-first component ordering.
    quaternion = quaternion / np.linalg.norm(quaternion)  # Local: normalize small floating-point drift; global: compute a consistent uprightness predicate.
    upright = 1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2)  # Local: compute body z-axis dot world z-axis; global: detect inversion independent of heading.
    return qpos[:3].copy(), quaternion_yaw(quaternion), float(upright < 0.0)  # Local: return position/yaw/inversion; global: share one physical measurement interface.


def run_episode(model: Any, model_key: str, seed: int, *, vector=None, alpha=0.0, warmup=100, max_steps=1000,
                deterministic=True, capture=False, video_path=None, action_bias=None, off_at=None) -> dict[str, np.ndarray]:
    """Local: execute one intervention rollout. Global: retain all transitions, matched prefixes, frozen weights, and streamed evidence."""
    if warmup < 0 or max_steps <= 0 or not np.isfinite(alpha):  # Local: validate the run schedule; global: reject unusable experiment specifications.
        raise ValueError("Require warmup >= 0, max_steps > 0, and finite alpha")  # Local: explain the contract; global: fail before acquiring simulator resources.
    spec, layer = MODEL_SPECS[model_key], actor_layer(model)  # Local: resolve environment and validated layer; global: preserve model/task coupling.
    if any(parameter.requires_grad for parameter in model.policy.parameters()):  # Local: inspect all parameters; global: require an explicitly frozen base policy.
        raise ValueError("Run episodes only after freezing all policy parameters")  # Local: identify the setup error; global: enforce the causal research boundary.
    before_hash = parameter_hash(model)  # Local: record actor/critic/buffer values; global: verify no hidden learning occurred.
    values = {name: [] for name in (*METRIC_NAMES, "observations", "actions")}  # Local: retain per-transition numerical evidence; global: enable independent recomputation.
    if capture:  # Local: allocate only requested hidden-state storage; global: keep evaluation memory bounded.
        values["activations"] = []  # Local: collect original hidden vectors; global: support difference-of-means extraction.
    onset, terminated, truncated = "", False, False  # Local: initialize episode status; global: preserve pre-onset failure explicitly.
    with ExitStack() as stack:  # Local: centralize cleanup ordering; global: remove hooks, video writers, and simulators even on error.
        env = stack.enter_context(gym.make(spec["env_id"], render_mode="rgb_array" if video_path else None))  # Local: keep normal task defaults; global: preserve upstream dynamics and horizon.
        physics = env.unwrapped  # Local: access official MuJoCo state; global: measure physical quantities rather than observation guesses.
        obs, _ = env.reset(seed=int(seed))  # Local: seed the simulator; global: create reproducible reset conditions.
        torch.manual_seed(int(seed))  # Local: seed policy sampling separately per episode; global: make stochastic fitting runs reproducible.
        floor = int(physics.model.geom("floor").id)  # Local: resolve floor by XML name; global: avoid brittle numeric geometry assumptions.
        forbidden = {int(physics.model.geom(name).id) for name in (("torso", "head") if model_key == "halfcheetah" else ("torso_geom",))}  # Local: choose actual body geoms; global: detect inappropriate support only.
        writer = None  # Local: default to no video resource; global: keep numerical experiments inexpensive.
        if video_path:  # Local: create visual evidence only when requested; global: avoid unnecessary rendering overhead.
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)  # Local: create the output directory; global: keep the video alongside experiment artifacts.
            writer = stack.enter_context(imageio.get_writer(str(video_path), fps=env.metadata.get("render_fps", round(1 / physics.dt)), codec="libx264", macro_block_size=1))  # Local: stream MP4 at simulation rate; global: avoid accumulating frame arrays in memory.
            writer.append_data(env.render())  # Local: show the initial pose; global: document reset conditions.
        intervention = stack.enter_context(ActivationIntervention(layer, vector, capture=capture))  # Local: attach one episode-scoped hook; global: guarantee condition cleanup.
        bias = None if action_bias is None else np.asarray(action_bias, dtype=np.float32)  # Local: prepare the comparator offset; global: keep action-bias controls explicit.
        if bias is not None and (bias.shape != env.action_space.shape or not np.isfinite(bias).all()):  # Local: validate action comparator dimensions; global: prevent silent broadcasting.
            raise ValueError("Action bias must be a finite action-shaped array")  # Local: describe the bad input; global: fail comparators transparently.
        for step in range(max_steps):  # Local: execute a bounded episode; global: let orchestration resume experiments between episodes.
            if step == warmup:  # Local: identify the exact first treatment action; global: establish matched causal onset.
                onset = state_hash(physics)  # Local: capture integration state before intervention; global: allow equality checks across conditions.
            active = step >= warmup and (off_at is None or step < off_at)  # Local: schedule the treatment interval; global: support off/on/off demonstrations.
            intervention.alpha = alpha if active else 0.0  # Local: enable only the intended offset; global: protect the unsteered prefix.
            position_before, yaw_before, _ = _pose(physics, model_key)  # Local: measure the transition's starting pose; global: compute actual displacement rates.
            if not np.isfinite(obs).all():  # Local: validate input state; global: do not hide simulator instability.
                raise FloatingPointError(f"Nonfinite observation at seed={seed} step={step}")  # Local: identify the failed transition; global: enable logged diagnosis.
            with torch.no_grad():  # Local: avoid graph construction during prediction; global: prevent unintentional optimization paths.
                action, _ = model.predict(obs, deterministic=deterministic)  # Local: reuse normal SB3 action scaling; global: preserve baseline policy semantics.
            if bias is not None and active:  # Local: apply comparator only after startup; global: give controls matched prefixes.
                action = np.clip(action + bias, env.action_space.low, env.action_space.high)  # Local: respect physical action bounds; global: compare valid control signals.
            if not np.isfinite(action).all():  # Local: inspect applied actions; global: fail rather than discarding bad transitions.
                raise FloatingPointError(f"Nonfinite action at seed={seed} step={step}")  # Local: locate numerical failure; global: preserve actionable logs.
            values["observations"].append(np.asarray(obs).copy())  # Local: store the action's input; global: maintain correct state/action alignment.
            values["actions"].append(np.asarray(action).copy())  # Local: store actual bounded controls; global: support effort and bias recomputation.
            if capture:  # Local: retain requested hidden values; global: make extraction independent of model reexecution.
                values["activations"].append(intervention.activation[0].copy())  # Local: remove the single-env batch axis; global: expose shape [time, hidden].
            obs, reward, terminated, truncated, _ = env.step(action)  # Local: call unmodified Gymnasium transition; global: preserve reward/termination semantics.
            position, yaw, inversion = _pose(physics, model_key)  # Local: read the resulting physical pose; global: measure effects after dynamics.
            velocity = (position[:2] - position_before[:2]) / physics.dt  # Local: divide physical displacement by elapsed simulation time; global: report m/s.
            metrics = dict(x_velocity=velocity[0], y_velocity=velocity[1], planar_speed=np.linalg.norm(velocity), height=position[2],  # Local: record kinematics; global: support speed/posture/direction targets.
                           yaw_rate=wrapped_angle_difference(yaw, yaw_before) / physics.dt, effort=np.mean(np.square(action)),  # Local: compute rad/s and squared-control proxy; global: avoid claiming physical energy.
                           inversion=inversion, bad_contact=bad_ground_contact(physics.data, floor, forbidden), reward=reward,  # Local: retain health indicators and original reward; global: evaluate locomotion competence.
                           saturation=np.mean((action <= env.action_space.low + 0.01) | (action >= env.action_space.high - 0.01)))  # Local: count controls near limits; global: detect intervention saturation.
            for name, value in metrics.items():  # Local: append one value per metric; global: preserve timestep alignment.
                if not np.isfinite(value):  # Local: reject undefined measurements; global: do not convert numerical failures into successful evidence.
                    raise FloatingPointError(f"Nonfinite {name} at seed={seed} step={step}")  # Local: identify the failing metric; global: support reliable diagnosis.
                values[name].append(float(value))  # Local: retain full scalar precision; global: support independent statistical analysis.
            if writer is not None:  # Local: render only requested runs; global: keep core evaluation efficient.
                writer.append_data(env.render())  # Local: encode the frame immediately; global: keep video memory constant.
            if terminated or truncated:  # Local: obey true Gymnasium episode boundaries; global: retain failed/short trajectories without extra simulation.
                break  # Local: finish this recorded episode; global: never silently reset within an artifact.
        dt = float(physics.dt)  # Local: save the physical timestep before closing; global: make distance calculations reproducible.
    if parameter_hash(model) != before_hash:  # Local: compare complete policy values; global: verify the intervention did not mutate the learner.
        raise RuntimeError("Policy state changed during an inference-only episode")  # Local: expose an invalid experiment; global: prohibit false frozen-policy claims.
    result = {name: np.asarray(value) for name, value in values.items()}  # Local: finalize non-executable numerical arrays; global: support compressed NPZ storage.
    result.update(seed=np.array(seed), onset_hash=np.array(onset), terminated=np.array(terminated), truncated=np.array(truncated), length=np.array(len(values["reward"])), dt=np.array(dt))  # Local: retain scalar episode metadata; global: keep failures and pair identities auditable.
    LOGGER.info("Episode model=%s seed=%d alpha=%g deterministic=%s length=%d return=%.3f terminated=%s", model_key, seed, alpha, deterministic, len(values["reward"]), sum(values["reward"]), terminated)  # Local: report completion in real time; global: allow progress and failure inspection.
    return result  # Local: expose the complete episode; global: separate numerical collection from storage/reporting.


def summarise_episode(episode: dict[str, np.ndarray], warmup: int = 100) -> dict[str, Any]:
    """Local: aggregate recorded post-startup metrics. Global: retain early failures in every comparison denominator."""
    length = int(episode["length"])  # Local: read actual recorded duration; global: avoid assuming every episode survived the horizon.
    summary = {name: float(np.mean(episode[name][warmup:])) if length > warmup else 0.0 for name in METRIC_NAMES}  # Local: average the intervention interval; global: use explicit zeros for missing post-onset behavior.
    summary.update(length=float(length), failure=float(bool(episode["terminated"]) or length <= warmup),  # Local: preserve early termination status; global: do not exclude failed trajectories.
                   return_=float(np.sum(episode["reward"])), distance=float(np.sum(episode["x_velocity"][warmup:]) * float(episode.get("dt", 0.05))))  # Local: retain original return and physical displacement; global: distinguish full-episode reward from post-onset behavior.
    summary["return"] = summary.pop("return_")  # Local: expose the public metric name; global: keep dictionaries natural despite Python's reserved keyword.
    summary.update(seed=int(episode.get("seed", -1)), onset_hash=str(episode.get("onset_hash", "")))  # Local: preserve pairing identity as Python scalars; global: let statistics verify matched simulator prefixes.
    return summary  # Local: provide comparable scalar measurements; global: support paired statistics without reloading simulator code.

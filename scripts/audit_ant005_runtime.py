"""Local: reproduce the Ant005 runtime checks; global: publish evidence without changing the experiment.

Run from the project root: uv run --no-sync python scripts/audit_ant005_runtime.py
This preserves the earlier inline audit: 28 actor checks and two existing 1000-action replays.
Sources: https://github.com/DLR-RM/stable-baselines3/blob/v2.4.1/stable_baselines3/sac/policies.py#L147-L176
https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/mujoco/ant_v5.py#L230-L393
https://github.com/davidbau/baukit/blob/9d51abd51ebf29769aecc38c4cbef459b731a36e/baukit/nethook.py#L65-L101
https://docs.python.org/3.12/library/urllib.request.html#urllib.request.urlopen
https://docs.python.org/3.12/library/hashlib.html
"""
import hashlib  # Local: fingerprint actual inputs; global: identify exactly what the audit verifies.
import importlib.metadata  # Local: record installed versions; global: accompany source hashes with reproducible dependency identities.
import inspect  # Local: locate installed official modules; global: compare the code actually imported by inference.
import json  # Local: serialize complete evidence; global: publish non-executable audit results.
import logging  # Local: reuse timestamped handlers; global: preserve the entire result in a terminal/file log.
import urllib.request  # Local: read pinned official source text; global: check local code against its stated provenance.
from pathlib import Path  # Local: keep all reads and outputs in this project; global: honor workspace scope.

import gymnasium as gym  # Local: instantiate original Ant-v5; global: replay without reward or dynamics adaptation.
import gymnasium.envs.mujoco.ant_v5 as ant  # Local: identify the imported environment implementation; global: verify upstream source equality.
import numpy as np  # Local: compare numerical bytes; global: avoid hiding drift behind a tolerance.
import stable_baselines3.sac.policies as sac  # Local: identify the imported actor; global: verify the original action path.
import torch  # Local: reconstruct the deterministic actor computation; global: test the hook against a separate calculation.
from rl_locomotion_steering_vectors.runtime import ActivationIntervention, _pose, actor_layer, bad_ground_contact, parameter_hash, prepare_model, state_hash, wrapped_angle_difference  # Local: exercise the production hook and physical metrics; global: compare them with saved evidence and a separate actor oracle.
from rl_locomotion_steering_vectors.storage import load_arrays, load_json, save_json, start_logging, utc_now  # Local: reuse artifact/logging utilities; global: leave original scientific inputs unchanged.


def verify_actions(model, run, vector):
    """Local: check saved actions and an explicit SAC chain; global: rule out an alternate hardcoded action path."""
    before, checks, displacements = parameter_hash(model), [], []  # Local: retain the frozen-state oracle and check locations; global: make the audit's bounded coverage explicit.
    assert all(not parameter.requires_grad for parameter in model.policy.parameters())  # Local: inspect every parameter; global: reject accidental learning during inference.
    assert not model.policy.training and not model.actor.training and not model.actor.use_sde  # Local: verify deterministic evaluation settings; global: justify the tanh-mean reconstruction below.
    for seed in (510000, 520000):  # Local: use one already measured validation and confirmation seed; global: consume no fresh resets.
        phase = "validation" if seed < 520000 else "confirmation"  # Local: locate each existing episode; global: keep the source split explicit.
        for condition in ("baseline", "lateral__-0.1000"):  # Local: compare both saved intervention arms; global: preserve paired evidence.
            episode = load_arrays(run / "episodes" / phase / condition / f"{seed}.npz")  # Local: read immutable arrays; global: never overwrite the experimental trajectory.
            for index in (0, 99, 100, 101, 250, 500, int(episode["length"]) - 1):  # Local: check startup, onset and later observations; global: match the original 28-location audit.
                obs = episode["observations"][index]  # Local: use the actual recorded policy input; global: compare identical physical observations.
                alpha = -.1 if condition != "baseline" and index >= 100 else 0  # Local: restore the registered intervention schedule; global: check the same condition as the cache.
                original = model.predict(obs, deterministic=True)[0]  # Local: obtain the unedited policy output; global: provide the identity/cleanup oracle.
                with ActivationIntervention(actor_layer(model), vector) as edit:  # Local: exercise actual Baukit integration; global: test the deployed implementation.
                    edit.alpha = alpha  # Local: set the already selected strength; global: perform no new calibration.
                    actual = model.predict(obs, deterministic=True)[0]  # Local: infer through the intervention; global: reproduce the original action.
                assert actual.dtype == episode["actions"][index].dtype and actual.tobytes() == episode["actions"][index].tobytes(), (seed, condition, index)  # Local: require exact stored-action agreement; global: detect an alternate action source.
                restored = model.predict(obs, deterministic=True)[0]  # Local: predict after context cleanup; global: verify restoration of the policy function.
                assert restored.tobytes() == original.tobytes()  # Local: compare identical inputs after hook removal; global: reject cross-condition contamination.
                with torch.no_grad():  # Local: build no training graph; global: preserve frozen inference behavior.
                    tensor = model.policy.obs_to_tensor(obs)[0]  # Local: retain SB3 input conversion; global: avoid an artificial preprocessing mismatch.
                    features = model.actor.extract_features(tensor, model.actor.features_extractor)  # Local: reuse the original feature extractor; global: isolate the hook as the tested difference.
                    hidden = model.actor.latent_pi[1](model.actor.latent_pi[0](features)) + alpha * torch.as_tensor(vector)  # Local: express the first-layer addition explicitly without the hook adapter; global: independently reconstruct its intended operation.
                    downstream = model.actor.latent_pi[3](model.actor.latent_pi[2](hidden))  # Local: retain the downstream nonlinearity; global: preserve the real actor architecture.
                    mean = model.actor.mu(downstream)  # Local: compute the actual SAC action mean; global: do not fit or substitute an action rule.
                    reconstructed = model.policy.unscale_action(torch.tanh(mean).cpu().numpy())[0]  # Local: apply SAC's deterministic squashing and unscaling; global: complete the independent action oracle.
                assert reconstructed.tobytes() == actual.tobytes(), (seed, condition, index, "manual")  # Local: require exact oracle agreement; global: verify the edit reaches the actuator output through SAC.
                checks.append([seed, condition, index])  # Local: retain every checked location; global: disclose audit coverage.
                if alpha:  # Local: describe active observations only; global: distinguish real displacement from prefix identity.
                    displacements.append((actual - original).tolist())  # Local: save action differences; global: test whether the candidate is merely a constant action bias.
    norms = np.linalg.norm(displacements, axis=1)  # Local: measure the observed intervention-induced changes; global: publish their range without new behavior selection.
    assert parameter_hash(model) == before and len(actor_layer(model)._forward_hooks) == 0  # Local: check state and hook cleanup; global: reject hidden learning or residual treatment.
    return {"check_locations": checks, "count": len(checks), "stored_actions_and_manual_chain_bitwise_equal": True, "zero_identity_and_cleanup_equal": True, "all_parameters_frozen": True, "policy_state_unchanged": True, "remaining_hooks": 0, "action_displacement": {"samples": len(displacements), "values": displacements, "coordinate_std": np.std(displacements, axis=0).tolist(), "min_norm": float(norms.min()), "max_norm": float(norms.max())}}  # Local: expose complete results; global: make the prose claim independently inspectable.


def verify_physics(run):
    """Local: replay existing controls through default Ant; global: detect reward, geometry, or state drift."""
    records, environment = [], None  # Local: collect both replay results and shared settings; global: preserve the exact audit scope.
    for condition in ("baseline", "lateral__-0.1000"):  # Local: retain baseline and treatment; global: test the same original validation comparison.
        episode = load_arrays(run / "episodes" / "validation" / condition / "510000.npz")  # Local: replay already measured actions; global: create no new policy experiment.
        errors = {key: 0 for key in ("observations", "actions", "reward", "x_velocity", "y_velocity", "height", "yaw_rate", "inversion", "bad_contact")}  # Local: count all discrepancies; global: avoid sampled-only simulator verification.
        maximum_info_difference, onset = 0., ""  # Local: retain the original metric convention comparison; global: distinguish physical metrics from reward intermediates.
        with gym.make("Ant-v5") as env:  # Local: use default task settings and wrappers; global: exclude reward or termination modifications.
            physics = env.unwrapped  # Local: inspect actual integration data; global: measure physical outcomes directly.
            observation, _ = env.reset(seed=510000)  # Local: restore the existing reset; global: preserve the original paired onset.
            floor, torso = int(physics.model.geom("floor").id), int(physics.model.geom("torso_geom").id)  # Local: resolve actual named collision geoms; global: avoid fabricated contact indices.
            environment = {"wrappers": str(env), "dt": physics.dt, "frame_skip": physics.frame_skip, "forward_reward_weight": physics._forward_reward_weight, "ctrl_cost_weight": physics._ctrl_cost_weight, "contact_cost_weight": physics._contact_cost_weight, "healthy_reward": physics._healthy_reward, "healthy_z_range": physics._healthy_z_range, "terminate_when_unhealthy": physics._terminate_when_unhealthy, "floor_id": floor, "torso_geom_id": torso, "observation_structure": physics.observation_structure}  # Local: expose actual settings; global: substantiate the unchanged-environment claim.
            for index, action in enumerate(episode["actions"]):  # Local: apply every stored control in order; global: reproduce the existing trajectory only.
                if index == 100:  # Local: locate the registered activation onset; global: compare the original complete state hash.
                    onset = state_hash(physics)  # Local: read integration state before the first edited action; global: preserve the pairing boundary.
                errors["observations"] += int(observation.tobytes() != episode["observations"][index].tobytes())  # Local: compare the entire policy-input sequence; global: detect state/preprocessing drift.
                before, yaw_before, _ = _pose(physics, "ant")  # Local: retain root pose before the transition; global: calculate the same declared physical quantities.
                observation, reward, terminated, truncated, info = env.step(action)  # Local: use the ordinary simulator and reward; global: avoid any artificial lateral forcing.
                after, yaw_after, inversion = _pose(physics, "ant")  # Local: inspect final root pose; global: verify the saved research measurements.
                velocity = (after[:2] - before[:2]) / physics.dt  # Local: derive actual world displacement velocity; global: avoid treating reward as a motion sensor.
                errors["actions"] += int(not np.array_equal(physics.data.ctrl.astype(action.dtype), action))  # Local: verify applied actuator controls; global: check delivery rather than input copying alone.
                metrics = {"reward": reward, "x_velocity": velocity[0], "y_velocity": velocity[1], "height": after[2], "yaw_rate": wrapped_angle_difference(yaw_after, yaw_before) / physics.dt, "inversion": inversion, "bad_contact": bad_ground_contact(physics.data, floor, {torso})}  # Local: reconstruct recorded descriptors; global: keep the numerical audit aligned with the original report.
                for key, value in metrics.items():  # Local: compare all measured fields at every transition; global: reject unnoticed output substitution.
                    errors[key] += int(np.asarray(value, dtype=episode[key].dtype).tobytes() != episode[key][index].tobytes())  # Local: preserve dtype and bytes; global: use no tolerance that could conceal drift.
                maximum_info_difference = max(maximum_info_difference, abs(info["x_velocity"] - velocity[0]), abs(info["y_velocity"] - velocity[1]))  # Local: expose root-qpos versus derived-xpos differences; global: document the original reward's distinct velocity convention.
                expected_reward = (info["reward_forward"] + info["reward_survive"]) - (-info["reward_ctrl"] - info["reward_contact"])  # Local: reconstruct the unmodified reward terms; global: test the reward-hack concern directly.
                assert np.float64(expected_reward).tobytes() == np.float64(reward).tobytes() and info["reward_forward"] == info["x_velocity"]  # Local: verify original forward reward arithmetic; global: rule out substituting the lateral label into reward.
                if terminated or truncated:  # Local: honor actual environment endings; global: preserve episode semantics during replay.
                    break  # Local: stop without reset or continuation; global: never hide a failure.
            assert not any(errors.values()) and index + 1 == int(episode["length"])  # Local: verify the whole recorded duration; global: fail visibly on any numerical discrepancy.
            assert onset == str(episode["onset_hash"]) and terminated == bool(episode["terminated"]) and truncated == bool(episode["truncated"])  # Local: check causal onset and final metadata; global: retain exact paired semantics.
        records.append({"condition": condition, "seed": 510000, "steps": index + 1, "field_mismatch_counts": errors, "onset_equal": True, "termination_equal": True, "truncation_equal": True, "original_reward_decomposition_equal": True, "max_qpos_vs_info_velocity_difference": maximum_info_difference})  # Local: publish the full comparison result; global: keep the audit reviewable without replaying it first.
    return {"environment": environment, "replays": records}  # Local: return numerical evidence; global: leave serialization to the thin wrapper.


def verify_sources():
    """Local: compare installed sources with pinned originals; global: validate environment and actor provenance."""
    sources = [(Path(inspect.getfile(ant)), "https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v1.0.0/gymnasium/envs/mujoco/ant_v5.py"), (Path(inspect.getfile(sac)), "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/v2.4.1/stable_baselines3/sac/policies.py")]  # Local: use actual imported modules; global: compare exact pinned implementations.
    with gym.make("Ant-v5") as env:  # Local: resolve the environment's actual XML path; global: avoid assuming which asset was loaded.
        sources.append((Path(env.unwrapped.fullpath), "https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v1.0.0/gymnasium/envs/mujoco/assets/ant.xml"))  # Local: include collision and actuator definitions; global: test physical-model provenance too.
    result = []  # Local: collect source equality records; global: expose every external check.
    for path, url in sources:  # Local: check each pinned source; global: fail rather than silently fall back to another version.
        with urllib.request.urlopen(url, timeout=30) as response:  # Local: bound the source read and close it; global: keep network failures explicit.
            upstream = response.read().decode("utf-8").replace("\r\n", "\n")  # Local: normalize platform newlines; global: compare source content rather than checkout convention.
        local = path.read_text(encoding="utf-8")  # Local: read the actually installed code; global: retain equivalent universal-newline semantics.
        assert local == upstream, str(path)  # Local: require full text equality; global: reject an undocumented patched dependency.
        result.append({"path": str(path.relative_to(Path.cwd())), "url": url, "normalized_text_sha256": hashlib.sha256(local.encode()).hexdigest(), "text_equal": True})  # Local: retain actual source identity; global: make the provenance check reproducible.
    return result  # Local: return complete source evidence; global: share one output artifact.


def main():
    """Local: run the preserved bounded audit; global: publish evidence without changing research inputs."""
    if not __debug__:  # Local: detect disabled assertions; global: prevent an unchecked invocation from claiming success.
        raise RuntimeError("Run this audit without Python -O; its checks use assertions")  # Local: explain the invocation requirement; global: preserve audit integrity.
    root = Path.cwd()  # Local: anchor all work to the current project; global: avoid unrelated files.
    run, folder = root / "runs" / "ant-classic-005", root / "reports" / "ant-classic-005"  # Local: separate immutable inputs and published audit outputs; global: preserve experiment history.
    start_logging(folder)  # Local: stream complete terminal/file records; global: satisfy reviewable audit logging.
    torch.set_num_threads(1)  # Local: retain original inference execution settings; global: avoid numerical differences from thread configuration.
    model, metadata = prepare_model("ant", root)  # Local: verify and freeze the exact upstream checkpoint; global: exercise the actual project load path.
    vector = load_arrays(run / "vectors.npz")["lateral"]  # Local: use the deployed payload; global: never refit or select another direction.
    identity = load_json(run / "episodes" / "validation" / "lateral__-0.1000" / "identity.json")  # Local: read the original condition; global: reject a changed policy/vector/runtime.
    runtime_hash = hashlib.sha256((root / "src" / "rl_locomotion_steering_vectors" / "runtime.py").read_text(encoding="utf-8").encode()).hexdigest()  # Local: use the cache's normalized-newline convention; global: match the original code identity.
    assert metadata["parameter_sha256"] == identity["policy_sha256"] and runtime_hash == identity["runtime_sha256"]  # Local: require original runtime/policy state; global: disallow auditing a replacement as if it were the experiment.
    assert hashlib.sha256(vector.tobytes()).hexdigest() == identity["vector_sha256"] and identity["bias_sha256"] is None and identity["alpha"] == -.1  # Local: check the actual activation-only condition; global: prevent comparator or sign substitution.
    watched = [run / "manifest.json", run / "vectors.npz", run / "selection.json"] + [run / "episodes" / phase / condition / f"{seed}.npz" for phase, seed in (("validation", 510000), ("confirmation", 520000)) for condition in ("baseline", "lateral__-0.1000")]  # Local: fingerprint the audit's scientific inputs; global: prove the checks leave them unchanged.
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}  # Local: snapshot actual input bytes; global: bind output evidence to immutable source artifacts.
    output = {"created_utc": utc_now(), "scope": "Publication replay of the original bounded inline runtime audit; no new experimental seeds or selection", "run": str(run.relative_to(root)), "identity": identity, "vector_shape": list(vector.shape), "vector_norm": float(np.linalg.norm(vector)), "checkpoint_sha256": metadata["sha256"], "source_files": hashes, "audit_script_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(), "actions": verify_actions(model, run, vector), "physics": verify_physics(run), "official_sources": verify_sources(), "versions": {name: importlib.metadata.version(name) for name in ("gymnasium", "mujoco", "stable-baselines3", "sb3-contrib", "numpy", "torch", "baukit")}}  # Local: compose preserved checks; global: publish one complete machine-readable audit.
    assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())  # Local: recheck every watched input; global: verify the audit changed no scientific evidence.
    output["scientific_inputs_unchanged"] = True  # Local: record the passed immutability check; global: distinguish audit outputs from experiment updates.
    save_json(folder / "runtime-audit.json", output)  # Local: publish compact strict JSON atomically; global: make all exact checks available beside the prose report.
    logging.info("Ant005 runtime audit complete\n%s", json.dumps(output, indent=2, allow_nan=False))  # Local: emit complete findings without truncation; global: preserve realtime terminal and timestamped file evidence.


if __name__ == "__main__":
    main()  # Local: run only when explicitly invoked; global: keep imports free of audit side effects.

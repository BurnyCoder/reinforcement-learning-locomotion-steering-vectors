"""Local: render existing Ant validation actions from above; global: preserve numerical evidence.

Run: uv run --no-sync python scripts/render_topdown.py --run-dir runs/ant-classic-005 --vector lateral --alpha -0.1 --seed 510000
Sources: https://mujoco.readthedocs.io/en/3.1.5/python.html#rendering
https://github.com/google-deepmind/mujoco/blob/3.1.5/src/render/render_gl3.c#L296-L312
https://imageio.readthedocs.io/en/stable/examples.html
https://mujoco.readthedocs.io/en/3.1.5/APIreference/APIfunctions.html#mjv-connector
This artifact helper changes the render scene only; it is not a new policy experiment.
"""
import argparse  # Local: expose saved-run inputs explicitly; global: make published media reproducible.
import hashlib  # Local: identify exported media; global: make the verified result traceable.
import logging  # Local: report completed replays; global: preserve timestamped terminal/file progress.
from contextlib import ExitStack  # Local: close encoder, renderer and simulator; global: handle failures without resource leaks.
from pathlib import Path  # Local: anchor artifacts beside this helper; global: stay within the project.

import gymnasium as gym  # Local: reuse the original Ant dynamics; global: avoid custom simulation behavior.
import imageio.v2 as imageio  # Local: encode one frame at a time; global: bound memory regardless of duration.
import mujoco  # Local: use the public renderer and camera API; global: isolate appearance from physical state.
import numpy as np  # Local: compare complete numerical arrays; global: reject replay drift.
from PIL import Image, ImageDraw, ImageFont  # Local: add factual labels; global: preserve original simulator imagery.

from rl_locomotion_steering_vectors.demonstration import _paired_video  # Local: reuse tested streaming composition; global: preserve paired timing and early endings.
from rl_locomotion_steering_vectors.experiment import condition_name  # Local: use the actual saved cache label; global: do not guess intervention directory names.
from rl_locomotion_steering_vectors.runtime import state_hash, summarise_episode  # Local: audit simulator identity; global: share the existing scientific measurements.
from rl_locomotion_steering_vectors.storage import load_arrays, load_json, save_json, start_logging, utc_now  # Local: reuse atomic artifact handling; global: retain provenance and errors.


def render_condition(run: Path, folder: Path, seed: int, condition: str, label: str, fixed: dict | None = None) -> dict:
    """Local: replay every saved action and verify its outcome; global: show the measured trajectory faithfully."""
    expected = load_arrays(run / "episodes" / "validation" / condition / f"{seed}.npz")  # Local: read existing evidence only; global: consume no fresh reset seeds.
    identity = load_json(run / "episodes" / "validation" / condition / "identity.json")  # Local: restore the actual cached onset; global: avoid current configuration affecting a replay.
    warmup, dt = int(identity["warmup"]), float(expected["dt"])  # Local: retain the recorded timing; global: preserve the original protocol.
    video = folder / f"seed-{seed}-{label}-topdown.mp4"  # Local: distinguish both conditions; global: avoid overwriting the preliminary side view.
    output = {name: [] for name in ("observations", "actions", "reward")}  # Local: reconstruct full requested arrays; global: verify all transitions rather than sampled values.
    font = ImageFont.load_default(size=14)  # Local: use a portable bundled font; global: keep labels reproducible.
    with ExitStack() as stack:  # Local: scope all resources; global: shut them down after exceptions too.
        env = stack.enter_context(gym.make("Ant-v5"))  # Local: retain default physics and episode semantics; global: change only the camera and scene.
        physics = env.unwrapped  # Local: access original MuJoCo state; global: compare full integration-state hashes.
        obs, _ = env.reset(seed=seed)  # Local: reproduce the stored reset; global: preserve initial conditions.
        floor = int(physics.model.geom("floor").id)  # Local: resolve the actual named ground; global: avoid geometry-index assumptions.
        sizes = physics.model.geom_size.copy()  # Local: retain a model oracle; global: prove the scene edit leaves collision geometry unchanged.
        repeats = physics.model.mat_texrepeat.copy()  # Local: retain material settings; global: keep model texture state unchanged too.
        material = int(physics.model.geom_matid[floor])  # Local: find the floor material; global: preserve the original checker size.
        repeat = repeats[material] / sizes[floor, :2]  # Local: convert finite-plane repeats to world units; global: extend the same visible pattern.
        renderer = stack.enter_context(mujoco.Renderer(physics.model, height=480, width=480))  # Local: own a public offscreen renderer; global: avoid monkeypatching Gymnasium or project runtime.
        camera = mujoco.MjvCamera()  # Local: create a dedicated visual camera; global: leave model cameras and dynamics unchanged.
        camera.type, camera.azimuth, camera.elevation, camera.distance = mujoco.mjtCamera.mjCAMERA_FREE, 90, -90, 8  # Local: fix world orientation and top-down elevation; global: make lateral motion visually legible.
        if fixed is not None:  # Local: use one frame computed from both saved paths; global: make distant displacement comparisons share an identical camera.
            camera.lookat[:] = [*(np.asarray(fixed["center_xy_relative"]) + physics.data.qpos[:2]), 0]  # Local: anchor the common center in the actual reset's world coordinates; global: keep the camera fixed throughout the movie.
            camera.distance = fixed["span_with_margin"] / (2 * np.tan(np.deg2rad(physics.model.vis.global_.fovy / 2)))  # Local: fit the whole shared square within the actual camera field of view; global: avoid cropping either trajectory.
        camera_spec = dict(lookat=camera.lookat.copy().tolist(), distance=float(camera.distance), azimuth=90, elevation=-90)  # Local: retain exact visual framing; global: compare both panels and verify fixed-camera invariance.
        trail = []  # Local: retain only the physical positions needed for annotations; global: never substitute markers for simulation dynamics.
        writer = stack.enter_context(imageio.get_writer(str(video), fps=round(1 / dt), codec="libx264", macro_block_size=1))  # Local: stream at the recorded timestep; global: preserve physical elapsed time.

        def frame(index: int) -> None:
            """Local: draw and encode the current top-down scene; global: preserve physical state exactly."""
            before = state_hash(physics)  # Local: snapshot integration data; global: prove rendering itself causes no state transition.
            if fixed is None:  # Local: retain close tracking as the default mode; global: keep previous top-down output reproducible.
                camera.lookat[:] = physics.data.body("torso").xpos  # Local: track position without rotating; global: keep the body visible and world orientation fixed.
            else:  # Local: enforce the distant-camera contract; global: never follow or recenter either condition.
                assert camera.lookat.tolist() == camera_spec["lookat"] and camera.distance == camera_spec["distance"]  # Local: verify fixed position and scale each frame; global: prevent misleading displacement comparisons.
            renderer.update_scene(physics.data, camera=camera)  # Local: obtain current render geoms; global: preserve simulator data and model geometry.
            ground = next(geom for geom in renderer.scene.geoms[:renderer.scene.ngeom] if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == floor)  # Local: identify only the scene's ground copy; global: avoid changing any physical geom.
            ground.dataid = -1  # Local: select MuJoCo's generic plane display list; global: permit visual resizing without rebuilding the model context.
            ground.size[:2], ground.texuniform, ground.texrepeat[:] = 1000, 1, repeat  # Local: expand the render rectangle and keep checker scale; global: make ground visible throughout these trajectories.
            if fixed is not None:  # Local: make distant travel visible with explicit scene annotations; global: preserve actual robot rendering and every physical field.
                ground.texrepeat[:], ground.reflectance = .05, 0  # Local: use 20-metre checker squares without reflections; global: avoid subpixel checker aliasing in the far view.
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0  # Local: disable decorative distance fog; global: keep a 300-metre trajectory visible from the necessary altitude.
                for eye in renderer.scene.camera:  # Local: update both stereo frusta identically; global: keep the public scene camera independent of model clipping parameters.
                    eye.frustum_far = max(eye.frustum_far, camera.distance * 4)  # Local: extend only the visual clipping plane; global: avoid clipping the distant floor and robot.
                position = physics.data.qpos[:3].copy()  # Local: read the actual body position; global: anchor annotations to simulator coordinates.
                position[2] = .03  # Local: draw trails just above the floor; global: label them as schematic position markers rather than robot geometry.
                trail.append(position)  # Local: accumulate one true position per frame; global: display only already-visited trajectory segments.
                color = [.15, .55, 1, 1] if label == "baseline" else [1, .25, .1, 1]  # Local: distinguish the two annotation colors; global: preserve consistent baseline/treatment identities.

                def line(start, end) -> None:
                    """Local: append a decorative line; global: keep annotations in the visual scene only."""
                    geom = renderer.scene.geoms[renderer.scene.ngeom]  # Local: use the next free render slot; global: never add a physical collision geom.
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))  # Local: initialize one colored decoration; global: reuse MuJoCo's documented primitive.
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.5, start, end)  # Local: connect world-space endpoints with a pixel-width line; global: keep annotations readable at distance.
                    renderer.scene.ngeom += 1  # Local: expose the completed decoration to rendering; global: leave model and simulator geom counts unchanged.

                points = trail[::5]  # Local: sample the existing trail every quarter-second; global: bound decorative geometry without changing measured data.
                for start, end in zip(points[:-1], points[1:]):  # Local: draw the visited path in temporal order; global: prevent showing future motion before it occurs.
                    line(start, end)  # Local: render a segment; global: make the distant body's displacement visible.
                angles = np.linspace(0, 2 * np.pi, 21)  # Local: define a closed ring; global: mark current position without covering the native robot at its center.
                ring = position + np.c_[3 * np.cos(angles), 3 * np.sin(angles), np.zeros(len(angles))]  # Local: place a clearly enlarged three-metre-radius marker around the current location; global: distinguish the annotation from the actual body size.
                for start, end in zip(ring[:-1], ring[1:]):  # Local: draw the ring perimeter; global: keep the actual robot visible inside it.
                    line(start, end)  # Local: append one marker edge; global: reuse the same scene-only primitive.
            pixels = renderer.render()  # Local: render original simulated bodies from above; global: avoid generative or reconstructed motion imagery.
            assert state_hash(physics) == before  # Local: compare every rendered state; global: reject any numerical side effect.
            header = 72 if fixed is not None else 48  # Local: reserve an extra legend line for distant annotations; global: make their meaning explicit.
            canvas = Image.new("RGB", (480, 480 + header), "black")  # Local: allocate one labeled frame; global: keep memory bounded.
            canvas.paste(Image.fromarray(pixels), (0, header))  # Local: preserve the complete original frame; global: distinguish simulation from its captions.
            draw = ImageDraw.Draw(canvas)  # Local: draw only labels; global: make exported context self-contained.
            view = "FIXED CAMERA" if fixed is not None else "TOP DOWN"  # Local: identify the actual camera mode; global: distinguish fixed framing from tracking.
            draw.text((6, 4), f"PRELIMINARY VALIDATION | {view} | {label.upper()}", font=font, fill="#ffd56b")  # Local: state the evidence stage; global: avoid implying a completed confirmation.
            begin = max(0, index - round(1 / dt))  # Local: select up to one second of existing numerical data; global: keep velocity labels stable through gait oscillations.
            vx, vy = [float(expected[name][begin:index].mean()) if index else 0.0 for name in ("x_velocity", "y_velocity")]  # Local: show recorded trailing means; global: do not infer speed from pixels.
            draw.text((6, 25), f"t={index * dt:05.2f}s | 1s means: vx={vx:+.2f}, vy={vy:+.2f} m/s", font=font, fill="white")  # Local: label time and both velocity components; global: quantify forward retention and lateral change.
            if fixed is not None:  # Local: explain the distant annotations; global: avoid presenting enlarged markers as physical robot size.
                draw.text((6, 48), "20m grid | +X right, +Y up | ring/trail = annotations", font=font, fill="white")  # Local: state verified camera axes and world scale; global: make displacement legible and quantitative.
            writer.append_data(np.asarray(canvas))  # Local: encode this frame immediately; global: avoid storing the full movie in RAM.
            if index in (100, 200, 900):  # Local: retain onset and later examples; global: permit independent visual inspection.
                canvas.save(folder / f"seed-{seed}-{label}-step-{index}.png")  # Local: save original labeled samples; global: show the same frames as the movie.

        onset = ""  # Local: retain the original empty onset for pre-onset failures; global: do not fabricate a reached intervention boundary.
        frame(0)  # Local: include the reset pose; global: align movie time zero with the saved simulator reset.
        for step, recorded_action in enumerate(expected["actions"]):  # Local: apply every existing control in order; global: do not sample new actions or alter the intervention.
            if step == warmup:  # Local: locate the exact treatment onset; global: preserve causal pairing.
                onset = state_hash(physics)  # Local: record before the first treatment action; global: compare with the original validation hash.
            output["observations"].append(obs.copy())  # Local: retain each actual action input; global: verify the complete observation sequence.
            action = recorded_action.copy()  # Local: keep source artifacts immutable; global: replay exactly the already-measured controls.
            output["actions"].append(action.copy())  # Local: record every applied action; global: make action correspondence explicit.
            obs, reward, terminated, truncated, _ = env.step(action)  # Local: use the original environment transition; global: preserve reward and ending semantics.
            assert np.array_equal(physics.data.ctrl.astype(action.dtype), action)  # Local: check the actual simulator controls; global: verify action delivery rather than only input copying.
            output["reward"].append(reward)  # Local: retain actual rewards; global: detect hidden changes to dynamics or reward computation.
            frame(step + 1)  # Local: depict the resulting state; global: align frame indices and physical timesteps.
            if terminated or truncated:  # Local: obey the environment ending; global: never hide an early failure with an automatic reset.
                break  # Local: stop exactly at the recorded boundary; global: preserve the episode denominator.
        output = {name: np.asarray(values) for name, values in output.items()}  # Local: materialize the checked arrays; global: support byte-level equality.
        output.update(seed=np.array(seed), onset_hash=np.array(onset), terminated=np.array(terminated), truncated=np.array(truncated), length=np.array(len(output["reward"])), dt=np.array(float(physics.dt)))  # Local: reconstruct episode metadata; global: verify ending flags, horizon, onset and timing.
        for name, actual in output.items():  # Local: compare every timestep and scalar field; global: reject tolerance-masked drift.
            assert actual.shape == expected[name].shape and actual.dtype == expected[name].dtype and actual.tobytes() == expected[name].tobytes(), name  # Local: require identical arrays; global: certify numerical-video correspondence.
        assert np.array_equal(physics.model.geom_size, sizes) and np.array_equal(physics.model.mat_texrepeat, repeats)  # Local: recheck model fields after all rendering; global: demonstrate scene-only visual changes.
    logging.info("Verified top-down replay seed=%s condition=%s video=%s", seed, condition, video)  # Local: report success only after every equality check; global: preserve actionable progress logs.
    return dict(seed=seed, condition=condition, video=str(video), camera=camera_spec, fixed_camera=fixed is not None, arrays_verified=sorted(output), all_verified_arrays_bitwise_equal=True, controls_equal_applied=True, model_geometry_and_material_unchanged=True, render_preserves_state=True, metrics=summarise_episode(expected, warmup), sha256=hashlib.sha256(video.read_bytes()).hexdigest())  # Local: expose complete audit evidence; global: make the result independently inspectable.


def main() -> None:
    """Local: render a specified existing validation pair; global: provide a reproducible top-down comparison."""
    parser = argparse.ArgumentParser(description="Render existing Ant validation trajectories from above; no new policy experiments")  # Local: explain the exact artifact workflow; global: avoid implying fresh evidence.
    parser.add_argument("--run-dir", type=Path, required=True)  # Local: identify the saved run; global: keep checkpoint and intervention provenance together.
    parser.add_argument("--vector", required=True)  # Local: identify the saved vector key; global: prevent unlabeled treatments.
    parser.add_argument("--alpha", type=float, required=True)  # Local: make strength explicit; global: do not silently choose a favorable setting.
    parser.add_argument("--seed", type=int, required=True)  # Local: make the recorded reset explicit; global: consume only the chosen existing evidence.
    parser.add_argument("--camera", choices=("tracking", "fixed"), default="tracking")  # Local: preserve tracking by default; global: offer the explicitly requested fixed distant comparison.
    args = parser.parse_args()  # Local: use standard CLI validation; global: provide reproducible help and invocation errors.
    run = args.run_dir.resolve()  # Local: canonicalize the selected run; global: keep paths unambiguous.
    if not run.is_relative_to(Path.cwd().resolve()) or load_json(run / "manifest.json")["model_key"] != "ant":  # Local: constrain the verified renderer to project-local Ant artifacts; global: avoid unsupported environment claims.
        raise ValueError("Require an existing Ant run within the current project")  # Local: explain the supported scope; global: fail before opening a simulator.
    condition = condition_name(args.vector, args.alpha)  # Local: resolve the exact saved condition; global: preserve actual strength precision checks from the run.
    identity = load_json(run / "episodes" / "validation" / condition / "identity.json")  # Local: read the original treatment identity; global: verify the requested vector and strength before rendering.
    vector = load_arrays(run / "vectors.npz")[args.vector]  # Local: inspect the named saved direction; global: retain its identity without applying new steering.
    if identity["alpha"] != args.alpha or identity["vector_sha256"] != hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest():  # Local: match exact cached inputs; global: reject stale or mislabeled vector artifacts.
        raise ValueError("Requested vector or strength differs from cached validation identity")  # Local: expose the mismatch; global: prevent false video provenance.
    mode = "fixed-camera" if args.camera == "fixed" else "topdown"  # Local: name the selected visual mode; global: preserve both outputs separately.
    folder = run / "videos" / mode / condition  # Local: separate distinct treatments; global: avoid overwriting videos from other signed strengths.
    fixed = None  # Local: tracking needs no shared bounds; global: retain the existing default behavior.
    if args.camera == "fixed":  # Local: precompute one common view from both existing trajectories; global: make neither panel's framing depend on its own path alone.
        episodes = [load_arrays(run / "episodes" / "validation" / selected / f"{args.seed}.npz") for selected in ("baseline", condition)]  # Local: read saved data only; global: avoid consuming additional seeds or policy observations.
        paths = [np.vstack((np.zeros((1, 2)), np.cumsum(np.c_[episode["x_velocity"], episode["y_velocity"]], axis=0) * float(episode["dt"]))) for episode in episodes]  # Local: integrate recorded displacement rates from reset; global: include both full paths before choosing fixed framing.
        low, high = np.min(np.vstack(paths), axis=0), np.max(np.vstack(paths), axis=0)  # Local: find shared world-relative bounds; global: retain all baseline and treatment travel.
        fixed = dict(center_xy_relative=((low + high) / 2).tolist(), span_with_margin=float(max(high - low) * 1.25), bounds_relative=[low.tolist(), high.tolist()])  # Local: add a fixed 25% margin; global: keep start/end markers inside the same camera view.
    start_logging(folder)  # Local: log terminal and file output; global: preserve exact errors if verification fails.
    logging.info("command run=%s vector=%s alpha=%s seed=%s camera=%s framing=%s", run, args.vector, args.alpha, args.seed, args.camera, fixed)  # Local: log all explicit inputs; global: make the run reproducible from its log.
    records = [render_condition(run, folder, args.seed, selected, label, fixed) for selected, label in (("baseline", "baseline"), (condition, "steered"))]  # Local: replay only the requested seed; global: avoid selecting another example from its outcome.
    if fixed is not None:  # Local: audit both panels' framing; global: prohibit separate camera scales from exaggerating displacement.
        assert records[0]["camera"] == records[1]["camera"]  # Local: compare actual lookat, distance and orientation; global: enforce the common fixed camera specification.
    paired = folder / f"seed-{args.seed}-preliminary-{mode}-paired.mp4"  # Local: name the final comparison clearly; global: keep it distinguishable from confirmed evidence.
    _paired_video(Path(records[0]["video"]), Path(records[1]["video"]), paired, args.seed, args.alpha)  # Local: reuse labeled paired composition; global: synchronize both original trajectories.
    save_json(folder / f"seed-{args.seed}-audit.json", dict(created_utc=utc_now(), status="preliminary validation replay", scene_only=True, camera_mode=args.camera, common_fixed_framing=fixed, annotation_legend="In fixed mode: colored trail and 3m-radius ring are visual position annotations; original robot geometry is also rendered. Checker squares span 20m.", records=records, paired_video=str(paired), paired_sha256=hashlib.sha256(paired.read_bytes()).hexdigest(), source_identity=identity, helper_sha256=hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode("utf-8")).hexdigest()))  # Local: persist numerical and visual provenance; global: prevent an ambiguous unsupported video claim.
    logging.info("Top-down pair complete: %s", paired)  # Local: expose the exact finished path; global: make the requested artifact easy to open.


if __name__ == "__main__":  # Local: execute only when called as a script; global: allow safe inspection without starting a replay.
    main()  # Local: run the reproducible artifact workflow; global: preserve its explicit scope.

"""Independent saved-data checks: analytical fixtures, with no policy or simulator.

Sources: https://numpy.org/doc/1.26/reference/routines.testing.html and
https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats.bootstrap.html.
"""
import subprocess
import sys

import numpy as np
import pytest

from rl_locomotion_steering_vectors.factcheck.numerical import episode_summary, paired, contrast, fitting


def episode(seed=1, length=6, speed=2.0):
    """Make arrays with known outcomes, not outputs from a synthetic neural model."""
    item = {name: np.zeros(length) for name in (
        "x_velocity", "y_velocity", "height", "yaw_rate", "effort", "reward",
        "planar_speed", "saturation", "inversion", "bad_contact")}
    item.update(seed=seed, length=length, terminated=False, truncated=True, dt=.05, onset_hash=str(seed))
    item["x_velocity"][:] = speed
    item["planar_speed"][:] = abs(speed)
    return item


def test_reference_preserves_prefix_failures_and_episode_weighting():
    early = episode(length=2, speed=100)
    full = episode(seed=2, speed=3)
    assert episode_summary(early, 2)["failure"] == 1
    assert episode_summary(early, 2)["speed"] == 0
    assert episode_summary(full, 2)["speed"] == 3
    changed = [episode(length=2, speed=100), episode(seed=2, speed=4)]
    result = paired([early, full], changed, "speed", warmup=2, seed=7)
    assert result["effect"] == .5
    assert not result["quality_pass"]
    assert result["baseline_means"]["speed"] == 1.5


def test_reference_uses_whole_seed_pairs_and_rejects_missing_or_duplicate_pairs():
    baseline = [episode(seed=i) for i in range(3)]
    treated = [episode(seed=i, length=4+i, speed=2.5) for i in reversed(range(3))]
    result = paired(baseline, treated, "speed", warmup=2, seed=17)
    assert result["ci95"] == [.5, .5]
    assert result["gate"]
    with pytest.raises(ValueError, match="seed"):
        paired(baseline, treated[:2], "speed", warmup=2, seed=17)
    with pytest.raises(ValueError, match="seed"):
        paired(baseline, [treated[0]] * 3, "speed", warmup=2, seed=17)
    treated[0]["onset_hash"] = "different"
    with pytest.raises(ValueError, match="onset"):
        paired(baseline, treated, "speed", warmup=2, seed=17)


def test_failure_threshold_is_strict_and_bad_arrays_are_rejected():
    item = episode(length=22)
    item["bad_contact"][2] = 1
    assert episode_summary(item, 2)["failure"] == 0
    item["bad_contact"][3] = 1
    assert episode_summary(item, 2)["failure"] == 1
    item["height"][4] = np.nan
    with pytest.raises(ValueError, match="height"):
        episode_summary(item, 2)


def test_contrast_balances_episodes_and_fitting_preserves_within_window_variance():
    records = []
    for seed in range(16):
        item = episode(seed, length=8)
        item["x_velocity"] = np.array([0, 0, 0, 0, 2, 2, 2, 2], dtype=float)
        item["activations"] = np.column_stack([item["x_velocity"], np.tile([-3, 3], 4)])
        records.append(item)
    table, stats, scale = fitting(records, warmup=0, window=2, fraction=0)
    raw, details = contrast(table, "speed")
    np.testing.assert_allclose(raw, [2, 0])
    assert scale == pytest.approx(np.sqrt(10))
    assert details["high_episodes"] == details["low_episodes"] == 16
    assert stats["eligible_windows"] == 64


def test_audit_modules_do_not_import_production_estimators_or_models():
    code = ("import sys; from rl_locomotion_steering_vectors.factcheck import numerical, audit, provenance, record, display; "
            "assert not any(x in sys.modules for x in ('torch','mujoco','gymnasium',"
            "'stable_baselines3','rl_locomotion_steering_vectors.analysis'))")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_numeric_evidence_rejects_corrupted_measurements_and_missing_fields():
    from rl_locomotion_steering_vectors.factcheck.audit import check_measurements, differences
    item = episode()
    item['actions'] = np.zeros((6, 2), dtype=np.float32)
    assert check_measurements([item])['effort_max_abs_error'] == 0
    item['effort'][0] = .01
    with pytest.raises(ValueError, match='effort'):
        check_measurements([item])
    item['actions'][0,0] = np.nan
    with pytest.raises(ValueError, match='actions'):
        check_measurements([item])
    assert differences({'gate': False}, {}, require_fields=True)
    assert differences([None,0.,2.], [None,0.,2.]) == []
    assert [x['field'] for x in differences([None,0.,2.], [0.,None,3.], 'values')] == ['values/0','values/1','values/2']


def test_display_trace_retains_prefix_only_failure_and_partial_reward_block():
    from rl_locomotion_steering_vectors.factcheck.display import trace
    short = episode(length=14, speed=2.)
    short['reward'][:] = np.arange(14)
    value = trace(short)
    assert value['failed'] and value['prefix_only']
    assert value['steps'] == [0,10,14]
    assert value['x'] == [0.,1.,1.4]
    assert value['reward'] == [None,4.5,11.5]
    assert value['summary']['reward'] == 0
    assert value['summary']['total_return'] == 91


def test_reference_kinematics_uses_the_common_horizon_and_correct_frame():
    from rl_locomotion_steering_vectors.factcheck.secondary import pair_geometry, crossed_interval
    left, right = episode(1, 500), episode(1, 400)
    for item in (left, right):
        item['observations'] = np.zeros((item['length'], 5))
        item['observations'][:, 1] = 1
    right['x_velocity'][:] = 0
    right['y_velocity'][:] = 2
    right['observations'][:, 1] = np.cos(np.pi/4)
    right['observations'][:, 4] = np.sin(np.pi/4)
    value = pair_geometry(left, right)
    assert value['common_steps'] == 300
    assert value['delta_travel_angle_deg'] == pytest.approx(90)
    assert value['delta_mean_yaw_deg'] == pytest.approx(90)
    assert value['delta_gait_lateral_velocity'] == pytest.approx(0, abs=1e-12)
    assert crossed_interval(np.full((4, 3), 7), resamples=50)['ci'] == [7., 7.]


def test_fixed_publication_date_is_deterministic_and_preserves_inputs(tmp_path):
    from pathlib import Path
    from reportlab import rl_config
    from rl_locomotion_steering_vectors.reporting import write_report
    rl_config.invariant = 1
    source = Path(__file__).resolve().parents[1]/'reports/hc-classic-001'
    saved = {name:(source/name).read_bytes() for name in ('manifest.json','results.json')}
    for name,data in saved.items(): (tmp_path/name).write_bytes(data)
    write_report(tmp_path,timestamp='2026-10-29',notice='Saved-data erratum')
    before = [(tmp_path/name).read_bytes() for name in ('report.md','report.pdf')]
    assert b'\r\n' not in before[0]
    write_report(tmp_path,timestamp='2026-10-29',notice='Saved-data erratum')
    assert before == [(tmp_path/name).read_bytes() for name in ('report.md','report.pdf')]
    assert saved == {name:(tmp_path/name).read_bytes() for name in saved}


def test_temporary_experiment_build_preserves_evidence_links_and_exact_outputs(tmp_path):
    from pathlib import Path
    from scripts.build_publication import EXPERIMENTS, render_experiments
    root = Path(__file__).resolve().parents[1]
    originals = {(name, file):(root/'reports'/name/file).read_bytes()
                 for name in EXPERIMENTS for file in ('manifest.json','results.json','vector_diagnostics.json')}
    outputs = render_experiments(root, tmp_path)
    for path in outputs:
        assert path.read_bytes() == (root/path.relative_to(tmp_path)).read_bytes(), str(path)
    assert all(data == (root/'reports'/name/file).read_bytes() for (name,file),data in originals.items())

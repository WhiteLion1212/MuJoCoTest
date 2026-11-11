import numpy as np
import pytest

from mm3dgs_bench import Config, SLAMEnvironment
from mm3dgs_bench.effects import environmental_effect
from mm3dgs_bench.evaluation import evaluate


@pytest.mark.parametrize('scenario', ['motion_blur', 'dust', 'rain'])
def test_partial_impairments_preserve_signal_and_recover(scenario):
    config = dict(width=160, height=120)
    with SLAMEnvironment(Config(scenario='rich', **config)) as clean, \
            SLAMEnvironment(Config(scenario=scenario, **config)) as impaired:
        for time_s in (4., 9., 13.):
            clean._set_state(time_s)
            impaired._set_state(time_s)
            a, b = clean._camera(time_s), impaired._camera(time_s)
            if time_s == 9.:
                assert np.mean(abs(a['rgb'].astype(float)-b['rgb'])) > 1.
                assert b['rgb'].std() > 8.
                assert .1 < np.mean(b['depth'] > 0) < 1.
                # Exposure rendering must restore the actual sensor state.
                np.testing.assert_allclose(clean.data.qpos, impaired.data.qpos)
                np.testing.assert_allclose(clean.data.sensor('accel').data, impaired.data.sensor('accel').data)
            else:
                np.testing.assert_array_equal(a['rgb'], b['rgb'])
                np.testing.assert_array_equal(a['depth'], b['depth'])


def test_motion_exposure_is_causal_and_uses_multiple_real_poses():
    with SLAMEnvironment(Config(scenario='motion_blur', width=80, height=60)) as env:
        times = []
        original = env._set_state
        def record(t):
            times.append(t)
            original(t)
        env._set_state = record
        env._set_state(9.)
        env._camera(9.)
        assert len(set(times)) == env.cfg.blur_samples
        assert min(times) == pytest.approx(9-env.cfg.blur_exposure_s)
        assert max(times) == 9.
        assert times[-1] == 9.


@pytest.mark.parametrize('scenario', ['motion_blur', 'dust', 'rain'])
def test_new_scenarios_have_identical_noisy_imu_and_deterministic_reset(scenario):
    with SLAMEnvironment(Config(scenario='rich'), render=False) as a, \
            SLAMEnvironment(Config(scenario=scenario), render=False) as b:
        for _ in range(121):
            x, y = a.next_tick(), b.next_tick()
            if 'imu' in x:
                np.testing.assert_array_equal(x['imu']['gyro'], y['imu']['gyro'])
                np.testing.assert_array_equal(x['imu']['accel'], y['imu']['accel'])
    with SLAMEnvironment(Config(scenario=scenario, width=80, height=60)) as env:
        env._set_state(9.)
        first = env._camera(9.)
        env.reset()
        env._set_state(9.)
        again = env._camera(9.)
        np.testing.assert_array_equal(first['rgb'], again['rgb'])
        np.testing.assert_array_equal(first['depth'], again['depth'])


def test_environment_effects_are_deterministic_and_dust_is_depth_dependent():
    rgb = np.full((80, 120, 3), 40, np.uint8)
    depth = np.full((80, 120), 2., np.float32)
    for effect in ('dust', 'rain'):
        a, ma = environmental_effect(rgb, depth, effect, 1., 9., 7)
        b, mb = environmental_effect(rgb, depth, effect, 1., 9., 7)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ma, mb)
    near, _ = environmental_effect(rgb, depth, 'dust', 1., 9., 7)
    far, _ = environmental_effect(rgb, depth*4, 'dust', 1., 9., 7)
    assert far.mean() > near.mean()


def test_prediction_scoring_does_not_hide_error_with_its_own_alignment(tmp_path):
    rows = np.array([[0., 0, 0, 0, 0, 0, 0, 1], [1., 1, 0, 0, 0, 0, 0, 1]])
    truth = tmp_path/'truth.tum'
    estimate = tmp_path/'prediction.tum'
    np.savetxt(truth, rows)
    rows[:, 1] += 1.
    np.savetxt(estimate, rows)
    assert evaluate(truth, estimate)['ate_rmse_m'] < 1e-12
    report = evaluate(truth, estimate, alignment=(np.eye(3), np.zeros(3)))
    assert report['ate_rmse_m'] == pytest.approx(1.)


def test_exposure_and_impairment_parameters_reject_invalid_values():
    for kwargs in ({'blur_exposure_s': .1}, {'blur_samples': 1}, {'blur_gain': 0.},
                   {'blur_depth_keep': 0.}, {'blackout_end': 23.}):
        with pytest.raises(ValueError):
            Config(scenario='motion_blur', **kwargs)


def test_counteraction_requires_matched_clean_control_and_visual_continuity():
    from copy import deepcopy
    from mm3dgs_bench.comparison import controlled_summary
    def report(scenario, visual_error, fused_error):
        variants={}
        for mode,value in [('visual',visual_error),('imu_fused',fused_error)]:
            variants[mode]={'prediction_increment':{'during_impairment':{'translation_rmse_m':value}},
                            'tracking':{'during_outage':{'ate_rmse_m':value}}}
        return {'scenario':scenario,'seed':7,'matched_measurement_id':'same-sensors',
                'imu_benefit_validated':True,'variants':variants}
    healthy=report('rich',.01,.003)
    impaired=report('rain',.02,.005)
    assert controlled_summary([impaired])['matched_clean_controls']==[]
    paired=controlled_summary([healthy,impaired])['matched_clean_controls'][0]
    assert paired['camera_impairment_counteracted']
    assert paired['camera_induced_prediction_penalty_reduction']==pytest.approx(.8)
    mismatched=deepcopy(healthy)
    mismatched['matched_measurement_id']='different-sensors'
    assert controlled_summary([mismatched,impaired])['matched_clean_controls']==[]
    impaired['imu_benefit_validated']=False  # e.g. the tracker stopped visual updates.
    assert not controlled_summary([healthy,impaired])['matched_clean_controls'][0]['camera_impairment_counteracted']

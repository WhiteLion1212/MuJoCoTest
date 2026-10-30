import numpy as np
import pytest
from mm3dgs_bench import Config,SLAMEnvironment
from mm3dgs_bench.trajectory import state
from mm3dgs_bench.scene import R_B_C,P_B_C


@pytest.mark.parametrize('t',[0.,1.,2.5,7.1,9.9,15.7,23.])
def test_real_mujoco_imu_matches_rigid_body_kinematics(t):
    with SLAMEnvironment(render=False) as e:
        e._set_state(t)
        p,r,v,a,w,alpha=state(t,e.cfg.duration)
        np.testing.assert_allclose(e.data.sensor('accel').data,r.inv().apply(a-np.array([0,0,-9.81])),atol=1e-10)
        np.testing.assert_allclose(e.data.sensor('gyro').data,w,atol=1e-10)
        np.testing.assert_allclose(e.data.qacc,np.r_[a,alpha],atol=1e-9)


def test_trajectory_derivatives_and_body_angular_velocity():
    h=1e-5
    for t in [3.,8.,10.,14.,20.]:
        p,r,v,a,w,_=state(t,24)
        pm,rm,vm,*_=state(t-h,24); pp,rp,vp,*_=state(t+h,24)
        np.testing.assert_allclose((pp-pm)/(2*h),v,atol=1e-8)
        np.testing.assert_allclose((vp-vm)/(2*h),a,atol=1e-8)
        np.testing.assert_allclose((rm.inv()*rp).as_rotvec()/(2*h),w,atol=1e-5)


def test_stationary_endpoints_and_extrinsics():
    with SLAMEnvironment(render=False) as e:
        for t in [0,1,23,24]:
            e.tick=round(t*e.cfg.physics_hz); e._set_state(t)
            gt=e.ground_truth()
            np.testing.assert_allclose(gt['velocity_W'],0,atol=1e-10)
            np.testing.assert_allclose(gt['T_W_C'][:3,3],gt['T_W_B'][:3,3]+gt['T_W_B'][:3,:3]@P_B_C)
            np.testing.assert_allclose(gt['T_W_C'][:3,:3],gt['T_W_B'][:3,:3]@R_B_C)


def test_sensor_noise_determinism_and_reset():
    with SLAMEnvironment(render=False) as a,SLAMEnvironment(render=False) as b:
        first=a.next_tick()['imu']
        a.reset(); b.reset()
        for _ in range(100):
            x,y=a.next_tick(),b.next_tick()
            if 'imu' in x:
                np.testing.assert_array_equal(x['imu']['accel'],y['imu']['accel'])
                np.testing.assert_array_equal(x['imu']['gyro'],y['imu']['gyro'])
        a.reset(); np.testing.assert_array_equal(a.next_tick()['imu']['accel'],first['accel'])


@pytest.mark.parametrize('kwargs',[{'imu_hz':99},{'duration':3},{'blackout_end':23},{'depth_noise':-1},{'width':12}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError): Config(**kwargs)


def test_public_calibration_hides_planted_bias_and_generation_seed():
    with SLAMEnvironment(render=False) as e:
        cal=e.calibration()
        assert 'config' not in cal
        assert set(cal['imu']['noise'])=={'accel_noise_density','gyro_noise_density','accel_bias_rw','gyro_bias_rw'}
        assert e.calibration(include_generation=True)['config']['seed']==7

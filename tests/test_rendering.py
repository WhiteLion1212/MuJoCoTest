import mujoco
import numpy as np
from mm3dgs_bench import Config,SLAMEnvironment


def test_depth_matches_geometric_ray_and_camera_calibration():
    with SLAMEnvironment(Config(width=160,height=120,depth_noise=0)) as e:
        e._set_state(4.)
        frame=e._camera(4.)
        assert frame['rgb'].shape==(120,160,3)
        assert frame['depth'].dtype==np.float32
        e.tick=4*e.cfg.physics_hz
        T=e.ground_truth()['T_W_C']; K=np.array(e.calibration()['camera']['K'])
        for u,v in [(80,60),(30,80),(125,85)]:
            ray=np.linalg.inv(K)@np.array([u,v,1.])
            direction=T[:3,:3]@ray; length=np.linalg.norm(direction); direction/=length
            geom=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(e.model,e.data,T[:3,3].copy(),direction,None,1,-1,geom)
            np.testing.assert_allclose(frame['depth'][v,u],distance/length,atol=3e-4)


def test_blackout_is_uninformative_and_depth_is_gated_with_recovery():
    with SLAMEnvironment(Config(width=80,height=60)) as e:
        for t in [7.9,8.,9.,11.99,12.,13.]:
            e._set_state(t); frame=e._camera(t)
            if 8<=t<12:
                assert not frame['rgb'].any()
                assert not frame['depth'].any()
            else:
                assert frame['rgb'].std()>10
                assert np.mean(frame['depth']>0)>.5


def test_clean_depth_is_explicit_ablation_and_mono_is_off():
    for mode in ['off','clean']:
        with SLAMEnvironment(Config(width=80,height=60,depth_mode=mode)) as e:
            e._set_state(9.); frame=e._camera(9.)
            assert not frame['rgb'].any()
            assert bool(frame['depth'].any())==(mode=='clean')


def test_matched_scenarios_have_identical_imu_and_geometry():
    with SLAMEnvironment(Config(scenario='rich'),render=False) as a,SLAMEnvironment(Config(scenario='low_texture'),render=False) as b:
        np.testing.assert_array_equal(a.model.geom_pos,b.model.geom_pos)
        for _ in range(301):
            x,y=a.next_tick(),b.next_tick()
            if 'imu' in x: np.testing.assert_array_equal(x['imu']['accel'],y['imu']['accel'])


def test_low_texture_has_rich_initialization_and_identical_depth_after_recovery():
    with SLAMEnvironment(Config(scenario='rich',width=80,height=60)) as a,SLAMEnvironment(Config(scenario='low_texture',width=80,height=60)) as b:
        for t in [4.,9.,13.]:
            a._set_state(t); b._set_state(t)
            x,y=a._camera(t),b._camera(t)
            if t==9.:
                assert y['rgb'].mean()<x['rgb'].mean()/2
                assert not y['depth'].any()
            else:
                np.testing.assert_array_equal(x['rgb'],y['rgb'])
                np.testing.assert_array_equal(x['depth'],y['depth'])


def test_studio_furniture_keeps_camera_path_clear():
    from mm3dgs_bench.scene import P_B_C
    from mm3dgs_bench.trajectory import state
    with SLAMEnvironment(render=False) as env:
        for t in np.linspace(0,24,97):
            env._set_state(t)
            p,r,*_ = state(t,24)
            origin = p+r.apply(P_B_C)
            for axis in np.r_[np.eye(3),-np.eye(3)]:
                geom = np.array([-1],dtype=np.int32)
                distance = mujoco.mj_ray(env.model,env.data,origin,axis,None,1,-1,geom)
                assert distance > .25, (t, distance, geom)

import json
import numpy as np
import pytest
from mm3dgs_bench import Config
from mm3dgs_bench.dataset import export_dataset,measurements,replay
from mm3dgs_bench.evaluation import evaluate


@pytest.fixture
def dataset(tmp_path):
    root=tmp_path/'sequence'
    c=Config(scenario='rich',duration=6.0,width=64,height=48,physics_hz=100,camera_hz=10,imu_hz=100)
    report=export_dataset(c,root)
    assert report['frames']==61
    assert report['imu_samples']==601
    return root


def test_export_sync_and_chronological_replay(dataset):
    previous=-1; equal=[]
    for kind,m in measurements(dataset):
        ts=m['timestamp_ns']; assert ts>=previous
        if ts==previous: assert kind=='camera'
        previous=ts
        if kind=='camera':
            assert m['rgb'].dtype==np.uint8
            assert m['depth'].shape==(48,64)
    assert previous==6_000_000_000


def test_replay_keeps_truth_and_planted_biases_out_of_estimator(dataset,tmp_path):
    class Dummy:
        def reset(self,cal):
            assert 'config' not in cal
            assert 'groundtruth' not in str(cal)
            assert 'accel_bias' not in cal['imu']['noise']
            self.imu_time=-1
        def process_imu(self,m): self.imu_time=m['timestamp_ns']
        def process_frame(self,m):
            assert set(m)=={'timestamp_ns','rgb','depth'}
            assert self.imu_time==m['timestamp_ns']
            return np.eye(4)
    output=tmp_path/'estimate.tum'
    replay(dataset,Dummy(),output)
    assert len(np.loadtxt(output))==61


def test_evaluation_identity_missing_frames_and_no_scale_fit(dataset,tmp_path):
    gt=dataset/'groundtruth'/'camera.tum'
    report=evaluate(gt,gt)
    assert report['ate_rmse_m']<1e-10
    assert report['rpe_1s_translation_rmse_m']<1e-10
    assert report['coverage']==1
    assert report['rpe_1s_pairs']==51
    rows=np.loadtxt(gt)
    rows[:,1:4]*=2
    dest=tmp_path/'scaled.tum'; np.savetxt(dest,rows)
    assert evaluate(gt,dest)['ate_rmse_m']>.5
    np.savetxt(dest,rows[::2])
    assert evaluate(gt,dest)['coverage']==pytest.approx(31/61)


def test_overwrite_refused(dataset):
    with pytest.raises(FileExistsError): export_dataset(Config(scenario='rich'),dataset)


def test_official_dataset_bridge_packing_and_pose_conventions(dataset):
    torch=pytest.importorskip('torch')
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'scripts'/'mm3dgs_dataset.py'
    spec=importlib.util.spec_from_file_location('mm3dgs_dataset',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    bridge=module.MuJoCoDataset({},str(dataset.parent),dataset.name,device='cpu',desired_width=64,desired_height=48)
    assert len(bridge)==61
    assert bridge[0][4].shape==(0,37)
    np.testing.assert_allclose(bridge[0][3].numpy(),np.eye(4),atol=1e-6)
    raw=np.genfromtxt(dataset/'imu.csv',delimiter=',',skip_header=1)
    for index in [1,10,40,60]:
        color,depth,K,pose,packed=bridge[index]
        assert color.shape==(48,64,3) and depth.shape==(48,64,1)
        assert packed.shape==(10,37)
        end=round(bridge.tstamps[index]*1e9)
        block=raw[(raw[:,0]>end-100_000_000)&(raw[:,0]<=end)]
        np.testing.assert_allclose(packed[:,13:16].numpy(),block[:,1:4],rtol=1e-6)
        np.testing.assert_allclose(packed[:,25:28].numpy(),block[:,4:7],rtol=1e-6)
    cal=json.loads((dataset/'calibration.json').read_text())
    np.testing.assert_allclose(bridge.get_c2i_tf().numpy(),cal['T_B_C'],atol=1e-7)


def test_official_results_conversion_round_trip(dataset,tmp_path):
    import subprocess,sys
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    truth=np.loadtxt(dataset/'groundtruth'/'camera.tum')
    c2w=[]
    for row in truth:
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(row[4:8]).as_matrix();T[:3,3]=row[1:4];c2w.append(T)
    origin=np.linalg.inv(c2w[0]); poses=[]
    for T in c2w:
        w2c=np.linalg.inv(origin@T);q=Rotation.from_matrix(w2c[:3,:3]).as_quat()
        poses.append(np.r_[q[[3,0,1,2]],w2c[:3,3]])
    result=tmp_path/'results.npz';np.savez(result,pose_est=poses)
    output=tmp_path/'converted.tum'
    subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts'/'results_to_tum.py'),str(dataset),str(result),'--out',str(output)],check=True)
    report=evaluate(dataset/'groundtruth'/'camera.tum',output)
    assert report['ate_rmse_m']<1e-9
    assert report['coverage']==1


@pytest.mark.parametrize('mode,dynamics,loss',[('cv','const_velocity',False),('imu','imu',False),('imu-loss','imu',True)])
def test_launcher_prepares_native_config(dataset,tmp_path,mode,dynamics,loss):
    import subprocess,sys
    from pathlib import Path
    repo=tmp_path/'upstream';(repo/'configs').mkdir(parents=True)
    (repo/'configs'/'UTMM.yml').write_text('dataset: utmm\n')
    output=tmp_path/'output'
    subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts'/'run_mm3dgs.py'),
                    '--repo',str(repo),'--dataset',str(dataset),'--output',str(output),'--mode',mode,'--prepare-only'],check=True)
    config=json.loads((output/'config.json').read_text())
    assert config['tracking']['dynamics_model']==dynamics
    assert config['tracking']['use_imu_loss'] is loss
    assert config['tracking']['imu_weight']==1.
    assert config['tracking']['use_gt_pose'] is False
    assert config['cam']['png_depth_scale']==1
    assert config['cam']['image_width']==64
    assert config['stride']==1


def test_exported_scene_and_textures_reload_without_generator(dataset):
    import cv2
    import mujoco
    from mm3dgs_bench import SLAMEnvironment
    assert len(list((dataset/'assets').glob('*.png'))) == 6
    saved = mujoco.MjModel.from_xml_path(str(dataset/'scene.xml'))
    assets = {str(p.relative_to(dataset)):p.read_bytes() for p in (dataset/'assets').glob('*.png')}
    cfg = Config(**json.loads((dataset/'calibration.json').read_text())['config'])
    with SLAMEnvironment(cfg, scene_xml=(dataset/'scene.xml').read_text(), assets=assets) as env:
        assert env.model.ngeom == saved.ngeom
        frame = env.next_tick()['camera']
        original = cv2.cvtColor(cv2.imread(str(dataset/'rgb'/'000000000000.png')),cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(frame['rgb'],original)
        np.testing.assert_array_equal(frame['depth'],np.load(dataset/'depth'/'000000000000.npy'))


def test_comparison_predicts_before_image_and_hides_generation_parameters(dataset,tmp_path,monkeypatch):
    import mm3dgs_bench.comparison as comparison
    instances=[]
    class Adapter:
        def reset(self, calibration):
            assert 'config' not in calibration
            self.camera_time=self.imu_time=-1
            self.imu_calls=0
        def process_imu(self, sample):
            self.imu_time=sample['timestamp_ns']; self.imu_calls+=1
        def predict_pose(self, timestamp_ns):
            assert self.camera_time < timestamp_ns
            assert self.imu_time <= timestamp_ns
            return None if self.camera_time < 0 else np.eye(4)
        def process_frame(self, frame):
            self.camera_time=frame['timestamp_ns']
            return np.eye(4)
        def tracking_status(self):
            return {'visual_update':False}
    def instantiate(_):
        result=Adapter(); instances.append(result); return result
    monkeypatch.setattr(comparison,'_instantiate',instantiate)
    output=tmp_path/'paired'
    report=comparison.compare(dataset,output,'test:Adapter')
    assert instances[0].imu_calls==0 and instances[1].imu_calls==601
    assert len(np.loadtxt(output/'visual'/'prediction.tum'))==60
    assert report['visual_tracking_continues'] is False
    assert report['imu_benefit_validated'] is False
    with pytest.raises(FileExistsError): comparison.compare(dataset,output,'test:Adapter')


def test_matched_controls_allow_exposure_intervention_but_reject_different_depth_sensor(dataset,tmp_path):
    import shutil
    from mm3dgs_bench.comparison import control_identity
    cloned=tmp_path/'impaired'
    shutil.copytree(dataset,cloned)
    original=control_identity(dataset)
    meta=json.loads((cloned/'calibration.json').read_text())
    meta['config']['scenario']='motion_blur'
    meta['camera']['shutter']='finite exposure'
    (cloned/'calibration.json').write_text(json.dumps(meta))
    assert control_identity(cloned)==original
    meta['config']['depth_noise']*=2
    (cloned/'calibration.json').write_text(json.dumps(meta))
    assert control_identity(cloned)!=original

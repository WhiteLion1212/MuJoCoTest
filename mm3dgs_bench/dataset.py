"""Portable offline format and estimator replay without ground-truth exposure."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .environment import SLAMEnvironment


def export_dataset(config, destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('rgb', 'depth', 'groundtruth'):
        (root/folder).mkdir()
    with SLAMEnvironment(config) as env:
        (root/'calibration.json').write_text(json.dumps(env.calibration(include_generation=True), indent=2))
        (root/'scene.xml').write_text(env.xml)
        for name, blob in env.assets.items():
            asset_path = root/name
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(blob)
        with (root/'imu.csv').open('w', newline='') as fi, (root/'frames.csv').open('w', newline='') as ff, (root/'groundtruth'/'states.csv').open('w', newline='') as fg, (root/'groundtruth'/'camera.tum').open('w') as ft:
            iw, fw, gw = csv.writer(fi), csv.writer(ff), csv.writer(fg)
            iw.writerow(['timestamp_ns','wx','wy','wz','ax','ay','az'])
            fw.writerow(['timestamp_ns','rgb','depth'])
            gw.writerow(['timestamp_ns','px','py','pz','qx','qy','qz','qw','vx','vy','vz','bax','bay','baz','bgx','bgy','bgz'])
            frames = samples = 0
            while True:
                try: obs = env.next_tick()
                except StopIteration: break
                truth = env.ground_truth()
                if 'imu' in obs:
                    m = obs['imu']; samples += 1
                    iw.writerow([m['timestamp_ns'], *m['gyro'], *m['accel']])
                    T = truth['T_W_B']
                    gw.writerow([m['timestamp_ns'], *T[:3,3], *Rotation.from_matrix(T[:3,:3]).as_quat(), *truth['velocity_W'], *truth['accel_bias'], *truth['gyro_bias']])
                if 'camera' in obs:
                    m = obs['camera']; frames += 1
                    ts = m['timestamp_ns']; rgb = f'rgb/{ts:012d}.png'; dep = f'depth/{ts:012d}.npy'
                    if not cv2.imwrite(str(root/rgb), cv2.cvtColor(m['rgb'], cv2.COLOR_RGB2BGR)):
                        raise OSError('Failed to write RGB image')
                    np.save(root/dep, m['depth'].astype(np.float32))
                    fw.writerow([ts, rgb, dep])
                    T = truth['T_W_C']; q = Rotation.from_matrix(T[:3,:3]).as_quat()
                    ft.write(' '.join(map(str, [ts/1e9, *T[:3,3], *q]))+'\n')
        summary = {'frames': frames, 'imu_samples': samples, 'scenario': config.scenario}
        (root/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def measurements(root):
    """Yield IMU before image at equal timestamps, including samples after last image."""
    root = Path(root)
    with (root/'imu.csv').open() as fi, (root/'frames.csv').open() as ff:
        imu, frames = csv.DictReader(fi), csv.DictReader(ff)
        pending = next(imu, None)
        for frame in frames:
            ts = int(frame['timestamp_ns'])
            while pending is not None and int(pending['timestamp_ns']) <= ts:
                yield 'imu', _imu(pending)
                pending = next(imu, None)
            bgr = cv2.imread(str(root/frame['rgb']))
            if bgr is None: raise OSError(f"Missing image: {frame['rgb']}")
            yield 'camera', {'timestamp_ns':ts, 'rgb':cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 'depth':np.load(root/frame['depth'], allow_pickle=False)}
        while pending is not None:
            yield 'imu', _imu(pending)
            pending = next(imu, None)


def _imu(row):
    return {'timestamp_ns':int(row['timestamp_ns']),
            'gyro':np.array([float(row[k]) for k in ('wx','wy','wz')]),
            'accel':np.array([float(row[k]) for k in ('ax','ay','az')])}


def replay(root, estimator, output, use_imu=True):
    root = Path(root)
    calibration = json.loads((root/'calibration.json').read_text())
    # Simulation generation parameters contain planted biases. Estimators receive
    # only observable calibration and noise specifications, never planted values.
    cfg = calibration.pop('config')
    calibration['imu']['noise'] = {k:v for k,v in cfg.items() if k in ('accel_noise_density','gyro_noise_density','accel_bias_rw','gyro_bias_rw')}
    estimator.reset(calibration)
    with Path(output).open('w') as stream:
        for kind, m in measurements(root):
            if kind == 'imu':
                if use_imu: estimator.process_imu(m)
            else:
                T = estimator.process_frame(m)
                if T is None: continue  # Lost tracking is reflected in coverage.
                T = np.asarray(T, dtype=float)
                if T.shape != (4,4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0,0,0,1]):
                    raise ValueError('Estimator must return finite 4x4 T_W_C or None')
                R = T[:3,:3]
                if not np.allclose(R.T@R, np.eye(3), atol=1e-5) or np.linalg.det(R) < .999:
                    raise ValueError('Estimator rotation is not in SO(3)')
                q = Rotation.from_matrix(R).as_quat()
                stream.write(' '.join(map(str,[m['timestamp_ns']/1e9,*T[:3,3],*q]))+'\n')

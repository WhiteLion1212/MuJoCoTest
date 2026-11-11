"""Paired causal replay: image tracking, IMU initialization, and IMU residuals."""
import importlib
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .dataset import measurements
from .evaluation import evaluate

REFERENCE = 'mm3dgs_bench.reference:Estimator'
INITIALIZATION = 'mm3dgs_bench.reference:InitializationOnlyEstimator'


def control_identity(dataset):
    """Match actual motion, noisy IMU, scene, timing and sensor calibration."""
    dataset = Path(dataset)
    calibration = json.loads((dataset/'calibration.json').read_text())
    config = calibration.pop('config')
    calibration['camera'].pop('shutter', None)  # Exposure is the intervention.
    calibration['depth_mode'] = config['depth_mode']
    calibration['depth_noise_std_m'] = config['depth_noise']
    calibration['evaluation_interval'] = [config['blackout_start'], config['blackout_end']]
    digest = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode())
    for name in ('imu.csv', 'groundtruth/camera.tum', 'scene.xml'):
        digest.update((dataset/name).read_bytes())
    for asset in sorted((dataset/'assets').glob('*')):
        digest.update(asset.name.encode())
        digest.update(asset.read_bytes())
    return digest.hexdigest()


def _row(timestamp_ns, pose):
    pose = np.asarray(pose, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])):
        raise ValueError('Estimator must return finite T_W_C or None')
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1., atol=1e-5):
        raise ValueError('Estimator rotation is not in SO(3)')
    return np.r_[timestamp_ns/1e9, pose[:3, 3], Rotation.from_matrix(r).as_quat()]


def _instantiate(spec):
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)()


def _increment_errors(truth, predictions, anchors, interval, output=None):
    """Relative motion from the most recent image to the predicted next pose.

    This is a prediction-horizon RPE, independent of historical map drift.
    Absolute errors remain available in the separately aligned TUM evaluation.
    """
    errors = []
    for predicted, anchor in zip(predictions, anchors):
        i = np.argmin(abs(truth[:, 0]-anchor[0]))
        j = np.argmin(abs(truth[:, 0]-predicted[0]))
        if max(abs(truth[i, 0]-anchor[0]), abs(truth[j, 0]-predicted[0])) > .002:
            continue
        ra, rp = Rotation.from_quat(anchor[4:8]), Rotation.from_quat(predicted[4:8])
        rg, rt = Rotation.from_quat(truth[i, 4:8]), Rotation.from_quat(truth[j, 4:8])
        dp = ra.inv().apply(predicted[1:4]-anchor[1:4])
        dg = rg.inv().apply(truth[j, 1:4]-truth[i, 1:4])
        angle = ((rg.inv()*rt).inv()*(ra.inv()*rp)).magnitude()*180/np.pi
        errors.append([predicted[0], np.linalg.norm(dp-dg), angle, predicted[0]-anchor[0]])
    errors = np.array(errors).reshape(-1, 4)
    if output is not None:
        np.savetxt(output, errors, delimiter=',', comments='',
                   header='timestamp_s,translation_error_m,rotation_error_deg,horizon_s')
    result = {}
    for name, mask in (('all', np.ones(len(errors), bool)),
                       ('during_impairment', (errors[:, 0] >= interval[0]) & (errors[:, 0] < interval[1]))):
        data = errors[mask]
        result[name] = {'pairs': len(data),
                        'translation_rmse_m': float(np.sqrt(np.mean(data[:, 1]**2))) if len(data) else None,
                        'rotation_rmse_deg': float(np.sqrt(np.mean(data[:, 2]**2))) if len(data) else None,
                        'mean_horizon_s': float(np.mean(data[:, 3])) if len(data) else None}
    return result


def _run(dataset, destination, estimator, use_imu, interval):
    calibration = json.loads((dataset/'calibration.json').read_text())
    calibration.pop('config', None)
    estimator.reset(calibration)
    poses, predictions, anchors, status = [], [], [], []
    for kind, sample in measurements(dataset):
        if kind == 'imu':
            if use_imu:
                estimator.process_imu(sample)
            continue
        ts = sample['timestamp_ns']
        # Crucially, predict BEFORE the image at this timestamp is processed.
        prediction = estimator.predict_pose(ts)
        if prediction is not None and poses:
            predictions.append(_row(ts, prediction))
            anchors.append(poses[-1])
        pose = estimator.process_frame(sample)
        if pose is not None:
            poses.append(_row(ts, pose))
        info = estimator.tracking_status() if hasattr(estimator, 'tracking_status') else {}
        status.append({'time_s': ts/1e9, 'pose_returned': pose is not None, **info})
    destination.mkdir()
    np.savetxt(destination/'tracking.tum', np.array(poses).reshape(-1, 8))
    np.savetxt(destination/'prediction.tum', np.array(predictions).reshape(-1, 8))
    (destination/'tracking_status.json').write_text(json.dumps(status, indent=2)+'\n')
    truth_file = dataset/'groundtruth'/'camera.tum'
    if not poses:
        return {'tracking': {'coverage': 0.}, 'prediction': {'coverage': 0.},
                'visual_update_fraction_during_impairment': 0., 'max_visual_gap_s': None}
    truth = np.atleast_2d(np.loadtxt(truth_file))
    # Fix BOTH tracking and prediction alignment from the same tracking origin.
    first = poses[0]
    idx = np.argmin(abs(truth[:, 0]-first[0]))
    R = (Rotation.from_quat(truth[idx, 4:8])*Rotation.from_quat(first[4:8]).inv()).as_matrix()
    offset = truth[idx, 1:4]-R @ first[1:4]
    tracking = evaluate(truth_file, destination/'tracking.tum', interval, alignment=(R, offset))
    prediction = evaluate(truth_file, destination/'prediction.tum', interval, alignment=(R, offset))
    affected = [s for s in status if interval[0] <= s['time_s'] < interval[1]]
    known_status = affected and all('visual_update' in s for s in affected)
    update_fraction = (float(np.mean([s['visual_update'] for s in affected])) if known_status else None)
    updates = [s['time_s'] for s in status if s.get('visual_update')]
    gaps = np.diff([truth[0, 0], *updates, truth[-1, 0]]) if updates else None
    return {'tracking': tracking, 'prediction': prediction,
            'prediction_increment': _increment_errors(truth, predictions, anchors, interval, destination/'prediction_errors.csv'),
            'visual_update_fraction_during_impairment': update_fraction,
            'max_visual_gap_s': float(max(gaps)) if gaps is not None else None}


def compare(dataset, destination, estimator=REFERENCE, initialization_estimator=None):
    """Same images/IMU, fresh instances, fixed origin, no future image access.

    An external adapter implements reset/process_imu/process_frame plus
    predict_pose(timestamp_ns). tracking_status() returns visual_update: bool;
    without it continuity is unknown and the acceptance check cannot pass.
    """
    dataset, destination = Path(dataset), Path(destination)
    config = json.loads((dataset/'calibration.json').read_text())['config']
    interval = (config['blackout_start'], config['blackout_end'])
    # Validate adapter interfaces before starting any expensive replay.
    jobs = [('visual', _instantiate(estimator), False), ('imu_fused', _instantiate(estimator), True)]
    if initialization_estimator is not None:
        jobs.insert(1, ('imu_initialization', _instantiate(initialization_estimator), True))
    for _, adapter, _ in jobs:
        for method in ('reset', 'process_imu', 'process_frame', 'predict_pose'):
            if not callable(getattr(adapter, method, None)):
                raise TypeError(f'Estimator adapter must implement {method}')
    destination.mkdir(parents=True, exist_ok=False)
    variants = {name: _run(dataset, destination/name, adapter, imu, interval)
                for name, adapter, imu in jobs}
    visual, fused = variants['visual'], variants['imu_fused']
    before = visual.get('prediction_increment', {}).get('during_impairment', {})
    after = fused.get('prediction_increment', {}).get('during_impairment', {})

    def reduction(key):
        a, b = before.get(key), after.get(key)
        return float(1-b/a) if a is not None and b is not None and a > 1e-12 else None

    position_gain = reduction('translation_rmse_m')
    rotation_gain = reduction('rotation_rmse_deg')
    continuity = all(v['visual_update_fraction_during_impairment'] is not None
                     and v['visual_update_fraction_during_impairment'] >= .9
                     and v['max_visual_gap_s'] <= .5
                     and v['tracking']['coverage'] >= .98
                     and v['prediction']['coverage'] >= .98 for v in (visual, fused))
    report = {'scenario': config['scenario'], 'seed': config['seed'], 'impairment_interval_s': interval,
              'matched_measurement_id': control_identity(dataset),
              'protocol': 'One-step camera pose prediction before current image; shared fixed tracking origin; no scale fit',
              'estimator': estimator, 'variants': variants,
              'translation_prediction_reduction': position_gain,
              'rotation_prediction_reduction': rotation_gain,
              'visual_tracking_continues': bool(continuity),
              'imu_benefit_validated': bool(continuity and position_gain is not None and position_gain >= .15
                                            and rotation_gain is not None and rotation_gain >= -.1),
              'criterion': 'Both modes: >=90% visual updates, <=0.5s visual gap, >=98% pose/prediction coverage; >=15% lower impaired-interval next-frame translation RMSE; rotation RMSE regression <=10%'}
    if 'imu_initialization' in variants:
        init = variants['imu_initialization']
        base = init.get('prediction_increment', {}).get('during_impairment', {}).get('translation_rmse_m')
        optimized = after.get('translation_rmse_m')
        gain = float(1-optimized/base) if base is not None and optimized is not None and base > 1e-12 else None
        report['translation_prediction_reduction_vs_initialization'] = gain
        report['imu_optimization_benefit_validated'] = bool(
            report['imu_benefit_validated'] and gain is not None and gain >= .15
            and init['visual_update_fraction_during_impairment'] is not None
            and init['visual_update_fraction_during_impairment'] >= .9
            and init['max_visual_gap_s'] <= .5)
    (destination/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def controlled_summary(reports):
    """Separate generic IMU benefit from mitigation of camera-induced error.

    A matched rich control must share the seed, motion and sensor settings.
    Positive results require the effect to worsen visual-only forecasting and
    fusion to reduce that added error, while both modes keep using the camera.
    """
    controls = {(r['seed'], r.get('matched_measurement_id')): r for r in reports
                if r['scenario'] == 'rich' and r.get('matched_measurement_id')}
    cases = []
    for report in reports:
        key = (report['seed'], report.get('matched_measurement_id'))
        if report['scenario'] == 'rich' or key not in controls:
            continue
        healthy = controls[key]
        penalties = {}
        for mode in ('visual', 'imu_fused'):
            impaired = report['variants'][mode]
            clean = healthy['variants'][mode]
            bad = impaired.get('prediction_increment', {}).get('during_impairment', {}).get('translation_rmse_m')
            good = clean.get('prediction_increment', {}).get('during_impairment', {}).get('translation_rmse_m')
            bad_ate = impaired['tracking'].get('during_outage', {}).get('ate_rmse_m')
            good_ate = clean['tracking'].get('during_outage', {}).get('ate_rmse_m')
            penalties[mode] = {'prediction_rmse_increase_m': bad-good if bad is not None and good is not None else None,
                              'trajectory_rmse_increase_m': bad_ate-good_ate if bad_ate is not None and good_ate is not None else None}
        added = penalties['visual']['prediction_rmse_increase_m']
        fused_added = penalties['imu_fused']['prediction_rmse_increase_m']
        mitigation = 1-fused_added/added if added is not None and added > 1e-9 and fused_added is not None else None
        cases.append({'scenario': report['scenario'], 'seed': report['seed'],
                      'penalty_relative_to_matched_clean_camera': penalties,
                      'camera_induced_prediction_penalty_reduction': mitigation,
                      'camera_impairment_counteracted': bool(report['imu_benefit_validated']
                                                            and mitigation is not None and mitigation >= .15)})
    return {'runs': reports, 'matched_clean_controls': cases,
            'control_criterion': 'Visual-only forecast worsens under impairment; IMU fusion reduces the added forecast RMSE by >=15%, with visual continuity and paired IMU-benefit criteria also passing'}

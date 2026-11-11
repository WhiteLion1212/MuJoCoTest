#!/usr/bin/env python3
"""Run the official CUDA implementation through a local dataset bridge.

Run in the upstream Conda environment. Does not edit upstream files or change
its inertial dynamics algorithm. --prepare-only writes a complete config using
only the standard library; it also works in this simulator's CPU environment.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',choices=['cv','imu','imu-loss'],default='imu',
                   help='Visual prediction, IMU initialization, or IMU initialization plus pose-prior loss')
    p.add_argument('--imu-weight',type=float,default=1.)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if not math.isfinite(args.imu_weight) or args.imu_weight < 0 or (args.mode=='imu-loss' and args.imu_weight==0):
        p.error('IMU weight must be finite and nonnegative; imu-loss requires a positive weight')
    repo=args.repo.resolve(); dataset=args.dataset.resolve(); output=args.output.resolve()
    if not (repo/'configs'/'UTMM.yml').is_file(): p.error('Missing official configs/UTMM.yml')
    cal=json.loads((dataset/'calibration.json').read_text())
    if cal['config']['imu_hz']!=100: p.error('Official implementation requires --imu-hz 100')
    camera=cal['camera']; K=camera['K']
    # JSON is valid YAML. Keep a distinct config for each mode/output.
    # Load upstream YAML lazily, so prepare-only can use the supplied JSON template.
    template=Path(__file__).with_name('upstream_UTMM_config.json')
    cfg=json.loads(template.read_text())
    cfg.update(inputdir=str(dataset.parent),scene=dataset.name,outputdir=str(output),stride=1,start_idx=0,
               desired_height=camera['height'],desired_width=camera['width'])
    cfg['cam'].update(image_height=camera['height'],image_width=camera['width'],fx=K[0][0],fy=K[1][1],
                      cx=K[0][2],cy=K[1][2],crop_edge=0,png_depth_scale=1.,fps=camera['rate_hz'])
    cfg['tracking'].update(dynamics_model='const_velocity' if args.mode=='cv' else 'imu',use_gt_pose=False,
                           use_imu_loss=args.mode=='imu-loss',imu_weight=args.imu_weight)
    # Invalid depth and NIQE on black frames can break the native mapper. Keep
    # native behavior visible: the bridge does not invent replacement measurements.
    output.mkdir(parents=True,exist_ok=True)
    config_path=output/'config.json'; config_path.write_text(json.dumps(cfg,indent=2)+'\n')
    print(f'Config: {config_path}')
    if args.prepare_only: return
    sys.path.insert(0,str(repo))
    from mm3dgs_dataset import MuJoCoDataset
    slam_module=importlib.import_module('slam.SLAM')
    original=slam_module.get_dataset_type
    slam_module.get_dataset_type=lambda name: MuJoCoDataset if name.lower()=='utmm' else original(name)
    # Use simulator gravity magnitude while preserving the upstream convention:
    # its G is negative gravity in the first-camera (relative world) frame.
    import torch
    import utils.pose_utils as pose_utils
    pose_utils.G=torch.tensor([0.,-9.81,0.])
    slam_module.SLAM(cfg).run()


if __name__=='__main__': main()

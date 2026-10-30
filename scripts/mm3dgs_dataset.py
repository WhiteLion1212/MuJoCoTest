"""Dataset bridge for official MM3DGS, independent of the simulator package.

Ground truth is returned only for official initialization/evaluation, following
its native interface. Never enable tracking.use_gt_pose for benchmark results.
"""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch


class MuJoCoDataset:
    def __init__(self, config_dict, basedir, sequence, stride=1, start=0, end=-1,
                 desired_height=None, desired_width=None, device='cuda:0',
                 relative_pose=True, **kwargs):
        if start != 0 or stride != 1:
            raise ValueError('Bridge requires start_idx=0 and stride=1 to preserve IMU grouping')
        self.root=Path(basedir)/sequence
        cal=json.loads((self.root/'calibration.json').read_text())
        if cal['config']['imu_hz']!=100:
            raise ValueError('Official MM3DGS hardcodes 100 Hz IMU; generate with --imu-hz 100')
        self.device=device
        with (self.root/'frames.csv').open() as f: self.frames=list(csv.DictReader(f))
        if end!=-1: self.frames=self.frames[:end]
        self.tstamps=[int(f['timestamp_ns'])/1e9 for f in self.frames]
        self.K=np.array(cal['camera']['K'])
        self.height=desired_height or cal['camera']['height']
        self.width=desired_width or cal['camera']['width']
        # No resize keeps the pixel-center convention exact.
        if (self.width,self.height)!=(cal['camera']['width'],cal['camera']['height']):
            raise ValueError('Use native dataset dimensions in the generated MM3DGS config')
        self.T_B_C=np.array(cal['T_B_C'])
        truth=np.atleast_2d(np.loadtxt(self.root/'groundtruth'/'camera.tum'))
        self.poses=[]
        for row in truth[:len(self.frames)]:
            T=np.eye(4); T[:3,:3]=Rotation.from_quat(row[4:8]).as_matrix(); T[:3,3]=row[1:4]
            self.poses.append(T)
        if relative_pose:
            origin=np.linalg.inv(self.poses[0])
            self.poses=[origin@T for T in self.poses]
        # Official propagate_imu expects the 37 columns of a flattened ROS IMU,
        # with angular velocity at [13:16] and specific force at [25:28].
        raw=np.atleast_2d(np.genfromtxt(self.root/'imu.csv',delimiter=',',skip_header=1))
        ts=raw[:,0].astype(np.int64)
        self.imus=[]; previous=0
        for i,f in enumerate(self.frames):
            current=int(f['timestamp_ns'])
            block=raw[(ts>previous)&(ts<=current)] if i else raw[:0]
            packed=np.zeros((len(block),37),np.float32)
            packed[:,13:16]=block[:,1:4]; packed[:,25:28]=block[:,4:7]
            self.imus.append(packed); previous=current
        self.num_imgs=len(self.frames)

    def __len__(self): return self.num_imgs

    def __getitem__(self,index):
        f=self.frames[index]
        color=cv2.cvtColor(cv2.imread(str(self.root/f['rgb'])),cv2.COLOR_BGR2RGB)
        depth=np.load(self.root/f['depth'],allow_pickle=False)[...,None]
        K=np.eye(4); K[:3,:3]=self.K
        return tuple(torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in
                     (color,depth,K,self.poses[index],self.imus[index]))

    def get_c2i_tf(self):
        return torch.as_tensor(self.T_B_C,dtype=torch.float32,device=self.device)

    def get_cam_K(self):
        return torch.as_tensor(self.K,dtype=torch.float32,device=self.device)

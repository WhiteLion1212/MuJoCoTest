#!/usr/bin/env python3
"""Convert official MM3DGS w2c, wxyz estimates to timestamped c2w, xyzw."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('dataset',type=Path); p.add_argument('results',type=Path); p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
poses=np.load(a.results,allow_pickle=False)['pose_est']
with (a.dataset/'frames.csv').open() as f: frames=list(csv.DictReader(f))
if len(poses)>len(frames): raise ValueError('More poses than dataset frames')
with a.out.open('w') as f:
    for frame,pose in zip(frames,poses):
        r=Rotation.from_quat(pose[[1,2,3,0]])
        ri=r.inv(); pos=-ri.apply(pose[4:7])
        f.write(' '.join(map(str,[int(frame['timestamp_ns'])/1e9,*pos,*ri.as_quat()]))+'\n')

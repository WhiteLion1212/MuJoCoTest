#!/usr/bin/env python3
"""Build a side-by-side reference/camera preview from an exported sequence.

Requires ffmpeg for browser-compatible H.264 encoding. The clean reference is
for visualization only and must never be supplied to a tested estimator.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from mm3dgs_bench import Config, SLAMEnvironment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        p.error('Install ffmpeg to encode the preview as H.264')
    meta = json.loads((a.dataset/'calibration.json').read_text())
    cfg = Config(**{**meta['config'], 'scenario': 'rich'})
    width, height = cfg.width, cfg.height
    with (a.dataset/'frames.csv').open() as f:
        frames = list(csv.DictReader(f))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mm3dgs-preview-') as temp:
        raw = Path(temp)/'raw.mp4'
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), cfg.camera_hz, (2*width,height+60))
        if not writer.isOpened():
            raise RuntimeError('OpenCV could not initialize its MP4 writer')
        try:
            with SLAMEnvironment(cfg) as env:
                for row in frames:
                    t = int(row['timestamp_ns'])/1e9
                    measured = cv2.imread(str(a.dataset/row['rgb']))
                    if measured is None:
                        raise OSError(f"Cannot read {row['rgb']}")
                    bad = cfg.blackout_start <= t < cfg.blackout_end and meta['config']['scenario'] != 'rich'
                    if bad:
                        env._set_state(t)
                        reference = cv2.cvtColor(env._camera(t)['rgb'], cv2.COLOR_RGB2BGR)
                    else:
                        reference = measured
                    canvas = np.full((height+60,2*width,3),(26,24,22),np.uint8)
                    canvas[32:height+32,:width] = reference
                    canvas[32:height+32,width:] = measured
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(canvas,'Clean reference (diagnostic only)',(8,22),font,.46,(230,230,230),1,cv2.LINE_AA)
                    cv2.putText(canvas,'Camera input to the estimator',(width+8,22),font,.46,(230,230,230),1,cv2.LINE_AA)
                    valid_depth = np.load(a.dataset/row['depth'],allow_pickle=False).any()
                    label = f't={t:05.2f}s | depth '+('available' if valid_depth else 'unavailable')+' | IMU continues'
                    cv2.putText(canvas,label,(8,height+51),font,.49,(60,190,255) if bad else (220,220,220),1,cv2.LINE_AA)
                    writer.write(canvas)
        finally:
            writer.release()
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(raw),'-c:v','libx264',
                        '-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out)],check=True)
    print(f'Wrote {a.out}: {len(frames)} frames at {cfg.camera_hz} fps')


if __name__ == '__main__':
    main()

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


def prediction_chart(results, width, duration, interval):
    curves = []
    for mode, label, color in [('visual', 'Visual', (80, 160, 255)),
                                ('imu_initialization', 'IMU init', (255, 170, 100)),
                                ('imu_fused', 'IMU fused', (100, 220, 100))]:
        path = results/mode/'prediction_errors.csv'
        if path.exists():
            data = np.atleast_2d(np.loadtxt(path, delimiter=',', skiprows=1))
            if data.size:
                curves.append((label, color, data))
    if not curves:
        raise ValueError('Results contain no prediction_errors.csv; run compare first')
    ymax = max(10., max(float(data[:, 1].max()*1000) for _, _, data in curves)*1.05)
    x0, x1, top, bottom = 48, width-12, 45, 143
    base = np.full((180, width, 3), (26, 24, 22), np.uint8)
    x = lambda t: round(x0+(x1-x0)*t/duration)
    cv2.rectangle(base, (x(interval[0]), top), (x(interval[1]), bottom), (45, 40, 32), -1)
    cv2.rectangle(base, (x0, top), (x1, bottom), (100, 100, 100), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(base, 'Next-frame motion prediction error (mm)', (8, 17), font, .43, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(base, f'{ymax:.0f}', (4, top+5), font, .38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(base, '0', (29, bottom), font, .38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(base, '0s', (x0, bottom+18), font, .38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(base, f'{duration:g}s', (x1-35, bottom+18), font, .38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(base, 'Predictions before each new image; shaded interval = impairment', (8, 176), font, .35, (190, 190, 190), 1, cv2.LINE_AA)
    paths = []
    for label, color, data in curves:
        px = x0+(x1-x0)*data[:, 0]/duration
        py = bottom-(bottom-top)*data[:, 1]*1000/ymax
        paths.append((label, color, data, np.rint(np.c_[px, py]).astype(np.int32)))

    def draw(time_s):
        panel = base.copy()
        for i, (label, color, data, points) in enumerate(paths):
            mask = data[:, 0] <= time_s+1e-9
            if mask.sum() >= 2:
                cv2.polylines(panel, [points[mask]], False, color, 1, cv2.LINE_AA)
            value = f'{data[mask][-1, 1]*1000:.1f}' if mask.any() else '--'
            cv2.putText(panel, f'{label}: {value}', (8+i*(width//3), 35), font, .4, color, 1, cv2.LINE_AA)
        cv2.line(panel, (x(time_s), top), (x(time_s), bottom), (200, 200, 200), 1)
        return panel
    return draw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--reference-dataset', type=Path, help='Optional matched rich sequence, for faster rendering')
    p.add_argument('--results', type=Path, help='Comparison directory; adds per-frame prediction-error traces')
    a = p.parse_args()
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        p.error('Install ffmpeg to encode the preview as H.264')
    meta = json.loads((a.dataset/'calibration.json').read_text())
    cfg = Config(**{**meta['config'], 'scenario': 'rich'})
    width, height = cfg.width, cfg.height
    chart = (prediction_chart(a.results, 2*width, cfg.duration, (cfg.blackout_start, cfg.blackout_end))
             if a.results is not None else None)
    output_height = height+60+(180 if chart else 0)
    with (a.dataset/'frames.csv').open() as f:
        frames = list(csv.DictReader(f))
    if a.reference_dataset is not None:
        if ((a.reference_dataset/'groundtruth'/'camera.tum').read_bytes()
                != (a.dataset/'groundtruth'/'camera.tum').read_bytes()):
            p.error('Reference dataset must have the identical camera trajectory and timestamps')
        reference_meta = json.loads((a.reference_dataset/'calibration.json').read_text())
        if (reference_meta['config']['scenario'] != 'rich'
                or reference_meta['camera']['K'] != meta['camera']['K']
                or reference_meta['scene_version'] != meta['scene_version']):
            p.error('Reference must use the same rich scene and camera calibration')
        with (a.reference_dataset/'frames.csv').open() as stream:
            reference_paths = {row['timestamp_ns']: row['rgb'] for row in csv.DictReader(stream)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mm3dgs-preview-') as temp:
        raw = Path(temp)/'raw.mp4'
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), cfg.camera_hz, (2*width,output_height))
        if not writer.isOpened():
            raise RuntimeError('OpenCV could not initialize its MP4 writer')
        try:
            scene_xml = (a.dataset/'scene.xml').read_text()
            assets = {str(path.relative_to(a.dataset)): path.read_bytes() for path in (a.dataset/'assets').glob('*.png')}
            with SLAMEnvironment(cfg, scene_xml=scene_xml, assets=assets) as env:
                for row in frames:
                    t = int(row['timestamp_ns'])/1e9
                    measured = cv2.imread(str(a.dataset/row['rgb']))
                    if measured is None:
                        raise OSError(f"Cannot read {row['rgb']}")
                    bad = cfg.blackout_start <= t < cfg.blackout_end and meta['config']['scenario'] != 'rich'
                    if bad:
                        if a.reference_dataset is not None:
                            reference = cv2.imread(str(a.reference_dataset/reference_paths[row['timestamp_ns']]))
                            if reference is None or reference.shape != measured.shape:
                                raise OSError('Missing or incompatible reference image')
                        else:
                            env._set_state(t)
                            reference = cv2.cvtColor(env._camera(t)['rgb'], cv2.COLOR_RGB2BGR)
                    else:
                        reference = measured
                    canvas = np.full((output_height,2*width,3),(26,24,22),np.uint8)
                    canvas[32:height+32,:width] = reference
                    canvas[32:height+32,width:] = measured
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(canvas,'Clean reference',(8,22),font,.46,(230,230,230),1,cv2.LINE_AA)
                    cv2.putText(canvas,meta['config']['scenario']+' camera input',(width+8,22),font,.46,(230,230,230),1,cv2.LINE_AA)
                    valid_depth = np.mean(np.load(a.dataset/row['depth'],allow_pickle=False)>0)
                    phase = 'IMPAIRED' if bad else 'normal'
                    label = f't={t:05.2f}s | {phase} | valid depth {valid_depth:.0%} | IMU active'
                    cv2.putText(canvas,label,(8,height+51),font,.49,(60,190,255) if bad else (220,220,220),1,cv2.LINE_AA)
                    if chart:
                        canvas[height+60:] = chart(t)
                    writer.write(canvas)
        finally:
            writer.release()
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(raw),'-c:v','libx264',
                        '-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out)],check=True)
    print(f'Wrote {a.out}: {len(frames)} frames at {cfg.camera_hz} fps')


if __name__ == '__main__':
    main()

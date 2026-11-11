#!/usr/bin/env python3
"""Generate matched camera impairments, causal comparisons and H.264 previews."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mm3dgs_bench import Config
from mm3dgs_bench.comparison import compare, controlled_summary, REFERENCE, INITIALIZATION
from mm3dgs_bench.dataset import export_dataset
from mm3dgs_bench.environment import SCENARIOS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 19])
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--height', type=int, default=240)
    p.add_argument('--estimator', default=REFERENCE, help='module:Class adapter')
    p.add_argument('--initialization-estimator', help='Optional IMU-initialization-only adapter')
    p.add_argument('--generate-only', action='store_true', help='Export datasets/videos without running an estimator')
    a = p.parse_args()
    if shutil.which('ffmpeg') is None:
        p.error('ffmpeg is required for the scenario videos')
    jobs = []
    for seed in a.seeds:
        for scenario in a.scenarios:
            config = Config(scenario=scenario, seed=seed, width=a.width, height=a.height)
            dataset = a.out/'datasets'/f'seed_{seed}'/scenario
            result = a.out/'results'/f'seed_{seed}'/scenario
            video = a.out/'videos'/f'{scenario}_seed_{seed}.mp4'
            if any(path.exists() for path in (dataset, result, video)):
                p.error(f'Output already exists for seed {seed}, scenario {scenario}')
            jobs.append((config, dataset, result, video))
    reports = []
    for config, dataset, result, video in jobs:
        print(f'Generating {config.scenario}, seed {config.seed}', flush=True)
        export_dataset(config, dataset)
        if not a.generate_only:
            init = a.initialization_estimator or (INITIALIZATION if a.estimator == REFERENCE else None)
            report = compare(dataset, result, a.estimator, init)
            reports.append(report)
            print(json.dumps({key: report[key] for key in ('scenario', 'seed', 'visual_tracking_continues',
                             'translation_prediction_reduction', 'imu_benefit_validated')}), flush=True)
        command = [sys.executable, str(Path(__file__).with_name('make_preview.py')),
                   str(dataset), '--out', str(video)]
        rich = dataset.with_name('rich')
        if (rich/'summary.json').exists():
            command += ['--reference-dataset', str(rich)]
        subprocess.run(command, check=True)
        (a.out/'metrics.json').write_text(json.dumps(controlled_summary(reports), indent=2)+'\n')
    print(f'Finished {len(jobs)} sequences and videos in {a.out}', flush=True)


if __name__ == '__main__':
    main()

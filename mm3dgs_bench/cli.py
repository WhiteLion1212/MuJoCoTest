import argparse
import importlib
import json
from pathlib import Path
from .environment import Config, SCENARIOS
from .dataset import export_dataset, replay
from .evaluation import evaluate


def main():
    parser=argparse.ArgumentParser(description='MuJoCo camera/IMU stress benchmark for MM3DGS')
    sub=parser.add_subparsers(dest='command',required=True)
    gen=sub.add_parser('generate',help='Export one scenario or a matched suite')
    gen.add_argument('--out',type=Path,required=True)
    gen.add_argument('--scenario',choices=(*SCENARIOS,'suite'),default='blackout')
    gen.add_argument('--width',type=int,default=640); gen.add_argument('--height',type=int,default=480)
    gen.add_argument('--duration',type=float,default=24.)
    gen.add_argument('--seed',type=int,default=7)
    gen.add_argument('--imu-hz',type=int,default=100); gen.add_argument('--camera-hz',type=int,default=30)
    gen.add_argument('--physics-hz',type=int,default=600)
    gen.add_argument('--blackout-start',type=float,default=8.); gen.add_argument('--blackout-end',type=float,default=12.)
    gen.add_argument('--depth-mode',choices=('gated','off','clean'),default='gated')
    prev=sub.add_parser('preview',help='Render sample views and produce a sensor sanity report')
    prev.add_argument('--out',type=Path,required=True)
    run=sub.add_parser('replay',help='Run an estimator exposing reset/process_imu/process_frame')
    run.add_argument('dataset',type=Path); run.add_argument('--estimator',required=True,help='module:Class')
    run.add_argument('--out',required=True,type=Path); run.add_argument('--no-imu',action='store_true')
    ev=sub.add_parser('evaluate',help='ATE, 1s RPE, and coverage, including blackout segments')
    ev.add_argument('dataset',type=Path); ev.add_argument('estimate',type=Path)
    ev.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.command=='generate':
        names=SCENARIOS if args.scenario=='suite' else (args.scenario,)
        # Validate every config and output path before writing anything.
        jobs=[]
        for name in names:
            cfg=Config(scenario=name,width=args.width,height=args.height,duration=args.duration,seed=args.seed,
                       imu_hz=args.imu_hz,camera_hz=args.camera_hz,physics_hz=args.physics_hz,
                       blackout_start=args.blackout_start,blackout_end=args.blackout_end,depth_mode=args.depth_mode)
            dest=args.out/name if args.scenario=='suite' else args.out
            if dest.exists(): parser.error(f'Output already exists: {dest}')
            jobs.append((cfg,dest))
        for cfg,dest in jobs:
            print(f'Generating {cfg.scenario} into {dest}',flush=True)
            print(json.dumps(export_dataset(cfg,dest)),flush=True)
    elif args.command=='preview':
        from .diagnostics import preview
        print(json.dumps(preview(args.out),indent=2))
    elif args.command=='replay':
        module,cls=args.estimator.split(':')
        replay(args.dataset,getattr(importlib.import_module(module),cls)(),args.out,not args.no_imu)
    elif args.command=='evaluate':
        config=json.loads((args.dataset/'calibration.json').read_text())['config']
        report=evaluate(args.dataset/'groundtruth'/'camera.tum',args.estimate,(config['blackout_start'],config['blackout_end']))
        body=json.dumps(report,indent=2)
        if args.out: args.out.write_text(body+'\n')
        print(body)


if __name__=='__main__': main()

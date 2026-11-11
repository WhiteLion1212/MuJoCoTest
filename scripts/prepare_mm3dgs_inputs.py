#!/usr/bin/env python3
"""Verify and extract the committed MM3DGS inputs (standard library only).

git lfs pull
python scripts/prepare_mm3dgs_inputs.py --out inputs
python scripts/run_mm3dgs.py --repo /path/to/MM3DGS-SLAM \
    --dataset inputs/datasets/seed_7/motion_blur --mode imu-loss --output runs/blur_imu_loss
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def extract_inputs(archive, manifest, destination):
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    with archive.open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Dataset is a Git LFS pointer. Run git lfs pull in this repository first.')
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != manifest['archive_sha256']:
        raise ValueError('Dataset archive checksum mismatch; retry git lfs pull')
    with zipfile.ZipFile(archive) as bundle:
        # Validate all paths before creating output; the shared measurements
        # remain beside the per-sequence folders that refer to them in CSV.
        root = destination.resolve()
        for entry in bundle.infolist():
            target = (root/entry.filename).resolve()
            if not target.is_relative_to(root) or ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f'Unsafe archive entry: {entry.filename}')
        destination.mkdir(parents=True)
        bundle.extractall(destination)
    for sequence in manifest['sequences']:
        folder = destination/sequence['path']
        for name in ('calibration.json', 'frames.csv', 'imu.csv', 'groundtruth/camera.tum'):
            if not (folder/name).is_file():
                raise ValueError(f'Incomplete dataset: {folder/name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--manifest', type=Path, default=root/'test_inputs/mm3dgs/manifest.json')
    parser.add_argument('--out', type=Path, default=root/'inputs')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    archive = args.manifest.parent/manifest['archive']
    extract_inputs(archive, manifest, args.out)
    print(f'Extracted {len(manifest["sequences"])} sequences to {args.out}')
    print('Keep datasets/shared together with the sequence folders; frames.csv references the shared files.')


if __name__ == '__main__':
    main()

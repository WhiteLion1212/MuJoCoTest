import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def extractor():
    source = Path(__file__).parents[1]/'scripts/prepare_mm3dgs_inputs.py'
    spec = importlib.util.spec_from_file_location('prepare_inputs', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_inputs


def test_extracts_shared_dataset_paths_and_preserves_existing_output(tmp_path, extractor):
    archive = tmp_path/'data.zip'
    folder = 'datasets/seed_7/rich'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name in ('calibration.json', 'frames.csv', 'imu.csv', 'groundtruth/camera.tum'):
            bundle.writestr(f'{folder}/{name}', 'sample')
        bundle.writestr('datasets/shared/frame.png', b'pixels')
    manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'sequences': [{'path': folder}]}
    destination = tmp_path/'inputs'
    extractor(archive, manifest, destination)
    assert (destination/folder/'../../shared/frame.png').read_bytes() == b'pixels'
    with pytest.raises(FileExistsError):
        extractor(archive, manifest, destination)


def test_rejects_lfs_pointer_checksum_mismatch_and_archive_traversal(tmp_path, extractor):
    pointer = tmp_path/'pointer.zip'
    pointer.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='git lfs pull'):
        extractor(pointer, {}, tmp_path/'out1')
    archive = tmp_path/'data.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('../escape', b'bad')
    with pytest.raises(ValueError, match='checksum'):
        extractor(archive, {'archive_sha256': 'incorrect'}, tmp_path/'out2')
    manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='Unsafe archive'):
        extractor(archive, manifest, tmp_path/'out3')
    assert not (tmp_path/'out3').exists()

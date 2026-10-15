"""Prepare pinned native/opt-in ESMFold2 qualification requests on DiamondHill."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys


def prepare(root):
    root = Path(root).absolute()
    sys.path.insert(0, str(root / 'python'))
    from fireprot_workers.esmfold2 import SOURCE_COMMIT, ASSET_HASHES, model_directory
    source = Path('/media/PM982/engramfold/folding_e2e_20260921/repos/esm')
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE_COMMIT
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD'], check=True)
    inventory = root / 'environment.json'
    inv = json.loads(inventory.read_text())
    assert len(inv['files']) > 100000
    assets = json.loads((root.parent / 'esmfold2-assets-20261015/verified-files.json').read_text())
    by_name = {Path(f['path']).name: f for f in assets['files']}
    pins = {}
    for name, digest in ASSET_HASHES.items():
        filename = ('model.safetensors.index.json' if name == 'model' else
                    'config.json' if name == 'model_config' else 'ccd.pkl' if name == 'ccd' else
                    f'model-{name[-5:]}-of-00006.safetensors')
        file = by_name[filename]
        assert file['sha256'] == digest, name
        pins[name] = dict(path=file['path'], sha256=digest)

    def pin(path):
        return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def save(name, value):
        path = root / name
        with path.open('x') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
        return pin(path)

    profile = dict(python='/media/PM982/engramfold/folding_e2e_20260921/venvs/esmfold2-amd/bin/python',
        workers=str(root / 'python'), hip_probe='/media/IntelSSD/fireprot/runtime/fireprot-hip-probe',
        amd_smi='/usr/bin/amd-smi', source_manifest=None, environment_inventory=pin(inventory),
        environment=dict(ESMCFOLD_CCD_PATH=pins['ccd']['path'], PYTHONHASHSEED='101',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', FIREPROT_ESMFOLD2_PROFILE='1',
            FIREPROT_BRIDGE_LIBRARY=str(root / 'target/release/lib_fireprot_bridge.so')))
    profile_pin = save('qualification-profile.json', profile)
    requests = []
    for arm in ('native', 'fireprot'):
        run_tag = root.name.removeprefix('esmfold2-integration-')
        tag = f'esmfold2-n600-{arm}-qualification-{run_tag}'
        input_path = Path('/media/IntelSSD/fireprot/benchmarks/protenix/inputs/umem_L01000_prefix600.json')
        data = json.loads(input_path.read_text())
        assert len(data[0]['sequences'][0]['proteinChain']['sequence']) == 600
        request = dict(schema_version=1, run_id=tag, backend=arm, seed=101, hip_device=7,
            workload=dict(model='esmfold2', model_name='biohub/ESMFold2', loops=3,
                sampling_steps=50, samples=1, precision='native', chunk_size=64, deterministic=True),
            source=dict(root=str(source), commit=SOURCE_COMMIT), checkpoints=dict(model=pins['model']),
            resources={k:v for k,v in pins.items() if k != 'model'}, input=pin(input_path),
            output_dir=f'/media/IntelSSD/fireprot/outputs/{tag}')
        assert not Path(request['output_dir']).exists()
        model_directory(request)
        request_pin = save(arm + '-request.json', request)
        requests.append(save(arm + '-spec.json', dict(schema_version=1, timeout_seconds=3600,
            request=request_pin, profile=profile_pin)))
    save('preparation.json', dict(schema_version=1, cli=pin(root / 'target/release/fireprot'),
        bridge=pin(root / 'target/release/lib_fireprot_bridge.so'), profile=profile_pin, specs=requests,
        assets=pin(root.parent / 'esmfold2-assets-20261015/verified-files.json'),
        screen=pin(root / 'triangle-screen.json'),
        scope='First monomer qualification only. Includes bundled ESMC; native precision, loops+1 passes, actual denoiser calls recorded separately from requested schedule size. No E2E claim before acceptance.'))


if __name__ == '__main__':
    prepare(sys.argv[1])

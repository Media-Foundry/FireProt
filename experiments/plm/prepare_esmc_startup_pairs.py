"""Measure CPU module imports overlapping the unchanged Rust ready verification."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


root = Path('/media/IntelSSD/fireprot')
work = Path(sys.argv[1]).resolve()
reference = sys.argv[2] if len(sys.argv) > 2 else 'current'
assert reference in ('current', 'native')
work.mkdir(exist_ok=False)
shutil.copy2(__file__, work/'prepare-measured.py')
original = root/'outputs/experiments/esmc-load-20261016/validation/qualified-loading'
deployment = work/'qualified-import-overlap'
shutil.copytree(original, deployment)
transport = deployment/'python/fireprot_workers/transport.py'
source = transport.read_text()
assert source.count('    def ready(self, torch):') == 1
source = source.replace('    def ready(self, torch):', '    def ready(self, torch, prepare_cpu=None):')
needle = "        with self.socket.makefile('rb') as reader:"
assert source.count(needle) == 1
source = source.replace(needle, '        if prepare_cpu is not None:\n            prepare_cpu()\n'+needle)
transport.write_text(source)
main = deployment/'python/fireprot_workers/__main__.py'
source = main.read_text()
needle = '        channel.ready(torch)'
assert source.count(needle) == 1
source = source.replace(needle, '''        prepare_cpu = None
        mode = os.environ.get('FIREPROT_ESMC_IMPORT_OVERLAP', '0')
        if mode not in ('0', '1'):
            raise ValueError('FIREPROT_ESMC_IMPORT_OVERLAP must be 0 or 1')
        if mode == '1':
            if request['workload']['model'] != 'plm' or request['workload']['model_name'] != 'esmc-fold2-6b':
                raise ValueError('Import overlap is qualified only for standalone ESMC')
            def prepare_cpu():
                import esm
                from .plm import ESM_COMMIT
                if request['source']['commit'] != ESM_COMMIT or Path(esm.__file__).resolve().parent.parent != Path(request['source']['root']).resolve():
                    raise ValueError('Biohub ESM imported source differs from its pin')
                # Import definitions only. Model construction and weights wait for start.
                from esm.models.esmc.model import EsmcModel
        channel.ready(torch, prepare_cpu=prepare_cpu)''')
main.write_text(source)
for path in (main, transport):
    compile(path.read_text(), str(path), 'exec')
write(work/'deployment-manifest.json', dict(original=str(original), files={
    str(p.relative_to(deployment)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(deployment.rglob('*')) if p.is_file()}))
request = json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
profiles = {}
for arm, flag in (('baseline', '0'), ('candidate', '1')):
    profile = json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-pinned-loading.json').read_text())
    profile['workers'] = str(deployment/'python')
    profile['environment']['FIREPROT_BRIDGE_LIBRARY'] = str(deployment/'target/release/lib_fireprot_bridge.so')
    profile['environment']['FIREPROT_ESMC_IMPORT_OVERLAP'] = flag
    profiles[arm] = write(work/f'{arm}-profile.json', profile)
pairs = []
for index in range(3):
    pair = dict(first='candidate' if index == 1 else 'baseline')
    for arm in ('baseline', 'candidate'):
        backend = 'native' if reference == 'native' and arm == 'baseline' else 'fireprot'
        value = dict(request, backend=backend, run_id=f'esmc-import-{reference}-{arm}-{index+1}',
                     output_dir=str(work/f'{arm}-{index+1}'), hip_device=6)
        pair[arm] = dict(request=write(work/f'{arm}-{index+1}.json', value), profile=profiles[arm])
    pairs.append(pair)
spec = dict(schema_version=1,
    execution=dict(schema_version=1, timeout_seconds=3600, invocation_timeout_seconds=600, pairs=pairs),
    comparison=dict(scope='end_to_end',
        cache_state='Fresh workers; caches not flushed; original checkpoint verification; existing HIP binaries.',
        treatment_description=f'ESMC N1000 x4/B4, native BF16, seed101, HIP6. Reference {reference}. Candidate uses qualified exact operators, pinned loading, and imports unchanged ESMC definitions after ready emission, overlapping Rust verification; construction and checkpoint loading still wait for start.',
        expected_changes={
            'launch.environment.FIREPROT_ESMC_IMPORT_OVERLAP': dict(baseline='0', candidate='1'),
            'invocation.profile_sha256': dict(baseline=profiles['baseline']['sha256'], candidate=profiles['candidate']['sha256'])}))
if reference == 'native':
    spec['comparison']['expected_changes']['backend'] = dict(baseline='native', candidate='fireprot')
write(work/'spec.json', spec)
env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
for name in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
    env.pop(name, None)
with (work/'pairs.log').open('x') as log:
    subprocess.run([str(deployment/'target/release/fireprot'), 'benchmark-pairs', str(work/'spec.json'), str(work/'measurements')],
                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Import overlap pairs complete', flush=True)

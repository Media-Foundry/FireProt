"""Measure full-file SHA256 verification with four versus eight Rust workers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
    return pin(path)


root = Path('/media/IntelSSD/fireprot')
work = Path(sys.argv[1]).resolve()
reference = sys.argv[2] if len(sys.argv) > 2 else 'current'
assert reference in ('current', 'native')
mode = sys.argv[3] if len(sys.argv) > 3 else 'hash'
assert mode in ('hash', 'affinity', 'combined')
original = root/'outputs/experiments/esmc-startup-20261016/native/qualified-import-overlap'
deployment = work/'qualified-hash8'
shutil.copytree(original, deployment)
shutil.copy2(__file__, work/'prepare-measured.py')
binaries = json.loads((work/'binaries.json').read_text())
for name, digest in binaries.items():
    source = work/name
    assert pin(source)['sha256'] == digest
    shutil.copy2(source, deployment/'target/release'/name)
write(work/'deployment-manifest.json', dict(original=str(original), files={
    str(p.relative_to(deployment)): pin(p)['sha256'] for p in sorted(deployment.rglob('*')) if p.is_file()}))
request = json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
profiles = {}
settings = {}
for arm in ('baseline', 'candidate'):
    threads = '8' if arm == 'candidate' or mode == 'affinity' else '4'
    profile = json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-startup.json').read_text())
    profile['workers'] = str(deployment/'python')
    profile['environment']['FIREPROT_BRIDGE_LIBRARY'] = str(deployment/'target/release/lib_fireprot_bridge.so')
    profile['environment']['FIREPROT_PREFLIGHT_HASH_THREADS'] = threads
    profile['environment']['FIREPROT_WORKER_CPU_AFFINITY'] = '64-127,192-255' if arm == 'candidate' and mode != 'hash' else 'inherit'
    if reference == 'native' and arm == 'baseline':
        profile['environment']['FIREPROT_ESMC_IMPORT_OVERLAP'] = '0'
        profile['environment']['FIREPROT_ESMC_CPU_LOADING'] = '0'
    settings[arm] = profile['environment']
    profiles[arm] = write(work/f'{arm}-profile.json', profile)
pairs = []
for index in range(3):
    pair = dict(first='candidate' if index == 1 else 'baseline')
    for arm in ('baseline', 'candidate'):
        backend = 'native' if reference == 'native' and arm == 'baseline' else 'fireprot'
        value = dict(request, backend=backend, run_id=f'esmc-hash8-{reference}-{arm}-{index+1}', hip_device=6,
                     output_dir=str(work/f'{arm}-{index+1}'))
        pair[arm] = dict(request=write(work/f'{arm}-{index+1}.json', value), profile=profiles[arm])
    pairs.append(pair)
changes = {'invocation.profile_sha256': dict(baseline=profiles['baseline']['sha256'], candidate=profiles['candidate']['sha256'])}
for name in settings['baseline']:
    if settings['baseline'][name] != settings['candidate'][name]:
        changes['launch.environment.'+name] = dict(baseline=settings['baseline'][name], candidate=settings['candidate'][name])
spec = dict(schema_version=1, execution=dict(schema_version=1, timeout_seconds=3600, invocation_timeout_seconds=600, pairs=pairs),
    comparison=dict(scope='end_to_end', cache_state='Fresh workers; caches not flushed; original full checkpoint hashes and all exports retained.',
        treatment_description=f'ESMC N1000 x4/B4, native BF16, seed101, HIP6. Reference {reference}; experiment {mode}. Explicit environment differences record hash workers and Rust-applied worker CPU affinity. Candidate retains exact HIP operators, pinned loading and import overlap. Both arms use the same new Rust binary. No converted checkpoint cache, skipped assets or omitted output work.',
        expected_changes=changes))
if reference == 'native':
    spec['comparison']['expected_changes']['backend'] = dict(baseline='native', candidate='fireprot')
write(work/'spec.json', spec)
env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
for name in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
    env.pop(name, None)
with (work/'pairs.log').open('x') as log:
    subprocess.run([str(deployment/'target/release/fireprot'), 'benchmark-pairs', str(work/'spec.json'), str(work/'measurements')],
                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Hash8 pairs complete:', reference, flush=True)

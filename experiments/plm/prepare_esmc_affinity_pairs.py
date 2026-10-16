"""Compare current ESMC scheduling with CPU affinity local to HIP6."""
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
original = root/'outputs/experiments/esmc-startup-20261016/native/qualified-import-overlap'
deployment = work/'qualified-affinity'
shutil.copytree(original, deployment)
shutil.copy2(__file__, work/'prepare-measured.py')
shutil.copy2(work/'esmc_cpu_affinity.py', deployment/'python/fireprot_workers/cpu_affinity.py')
main = deployment/'python/fireprot_workers/__main__.py'
s = main.read_text()
assert s.count('        import torch\n') == 1
s = s.replace('        import torch\n', '        from .cpu_affinity import begin, instrument_loading, finish\n        cpu_diagnostic = begin(request)\n        import torch\n')
s = s.replace('        bundle = predict(request, channel)', '        instrument_loading(cpu_diagnostic)\n        bundle = predict(request, channel)\n        finish(cpu_diagnostic)')
main.write_text(s)
transport = deployment/'python/fireprot_workers/transport.py'
s = transport.read_text()
needle = "('operator-runtime.json', 'atlasfold-model.json')"
assert s.count(needle) == 1
transport.write_text(s.replace(needle, "('operator-runtime.json', 'atlasfold-model.json', 'cpu-placement.json')"))
for path in (main, transport, deployment/'python/fireprot_workers/cpu_affinity.py'):
    compile(path.read_text(), str(path), 'exec')
write(work/'deployment-manifest.json', dict(original=str(original), files={
    str(p.relative_to(deployment)): pin(p)['sha256'] for p in sorted(deployment.rglob('*')) if p.is_file()}))
request = json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
profiles = {}
for arm, flag in (('baseline', '0'), ('candidate', '1')):
    profile = json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-startup.json').read_text())
    profile['workers'] = str(deployment/'python')
    profile['environment']['FIREPROT_BRIDGE_LIBRARY'] = str(deployment/'target/release/lib_fireprot_bridge.so')
    profile['environment']['FIREPROT_ESMC_CPU_LOCAL'] = flag
    profiles[arm] = write(work/f'{arm}-profile.json', profile)
pairs = []
for index in range(3):
    pair = dict(first='candidate' if index == 1 else 'baseline')
    for arm in ('baseline', 'candidate'):
        value = dict(request, run_id=f'esmc-affinity-{arm}-{index+1}', hip_device=6,
                     output_dir=str(work/f'{arm}-{index+1}'))
        pair[arm] = dict(request=write(work/f'{arm}-{index+1}.json', value), profile=profiles[arm])
    pairs.append(pair)
spec = dict(schema_version=1, execution=dict(schema_version=1, timeout_seconds=3600, invocation_timeout_seconds=600, pairs=pairs),
    comparison=dict(scope='end_to_end', cache_state='Fresh workers; caches not flushed; original checkpoint hashes and all exports retained.',
        treatment_description='ESMC N1000 x4/B4, native BF16, seed101, HIP6. Both arms retain pinned loading and import overlap. Candidate restricts this worker before Torch import to GPU-local CPUs64-127,192-255. Parent Rust CPU affinity and memory policy are unchanged. Sealed diagnostic records actual CPU masks and checkpoint placement time.',
        expected_changes={
            'launch.environment.FIREPROT_ESMC_CPU_LOCAL': dict(baseline='0', candidate='1'),
            'invocation.profile_sha256': dict(baseline=profiles['baseline']['sha256'], candidate=profiles['candidate']['sha256'])}))
write(work/'spec.json', spec)
env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
for name in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
    env.pop(name, None)
with (work/'pairs.log').open('x') as log:
    subprocess.run([str(deployment/'target/release/fireprot'), 'benchmark-pairs', str(work/'spec.json'), str(work/'measurements')],
                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print('CPU affinity pairs complete', flush=True)

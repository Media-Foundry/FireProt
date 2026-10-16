"""Collect immutable diagnostic pins and compact ESM qualification evidence."""
import hashlib
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
destination=Path(sys.argv[2])

def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

models={}
for name in ('esm3-open','esmc-fold2-6b','esm2-3b'):
    profile_path=root/f'{name}-profile/report.json'
    profile=json.loads(profile_path.read_text())
    warm_path=root/f'{name}-packaged/report.json'
    warm=json.loads(warm_path.read_text())
    assert [c['batch_size'] for c in warm['cases']]==[1,2,4]
    assert all(c['all_arrays_bitwise_equal'] for c in warm['cases'])
    for c in warm['cases']:
        assert len(c['pairs'])==3
        assert all(len(p[arm])==3 for p in c['pairs'] for arm in ('native','candidate'))
    blas_path=root/f'{name}-blas/report.json'
    blas=json.loads(blas_path.read_text())
    models[name]=dict(profile=pin(profile_path),profile_aten_rows=[dict(batch=c['batch'],rows=[x for x in c['rows'] if x['key'].startswith('aten::')][:12]) for c in profile['cases']],
        warm_artifact=pin(warm_path),warm=warm,blas_artifact=pin(blas_path),blas=blas)
    if name!='esm3-open':
        inc_path=root/f'{name}-increment/report.json'
        inc=json.loads(inc_path.read_text())
        assert len(inc['cases'])==1 and inc['cases'][0]['batch_size']==4
        assert inc['cases'][0]['all_arrays_bitwise_equal']
        models[name]['increment_artifact']=pin(inc_path)
        models[name]['increment']=inc

loaders={}
for prefix,modes in [('esm2',('cpu_float','gpu_float','combined')),
                     ('esmc',('cpu_layout','gpu_layout','gpu_bf16_layout','streamed_gpu_bf16'))]:
    for mode in modes:
        path=root/f'{prefix}-transfer-{mode}/report.json'
        value=json.loads(path.read_text())
        assert all(value['exact'].values())
        loaders[f'{prefix}-{mode}']=dict(artifact=pin(path),result=value)
build_path=root/'trials-v2/build-receipt.json'
build=json.loads(build_path.read_text())
library=Path(build['directory'])/next(iter(build['receipt']['artifacts']))
assert pin(library)['sha256']==next(iter(build['receipt']['artifacts'].values()))
deployment=root/'qualified-v2'
sources={str(p.relative_to(deployment)):pin(p) for p in [
    deployment/'target/release/fireprot',deployment/'target/release/lib_fireprot_bridge.so',
    deployment/'python/fireprot/adapters/plm/esm.py',deployment/'python/fireprot/adapters/plm/runtime.py',
    deployment/'python/fireprot_workers/plm.py',deployment/'python/fireprot_workers/plm_loading.py',
    deployment/'kernels/amd/fireprot_esm.hip']}
diagnostic_scripts=[p for p in root.glob('*.py')]+[p for p in (root/'trials-v2').glob('*.py')]
report=dict(schema_version=1,hip_device=6,architecture='gfx90a',seed=101,length=1000,
    torch='2.12.0a0+git78258b9',hip='7.14.60850',precision='native',
    boundary='Warm diagnostics and single-observation loading screens; complete invocation results are separate.',
    cache_state='unflushed; warm operators use two warmups per arm per pair; no output cache',
    inputs='Distinct deterministic N1000 development-sequence rotations, not an independent biological test set.',
    models=models,loaders=loaders,build=pin(build_path),library=pin(library),deployment=sources,
    scripts=[pin(p) for p in sorted(diagnostic_scripts)])
with destination.open('x') as f:json.dump(report,f,indent=2);f.write('\n')

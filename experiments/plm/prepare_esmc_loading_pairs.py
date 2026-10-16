"""Freeze the bounded pinned loader and run current/native Rust E2E comparisons."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def pin(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return pin(path)


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve()
    original=root/'outputs/experiments/esm-speed-20261016/qualified-v2'
    deployment=work/'qualified-loading';shutil.copytree(original,deployment)
    target=deployment/'python/fireprot_workers/esmc_cpu_loading.py';assert not target.exists()
    shutil.copy2(work/'esmc_cpu_loading.py',target)
    loader=deployment/'python/fireprot_workers/plm_loading.py'
    source=loader.read_text();before=hashlib.sha256(source.encode()).hexdigest()
    needle='def streamed_esmc_state(root, mapping, published_to_native):'
    assert source.count(needle)==1
    source=source.replace(needle,'def _gpu_streamed_esmc_state(root, mapping, published_to_native):')
    source+='''\n\ndef streamed_esmc_state(root, mapping, published_to_native):
    import os
    mode = os.environ.get('FIREPROT_ESMC_CPU_LOADING', '0')
    if mode == '0':
        return _gpu_streamed_esmc_state(root, mapping, published_to_native)
    if mode != '1':
        raise ValueError('FIREPROT_ESMC_CPU_LOADING must be 0 or 1')
    from .esmc_cpu_loading import reused_pinned_esmc_state
    return reused_pinned_esmc_state(root, mapping, published_to_native)
'''
    loader.write_text(source)
    files=sorted(p for p in deployment.rglob('*') if p.is_file())
    write(work/'deployment-manifest.json',dict(original=str(original),loader_before_sha256=before,
        files={str(p.relative_to(deployment)):pin(p)['sha256'] for p in files}))
    cases=[]
    for reference in ('current','native'):
        case=work/f'{reference}-pairs';case.mkdir(exist_ok=False)
        request=json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
        profiles={}
        for arm,flag in (('baseline','0'),('candidate','1')):
            profile=json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-gpu-loading.json').read_text())
            profile['workers']=str(deployment/'python')
            profile['environment']['FIREPROT_BRIDGE_LIBRARY']=str(deployment/'target/release/lib_fireprot_bridge.so')
            profile['environment']['FIREPROT_ESMC_CPU_LOADING']=flag
            profiles[arm]=write(case/f'{arm}-profile.json',profile)
        pairs=[]
        for index in range(3):
            pair=dict(first='candidate' if index==1 else 'baseline')
            for arm in ('baseline','candidate'):
                backend='native' if arm=='baseline' and reference=='native' else 'fireprot'
                value=dict(request,backend=backend,run_id=f'esmc-load-{reference}-{arm}-{index+1}',
                           output_dir=str(case/f'{arm}-{index+1}'),hip_device=6)
                pair[arm]=dict(request=write(case/f'{arm}-{index+1}.json',value),profile=profiles[arm])
            pairs.append(pair)
        changes={'launch.environment.FIREPROT_ESMC_CPU_LOADING':dict(baseline='0',candidate='1'),
                 'invocation.profile_sha256':dict(baseline=profiles['baseline']['sha256'],candidate=profiles['candidate']['sha256'])}
        if reference=='native':changes['backend']=dict(baseline='native',candidate='fireprot')
        spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=3600,invocation_timeout_seconds=600,pairs=pairs),
            comparison=dict(scope='end_to_end',cache_state='Fresh workers; caches not flushed; no converted checkpoint or output cache; existing HIP binaries.',
                treatment_description=f'ESMC N1000 x4, B4, seed101, native BF16. Reference {reference}; candidate exact HIP operators plus bounded CPU BF16 conversion directly into two reusable pinned blocks and async H2D. Full checkpoint verification and all output exports retained.',
                expected_changes=changes))
        write(case/'spec.json',spec);cases.append(case)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for name in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(name,None)
    for case in cases:
        with (case/'pairs.log').open('x') as log:
            subprocess.run([str(deployment/'target/release/fireprot'),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],
                           cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(case.name,'complete',flush=True)


if __name__=='__main__':main()

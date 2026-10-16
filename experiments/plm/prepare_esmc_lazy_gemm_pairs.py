"""Measure lazy FFN plan creation after the original first QKV projection."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return pin(path)


def main():
    root=Path('/media/IntelSSD/fireprot')
    work=Path(sys.argv[1]).resolve()
    original=root/'outputs/experiments/esmc-startup-20261016/native/qualified-import-overlap'
    deployment=work/'qualified-fixed'
    # Rust inventories root shim, worker trees and AMD source trees as well as
    # the binaries. Copy the complete small frozen deployment, without weights.
    shutil.copytree(original,deployment)
    shutil.copy2(__file__,work/'prepare-measured.py')
    module=deployment/'python/fireprot/adapters/plm/gemm_fixed.py'
    assert not module.exists()
    shutil.copy2(work/'esmc_lt_lazy_scope.py',module)
    runtime=deployment/'python/fireprot/adapters/plm/runtime.py'
    text=runtime.read_text();before=hashlib.sha256(text.encode()).hexdigest()
    needle='        with esm_scope(model, request, batch_size) as state:\n            yield state\n'
    assert text.count(needle)==1
    text=text.replace(needle,
        '        from .gemm_fixed import fixed_gemm_scope\n'
        '        with fixed_gemm_scope(request, batch_size) as gemm, esm_scope(model, request, batch_size) as state:\n'
        '            if gemm is not None: state["fixed_gemm"] = gemm\n'
        '            yield state\n')
    runtime.write_text(text)
    loader=deployment/'python/fireprot_workers/plm.py'
    text=loader.read_text();loader_before=hashlib.sha256(text.encode()).hexdigest()
    needle="if k not in ('hip_rope', 'hip_batch', 'hip_esm')"
    assert text.count(needle)==1
    # Keep the upstream asset set strict; the separately pinned GEMM shared library
    # is a framework resource, not an ESMFold2 checkpoint shard.
    loader.write_text(text.replace(needle,"if k not in ('hip_rope', 'hip_batch', 'hip_esm', 'hip_esmc_lt')"))
    files=sorted(p for p in deployment.rglob('*') if p.is_file())
    write(work/'deployment-manifest.json',dict(original=str(original),runtime_before_sha256=before,loader_before_sha256=loader_before,
          files={str(p.relative_to(deployment)):pin(p)['sha256'] for p in files}))
    receipt=json.loads((root/'outputs/experiments/esmc-gemm-order-20261016/lt-build-receipt.json').read_text())
    table=Path(receipt['directory'])/next(iter(receipt['receipt']['artifacts']))
    assert pin(table)['sha256']=='7b554428ba5a79ed1ef2f107f8e4d8ba8e306352a8efa88c0b116f04f1fdedb4'
    specs=[]
    for count in (4,):
        label=f'esmc-fixed-{count}';case=work/f'{label}-pairs';case.mkdir(exist_ok=False)
        request=json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
        request['resources']['hip_esmc_lt']=pin(table)
        if count==64:
            base=json.loads(Path(request['input']['path']).read_text())['sequences'][0]['sequence']
            seqs=[dict(id=f'item-{i}',sequence=base[(i*137)%1000:]+base[:(i*137)%1000]) for i in range(count)]
            request['input']=write(case/'input.json',dict(schema_version=1,sequences=seqs))
            request['workload']['batch']['items']=count
        profiles={}
        for arm,flag in (('baseline','0'),('candidate','1')):
            profile=json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-startup.json').read_text())
            profile['workers']=str(deployment/'python')
            profile['environment']['FIREPROT_BRIDGE_LIBRARY']=str(deployment/'target/release/lib_fireprot_bridge.so')
            profile['environment']['FIREPROT_ESMC_FIXED_GEMM']=flag
            profiles[arm]=write(case/f'{arm}-profile.json',profile)
        pairs=[]
        for index in range(3):
            pair=dict(first='candidate' if index==1 else 'baseline')
            for arm in ('baseline','candidate'):
                value=dict(request,run_id=f'{label}-{arm}-{index+1}',hip_device=6,
                           output_dir=str(case/f'{arm}-{index+1}'))
                pair[arm]=dict(request=write(case/f'{arm}-{index+1}.json',value),profile=profiles[arm])
            pairs.append(pair)
        spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=3600,invocation_timeout_seconds=600,pairs=pairs),
            comparison=dict(scope='end_to_end',cache_state='Fresh processes; resident-file/GPU caches not flushed; no output cache; published HIP library already built.',
                treatment_description=f'Incremental lazy direct hipBLASLt FFN up/gate composition; plan created at first FFN after native QKV; current pinned loading and import overlap retained. ESMC N1000 x {count}, B4, seed101, native BF16. Same worker, resources, full hash preflight and exported outputs.',
                expected_changes={
                    'launch.environment.FIREPROT_ESMC_FIXED_GEMM':dict(baseline='0',candidate='1'),
                    'invocation.profile_sha256':dict(baseline=profiles['baseline']['sha256'],candidate=profiles['candidate']['sha256'])}))
        write(case/'spec.json',spec);specs.append(case)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for name in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(name,None)
    for case in specs:
        with (case/'pairs.log').open('x') as log:
            subprocess.run([str(deployment/'target/release/fireprot'),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],
                           cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert pin(table)['sha256']=='7b554428ba5a79ed1ef2f107f8e4d8ba8e306352a8efa88c0b116f04f1fdedb4'
        print(case.name,'complete',flush=True)


if __name__=='__main__':main()

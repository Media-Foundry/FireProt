"""Create an isolated frozen worker overlay and Rust incremental E2E pairs."""
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
    original=root/'outputs/experiments/esm-speed-20261016/qualified-v2'
    deployment=work/'qualified-tuned'
    # Rust inventories root shim, worker trees and AMD source trees as well as
    # the binaries. Copy the complete small frozen deployment, without weights.
    shutil.copytree(original,deployment)
    module=deployment/'python/fireprot/adapters/plm/gemm_tuned.py'
    assert not module.exists()
    shutil.copy2(work/'esmc_tuned_scope.py',module)
    runtime=deployment/'python/fireprot/adapters/plm/runtime.py'
    text=runtime.read_text();before=hashlib.sha256(text.encode()).hexdigest()
    needle='        with esm_scope(model, request, batch_size) as state:\n            yield state\n'
    assert text.count(needle)==1
    text=text.replace(needle,
        '        from .gemm_tuned import tuned_gemm_scope\n'
        '        with tuned_gemm_scope(request, batch_size) as gemm, esm_scope(model, request, batch_size) as state:\n'
        '            if gemm is not None: state["tuned_gemm"] = gemm\n'
        '            yield state\n')
    runtime.write_text(text)
    loader=deployment/'python/fireprot_workers/plm.py'
    text=loader.read_text();loader_before=hashlib.sha256(text.encode()).hexdigest()
    needle="if k not in ('hip_rope', 'hip_batch', 'hip_esm')"
    assert text.count(needle)==1
    # Keep the upstream asset set strict; the separately pinned selector table
    # is a framework resource, not an ESMFold2 checkpoint shard.
    loader.write_text(text.replace(needle,"if k not in ('hip_rope', 'hip_batch', 'hip_esm', 'esmc_gemm_algorithms')"))
    files=sorted(p for p in deployment.rglob('*') if p.is_file())
    write(work/'deployment-manifest.json',dict(original=str(original),runtime_before_sha256=before,loader_before_sha256=loader_before,
          files={str(p.relative_to(deployment)):pin(p)['sha256'] for p in files}))
    table=work/'tunable-v3/algorithms.csv'
    assert pin(table)['sha256']=='9980c71947c81b3c759cdff4707376031126b0fa35045b7f0b3ece6345e3c060'
    specs=[]
    for count in (4,64):
        label=f'esmc-tuned-{count}';case=work/f'{label}-pairs';case.mkdir(exist_ok=False)
        request=json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
        request['resources']['esmc_gemm_algorithms']=pin(table)
        if count==64:
            base=json.loads(Path(request['input']['path']).read_text())['sequences'][0]['sequence']
            seqs=[dict(id=f'item-{i}',sequence=base[(i*137)%1000:]+base[:(i*137)%1000]) for i in range(count)]
            request['input']=write(case/'input.json',dict(schema_version=1,sequences=seqs))
            request['workload']['batch']['items']=count
        profiles={}
        for arm,flag in (('baseline','0'),('candidate','1')):
            profile=json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-exact.json').read_text())
            profile['workers']=str(deployment/'python')
            profile['environment']['FIREPROT_BRIDGE_LIBRARY']=str(deployment/'target/release/lib_fireprot_bridge.so')
            profile['environment']['FIREPROT_ESMC_TUNED_GEMM']=flag
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
            comparison=dict(scope='end_to_end',cache_state='Fresh processes; resident-file/GPU caches not flushed; no output cache; offline table already built.',
                treatment_description=f'Incremental frozen GEMM selection over current exact FireProt. ESMC N1000 x {count}, B4, seed101, native BF16. Same worker, resources, full hash preflight and exported outputs.',
                expected_changes={
                    'launch.environment.FIREPROT_ESMC_TUNED_GEMM':dict(baseline='0',candidate='1'),
                    'invocation.profile_sha256':dict(baseline=profiles['baseline']['sha256'],candidate=profiles['candidate']['sha256'])}))
        write(case/'spec.json',spec);specs.append(case)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for name in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(name,None)
    for case in specs:
        with (case/'pairs.log').open('x') as log:
            subprocess.run([str(deployment/'target/release/fireprot'),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],
                           cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert pin(table)['sha256']=='9980c71947c81b3c759cdff4707376031126b0fa35045b7f0b3ece6345e3c060'
        print(case.name,'complete',flush=True)


if __name__=='__main__':main()

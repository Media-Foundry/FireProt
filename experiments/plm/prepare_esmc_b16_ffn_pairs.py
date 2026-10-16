"""Freeze direct B16 FFN workers and run same-grouping Rust E2E pairs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from collect_esmc_pointwise_pairs import pin


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return pin(path)


def replace(path,needle,value):
    text=path.read_text();assert text.count(needle)==1,(path,needle);path.write_text(text.replace(needle,value))


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve()
    screen=json.loads((work/'direct-screen/report.json').read_text())
    assert screen['passed'] and screen['assets_rehashed_after'] and screen['all_80_projections_exact'] and screen['full_model_exact']
    library=screen['library'];assert pin(Path(library['path']))==library
    prior=root/'outputs/experiments/esmc-large-batch-e2e-20261016'
    oldprofile=json.loads((prior/'b16-profile.json').read_text());original=Path(oldprofile['workers']).parent
    deployment=work/'qualified-b16-ffn';shutil.copytree(original,deployment)
    target=deployment/'python/fireprot/adapters/plm/gemm_b16.py'
    source=Path(__file__).with_name('esmc_b16_lt_scope.py');assert not target.exists()
    text=source.read_text().replace('from esmc_lt_lazy_scope import FixedProjection','from .gemm_lazy import FixedProjection')
    text=text.replace("    resource=request['resources']['hip_esmc_b16_lt'];key=",f"    resource=request['resources']['hip_esmc_b16_lt']\n    if resource['sha256']!='{library['sha256']}':raise ValueError('Unqualified B16 FFN library')\n    key=")
    text=text.replace("        state['library']=resource;state['online_tuning']=False", "        state['library']=resource;state['online_tuning']=False;state['algorithm']=3903")
    target.write_text(text)
    large=deployment/'python/fireprot/adapters/plm/large_batch.py'
    needle="    with pointwise.scope(model,'residual') as residual,esm_scope(model,selected,batch_size) as state:"
    replace(large,needle,"    from .gemm_b16 import b16_gemm_scope\n    with b16_gemm_scope(request,batch_size) as gemm,pointwise.scope(model,'residual') as residual,esm_scope(model,selected,batch_size) as state:")
    replace(large,"        state['fixed_gemm_used']=False","        state['fixed_gemm_used']=gemm is not None\n        if gemm is not None:state['b16_fixed_gemm']=gemm")
    loader=deployment/'python/fireprot_workers/plm.py'
    replace(loader,"'hip_esmc_lt', 'hip_esmc_large_batch')","'hip_esmc_lt', 'hip_esmc_large_batch', 'hip_esmc_b16_lt')")
    for p in (target,large,loader):compile(p.read_text(),str(p),'exec')
    write(work/'deployment.json',dict(original=str(original),files={str(p.relative_to(deployment)):pin(p)['sha256'] for p in sorted(deployment.rglob('*')) if p.is_file()}))
    base=json.loads((prior/'pair-1-b16-request.json').read_text());base['resources']['hip_esmc_b16_lt']=library
    cases=[]
    for count in (16,64):
        case=work/f'b{count}-pairs';case.mkdir(exist_ok=False)
        request=json.loads(json.dumps(base));request['workload']['batch']['items']=count
        if count==64:request['input']=json.loads((root/'configs/requests/plm-esmc-fold2-6b-resident64-gemm.json').read_text())['input']
        profiles={}
        for arm,flag in (('baseline','0'),('candidate','1')):
            profile=json.loads(json.dumps(oldprofile));profile['workers']=str(deployment/'python')
            profile['environment']['FIREPROT_BRIDGE_LIBRARY']=str(deployment/'target/release/lib_fireprot_bridge.so')
            profile['environment']['FIREPROT_ESMC_B16_FFN']=flag
            profiles[arm]=write(case/f'{arm}-profile.json',profile)
        pairs=[]
        for index in range(3):
            pair=dict(first='candidate' if index==1 else 'baseline')
            for arm in ('baseline','candidate'):
                value=dict(request,run_id=f'esmc-b16-ffn-{count}-{arm}-{index+1}',output_dir=str(case/f'{arm}-{index+1}'))
                pair[arm]=dict(request=write(case/f'{arm}-{index+1}.json',value),profile=profiles[arm])
            pairs.append(pair)
        spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=3600,invocation_timeout_seconds=600,pairs=pairs),
            comparison=dict(scope='end_to_end',cache_state='Fresh workers; unflushed caches; offline fixed HIP library; all checkpoint hashes and exports retained.',
                treatment_description=f'ESMC N1000 x{count}, B16 groups, BF16, seed101, HIP6: native FFN versus lazy direct hipBLASLt3903; same HIP pointwise/RoPE, workers, CLI, assets, grouping and snapshot monitor.',
                expected_changes={'launch.environment.FIREPROT_ESMC_B16_FFN':dict(baseline='0',candidate='1'),
                                  'invocation.profile_sha256':dict(baseline=profiles['baseline']['sha256'],candidate=profiles['candidate']['sha256'])}))
        write(case/'spec.json',spec);cases.append(case)
        if count==64:
            reference=dict(request,backend='native',run_id='esmc-b16-native64-reference',output_dir=str(work/'native64'))
            write(work/'native64-request.json',reference)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for key in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):env.pop(key,None)
    cli=deployment/'target/release/fireprot';write(work/'benchmark-cli.json',pin(cli))
    # Produce the official B16 reference for all 64 distinct inputs before timing.
    with (work/'native64.log').open('x') as log:
        subprocess.run([str(cli),'esm',str(work/'native64-request.json'),str(work/'b64-pairs/baseline-profile.json')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Native B16 reference for64 inputs complete',flush=True)
    for case in cases:
        with (case/'pairs.log').open('x') as log:
            subprocess.run([str(cli),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert pin(Path(library['path']))==library
        print(case.name,'complete',flush=True)


if __name__=='__main__':main()

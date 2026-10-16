"""Compare ESMC resident B4/B16 grouping with per-group official acceptance.

Different grouping is not submitted to the identical-workload comparator.
Every invocation is measured/sealed by Rust, then checked against its own
frozen official same-grouping outputs. Cross-group drift remains separate.
"""
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import zipfile

from collect_esmc_pointwise_pairs import archive, checked, pin


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return pin(path)


def replace(path,needle,value):
    s=path.read_text();assert s.count(needle)==1,(path,needle);path.write_text(s.replace(needle,value))


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve();work.mkdir(exist_ok=False)
    ops=root/'outputs/experiments/esmc-large-batch-ops-20261016'
    qualification=json.loads((ops/'screen/report.json').read_text());assert qualification['passed'] and qualification['assets_rehashed_after']
    assert all(v['all_arrays_exact'] for v in qualification['cases'].values())
    library=json.loads((ops/'library.json').read_text());assert pin(Path(library['path']))==library
    profile=json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-resident-snapshot.json').read_text())
    original=Path(profile['workers']).parent;deployment=work/'qualified-large-batch';shutil.copytree(original,deployment)
    cli=deployment/'target/release/fireprot';shutil.copy2(root/'target/release/fireprot',cli)
    module=deployment/'python/fireprot/adapters/plm/large_batch.py';assert not module.exists()
    shutil.copy2(Path(__file__).with_name('esmc_large_batch_scope.py'),module)
    runtime=deployment/'python/fireprot/adapters/plm/runtime.py'
    needle="    name = request['workload']['model_name']\n"
    replace(runtime,needle,needle+'''    if batch_size in (8,16) and 'hip_esmc_large_batch' in request.get('resources',{}):
        from .large_batch import large_batch_scope
        with large_batch_scope(model,request,length,batch_size) as state:
            yield state
        return
''')
    worker=deployment/'python/fireprot_workers/plm_batch.py'
    needle="        if any(len(item['sequence']) != 1000 for item in plan['sequences']) or any(len(g) not in (1,2,4) for g in plan['groups']):"
    replace(worker,needle,"        sizes=(1,2,4,8,16) if request['workload']['model_name']=='esmc-fold2-6b' and 'hip_esmc_large_batch' in request.get('resources',{}) else (1,2,4)\n        if any(len(item['sequence']) != 1000 for item in plan['sequences']) or any(len(g) not in sizes for g in plan['groups']):")
    loader=deployment/'python/fireprot_workers/plm.py'
    needle="'hip_esmc_pointwise', 'hip_esmc_lt')";replace(loader,needle,"'hip_esmc_pointwise', 'hip_esmc_lt', 'hip_esmc_large_batch')")
    for p in (module,runtime,worker,loader):compile(p.read_text(),str(p),'exec')
    write(work/'deployment.json',dict(original=str(original),files={str(p.relative_to(deployment)):pin(p)['sha256'] for p in sorted(deployment.rglob('*')) if p.is_file()}))
    reference=root/'outputs/experiments/esmc-large-batch-20261016/screen'
    diagnostic=json.loads((reference/'report.json').read_text());assert diagnostic['passed'] and diagnostic['assets_rehashed_after']
    inputs=write(work/'input.json',dict(schema_version=1,sequences=diagnostic['inputs']))
    base=json.loads((root/'configs/requests/plm-esmc-fold2-6b-resident64-gemm.json').read_text())
    base['resources']['hip_esmc_large_batch']=library;base['input']=inputs
    profiles={}
    for size in (4,16):
        p=json.loads(json.dumps(profile));p['workers']=str(deployment/'python')
        p['environment']['FIREPROT_BRIDGE_LIBRARY']=str(deployment/'target/release/lib_fireprot_bridge.so')
        p['environment']['FIREPROT_ESMC_FIXED_GEMM']='1' if size==4 else '0'
        profiles[size]=write(work/f'b{size}-profile.json',p)
    report=dict(schema_version=1,model='ESMC bundled 6B',length=1000,items=16,dtype='native BF16',seed=101,hip_device=6,
        boundary='Complete Rust-measured invocations; same 16 inputs, one model load, B4 versus B16 grouping. Post-measurement reference verification excluded in both arms.',
        cache_state='Fresh workers; unflushed caches; offline-built HIP kernels; all checkpoint hashes and exports retained.',
        acceptance='Official same-grouping byte equality; no cross-group equality assertion or tolerance relaxation.',
        cli=pin(cli),input=inputs,profiles={str(k):v for k,v in profiles.items()},library=library,qualification=pin(ops/'screen/report.json'),
        grouping_drift=diagnostic['native_grouping_drift_vs_b4']['16'],pairs=[])
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for key in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):env.pop(key,None)
    for index,order in enumerate(((4,16),(16,4),(4,16))):
        pair=dict(order=list(order))
        for size in order:
            name=f'pair-{index+1}-b{size}';request=json.loads(json.dumps(base))
            request.update(run_id='esmc-large-'+name,output_dir=str(work/name))
            request['workload']['batch'].update(items=16,max_batch_size=size,max_padded_tokens=size*1002)
            req=write(work/f'{name}-request.json',request)
            spec=write(work/f'{name}-spec.json',dict(schema_version=1,timeout_seconds=300,request=req,profile=profiles[size]))
            measurement=work/f'{name}-measurement'
            with (work/f'{name}.log').open('x') as log:
                subprocess.run([str(cli),'measure-inference',spec['path'],str(measurement)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            invocation=checked(pin(measurement/'invocation.json'));assert invocation['exit_code']==0
            out=Path(request['output_dir']);assert pin(out/'accepted.json')==invocation['accepted']
            predictions,arrays=archive(out);assert len(arrays)==16
            assert predictions['batch_groups']==[list(range(i,i+size)) for i in range(0,16,size)]
            for i,actual in enumerate(arrays):
                with zipfile.ZipFile(reference/f'native-b{size}-item-{i}.npz') as ref:
                    assert actual=={k:ref.read(k) for k in ref.namelist()}
            supervisor=json.loads((out/'supervisor.json').read_text())
            assert supervisor['phase']=='succeeded' and supervisor['device_verified'] and supervisor['isolation_samples']>=2
            assert supervisor['physical_device']['pci_address']=='0000:8e:00.0'
            runtimes=json.loads((out/'operator-runtime.json').read_text());assert len(runtimes)==16//size
            for r in runtimes:
                assert r['calls']==160 and r['rope_calls']==80 and r['swiglu_calls']==80 and r['batch_size']==size
                assert r['pointwise']['residual_calls']==160
                if size==16:assert r['large_batch_prototype'] and not r['fixed_gemm_used'] and r['library']==library
                else:assert r['fixed_gemm']['calls']==80 and r['fixed_gemm']['algorithm']==4139
            pair[str(size)]=dict(invocation=pin(measurement/'invocation.json'),accepted=invocation['accepted'],
                e2e_seconds=invocation['elapsed_seconds'],model_seconds=predictions['model_seconds'],
                supervisor_seconds=supervisor['elapsed_seconds'],isolation_samples=supervisor['isolation_samples'],official_arrays_exact=True)
            print(name,pair[str(size)]['e2e_seconds'],flush=True)
        pair['speedup']=pair['4']['e2e_seconds']/pair['16']['e2e_seconds'];report['pairs'].append(pair)
        (work/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['medians']={field:{str(n):statistics.median(p[str(n)][field] for p in report['pairs']) for n in (4,16)}
        for field in ('e2e_seconds','model_seconds','supervisor_seconds','isolation_samples')}
    report['median_speedup']=report['medians']['e2e_seconds']['4']/report['medians']['e2e_seconds']['16']
    report['all_pairs_faster']=all(p['speedup']>1 for p in report['pairs']);report['passed']=True
    assert pin(cli)==report['cli'] and pin(Path(library['path']))==library
    write(work/'report.json',report);print(json.dumps(report['medians']),report['median_speedup'],flush=True)


if __name__=='__main__':main()

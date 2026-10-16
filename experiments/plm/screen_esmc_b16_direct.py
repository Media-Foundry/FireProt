"""Qualify the direct B16 FFN plan against native projections and model outputs."""
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time
import zipfile

from screen_esmc_large_batch import pin


def main():
    request_path,work=map(Path,sys.argv[1:3]);out=work/'direct-screen';out.mkdir(exist_ok=False)
    request=json.loads(request_path.read_text());selection=json.loads((work/'direct/selection.json').read_text())
    library=selection['library'];assert pin(library['path'])==library and selection['algorithm']==3903
    assets=list(request['checkpoints'].values())+list(request['resources'].values())+[request['input']]
    def verify(value):assert pin(value['path'])==value
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,assets))
    import torch
    from fireprot_workers.plm import load_model
    from fireprot_workers.plm_batch import prepare_forward
    from fireprot_workers.transport import save_arrays
    from fireprot.adapters.plm.runtime import optimized_scope
    from esmc_b16_lt_scope import B16Projection
    assert os.environ['HIP_VISIBLE_DEVICES']=='6' and not torch.cuda.tunable.is_enabled()
    torch.set_num_threads(8);torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sequences=[s['sequence'] for s in json.loads(Path(request['input']['path']).read_text())['sequences']]
    assert len(sequences)==16 and all(len(s)==1000 for s in sequences)
    report=dict(schema_version=1,library=library,algorithm=3903,request=pin(request_path),script=pin(__file__),
        scope=pin(Path(__file__).with_name('esmc_b16_lt_scope.py')),base_scope=pin(Path(__file__).with_name('esmc_lt_lazy_scope.py')),
        boundary='Warm B16 forward; lazy direct FFN; no online tuning; not E2E.',seed=101,hip_device=6,dtype='native BF16',pairs=[])
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def exact(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    with torch.inference_mode():
        model,_,_=load_model(request,sequences[0]);forward=prepare_forward(model,'esmc-fold2-6b',sequences)
        expected={k:v.clone() for k,v in forward().items()}
        projection=B16Projection(library['path'])
        def run(enabled,audit=False):
            with optimized_scope(model,request,1000,batch_size=16):
                if not enabled:return forward(),None
                with projection.scope(check_each=audit) as state:return forward(),state
        try:
            actual,state=run(True,True);assert state['calls']==80
            assert all(exact(v,actual[k]) for k,v in expected.items())
            report.update(all_80_projections_exact=True,full_model_exact=True,initial_plan=state);save()
            for i in range(16):
                path=out/f'direct-{i}.npz';save_arrays(path,{k:v[i:i+1] for k,v in actual.items()})
                ref=work/'screen'/f'native-{i}.npz'
                with zipfile.ZipFile(path) as a,zipfile.ZipFile(ref) as b:
                    assert set(a.namelist())==set(b.namelist()) and all(a.read(k)==b.read(k) for k in a.namelist())
            # Exercise explicit unsupported-shape rejection without launching kernels.
            lib=projection.lib;bad=[]
            for rows in (0,1002,4008,8016,16031,16033):
                plan=ctypes.c_void_p();size=ctypes.c_size_t();rc=lib.esmc_lt_create(rows,ctypes.byref(plan),ctypes.byref(size))
                assert rc!=0 and not plan.value;bad.append(dict(rows=rows,status=rc))
            report['unsupported_rows']=bad
            for order in ((False,True),(True,False),(False,True)):
                pair={}
                for enabled in order:
                    for _ in range(2):run(enabled)
                    times=[]
                    for _ in range(3):
                        torch.cuda.synchronize();begin=time.perf_counter();actual,state=run(enabled);torch.cuda.synchronize()
                        times.append(time.perf_counter()-begin);assert all(exact(v,actual[k]) for k,v in expected.items())
                        if enabled:assert not state['plan_created'] and state['calls']==80
                    pair['candidate' if enabled else 'baseline']=times
                report['pairs'].append(pair);save();print('Direct pair',pair,flush=True)
            report['medians']={a:statistics.median(t for p in report['pairs'] for t in p[a]) for a in ('baseline','candidate')}
            report['speedup']=report['medians']['baseline']/report['medians']['candidate'];report['passed']=True;save()
        finally:torch.cuda.synchronize();projection.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,assets))
    assert pin(library['path'])==library;report['assets_rehashed_after']=True;save()
    print(json.dumps(dict(medians=report['medians'],speedup=report['speedup'],initial_plan=report['initial_plan']),indent=2),flush=True)


if __name__=='__main__':main()

"""Offline B16 FFN algorithm screen with exact all-layer and output gates."""
from contextlib import contextmanager
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import sys
import time
import zipfile

from screen_esmc_large_batch import pin


def main():
    assert os.environ['HIP_VISIBLE_DEVICES']=='6'
    request_path,output=map(Path,sys.argv[1:3]);output.mkdir(exist_ok=False)
    request=json.loads(request_path.read_text());assert request['workload']['batch']['max_batch_size']==16
    assets=list(request['checkpoints'].values())+list(request['resources'].values())+[request['input']]
    def verify(value):assert pin(value['path'])==value
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,assets))
    import torch
    import torch.nn.functional as F
    import torch.cuda.tunable as tunable
    from fireprot_workers.plm import load_model
    from fireprot_workers.plm_batch import prepare_forward
    from fireprot_workers.transport import save_arrays
    from fireprot.adapters.plm.runtime import optimized_scope
    torch.set_num_threads(8);torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert str(torch.backends.cuda.preferred_blas_library()).endswith('Cublaslt')
    tunable.enable(False);tunable.tuning_enable(False)
    tunable.set_filename(str(output/'algorithms.csv'))
    tunable.set_numerical_check_tolerances(True,atol=1e-45,rtol=1e-45)
    tunable.set_max_tuning_duration(10);tunable.set_max_tuning_iterations(20)
    items=json.loads(Path(request['input']['path']).read_text())['sequences'];assert len(items)==16
    report=dict(schema_version=1,model='ESMC bundled 6B',length=1000,batch=16,seed=101,dtype='native BF16',hip_device=6,
        request=pin(request_path),assets=assets,script=pin(__file__),torch=str(torch.__version__),hip=str(torch.version.hip),
        boundary='Warm B16 FFN and model forward; offline tuning excluded; not E2E.',
        cache_state='Resident weights; two warmups per arm; unflushed caches; AB/BA/AB, three observations per arm per pair.',
        internal_tolerances=dict(atol=1e-45,rtol=1e-45),acceptance='Independent raw-byte equality, including signed zero.')
    def save(): (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    linear=F.linear
    try:
        with torch.inference_mode():
            model,_,_=load_model(request,items[0]['sequence'])
            forward=prepare_forward(model,'esmc-fold2-6b',[s['sequence'] for s in items])
            captured=[]
            def capture(x,w,b=None):
                if w is model.transformer.blocks[0].ffn.fc1_weight:captured.append((x.clone(),w,b))
                return linear(x,w,b)
            F.linear=capture
            try:expected={k:v.clone() for k,v in forward().items()}
            finally:F.linear=linear
            reference_root=Path('/media/IntelSSD/fireprot/outputs/experiments/esmc-large-batch-20261016/screen')
            for i in range(16):
                path=output/f'native-{i}.npz';save_arrays(path,{k:v[i:i+1] for k,v in expected.items()})
                with zipfile.ZipFile(path) as a,zipfile.ZipFile(reference_root/f'native-b16-item-{i}.npz') as b:
                    assert set(a.namelist())==set(b.namelist()) and all(a.read(k)==b.read(k) for k in a.namelist())
            report['native_reference_exact']=True;assert len(captured)==1
            x,w,b=captured[0];reference=linear(x,w,b);torch.cuda.synchronize();save()
            print('Tuning FFN',list(x.shape),list(w.shape),flush=True)
            tunable.enable(True);tunable.tuning_enable(True);start=time.perf_counter()
            actual=linear(x,w,b);torch.cuda.synchronize();report['tuning_seconds']=time.perf_counter()-start
            tunable.tuning_enable(False);tunable.enable(False)
            report['algorithms']=tunable.get_results();report['first_projection_exact']=exact(reference,actual);save()
            assert report['first_projection_exact'],'Tuned first projection changed bits'
            pairs=[]
            for order in ((False,True),(True,False),(False,True)):
                pair={}
                for enabled in order:
                    tunable.enable(enabled)
                    for _ in range(2):linear(x,w,b)
                    values=[]
                    for _ in range(3):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        begin.record();actual=linear(x,w,b);end.record();end.synchronize()
                        values.append(begin.elapsed_time(end));assert exact(reference,actual)
                    pair['candidate' if enabled else 'baseline']=values
                pairs.append(pair)
            tunable.enable(False);report['operator_pairs_ms']=pairs
            report['operator_medians_ms']={a:statistics.median(t for p in pairs for t in p[a]) for a in ('baseline','candidate')};save()
            del captured,x,w,b,reference,actual
            @contextmanager
            def scope(enabled,audit=False):
                calls=0
                def dispatch(x,w,b=None):
                    nonlocal calls
                    if tuple(w.shape)!=(13824,2560):return linear(x,w,b)
                    assert tuple(x.shape)==(16,1002,2560) and b is None and x.dtype==w.dtype==torch.bfloat16
                    assert x.is_contiguous() and w.is_contiguous()
                    tunable.enable(enabled)
                    try:out=linear(x,w,b)
                    finally:tunable.enable(False)
                    if audit:assert exact(out,linear(x,w,b)),f'FFN projection {calls} differs'
                    calls+=1;return out
                F.linear=dispatch
                try:
                    yield
                    assert calls==80
                finally:F.linear=linear;tunable.enable(False)
            def run(enabled,audit=False):
                with scope(enabled,audit),optimized_scope(model,request,1000,batch_size=16):return forward()
            actual=run(True,audit=True)
            report['all_80_projections_exact']=True
            report['full_model_exact']=all(exact(v,actual[k]) for k,v in expected.items());save()
            assert report['full_model_exact']
            for i in range(16):save_arrays(output/f'tuned-{i}.npz',{k:v[i:i+1] for k,v in actual.items()})
            pairs=[]
            for order in ((False,True),(True,False),(False,True)):
                pair={}
                for enabled in order:
                    for _ in range(2):run(enabled)
                    values=[]
                    for _ in range(3):
                        torch.cuda.synchronize();start=time.perf_counter();actual=run(enabled);torch.cuda.synchronize()
                        values.append(time.perf_counter()-start);assert all(exact(v,actual[k]) for k,v in expected.items())
                    pair['candidate' if enabled else 'baseline']=values
                pairs.append(pair);report['forward_pairs']=pairs;save()
                print('Forward pair',pair,flush=True)
            report['forward_medians']={a:statistics.median(t for p in pairs for t in p[a]) for a in ('baseline','candidate')}
            report['forward_speedup']=report['forward_medians']['baseline']/report['forward_medians']['candidate']
            report['passed']=True;save()
    finally:F.linear=linear;tunable.tuning_enable(False);tunable.enable(False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,assets))
    report['assets_rehashed_after']=True;save();print(json.dumps({k:report[k] for k in ('algorithms','operator_medians_ms','forward_medians','forward_speedup')},indent=2),flush=True)


if __name__=='__main__':main()

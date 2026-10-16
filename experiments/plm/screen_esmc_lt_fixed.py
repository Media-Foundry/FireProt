"""Qualify direct fixed FFN GEMM against all 80 native projections and final outputs."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch
from fireprot.adapters.plm.runtime import optimized_scope
from fireprot.ops._bridge import bridge
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from esmc_lt_scope import FixedProjection
from screen_esmc_ck import difference


def main():
    assert os.environ['HIP_VISIBLE_DEVICES']=='6'
    request=json.loads(Path(sys.argv[1]).read_text());library=Path(sys.argv[2]);output=Path(sys.argv[3]);output.mkdir(exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not torch.cuda.tunable.is_enabled()
    src=Path(request['input']['path']);sha=hashlib.sha256(src.read_bytes()).hexdigest();assert sha==request['input']['sha256']
    plan=json.loads(bridge.plan_plm_batch(src.read_text(),json.dumps(request['workload']['batch'])))
    sequences=[x['sequence'] for x in plan['sequences']]
    report=dict(model=request['workload']['model_name'],length=1000,hip_device=6,seed=request['seed'],
        torch=torch.__version__,hip=torch.version.hip,precision='native BF16',input_sha256=sha,
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope_sha256=hashlib.sha256(Path(__file__).with_name('esmc_lt_scope.py').read_bytes()).hexdigest(),
        boundary='warm direct FFN plus existing exact HIP scope; not E2E',
        cache_state='resident model, warmed native library, caches not flushed',cases=[])
    def save():(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with torch.inference_mode():
        model,_,_=load_model(request,sequences[0])
        for size in (1,4):
            forward=prepare_forward(model,request['workload']['model_name'],sequences[:size])
            expected={k:v.clone() for k,v in forward().items()}
            torch.cuda.synchronize()
            start=time.perf_counter();projection=FixedProjection(library,size);torch.cuda.synchronize()
            case=dict(batch=size,plan_seconds_after_native_warmup=time.perf_counter()-start);report['cases'].append(case);save()
            def run(candidate,check=False):
                with optimized_scope(model,request,1000,batch_size=size):
                    if not candidate:return forward()
                    with projection.scope(check_each=check):return forward()
            try:
                actual=run(True,check=True)
                case['all_80_projections_exact']=True
                case['initial_difference']={k:difference(v,actual[k]) for k,v in expected.items()}
                assert all(d['bitwise_equal'] for d in case['initial_difference'].values());save()
                pairs=[]
                for order in ((False,True),(True,False),(False,True)):
                    pair={}
                    for candidate in order:
                        for _ in range(2):run(candidate)
                        ts=[]
                        for _ in range(3):
                            torch.cuda.synchronize();start=time.perf_counter();actual=run(candidate);torch.cuda.synchronize()
                            ts.append(time.perf_counter()-start)
                            assert all(torch.equal(v.view(torch.uint8),actual[k].view(torch.uint8)) for k,v in expected.items())
                        pair['candidate' if candidate else 'current_fireprot']=ts
                    pairs.append(pair)
                case['pairs']=pairs;case['medians']={a:statistics.median(t for p in pairs for t in p[a]) for a in ('candidate','current_fireprot')}
                case['speedup']=case['medians']['current_fireprot']/case['medians']['candidate'];save()
                print(json.dumps({k:v for k,v in case.items() if k!='pairs'}),flush=True)
            finally:torch.cuda.synchronize();projection.close()


if __name__=='__main__':main()

"""Tune ESMC below BF16 spacing, then independently check all output bits."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import torch.cuda.tunable as tunable
from fireprot.adapters.plm.runtime import optimized_scope
from fireprot.ops._bridge import bridge
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from screen_esmc_ck import difference


def main():
    assert os.environ['HIP_VISIBLE_DEVICES']=='6'
    req=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(req['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert str(torch.backends.cuda.preferred_blas_library()).endswith('Cublaslt')
    tunable.enable(False);tunable.tuning_enable(False)
    tunable.set_filename(str(out/'algorithms.csv'))
    # This Torch build rejects zero tolerances. 1e-45 is below the smallest
    # BF16 spacing (2**-133); raw-byte validation also distinguishes signed zero.
    tunable.set_numerical_check_tolerances(True,atol=1e-45,rtol=1e-45)
    tunable.set_max_tuning_duration(10);tunable.set_max_tuning_iterations(20)
    src=Path(req['input']['path']);sha=hashlib.sha256(src.read_bytes()).hexdigest()
    assert sha==req['input']['sha256']
    plan=json.loads(bridge.plan_plm_batch(src.read_text(),json.dumps(req['workload']['batch'])))
    sequences=[s['sequence'] for s in plan['sequences']]
    report=dict(model=req['workload']['model_name'],hip_device=6,seed=req['seed'],length=1000,
                precision='native BF16',torch=torch.__version__,hip=torch.version.hip,input_sha256=sha,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                boundary='warm forward, excludes offline tuning and loading; not E2E',
                cache_state='resident weights and warmed operators, caches not flushed',
                internal_tolerances=dict(atol=1e-45,rtol=1e-45),acceptance='independent raw-byte equality',cases=[])
    linear=F.linear
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        with torch.inference_mode():
            model,_,_=load_model(req,sequences[0])
            for size in (1,4):
                tunable.enable(False)
                forward=prepare_forward(model,req['workload']['model_name'],sequences[:size])
                captured=[]
                def capture(x,w,b=None):
                    if len(captured)<4:captured.append((x.clone(),w,b))
                    return linear(x,w,b)
                F.linear=capture
                try:expected={k:v.clone() for k,v in forward().items()}
                finally:F.linear=linear
                case=dict(batch=size,operators=[]);report['cases'].append(case);save()
                for index,(x,w,b) in enumerate(captured):
                    tunable.enable(False);reference=linear(x,w,b)
                    print(json.dumps(dict(stage='tuning',batch=size,operator=index,shape=list(x.shape),weight=list(w.shape))),flush=True)
                    tunable.enable(True);tunable.tuning_enable(True)
                    start=time.perf_counter();actual=linear(x,w,b);torch.cuda.synchronize()
                    tuning_seconds=time.perf_counter()-start
                    tunable.tuning_enable(False)
                    row=dict(index=index,tuning_seconds=tuning_seconds,difference=difference(reference,actual))
                    # This installed Torch exports its CSV automatically at process exit.
                    case['operators'].append(row);report['algorithms']=tunable.get_results();save()
                    print(json.dumps(dict(batch=size,operator=index,**row)),flush=True)
                del captured
                # Even a first-block match must pass the complete 80-block model.
                def run(candidate):
                    tunable.enable(candidate)
                    with optimized_scope(model,req,1000,batch_size=size):return forward()
                actual=run(True)
                case['initial_forward_difference']={k:difference(v,actual[k]) for k,v in expected.items()};save()
                if not all(d['bitwise_equal'] for d in case['initial_forward_difference'].values()):
                    case['accepted']=False;save();print(json.dumps(dict(batch=size,rejected=case['initial_forward_difference'])),flush=True)
                    continue
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
                case.update(accepted=True,pairs=pairs)
                case['medians']={a:statistics.median(t for p in pairs for t in p[a]) for a in ('candidate','current_fireprot')}
                case['speedup']=case['medians']['current_fireprot']/case['medians']['candidate'];save()
                print(json.dumps(dict(batch=size,medians=case['medians'],speedup=case['speedup'],bitwise_equal=True)),flush=True)
    finally:
        F.linear=linear;tunable.tuning_enable(False);tunable.enable(False)


if __name__=='__main__':main()

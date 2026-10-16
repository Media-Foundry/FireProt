"""Screen alternate installed BLAS backends without accepting numerical drift."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from fireprot_workers.atlasfold import read_sequence

request=json.loads(Path(sys.argv[1]).read_text())
output=Path(sys.argv[2]);output.mkdir(exist_ok=False)
assert os.environ['HIP_VISIBLE_DEVICES']=='6'
torch.set_num_threads(8);torch.manual_seed(request['seed'])
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
_,seq=read_sequence(request['input']['path'])
report=dict(model=request['workload']['model_name'],hip_device=6,batch=4,length=1000,precision='native',cases=[])
with torch.inference_mode():
    model,_,_=load_model(request,seq)
    forward=prepare_forward(model,request['workload']['model_name'],[seq[i*137:]+seq[:i*137] for i in range(4)])
    preferred=torch.backends.cuda.preferred_blas_library
    original=preferred()
    expected={k:v.clone() for k,v in forward().items()}
    report['original']=str(original)
    try:
        for choice in ('cublas','cublaslt'):
            preferred(choice)
            actual=forward()
            exact={k:torch.equal(v.view(torch.uint8),actual[k].view(torch.uint8)) for k,v in expected.items()}
            case=dict(choice=choice,exact=exact,max_abs={k:float((v.float()-actual[k].float()).abs().max()) for k,v in expected.items()})
            if all(exact.values()):
                pairs=[]
                for order in ((False,True),(True,False),(False,True)):
                    pair={}
                    for candidate in order:
                        preferred(choice if candidate else original)
                        for _ in range(2):forward()
                        ts=[]
                        for _ in range(3):
                            torch.cuda.synchronize();t=time.perf_counter();actual=forward();torch.cuda.synchronize()
                            ts.append(time.perf_counter()-t)
                            assert all(torch.equal(v.view(torch.uint8),actual[k].view(torch.uint8)) for k,v in expected.items())
                        pair['candidate' if candidate else 'baseline']=ts
                    pairs.append(pair)
                med={a:statistics.median([t for p in pairs for t in p[a]]) for a in ('baseline','candidate')}
                case.update(pairs=pairs,medians=med,speedup=med['baseline']/med['candidate'])
            report['cases'].append(case)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(case),flush=True)
    finally:preferred(original)

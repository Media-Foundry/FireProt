"""AB/BA/AB native-output gates and warm latency for ESM candidates."""
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
from fireprot_workers.transport import save_arrays
from esm_trials import candidate_scope

request=json.loads(Path(sys.argv[1]).read_text())
output=Path(sys.argv[2]);output.mkdir(exist_ok=False)
library=sys.argv[3] if len(sys.argv)>3 else None
name=request['workload']['model_name']
assert os.environ['HIP_VISIBLE_DEVICES']=='6'
torch.set_num_threads(8);torch.manual_seed(request['seed'])
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
_,seq=read_sequence(request['input']['path'])
report=dict(model=name,hip_device=6,seed=request['seed'],length=1000,precision='native',
            boundary='warm forward including scope entry and exit, not E2E',cases=[])
with torch.inference_mode():
    model,_,_=load_model(request,seq)
    originals=[(m,m.forward) for m in model.modules()]
    modes=['geometry'] if name=='esm3-open' else []
    if library: modes+=['hip']+(['geometry+hip'] if name=='esm3-open' else [])
    for size in (1,4):
        forward=prepare_forward(model,name,[seq[i*137:]+seq[:i*137] for i in range(size)])
        expected={k:v.clone() for k,v in forward().items()}
        save_arrays(output/f'b{size}-native.npz',expected)
        for mode in modes:
            def run(candidate):
                if not candidate: return forward(),None
                with candidate_scope(model,name,mode,library) as state: actual=forward()
                return actual,state
            actual,state=run(True)
            equality={k:torch.equal(v.view(torch.uint8),actual[k].view(torch.uint8)) for k,v in expected.items()}
            case=dict(batch=size,mode=mode,exact=equality,state=state)
            if all(equality.values()):
                pairs=[]
                for order in ((False,True),(True,False),(False,True)):
                    pair={}
                    for candidate in order:
                        for _ in range(2): run(candidate)
                        values=[]
                        for _ in range(3):
                            torch.cuda.synchronize();start=time.perf_counter()
                            actual,state=run(candidate);torch.cuda.synchronize()
                            values.append(time.perf_counter()-start)
                            assert all(torch.equal(v.view(torch.uint8),actual[k].view(torch.uint8)) for k,v in expected.items())
                            assert all(m.forward==f for m,f in originals)
                        pair['candidate' if candidate else 'native']=values
                    pairs.append(pair)
                medians={arm:statistics.median([t for p in pairs for t in p[arm]]) for arm in ('native','candidate')}
                case.update(pairs=pairs,medians=medians,speedup=medians['native']/medians['candidate'])
                save_arrays(output/f'b{size}-{mode}.npz',actual)
            report['cases'].append(case)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(case),flush=True)

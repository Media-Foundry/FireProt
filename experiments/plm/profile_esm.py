"""Profile native N1000 ESM forwards; diagnostics, not E2E measurements."""
import json
import os
from pathlib import Path
import sys
import time

import torch
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from fireprot_workers.atlasfold import read_sequence


request = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2]); output.mkdir(exist_ok=False)
assert os.environ['HIP_VISIBLE_DEVICES'] == '6'
torch.set_num_threads(8); torch.manual_seed(request['seed'])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
_, sequence = read_sequence(request['input']['path'])
with torch.inference_mode():
    start = time.perf_counter()
    model, _, _ = load_model(request, sequence)
    torch.cuda.synchronize()
    report = dict(model=request['workload']['model_name'], load_seconds=time.perf_counter()-start,
                  hip_device=6, seed=request['seed'], length=1000, precision='native', cases=[])
    for size in (1,4):
        forward = prepare_forward(model, request['workload']['model_name'],
                                  [sequence[i*137:]+sequence[:i*137] for i in range(size)])
        for _ in range(2): forward()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            forward(); torch.cuda.synchronize()
        rows = [dict(key=x.key, count=x.count, self_device_us=x.self_device_time_total,
                     self_cpu_us=x.self_cpu_time_total, shapes=x.input_shapes)
                for x in prof.key_averages(group_by_input_shape=True)]
        rows.sort(key=lambda x:x['self_device_us'],reverse=True)
        case=dict(batch=size,rows=rows)
        report['cases'].append(case)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(batch=size,top=rows[:22])),flush=True)
    (output/'modules.txt').write_text(str(model))

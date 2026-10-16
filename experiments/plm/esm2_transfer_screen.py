"""Fresh-process exact ESM2 CPU/GPU FP16-to-FP32 promotion diagnostics."""
import inspect
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import fireprot_workers.plm as worker
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays

request=json.loads(Path(sys.argv[1]).read_text())
mode=sys.argv[2]
output=Path(sys.argv[3]);output.mkdir(exist_ok=False)
reference=Path(sys.argv[4])
assert os.environ['HIP_VISIBLE_DEVICES']=='6'
request['backend']='fireprot'
torch.set_num_threads(8);torch.manual_seed(request['seed'])
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
_,seq=read_sequence(request['input']['path'])
source=inspect.getsource(worker.load_model)
old='model = model.float().eval().cuda()'
assert source.count(old)==1
if mode=='gpu_float':source=source.replace(old,'model = model.cuda().float().eval()')
elif mode=='combined':source=source.replace(old,"model = model.to(device='cuda', dtype=torch.float32).eval()")
else:assert mode=='cpu_float'
namespace=dict(worker.__dict__)
exec(compile(source,'<esm2-transfer-screen>','exec'),namespace)
with torch.inference_mode():
    torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();start=time.perf_counter()
    model,_,forward=namespace['load_model'](request,seq)
    torch.cuda.synchronize();load=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    arrays=forward();torch.cuda.synchronize()
    save_arrays(output/'sample.npz',arrays)
with np.load(reference,allow_pickle=False) as expected,np.load(output/'sample.npz',allow_pickle=False) as actual:
    exact={key:expected[key].dtype==actual[key].dtype and expected[key].shape==actual[key].shape
           and expected[key].tobytes()==actual[key].tobytes() for key in expected.files}
report=dict(mode=mode,load_seconds=load,load_peak_bytes=peak,exact=exact,
            boundary='loader diagnostic only; import/CUDA initialization excluded; no E2E claim')
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
assert all(exact.values())

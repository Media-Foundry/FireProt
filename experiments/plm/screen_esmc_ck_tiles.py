"""Screen six direct CK tiles on actual first-block ESMC projections."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F
from fireprot.ops._bridge import bridge
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from screen_esmc_ck import difference


def main():
    assert os.environ['HIP_VISIBLE_DEVICES'] == '6'
    request = json.loads(Path(sys.argv[1]).read_text())
    library = Path(sys.argv[2])
    output = Path(sys.argv[3]); output.mkdir(exist_ok=False)
    lib = ctypes.CDLL(str(library)); gemm = lib.esmc_ck_tile
    gemm.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    gemm.restype = ctypes.c_int
    torch.set_num_threads(8); torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert str(torch.backends.cuda.preferred_blas_library()).endswith('Cublaslt')
    input_path = Path(request['input']['path'])
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == request['input']['sha256']
    plan = json.loads(bridge.plan_plm_batch(input_path.read_text(), json.dumps(request['workload']['batch'])))
    sequences = [x['sequence'] for x in plan['sequences']]
    report = dict(model=request['workload']['model_name'], hip_device=6, seed=request['seed'],
                  length=1000, dtype='bfloat16', torch=torch.__version__, hip=torch.version.hip,
                  input_sha256=request['input']['sha256'],
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  boundary='warm first-block operators only; no E2E claim',
                  cache_state='resident weights and activations; warmed operators; caches not flushed',
                  variants=['128x128x32-mfma32','128x128x64-mfma32','128x128x128-mfma32',
                            '256x128x32-mfma32','128x256x32-mfma32','128x128x64-mfma16'], cases=[])
    linear = F.linear
    def save(): (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    with torch.inference_mode():
        model, _, _ = load_model(request, sequences[0])
        for size in (1,4):
            captured = []
            def capture(x,w,b=None):
                if len(captured)<4: captured.append((x.clone(),w,b))
                return linear(x,w,b)
            F.linear = capture
            try: prepare_forward(model, request['workload']['model_name'], sequences[:size])()
            finally: F.linear = linear
            for index,(x,w,b) in enumerate(captured):
                assert x.is_contiguous() and w.is_contiguous() and b is None
                m=x.numel()//x.shape[-1]; n,k=w.shape
                reference=linear(x,w,b)
                dest=torch.empty_like(reference)
                row=dict(batch=size, operator=index, m=m,n=n,k=k,variants=[])
                report['cases'].append(row)
                for variant in range(6):
                    def run():
                        rc=gemm(variant,x.data_ptr(),w.data_ptr(),dest.data_ptr(),m,n,k,
                                torch.cuda.current_stream().cuda_stream)
                        if rc: raise RuntimeError(f'CK variant {variant}: status {rc}')
                        return dest
                    try: actual=run(); torch.cuda.synchronize()
                    except RuntimeError as error:
                        row['variants'].append(dict(variant=variant,error=str(error))); save(); continue
                    entry=dict(variant=variant,difference=difference(reference,actual),pairs=[])
                    first=actual.clone()
                    for order in ((False,True),(True,False),(False,True)):
                        pair={}
                        for candidate in order:
                            fn=run if candidate else lambda:linear(x,w,b)
                            for _ in range(4):fn()
                            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(20):actual=fn()
                            end.record(); end.synchronize()
                            assert torch.equal(actual.view(torch.uint8),(first if candidate else reference).view(torch.uint8))
                            pair['ck' if candidate else 'hipblaslt']=start.elapsed_time(end)/20
                        entry['pairs'].append(pair)
                    entry['median_ms']={a:statistics.median(p[a] for p in entry['pairs']) for a in ('ck','hipblaslt')}
                    entry['speedup']=entry['median_ms']['hipblaslt']/entry['median_ms']['ck']
                    row['variants'].append(entry);save()
                valid=[v for v in row['variants'] if 'speedup' in v]
                print(json.dumps(dict(batch=size,operator=index,best=max(valid,key=lambda v:v['speedup']) if valid else None,
                                      exact_variants=[v['variant'] for v in valid if v['difference']['bitwise_equal']])),flush=True)
            del captured
    save()


if __name__ == '__main__': main()

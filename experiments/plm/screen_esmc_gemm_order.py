"""Diagnose BF16 GEMM accumulation/layout differences on real ESMC activations."""
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from fireprot.ops._bridge import bridge
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from screen_esmc_ck import difference


def main():
    assert os.environ['HIP_VISIBLE_DEVICES'] == '6'
    req=json.loads(Path(sys.argv[1]).read_text())
    library=Path(sys.argv[2]); out=Path(sys.argv[3]);out.mkdir(exist_ok=False)
    lib=ctypes.CDLL(str(library)); gemm=lib.esmc_ck_tile
    gemm.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]
    gemm.restype=ctypes.c_int
    torch.set_num_threads(8);torch.manual_seed(req['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    preferred=torch.backends.cuda.preferred_blas_library;original=preferred()
    assert str(original).endswith('Cublaslt')
    source=Path(req['input']['path']); source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash==req['input']['sha256']
    plan=json.loads(bridge.plan_plm_batch(source.read_text(),json.dumps(req['workload']['batch'])))
    seq=plan['sequences'][0]['sequence']
    report=dict(model=req['workload']['model_name'],length=len(seq),batch=1,seed=req['seed'],hip_device=6,
                dtype='native BF16',input_sha256=source_hash,torch=torch.__version__,hip=torch.version.hip,
                library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='first block FFN up/gate; correctness diagnostics only; no timing claim',layouts=[],permutations=[])
    linear=F.linear;captured=[]
    def capture(x,w,b=None):
        if len(captured)<4:captured.append((x.clone(),w,b))
        return linear(x,w,b)
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        with torch.inference_mode():
            model,_,_=load_model(req,seq)
            F.linear=capture
            try:prepare_forward(model,req['workload']['model_name'],[seq])()
            finally:F.linear=linear
            x,w,b=captured[2];assert w.shape==(13824,2560) and b is None
            x=x.reshape(-1,2560);reference=linear(x,w)
            report['shape']=[x.shape[0],w.shape[0],w.shape[1]]
            for backend in ('cublaslt','ck'):
                preferred(backend)
                for name,fn in (
                    ('native_linear',lambda:linear(x,w)),
                    ('contiguous_rhs',lambda:x@w.t().contiguous()),
                    ('swapped_operands',lambda:(w@x.t()).t().contiguous()),
                    ('swapped_contiguous_rhs',lambda:(w@x.t().contiguous()).t().contiguous())):
                    try: actual=fn()
                    except RuntimeError as error:
                        if 'does not support this GEMM problem' not in str(error): raise
                        row=dict(backend=backend,layout=name,unsupported=str(error))
                        report['layouts'].append(row);save();print(json.dumps(row),flush=True)
                        continue
                    row=dict(backend=backend,layout=name,difference=difference(reference,actual))
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        fn();torch.cuda.synchronize()
                    row['kernels']=sorted({e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA})
                    report['layouts'].append(row);save()
                    print(json.dumps({k:v for k,v in row.items() if k!='kernels'}),flush=True)
            preferred(original)
            dest=torch.empty_like(reference)
            indices=torch.arange(x.shape[1],device=x.device)
            for bits in itertools.permutations(range(3)):
                for xor in range(8):
                    low=indices%8
                    p=(indices//8)*8
                    for dst,src in enumerate(bits): p=p+(((low>>src)&1)<<dst)
                    p=(p//8)*8+((p%8)^xor)
                    a=x.index_select(1,p);weight=w.index_select(1,p)
                    rc=gemm(0,a.data_ptr(),weight.data_ptr(),dest.data_ptr(),x.shape[0],w.shape[0],w.shape[1],
                            torch.cuda.current_stream().cuda_stream)
                    assert rc==0,rc
                    report['permutations'].append(dict(bits=list(bits),xor=xor,difference=difference(reference,dest)))
                    save()
            best=min(report['permutations'],key=lambda v:v['difference']['unequal_elements'])
            report['best_permutation']=best
            report['exact_permutations']=[v for v in report['permutations'] if v['difference']['bitwise_equal']]
            save();print(json.dumps(dict(best_permutation=best,exact_count=len(report['exact_permutations']))),flush=True)
    finally:
        F.linear=linear;preferred(original)


if __name__=='__main__':main()

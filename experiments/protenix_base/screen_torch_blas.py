"""Compare Torch BLAS preferences on padded contractions, including preference swaps."""
import json
import os
from pathlib import Path
import statistics
import sys
import torch
import torch.utils.deterministic


def main(destination):
    if os.environ.get('HIP_VISIBLE_DEVICES') not in ('6','7'):raise RuntimeError('HIP6/7 required')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('hipblas')
    torch.manual_seed(101)
    rows=[]
    with torch.inference_mode():
        a,b=[torch.empty_strided((128,1000,1000),(1000128,1000,1),device='cuda',dtype=torch.float32) for _ in range(2)]
        for x in (a,b):x.normal_().div_(1000**0.5)
        for outgoing in (0,1):
            left,right=(a,b.transpose(-1,-2)) if outgoing else (a.transpose(-1,-2),b)
            reference=torch.bmm(left,right)
            def call(backend):
                previous=torch.backends.cuda.preferred_blas_library()
                try:
                    torch.backends.cuda.preferred_blas_library(backend)
                    return torch.bmm(left,right)
                finally:torch.backends.cuda.preferred_blas_library(previous)
            candidate=call('hipblaslt');torch.testing.assert_close(candidate,reference,atol=2e-5,rtol=2e-5)
            numeric=dict(max_abs_delta=(candidate-reference).abs().max().item(),bitwise_equal=torch.equal(candidate.view(torch.int32),reference.view(torch.int32)))
            del candidate,reference
            times={name:[] for name in ['hipblas','hipblaslt']}
            for i in range(24):
                for backend in (['hipblas','hipblaslt'] if i%2==0 else ['hipblaslt','hipblas']):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();result=call(backend);end.record();end.synchronize()
                    if i>=4:times[backend].append(start.elapsed_time(end))
                    del result
            rows.append(dict(outgoing=outgoing,numerics=numeric,timings={k:dict(median_ms=statistics.median(v),values_ms=v) for k,v in times.items()}))
    report=dict(scope='Synthetic Torch BLAS preference screen, not proof of internal kernel selection or model benefit',shape=[128,1000,1000],strides=[1000128,1000,1],torch=str(torch.__version__),hip=str(torch.version.hip),seed=101,hip_device=os.environ['HIP_VISIBLE_DEVICES'],rows=rows)
    Path(destination).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main(sys.argv[1])

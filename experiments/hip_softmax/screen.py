"""Tensor-only correctness screen; compilation/cache ownership remains in Rust."""
import ctypes
import json
from pathlib import Path
import sys

import torch


def main(destination):
    from fireprot.backends.amd._native_hip import load
    root = Path(__file__).resolve().parent
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    library, receipt = load('fireprot_experimental_softmax_f32',
        root/'fused_softmax.cpp', Path('/opt/rocm'), extra_files=[__file__],
        versions={'c_abi': 'experimental-softmax-f32-v1'})
    (destination/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    entry = library.fireprot_softmax_f32
    entry.argtypes = ([ctypes.c_void_p]*4 + [ctypes.c_int]*2
                     + [ctypes.c_int64]*5 + [ctypes.c_int]*2 + [ctypes.c_void_p])
    entry.restype = ctypes.c_int
    torch.manual_seed(101)
    records = []
    with torch.no_grad():
        for n, batch in [(1000,8), (1536,8), (1536,128)]:
            scores = torch.randn(batch,4,n,n,device='cuda',dtype=torch.float32)*0.5
            # Real upstream linear_z output is head-last before permutation.
            bias = torch.randn(n,n,4,device='cuda').permute(2,0,1).unsqueeze(0)*0.1
            mask = torch.zeros(batch,1,1,n,device='cuda')
            cases = ['zero','partial','nonzero','extreme'] if batch == 8 else ['large-address']
            for case in cases:
                if case == 'partial': mask[..., -7:] = -float('inf')
                elif case == 'nonzero': mask.normal_(0,0.1)
                elif case == 'extreme':
                    mask.zero_()
                    scores[0,0,0,:] = -110.0
                    scores[0,0,0,0] = 0.0
                    scores[0,0,1,:] = -0.0
                    scores[0,0,2,:] = -float('inf')
                    scores[0,0,3,0] = float('nan')
                    scores[0,0,4,0] = float('inf')
                summed = scores.clone()
                summed.add_(mask).add_(bias)
                reference = torch.softmax(summed,dim=-1)
                del summed
                for waves in (2,4):
                    for reuse in (0,1):
                        output = torch.empty_like(scores)
                        status = entry(scores.data_ptr(),mask.data_ptr(),bias.data_ptr(),output.data_ptr(),
                            batch,n,mask.stride(0),mask.stride(-1),*bias.stride()[-3:],
                            waves,reuse,torch.cuda.current_stream().cuda_stream)
                        if status: raise RuntimeError(f'HIP status {status}')
                        torch.cuda.synchronize()
                        different = torch.count_nonzero(output.view(torch.int32) != reference.view(torch.int32)).item()
                        record = dict(n=n,batch=batch,case=case,waves=waves,reuse=reuse,
                            score_strides=list(scores.stride()),bias_strides=list(bias.stride()),
                            mismatched_bits_elements=different,passed=different==0)
                        records.append(record)
                        print(json.dumps(record),flush=True)
                        del output
                del reference
            del scores,bias,mask
    result = dict(schema_version=1,torch=str(torch.__version__),hip=str(torch.version.hip),
        device=torch.cuda.get_device_name(),seed=101,records=records,
        passed=all(r['passed'] for r in records))
    (destination/'screen.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main(sys.argv[1])

"""Synthetic CK tile screen; not trained-model accuracy or E2E evidence."""
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import torch
import torch.utils.deterministic
from fireprot.backends.amd._native_hip import load
from fireprot.backends.amd import ck_attention


def main(destination, source_name='ck_tiles.cpp', tiles=None):
    if os.environ.get('HIP_VISIBLE_DEVICES') not in ('6','7'):
        raise RuntimeError('Only HIP6/7 authorized')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a':
        raise RuntimeError('gfx90a required')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.manual_seed(101)
    lib,receipt=load('fireprot_base_ck_tiles',Path(__file__).with_name(source_name),Path('/opt/rocm'),extra_files=[__file__],versions={'c_abi':'base-ck-tiles-v1'})
    outpath=Path(destination)
    outpath.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    entry=lib.fireprot_base_ck_tiles
    entry.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];entry.restype=ctypes.c_int
    if ck_attention.load_extension() is None:raise RuntimeError('Reference CK unavailable')
    rows=[]; timings={v:[] for v in range(4)}
    with torch.inference_mode():
        q,k,v=[torch.randn(1000,1000,4,32,device='cuda').transpose(1,2) for _ in range(3)]
        q.div_(32**0.5)
        bias=torch.randn(1,4,1000,1000,device='cuda')
        def call(index):
            result=torch.empty((1000,1000,4,32),device='cuda')
            status=entry(q.data_ptr(),k.data_ptr(),v.data_ptr(),bias.data_ptr(),result.data_ptr(),index,torch.cuda.current_stream().cuda_stream)
            if status:raise RuntimeError(f'HIP status {status}')
            return result
        for pattern in ('random','large_bias'):
            if pattern=='large_bias':bias.mul_(20)
            reference=ck_attention.attention(module=None,q=q,k=k,v=v,biases=[bias],native=lambda:None)
            for index in range(4):
                candidate=call(index)
                torch.testing.assert_close(candidate,reference,atol=2e-5,rtol=2e-5)
                rows.append(dict(pattern=pattern,variant=index,max_abs_delta=(candidate-reference).abs().max().item(),bitwise_equal=torch.equal(candidate.view(torch.int32),reference.contiguous().view(torch.int32))))
                del candidate
            del reference
        bias.div_(20)
        # Interleave directions to avoid confounding variant with device drift.
        for iteration in range(20):
            for index in (range(4) if iteration%2==0 else reversed(range(4))):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); result=call(index);end.record();end.synchronize()
                if iteration>=4:timings[index].append(start.elapsed_time(end))
                del result
    report=dict(scope='Synthetic N1000 attention; allocations and deterministic filling included; not model qualification',hip_device=os.environ['HIP_VISIBLE_DEVICES'],seed=101,torch=str(torch.__version__),hip=str(torch.version.hip),numerics=rows,records=[dict(variant=i,tile=tile,median_ms=statistics.median(timings[i]),values_ms=timings[i]) for i,tile in enumerate(tiles or ['128/32/4','64/32/4','64/64/4','128/64/4'])],build_library=receipt['library'],build_library_sha256=receipt['library_sha256'])
    outpath.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main(sys.argv[1], *(sys.argv[2:3]), tiles=json.loads(sys.argv[3]) if len(sys.argv)>3 else None)

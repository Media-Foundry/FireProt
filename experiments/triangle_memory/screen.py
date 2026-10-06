"""Bitwise/canary/stream tests and isolated copy timings on gfx90a."""
import json
from pathlib import Path
import statistics
import sys
import torch
import torch.utils.deterministic
from copy_runtime import CopyRuntime


def bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def measure(fn):
    samples = []
    for _ in range(9):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples[2:])


@torch.inference_mode()
def main(build, destination):
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    rt = CopyRuntime(json.loads(Path(build).read_text())['library'])
    # All int32 bit patterns, including explicit IEEE edge values.
    src = torch.randint(-(2**31), 2**31-1, (256,1536,128), device='cuda', dtype=torch.int32)
    src.flatten()[:4] = torch.tensor([0,-2147483648,2143289345,2139095040],device='cuda',dtype=torch.int32)
    src = src.view(torch.float32)
    results = []
    for mode in ('row','column','bmm','norm'):
        for offset in ((0,256,1280) if mode in ('row','column') else (0,)):
            if mode == 'row':
                inp = src.permute(2,0,1)
                parent = torch.full((128,1536,1536),17.,device='cuda')
                out = parent[:,offset:offset+256,:]
            elif mode == 'column':
                inp = src.permute(2,1,0)
                parent = torch.full((128,1536,1536),17.,device='cuda')
                out = parent[:,:,offset:offset+256]
            elif mode == 'bmm':
                inp = src.permute(2,1,0)
                parent = out = torch.empty(inp.shape,device='cuda')
            else:
                base = src.permute(2,1,0).contiguous()
                inp = base.permute(1,2,0)
                parent = out = torch.empty(inp.shape,device='cuda')
            expected = parent.clone()
            ref = (expected[:,offset:offset+256,:] if mode=='row' else
                   expected[:,:,offset:offset+256] if mode=='column' else expected)
            ref.copy_(inp)
            before = rt.calls
            rt.copy_(out,inp)
            assert rt.calls == before+1 and bits_equal(parent,expected)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                rt.copy_(out,inp)
            torch.cuda.current_stream().wait_stream(stream)
            assert bits_equal(parent,expected)
            native, hip = [], []
            for order in (False,True,False):
                if order:
                    hip.append(measure(lambda:rt.copy_(out,inp)))
                    native.append(measure(lambda:out.copy_(inp)))
                else:
                    native.append(measure(lambda:out.copy_(inp)))
                    hip.append(measure(lambda:rt.copy_(out,inp)))
            results.append(dict(mode=mode,offset=offset,native_ms=statistics.median(native),
                hip_ms=statistics.median(hip),exact=True))
            del parent,out,expected,ref,inp
    n=rt.fallbacks
    a=torch.arange(32,device='cuda',dtype=torch.float32)
    rt.copy_(a,a); assert rt.fallbacks==n+1
    b=torch.empty_like(a); rt.copy_(b,a); assert bits_equal(a,b) and rt.fallbacks==n+2
    Path(destination).write_text(json.dumps(dict(correctness_passed=True,calls=rt.calls,
        fallbacks=rt.fallbacks,results=results,torch=str(torch.__version__),hip=torch.version.hip,
        device=torch.cuda.get_device_name(),strict=True,seed=101),indent=2)+'\n')


if __name__ == '__main__':
    main(*sys.argv[1:])

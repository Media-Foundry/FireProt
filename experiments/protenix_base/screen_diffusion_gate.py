"""Exact-output and allocation-inclusive screen of the diffusion gate HIP kernel."""
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import torch
from fireprot.backends.amd._native_hip import load


def main(destination):
    if os.environ.get('HIP_VISIBLE_DEVICES') not in ('6', '7'):
        raise RuntimeError('HIP6/7 only')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a':
        raise RuntimeError('gfx90a required')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    lib, receipt = load('fireprot_base_diffusion_gate', Path(__file__).with_name('diffusion_gate.cpp'),
                        Path('/opt/rocm'), link_flags=['-ffp-contract=off'],
                        extra_files=[__file__], versions={'c_abi': 'base-diffusion-gate-v1'})
    dest = Path(destination)
    dest.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    entry = lib.base_diffusion_gate
    entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    entry.restype = ctypes.c_int
    checks, timings = [], {}
    with torch.no_grad():
        for layout in ('token_major', 'head_major'):
            o = torch.randn((1, 1000, 16, 48), device='cuda') if layout == 'token_major' else torch.randn((1, 16, 1000, 48), device='cuda').transpose(1, 2)
            source = torch.randn((1, 1000, 768), device='cuda')
            def native():
                gate = source.clone()  # Common stand-in for fresh linear_g output.
                return (o * torch.sigmoid(gate).view(1, 1000, 16, 48)).reshape(1, 1000, 768)
            def fused():
                gate = source.clone()
                status = entry(o.data_ptr(), gate.data_ptr(), o.stride(1), o.stride(2), torch.cuda.current_stream().cuda_stream)
                if status:
                    raise RuntimeError(f'HIP launch status {status}')
                return gate
            for scale in (1, 20):
                if scale == 20:
                    source.mul_(20)
                ref, actual = native(), fused()
                exact = torch.equal(ref.view(torch.int32), actual.view(torch.int32))
                checks.append(dict(layout=layout, scale=scale, bitwise_equal=exact,
                                   max_abs_delta=(ref-actual).abs().max().item()))
                if not exact:
                    raise RuntimeError('Diffusion gate exact-output check failed')
            source.div_(20)
            values = {'native': [], 'fused': []}
            methods = [('native', native), ('fused', fused)]
            for iteration in range(44):
                for name, fn in (methods if iteration % 2 == 0 else list(reversed(methods))):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(); result = fn(); end.record(); end.synchronize()
                    if iteration >= 4:
                        values[name].append(begin.elapsed_time(end))
                    del result
            timings[layout] = {name: dict(median_ms=statistics.median(v), values_ms=v) for name, v in values.items()}
    report = dict(scope='Synthetic gate/layout conversion; common gate allocation included; excludes linear_g and linear_o; not E2E',
                  seed=101, dtype='float32', hip_device=os.environ['HIP_VISIBLE_DEVICES'],
                  deterministic_fill=True, checks=checks, timings=timings,
                  library_sha256=receipt['library_sha256'], library=receipt['library'])
    dest.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])

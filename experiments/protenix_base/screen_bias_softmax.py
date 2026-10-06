"""Qualify optional-mask HIP softmax on synthetic private FP32 score buffers.

This is an operator screen, not model correctness or E2E performance evidence.
Run only when no timed model experiment is active. Rust owns the native build.
"""
import ctypes
import json
from pathlib import Path
import sys

import torch
import torch.utils.deterministic
from fireprot.backends.amd._native_hip import load


def main(destination):
    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a':
        raise RuntimeError('gfx90a HIP required')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    library, receipt = load('fireprot_base_bias_softmax',
        Path(__file__).with_name('bias_softmax.cpp'), Path('/opt/rocm'),
        extra_files=[__file__], versions={'c_abi': 'base-bias-softmax-f32-v1'})
    entry = library.fireprot_base_bias_softmax_f32
    entry.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
                     + [ctypes.c_int64] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    entry.restype = ctypes.c_int
    records = []
    with torch.inference_mode():
        for n in (600, 1000):
            for batch in (1, 2, 8):
                for pattern in ('random', 'zero', 'large', 'partial_mask', 'all_mask'):
                    scores = torch.randn(batch, 4, n, n, device='cuda', dtype=torch.float32)
                    bias = torch.randn(1, 4, n, n, device='cuda', dtype=torch.float32)
                    mask = None
                    if pattern == 'zero':
                        scores.zero_()
                        bias.zero_()
                    elif pattern == 'large':
                        scores.mul_(80)
                    elif pattern in ('partial_mask', 'all_mask'):
                        mask = torch.zeros(batch, 1, 1, n, device='cuda')
                        mask[..., :n if pattern == 'all_mask' else n//2] = -torch.inf
                    expected = scores.clone()
                    if mask is not None:
                        expected.add_(mask)
                    expected.add_(bias)
                    expected = torch.softmax(expected, dim=-1)
                    actual = scores.clone()  # Private storage, no additional consumers.
                    code = entry(actual.data_ptr(), None if mask is None else mask.data_ptr(),
                        bias.data_ptr(), actual.data_ptr(), batch, n,
                        0 if mask is None else mask.stride(0), 0 if mask is None else mask.stride(-1),
                        *bias.stride()[-3:], 2, 1, torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize()
                    if code:
                        raise RuntimeError(f'HIP status {code}')
                    mismatches = int((actual.view(torch.int32) != expected.view(torch.int32)).sum())
                    records.append(dict(n=n, batch=batch, pattern=pattern, bitwise_mismatches=mismatches))
    report = dict(torch=str(torch.__version__), hip=torch.version.hip,
                  build=receipt, records=records, passed=all(r['bitwise_mismatches']==0 for r in records))
    Path(destination).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed':report['passed'],'records':records}), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main(sys.argv[1])

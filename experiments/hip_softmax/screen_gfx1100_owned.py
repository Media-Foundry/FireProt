"""Strict N1000 operator and complete-attention screen; not E2E evidence."""
import json
from pathlib import Path
from statistics import median
import sys

import torch
import torch.utils.deterministic
from gfx1100_owned import Gfx1100OwnedSoftmax


def exact(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


@torch.inference_mode()
def main(root):
    root = Path(root)
    build = json.loads((root/'build.json').read_text())
    old = '/media/990Pro/fireprot/runtime/kernels/openfold3-gfx1100-softmax-screen-20261017/wave32-v1.so'
    digest = 'df2c109098b48b019dc95bd120503af301d6d4cec5762f0bbc1fedab05303ad5'
    runtime = Gfx1100OwnedSoftmax(old, digest, build['library'], build['library_sha256'], True)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    results = []
    for batch in (1, 8, 232, 256):
        for ending in (False, True):
            q, k, v = [torch.randn(batch, 4, 1000, 32, device='cuda')*.2 for _ in range(3)]
            bias = torch.randn(1, 4, 1000, 1000, device='cuda')*.1
            if ending:
                bias = bias.transpose(-1, -2)
            mask = torch.zeros(batch, 1, 1, 1000, device='cuda')
            for mode in ('zero', 'partial', 'nonzero', 'special'):
                if mode == 'partial':
                    mask[..., 750:] = -1e9
                elif mode == 'nonzero':
                    mask.normal_(0, .1)
                scores = torch.einsum('...qc,...kc->...qk', q, k)
                if mode == 'special':
                    scores[:, :, 0, :] = -float('inf')
                    scores[:, :, 1, 0] = float('inf')
                    scores[:, :, 2, 0] = float('nan')
                    scores[:, :, 3, :] = -0.
                runtime.enabled = False
                expected = runtime(scores, [mask, bias])
                runtime.enabled = True
                pointer = scores.data_ptr()
                actual = runtime(scores, [mask, bias])
                assert pointer == actual.data_ptr() and exact(actual, expected)
                if mode != 'special':
                    native = torch.einsum('...qc,...kc->...qk', q, k)
                    native += mask
                    native += bias
                    native = native.softmax(-1)
                    assert exact(actual, native)
                    del native
                del scores, actual, expected
            mask.zero_()

            def attention(enabled):
                runtime.enabled = enabled
                scores = torch.einsum('...qc,...kc->...qk', q, k)
                probs = runtime(scores, [mask, bias])
                return torch.einsum('...qk,...kc->...qc', probs, v)

            expected = attention(False)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = attention(True)
            torch.cuda.current_stream().wait_stream(stream)
            assert exact(actual, expected)
            samples = {'baseline': [], 'candidate': []}
            peaks = {}
            for repeat in range(3):
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    arm = 'candidate' if enabled else 'baseline'
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(3):
                        output = attention(enabled)
                        del output
                    end.record()
                    end.synchronize()
                    samples[arm].append(begin.elapsed_time(end)/3)
                    peaks[arm] = torch.cuda.max_memory_allocated()
            medians = {a: median(v) for a, v in samples.items()}
            results.append(dict(batch=batch, ending=ending, exact=True, samples_ms=samples,
                medians_ms=medians, speedup=medians['baseline']/medians['candidate'],
                peak_allocated_bytes=peaks))
            del q, k, v, bias, mask, expected, actual
    # Offset storage and bias alias must keep the out-of-place route.
    parent = torch.randn(4000001, device='cuda')
    scores = parent[1:].view(1, 4, 1000, 1000)
    mask = torch.zeros(1, 1, 1, 1000, device='cuda')
    bias = torch.zeros(1, 4, 1000, 1000, device='cuda')
    runtime.enabled = True
    before = runtime.receipt['owned_scores']['fallbacks']
    assert runtime(scores, [mask, bias]).data_ptr() != scores.data_ptr()
    scores = scores.clone()
    assert runtime(scores, [mask, scores]).data_ptr() != scores.data_ptr()
    assert runtime.receipt['owned_scores']['fallbacks'] == before+2
    with (root/'screen.json').open('x') as out:
        json.dump(dict(correctness_passed=True, results=results, receipt=runtime.receipt,
            scope='Operator and full QK-softmax-PV only; no model E2E claim',
            strict=True, seed=101, torch=str(torch.__version__), hip=torch.version.hip), out, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])

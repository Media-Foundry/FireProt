"""GPU numerical and fallback checks for the experimental zero-mask filter."""
import importlib.util
import json
from pathlib import Path

import torch



def test_zero_mask_exact_and_nonzero_fallback(tmp_path):
    import openfold3.core.model.primitives.attention as attention
    spec = importlib.util.spec_from_file_location('zero_mask', Path(__file__).with_name('zero_mask.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = attention._attention
    torch.manual_seed(1043)
    q = torch.randn(2, 4, 1000, 32, device='cuda') * 0.125
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    pair = torch.randn(1, 4, 1000, 1000, device='cuda') * 0.1
    zero = torch.zeros(2, 1, 1, 1000, device='cuda')
    masks = [zero, -zero, torch.ones_like(zero)]
    partial = zero.clone()
    partial[..., -1] = -float('inf')
    masks.append(partial)
    import os
    old = os.environ.get('FIREPROT_SKIP_ZERO_MASK')
    os.environ['FIREPROT_SKIP_ZERO_MASK'] = '1'
    try:
        with torch.no_grad(), module.zero_mask_probe(tmp_path):
            for mask in masks:
                for precision in [False, True]:
                    reference = original(q, k, v, [mask, pair], use_high_precision=precision)
                    candidate = attention._attention(q, k, v, [mask, pair], use_high_precision=precision)
                    assert torch.equal(reference.view(torch.int32), candidate.view(torch.int32))
            # Shape mismatch must use the original bias list and arithmetic.
            reference = original(q[..., :8, :], k, v, [zero, pair[..., :8, :]])
            candidate = attention._attention(q[..., :8, :], k, v, [zero, pair[..., :8, :]])
            assert torch.equal(reference.view(torch.int32), candidate.view(torch.int32))
        assert attention._attention is original
        report = json.loads((tmp_path/'zero-mask-audit.json').read_text())
        assert report['calls'] == 9
        assert report['eligible_calls'] == report['checked'] == 8
        assert report['skipped'] == report['nonzero'] == 4
    finally:
        if old is None:
            os.environ.pop('FIREPROT_SKIP_ZERO_MASK', None)
        else:
            os.environ['FIREPROT_SKIP_ZERO_MASK'] = old


if __name__ == '__main__':
    import tempfile
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('Select an explicit ROCm device for this test')
    with tempfile.TemporaryDirectory() as directory:
        test_zero_mask_exact_and_nonzero_fallback(Path(directory))
    print('PASS: eight exact mask/precision cases, shape fallback and restoration')

"""Exact attention, fallback, restoration and manifest-pin checks after publication."""
import json
import os
from pathlib import Path
import sys

import torch
import openfold3.core.model.primitives.attention as attention
from fireprot.adapters.openfold3.attention import attention_scope
from fireprot.adapters.openfold3.gfx1100.dispatch import Gfx1100Dispatcher
from fireprot.backends.amd.softmax_gfx1100 import Gfx1100Softmax, CASES

destination = Path(sys.argv[1]).resolve()
torch.set_num_threads(8)
torch.manual_seed(101)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
kernel = Gfx1100Softmax()
dispatcher = Gfx1100Dispatcher(kernel)
original = attention._attention
report = dict(schema_version=1, status='running', records=[])
try:
    with torch.inference_mode():
        for batch, n in sorted(CASES | {(1, 124), (2, 1000)}):
            q, k, v = [torch.randn(batch, n, 4, 32, device='cuda').transpose(1, 2) * 0.125 for _ in range(3)]
            for ending in (False, True):
                pair = torch.randn(n, n, 4, device='cuda') * 0.1
                bias = pair.permute((2, 1, 0) if ending else (2, 0, 1)).unsqueeze(0)
                for kind in ('nonzero', 'partial', 'fully-masked'):
                    storage = torch.randn(batch, 1, 1, n * 2, device='cuda') * 0.1
                    mask = storage[..., ::2]
                    if kind == 'partial':
                        mask[..., -7:] = -float('inf')
                    elif kind == 'fully-masked':
                        mask.fill_(-float('inf'))
                    expected = original(q, k, v, [mask, bias])
                    before = kernel.launches_submitted
                    with attention_scope(dispatcher):
                        actual = attention._attention(q, k, v, [mask, bias])
                    exact = torch.equal(expected.contiguous().view(torch.int32), actual.contiguous().view(torch.int32))
                    launches = kernel.launches_submitted - before
                    assert exact and launches == int((batch, n) in CASES)
                    assert attention._attention is original
                    report['records'].append(dict(batch=batch, length=n, ending=ending,
                        mask=kind, exact=exact, launches=launches))
                    del storage, mask, expected, actual
                del pair, bias
            del q, k, v
        sentinel = RuntimeError('intentional restoration test')
        try:
            with attention_scope(dispatcher):
                raise sentinel
        except RuntimeError as error:
            assert error is sentinel
        assert attention._attention is original
        report['exception_restoration'] = True
    name = 'FIREPROT_SOFTMAX_GFX1100_BUNDLE_SHA256'
    original_pin = os.environ[name]
    try:
        os.environ[name] = '0' * 64
        try:
            Gfx1100Softmax()
        except (ValueError, RuntimeError, OSError) as error:
            report['wrong_manifest_rejected'] = str(error)
        else:
            raise AssertionError('Wrong manifest pin was accepted')
    finally:
        os.environ[name] = original_pin
    report['status'] = 'passed'
finally:
    report['runtime'] = dispatcher.manifest()
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
print(json.dumps(dict(status=report['status'], checks=len(report['records']),
                      launches=kernel.launches_submitted)))

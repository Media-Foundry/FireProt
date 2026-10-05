"""Experimental removal of verified zero mask additions; native math retained."""
from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def zero_mask_probe(output):
    import torch
    import openfold3.core.model.primitives.attention as attention
    original = attention._attention
    enabled = os.environ.get('FIREPROT_SKIP_ZERO_MASK') == '1'
    report = dict(enabled=enabled, calls=0, checked=0, skipped=0,
                  nonzero=0, eligible_calls=0,
                  policy='FP32 H4 D32 N1000/N1536, no grad; inspect each mask value')

    def call(query, key, value, biases, use_high_precision=False):
        report['calls'] += 1
        filtered = biases
        if (enabled and not torch.is_grad_enabled() and query.is_cuda
                and query.dtype == key.dtype == value.dtype == torch.float32
                and query.ndim == 4 and query.shape[-3] == 4
                and query.shape[-1] == 32 and query.shape[-2] in (1000, 1536)
                and key.shape[-2] == query.shape[-2]):
            report['eligible_calls'] += 1
            filtered = []
            for bias in biases:
                if (bias.dtype == torch.float32 and bias.device == query.device
                        and bias.ndim == query.ndim
                        and tuple(bias.shape[-3:]) == (1, 1, key.shape[-2])):
                    report['checked'] += 1
                    # Includes NaN/Inf as nonzero; no cached pointer-based assumption.
                    if torch.count_nonzero(bias).item() == 0:
                        report['skipped'] += 1
                        continue
                    report['nonzero'] += 1
                filtered.append(bias)
        return original(query, key, value, filtered, use_high_precision=use_high_precision)

    attention._attention = call
    try:
        yield
    finally:
        attention._attention = original
        Path(output, 'zero-mask-audit.json').write_text(json.dumps(report,indent=2)+'\n')

"""Experimental exact duplicate-template reuse; native reductions stay intact.

Only identical FP32 template slots in the measured N1536 inference bucket are
eligible. The native block computes the first slot and its exact bytes populate
the other slots. The stack's final normalization and four-slot reduction remain
native. Distinct templates, training and unsupported calls use the original.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path

import torch


@contextmanager
def template_reuse_scope(model, output, source_sha256):
    stack = model.template_embedder.template_pair_stack
    source = Path(inspect.getsourcefile(type(stack)))
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError('Template source pin differs')
    originals = []
    counts = dict(eligible=0, distinct=0, unsupported=0, reused_slots=0)

    def install(block):
        original = block.forward
        had_local = 'forward' in block.__dict__
        local = block.__dict__.get('forward')

        def forward(t, mask, *args, **kwargs):
            supported = (
                not args and not block.training
                and torch.is_inference_mode_enabled()
                and torch.are_deterministic_algorithms_enabled()
                and not torch.is_deterministic_algorithms_warn_only_enabled()
                and kwargs.get('inplace_safe') is True
                and type(t) is torch.Tensor and type(mask) is torch.Tensor
                and t.is_cuda and mask.device == t.device
                and t.dtype == mask.dtype == torch.float32
                and tuple(t.shape) == (1, 4, 1536, 1536, 64)
                and tuple(mask.shape) == (1, 4, 1536, 1536)
                and t.is_contiguous() and mask.stride(-1) == 1
                and not t.requires_grad and not mask.requires_grad
                and not t.is_neg() and not mask.is_neg()
                and not t.is_conj() and not mask.is_conj()
                and not any(kwargs.get(k, False) for k in (
                    'use_deepspeed_evo_attention', 'use_cueq_triangle_kernels',
                    'use_triton_triangle_kernels', 'use_lma'))
            )
            if not supported:
                counts['unsupported'] += 1
                return original(t, mask, *args, **kwargs)
            # Integer views compare signed zero and NaN payloads as actual bits.
            first = t[:, :1]
            first_mask = mask[:, :1]
            same = all(
                torch.equal(first.view(torch.int32), t[:, i:i+1].view(torch.int32))
                and torch.equal(first_mask.view(torch.int32), mask[:, i:i+1].view(torch.int32))
                for i in range(1, 4)
            )
            if not same:
                counts['distinct'] += 1
                return original(t, mask, *args, **kwargs)
            value = original(first, first_mask, **kwargs)
            if value.data_ptr() != first.data_ptr():
                raise RuntimeError('Native in-place template block changed aliasing')
            for i in range(1, 4):
                t[:, i:i+1].copy_(value)
            counts['eligible'] += 1
            counts['reused_slots'] += 3
            return t

        block.forward = forward
        originals.append((block, had_local, local))

    try:
        for block in stack.blocks:
            install(block)
        yield
    finally:
        for block, had_local, local in reversed(originals):
            if had_local:
                block.forward = local
            else:
                delattr(block, 'forward')
        Path(output, 'template-reuse.json').write_text(json.dumps(dict(
            source_sha256=source_sha256, counts=counts,
            policy='Bitwise-identical input slots and masks; native final reduction',
        ), indent=2) + '\n')

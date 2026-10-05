"""Preserve the measured N1536 native confidence chunk ceiling during reuse.

This is an experimental, source-pinned allocation policy for fresh model runs.
It changes neither recycle/step counts nor precision. Exact model validation is
still mandatory; a chunk ceiling alone does not prove arithmetic equivalence.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import torch

SOURCE_PINS = {
    'chunk_utils.py': '4c9ab484854868e3c3d9a28a0d2e8fe0f9b1c75a35dca4984e0631591489fac5',
    'pairformer.py': 'cdb66c9553ef42b93835dec92f3e6dcf711dbf139a5a59aa1f2561216435af9f',
}


def validate_sources(stack, tuner):
    for function in (type(stack)._prep_blocks, type(tuner).tune_chunk_size):
        path = Path(inspect.getfile(function)).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_PINS.get(path.name):
            raise RuntimeError('Unsupported confidence chunk source')


def chunk_limit(args, requested):
    if (len(args) == 2 and all(type(t) is torch.Tensor for t in args)
            and all(t.dtype == torch.float32 for t in args)
            and args[0].ndim >= 2 and args[0].shape[-2] == 1536
            and args[1].ndim >= 3 and tuple(args[1].shape[-3:-1]) == (1536, 1536)):
        return min(requested, 256)
    return requested


@contextmanager
def confidence_chunk_cap(model, report):
    stack = model.aux_heads.pairformer_embedding.pairformer_stack
    tuner = stack.chunk_size_tuner
    if (tuner is None or stack.training or torch.is_grad_enabled()
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError('Confidence chunk policy requires strict inference')
    validate_sources(stack, tuner)
    if tuner.cached_arg_data is not None or tuner.cached_chunk_size is not None:
        raise RuntimeError('Confidence chunk policy requires a fresh unused tuner')
    original = tuner.tune_chunk_size
    had_local = 'tune_chunk_size' in tuner.__dict__
    local = tuner.__dict__.get('tune_chunk_size')
    report.update(mode='measured_native_confidence_ceiling', calls=[])

    def call(representative_fn, args, max_chunk_size):
        maximum = chunk_limit(args, max_chunk_size)
        actual = original(representative_fn, args, maximum)
        report['calls'].append(dict(requested_max=max_chunk_size, bounded_max=maximum,
                                    selected=actual, n=int(args[1].shape[-2])))
        return actual

    tuner.tune_chunk_size = call
    try:
        yield
    finally:
        if had_local:
            tuner.tune_chunk_size = local
        else:
            del tuner.tune_chunk_size

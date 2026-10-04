"""Lazy HIP implementation of sequential BF16 mask and bias addition.

The kernel mutates contiguous [1, outer_chunk, 4, N, N] logits. Each addition
rounds to BF16 independently, matching native in-place Torch arithmetic.
"""
from __future__ import annotations

from fireprot._paths import KERNEL_ROOT

import os
from pathlib import Path
import threading

_lock = threading.RLock()
_extension = None
_error = None
_name = None
_build = None


def load():
    """Return the isolated extension, or None so the adapter retains native math."""
    global _extension, _error, _name, _build
    with _lock:
        if _extension is not None or _error is not None:
            return _extension
        try:
            import torch
            if os.environ.get('FIREPROT_DISABLE_MASK_BIAS_HIP') == '1':
                raise RuntimeError('disabled by FIREPROT_DISABLE_MASK_BIAS_HIP')
            if not torch.version.hip or not torch.cuda.is_available():
                raise RuntimeError('HIP is unavailable')
            bundle = os.environ.get('FIREPROT_MASK_BIAS_BUNDLE')
            if bundle:
                digest = os.environ.get('FIREPROT_MASK_BIAS_BUNDLE_SHA256')
                if not digest:
                    raise RuntimeError('FIREPROT_MASK_BIAS_BUNDLE_SHA256 is required')
                from ._published import load as published_load
                _extension, _build = published_load('fireprot_mask_bias', bundle, digest)
                _name = _extension.__name__
                return _extension
            from ._native_build import load as rust_load
            source = KERNEL_ROOT/'fireprot_mask_bias.cu'
            _extension, _build = rust_load('fireprot_mask_bias', [source])
            _name = _extension.__name__
        except Exception as error:
            _error = f'{type(error).__name__}: {error}'
        return _extension


def manifest():
    with _lock:
        return {'loaded': _extension is not None, 'name': _name, 'error': _error, 'build': _build}

"""Lazy gfx90a pair softmax and bit-preserving transpose kernels.

Softmax follows the validated Torch persistent wave64 reduction order.
Model-specific guards and native fallbacks belong to the adapter.
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
            if os.environ.get('FIREPROT_DISABLE_PAIR_FUSIONS_HIP') == '1':
                raise RuntimeError('disabled by FIREPROT_DISABLE_PAIR_FUSIONS_HIP')
            if not torch.version.hip or not torch.cuda.is_available():
                raise RuntimeError('HIP is unavailable')
            if str(torch.__version__) != '2.12.0a0+git78258b9' or torch.version.hip != '7.14.60850':
                raise RuntimeError('unvalidated Torch/HIP softmax implementation')
            from ._published import select_python
            published = select_python('fireprot_pair_fusions', 'FIREPROT_PAIR_FUSIONS')
            if published is not None:
                _extension, _build = published
                _name = _extension.__name__
                return _extension
            from ._native_build import load as rust_load
            sources = [KERNEL_ROOT/name for name in (
                'fireprot_pair_fusions.cpp', 'fireprot_pair_softmax.cu', 'fireprot_pair_transpose.cu',
                'fireprot_pair_qkv.cu')]
            _extension, _build = rust_load('fireprot_pair_fusions', sources)
            _name = _extension.__name__
        except Exception as error:
            _error = f'{type(error).__name__}: {error}'
        return _extension


def manifest():
    with _lock:
        return {'loaded': _extension is not None, 'name': _name, 'error': _error, 'build': _build}

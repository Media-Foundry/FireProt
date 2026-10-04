"""Lazy HIP implementation of sequential BF16 mask and bias addition.

The kernel mutates contiguous [1, outer_chunk, 4, N, N] logits. Each addition
rounds to BF16 independently, matching native in-place Torch arithmetic.
"""
from __future__ import annotations

import hashlib
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
            from torch.utils.cpp_extension import load as compile_extension
            source = Path(__file__).with_name('csrc')/'fireprot_mask_bias.cu'
            if os.environ.get('FIREPROT_RUST_BUILD') == '1':
                from ._native_build import load as rust_load
                _extension, _build = rust_load('fireprot_mask_bias', [source])
                _name = _extension.__name__
                return _extension
            digest = hashlib.sha256(source.read_bytes())
            digest.update(f'{torch.__version__}|{torch.version.hip}|gfx90a|-O3'.encode())
            _name = 'fireprot_mask_bias_' + digest.hexdigest()[:16]
            cache = Path(os.environ.get('FIREPROT_EXTENSION_CACHE',
                         str(Path(__file__).resolve().parents[3]/'runtime/torch_extensions')))
            build = cache/_name
            build.mkdir(parents=True, exist_ok=True)
            _extension = compile_extension(name=_name, sources=[str(source)],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--offload-arch=gfx90a'],
                build_directory=str(build), with_cuda=True,
                verbose=os.environ.get('FIREPROT_EXTENSION_VERBOSE') == '1')
        except Exception as error:
            _error = f'{type(error).__name__}: {error}'
        return _extension


def manifest():
    with _lock:
        return {'loaded': _extension is not None, 'name': _name, 'error': _error, 'build': _build}

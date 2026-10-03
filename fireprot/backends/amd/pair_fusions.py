"""Lazy gfx90a pair softmax and bit-preserving transpose kernels.

Softmax follows the validated Torch persistent wave64 reduction order.
Model-specific guards and native fallbacks belong to the adapter.
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


def load():
    """Return the isolated extension, or None so the adapter retains native math."""
    global _extension, _error, _name
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
            from torch.utils.cpp_extension import load as compile_extension
            sources = [Path(__file__).with_name('csrc')/name for name in (
                'fireprot_pair_fusions.cpp', 'fireprot_pair_softmax.cu', 'fireprot_pair_transpose.cu')]
            digest = hashlib.sha256()
            for source in sources:
                digest.update(source.name.encode())
                digest.update(source.read_bytes())
            digest.update(f'{torch.__version__}|{torch.version.hip}|gfx90a|-O3'.encode())
            _name = 'fireprot_pair_fusions_' + digest.hexdigest()[:16]
            cache = Path(os.environ.get('FIREPROT_EXTENSION_CACHE',
                         str(Path(__file__).resolve().parents[3]/'runtime/torch_extensions')))
            build = cache/_name
            build.mkdir(parents=True, exist_ok=True)
            _extension = compile_extension(name=_name, sources=[str(source) for source in sources],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--offload-arch=gfx90a'],
                build_directory=str(build), with_cuda=True,
                verbose=os.environ.get('FIREPROT_EXTENSION_VERBOSE') == '1')
        except Exception as error:
            _error = f'{type(error).__name__}: {error}'
        return _extension


def manifest():
    with _lock:
        return {'loaded': _extension is not None, 'name': _name, 'error': _error}

"""Opt-in C128 normalization model qualification; unmatched calls remain native."""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.utils.deterministic

from .screen import Kernel
from .screen_strided import StridedKernel


@contextmanager
def normalization_scope(model, output, library, library_sha256, strided_library, strided_sha256):
    from openfold3.core.model.primitives.normalization import LayerNorm
    source = Path(inspect.getsourcefile(LayerNorm))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != 'e39b9599c1711a232b7a52767732920e45e069dc37850192090c1f59440b3d79':
        raise RuntimeError('Upstream normalization source pin differs')
    if hashlib.sha256(Path(library).read_bytes()).hexdigest() != library_sha256:
        raise RuntimeError('Qualified normalization library pin differs')
    if 'gfx90a' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('Normalization scope requires gfx90a')
    if hashlib.sha256(Path(strided_library).read_bytes()).hexdigest() != strided_sha256:
        raise RuntimeError("Strided normalization library pin differs")
    strided_kernel = StridedKernel(strided_library)
    kernel = Kernel(library, 32)
    counts = dict(accepted=0, fallback=0, contiguous_copies=0, strided_calls=0)
    signatures = {}
    originals = []

    def install(module):
        original = module.forward
        had_local = 'forward' in module.__dict__
        local = module.__dict__.get('forward')

        def forward(x, *args, **kwargs):
            supported = (
                not args and not kwargs and not module.training
                and torch.is_inference_mode_enabled()
                and torch.are_deterministic_algorithms_enabled()
                and not torch.is_deterministic_algorithms_warn_only_enabled()
                and torch.utils.deterministic.fill_uninitialized_memory
                and type(x) is torch.Tensor and x.is_cuda and x.dtype == torch.float32
                and x.ndim >= 2 and x.shape[-1] == 128
                and x.numel() // 128 in (393216, 2359296)
                and not x.is_neg() and not x.is_conj() and module.eps == 1e-5
            )
            if supported:
                for p in (module.weight, module.bias):
                    if p is not None and (p.shape != (128,) or p.dtype != x.dtype
                        or p.device != x.device or not p.is_contiguous() or p.data_ptr() % 16):
                        supported = False
                if x.is_contiguous() and x.data_ptr() % 16:
                    supported = False
            if not supported:
                counts['fallback'] += 1
                return original(x, *args, **kwargs)
            key = str((tuple(x.shape), tuple(x.stride())))
            signatures[key] = signatures.get(key, 0) + 1
            counts['accepted'] += 1
            if not x.is_contiguous():
                layout = (tuple(x.shape), tuple(x.stride()))
                qualified = {
                    ((1, 1536, 256, 128), (301989888, 196608, 128, 1)),
                    ((1, 256, 1536, 128), (150994944, 128, 98304, 1)),
                    ((1, 1536, 1536, 128), (301989888, 128, 196608, 1)),
                }
                if layout in qualified and x.data_ptr() % 16 == 0:
                    counts['strided_calls'] += 1
                    return strided_kernel(x, module.weight, module.bias)
                counts['contiguous_copies'] += 1
            # Native LayerNorm also materializes noncontiguous input first.
            return kernel(x.contiguous(), module.weight, module.bias)

        module.forward = forward
        originals.append((module, had_local, local))

    try:
        for module in model.modules():
            if isinstance(module, LayerNorm) and module.c_in == (128,):
                install(module)
        yield
    finally:
        for module, had_local, local in reversed(originals):
            if had_local:
                module.forward = local
            else:
                delattr(module, 'forward')
        Path(output, 'normalization.json').write_text(json.dumps(dict(
            source_sha256=source_sha, library_sha256=library_sha256,
            strided_library_sha256=strided_sha256,
            counts=counts, signatures=signatures, width=32,
        ), indent=2) + '\n')

# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Outpace Bio, Inc.
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0; see licenses/Apache-2.0.txt.
# FireProt modification: guarded N1000 gfx1100 copy substitutions.
"""Experimental N1000 triangle layout copies; preserve native arithmetic."""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import threading

import torch
import torch.utils.deterministic

if __package__:
    from .copy_runtime import CopyRuntime
else:
    from copy_runtime import CopyRuntime


class Gfx1100Copy(CopyRuntime):
    def __init__(self, library, sha256):
        if hashlib.sha256(Path(library).read_bytes()).hexdigest() != sha256:
            raise ValueError('Triangle copy library digest differs')
        if (str(torch.__version__) != '2.13.0+rocm10.0.0'
                or torch.version.hip != '7.15.26333'
                or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1100'):
            raise RuntimeError('Unqualified triangle copy environment')
        super().__init__(library)

    @staticmethod
    def layout(t):
        shape, strides = list(t.shape), list(t.stride())
        if len(shape) == 4 and shape[0] == 1:
            shape, strides = shape[1:], strides[1:]
        if len(shape) != 3 or sorted(shape) not in ([128, 232, 1000], [128, 256, 1000]):
            return None
        extent = 1
        for n, s in sorted(zip(shape, strides), key=lambda v: v[1]):
            if s < extent:
                return None
            extent = s * n
        return shape, strides


def patched(runtime):
    from openfold3.core.model.layers import triangular_multiplicative_update as native
    cls = native.TriangleMultiplicativeUpdate
    path = Path(inspect.getsourcefile(cls))
    if hashlib.sha256(path.read_bytes()).hexdigest() != '612caf7abd07a86a7fdf4eda7e0a2170c04b407f9121fc286ab864a648db1ba7':
        raise RuntimeError('Triangle source differs from the audited implementation')
    source = textwrap.dedent(inspect.getsource(cls._inference_forward))
    replacements = {
        'p[..., i : i + inplace_chunk_size] = pair_chunk':
            '_copy.copy_(p[..., i : i + inplace_chunk_size], pair_chunk)',
        'p[..., i : i + inplace_chunk_size, :] = pair_chunk':
            '_copy.copy_(p[..., i : i + inplace_chunk_size, :], pair_chunk)',
        'else:\n                x_chunk = torch.einsum("...ij,...jk->...ik", a, b_chunk)':
            'else:\n                x_chunk = torch.einsum("...ij,...jk->...ik", a, _copy.contiguous(b_chunk))',
        'x_chunk = self.layer_norm_out(x_chunk)':
            'x_chunk = self.layer_norm_out(_copy.contiguous(x_chunk))',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Expected triangle substitution site differs')
        source = source.replace(old, new)
    namespace = dict(vars(native), _copy=runtime)
    exec(compile(source, '<source-pinned-gfx1100-triangle-copy>', 'exec'), namespace)
    return cls, namespace['_inference_forward']


@contextmanager
def triangle_copy_scope(model, output, library, sha256, enabled):
    runtime = Gfx1100Copy(library, sha256)
    cls, candidate = patched(runtime)
    original = cls._inference_forward
    owner = threading.get_ident()
    counts = dict(accepted=0, fallback=0)
    completed = False

    def forward(self, z, mask=None, inplace_chunk_size=None, with_add=True,
                use_triton_triangle_kernels=False):
        eligible = (enabled and threading.get_ident() == owner
            and type(z) is torch.Tensor and z.is_cuda and z.device.index == 0
            and tuple(z.shape) == (1, 1000, 1000, 128) and z.dtype == torch.float32
            and z.is_contiguous() and not z.is_neg() and not z.is_conj()
            and not self.training and torch.is_inference_mode_enabled()
            and not torch.is_grad_enabled() and torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled()
            and torch.utils.deterministic.fill_uninitialized_memory
            and inplace_chunk_size == 256 and with_add
            and not use_triton_triangle_kernels and self.c_hidden == 128)
        counts['accepted' if eligible else 'fallback'] += 1
        function = candidate if eligible else original
        return function(self, z, mask, inplace_chunk_size, with_add, use_triton_triangle_kernels)

    cls._inference_forward = forward
    try:
        yield
        if enabled and (counts['accepted'] == 0 or runtime.calls == 0):
            raise RuntimeError('Triangle copy candidate had no HIP coverage')
        completed = True
    finally:
        cls._inference_forward = original
        Path(output, 'triangle-copy.json').write_text(json.dumps(dict(
            enabled=enabled, completed=completed, library_sha256=sha256,
            counts=counts, copies=runtime.calls, copy_fallbacks=runtime.fallbacks), indent=2)+'\n')

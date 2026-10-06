"""Opt-in diagnostic model scope; unsupported triangle calls remain native."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from .copy_runtime import CopyRuntime
from .replay import patched, native


@contextmanager
def triangle_scope(model, output, library):
    if not torch.is_inference_mode_enabled():
        raise RuntimeError('Triangle scope requires inference mode')
    runtime = CopyRuntime(library)
    candidate = patched(runtime)
    cls = native.TriangleMultiplicativeUpdate
    original = cls._inference_forward
    accepted = fallback = 0

    def forward(self, z, mask=None, inplace_chunk_size=None, with_add=True,
                use_triton_triangle_kernels=False):
        nonlocal accepted, fallback
        if (tuple(z.shape)==(1,1536,1536,128) and z.dtype==torch.float32
            and z.is_contiguous() and inplace_chunk_size==256 and with_add
            and not use_triton_triangle_kernels and self.c_hidden==128):
            accepted += 1
            return candidate(self,z,mask,inplace_chunk_size,with_add,False)
        fallback += 1
        return original(self,z,mask,inplace_chunk_size,with_add,use_triton_triangle_kernels)

    cls._inference_forward = forward
    try:
        yield
    finally:
        cls._inference_forward = original
        Path(output,'triangle-copy.json').write_text(json.dumps(dict(
            accepted=accepted,fallback=fallback,copies=runtime.calls,
            copy_fallbacks=runtime.fallbacks),indent=2)+'\n')

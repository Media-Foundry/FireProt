"""Opt-in triangle attention adapter for the pinned AtlasFold monomer.

Native model loading, AtlasLM, sampling, and confidence collection stay upstream.
Unsupported calls retain the original method. An unavailable HIP build uses the
bounded-memory native-math reference and records that selection explicitly.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import types

SOURCE_HASHES = {
    'model/network/primitives/triangle_update.py': '11e36afda7255016be645c647ad77b9911485fe801bd52c8d60f9a0c68f3bcc9',
    'model/network/primitives/linear.py': '3f5514681e542a38a2e3a7154190fefcc1d93fa5b819a2e84b9e8b4de05217fa',
    'model/network/primitives/normalization.py': '33a08703e83f76ca35c1a85319cf2333d061fc3b62386a074b37403bbe80062e',
    'model/model.py': 'bdb34a36ae217cc2cf107ddae3fc1c2f319d20c2591b41ae8bbc13c2ff13c8df',
    'runner.py': 'b5d8af2e6f48d416cd2a73509a8d887b2bcb26e0677a9ca0f70a8d9419b73c7f',
}


def verify_source(root: Path):
    for relative, expected in SOURCE_HASHES.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Unsupported AtlasFold source: {relative}')


def triangle_attention(module, z, mask, *, ending, extension, valid_length, chunk_size=128, fuse_softmax=False):
    """Original attention arithmetic with bounded outer-axis temporary storage."""
    import torch
    if ending:
        z = z.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
    z = module.layernorm(z)
    batch, length, _, channels = z.shape
    heads, width = module.num_heads, module.channel_hidden
    qkv = module.linear_qkv(z)
    q, k, v = qkv.chunk(3, dim=-1)
    q, k, v = [t.reshape(batch, length, length, heads, width).transpose(-2, -3)
               for t in (q, k, v)]
    bias = module.linear_bias(z).permute(0, 3, 1, 2).unsqueeze(1)
    if extension is not None:
        bias = bias.contiguous()
    mask = mask[:, :, None, None, :]
    gate = module.linear_g(z).sigmoid().reshape(batch, length, length, heads, width)
    q *= module.scale
    if extension is not None and fuse_softmax:
        # Scaling remains native; packing copies BF16 bits into native BLAS layouts.
        q, k, v = extension.pack_qkv(qkv)
    del qkv
    chunks = []
    for offset in range(0, length, chunk_size):
        stop = min(offset + chunk_size, length)
        logits = q[:, offset:stop] @ k[:, offset:stop].transpose(-1, -2)
        if extension is not None and fuse_softmax:
            probabilities = extension.mask_softmax(logits, bias, offset, valid_length, 2)
        else:
            if extension is None:
                logits += (-module.inf) * (~mask[:, offset:stop].bool()).to(q.dtype)
                logits += bias
            else:
                extension.mask_bias_(logits, bias, offset, valid_length)
            probabilities = logits.softmax(-1)
        chunks.append((probabilities @ v[:, offset:stop]).transpose(-2, -3))
        del logits, probabilities
    output = module.linear_out((torch.cat(chunks, dim=1) * gate).flatten(-2))
    return output.transpose(-2, -3) if ending else output


def triangle_multiplication(module, z, mask, *, outgoing, extension):
    """Pack copies with HIP while retaining native physical BLAS operand layouts."""
    z = module.layernorm_in(z)
    z = z * mask.unsqueeze(-1)
    projections = module.linear_g_in(z).sigmoid() * module.linear_in(z)
    n = z.shape[-2]
    packed = extension.transpose_bf16(projections.reshape(n*n, 256)).reshape(1, 256, n, n)
    a, b = packed.chunk(2, dim=1)
    gate = module.linear_g_out(z).sigmoid()
    # Materialize the transposed operand in the native physical BLAS layout.
    product = (a @ extension.batched_transpose(b[0]).unsqueeze(0) if outgoing
               else extension.batched_transpose(a[0]).unsqueeze(0) @ b)
    product = extension.transpose_bf16(product.reshape(128, n*n)).reshape(1, n, n, 128)
    return gate * module.linear_out(module.layernorm_out(product))


class AtlasFoldRuntime:
    """Reversible instance hooks; no global class or upstream source mutations."""
    def __init__(self, model, sequence_length, backend='hip', *, allow_long1536=False):
        import torch
        import atlasfold
        from atlasfold.model.model import AtlasFold
        from atlasfold.model.network.primitives.triangle_update import (
            TriangleAttentionStartingNode, TriangleAttentionEndingNode,
            TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)
        from atlasfold.model.network.primitives.normalization import LayerNorm
        from atlasfold.model.network.primitives.linear import LinearNoBias
        if backend not in ('hip', 'chunked', 'hip-fused'):
            raise ValueError('backend must be hip, hip-fused or chunked')
        if sequence_length not in (1000, 1536) or (sequence_length == 1536 and not allow_long1536):
            raise ValueError('Only N1000 and explicit experimental N1536 are supported')
        verify_source(Path(atlasfold.__file__).resolve().parent)
        if type(model) is not AtlasFold or model.cfg.name != 'atlasfold-260703' or model.training:
            raise ValueError('Expected the pinned AtlasFold 260703 monomer in eval mode')
        if getattr(model, '_fireprot_atlasfold_runtime', None) is not None:
            raise ValueError('A FireProt AtlasFold runtime is already installed')
        self.backend = backend
        self.normalization_type = LayerNorm
        self.linear_type = LinearNoBias
        backend_root = Path(__file__).resolve().parents[2]/'backends/amd'
        sources = (backend_root/'pair_fusions.py', backend_root/'csrc/fireprot_pair_fusions.cpp',
                   backend_root/'csrc/fireprot_pair_softmax.cu', backend_root/'csrc/fireprot_pair_transpose.cu',
                   backend_root/'csrc/fireprot_pair_qkv.cu') if backend == 'hip-fused' else (
                       backend_root/'mask_bias.py', backend_root/'csrc/fireprot_mask_bias.cu')
        self.implementation_hashes = {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), *sources)}
        self.valid_length = sequence_length
        self.bucket = 1024 if sequence_length == 1000 else 1536
        self.counts = {'hip_attention': 0, 'chunked_attention': 0, 'native_fallback': 0,
                       'fused_attention': 0, 'hip_multiplication': 0, 'native_multiplication': 0}
        self.originals = []
        self.extension = None
        if backend == 'hip':
            from fireprot.backends.amd.mask_bias import load
            self.extension = load()
        if backend == 'hip-fused':
            from fireprot.backends.amd.pair_fusions import load
            self.extension = load()
        self.expected_mask = None
        types_to_hook = (TriangleAttentionStartingNode, TriangleAttentionEndingNode)
        multiplication_types = (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)
        if backend == 'hip-fused':
            types_to_hook += multiplication_types
        for module in model.modules():
            if type(module) not in types_to_hook:
                continue
            if 'forward' in module.__dict__ or hasattr(type(module).forward, '__wrapped__'):
                raise ValueError('Refusing an already modified triangle forward')
            if Path(type(module).forward.__code__.co_filename).resolve() != (
                    Path(atlasfold.__file__).resolve().parent/'model/network/primitives/triangle_update.py'):
                raise ValueError('Refusing a replaced triangle class method')
        for module in model.modules():
            if type(module) not in types_to_hook:
                continue
            original = module.forward
            ending = type(module) is TriangleAttentionEndingNode
            multiplication = type(module) in multiplication_types
            outgoing = type(module) is TriangleMultiplicationOutgoing
            def forward(current, z, mask, kernel_backend='torch', _original=original,
                        _ending=ending, _multiplication=multiplication, _outgoing=outgoing):
                if not self.supported(current, z, mask, kernel_backend, multiplication=_multiplication):
                    self.counts['native_fallback'] += 1
                    return _original(z, mask, kernel_backend)
                if _multiplication:
                    if self.extension is None:
                        self.counts['native_multiplication'] += 1
                        return _original(z, mask, kernel_backend)
                    self.counts['hip_multiplication'] += 1
                    return triangle_multiplication(current, z, mask, outgoing=_outgoing, extension=self.extension)
                route = ('fused_attention' if self.backend == 'hip-fused' else 'hip_attention') if self.extension is not None else 'chunked_attention'
                self.counts[route] += 1
                return triangle_attention(current, z, mask, ending=_ending,
                    extension=self.extension, valid_length=self.valid_length, fuse_softmax=self.backend == 'hip-fused')
            module.forward = types.MethodType(forward, module)
            self.originals.append((module, original))
        model._fireprot_atlasfold_runtime = self
        self.model = model

    def supported(self, module, z, mask, kernel_backend, *, multiplication=False):
        import torch
        if (kernel_backend != 'torch' or module.training or torch.is_grad_enabled()
                or z.requires_grad or not z.is_cuda or not torch.version.hip
                or z.dtype != torch.bfloat16 or tuple(z.shape) != (1, self.bucket, self.bucket, 128)
                or mask.device != z.device or mask.dtype != torch.bool
                or tuple(mask.shape) != (1, self.bucket, self.bucket)
                or not torch.is_autocast_enabled('cuda')
                or torch.get_autocast_dtype('cuda') != torch.bfloat16):
            return False
        if torch.cuda.get_device_properties(z.device).gcnArchName.split(':')[0] != 'gfx90a':
            return False
        if multiplication:
            norms = (module.layernorm_in, module.layernorm_out)
            linears = (module.linear_in, module.linear_g_in, module.linear_out, module.linear_g_out)
            if module.linear_in.weight.shape != (256, 128) or module.linear_out.weight.shape != (128, 128):
                return False
        else:
            if (module.num_heads != 4 or module.channel_hidden != 32 or module.inf != 1e9
                    or module.scale != 1.0 / math.sqrt(32)):
                return False
            norms = (module.layernorm,)
            linears = (module.linear_qkv, module.linear_bias, module.linear_g, module.linear_out)
        if any(type(norm) is not self.normalization_type for norm in norms):
            return False
        for linear in linears:
            if type(linear) is not self.linear_type or linear.precision is not None or linear.bias is not None:
                return False
        hooks = torch.nn.modules.module
        if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
            return False
        if any(child._forward_hooks or child._forward_pre_hooks or 'forward' in child.__dict__
               for child in module.children()):
            return False
        if self.expected_mask is None or self.expected_mask.device != mask.device:
            valid = torch.arange(self.bucket, device=mask.device) < self.valid_length
            self.expected_mask = valid[None, :, None] & valid[None, None, :]
        return torch.equal(mask, self.expected_mask)

    def manifest(self):
        if self.backend == 'hip-fused':
            from fireprot.backends.amd.pair_fusions import manifest
        else:
            from fireprot.backends.amd.mask_bias import manifest
        return {'adapter': 'atlasfold', 'requested_backend': self.backend,
                'valid_length': self.valid_length, 'bucket_length': self.bucket,
                'implementation_sha256': dict(self.implementation_hashes),
                'chunk_size': 128, 'counts': dict(self.counts),
                'extension': manifest() if self.backend in ('hip', 'hip-fused') else None}

    def uninstall(self):
        for module, original in self.originals:
            del module.forward
        self.originals.clear()
        if getattr(self.model, '_fireprot_atlasfold_runtime', None) is self:
            del self.model._fireprot_atlasfold_runtime

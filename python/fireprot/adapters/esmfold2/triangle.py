"""Experimental reuse of right-hand packing within one triangular contraction.

The chunk size, contraction order and native autocast dtype are retained. No
cross-loop tensor cache is introduced. Rust owns capability selection. This
candidate requires operator and full-model qualification before promotion.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import types

import torch

from fireprot.ops.registry import OperationRegistry, signature_from_tensor

LAYERS_SHA256 = '35f25607bd789d36fb4f46e184c3ac09f3c435538806d3531f42525bbdc372f2'


def packed_contract(left, right, chunk_size, flow):
    """Keep native chunked BMM geometry; pack its invariant right operand once."""
    batch, n, _, channels = left.shape
    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else left.dtype
    right_axes = (0, 3, 2, 1) if flow == 'outgoing' else (0, 3, 1, 2)
    rhs = right.to(dtype).permute(right_axes).contiguous().view(batch * channels, n, n)
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sl = left[:, start:end] if flow == 'outgoing' else left[:, :, start:end]
        left_axes = (0, 3, 1, 2) if flow == 'outgoing' else (0, 3, 2, 1)
        lhs = sl.to(dtype).permute(left_axes).contiguous().view(batch * channels, end - start, n)
        product = torch.bmm(lhs, rhs)
        chunks.append(product.view(batch, channels, end - start, n).permute(0, 2, 3, 1))
    return torch.cat(chunks, dim=1)


class TriangleDispatcher:
    def __init__(self):
        self.registry = OperationRegistry()
        self.calls = 0
        self.registry.register(operation='esmfold2_triangle_contract',
            name='reuse_right_pack', function=self._run, priority=10, version='experimental-v1',
            supports=lambda s: (dict(s.attributes).get('eligible') == 'True',
                                'Unqualified tensor, dtype, layout, chunk or device'))

    def _run(self, left, right, chunk_size, flow):
        self.calls += 1
        return packed_contract(left, right, chunk_size, flow)

    def __call__(self, native, left, right, chunk_size, flow):
        if type(left) is not torch.Tensor or type(right) is not torch.Tensor:
            return native(left, right, chunk_size)
        architecture = torch.cuda.get_device_properties(left.device).gcnArchName.split(':')[0] if left.is_cuda else 'cpu'
        eligible = (not torch.is_grad_enabled() and left.is_cuda and right.device == left.device
                    and architecture == 'gfx90a' and left.dtype == right.dtype == torch.float32
                    and left.ndim == 4 and left.shape == right.shape
                    and left.shape[0] == 1 and left.shape[1] == left.shape[2]
                    and left.shape[1] in (600, 1000) and left.shape[3] == 256
                    and chunk_size == 64 and flow in ('outgoing', 'incoming')
                    and not any(t.requires_grad or t.is_conj() or t.is_neg() for t in (left, right))
                    and str(torch.__version__) == '2.12.0a0+git78258b9'
                    and torch.version.hip == '7.14.60850')
        sig = signature_from_tensor('esmfold2_triangle_contract', left,
            architecture=architecture, mode='inference', mask='already_applied', inplace=False,
            attributes=dict(eligible=eligible, flow=flow, chunk_size=chunk_size,
                            right_strides=tuple(right.stride()),
                            autocast=str(torch.get_autocast_dtype('cuda')) if torch.is_autocast_enabled('cuda') else 'off'))
        return self.registry.dispatch(sig, lambda a, b, c, _: native(a, b, c),
                                      left, right, chunk_size, flow)

    def manifest(self):
        return dict(experimental=True, implementation='reuse_right_pack',
                    completed_dispatches=self.calls, registry=self.registry.manifest(),
                    scope='Trunk/LM encoder/coda/confidence/MSA triangle contractions only; diffusion unchanged.')


@contextmanager
def triangle_scope(model):
    import esm.models.esmfold2.layers as layers
    source = Path(layers.__file__).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != LAYERS_SHA256:
        raise RuntimeError('Unsupported ESMFold2 triangle source')
    dispatcher = TriangleDispatcher()
    changed = []
    try:
        for name, module in model.named_modules():
            if type(module) is not layers.TriangleMultiplicativeBlock:
                continue
            if not name.startswith(('folding_trunk.', 'lm_encoder.', 'parcae_coda.',
                                    'confidence_head.folding_trunk.', 'msa_encoder.')):
                continue
            method = '_triangular_contract_chunked'
            if method in module.__dict__:
                raise RuntimeError('ESMFold2 contraction already replaced')
            native = getattr(module, method)
            def call(self, left, right, chunk_size, _native=native):
                return dispatcher(_native, left, right, chunk_size, self.flow)
            replacement = types.MethodType(call, module)
            setattr(module, method, replacement)
            changed.append((module, method, replacement))
        if not changed:
            raise RuntimeError('No compatible ESMFold2 triangle modules')
        yield dispatcher
    finally:
        for module, method, replacement in reversed(changed):
            if getattr(module, method) is not replacement:
                raise RuntimeError('ESMFold2 triangle hook changed during scope')
            delattr(module, method)

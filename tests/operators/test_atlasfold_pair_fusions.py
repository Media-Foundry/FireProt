"""Exact fused pair arithmetic, trained operators, and safe dispatch contracts."""
import os
from types import SimpleNamespace

import pytest
import torch
from fireprot.adapters.atlasfold import runtime
from fireprot.backends.amd import pair_fusions

HIP = pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS') != '1' or not torch.version.hip,
                         reason='explicit HIP opt-in required')


def test_disabled_pair_extension(monkeypatch):
    monkeypatch.setattr(pair_fusions, '_extension', None)
    monkeypatch.setattr(pair_fusions, '_error', None)
    monkeypatch.setenv('FIREPROT_DISABLE_PAIR_FUSIONS_HIP', '1')
    assert pair_fusions.load() is None
    assert 'disabled' in pair_fusions.manifest()['error']


@HIP
@pytest.mark.parametrize('n', [1024, 1536])
@pytest.mark.parametrize('valid_kind', ['none', 'prefix', 'all'])
@pytest.mark.parametrize('last', [False, True])
@pytest.mark.parametrize('outer', [2, 8, 16])
def test_softmax_rounding_nonfinite_and_stream(n, valid_kind, last, outer):
    ext = pair_fusions.load()
    assert ext is not None, pair_fusions.manifest()
    valid = {'none': 0, 'prefix': 1000, 'all': n}[valid_kind]
    offset = n-outer if last else 0
    torch.manual_seed(1073)
    with torch.inference_mode():
        x = torch.randn(1, outer, 4, n, n, device='cuda', dtype=torch.bfloat16)*8
        bias = torch.randn(1, 1, 4, n, n, device='cuda', dtype=torch.bfloat16)*3
        x[0, 0, 0, 0, 0] = float('inf')
        x[0, 0, 0, 1, 0] = float('nan')
        x[0, 0, 0, 2, :] = -float('inf')
        mask = ((torch.arange(offset, offset+outer, device='cuda')[:, None] < valid)
                & (torch.arange(n, device='cuda')[None, :] < valid))
        ref = x.clone()
        ref += (-1e9)*(~mask[None, :, None, None, :]).to(x.dtype)
        ref += bias
        ref = ref.float().softmax(-1).bfloat16()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            output = ext.mask_softmax(x, bias, offset, valid, 2)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(output, ref, atol=0, rtol=0, equal_nan=True)
        with pytest.raises(RuntimeError):
            ext.mask_softmax(x.transpose(-1, -2), bias, offset, valid, 2)


@HIP
@pytest.mark.parametrize('shape', [(1, 1), (33, 65), (128, 1024*1024)])
def test_transpose_preserves_bits_and_stream(shape):
    ext = pair_fusions.load()
    assert ext is not None, pair_fusions.manifest()
    # Include every BF16 bit pattern, including signed zeros and NaN payloads.
    bits = torch.arange(shape[0]*shape[1], device='cuda', dtype=torch.int32).to(torch.int16).reshape(shape)
    x = bits.view(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = ext.transpose_bf16(x)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(y.view(torch.int16), bits.T.contiguous())


@HIP
@pytest.mark.parametrize('kind', ['tri_attn_start', 'tri_attn_end', 'tri_mul_out', 'tri_mul_in'])
def test_trained_operator_fused_dispatch_and_fallback(kind, monkeypatch):
    from atlasfold.model.model import AtlasFold
    from atlasfold.model.network.primitives.triangle_update import (
        TriangleAttentionStartingNode, TriangleAttentionEndingNode,
        TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)
    classes = dict(zip(['tri_attn_start', 'tri_attn_end', 'tri_mul_out', 'tri_mul_in'],
                       [TriangleAttentionStartingNode, TriangleAttentionEndingNode,
                        TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming]))
    multiplication = kind.startswith('tri_mul')
    model = AtlasFold.__new__(AtlasFold)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(name='atlasfold-260703')
    model.triangle = (classes[kind](128) if multiplication else classes[kind](128, 4)).cuda().bfloat16()
    state = torch.load('/media/PM982/engramfold/folding_e2e_20260921/weights/atlasfold/snapshot/weights/atlasfold-260703.pth',
                       map_location='cpu', weights_only=True, mmap=True)
    prefix = 'main_stack.blocks.0.'+kind+'.'
    model.triangle.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    model.eval()
    torch.manual_seed(1103)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        z = torch.randn(1, 1024, 1024, 128, device='cuda', dtype=torch.bfloat16)
        valid = torch.arange(1024, device='cuda') < 1000
        mask = valid[None, :, None] & valid[None, None, :]
        reference = model.triangle(z, mask)
        adapter = runtime.AtlasFoldRuntime(model, 1000, 'hip-fused')
        assert torch.equal(model.triangle(z, mask), reference)
        assert adapter.counts['hip_multiplication' if multiplication else 'fused_attention'] == 1
        child = model.triangle.linear_in if multiplication else model.triangle.linear_qkv
        hook = child.register_forward_hook(lambda *args: None)
        assert not adapter.supported(model.triangle, z, mask, 'torch', multiplication=multiplication)
        hook.remove()
        child.precision = torch.float32
        assert not adapter.supported(model.triangle, z, mask, 'torch', multiplication=multiplication)
        child.precision = None
        wrong = mask.clone(); wrong[0, 0, 0] = False
        assert not adapter.supported(model.triangle, z, wrong, 'torch', multiplication=multiplication)
        adapter.uninstall()
        monkeypatch.setattr(pair_fusions, 'load', lambda: None)
        fallback = runtime.AtlasFoldRuntime(model, 1000, 'hip-fused')
        assert torch.equal(model.triangle(z, mask), reference)
        assert fallback.counts['native_multiplication' if multiplication else 'chunked_attention'] == 1
        fallback.uninstall()
        assert 'forward' not in model.triangle.__dict__


def test_unvalidated_torch_version_falls_back(monkeypatch):
    monkeypatch.setattr(pair_fusions, '_extension', None)
    monkeypatch.setattr(pair_fusions, '_error', None)
    monkeypatch.delenv('FIREPROT_DISABLE_PAIR_FUSIONS_HIP', raising=False)
    monkeypatch.setattr(torch.version, 'hip', '7.14.60850')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch, '__version__', 'unvalidated')
    assert pair_fusions.load() is None
    assert 'unvalidated' in pair_fusions.manifest()['error']


@HIP
@pytest.mark.parametrize('n', [1024, 1536])
def test_batched_transpose_bits_stream_and_rejections(n):
    ext = pair_fusions.load()
    assert ext is not None, pair_fusions.manifest()
    bits = torch.arange(128*n*n, device='cuda', dtype=torch.int32).to(torch.int16).reshape(128, n, n)
    x = bits.view(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = ext.batched_transpose(x)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(y.view(torch.int16), bits.mT.contiguous())
    for invalid in (x.mT, x[:1], x[:, :32, :32].contiguous()):
        with pytest.raises(RuntimeError):
            ext.batched_transpose(invalid)

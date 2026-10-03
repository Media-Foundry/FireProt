"""Pinned-source, capability, reversible dispatch and HIP arithmetic contracts."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fireprot.adapters.atlasfold import runtime
from fireprot.backends.amd import mask_bias

HIP = pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS') != '1' or not torch.version.hip,
                         reason='explicit HIP opt-in required')


def test_changed_upstream_source_rejected(tmp_path, monkeypatch):
    source = tmp_path/'upstream.py'
    source.write_text('original\n')
    monkeypatch.setattr(runtime, 'SOURCE_HASHES', {'upstream.py': hashlib.sha256(source.read_bytes()).hexdigest()})
    runtime.verify_source(tmp_path)
    source.write_text('changed\n')
    with pytest.raises(RuntimeError, match='Unsupported AtlasFold source'):
        runtime.verify_source(tmp_path)


def test_disabled_extension_has_explicit_reason(monkeypatch):
    monkeypatch.setattr(mask_bias, '_extension', None)
    monkeypatch.setattr(mask_bias, '_error', None)
    monkeypatch.setenv('FIREPROT_DISABLE_MASK_BIAS_HIP', '1')
    assert mask_bias.load() is None
    assert 'disabled' in mask_bias.manifest()['error']


def test_cpu_call_is_not_supported():
    adapter = object.__new__(runtime.AtlasFoldRuntime)
    assert not adapter.supported(SimpleNamespace(training=False), torch.zeros(1), torch.ones(1), 'torch')


@HIP
@pytest.mark.parametrize('valid', [0, 8, 16])
@pytest.mark.parametrize('offset', [0, 8])
def test_hip_sequential_rounding_and_stream(valid, offset):
    extension = mask_bias.load()
    assert extension is not None, mask_bias.manifest()
    torch.manual_seed(1031)
    with torch.inference_mode():
        logits = torch.randn(1, 8, 4, 16, 16, device='cuda', dtype=torch.bfloat16)
        logits.flatten()[:8] = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1e9, -1e9, 1e-20],
                                           device='cuda', dtype=torch.bfloat16)
        bias = torch.randn(1, 1, 4, 16, 16, device='cuda', dtype=torch.bfloat16)
        mask = ((torch.arange(offset, offset+8, device='cuda')[:, None] < valid)
                & (torch.arange(16, device='cuda')[None, :] < valid))
        reference = logits.clone()
        reference += (-1e9) * (~mask[None, :, None, None, :]).to(logits.dtype)
        reference += bias
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            extension.mask_bias_(logits, bias, offset, valid)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(logits, reference, atol=0, rtol=0, equal_nan=True)
        with pytest.raises(RuntimeError):
            extension.mask_bias_(logits.transpose(-1, -2), bias, offset, valid)


@HIP
@pytest.mark.parametrize('ending', [False, True])
def test_actual_triangle_selection_mask_fallback_and_reinstall(ending):
    pytest.importorskip('atlasfold')
    from atlasfold.model.model import AtlasFold
    from atlasfold.model.network.primitives.triangle_update import TriangleAttentionStartingNode, TriangleAttentionEndingNode
    cls = TriangleAttentionEndingNode if ending else TriangleAttentionStartingNode
    model = AtlasFold.__new__(AtlasFold)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(name='atlasfold-260703')
    model.triangle = cls(128, 4).cuda().bfloat16()
    state = torch.load('/media/PM982/engramfold/folding_e2e_20260921/weights/atlasfold/snapshot/weights/atlasfold-260703.pth',
                       map_location='cpu', weights_only=True, mmap=True)
    prefix = 'main_stack.blocks.0.' + ('tri_attn_end.' if ending else 'tri_attn_start.')
    model.triangle.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    model.eval()
    torch.manual_seed(1049)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        z = torch.randn(1, 1024, 1024, 128, device='cuda', dtype=torch.bfloat16)
        valid = torch.arange(1024, device='cuda') < 1000
        mask = valid[None, :, None] & valid[None, None, :]
        reference = model.triangle(z, mask)
        adapter = runtime.AtlasFoldRuntime(model, 1000, 'hip')
        assert adapter.supported(model.triangle, z, mask, 'torch')
        output = model.triangle(z, mask)
        assert adapter.counts['hip_attention'] == 1
        assert torch.equal(output, reference)
        wrong_mask = mask.clone(); wrong_mask[0, 0, 0] = False
        assert not adapter.supported(model.triangle, z, wrong_mask, 'torch')
        hook = model.triangle.linear_qkv.register_forward_hook(lambda *args: None)
        assert not adapter.supported(model.triangle, z, mask, 'torch')
        hook.remove()
        model.triangle.linear_qkv.precision = torch.float32
        assert not adapter.supported(model.triangle, z, mask, 'torch')
        model.triangle.linear_qkv.precision = None
        model.triangle.train()
        assert not adapter.supported(model.triangle, z, mask, 'torch')
        model.triangle.eval()
        adapter.uninstall()
        assert 'forward' not in model.triangle.__dict__
        second = runtime.AtlasFoldRuntime(model, 1000, 'chunked')
        output = model.triangle(z, mask)
        assert second.counts['chunked_attention'] == 1
        assert torch.equal(output, reference)
        second.uninstall()

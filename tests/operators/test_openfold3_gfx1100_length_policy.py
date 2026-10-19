"""CPU metadata/lifecycle tests, not new GPU or full-model qualification."""
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest
import torch


class TensorMetadata:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        self.device = torch.device('cuda:0')
        self.dtype = torch.float32
        self.is_cuda = True
        self.requires_grad = False

    def is_neg(self):
        return False

    def is_conj(self):
        return False


@pytest.mark.parametrize('name', ['pair_cache', 'atom_cache'])
@pytest.mark.parametrize('length', [1, 124, 800, 1000, 1001, 1536, 1537])
def test_reuse_metadata_accepts_positive_lengths(monkeypatch, name, length):
    module = importlib.import_module('fireprot.adapters.openfold3.gfx1100.' + name)
    monkeypatch.setattr(module, 'torch', SimpleNamespace(Tensor=TensorMetadata, float32=torch.float32))
    inputs = dict(zij_trunk=TensorMetadata((1, 1, length, length, 128)),
                  si_trunk=TensorMetadata((1, 1, length, 384)),
                  rl=TensorMetadata((1, 1, length * 4, 3)),
                  use_conditioning=True, chunk_size=1024)
    assert module.eligible(inputs)
    inputs['zij_trunk'] = TensorMetadata((1, 1, length, length + 1, 128))
    assert not module.eligible(inputs)
    inputs['zij_trunk'] = TensorMetadata((1, 1, length, length, 64))
    assert not module.eligible(inputs)
    inputs['zij_trunk'] = TensorMetadata((1, 1, 0, 0, 128))
    assert not module.eligible(inputs)


class FakeCudaTensor:
    """CPU storage with CUDA metadata for testing branch conditions only."""
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape
        self.ndim = tensor.ndim
        self.dtype = tensor.dtype
        self.device = torch.device('cuda:0')
        self.is_cuda = True
        self.requires_grad = False

    def __getitem__(self, item):
        return type(self)(self.tensor[item])

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def copy_(self, other):
        self.tensor.copy_(other.tensor)
        return self


class FakeTorch:
    Tensor = FakeCudaTensor

    def __getattr__(self, name):
        if name in ('is_inference_mode_enabled', 'are_deterministic_algorithms_enabled'):
            return lambda: True
        if name == 'is_deterministic_algorithms_warn_only_enabled':
            return lambda: False
        return getattr(torch, name)


@pytest.mark.parametrize('length', [17, 31, 41])
@pytest.mark.parametrize('kind', ['same', 'different_slot', 'different_mask', 'signed_zero'])
def test_template_reuse_is_semantic_not_length_gated(monkeypatch, tmp_path, length, kind):
    module = importlib.import_module('fireprot.adapters.openfold3.gfx1100.template_reuse')
    monkeypatch.setattr(module, 'torch', FakeTorch())
    source = tmp_path / 'source.py'
    source.write_text('# pinned test source\n')
    monkeypatch.setattr(module.inspect, 'getsourcefile', lambda _: str(source))
    calls = []
    def native(t, mask, **kwargs):
        calls.append(t.shape[1])
        t.tensor.add_(1)
        return t
    block = SimpleNamespace(training=False, forward=native)
    model = SimpleNamespace(template_embedder=SimpleNamespace(
        template_pair_stack=SimpleNamespace(blocks=[block])))
    t = FakeCudaTensor(torch.zeros((1, 4, length, length, 64)))
    mask = FakeCudaTensor(torch.ones((1, 4, length, length)))
    if kind == 'different_slot':
        t.tensor[:, 2, 0, 0, 0] = 2
    elif kind == 'different_mask':
        mask.tensor[:, 2, 0, 0] = 0
    elif kind == 'signed_zero':
        t.tensor[:, 2, 0, 0, 0] = -0.0
    expected = t.tensor + 1
    with module.template_reuse_scope(model, tmp_path, hashlib.sha256(source.read_bytes()).hexdigest()):
        assert block.forward(t, mask, inplace_safe=True) is t
    assert block.forward is native
    assert torch.equal(t.tensor, expected)
    counts = json.loads((tmp_path / 'template-reuse.json').read_text())['counts']
    assert calls == ([1] if kind == 'same' else [4])
    assert counts['reused_slots'] == (3 if kind == 'same' else 0)

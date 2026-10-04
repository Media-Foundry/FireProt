"""AtlasFold boundary facts must invalidate cached Rust route decisions."""
from types import SimpleNamespace

import torch

from fireprot.adapters.atlasfold import runtime


def adapter(monkeypatch):
    instance = object.__new__(runtime.AtlasFoldRuntime)
    instance.backend = 'hip-fused'
    instance.extension = None
    instance.valid_length = 1000
    checked = []

    def supported(module, z, mask, kernel_backend, **kwargs):
        checked.append(bool(mask.item()))
        return bool(mask.item())

    instance.supported = supported
    instance._setup_registry()
    monkeypatch.setattr(runtime, 'triangle_attention',
                        lambda *a, extension, **k: 'chunked' if extension is None else 'fused')
    monkeypatch.setattr(runtime, 'triangle_multiplication', lambda *a, **k: 'hip multiplication')
    return instance, checked


def test_mask_values_and_extension_availability_are_checked_each_call(monkeypatch):
    instance, checked = adapter(monkeypatch)
    z, mask = torch.zeros(1), torch.ones(1, dtype=torch.bool)

    def native(actual_z, actual_mask, kernel):
        assert actual_z is z and actual_mask is mask and kernel == 'torch'
        return 'native'

    def call():
        return instance._dispatch(SimpleNamespace(), z, mask, 'torch', original=native,
                                  ending=False, multiplication=False, outgoing=False)

    assert call() == 'chunked'
    instance.extension = object()
    assert call() == 'fused'
    mask.zero_()
    assert call() == 'native'
    mask.fill_(True)
    assert call() == 'fused'
    assert checked == [True, True, False, True]
    assert instance.counts == dict(hip_attention=0, chunked_attention=1, native_fallback=1,
                                  fused_attention=2, hip_multiplication=0, native_multiplication=0)
    assert instance.registry.manifest()['policy_engine']['name'] == 'fireprot-core'


def test_missing_multiplication_extension_and_unsupported_call_have_distinct_records(monkeypatch):
    instance, _ = adapter(monkeypatch)
    z, mask = torch.zeros(1), torch.ones(1, dtype=torch.bool)

    def call():
        return instance._dispatch(SimpleNamespace(), z, mask, 'torch', original=lambda *a: 'native',
                                  ending=False, multiplication=True, outgoing=True)

    assert call() == 'native'
    mask.zero_()
    assert call() == 'native'
    mask.fill_(True)
    instance.extension = object()
    assert call() == 'hip multiplication'
    assert instance.counts['native_multiplication'] == 1
    assert instance.counts['native_fallback'] == 1
    assert instance.counts['hip_multiplication'] == 1

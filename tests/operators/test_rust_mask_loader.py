"""The experimental loader must expose Rust receipts and retain native fallback."""
from types import SimpleNamespace

import torch
import torch.utils.cpp_extension  # Resolve host build tools before mocking GPU availability.
from fireprot.backends.amd import mask_bias, _native_build


def prepare(monkeypatch):
    monkeypatch.setenv('FIREPROT_RUST_BUILD', '1')
    monkeypatch.delenv('FIREPROT_DISABLE_MASK_BIAS_HIP', raising=False)
    for field in ('_extension', '_error', '_name', '_build'):
        monkeypatch.setattr(mask_bias, field, None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.version, 'hip', 'test')


def test_rust_build_receipt_is_exposed(monkeypatch):
    prepare(monkeypatch)
    extension = SimpleNamespace(__name__='rust_test_extension')
    receipt = {'cache_hit': True, 'receipt': {'test': 'identity'}}
    calls = []
    def load(package, sources):
        calls.append((package, sources))
        return extension, receipt
    monkeypatch.setattr(_native_build, 'load', load)
    assert mask_bias.load() is extension
    assert mask_bias.load() is extension
    assert len(calls) == 1
    assert calls[0][0] == 'fireprot_mask_bias'
    assert mask_bias.manifest()['build'] is receipt
    assert mask_bias.manifest()['name'] == 'rust_test_extension'


def test_rust_build_failure_preserves_fallback(monkeypatch):
    prepare(monkeypatch)
    calls = []
    def fail(*args):
        calls.append(1)
        raise OSError('corrupt cached library')
    monkeypatch.setattr(_native_build, 'load', fail)
    assert mask_bias.load() is None
    assert mask_bias.load() is None
    assert len(calls) == 1
    assert mask_bias.manifest()['error'] == 'OSError: corrupt cached library'
    assert mask_bias.manifest()['build'] is None


def test_published_choice_precedes_build_and_requires_pin(monkeypatch):
    from fireprot.backends.amd import _published
    prepare(monkeypatch)
    monkeypatch.setenv('FIREPROT_MASK_BIAS_BUNDLE', '/declared/bundle')
    monkeypatch.delenv('FIREPROT_MASK_BIAS_BUNDLE_SHA256', raising=False)
    def forbidden(*args):
        raise AssertionError('must not enter build path')
    monkeypatch.setattr(_native_build, 'load', forbidden)
    assert mask_bias.load() is None
    assert 'SHA256 is required' in mask_bias.manifest()['error']
    monkeypatch.setattr(mask_bias, '_error', None)
    monkeypatch.setenv('FIREPROT_MASK_BIAS_BUNDLE_SHA256', 'a' * 64)
    extension = SimpleNamespace(__name__='published_extension')
    def published(package, directory, digest):
        assert (package, directory, digest) == ('fireprot_mask_bias', '/declared/bundle', 'a' * 64)
        return extension, {'mode': 'published'}
    monkeypatch.setattr(_published, 'load', published)
    assert mask_bias.load() is extension
    assert mask_bias.manifest()['build']['mode'] == 'published'

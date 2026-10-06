"""Boundary tests for the isolated Base template experiment, not GPU qualification."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fireprot.adapters.protenix import features

spec = spec_from_file_location('base_unused_templates', Path(__file__).resolve().parents[2]
                              / 'experiments/protenix_base/unused_templates.py')
experiment = module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.mark.parametrize('reason', ['allowed', 'disabled', 'used', 'blocks', 'dtype', 'source'])
def test_base_omission_is_limited_to_verified_unused_consumer(monkeypatch, reason):
    cfg = NS(model_name='protenix_base_default_v0.5.0', dtype='fp32', use_template=False,
             model=NS(template_embedder=NS(n_blocks=0)))
    monkeypatch.setenv('FIREPROT_SKIP_UNUSED_TEMPLATES', '0' if reason == 'disabled' else '1')
    monkeypatch.setattr(features, '_verified_consumer', lambda: reason != 'source')
    if reason == 'blocks':
        cfg.model.template_embedder.n_blocks = 2
    if reason == 'dtype':
        cfg.dtype = 'bf16'
    calls = []
    sentinel = object()
    def native():
        calls.append('native')
        return sentinel
    original = features._supported
    with experiment.install():
        result = features.build_template_features(configs=cfg, use_template=reason == 'used', native=native)
    assert features._supported is original
    if reason == 'allowed':
        assert result == {} and not calls
    else:
        assert result is sentinel and calls == ['native']


def test_other_models_delegate_and_exception_restores_hook(monkeypatch):
    calls = []
    def original(cfg, used):
        calls.append((cfg, used))
        return 'existing-policy'
    monkeypatch.setattr(features, '_supported', original)
    cfg = NS(model_name='protenix_mini_default_v0.5.0')
    with pytest.raises(ValueError, match='worker failed'):
        with experiment.install():
            assert features._supported(cfg, False) == 'existing-policy'
            raise ValueError('worker failed')
    assert features._supported is original and calls == [(cfg, False)]

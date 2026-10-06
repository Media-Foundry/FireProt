"""Opt-in workload boundaries and restoration of composed Base contexts."""
from contextlib import contextmanager
from types import SimpleNamespace
import json
import pytest
from fireprot.adapters.protenix import base_n1000 as base


def inputs(atoms=7979):
    request = dict(backend='fireprot', workload=dict(model='protenix',
        model_name='protenix_base_default_v0.5.0', precision='float32',
        cycles=10, diffusion_steps=200, samples=1))
    data = dict(input_feature_dict=dict(token_index=SimpleNamespace(shape=(1000,)),
                                      atom_to_token_idx=SimpleNamespace(shape=(atoms,))))
    return request, data


def test_default_off_preserves_other_workloads(tmp_path, monkeypatch):
    monkeypatch.delenv(base.FLAG, raising=False)
    with base.install(None, tmp_path, {}, {}):
        pass
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field,value', [('model_name', 'protenix_mini_default_v0.5.0'),
    ('precision', 'bfloat16'), ('cycles', 4), ('diffusion_steps', 5), ('samples', 5)])
def test_explicit_enable_rejects_unqualified_workloads(field, value):
    request, data = inputs()
    request['workload'][field] = value
    with pytest.raises(ValueError):
        base.validate_contract(request, data)


def test_native_and_non1000_not_silently_accelerated():
    request, data = inputs()
    request['backend'] = 'native'
    with pytest.raises(ValueError):
        base.validate_contract(request, data)
    request['backend'] = 'fireprot'
    data['input_feature_dict']['token_index'].shape = (600,)
    with pytest.raises(ValueError):
        base.validate_contract(request, data)


@pytest.mark.parametrize('atoms', [7979, 7711])
def test_atom_count_is_taken_from_features(atoms):
    assert base.validate_contract(*inputs(atoms)) == atoms


def test_failure_unwinds_all_contexts_and_records_error(tmp_path, monkeypatch):
    monkeypatch.setenv(base.FLAG, '1')
    events = []
    monkeypatch.setattr(base, '_select', lambda model, atoms: SimpleNamespace(
        name='test-context', function=base._components))
    names = ('triangle_lt', 'dual_storage', 'global_bias', 'local_bias', 'transition')
    def stub(name):
        @contextmanager
        def install(*args):
            if name == 'transition':
                assert args[2] == 7711
            events.append(('enter', name))
            try:
                yield
            finally:
                events.append(('exit', name))
        return SimpleNamespace(install=install)
    for name in names:
        monkeypatch.setattr(base, name, stub(name), raising=False)
    with pytest.raises(RuntimeError, match='model failure'):
        with base.install(None, tmp_path, *inputs(7711)):
            raise RuntimeError('model failure')
    assert events == [('enter', n) for n in names] + [('exit', n) for n in reversed(names)]
    record = json.loads((tmp_path/'base-n1000-optimizations.json').read_text())
    assert record['atom_count'] == 7711
    assert record['error'] == 'RuntimeError: model failure'


def test_invalid_flag_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(base.FLAG, 'on')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        with base.install(None, tmp_path, *inputs()):
            pass


@pytest.mark.parametrize('field,value', [('architecture', 'gfx1100'), ('dtype', 'bfloat16'),
    ('mode', 'training'), ('shape', (600, 4500)), ('shape', (1000, 0))])
def test_capability_rejects_unsupported_signatures(field, value):
    signature = SimpleNamespace(device_type='cuda', architecture='gfx90a', dtype='float32',
        mode='inference', layout='model_features', mask='model_defined', inplace=True,
        shape=(1000, 7711), attributes=(('cycles', '10'), ('steps', '200'), ('samples', '1')))
    assert base._supports(signature)
    setattr(signature, field, value)
    assert not base._supports(signature)


def test_real_rust_registry_resolves_and_rejects_cpu(monkeypatch):
    import sys
    from fireprot.ops.registry import registry
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(gcnArchName='gfx90a'))))
    parameter = SimpleNamespace(is_cuda=True, device=SimpleNamespace(type='cuda'), dtype='torch.float32')
    model = SimpleNamespace(parameters=lambda: iter([parameter]), training=False)
    try:
        selected = base._select(model, 7711)
        assert selected.name == 'amd-gfx90a-base-n1000-cached-v1'
        assert selected.function is base._components
        parameter.is_cuda = False
        parameter.device.type = 'cpu'
        with pytest.raises(RuntimeError, match='No qualified Base context'):
            base._select(model, 7711)
    finally:
        registry.unregister('protenix_base_n1000_context', 'amd-gfx90a-base-n1000-cached-v1')
        base._registered_registry.cache_clear()

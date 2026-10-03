"""Guard the omission of template tensors unused by Mini inference."""
from types import SimpleNamespace as NS
import pytest
from fireprot.adapters.protenix import features


def config():
    return NS(model_name='protenix_mini_default_v0.5.0',dtype='fp32',use_template=False,
              model=NS(template_embedder=NS(n_blocks=0)))


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv('FIREPROT_SKIP_UNUSED_TEMPLATES','1')
    monkeypatch.setattr(features,'_verified_consumer',lambda:True)
    monkeypatch.setattr(features,'_counts',dict(skipped_templates=0,native_templates=0))


def test_unused_templates_do_not_call_builder(enabled):
    def fail():raise AssertionError('unused template builder was called')
    assert features.build_template_features(configs=config(),use_template=False,native=fail)=={}
    assert features.manifest()['counts']['skipped_templates']==1


@pytest.mark.parametrize('reason',['disabled','model','dtype','configured_templates','actual_templates','blocks','source','incomplete'])
def test_unsupported_preserves_native_result(enabled,monkeypatch,reason):
    cfg=config();actual=False
    if reason=='disabled':monkeypatch.setenv('FIREPROT_SKIP_UNUSED_TEMPLATES','0')
    elif reason=='model':cfg.model_name='protenix_base_default_v1.0.0'
    elif reason=='dtype':cfg.dtype='bf16'
    elif reason=='configured_templates':cfg.use_template=True
    elif reason=='actual_templates':actual=True
    elif reason=='blocks':cfg.model.template_embedder.n_blocks=2
    elif reason=='source':monkeypatch.setattr(features,'_verified_consumer',lambda:False)
    elif reason=='incomplete':cfg=NS()
    sentinel=object();calls=[]
    def native():calls.append(1);return sentinel
    assert features.build_template_features(configs=cfg,use_template=actual,native=native) is sentinel
    assert calls==[1] and features.manifest()['counts']==dict(skipped_templates=0,native_templates=1)


def test_native_exception_propagates(enabled):
    def native():raise ValueError('template error')
    with pytest.raises(ValueError,match='template error'):
        features.build_template_features(configs=config(),use_template=True,native=native)

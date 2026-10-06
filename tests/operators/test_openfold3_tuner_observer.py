"""Observer selection/exception/cache fidelity using a small tuner test double."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
import hashlib

import pytest

from experiments.mfma_qk.tuner_observer import _checked_tuner, _observe


class Tuner:
    def __init__(self):
        self.cached_chunk_size = None
        self.caught = []

    def tune_chunk_size(self, representative_fn, args, max_chunk_size):
        if self.cached_chunk_size is not None:
            return self.cached_chunk_size
        for size in (max_chunk_size, max_chunk_size // 2):
            try:
                representative_fn(*args, chunk_size=size)
            except RuntimeError as error:
                self.caught.append(error)
            else:
                self.cached_chunk_size = size
                return size
        raise RuntimeError('no viable candidate')


def test_failures_selection_and_cached_calls_match_original():
    failure = RuntimeError('not necessarily an out-of-memory error')
    args = (object(),)
    calls = []
    def representative(token, *, chunk_size):
        assert token is args[0]
        calls.append(chunk_size)
        if chunk_size == 1024:
            raise failure
    baseline = Tuner()
    expected = baseline.tune_chunk_size(representative, args, 1024)
    original = Tuner.tune_chunk_size
    candidate = Tuner()
    report = {}
    with _observe(Tuner, report, {id(candidate): 'pairformer'}):
        assert candidate.tune_chunk_size(representative, args, 1024) == expected
        assert candidate.tune_chunk_size(representative, args, 1024) == expected
    assert calls == [1024, 512, 1024, 512]
    assert candidate.caught[0] is baseline.caught[0] is failure
    assert Tuner.tune_chunk_size is original and report['restored']
    first, cached = report['tuners']
    assert first['module'] == 'pairformer' and first['selected'] == 512
    assert first['attempts'][0]['error_type'] == 'builtins.RuntimeError'
    assert first['attempts'][0]['message'] == str(failure)
    assert cached['attempts'] == [] and cached['cached_before'] == 512


@pytest.mark.parametrize('error', [ValueError('model failed'), KeyboardInterrupt()])
def test_escaping_errors_keep_identity_and_restore_hook(error):
    original = Tuner.tune_chunk_size
    def fail(*args, **kwargs):
        raise error
    report = {}
    with pytest.raises(type(error)) as caught:
        with _observe(Tuner, report, {}):
            Tuner().tune_chunk_size(fail, (), 1024)
    assert caught.value is error
    assert Tuner.tune_chunk_size is original and report['restored']
    assert not report['tuners'][0]['completed']


def test_nested_and_concurrent_install_rejected_and_other_context_not_logged():
    report = {}
    with _observe(Tuner, report, {}):
        with pytest.raises(RuntimeError, match='already installed'):
            with _observe(Tuner, {}, {}):
                pass
        def independent():
            with pytest.raises(RuntimeError, match='already installed'):
                with _observe(Tuner, {}, {}):
                    pass
            return Tuner().tune_chunk_size(lambda **kw: None, (), 512)
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(independent).result() == 512
    assert report['tuners'] == [] and report['restored']


def test_foreign_replacement_is_reported_without_overwriting_it():
    original = Tuner.tune_chunk_size
    foreign = lambda *args: 123
    report = {}
    try:
        with pytest.raises(RuntimeError, match='hook changed'):
            with _observe(Tuner, report, {}):
                Tuner.tune_chunk_size = foreign
        assert Tuner.tune_chunk_size is foreign and not report['restored']
    finally:
        Tuner.tune_chunk_size = original
    with _observe(Tuner, {}, {}):
        pass


def test_source_hash_and_replaced_method_gates(tmp_path):
    source = tmp_path / 'chunk_utils.py'
    source.write_text('class ChunkSizeTuner:\n    def tune_chunk_size(self): pass\n')
    namespace = {}
    exec(compile(source.read_text(), str(source), 'exec'), namespace)
    module = SimpleNamespace(__file__=str(source), ChunkSizeTuner=namespace['ChunkSizeTuner'])
    with pytest.raises(RuntimeError, match='source hash'):
        _checked_tuner(module)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with patch('experiments.mfma_qk.tuner_observer.SOURCE_SHA256', digest):
        assert _checked_tuner(module)[0] is module.ChunkSizeTuner
        module.ChunkSizeTuner.tune_chunk_size = lambda *args: 123
        with pytest.raises(RuntimeError, match='replaced'):
            _checked_tuner(module)


def test_confidence_cap_validates_first_and_preserves_its_instance_hook():
    import torch
    from experiments.hip_softmax.confidence_chunk import confidence_chunk_cap

    original = Tuner.tune_chunk_size
    confidence = Tuner()
    confidence.cached_arg_data = None
    stack = SimpleNamespace(chunk_size_tuner=confidence, training=False)
    model = SimpleNamespace(aux_heads=SimpleNamespace(pairformer_embedding=SimpleNamespace(pairformer_stack=stack)))
    def validate(stack, tuner):
        # The real source guard rejects our observer's different code filename.
        if type(tuner).tune_chunk_size is not original:
            raise RuntimeError('source changed before confidence validation')
    cap_report, observer_report = {}, {}
    with torch.inference_mode(), patch('torch.are_deterministic_algorithms_enabled', return_value=True), patch(
            'torch.is_deterministic_algorithms_warn_only_enabled', return_value=False), patch(
            'experiments.hip_softmax.confidence_chunk.validate_sources', side_effect=validate):
        with pytest.raises(RuntimeError, match='source changed'):
            with _observe(Tuner, {}, {}), confidence_chunk_cap(model, {}):
                pass
        with confidence_chunk_cap(model, cap_report), _observe(Tuner, observer_report, {}):
            args = (torch.empty(1536, 1, device='meta'), torch.empty(1536, 1536, 1, device='meta'))
            assert confidence.tune_chunk_size(lambda *args, **kw: None, args, 1024) == 256
            assert Tuner().tune_chunk_size(lambda **kw: None, (), 512) == 512
    assert cap_report['calls'][0]['bounded_max'] == 256
    assert len(observer_report['tuners']) == 1  # Existing confidence hook keeps its original callable.
    assert Tuner.tune_chunk_size is original and 'tune_chunk_size' not in confidence.__dict__

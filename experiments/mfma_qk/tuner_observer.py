"""Observe pinned upstream chunk trials without changing selection or failures."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
from pathlib import Path
import threading

SOURCE_SHA256 = '4c9ab484854868e3c3d9a28a0d2e8fe0f9b1c75a35dca4984e0631591489fac5'
_ACTIVE = ContextVar('fireprot_chunk_diagnostics', default=None)
_LOCK = threading.RLock()
_installed = False


def _checked_tuner(module):
    source = Path(module.__file__).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Chunk tuner source hash mismatch')
    tuner_type = module.ChunkSizeTuner
    method = tuner_type.tune_chunk_size
    if (Path(method.__code__.co_filename).resolve() != source
            or method.__name__ != 'tune_chunk_size'):
        raise RuntimeError('Chunk tuner method was replaced before observation')
    return tuner_type, dict(path=str(source), sha256=SOURCE_SHA256)


@contextmanager
def _observe(tuner_type, report, labels):
    global _installed
    with _LOCK:
        if _installed:
            raise RuntimeError('Chunk diagnostics already installed')
        original = tuner_type.tune_chunk_size

        @wraps(original)
        def observed(tuner, representative_fn, args, max_chunk_size):
            if _ACTIVE.get() is not report:
                return original(tuner, representative_fn, args, max_chunk_size)
            event = dict(module=labels.get(id(tuner), 'unmapped'), max_chunk_size=max_chunk_size,
                         cached_before=tuner.cached_chunk_size, attempts=[], completed=False)

            @wraps(representative_fn)
            def trial(*trial_args, **trial_kwargs):
                attempt = dict(chunk_size=trial_kwargs.get('chunk_size'))
                try:
                    result = representative_fn(*trial_args, **trial_kwargs)
                except RuntimeError as error:
                    attempt.update(accepted=False,
                        error_type=type(error).__module__ + '.' + type(error).__qualname__,
                        message=str(error))
                    event['attempts'].append(attempt)
                    raise
                event['attempts'].append(dict(**attempt, accepted=True))
                return result

            try:
                result = original(tuner, trial, args, max_chunk_size)
                event.update(completed=True, selected=result)
                return result
            finally:
                event['cached_after'] = tuner.cached_chunk_size
                report['tuners'].append(event)

        report.update(tuners=[], restored=False)
        tuner_type.tune_chunk_size = observed
        _installed = True
        token = _ACTIVE.set(report)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        with _LOCK:
            foreign = tuner_type.tune_chunk_size is not observed
            if not foreign:
                tuner_type.tune_chunk_size = original
                report['restored'] = True
            _installed = False
        if foreign:
            raise RuntimeError('Chunk tuner hook changed during observation')


@contextmanager
def tuner_observer(model, report):
    module = importlib.import_module('openfold3.core.utils.chunk_utils')
    tuner_type, source = _checked_tuner(module)
    labels = {id(layer.chunk_size_tuner): name for name, layer in model.named_modules()
              if getattr(layer, 'chunk_size_tuner', None) is not None}
    report.update(enabled=True, source=source)
    with _observe(tuner_type, report, labels):
        yield

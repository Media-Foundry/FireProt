"""Opt-in omission of template tensors unused by the verified Mini consumer."""
from __future__ import annotations
import functools
import hashlib
import inspect
import os
import threading
from typing import Any

_lock=threading.RLock()
_counts={'skipped_templates':0,'native_templates':0}
_contract: dict[str,Any]={}
_CONSUMER_SHA256='74cf338ebc53b572dc12d6519392a09972c2a807beb1017c7fe0773536e301e5'


@functools.lru_cache(maxsize=1)
def _verified_consumer() -> bool:
    try:
        from protenix.model.modules.pairformer import TemplateEmbedder
        function=TemplateEmbedder.forward
        digest=hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        _contract.update(consumer_sha256=digest)
        return digest==_CONSUMER_SHA256 and not hasattr(function,'__wrapped__')
    except Exception as error:
        _contract.update(error=f'{type(error).__name__}: {error}')
        return False


def _supported(configs: Any, use_template: bool) -> bool:
    if os.environ.get('FIREPROT_SKIP_UNUSED_TEMPLATES')!='1' or use_template:
        return False
    try:
        return (configs.model_name=='protenix_mini_default_v0.5.0'
                and configs.dtype=='fp32' and configs.use_template is False
                and configs.model.template_embedder.n_blocks==0
                and _verified_consumer())
    except (AttributeError,KeyError,TypeError):
        return False


def build_template_features(*,configs: Any,use_template: bool,native: Any) -> Any:
    if _supported(configs,use_template):
        with _lock:
            _counts['skipped_templates']+=1
        # The native dataset supplies its dummy template fields for an empty result.
        return {}
    with _lock:
        _counts['native_templates']+=1
    return native()


def manifest() -> dict[str,Any]:
    with _lock:
        return dict(enabled=os.environ.get('FIREPROT_SKIP_UNUSED_TEMPLATES')=='1',
                    counts=dict(_counts),contract=dict(_contract))

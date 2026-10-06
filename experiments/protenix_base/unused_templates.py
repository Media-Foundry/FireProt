"""Unqualified Base extension of Mini's unused-template guard.

Install around the worker's entire predict call, before dataset construction.
Use a frozen worker and compare complete outputs with the flag off/on. This
module is not loaded by maintained profiles.
"""
from contextlib import contextmanager
import os


@contextmanager
def install():
    from fireprot.adapters.protenix import features

    previous = features._supported

    def supported(configs, use_template):
        if getattr(configs, 'model_name', None) != 'protenix_base_default_v0.5.0':
            return previous(configs, use_template)
        if os.environ.get('FIREPROT_SKIP_UNUSED_TEMPLATES') != '1' or use_template:
            return False
        try:
            return (configs.dtype == 'fp32' and configs.use_template is False
                    and configs.model.template_embedder.n_blocks == 0
                    and features._verified_consumer())
        except (AttributeError, KeyError, TypeError):
            return False

    features._supported = supported
    try:
        yield
    finally:
        features._supported = previous

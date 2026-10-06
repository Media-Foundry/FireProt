"""Scoped exact key packing around the existing attention workspace."""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

from .attention_runtime import KeyPacker
from experiments.mfma_qk import workspace_attention as native


@contextmanager
def key_scope(output, library):
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()=='47d2efa63c4402367af9282f05411b2caa123dc9f8826b5314f82afae8cc966d'
    original=native.WorkspaceFusion.__call__
    source=textwrap.dedent(inspect.getsource(original))
    old='key.transpose(-1, -2).reshape(self.batch * 4, 32, 1536)'
    assert source.count(old)==1
    pack=KeyPacker(library)
    scope=dict(vars(native),_key_packer=pack)
    exec(compile(source.replace(old,'_key_packer(key)'),'<key-copy-model>','exec'),scope)
    native.WorkspaceFusion.__call__=scope['__call__']
    try:
        yield
    finally:
        native.WorkspaceFusion.__call__=original
        Path(output,'triangle-key-copy.json').write_text(json.dumps(dict(calls=pack.calls))+'\n')

"""Final-cycle experiment: retained native BMM output and exact HIP softmax."""
import hashlib
import inspect
from pathlib import Path
import textwrap

from experiments.mfma_qk import workspace_attention as native


def make_call(mode, key_packer, softmax):
    if mode not in ('fusion', 'split'):
        raise ValueError('Expected explicit fusion or split mode')
    expected = '47d2efa63c4402367af9282f05411b2caa123dc9f8826b5314f82afae8cc966d'
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Workspace source differs from the audited lifecycle')
    source = textwrap.dedent(inspect.getsource(native.WorkspaceFusion.__call__))
    packing = 'key.transpose(-1, -2).reshape(self.batch * 4, 32, 1536)'
    assert source.count(packing) == 1
    source = source.replace(packing, '_key_packer(key)')
    if mode == 'split':
        begin = source.index('        status = self.entry(')
        end = source.index('        if status:', begin)
        source = source[:begin] + '''        torch.bmm(packed_q, packed_k,
            out=self._scores.view(self.batch * 4, 1536, 1536))
        _softmax(self._scores, (mask, bias))
        status = 0
''' + source[end:]
    scope = dict(vars(native), _key_packer=key_packer, _softmax=softmax)
    exec(compile(source, '<retained-score-' + mode + '>', 'exec'), scope)
    return scope['__call__']

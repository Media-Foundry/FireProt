# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Outpace Bio, Inc.
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0; see licenses/Apache-2.0.txt.
# FireProt modification: source-pinned runtime copy substitutions and replay.
"""Replay trained triangle multiplication with unchanged native arithmetic."""
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import textwrap
import types

import torch
import torch.utils.deterministic
from copy_runtime import CopyRuntime
from screen import bits_equal
from openfold3.core.model.layers import triangular_multiplicative_update as native

SOURCE_SHA = '612caf7abd07a86a7fdf4eda7e0a2170c04b407f9121fc286ab864a648db1ba7'


def patched(rt):
    path = Path(inspect.getsourcefile(native.TriangleMultiplicativeUpdate))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA
    source = textwrap.dedent(inspect.getsource(native.TriangleMultiplicativeUpdate._inference_forward))
    substitutions = {
        'p[..., i : i + inplace_chunk_size] = pair_chunk':
            '_copy.copy_(p[..., i : i + inplace_chunk_size], pair_chunk)',
        'p[..., i : i + inplace_chunk_size, :] = pair_chunk':
            '_copy.copy_(p[..., i : i + inplace_chunk_size, :], pair_chunk)',
        'else:\n                x_chunk = torch.einsum("...ij,...jk->...ik", a, b_chunk)':
            'else:\n                x_chunk = torch.einsum("...ij,...jk->...ik", a, _copy.contiguous(b_chunk))',
        'x_chunk = self.layer_norm_out(x_chunk)':
            'x_chunk = self.layer_norm_out(_copy.contiguous(x_chunk))',
    }
    for old, new in substitutions.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    scope = dict(vars(native), _copy=rt)
    exec(compile(source, '<source-pinned-triangle-copy-experiment>', 'exec'), scope)
    return scope['_inference_forward']


@torch.inference_mode()
def main(build, captures, output):
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    rt = CopyRuntime(json.loads(Path(build).read_text())['library'])
    candidate = patched(rt)
    results = []
    for path in sorted(Path(captures).glob('trunk_pairformer-tri_mul_*.pt')):
        snap = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        cls = getattr(native, snap['module_type'])
        module = cls(c_z=128,c_hidden=128).eval().cuda()
        module.load_state_dict(snap['state_dict'],strict=True)
        z = snap['z'].cuda(); mask = snap['mask'].cuda()
        flags = snap['flags']
        assert flags['inplace_safe'] and flags['_add_with_inplace']
        samples={'native':[],'copy':[]}
        reference=None
        for mode in ['native','copy'] + ['native','copy','copy','native']*3:
            module._inference_forward = types.MethodType(
                candidate if mode=='copy' else native.TriangleMultiplicativeUpdate._inference_forward,module)
            inp=z.clone()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); value=module(inp,mask=mask,**flags);end.record();end.synchronize()
            assert value.data_ptr()==inp.data_ptr()
            if reference is None:reference=value.clone()
            else:assert bits_equal(reference,value), (path.name,mode)
            samples[mode].append(start.elapsed_time(end))
            del inp,value
        a=statistics.median(samples['native'][1:]);b=statistics.median(samples['copy'][1:])
        results.append(dict(capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            native_ms=a,copy_ms=b,speedup=a/b,exact=True,samples_ms=samples))
        del module,z,mask,reference,snap
    assert len(results)==2
    Path(output).write_text(json.dumps(dict(correctness_passed=True,results=results,
        calls=rt.calls,fallbacks=rt.fallbacks,source_sha256=SOURCE_SHA,
        torch=str(torch.__version__),hip=torch.version.hip,seed=101,strict=True),indent=2)+'\n')


if __name__=='__main__':
    main(*sys.argv[1:])

"""Screen full-model exactness and warm latency for AtlasLM N1000 operations."""
import ctypes
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from fireprot_workers.plm import load_model
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays

r = json.loads(Path(sys.argv[1]).read_text()); r['backend'] = 'native'
output = Path(sys.argv[2]); output.mkdir(exist_ok=False)
library = ctypes.CDLL(sys.argv[3])
entry = library.fireprot_atlaslm_swiglu
entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
entry.restype = ctypes.c_int
torch.set_num_threads(8); torch.manual_seed(r['seed'])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
_, sequence = read_sequence(r['input']['path'])
model, tokens, forward = load_model(r, sequence)
assert tokens.shape == (1,1002) and bool((tokens != model.alphabet.pad_idx).all())
changes = []
verify = False
verified = 0

def restore():
    for module, native in reversed(changes): module.forward = native
    changes.clear()

def install(mode):
    restore()
    if mode == 'swiglu':
        for block in model.transformer.blocks:
            module = block.ffn[2]; native = module.forward
            def call(x, _native=native):
                global verified
                assert x.shape == (1,1002,12288) and x.dtype == torch.bfloat16 and x.is_contiguous()
                y = torch.empty((1,1002,6144), dtype=x.dtype, device=x.device)
                status = entry(x.data_ptr(), y.data_ptr(), 1002, torch.cuda.current_stream().cuda_stream)
                if status: raise RuntimeError(status)
                if verify:
                    expected = _native(x)
                    if not torch.equal(expected, y):
                        raise RuntimeError(f'SwiGLU differs: {int((expected != y).sum())} entries')
                    verified += 1
                return y
            changes.append((module, native)); module.forward = call
    elif mode in ('efficient_unmasked', 'auto_unmasked'):
        for block in model.transformer.blocks:
            module = block.attn; native = module.forward
            def call(x, seq_id=None, pos_id=None, return_attn=False, return_attn_logits=False, _native=native):
                assert not return_attn and not return_attn_logits and pos_id is None
                # Input tokens were checked once above; the native model supplies
                # the same all-valid mask to every layer for this fixed input.
                if mode == 'efficient_unmasked':
                    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                        return _native(x, None, None, False, False)
                return _native(x, None, None, False, False)
            changes.append((module, native)); module.forward = call

report = dict(length=1000, seed=r['seed'], trials={})
with torch.inference_mode():
    expected = {k:v.clone() for k,v in forward().items()}
    save_arrays(output/'native.npz', expected)
    for mode in ['swiglu', 'efficient_unmasked', 'auto_unmasked']:
        trial = {}; report['trials'][mode] = trial
        try:
            verify = True; verified = 0; install(mode)
            result = forward(); torch.cuda.synchronize()
            verify = False
            trial['exact'] = {k:torch.equal(v,result[k]) for k,v in expected.items()}
            trial['verified_swiglu_calls'] = verified
            if not all(trial['exact'].values()):
                trial['decision'] = 'rejected: full output mismatch'; continue
            save_arrays(output/(mode+'.npz'), result)
            trial['pairs'] = []
            for order in [('native',mode),(mode,'native'),('native',mode)]:
                pair = {}
                for arm in order:
                    install(arm)
                    for _ in range(2): forward()
                    values = []
                    for _ in range(5):
                        torch.cuda.synchronize(); started = time.perf_counter()
                        actual = forward(); torch.cuda.synchronize()
                        values.append(time.perf_counter()-started)
                        if not all(torch.equal(v,actual[k]) for k,v in expected.items()):
                            raise RuntimeError('Repeated output drift')
                    pair[arm] = values
                trial['pairs'].append(pair)
            trial['medians'] = {arm:statistics.median(sum([p[arm] for p in trial['pairs']],[])) for arm in ['native',mode]}
        except Exception as error:
            trial['error'] = str(error)
        finally:
            verify = False; restore()
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)

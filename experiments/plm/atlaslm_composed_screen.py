"""Verify BF16 SwiGLU finite domain and time the packaged AtlasLM composition."""
import ctypes
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from fireprot_workers.plm import load_model
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays
from fireprot.adapters.plm.runtime import optimized_scope

r = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2]); output.mkdir(exist_ok=False)
torch.set_num_threads(8); torch.manual_seed(r['seed'])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
library = ctypes.CDLL(r['resources']['hip_swiglu']['path'])
entry = library.fireprot_atlaslm_swiglu
entry.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
entry.restype = ctypes.c_int
report = dict(length=1000, seed=r['seed'], pairs=[])
with torch.inference_mode():
    # All finite BF16 bit patterns, repeated with varying exactly representable
    # gates. Compare raw result bits, including signed zero and overflow results.
    bits = torch.arange(65536, dtype=torch.int32, device='cuda')
    bits = bits[(bits & 0x7f80) != 0x7f80].to(torch.uint16)
    values = bits.view(torch.bfloat16)
    x = torch.empty((1,1002,12288),dtype=torch.bfloat16,device='cuda')
    left,right = x.chunk(2,dim=-1)
    indices = torch.arange(left.numel(),device='cuda')
    left.copy_(values[indices % values.numel()].view_as(left))
    right.copy_((((indices // values.numel()) % 7 - 3) / 2).to(torch.bfloat16).view_as(right))
    expected = torch.nn.functional.silu(left)*right
    actual = torch.empty_like(expected)
    code = entry(x.data_ptr(),actual.data_ptr(),1002,torch.cuda.current_stream().cuda_stream)
    assert code == 0
    report['finite_domain_patterns'] = values.numel()
    report['finite_domain_mismatches'] = int((expected.view(torch.uint16) != actual.view(torch.uint16)).sum())
    del x,left,right,values,bits,indices,expected,actual
    torch.cuda.empty_cache()
    if report['finite_domain_mismatches']:
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        raise RuntimeError('BF16 SwiGLU finite-domain bit mismatch')
    _,sequence = read_sequence(r['input']['path'])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model,tokens,forward = load_model(r,sequence)
    torch.cuda.synchronize()
    report['loading_seconds_with_initialized_device'] = time.perf_counter()-started
    report['loading_peak_bytes'] = torch.cuda.max_memory_allocated()
    report['loading_reserved_bytes'] = torch.cuda.max_memory_reserved()
    expected = {k:v.clone() for k,v in forward().items()}
    originals = [(m,m.forward) for m in model.modules()]
    def run(candidate):
        if not candidate: return forward()
        with optimized_scope(model,r,len(sequence)) as state:
            result = forward()
        assert state['calls'] == state['swiglu_calls'] == 48 and state['rotary_modules'] == 1
        return result
    for order in [(False,True),(True,False),(False,True)]:
        pair = {}
        for candidate in order:
            for _ in range(2): run(candidate)
            values = []
            for _ in range(5):
                torch.cuda.synchronize(); started = time.perf_counter()
                actual = run(candidate); torch.cuda.synchronize()
                values.append(time.perf_counter()-started)
                assert all(torch.equal(v,actual[k]) for k,v in expected.items())
                assert all(m.forward == original for m,original in originals)
            pair['candidate' if candidate else 'native'] = values
        report['pairs'].append(pair)
    report['medians'] = {arm:statistics.median(sum([p[arm] for p in report['pairs']],[])) for arm in ['native','candidate']}
    report['speedup'] = report['medians']['native']/report['medians']['candidate']
    report['all_outputs_exact'] = True
    save_arrays(output/'native.npz',expected);save_arrays(output/'candidate.npz',actual)
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)

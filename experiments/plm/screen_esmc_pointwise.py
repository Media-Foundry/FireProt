"""Screen exact HIP residual fusion and vector SwiGLU using real ESMC activations."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from fireprot.adapters.plm.runtime import optimized_scope
from fireprot.ops._bridge import bridge
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from esmc_pointwise_scope import Pointwise


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ['HIP_VISIBLE_DEVICES'] == '6'
    request_path, library, output = map(Path, sys.argv[1:4])
    output.mkdir(exist_ok=False)
    request = json.loads(request_path.read_text())
    torch.set_num_threads(8); torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert not torch.cuda.tunable.is_enabled()
    source = Path(request['input']['path'])
    assert sha(source) == request['input']['sha256']
    plan = json.loads(bridge.plan_plm_batch(source.read_text(), json.dumps(request['workload']['batch'])))
    sequences = [x['sequence'] for x in plan['sequences']]
    report = dict(model=request['workload']['model_name'], seed=request['seed'], hip_device=6,
                  torch=torch.__version__, hip=torch.version.hip, dtype='native BF16', length=1000,
                  request_sha256=sha(request_path), library_sha256=sha(library), script_sha256=sha(__file__),
                  scope_sha256=sha(Path(__file__).with_name('esmc_pointwise_scope.py')),
                  boundary='warm operators and forward; excludes loading and export; not E2E',
                  cache_state='resident model; warmed operators; caches not flushed', cases=[])
    def save(): (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    def exact(a, b): return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    with torch.inference_mode():
        model, _, _ = load_model(request, sequences[0])
        for size in (1, 4):
            forward = prepare_forward(model, request['workload']['model_name'], sequences[:size])
            captured = {}
            block = model.transformer.blocks[0]
            native_attn, native_linear = block.attn.forward, F.linear
            def attn(x, *args, **kwargs):
                out = native_attn(x, *args, **kwargs)
                captured['x'], captured['r'] = x.clone(), out[0].clone()
                return out
            def linear(x, w, b=None):
                out = native_linear(x, w, b)
                if w is block.ffn.fc1_weight: captured['up'] = out.clone()
                return out
            block.attn.forward = attn; F.linear = linear
            try: expected = {k:v.clone() for k,v in forward().items()}
            finally: block.attn.forward = native_attn; F.linear = native_linear
            case = dict(batch=size, scaling_factor=block.scaling_factor, operators=[], forwards=[])
            report['cases'].append(case); save()
            old = ctypes_swiglu(request, size)
            for kind, name in ((0, 'residual'), (1, 'swiglu')):
                x = captured['x' if kind == 0 else 'up']
                r = captured['r'] if kind == 0 else None
                native = (lambda: x+r/block.scaling_factor) if kind == 0 else (lambda: old(x))
                ref = native()
                for vector in (1, 2, 4, 8):
                    op = Pointwise(library, vector)
                    candidate = lambda: op.run(kind, x, r, block.scaling_factor)
                    actual = candidate()
                    row = dict(kind=name, vector=vector, bitwise_equal=exact(ref, actual), pairs=[])
                    case['operators'].append(row); save()
                    if not row['bitwise_equal']: continue
                    for order in ((False, True), (True, False), (False, True)):
                        pair = {}
                        for enabled in order:
                            fn = candidate if enabled else native
                            for _ in range(4): fn()
                            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(30): actual = fn()
                            end.record(); end.synchronize()
                            assert exact(ref, actual)
                            pair['candidate' if enabled else 'baseline'] = start.elapsed_time(end)/30
                        row['pairs'].append(pair)
                    row['median_ms'] = {a:statistics.median(p[a] for p in row['pairs']) for a in ('baseline','candidate')}
                    row['speedup'] = row['median_ms']['baseline']/row['median_ms']['candidate']; save()
                valid = [r for r in case['operators'] if r['kind']==name and r['bitwise_equal']]
                if not valid: continue
                best = (next(r for r in valid if r['vector']==8) if name == 'swiglu'
                        else min(valid, key=lambda r:r['median_ms']['candidate']))
                op = Pointwise(library, best['vector'])
                candidate_request = dict(request, resources=dict(request['resources']))
                candidate_request['resources']['hip_esm'] = dict(path=str(library),sha256=sha(library))
                def run(enabled, audit=False):
                    if enabled and audit and name == 'swiglu':
                        # Audit all intermediate activations without replacing the
                        # qualified scope's accounting wrapper. Timing uses its ABI.
                        with op.scope(model, name, audit=True) as state: out = forward()
                        return out, state
                    selected = candidate_request if enabled and name == 'swiglu' else request
                    with optimized_scope(model, selected, 1000, batch_size=size) as hip:
                        if enabled and name == 'residual':
                            with op.scope(model, name, audit=audit) as state:
                                out = forward()
                            return out, state
                        return forward(), hip
                row = dict(mode=name, vector=best['vector'], pairs=[])
                case['forwards'].append(row)
                try:
                    actual, row['audit'] = run(True, audit=True)
                    row['bitwise_equal'] = all(exact(v, actual[k]) for k,v in expected.items())
                    assert row['bitwise_equal']
                except AssertionError as error:
                    row['rejected'] = str(error); save(); continue
                for order in ((False, True), (True, False), (False, True)):
                    pair = {}
                    for enabled in order:
                        for _ in range(2): run(enabled)
                        times = []
                        for _ in range(3):
                            torch.cuda.synchronize(); start = time.perf_counter()
                            actual, _ = run(enabled); torch.cuda.synchronize()
                            times.append(time.perf_counter()-start)
                            assert all(exact(v, actual[k]) for k,v in expected.items())
                        pair['candidate' if enabled else 'baseline'] = times
                    row['pairs'].append(pair)
                row['median_seconds'] = {a:statistics.median(t for p in row['pairs'] for t in p[a]) for a in ('baseline','candidate')}
                row['speedup'] = row['median_seconds']['baseline']/row['median_seconds']['candidate']; save()
                print(json.dumps(dict(batch=size, result=row)), flush=True)
            del captured, expected


def ctypes_swiglu(request, size):
    import ctypes
    lib = ctypes.CDLL(request['resources']['hip_esm']['path'])
    fn = lib.fireprot_esm_swiglu
    fn.argtypes = [ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]; fn.restype = ctypes.c_int
    def run(x):
        out = torch.empty((*x.shape[:-1],6912), device=x.device, dtype=x.dtype)
        rc = fn(x.data_ptr(),out.data_ptr(),size*1002,6912,torch.cuda.current_stream().cuda_stream)
        if rc: raise RuntimeError(rc)
        return out
    return run


if __name__ == '__main__': main()

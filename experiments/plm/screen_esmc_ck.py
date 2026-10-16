"""Diagnose installed CK GEMM on real ESMC activations; no production dispatch changes."""
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


def difference(expected, actual):
    delta = (expected.float() - actual.float()).abs()
    return dict(bitwise_equal=torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)),
                unequal_elements=int((expected != actual).sum()), max_abs=float(delta.max()),
                mean_abs=float(delta.mean()), finite=bool(torch.isfinite(actual).all()))


def main():
    assert os.environ['HIP_VISIBLE_DEVICES'] == '6'
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    output = Path(sys.argv[2]); output.mkdir(exist_ok=False)
    torch.set_num_threads(8); torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    preferred = torch.backends.cuda.preferred_blas_library
    original = preferred()
    assert str(original).endswith('Cublaslt'), str(original)
    report = dict(model=request['workload']['model_name'], hip_device=6, seed=request['seed'],
                  torch=torch.__version__, hip=torch.version.hip, device=str(torch.cuda.get_device_properties(0)),
                  precision='native BF16', length=1000, cache_state='resident weights; warmed operators; caches not flushed',
                  boundary='diagnostic operator and warm forward; not complete invocation',
                  request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_blas=str(original), ck_sdpa_available=torch._C._is_ck_sdpa_available(), cases=[])

    def save():
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    input_path = Path(request['input']['path'])
    plan = json.loads(bridge.plan_plm_batch(input_path.read_text(), json.dumps(request['workload']['batch'])))
    sequences = [item['sequence'] for item in plan['sequences']]
    sequence = sequences[0]
    assert len(sequences) == 4 and all(len(s) == 1000 for s in sequences)
    report['input_sha256'] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert report['input_sha256'] == request['input']['sha256']
    linear = F.linear
    try:
        with torch.inference_mode():
            model, _, _ = load_model(request, sequence)
            for size in (1, 4):
                forward = prepare_forward(model, request['workload']['model_name'],
                                          sequences[:size])
                captured = []

                def capture(x, weight, bias=None):
                    # First block: QKV, attention output, FFN up/gate, FFN down.
                    if len(captured) < 4:
                        captured.append((x.clone(), weight, bias))
                    return linear(x, weight, bias)

                F.linear = capture
                try:
                    expected = {k:v.clone() for k,v in forward().items()}
                finally:
                    F.linear = linear
                case = dict(batch=size, operators=[])
                report['cases'].append(case); save()
                for index, (x, weight, bias) in enumerate(captured):
                    preferred(original); reference = linear(x, weight, bias)
                    preferred('ck'); actual = linear(x, weight, bias)
                    row = dict(index=index, input_shape=list(x.shape), weight_shape=list(weight.shape),
                               difference=difference(reference, actual), pairs=[])
                    for order in ((False, True), (True, False), (False, True)):
                        pair = {}
                        for candidate in order:
                            preferred('ck' if candidate else original)
                            for _ in range(4): linear(x, weight, bias)
                            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(20): actual = linear(x, weight, bias)
                            end.record(); end.synchronize()
                            pair['ck' if candidate else 'hipblaslt'] = start.elapsed_time(end)/20
                            # Every timing arm must reproduce its own initial output exactly.
                            check = linear(x, weight, bias)
                            assert torch.equal(actual.view(torch.uint8), check.view(torch.uint8))
                        row['pairs'].append(pair)
                    row['median_ms'] = {a:statistics.median(p[a] for p in row['pairs']) for a in ('ck','hipblaslt')}
                    row['speedup'] = row['median_ms']['hipblaslt']/row['median_ms']['ck']
                    row['kernels'] = {}
                    for backend in ('cublaslt', 'ck'):
                        preferred(backend)
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
                            linear(x, weight, bias); torch.cuda.synchronize()
                        prof.export_chrome_trace(str(output/f'b{size}-op{index}-{backend}.json'))
                        row['kernels'][backend] = sorted({e.name for e in prof.events()
                                                         if e.device_type == torch.autograd.DeviceType.CUDA})
                    case['operators'].append(row); save()
                    print(json.dumps(dict(batch=size, operator=index, speedup=row['speedup'], difference=row['difference'])), flush=True)
                del captured
                # Isolate CK's incremental effect with the already qualified HIP scope on both arms.
                case['forward_pairs'] = []
                for order in ((False, True), (True, False), (False, True)):
                    pair = {}
                    for candidate in order:
                        preferred('ck' if candidate else original)
                        def run():
                            with optimized_scope(model, request, 1000, batch_size=size):
                                return forward()
                        for _ in range(2): run()
                        times = []; differences = []
                        for _ in range(3):
                            torch.cuda.synchronize(); start = time.perf_counter()
                            actual = run(); torch.cuda.synchronize()
                            times.append(time.perf_counter()-start)
                            diff = {k:difference(v, actual[k]) for k,v in expected.items()}
                            differences.append(diff)
                            if not candidate:
                                assert all(d['bitwise_equal'] for d in diff.values()), diff
                        pair['ck' if candidate else 'current_fireprot'] = dict(seconds=times, differences=differences)
                    case['forward_pairs'].append(pair); save()
                case['forward_median_seconds'] = {a:statistics.median(t for p in case['forward_pairs'] for t in p[a]['seconds'])
                                                  for a in ('ck','current_fireprot')}
                case['forward_speedup'] = case['forward_median_seconds']['current_fireprot']/case['forward_median_seconds']['ck']
                save(); print(json.dumps(dict(batch=size, forward_speedup=case['forward_speedup'], difference=diff)), flush=True)
                preferred(original)
    finally:
        F.linear = linear
        preferred(original)


if __name__ == '__main__':
    main()

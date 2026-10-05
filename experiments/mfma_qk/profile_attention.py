"""Correlated kernel-duration diagnostic for accepted complete-attention arms."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.utils.deterministic

from experiments.mfma_qk.benchmark_attention import PackedFusion, pin
from experiments.mfma_qk.probe_fusion import bitwise_mismatch_count
from experiments.hip_softmax.owned_scores import OwnedScoreSoftmax
from experiments.attention_trace.probe import trace_operations


def kernels_in_region(path):
    events = json.loads(path.read_text())['traceEvents']
    regions = [e for e in events if e.get('cat') == 'user_annotation'
               and e.get('name') == 'fireprot.attention.profile' and e.get('ph') == 'X']
    if len(regions) != 1:
        raise RuntimeError('Expected one complete synchronized attention region')
    region = regions[0]
    result = []
    for kernel in events:
        if kernel.get('cat') != 'kernel':
            continue
        args = kernel.get('args', {})
        correlation = args.get('correlation')
        launches = [e for e in events if e.get('cat') == 'cuda_runtime'
                    and e.get('args', {}).get('correlation') == correlation
                    and e.get('args', {}).get('External id') == args.get('External id')]
        if correlation is None or len(launches) != 1:
            raise RuntimeError('Ambiguous kernel/runtime correlation')
        launch = launches[0]
        if (launch.get('pid'), launch.get('tid')) != (region['pid'], region['tid']):
            continue
        if not region['ts'] <= launch['ts'] <= region['ts'] + region['dur']:
            continue
        result.append(dict(name=kernel['name'], duration_us=kernel['dur'],
            correlation=correlation, launch=launch['name'],
            device=args.get('device'), stream=args.get('stream')))
    if not result:
        raise RuntimeError('No correlated GPU kernels in attention region')
    return sorted(result, key=lambda r: r['duration_us'], reverse=True)


def main(controls_path, destination, batch):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, batch=batch, n=1536, heads=4, k=32, seed=101,
        controls=pin(controls_path), script=pin(__file__), observations=[],
        scope='One warmed profiled call per arm/layout; kernel-duration diagnosis, not latency or E2E comparison')
    try:
        controls = json.loads(Path(controls_path).read_text())
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE'] = controls['split_hip']['bundle']
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE_SHA256'] = controls['split_hip']['manifest_sha256']
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        torch.manual_seed(101)
        with torch.inference_mode():
            split = OwnedScoreSoftmax()
            candidates = {c['label']: PackedFusion(c, batch) for c in controls['fusion']}
            query, key, value = [(torch.randn(batch, 1536, 4, 32, device='cuda', dtype=torch.float32) * 0.125).transpose(1, 2) for _ in range(3)]
            mask = torch.zeros(batch, 1, 1, 1536, device='cuda', dtype=torch.float32)
            for outer in range(batch):
                mask[outer, ..., 1024 + outer % 512:] = -float('inf')
            base_bias = torch.randn(1, 4, 1536, 1536, device='cuda', dtype=torch.float32) * 0.125
            for layout in ('contiguous', 'transposed'):
                bias = base_bias if layout == 'contiguous' else base_bias.transpose(-1, -2)
                scores = torch.einsum('...qc, ...kc->...qk', query, key)
                scores.add_(mask).add_(bias)
                probability = torch.softmax(scores, -1)
                expected = torch.einsum('...qk, ...kc->...qc', probability, value)
                del scores, probability

                def split_call():
                    scores = torch.einsum('...qc, ...kc->...qk', query, key)
                    return torch.einsum('...qk, ...kc->...qc', split(scores, [mask, bias]), value)

                arms = dict(split_hip=split_call)
                arms.update({name: (lambda kernel=kernel: kernel(query, key, value, mask, bias))
                             for name, kernel in candidates.items()})
                for name, function in arms.items():
                    for _ in range(2):
                        output = function()
                        torch.cuda.synchronize()
                        assert bitwise_mismatch_count(expected, output) == 0
                        del output
                    trace = destination / f'{layout}-{name}.json'
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                        torch.cuda.synchronize()
                        live_bytes = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        with torch.profiler.record_function('fireprot.attention.profile'):
                            output = function()
                            torch.cuda.synchronize()
                        peak_bytes = torch.cuda.max_memory_allocated()
                    profiler.export_chrome_trace(str(trace))
                    mismatches = bitwise_mismatch_count(expected, output)
                    del output
                    if mismatches:
                        raise RuntimeError(f'Profiled attention mismatch: {name}/{layout}')
                    operations = dict(einsums=[dict(label='profile')])
                    trace_operations(operations, trace)
                    record = dict(arm=name, layout=layout, trace=pin(trace), bitwise_mismatches=0,
                        pytorch_live_bytes_before=live_bytes, pytorch_peak_bytes=peak_bytes,
                        pytorch_peak_increment_bytes=peak_bytes-live_bytes,
                        kernels=kernels_in_region(trace), operations=operations)
                    report['observations'].append(record)
                    print(name, layout, json.dumps(record['kernels'][:3]), flush=True)
                del expected
        report['completed'] = True
    finally:
        (destination / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controls')
    parser.add_argument('destination')
    parser.add_argument('--batch', type=int, choices=(32, 256), default=256)
    args = parser.parse_args()
    main(args.controls, args.destination, args.batch)

"""Workspace ownership/correctness gates and complete-attention timing."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

import torch
import torch.utils.deterministic

from experiments.mfma_qk import workspace_attention
from experiments.mfma_qk.benchmark_attention import PackedFusion, balanced_orders, pin
from experiments.mfma_qk.probe_fusion import bitwise_mismatch_count
from experiments.hip_softmax.owned_scores import OwnedScoreSoftmax
from fireprot.backends.amd import softmax_fp32


def exact(expected, actual):
    count = bitwise_mismatch_count(expected, actual)
    if count:
        raise RuntimeError(f'{count} bitwise mismatches')


def operands(batch, seed, layout):
    torch.manual_seed(seed)
    q, k, v = [(torch.randn(batch, 1536, 4, 32, device='cuda') * 0.125).transpose(1, 2)
               for _ in range(3)]
    mask = torch.zeros(batch, 1, 1, 1536, device='cuda')
    for outer in range(batch):
        mask[outer, ..., 1024 + (outer + seed) % 512:] = -float('inf')
    if seed == 103:
        mask[0].fill_(-float('inf'))
    bias = torch.randn(1, 4, 1536, 1536, device='cuda') * 0.125
    if layout == 'transposed':
        bias = bias.transpose(-1, -2)
    return q, k, v, mask, bias


def reference(args):
    q, k, v, mask, bias = args
    scores = torch.einsum('...qc, ...kc->...qk', q, k)
    scores.add_(mask).add_(bias)
    probability = torch.softmax(scores, -1)
    del scores
    return probability, torch.einsum('...qk, ...kc->...qc', probability, v)


def native(args):
    return reference(args)[1]


def main(controls_path, destination, batch):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, batch=batch, n=1536, heads=4, k=32,
        controls=pin(controls_path), source_pins=[pin(__file__), pin(workspace_attention.__file__),
            *[pin(sys.modules[obj.__module__].__file__) for obj in
              (PackedFusion, bitwise_mismatch_count, OwnedScoreSoftmax)], pin(softmax_fp32.__file__)],
        qualification=[], guards=[], records=[], cold_workspace_calls=[],
        scope='Experimental retained workspace; complete attention only, no model/E2E acceptance',
        cache_state='Published kernels and warm allocator. Two private score workspaces persist across all arms. No cache flush; exact checks outside timing.',
        retained_workspace_bytes_per_arm=batch * 4 * 1536 * 1536 * 4,
        visible_device=os.environ.get('HIP_VISIBLE_DEVICES'))
    pools = {}
    try:
        if str(torch.__version__) != softmax_fp32.TORCH_VERSION or torch.version.hip != softmax_fp32.HIP_VERSION:
            raise RuntimeError('Use the pinned Torch/HIP environment')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        report['settings'] = dict(torch=str(torch.__version__), hip=str(torch.version.hip),
            strict_deterministic=True, fill_uninitialized_memory=True, allow_tf32=False,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()))
        control = json.loads(Path(controls_path).read_text())
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE'] = control['split_hip']['bundle']
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE_SHA256'] = control['split_hip']['manifest_sha256']
        kernel = control['fusion']
        with torch.inference_mode():
            split = OwnedScoreSoftmax()
            fresh = PackedFusion(kernel, batch)
            pool_specs = [('workspace', kernel, False), ('workspace_contiguous_bias', kernel, True)]
            if 'workspace_reference' in control:
                pool_specs[0] = ('workspace_reference', control['workspace_reference'], True)
            pools = {name: workspace_attention.WorkspaceFusion(selected, batch,
                max_bytes=9 * 2**30, contiguous_bias=normalize)
                for name, selected, normalize in pool_specs}
            report['workspace_receipts'] = {name: dict(kernel=p.receipt,
                contiguous_bias=p.contiguous_bias) for name, p in pools.items()}
            report['kernel_receipt'] = fresh.receipt
            # Different seeds/layouts and poisoned previous scores exercise full overwrite.
            previous = {}
            for seed in (101, 102, 103):
                for layout in ('contiguous', 'transposed'):
                    args = operands(batch, seed, layout)
                    probability, expected = reference(args)
                    for name, pool in pools.items():
                        first = pool.allocations == 0
                        if not first:
                            pool._scores.fill_(float('nan') if seed % 2 else -12345.)
                        torch.cuda.synchronize()
                        live = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        output = pool(*args)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        peak = torch.cuda.max_memory_allocated()
                        exact(probability, pool._scores)
                        exact(expected, output)
                        prior_checked = name in previous
                        if prior_checked:
                            exact(*previous[name])
                        previous[name] = (output.clone(), output)
                        record = dict(arm=name, seed=seed, layout=layout, score_bits_exact=True,
                            output_bits_exact=True, prior_result_unchanged=True if prior_checked else None,
                            allocations=pool.allocations, retained_bytes=pool.retained_bytes)
                        report['qualification'].append(record)
                        if first:
                            report['cold_workspace_calls'].append(dict(arm=name, seconds=elapsed,
                                pytorch_live_bytes_before=live, pytorch_peak_bytes=peak,
                                note='First workspace allocation/fill included; artifacts and other Torch operations already warm.'))
                    del probability, expected, args
            previous.clear()
            del output
            args = operands(batch, 101, 'contiguous')
            for name, pool in pools.items():
                before = (pool.calls, pool.allocations, pool._scores.data_ptr())
                errors = []
                def other_thread():
                    try:
                        pool(*args)
                    except RuntimeError as error:
                        errors.append(str(error))
                thread = threading.Thread(target=other_thread)
                thread.start()
                thread.join()
                if len(errors) != 1 or 'different thread' not in errors[0]:
                    raise RuntimeError('Cross-thread rejection failed')
                with torch.cuda.stream(torch.cuda.Stream()):
                    try:
                        pool(*args)
                    except RuntimeError as error:
                        if 'one device and stream' not in str(error):
                            raise
                    else:
                        raise RuntimeError('Cross-stream rejection failed')
                if before != (pool.calls, pool.allocations, pool._scores.data_ptr()):
                    raise RuntimeError('Rejected access mutated the workspace')
                report['guards'].append(dict(arm=name, thread_rejected=True, stream_rejected=True,
                    no_submission_or_reallocation=True))
            del args
            for layout in ('contiguous', 'transposed'):
                args = operands(batch, 101, layout)
                expected = native(args)
                def split_call():
                    q, k, v, mask, bias = args
                    scores = torch.einsum('...qc, ...kc->...qk', q, k)
                    return torch.einsum('...qk, ...kc->...qc', split(scores, [mask, bias]), v)
                arms = dict(native=lambda: native(args), split_hip=split_call,
                    fresh_fusion=lambda: fresh(*args))
                arms.update({name: (lambda pool=pool: pool(*args)) for name, pool in pools.items()})
                for function in arms.values():
                    for _ in range(2):
                        output = function()
                        torch.cuda.synchronize()
                        exact(expected, output)
                        del output
                for round_index, order in enumerate(balanced_orders(list(arms))):
                    for position, name in enumerate(order):
                        torch.cuda.synchronize()
                        live = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        output = arms[name]()
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - start
                        peak = torch.cuda.max_memory_allocated()
                        exact(expected, output)
                        record = dict(arm=name, layout=layout, round=round_index, position=position,
                            seconds=seconds, bitwise_mismatches=0, pytorch_live_bytes_before=live,
                            pytorch_peak_bytes=peak, pytorch_peak_increment_bytes=peak-live)
                        report['records'].append(record)
                        print(json.dumps(record), flush=True)
                        del output
                del expected, args
            report['workspaces'] = {name: dict(allocations=p.allocations, calls=p.calls,
                retained_bytes=p.retained_bytes) for name, p in pools.items()}
            if any(p.allocations != 1 or p.calls != 30 for p in pools.values()):
                raise RuntimeError('Unexpected workspace allocation/submission count')
            if split.storage_fallbacks or split.launches_submitted != 24:
                raise RuntimeError('Unexpected split baseline execution')
            report['split_runtime'] = split.experiment_manifest()
            for name, pool in pools.items():
                before = torch.cuda.memory_allocated()
                pool.close()
                released = before - torch.cuda.memory_allocated()
                if released != pool.retained_bytes:
                    raise RuntimeError('Workspace release did not free its retained tensor')
                report['workspaces'][name]['released_bytes'] = released
            report['medians_seconds'] = {layout: {name: statistics.median(
                r['seconds'] for r in report['records'] if r['layout'] == layout and r['arm'] == name)
                for name in arms} for layout in ('contiguous', 'transposed')}
        report['completed'] = True
    finally:
        for pool in pools.values():
            pool.close()
        (destination / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controls')
    parser.add_argument('destination')
    parser.add_argument('--batch', type=int, choices=(32, 256), default=32)
    args = parser.parse_args()
    main(args.controls, args.destination, args.batch)

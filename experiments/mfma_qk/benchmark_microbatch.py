"""Full-batch reference versus bounded B256 workspace microbatching."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

from experiments.mfma_qk import microbatch_attention, workspace_attention
from experiments.mfma_qk.benchmark_attention import pin
from experiments.mfma_qk.benchmark_workspace import operands, exact
from experiments.hip_softmax.owned_scores import OwnedScoreSoftmax
from experiments.attention_trace.probe import trace_operations
from fireprot.backends.amd import softmax_fp32


def reference(args, split):
    q, k, v, mask, bias = args
    scores = torch.einsum('...qc, ...kc->...qk', q, k)
    probability = split(scores, [mask, bias])
    return probability, torch.einsum('...qk, ...kc->...qc', probability, v)


def main(controls_path, destination, batch, bias_layouts='baseline'):
    layouts = ('head_last', 'head_last_transposed') if bias_layouts == 'head-last' else ('contiguous', 'transposed')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, batch=batch, microbatch=256, n=1536, heads=4, k=32,
        controls=pin(controls_path), source_pins=[pin(__file__), pin(microbatch_attention.__file__),
            pin(workspace_attention.__file__), *[pin(sys.modules[x.__module__].__file__)
                for x in (operands, exact, OwnedScoreSoftmax, trace_operations)], pin(softmax_fp32.__file__)],
        qualification=[], records=[], seed=101, bias_layouts=list(layouts), visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),
        scope='Full-batch native QK + accepted in-place HIP softmax + native PV reference; not an out-of-place Torch-softmax or model comparison',
        cache_state='One initialized B256 score workspace retained throughout warm timing in both arms; no cache flush; exact checks outside timing.')
    candidate = None
    try:
        if str(torch.__version__) != softmax_fp32.TORCH_VERSION or torch.version.hip != softmax_fp32.HIP_VERSION:
            raise RuntimeError('Use pinned Torch/HIP')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        report['settings'] = dict(torch=str(torch.__version__), hip=str(torch.version.hip),
            strict_deterministic=True, fill_uninitialized_memory=True, allow_tf32=False,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()))
        free, total = torch.cuda.mem_get_info()
        estimated = batch * 4 * 1536 * 1536 * 4 + 8 * batch * 4 * 1536 * 32 * 4 + 11 * 2**30
        report['preflight_memory'] = dict(free_bytes=free, total_bytes=total,
            estimated_required_bytes=estimated, note='Full score + eight QKV/output-sized buffers +9GiB workspace +2GiB margin')
        if free < estimated:
            raise RuntimeError('Insufficient free device memory for full-reference validation')
        control = json.loads(Path(controls_path).read_text())
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE'] = control['split_hip']['bundle']
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE_SHA256'] = control['split_hip']['manifest_sha256']
        with torch.inference_mode():
            split = OwnedScoreSoftmax()
            candidate = microbatch_attention.MicrobatchFusion(control['fusion'], batch)
            report['kernel_receipt'] = candidate._pool.receipt
            previous = None
            for seed in (101, 102, 103):
                for layout in layouts:
                    args = operands(batch, seed, layout)
                    if seed == 102:
                        args = (*args[:3], args[3][:1], args[4])
                    if seed == 101 and layout == layouts[0]:
                        trace = destination / 'full-reference-trace.json'
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                            with torch.profiler.record_function('fireprot.attention.full_reference'):
                                probability, expected = reference(args, split)
                                torch.cuda.synchronize()
                        profiler.export_chrome_trace(str(trace))
                        observation = dict(einsums=[dict(label='full_reference')])
                        trace_operations(observation, trace)
                        bmms = [x for x in observation['einsums'][0]['operations'] if x['operation'] == 'aten::bmm']
                        wanted = [([batch*4, 1536, 32], [batch*4, 32, 1536]),
                                  ([batch*4, 1536, 1536], [batch*4, 1536, 32])]
                        if len(bmms) != 2 or any([x['inputs'][0]['shape'], x['inputs'][1]['shape']] != list(shapes)
                                for x, shapes in zip(bmms, wanted)):
                            raise RuntimeError('Full-shape QK/PV reference trace differs from contract')
                        report['reference_trace'] = dict(pin=pin(trace), observation=observation)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        first = candidate(*args)
                        torch.cuda.synchronize()
                        report['first_workspace_call_seconds'] = time.perf_counter() - start
                        exact(expected, first)
                        previous = (first.clone(), first)
                    else:
                        probability, expected = reference(args, split)
                    actual = candidate(*args, reference_probabilities=probability)
                    exact(expected, actual)
                    if actual.stride() != expected.stride():
                        raise RuntimeError('Microbatch output layout differs from native PV')
                    exact(*previous)
                    previous = (actual.clone(), actual)
                    report['qualification'].append(dict(seed=seed, layout=layout, bias_strides=list(args[-1].stride()),
                        mask_batch=args[3].shape[0], all_probability_bits_exact=True,
                        all_output_bits_exact=True, prior_result_unchanged=True,
                        output_strides=list(actual.stride())))
                    del probability, expected, args, actual
            del previous, first
            for layout in layouts:
                args = operands(batch, 101, layout)
                probability, expected = reference(args, split)
                del probability
                arms = dict(split_hip=lambda: reference(args, split)[1], microbatch=lambda: candidate(*args))
                for fn in arms.values():
                    for _ in range(2):
                        output = fn()
                        torch.cuda.synchronize()
                        exact(expected, output)
                        del output
                for round_index in range(10):
                    order = list(arms) if round_index % 2 == 0 else list(reversed(arms))
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
            if candidate.calls != 31 or candidate._pool.calls != 31 * (batch // 256) or candidate._pool.allocations != 1:
                raise RuntimeError('Unexpected microbatch execution accounting')
            if split.storage_fallbacks or split.launches_submitted != 32:
                raise RuntimeError('Unexpected full-reference execution accounting')
            report['workspace'] = dict(full_calls=candidate.calls, kernel_calls=candidate._pool.calls,
                allocations=candidate._pool.allocations, retained_bytes=candidate._pool.retained_bytes)
            report['split_runtime'] = split.experiment_manifest()
            before = torch.cuda.memory_allocated()
            candidate.close()
            released = before - torch.cuda.memory_allocated()
            if released != 9 * 2**30:
                raise RuntimeError('Workspace release accounting failed')
            report['workspace']['released_bytes'] = released
            report['medians_seconds'] = {layout: {name: statistics.median(r['seconds'] for r in report['records']
                if r['arm'] == name and r['layout'] == layout) for name in arms}
                for layout in layouts}
        report['completed'] = True
    finally:
        if candidate is not None:
            candidate.close()
        (destination / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controls')
    parser.add_argument('destination')
    parser.add_argument('--batch', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--bias-layouts', choices=('baseline', 'head-last'), default='baseline')
    args = parser.parse_args()
    main(args.controls, args.destination, args.batch, args.bias_layouts)

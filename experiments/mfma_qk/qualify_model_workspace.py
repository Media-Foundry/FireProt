"""Qualify one Rust-governed workspace across mixed full attention batches."""
import argparse
import json
import os
from pathlib import Path
import threading

import torch

from experiments.mfma_qk import model_workspace, workspace_attention
from experiments.mfma_qk.benchmark_attention import pin
from experiments.mfma_qk.benchmark_microbatch import reference
from experiments.mfma_qk.benchmark_workspace import operands, exact
from experiments.hip_softmax.owned_scores import OwnedScoreSoftmax
from fireprot.backends.amd import softmax_fp32
from fireprot.ops._bridge import identity


def main(controls_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, qualification=[], guards=[],
        scope='Mixed-batch operator correctness and lifetime only; no model/E2E claim',
        reference='Full-batch native QK, accepted in-place HIP softmax, full-batch native PV',
        controls=pin(controls_path), policy_engine=identity,
        sources=[pin(__file__), pin(model_workspace.__file__), pin(workspace_attention.__file__)],
        visible_device=os.environ.get('HIP_VISIBLE_DEVICES'), batch_order=[],
        cache_state='One lazily initialized fixed-capacity workspace reused across all complete calls; poisoned per microchunk for full overwrite checks')
    candidate = None
    try:
        if str(torch.__version__) != softmax_fp32.TORCH_VERSION or torch.version.hip != softmax_fp32.HIP_VERSION:
            raise RuntimeError('Use the pinned Torch/HIP environment')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        free, total = torch.cuda.mem_get_info()
        report['settings'] = dict(torch=str(torch.__version__), hip=str(torch.version.hip),
            strict_deterministic=True, fill_uninitialized_memory=True, allow_tf32=False,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()),
            device_name=torch.cuda.get_device_name(), architecture=torch.cuda.get_device_properties(0).gcnArchName)
        report['preflight'] = dict(free_bytes=free, total_bytes=total, required_free_bytes=54 * 2**30)
        if free < 54 * 2**30:
            raise RuntimeError('Insufficient free memory for full B1024 reference and shared pool')
        control = json.loads(Path(controls_path).read_text())
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE'] = control['split_hip']['bundle']
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE_SHA256'] = control['split_hip']['manifest_sha256']
        with torch.inference_mode():
            split = OwnedScoreSoftmax()
            candidate = model_workspace.ModelWorkspaceFusion(control['fusion'],
                microbatch=control.get('microbatch', 256),
                allocation_policy=control.get('allocation_policy', 'ordinary'))
            measured_batches = tuple(b for b in (128, 256, 512, 1024) if b % candidate.microbatch == 0)
            report.update(microbatch=candidate.microbatch, retained_workspace_bytes=candidate.retained_bytes)
            previous = None
            score_pointer = None
            torch.cuda.reset_peak_memory_stats()
            for seed in (101, 102, 103):
                for layout in ('head_last', 'head_last_transposed'):
                    batches = measured_batches if layout == 'head_last' else tuple(reversed(measured_batches))
                    for batch in batches:
                        args = operands(batch, seed, layout)
                        if seed == 102:
                            args = (*args[:3], args[3][:1], args[4])
                        probability, expected = reference(args, split)
                        actual = candidate(*args, reference_probabilities=probability)
                        exact(expected, actual)
                        if expected.stride() != actual.stride():
                            raise RuntimeError('Output strides differ from full-batch native PV')
                        prior_checked = previous is not None
                        if prior_checked:
                            exact(*previous)
                        previous = (actual.clone(), actual)
                        pointer = candidate._pool._scores.data_ptr()
                        if score_pointer is not None and score_pointer != pointer:
                            raise RuntimeError('Workspace storage changed across outer batches')
                        score_pointer = pointer
                        record = dict(seed=seed, batch=batch, layout=layout, mask_batch=args[3].shape[0],
                            bias_strides=list(args[-1].stride()), output_strides=list(actual.stride()),
                            all_probability_bits_exact=True, all_output_bits_exact=True,
                            previous_result_unchanged=True if prior_checked else None,
                            workspace_allocations=candidate._pool.allocations)
                        report['qualification'].append(record)
                        report['batch_order'].append(batch)
                        print(json.dumps(record), flush=True)
                        del args, probability, expected, actual
            del previous
            report['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            args = operands(256, 104, 'head_last')
            before = candidate.manifest()
            errors = []
            def wrong_thread():
                try:
                    candidate(*args)
                except RuntimeError as error:
                    errors.append(str(error))
            thread = threading.Thread(target=wrong_thread)
            thread.start()
            thread.join()
            if len(errors) != 1 or 'another thread' not in errors[0]:
                raise RuntimeError('Thread guard failed')
            with torch.cuda.stream(torch.cuda.Stream()):
                try:
                    candidate(*args)
                except RuntimeError as error:
                    if 'original device/stream' not in str(error):
                        raise
                else:
                    raise RuntimeError('Stream guard failed')
            if candidate.manifest() != before:
                raise RuntimeError('Rejected calls changed lease or submission state')
            report['guards'].append(dict(thread_rejected=True, stream_rejected=True, state_unchanged=True))
            del args
            report['workspace_before_close'] = candidate.manifest()
            r = report['workspace_before_close']
            expected_releases = int(candidate.allocation_policy == 'release_cached_once')
            if r['cache_release_attempts'] != expected_releases or r['cache_releases_completed'] != expected_releases:
                raise RuntimeError('One-time cache release accounting failed')
            expected_calls = {str(b): 6 for b in measured_batches}
            if (r['allocations'] != 1 or r['successful_kernel_calls'] != sum(
                        6 * b // candidate.microbatch for b in measured_batches)
                    or r['calls_by_batch'] != expected_calls
                    or r['lease']['allocation_count'] != 1
                    or r['lease']['completed_calls'] != sum(expected_calls.values())
                    or r['lease']['charged_bytes'] != candidate.retained_bytes):
                raise RuntimeError('Mixed-batch execution accounting failed')
            if split.storage_fallbacks or split.launches_submitted != sum(expected_calls.values()):
                raise RuntimeError('Reference execution accounting failed')
            report['split_runtime'] = split.experiment_manifest()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            candidate.close()
            report['released_tensor_bytes'] = before - torch.cuda.memory_allocated()
            report['workspace_after_close'] = candidate.manifest()
            if (report['released_tensor_bytes'] != candidate.retained_bytes
                    or report['workspace_after_close']['lease']['charged_bytes'] != 0
                    or report['workspace_after_close']['lease']['release_count'] != 1):
                raise RuntimeError('Workspace tensor release does not match Rust receipt')
        report['completed'] = True
    finally:
        try:
            if candidate is not None:
                candidate.close()
        finally:
            if candidate is not None:
                report['final_workspace'] = candidate.manifest()
            (destination / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controls')
    parser.add_argument('destination')
    args = parser.parse_args()
    main(args.controls, args.destination)

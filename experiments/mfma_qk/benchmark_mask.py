"""Complete attention timing including transposed pair-mask normalization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.mfma_qk.benchmark_attention import balanced_orders, pin
from experiments.mfma_qk.benchmark_microbatch import reference
from experiments.mfma_qk.benchmark_workspace import operands, exact
from experiments.mfma_qk.workspace_dispatch import load_workspace
from experiments.hip_softmax.owned_scores import OwnedScoreSoftmax
from fireprot.backends.amd import softmax_fp32


def main(controls, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, controls=pin(controls), source=pin(__file__), records=[],
        qualification=[], visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),
        scope='Complete attention with transposed pair masks; no model/E2E claim',
        cache_state='One4.5GiB pool retained in both arms; configured first-allocation policy before timing; no cache release during timing')
    pool = None
    try:
        if str(torch.__version__) != softmax_fp32.TORCH_VERSION or torch.version.hip != softmax_fp32.HIP_VERSION:
            raise RuntimeError('Use pinned Torch/HIP')
        control = json.loads(Path(controls).read_text())
        if control.get('microbatch') != 128 or control.get('normalize_transposed_mask') is not True:
            raise ValueError('Requires explicit capacity128 and transposed-mask normalization')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        report['settings'] = dict(torch=str(torch.__version__), hip=str(torch.version.hip), seed=101,
            strict_deterministic=True, fill_uninitialized_memory=True, allow_tf32=False,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()))
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE'] = control['split_hip']['bundle']
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE_SHA256'] = control['split_hip']['manifest_sha256']
        with torch.inference_mode():
            split = OwnedScoreSoftmax()
            pool = load_workspace(controls, hashlib.sha256(Path(controls).read_bytes()).hexdigest())
            for batch in (128, 256):
                for layout in ('head_last', 'head_last_transposed'):
                    args = operands(batch, 101, layout)
                    storage = torch.empty(1536, 1536, device='cuda')
                    mask = storage.T[17:17 + batch, None, None, :]
                    mask.copy_(args[3])
                    args = (*args[:3], mask, args[4])
                    del storage, mask
                    probabilities, expected = reference(args, split)
                    output = pool(*args, reference_probabilities=probabilities)
                    exact(expected, output)
                    report['qualification'].append(dict(batch=batch, layout=layout,
                        mask_strides=list(args[3].stride()), mask_offset=args[3].storage_offset(),
                        all_probability_bits_exact=True, all_output_bits_exact=True))
                    del probabilities, output
                    arms = dict(split_hip=lambda: reference(args, split)[1], workspace=lambda: pool(*args))
                    for fn in arms.values():
                        for _ in range(2):
                            output = fn()
                            torch.cuda.synchronize()
                            exact(expected, output)
                            del output
                    releases = pool.cache_release_attempts
                    for index, order in enumerate(balanced_orders(list(arms)) * 3):
                        for position, name in enumerate(order):
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            output = arms[name]()
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter() - start
                            exact(expected, output)
                            row = dict(batch=batch, layout=layout, arm=name, round=index,
                                position=position, seconds=elapsed, bitwise_mismatches=0)
                            report['records'].append(row)
                            print(json.dumps(row), flush=True)
                            del output
                    if releases != pool.cache_release_attempts:
                        raise RuntimeError('Cache policy changed during timing')
                    del args, expected
            report['medians_seconds'] = {f'B{b}_{layout}': {name: statistics.median(
                r['seconds'] for r in report['records'] if r['batch'] == b and r['layout'] == layout
                and r['arm'] == name) for name in arms}
                for b in (128, 256) for layout in ('head_last', 'head_last_transposed')}
            report['workspace_before_close'] = pool.manifest()
            before = torch.cuda.memory_allocated()
            pool.close()
            report['released_tensor_bytes'] = before - torch.cuda.memory_allocated()
            report['workspace_after_close'] = pool.manifest()
            if report['released_tensor_bytes'] != pool.retained_bytes or pool.manifest()['lease']['charged_bytes'] != 0:
                raise RuntimeError('Workspace release accounting failed')
            if split.storage_fallbacks:
                raise RuntimeError('Unexpected reference fallback')
            report['split_runtime'] = split.experiment_manifest()
        report['completed'] = True
    finally:
        try:
            if pool is not None:
                pool.close()
        finally:
            (destination / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controls')
    parser.add_argument('destination')
    args = parser.parse_args()
    main(args.controls, args.destination)

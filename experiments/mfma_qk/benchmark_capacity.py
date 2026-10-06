"""Complete-operator screen for qualified smaller pools at observed hot batches."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.mfma_qk.benchmark_attention import balanced_orders, pin
from experiments.mfma_qk.benchmark_microbatch import reference
from experiments.mfma_qk.benchmark_workspace import operands, exact
from experiments.mfma_qk.model_workspace import ModelWorkspaceFusion
from experiments.hip_softmax.owned_scores import OwnedScoreSoftmax
from fireprot.backends.amd import softmax_fp32


def main(controls, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, records=[], controls=[pin(p) for p in controls],
        source=pin(__file__), visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),
        scope='Complete attention operator at B128/B256, not model/E2E',
        cache_state='Both initialized pools retained during every arm:5.625GiB total; no cache flush; exact checks outside timing')
    pools = {}
    try:
        if str(torch.__version__) != softmax_fp32.TORCH_VERSION or torch.version.hip != softmax_fp32.HIP_VERSION:
            raise RuntimeError('Use pinned Torch/HIP')
        configs = [json.loads(Path(p).read_text()) for p in controls]
        if {c['microbatch'] for c in configs} != {32, 128} or len(configs) != 2:
            raise ValueError('Supply exactly one32 and one128 control')
        if configs[0]['split_hip'] != configs[1]['split_hip']:
            raise ValueError('Reference controls differ')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        report['settings'] = dict(torch=str(torch.__version__), hip=str(torch.version.hip),
            strict_deterministic=True, fill_uninitialized_memory=True, allow_tf32=False,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()), seed=101)
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE'] = configs[0]['split_hip']['bundle']
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE_SHA256'] = configs[0]['split_hip']['manifest_sha256']
        with torch.inference_mode():
            split = OwnedScoreSoftmax()
            pools = {f"capacity{c['microbatch']}": ModelWorkspaceFusion(c['fusion'], microbatch=c['microbatch'])
                     for c in configs}
            for batch in (128, 256):
                for layout in ('head_last', 'head_last_transposed'):
                    args = operands(batch, 101, layout)
                    probabilities, expected = reference(args, split)
                    # Keep reference storage live while initializing both pools.
                    for pool in pools.values():
                        output = pool(*args, reference_probabilities=probabilities)
                        exact(expected, output)
                        del output
                    del probabilities
                    arms = dict(split_hip=lambda: reference(args, split)[1])
                    arms.update({name: (lambda p=p: p(*args)) for name, p in pools.items()})
                    for fn in arms.values():
                        for _ in range(2):
                            output = fn()
                            torch.cuda.synchronize()
                            exact(expected, output)
                            del output
                    for index, order in enumerate(balanced_orders(list(arms)) * 2):
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
                    del args, expected
            report['medians_seconds'] = {f'B{b}_{layout}': {name: statistics.median(
                r['seconds'] for r in report['records'] if r['batch'] == b and r['layout'] == layout
                and r['arm'] == name) for name in arms}
                for b in (128, 256) for layout in ('head_last', 'head_last_transposed')}
            report['workspaces'] = {name: pool.manifest() for name, pool in pools.items()}
            for name, pool in pools.items():
                before = torch.cuda.memory_allocated()
                pool.close()
                released = before - torch.cuda.memory_allocated()
                if released != pool.retained_bytes:
                    raise RuntimeError('Pool release accounting failed')
                report['workspaces'][name]['released_tensor_bytes'] = released
            if split.storage_fallbacks:
                raise RuntimeError('Unexpected native reference storage fallback')
            report['split_runtime'] = split.experiment_manifest()
        report['completed'] = True
    finally:
        try:
            for pool in pools.values():
                pool.close()
        finally:
            (destination / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controls128')
    parser.add_argument('controls32')
    parser.add_argument('destination')
    args = parser.parse_args()
    main([args.controls128, args.controls32], args.destination)

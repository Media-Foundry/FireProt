"""Isolated complete-attention latency screen; no model or E2E speedup claim."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.utils.deterministic

from fireprot.backends.amd import softmax_fp32, _published
from experiments.hip_softmax import owned_scores
from experiments.mfma_qk import probe_fusion


def pin(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class PackedFusion:
    def __init__(self, control, batch):
        screen = control['numerical_screen']
        assert pin(screen['path'])['sha256'] == screen['sha256']
        evidence = json.loads(Path(screen['path']).read_text())
        assert evidence['completed'] and evidence['batch'] == batch
        assert (evidence['n'], evidence['heads'], evidence['k']) == (1536, 4, 32)
        assert len(evidence['records']) == 21 and all(
            r['passed'] and r['bitwise_mismatches'] == 0 for r in evidence['records'])
        assert evidence['kernel']['kernel']['manifest_sha256'] == control['manifest_sha256']
        os.environ['FIREPROT_QK_SOFTMAX_BUNDLE'] = control['bundle']
        os.environ['FIREPROT_QK_SOFTMAX_BUNDLE_SHA256'] = control['manifest_sha256']
        selected = _published.select_c('fireprot_experimental_qk_softmax',
            'FIREPROT_QK_SOFTMAX', 'experimental-qk-softmax-n1536-f32-v1')
        if selected is None:
            raise RuntimeError('Explicit fusion bundle required')
        self.library, self.receipt = selected
        self.entry = self.library.fireprot_qk_softmax_f32
        self.entry.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int]
            + [ctypes.c_int64] * 5 + [ctypes.c_int, ctypes.c_void_p])
        self.entry.restype = ctypes.c_int

    def __call__(self, query, key, value, mask, bias):
        batch = query.shape[0]
        # Include the real head-last input packing and output allocation/filling.
        packed_q = query.reshape(batch * 4, 1536, 32)
        packed_k = key.transpose(-1, -2).reshape(batch * 4, 32, 1536)
        assert probe_fusion.validate_inputs(packed_q, packed_k, mask, bias) == batch
        scores = torch.empty(batch, 4, 1536, 1536, device=query.device, dtype=query.dtype)
        status = self.entry(packed_q.data_ptr(), packed_k.data_ptr(), mask.data_ptr(),
            bias.data_ptr(), scores.data_ptr(), batch,
            mask.stride(0) if mask.shape[0] != 1 else 0, mask.stride(-1),
            bias.stride(1), bias.stride(2), bias.stride(3), 1,
            torch.cuda.current_stream(query.device).cuda_stream)
        if status:
            raise RuntimeError(f'Fusion launch failed: {status}')
        return torch.einsum('...qk, ...kc->...qc', scores, value).to(dtype=query.dtype)


def balanced_orders(labels):
    """Every arm occupies each ordinal position twice, with reverse rotations."""
    return [labels[offset:] + labels[:offset] for offset in range(len(labels))] + [
        list(reversed(labels[offset:] + labels[:offset])) for offset in range(len(labels))]


def main(controls_path, destination, batch):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, completed=False, batch=batch, n=1536, heads=4,
        k=32, seed=101, dtype='float32', controls=pin(controls_path), records=[],
        scope='Complete attention operator, including Q/K packing, allocations and native PV; not E2E',
        source_pins=[pin(module.__file__) for module in (
            probe_fusion, owned_scores, softmax_fp32, _published)] + [pin(__file__)],
        torch=str(torch.__version__), hip=str(torch.version.hip),
        visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),
        cache_state='Published artifacts; each arm warmed twice. No cache flush. Exact checks outside timing can affect device caches.')
    try:
        if str(torch.__version__) != softmax_fp32.TORCH_VERSION or torch.version.hip != softmax_fp32.HIP_VERSION:
            raise RuntimeError('Use the validated OpenFold3 Torch/HIP environment')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        torch.manual_seed(101)
        report['settings'] = dict(strict_deterministic=True, warn_only=False,
            fill_uninitialized_memory=True, allow_tf32=False,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()))
        controls = json.loads(Path(controls_path).read_text())
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE'] = controls['split_hip']['bundle']
        os.environ['FIREPROT_SOFTMAX_FP32_BUNDLE_SHA256'] = controls['split_hip']['manifest_sha256']
        with torch.inference_mode():
            split = owned_scores.OwnedScoreSoftmax()
            kernels = {entry['label']: PackedFusion(entry, batch) for entry in controls['fusion']}
            if len(kernels) != len(controls['fusion']) or any(k in ('native', 'split_hip') for k in kernels):
                raise ValueError('Fusion arm labels must be unique and separate from baselines')
            report['kernel_receipts'] = {name: kernel.receipt for name, kernel in kernels.items()}
            report['split_hip_receipt'] = split.receipt
            query, key, value = [(torch.randn(batch, 1536, 4, 32, device='cuda', dtype=torch.float32) * 0.125).transpose(1, 2)
                                 for _ in range(3)]
            mask = torch.zeros(batch, 1, 1, 1536, device='cuda', dtype=torch.float32)
            for outer in range(batch):
                mask[outer, ..., 1024 + outer % 512:] = -float('inf')
            bias_base = torch.randn(1, 4, 1536, 1536, device='cuda', dtype=torch.float32) * 0.125
            report['inputs'] = {name: dict(shape=list(t.shape), strides=list(t.stride()))
                for name, t in [('query', query), ('key', key), ('value', value), ('mask', mask)]}
            for layout in ('contiguous', 'transposed'):
                bias = bias_base if layout == 'contiguous' else bias_base.transpose(-1, -2)

                def native():
                    scores = torch.einsum('...qc, ...kc->...qk', query, key)
                    scores.add_(mask)
                    scores.add_(bias)
                    scores = torch.softmax(scores, -1)
                    return torch.einsum('...qk, ...kc->...qc', scores, value).to(dtype=query.dtype)

                def split_hip():
                    scores = torch.einsum('...qc, ...kc->...qk', query, key)
                    scores = split(scores, [mask, bias])
                    return torch.einsum('...qk, ...kc->...qc', scores, value).to(dtype=query.dtype)

                arms = dict(native=native, split_hip=split_hip)
                arms.update({name: (lambda kernel=kernel: kernel(query, key, value, mask, bias))
                             for name, kernel in kernels.items()})
                expected = native()
                torch.cuda.synchronize()
                for name, function in arms.items():
                    for _ in range(2):
                        actual = function()
                        torch.cuda.synchronize()
                        assert probe_fusion.bitwise_mismatch_count(expected, actual) == 0, name
                        del actual
                for round_index, order in enumerate(balanced_orders(list(arms))):
                    for position, name in enumerate(order):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        actual = arms[name]()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        mismatches = probe_fusion.bitwise_mismatch_count(expected, actual)
                        record = dict(layout=layout, round=round_index, position=position,
                            arm=name, seconds=elapsed, bitwise_mismatches=mismatches,
                            elements=actual.numel())
                        report['records'].append(record)
                        print(json.dumps(record), flush=True)
                        del actual
                        if mismatches:
                            raise RuntimeError(f'Complete attention mismatch in {name}')
                del expected
            assert split.storage_fallbacks == 0
            assert split.launches_submitted == 4 + 4 * len(arms)
            report['split_runtime'] = split.experiment_manifest()
            report['medians_seconds'] = {layout: {name: statistics.median(
                r['seconds'] for r in report['records'] if r['layout'] == layout and r['arm'] == name)
                for name in arms} for layout in ('contiguous', 'transposed')}
        report['completed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (destination / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controls')
    parser.add_argument('destination')
    parser.add_argument('--batch', type=int, choices=(32, 64, 128, 256, 512), default=32)
    args = parser.parse_args()
    main(args.controls, args.destination, args.batch)

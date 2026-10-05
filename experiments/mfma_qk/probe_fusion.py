"""Exact packed QK/LDS/softmax screen; no operator or model timing claims."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.utils.deterministic

from fireprot.backends.amd import softmax_fp32
from fireprot.backends.amd._published import select_c
from experiments.attention_trace import probe as trace_probe

N = 1536


def validate_inputs(query, key, mask, bias):
    """Bound the experimental ABI to allocated, dense inputs and known broadcasts."""
    if query.ndim != 3 or query.shape[0] % 4:
        raise ValueError('Q must contain four packed heads per outer batch')
    batch = query.shape[0] // 4
    if (not 1 <= batch <= 1536 or tuple(query.shape) != (batch * 4, N, 32)
            or tuple(key.shape) != (batch * 4, 32, N)
            or tuple(mask.shape) not in ((1, 1, 1, N), (batch, 1, 1, N))
            or tuple(bias.shape) != (1, 4, N, N)):
        raise ValueError('Unsupported packed operands or mask/bias broadcasts')
    if any(t.device != query.device or not t.is_cuda or t.dtype != torch.float32
           or t.requires_grad for t in (query, key, mask, bias)):
        raise ValueError('Inputs must be inference FP32 on one CUDA/HIP device')
    if not query.is_contiguous() or not key.is_contiguous() or not mask.is_contiguous():
        raise ValueError('Q, transposed K and mask must be contiguous')
    if not (bias.is_contiguous() or bias.transpose(-1, -2).is_contiguous()):
        raise ValueError('Bias must be contiguous or a dense query/key transpose')
    return batch


def native_scores(query, key, destination, report, case):
    if case != 'random':
        return torch.bmm(query, key)
    path = destination / 'native-qk-trace.json'
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
        with torch.profiler.record_function('fireprot.attention.QK'):
            scores = torch.bmm(query, key)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    observation = dict(einsums=[dict(label='QK')])
    trace_probe.trace_operations(observation, path)
    report['native_qk_observation'] = observation
    report['native_trace'] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    bmms = [op for op in observation['einsums'][0]['operations'] if op['operation'] == 'aten::bmm']
    if len(bmms) != 1 or bmms[0]['inputs'] != [
            dict(shape=list(query.shape), strides=list(query.stride())),
            dict(shape=list(key.shape), strides=list(key.stride()))]:
        raise RuntimeError('Native QK kernel/layout observation is incomplete')
    gemms = [k for k in bmms[0]['kernels'] if k['launch'] in (
        'hipExtModuleLaunchKernel', 'hipModuleLaunchKernel')]
    initialization = [k for k in bmms[0]['kernels'] if (
        k['launch'] == 'hipMemset' and k['name'] == '__amd_rocclr_fillBufferUnAligned.kd')]
    if len(gemms) != 1 or len(gemms) + len(initialization) != len(bmms[0]['kernels']):
        raise RuntimeError('Ambiguous or unexplained native QK kernels')
    report['native_qk_gemm'] = gemms[0]
    report['native_qk_initialization'] = initialization
    return scores


def operands(case, batch):
    query = torch.randn(batch * 4, N, 32, device='cuda', dtype=torch.float32) * 0.125
    key = torch.randn(batch * 4, 32, N, device='cuda', dtype=torch.float32) * 0.125
    if case == 'cancellation':
        query[:] = torch.tensor([16777216., 1., -16777216., 1.] + [0.] * 28,
                                device='cuda', dtype=torch.float32)
        key.fill_(1)
    elif case == 'subnormal':
        query.fill_(torch.finfo(torch.float32).tiny / 8)
        key.fill_(1)
        if query.view(torch.int32)[0, 0, 0].item() != 0x00100000:
            raise RuntimeError('Subnormal input was not preserved')
    elif case == 'signed_zero':
        query.fill_(-0.)
        key.fill_(1)
        if query.view(torch.int32)[0, 0, 0].item() != -2147483648:
            raise RuntimeError('Negative-zero input was not preserved')
    elif case == 'nonfinite':
        query[:, 0, 0] = float('nan')
        query[:, 1, 0] = float('inf')
        key[:, 0, 0] = 0
    return query, key


def staged_mismatches(scores, actual, reference_batch, mask=None, bias=None):
    """Keep full-shape native QK on CPU; compare all rows in bounded GPU batches.

    Optional softmax uses Torch on complete rows, partitioning outer batch only.
    This does not establish native full-batch softmax kernel selection parity.
    """
    if reference_batch < 1 or scores.shape != actual.shape:
        raise ValueError('Invalid staged comparison shape or reference batch')
    mismatches = 0
    for start in range(0, scores.shape[0], reference_batch):
        stop = min(start + reference_batch, scores.shape[0])
        expected = scores[start:stop].to(actual.device, copy=True)
        if mask is not None:
            expected.add_(mask if mask.shape[0] == 1 else mask[start:stop])
            expected.add_(bias)
            expected = torch.softmax(expected, -1)
        mismatches += int(torch.count_nonzero(
            expected.view(torch.int32) != actual[start:stop].view(torch.int32)).item())
        del expected
    return mismatches


def main(destination, batch, reference_batch=0):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, completed=False, batch=batch, n=N, heads=4, k=32,
                  seed=101, dtype='float32', records=[], reference_batch=reference_batch,
                  reference_method=('full_shape_qk_host_staged_outer_batch_softmax'
                      if reference_batch else 'full_shape_qk_and_softmax'),
                  scope='Synthetic packed operands only; no timing or model acceptance',
                  torch=str(torch.__version__), hip=str(torch.version.hip),
                  visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__), Path(trace_probe.__file__))})
    try:
        if reference_batch < 0 or (batch > 512 and not reference_batch):
            raise ValueError('Batch above512 requires explicit host-staged reference')
        if reference_batch:
            available = next(int(line.split()[1]) * 1024
                for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))
            required = (batch + 3 * reference_batch) * 4 * N * N * 4
            if available < required:
                raise MemoryError(f'Host reference requires at least {required} available bytes')
        if (str(torch.__version__) != softmax_fp32.TORCH_VERSION
                or torch.version.hip != softmax_fp32.HIP_VERSION):
            raise RuntimeError('Use the validated OpenFold3 Torch/HIP environment')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        torch.manual_seed(101)
        report['settings'] = dict(strict_deterministic=True, warn_only=False,
            fill_uninitialized_memory=True, allow_tf32=False,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()))
        selected = select_c('fireprot_experimental_qk_softmax', 'FIREPROT_QK_SOFTMAX',
                            'experimental-qk-softmax-n1536-f32-v1')
        if selected is None:
            raise RuntimeError('An explicitly pinned experimental bundle is required')
        library, report['kernel'] = selected
        entry = library.fireprot_qk_softmax_f32
        entry.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] + [ctypes.c_int64] * 5
                          + [ctypes.c_int, ctypes.c_void_p])
        entry.restype = ctypes.c_int

        def launch(query, key, mask, bias, mode):
            outer = validate_inputs(query, key, mask, bias)
            # Fresh allocation ensures disjoint output storage. No user output pointer.
            output = torch.empty(outer, 4, N, N, device=query.device, dtype=torch.float32)
            with torch.cuda.device(query.device):
                status = entry(query.data_ptr(), key.data_ptr(), mask.data_ptr(), bias.data_ptr(),
                    output.data_ptr(), outer, mask.stride(0) if mask.shape[0] != 1 else 0,
                    mask.stride(-1), bias.stride(1), bias.stride(2), bias.stride(3), mode,
                    torch.cuda.current_stream(query.device).cuda_stream)
                if status:
                    raise RuntimeError(f'Fusion launch failed: {status}')
                torch.cuda.synchronize(query.device)
            return output

        def compare(expected, actual, case, mode, mask, bias, layout):
            if reference_batch:
                mismatches = staged_mismatches(expected, actual, reference_batch,
                    mask if mode == 'softmax' else None, bias if mode == 'softmax' else None)
            else:
                mismatches = int(torch.count_nonzero(
                    expected.view(torch.int32) != actual.view(torch.int32)).item())
            record = dict(case=case, mode=mode, layout=layout, passed=mismatches == 0,
                elements=expected.numel(), bitwise_mismatches=mismatches,
                mask_shape=list(mask.shape), mask_stride=list(mask.stride()),
                bias_shape=list(bias.shape), bias_stride=list(bias.stride()))
            report['records'].append(record)
            print(json.dumps(record), flush=True)
            if mismatches:
                raise RuntimeError(f'Bitwise mismatch: {case}, {mode}, {layout}')

        with torch.inference_mode():
            for case in ('random', 'cancellation', 'subnormal', 'signed_zero', 'nonfinite'):
                query, key = operands(case, batch)
                scores = native_scores(query, key, destination, report, case).view(batch, 4, N, N)
                if reference_batch:
                    scores = scores.cpu()
                    torch.cuda.empty_cache()
                mask = torch.zeros(1, 1, 1, N, device='cuda', dtype=torch.float32)
                bias = torch.zeros(1, 4, N, N, device='cuda', dtype=torch.float32)
                actual = launch(query, key, mask, bias, 0)
                compare(scores, actual, case, 'raw_after_lds', mask, bias, 'contiguous')
                del actual
                variants = ('zero', 'partial_outer', 'additive', 'all_masked') if case == 'random' else ('zero',)
                for variant in variants:
                    if variant == 'partial_outer':
                        mask = torch.zeros(batch, 1, 1, N, device='cuda', dtype=torch.float32)
                        for outer in range(batch):
                            mask[outer, ..., N // 2 + outer % (N // 2):] = -float('inf')
                    elif variant == 'additive':
                        mask = torch.randn(1, 1, 1, N, device='cuda', dtype=torch.float32) * 0.25
                    elif variant == 'all_masked':
                        mask = torch.full((1, 1, 1, N), -float('inf'), device='cuda', dtype=torch.float32)
                    else:
                        mask = torch.zeros(1, 1, 1, N, device='cuda', dtype=torch.float32)
                    base_bias = torch.randn(1, 4, N, N, device='cuda', dtype=torch.float32) * 0.125
                    if case != 'random':
                        base_bias.zero_()
                    for layout in ('contiguous', 'transposed'):
                        bias = base_bias if layout == 'contiguous' else base_bias.transpose(-1, -2)
                        if reference_batch:
                            expected = scores
                        else:
                            biased = scores.clone()
                            biased.add_(mask)
                            biased.add_(bias)
                            expected = torch.softmax(biased, -1)
                            del biased
                        actual = launch(query, key, mask, bias, 1)
                        compare(expected, actual, f'{case}/{variant}', 'softmax', mask, bias, layout)
                        del expected, actual
                del query, key, scores, mask, bias, base_bias
        report['completed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (destination / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    parser.add_argument('--batch', type=int, choices=(32, 64, 128, 256, 512, 1024), default=32)
    parser.add_argument('--reference-batch', type=int, default=0,
        help='Stage full QK on CPU; use this many outer batches per reference softmax (0: full GPU)')
    args = parser.parse_args()
    main(args.destination, args.batch, args.reference_batch)

"""Pending lane calibration and packed K32 score comparison; no timing claims."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.utils.deterministic

from fireprot.backends.amd import softmax_fp32
from fireprot.backends.amd._published import select_c
from experiments.attention_trace import probe as trace_probe


def main(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, completed=False, seed=101, records=[],
                  scope='Unvalidated MFMA lane/K-order probe, not model or performance evidence',
                  torch=str(torch.__version__), hip=str(torch.version.hip),
                  visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),
                  trace_helper_sha256=hashlib.sha256(
                      Path(trace_probe.__file__).read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        if (str(torch.__version__) != softmax_fp32.TORCH_VERSION
                or torch.version.hip != softmax_fp32.HIP_VERSION):
            raise RuntimeError('Use the validated OpenFold3 Torch/HIP environment')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        # Match pinned OpenFold3 entry_points/import_utils.py::_configure_torch_backend.
        # On ROCm this Torch API spelling selects hipBLAS rather than hipBLASLt.
        torch.backends.cuda.preferred_blas_library('cublas')
        torch.manual_seed(101)
        report['settings'] = dict(strict_deterministic=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()))
        selected = select_c('fireprot_experimental_qk_mfma16', 'FIREPROT_QK_MFMA16',
                            'experimental-qk-mfma16-f32-v1')
        if selected is None:
            raise RuntimeError('An explicitly pinned experimental bundle is required')
        library, report['kernel'] = selected
        entry = library.fireprot_qk_mfma16_f32
        entry.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        entry.restype = ctypes.c_int

        def launch(query, key_transposed, raw=False):
            batch, n, k = query.shape
            if (k != 32 or n not in (16, 1536)
                    or tuple(key_transposed.shape) != (batch, 32, n)
                    or query.device != key_transposed.device
                    or any(not t.is_cuda or t.dtype != torch.float32
                           or not t.is_contiguous() or t.requires_grad
                           for t in (query, key_transposed))):
                raise ValueError('Unsupported packed score operands')
            output = torch.empty(batch, n, n, device=query.device, dtype=torch.float32)
            status = entry(query.data_ptr(), key_transposed.data_ptr(), output.data_ptr(),
                           batch, n, int(raw), torch.cuda.current_stream().cuda_stream)
            if status:
                raise RuntimeError(f'QK probe launch failed: {status}')
            torch.cuda.synchronize()
            return output

        with torch.inference_mode():
            query = torch.zeros(1, 16, 32, device='cuda', dtype=torch.float32)
            key = torch.zeros(1, 32, 16, device='cuda', dtype=torch.float32)
            query[..., 0] = 1
            query[..., 1] = torch.arange(1, 17, device='cuda') * 100
            key[:, 0, :] = torch.arange(1, 17, device='cuda')
            key[:, 1, :] = 1
            raw = launch(query, key, raw=True).view(64, 4)
            lanes = torch.arange(64, device='cuda')[:, None]
            registers = torch.arange(4, device='cuda')[None, :]
            expected = ((lanes % 16 + 1) * 100 + 4 * (lanes // 16) + registers + 1).float()
            passed = torch.equal(raw.view(torch.int32), expected.view(torch.int32))
            report['lane_calibration'] = dict(passed=passed,
                raw_accumulator_bits=raw.view(torch.int32).cpu().tolist(),
                expected_encoded_values=expected.int().cpu().tolist())
            if not passed:
                raise RuntimeError('Candidate input/output lane mapping rejected')
            del query, key, raw, expected

            for case in ('random', 'cancellation', 'subnormal', 'signed_zero', 'nonfinite'):
                # Matches the accepted trace's packed BMM layouts, not original QKV storage.
                query = torch.randn(128, 1536, 32, device='cuda', dtype=torch.float32) * 0.125
                key = torch.randn(128, 32, 1536, device='cuda', dtype=torch.float32) * 0.125
                if case == 'cancellation':
                    query[:] = torch.tensor([16777216., 0, 0, 0, 1, 0, 0, 0,
                        -16777216., 0, 0, 0, 1, 0, 0, 0] * 2, device='cuda', dtype=torch.float32)
                    key.fill_(1)
                elif case == 'subnormal':
                    query.fill_(torch.finfo(torch.float32).tiny / 8)
                    key.fill_(1)
                    if int(query.view(torch.int32)[0, 0, 0].item()) != 0x00100000:
                        raise RuntimeError('Subnormal input was not preserved')
                elif case == 'signed_zero':
                    query.fill_(-0.0)
                    key.fill_(1)
                    if int(query.view(torch.int32)[0, 0, 0].item()) != -2147483648:
                        raise RuntimeError('Negative-zero input was not preserved')
                elif case == 'nonfinite':
                    query[:, 0, 0] = float('nan')
                    query[:, 1, 0] = float('inf')
                    key[:, 0, 0] = 0
                if case == 'random':
                    trace_path = destination / 'native-qk-trace.json'
                    with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True) as profiler:
                        with torch.profiler.record_function('fireprot.attention.QK'):
                            expected = torch.bmm(query, key)
                        torch.cuda.synchronize()
                    profiler.export_chrome_trace(str(trace_path))
                    observation = dict(einsums=[dict(label='QK')])
                    trace_probe.trace_operations(observation, trace_path)
                    report['native_qk_observation'] = observation
                    report['native_trace'] = dict(path=str(trace_path), sha256=
                        hashlib.sha256(trace_path.read_bytes()).hexdigest())
                    operations = observation['einsums'][0]['operations']
                    bmms = [op for op in operations if op['operation'] == 'aten::bmm']
                    if (len(bmms) != 1
                            or bmms[0]['inputs'] != [
                                dict(shape=list(query.shape), strides=list(query.stride())),
                                dict(shape=list(key.shape), strides=list(key.stride()))]):
                        raise RuntimeError('Native QK kernel/layout observation is incomplete')
                    # The first BLAS call can include a uniquely correlated runtime
                    # initialization memset. Preserve it; do not confuse it with GEMM.
                    gemms = [k for k in bmms[0]['kernels'] if k['launch'] in (
                        'hipExtModuleLaunchKernel', 'hipModuleLaunchKernel')]
                    initialization = [k for k in bmms[0]['kernels'] if (
                        k['launch'] == 'hipMemset'
                        and k['name'] == '__amd_rocclr_fillBufferUnAligned.kd')]
                    if (len(gemms) != 1
                            or len(gemms) + len(initialization) != len(bmms[0]['kernels'])):
                        raise RuntimeError('Native QK contains ambiguous or unexplained kernels')
                    report['native_qk_gemm'] = gemms[0]
                    report['native_qk_initialization'] = initialization
                    # This trace is diagnostic only; no latency is reported from it.
                else:
                    expected = torch.bmm(query, key)
                actual = launch(query, key)
                mismatches = int(torch.count_nonzero(
                    expected.view(torch.int32) != actual.view(torch.int32)).item())
                record = dict(case=case, passed=mismatches == 0, bitwise_mismatches=mismatches,
                              query_shape=list(query.shape), query_stride=list(query.stride()),
                              key_shape=list(key.shape), key_stride=list(key.stride()))
                report['records'].append(record)
                print(json.dumps(record), flush=True)
                del query, key, expected, actual
                if mismatches:
                    raise RuntimeError(f'Native packed QK mismatch: {case}')
        report['completed'] = True
    finally:
        (destination / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])

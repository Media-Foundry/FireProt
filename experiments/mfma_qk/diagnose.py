"""Diagnose rejected K32 arithmetic on tiles of the full native BMM reference."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.utils.deterministic

from fireprot.backends.amd import softmax_fp32
from fireprot.backends.amd._published import select_c


def main(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, completed=False, seed=101, records=[],
                  scope='Diagnostic tiles only; not a full score/model acceptance or timing result',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  visible_device=os.environ.get('HIP_VISIBLE_DEVICES'))
    try:
        if (str(torch.__version__) != softmax_fp32.TORCH_VERSION
                or torch.version.hip != softmax_fp32.HIP_VERSION):
            raise RuntimeError('Unvalidated Torch/HIP environment')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library('cublas')
        torch.manual_seed(101)
        selected = select_c('fireprot_experimental_qk_mfma16', 'FIREPROT_QK_MFMA16',
                            'experimental-qk-mfma16-f32-v1')
        if selected is None:
            raise RuntimeError('An explicitly pinned experimental bundle is required')
        library, report['kernel'] = selected
        entry = library.fireprot_qk_mfma16_f32
        entry.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        entry.restype = ctypes.c_int
        report['settings'] = dict(torch=str(torch.__version__), hip=torch.version.hip,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()),
            strict_deterministic=torch.are_deterministic_algorithms_enabled(),
            fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        with torch.inference_mode():
            for case in ('dense_random', 'single_k', 'first_k4',
                         'cancellation_adjacent', 'cancellation_strided'):
                query = torch.randn(128, 1536, 32, device='cuda', dtype=torch.float32) * 0.125
                key = torch.randn(128, 32, 1536, device='cuda', dtype=torch.float32) * 0.125
                if case == 'single_k':
                    query.zero_()
                    indices = (torch.arange(1536, device='cuda')[None, :]
                               + 16 * (torch.arange(128, device='cuda')[:, None] % 2)) % 32
                    query.scatter_(-1, indices[..., None], 1)
                elif case == 'first_k4':
                    query[..., 4:] = 0
                elif case.startswith('cancellation_'):
                    query.zero_()
                    key.fill_(1)
                    indices = [0, 1, 2, 3] if case.endswith('adjacent') else [0, 8, 16, 24]
                    for k, value in zip(indices, (16777216., 1., -16777216., 1.)):
                        query[..., k] = value
                # Keep the native GEMM dimensions/layout identical to the model trace.
                native = torch.bmm(query, key)
                expected = native[:2, :16, :16].contiguous()
                q = query[:2, :16, :].contiguous()
                kt = key[:2, :, :16].contiguous()
                actual = torch.empty_like(expected)
                status = entry(q.data_ptr(), kt.data_ptr(), actual.data_ptr(),
                               2, 16, 0, torch.cuda.current_stream().cuda_stream)
                if status:
                    raise RuntimeError(f'QK diagnostic launch failed: {status}')
                torch.cuda.synchronize()
                mismatch = expected.view(torch.int32) != actual.view(torch.int32)
                same_sign = torch.signbit(expected) == torch.signbit(actual)
                finite = torch.isfinite(expected) & torch.isfinite(actual)
                eligible = finite & same_sign
                # Int-bit distance is an ULP distance only within the same sign.
                distance = (expected.view(torch.int32).long()
                            - actual.view(torch.int32).long()).abs()[eligible]
                path = destination / (case + '.npz')
                np.savez(path, query=q.cpu().numpy(), key_transposed=kt.cpu().numpy(),
                         expected=expected.cpu().numpy(), actual=actual.cpu().numpy())
                record = dict(case=case, compared_elements=expected.numel(),
                    bitwise_mismatches=int(mismatch.sum().item()),
                    max_abs_error=float((actual - expected).abs().max().item()),
                    same_sign_finite_elements=int(eligible.sum().item()),
                    max_same_sign_ulp=int(distance.max().item()) if distance.numel() else None,
                    native_query_shape=list(query.shape), native_key_shape=list(key.shape),
                    candidate_query_shape=list(q.shape), candidate_key_shape=list(kt.shape),
                    tile_archive=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                if case.startswith('cancellation_'):
                    record.update(expected_values=expected.unique().cpu().tolist(),
                                  actual_values=actual.unique().cpu().tolist())
                report['records'].append(record)
                print(json.dumps(record), flush=True)
                del query, key, native, q, kt, expected, actual
        # Completion means all diagnostics ran, never numerical acceptance.
        report['completed'] = True
    finally:
        (destination / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])

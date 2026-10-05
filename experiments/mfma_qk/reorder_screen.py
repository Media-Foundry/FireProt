"""Test explicit K order hypotheses using pinned native-reference diagnostic tiles."""
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


def main(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, completed=False, records=[],
                  scope='K-order hypothesis screen on retained tiles, not complete-score/model acceptance',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  visible_device=os.environ.get('HIP_VISIBLE_DEVICES'))
    try:
        if (str(torch.__version__) != softmax_fp32.TORCH_VERSION
                or torch.version.hip != softmax_fp32.HIP_VERSION):
            raise RuntimeError('Unvalidated Torch/HIP environment')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.backends.cuda.matmul.allow_tf32 = False
        selected = select_c('fireprot_experimental_qk_mfma16', 'FIREPROT_QK_MFMA16',
                            'experimental-qk-mfma16-f32-v1')
        if selected is None:
            raise RuntimeError('Pinned experimental artifact required')
        library, report['kernel'] = selected
        entry = library.fireprot_qk_mfma16_f32
        entry.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        entry.restype = ctypes.c_int
        data = json.loads((source / 'diagnosis.json').read_text())
        report['reference_sha256'] = hashlib.sha256((source / 'diagnosis.json').read_bytes()).hexdigest()
        if not data['completed']:
            raise RuntimeError('Incomplete tile diagnostic')
        orders = dict(
            contiguous=list(range(32)),
            swap_middle_k4=[base + i for base in range(0, 32, 4) for i in (0, 2, 1, 3)],
            even_odd_k8=[base + i for base in range(0, 32, 8) for i in (0, 2, 4, 6, 1, 3, 5, 7)],
            stride8=[base + 8 * lane for base in range(8) for lane in range(4)])
        with torch.inference_mode():
            for name, order in orders.items():
                indices = torch.tensor(order, device='cuda', dtype=torch.int64)
                record = dict(name=name, logical_k_order=order, mismatches={})
                for case in data['records']:
                    pin = case['tile_archive']
                    path = source / Path(pin['path']).name
                    if hashlib.sha256(path.read_bytes()).hexdigest() != pin['sha256']:
                        raise RuntimeError('Native tile archive changed')
                    with np.load(path, allow_pickle=False) as arrays:
                        q = torch.from_numpy(arrays['query'].copy()).cuda()
                        key = torch.from_numpy(arrays['key_transposed'].copy()).cuda()
                        expected = torch.from_numpy(arrays['expected'].copy()).cuda()
                    if (tuple(q.shape) != (2, 16, 32) or tuple(key.shape) != (2, 32, 16)
                            or tuple(expected.shape) != (2, 16, 16)
                            or any(t.dtype != torch.float32 for t in (q, key, expected))):
                        raise RuntimeError('Unexpected tile contract')
                    q = q.index_select(-1, indices).contiguous()
                    key = key.index_select(-2, indices).contiguous()
                    actual = torch.empty_like(expected)
                    status = entry(q.data_ptr(), key.data_ptr(), actual.data_ptr(),
                                   2, 16, 0, torch.cuda.current_stream().cuda_stream)
                    if status:
                        raise RuntimeError(f'QK order screen launch failed: {status}')
                    torch.cuda.synchronize()
                    record['mismatches'][case['case']] = int(torch.count_nonzero(
                        expected.view(torch.int32) != actual.view(torch.int32)).item())
                record['all_tiles_exact'] = all(v == 0 for v in record['mismatches'].values())
                report['records'].append(record)
                print(json.dumps(record), flush=True)
        report['completed'] = True
    finally:
        (destination / 'orders.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])

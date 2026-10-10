"""Exact synthetic projection gates and balanced allocation-inclusive timing."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.utils.deterministic


def load_entry(receipt, symbol, arguments):
    path = Path(receipt['library'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['library_sha256']
    library = ctypes.CDLL(str(path))
    entry = getattr(library, symbol)
    entry.argtypes = arguments
    entry.restype = ctypes.c_int
    return library, entry


@torch.inference_mode()
def main(root, qualification):
    root = Path(root)
    assert os.environ['HIP_VISIBLE_DEVICES'] == '7'
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library('cublas')
    torch.manual_seed(101)
    receipt = json.loads((root / 'build/build.json').read_text())
    library, entry = load_entry(receipt, 'fireprot_project_128_16',
                               [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    qualified = json.loads(Path(qualification).read_text())
    assert qualified['all11_exact'] and qualified['full_cif_exact']
    pack_receipt = qualified['global_softmax']['pack_kernel']
    pack_library, pack = load_entry(pack_receipt, 'fireprot_pack_bias_h16_f32',
                                   [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_void_p])
    stream = torch.cuda.current_stream().cuda_stream

    def mismatch(a, b):
        assert a.shape == b.shape
        return sum(int((a[i:i+4096].contiguous().view(torch.int32) !=
                        b[i:i+4096].contiguous().view(torch.int32)).sum().item())
                   for i in range(0, a.shape[0], 4096))

    def hip(x, w, packed, guarded=False):
        rows = x.shape[0]
        storage = torch.full((rows * 16 + 512,), 17., device='cuda') if guarded else None
        output = storage[256:-256] if guarded else torch.empty(rows * 16, device='cuda')
        rc = entry(x.data_ptr(), w.data_ptr(), output.data_ptr(), rows, int(packed), stream)
        assert rc == 0
        output = output.view(16, rows).t() if packed else output.view(rows, 16)
        return output, storage

    report = dict(correctness_passed=False, seed=101, strict=True, hip_device=7,
                  source_sha256=hashlib.sha256((root / 'narrow_projection.cpp').read_bytes()).hexdigest(),
                  library_sha256=receipt['library_sha256'], records=[],
                  scope='Synthetic128-to16 projection only; no trained/model/E2E claim; strict allocations included in timing.')
    cases = [('random',64), ('random',4096), ('random',1536*1536),
             ('cancellation',256), ('subnormal',256), ('signed_zero',256),
             ('nonfinite',256), ('scaled',256)]
    try:
        for case, rows in cases:
            x = torch.randn(rows,128,device='cuda') * .125
            w = torch.randn(16,128,device='cuda') * .125
            if case == 'cancellation':
                x[:] = torch.tensor([16777216.,1.,-16777216.,1.] * 32,device='cuda')
                w.fill_(1.)
            elif case == 'subnormal':
                x.fill_(torch.finfo(torch.float32).tiny / 8); w.fill_(1.)
            elif case == 'signed_zero':
                x.fill_(-0.); w.fill_(1.)
            elif case == 'nonfinite':
                x[0,0] = float('nan'); x[1,0] = float('inf'); w[0,0] = 0.
            elif case == 'scaled':
                x.mul_(1.e10); w.mul_(1.e-10)
            expected = torch.nn.functional.linear(x,w)
            checks = []
            for packed in (False,True):
                actual, storage = hip(x,w,packed,True)
                bad = mismatch(actual,expected)
                guards = bool((storage[:256] == 17.).all()) and bool((storage[-256:] == 17.).all())
                checks.append(dict(packed=packed,mismatches=bad,guards_unchanged=guards,bitwise_equal=bad==0))
                del actual, storage
            record = dict(case=case,rows=rows,checks=checks)
            report['records'].append(record); print(json.dumps(record),flush=True)
            del x,w,expected
        report['correctness_passed'] = all(c['bitwise_equal'] and c['guards_unchanged']
                                         for r in report['records'] for c in r['checks'])
        if not report['correctness_passed']:
            return
        rows = 1536 * 1536
        x = torch.randn(rows,128,device='cuda') * .125
        w = torch.randn(16,128,device='cuda') * .125
        expected = torch.nn.functional.linear(x,w)
        labels = ['torch','torch_hip_pack','hip','hip_packed']
        orders = [labels[i:]+labels[:i] for i in range(4)]
        orders += [list(reversed(x)) for x in orders]
        orders *= 2
        samples = {name: [] for name in labels}
        checks = {name: 0 for name in labels}

        def invoke(name, timed):
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            if name.startswith('torch'):
                actual = torch.nn.functional.linear(x,w)
                if name == 'torch_hip_pack':
                    packed = torch.empty((16,rows),device='cuda')
                    assert pack(actual.data_ptr(),packed.data_ptr(),64,stream) == 0
                    actual = packed.t()
            else:
                actual,_ = hip(x,w,name=='hip_packed')
            b.record(); b.synchronize()
            if timed:
                samples[name].append(a.elapsed_time(b))
            assert mismatch(actual,expected) == 0
            checks[name] += 1
        for name in labels:
            for _ in range(2): invoke(name,False)
        for order in orders:
            for name in order: invoke(name,True)
        report['timing'] = dict(samples_ms=samples,medians_ms={k:statistics.median(v) for k,v in samples.items()},
                                exact_checks=checks,orders=orders,cache_state='Unflushed; exact comparisons between samples')
        print(json.dumps(report['timing']['medians_ms']),flush=True)
    finally:
        (root / 'screen.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
